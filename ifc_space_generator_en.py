#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFC Space Generator
===================
Creates IfcSpace objects (with room numbers) for all fully enclosed rooms of an
IFC file, without requiring any rooms in the source model.
Developed for Archicad 27 exports (IFC2X3), also works with IFC4.

How it works
------------
1. Load the geometry of walls, columns, curtain walls, doors, windows and slabs.
2. Cut a floor plan section per storey at the cut height (default +1.0 m).
   Wall/column cross sections are united, doors and windows are added as
   barriers. The "holes" of the union are the fully enclosed rooms.
3. Multi-storey rooms (stairwells, atria): if the slab above a room covers
   little of its footprint (slab opening) and the footprint on the storey above
   is (almost) identical, both are joined into ONE IfcSpace.
4. Write the IfcSpaces as an extrusion of the footprint, assign them to the
   storey, and store the room number in Name + LongName +
   Pset_SpaceCommon.Reference.

Dependencies
------------
    pip install ifcopenshell shapely numpy PySide6

    Python  >= 3.9
    ifcopenshell >= 0.8 (0.7 should work)
    shapely >= 2.0
    numpy
    PySide6 (GUI only; CLI mode works without it)

Usage
-----
    python ifc_space_generator.py                  # GUI
    python ifc_space_generator.py model.ifc        # no GUI (CLI)
    python ifc_space_generator.py -h               # all options
"""
from __future__ import annotations

import argparse
import os
import queue
import statistics
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.guid
import ifcopenshell.util.placement
import ifcopenshell.util.unit

# --------------------------------------------------------------------------
# Configuration: which IFC classes count as what
# --------------------------------------------------------------------------
BARRIER_CLASSES = ("IfcWall", "IfcWallStandardCase", "IfcColumn",
                   "IfcCurtainWall", "IfcMember", "IfcPlate")
FILL_CLASSES = ("IfcDoor", "IfcWindow")      # close openings
SLAB_CLASSES = ("IfcSlab", "IfcRoof")        # used to detect slab openings


@dataclass
class Params:
    src: str
    dst: str
    cut_height: float = 1.0          # m above storey floor level
    min_area: float = 1.0            # m², smaller "rooms" are discarded
    gap_mm: float = 20.0             # gaps up to this width count as closed
    number_format: str = "{g:02d}.{n:03d}"   # {g}=storey no., {s}=storey name, {n}=running no.
    first_storey_no: int = 1         # number assigned to the lowest storey
    merge_multistorey: bool = True
    ceiling_deduction: float = 0.0   # m, subtracted from the room height (slab thickness)
    slab_open_ratio: float = 0.5     # slab covers < 50 % of the room => slab opening
    iou_min: float = 0.8             # minimum footprint overlap when joining storeys


class Cancelled(Exception):
    pass


class Reporter:
    """Bundles progress / status / log; same interface for GUI and CLI."""

    def __init__(self, on_progress=None, on_status=None, on_log=None, is_cancelled=None):
        self._p, self._s, self._l, self._c = on_progress, on_status, on_log, is_cancelled
        self._last = object()

    def progress(self, value):
        """value: 0..100 or None (indeterminate)."""
        v = None if value is None else round(float(value), 1)
        if v != self._last and self._p:
            self._last = v
            self._p(v)

    def status(self, text):
        if self._s:
            self._s(text)

    def log(self, text):
        if self._l:
            self._l(text)

    def check(self):
        if self._c and self._c():
            raise Cancelled()


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
@dataclass
class Item:
    kind: str            # "barrier" | "fill" | "slab"
    verts: np.ndarray    # (n,3) meters, world coordinates
    faces: np.ndarray    # (m,3)
    zmin: float
    zmax: float


def _parts(g):
    return list(g.geoms) if hasattr(g, "geoms") else [g]


def _polys(g):
    return [x for x in _parts(g) if x.geom_type == "Polygon" and not x.is_empty]


def slice_segments(verts, faces, z):
    """Intersect a triangle mesh with the plane Z=z, returns segments (k,2,2)."""
    tri = verts[faces]
    d = tri[:, :, 2] - z
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    pos = d > 0
    npos = pos.sum(axis=1)
    sel = (npos == 1) | (npos == 2)
    if not sel.any():
        return np.empty((0, 2, 2))
    tri, d, pos = tri[sel], d[sel], pos[sel]
    m = len(tri)
    cross = np.stack([pos[:, 0] != pos[:, 1],
                      pos[:, 1] != pos[:, 2],
                      pos[:, 2] != pos[:, 0]], axis=1)
    pts = np.empty((m, 3, 2))
    for e, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        den = d[:, a] - d[:, b]
        den = np.where(den == 0, 1.0, den)
        t = d[:, a] / den
        pts[:, e, :] = tri[:, a, :2] + (tri[:, b, :2] - tri[:, a, :2]) * t[:, None]
    order = np.argsort(~cross, axis=1, kind="stable")[:, :2]
    rows = np.arange(m)
    return np.stack([pts[rows, order[:, 0]], pts[rows, order[:, 1]]], axis=1)


def _drop_hole_faces(faces):
    """polygonize also returns holes of a cross section as separate faces -> remove them."""
    faces = sorted(faces, key=lambda f: f.area, reverse=True)
    kept = []
    for f in faces:
        rp = f.representative_point()
        in_hole = False
        for a in kept:
            if len(a.interiors) and Polygon(a.exterior).contains(rp) and not a.contains(rp):
                in_hole = True
                break
        if not in_hole:
            kept.append(f)
    return kept


def footprint(item, z, tol):
    segs = slice_segments(item.verts, item.faces, z)
    if len(segs) == 0:
        return None
    segs = np.round(segs, 5)
    segs = segs[(segs[:, 0] != segs[:, 1]).any(axis=1)]
    if len(segs) == 0:
        return None
    if item.kind == "fill":      # door/window: convex hull of the cut as a barrier
        return shapely.MultiPoint(segs.reshape(-1, 2)).convex_hull.buffer(max(tol, 0.005))
    lines = shapely.union_all(shapely.linestrings(segs))
    faces = _drop_hole_faces(list(polygonize(_parts(lines))))
    if not faces:                # mesh not closed -> use the outline as a thin barrier
        return lines.buffer(0.003)
    return shapely.union_all(faces)


def build_barrier(items, z, tol):
    geoms = []
    for it in items:
        if it.zmin - 1e-6 > z or it.zmax + 1e-6 < z:
            continue
        g = footprint(it, z, tol)
        if g is None or g.is_empty:
            continue
        if not g.is_valid:
            g = shapely.make_valid(g)
        geoms.append(g)
    if not geoms:
        return None
    return shapely.union_all(geoms).buffer(tol)


def extract_rooms(barrier, tol, min_area):
    rooms = []
    for poly in _polys(barrier):
        for ring in poly.interiors:
            hole = Polygon(ring)
            if hole.area < min_area * 0.25:
                continue
            free = hole.difference(barrier)      # subtract islands (free-standing columns)
            for part in _polys(free):
                r = part.buffer(tol, join_style="mitre").simplify(0.002)
                if r.is_empty or r.geom_type != "Polygon" or r.area < min_area:
                    continue
                rooms.append(r)
    return rooms


def slab_footprint(slab_items, z_level, win=0.6):
    """Footprint of all slabs whose horizontal faces lie near z_level (including openings)."""
    tris = []
    for it in slab_items:
        if it.zmax < z_level - win or it.zmin > z_level + win:
            continue
        v = it.verts[it.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12
        nz = np.zeros(len(n))
        nz[ok] = np.abs(n[ok, 2]) / ln[ok]
        zc = v[:, :, 2].mean(axis=1)
        m = (nz > 0.95) & (np.abs(zc - z_level) <= win)
        if m.any():
            tris.append(v[m][:, :, :2])
    if not tris:
        return None
    t = np.concatenate(tris)
    rings = np.concatenate([t, t[:, :1, :]], axis=1)
    polys = shapely.polygons(rings)
    polys = polys[shapely.area(polys) > 1e-9]
    if len(polys) == 0:
        return None
    return shapely.union_all(polys)


# --------------------------------------------------------------------------
# Reading the IFC
# --------------------------------------------------------------------------
@dataclass
class Storey:
    ent: object
    name: str
    M: np.ndarray    # 4x4 world placement in file units
    elev: float      # meters


def read_storeys(model, scale, rep):
    storeys = []
    for st in model.by_type("IfcBuildingStorey"):
        if st.ObjectPlacement is None:
            continue
        M = np.array(ifcopenshell.util.placement.get_local_placement(st.ObjectPlacement), dtype=float)
        storeys.append(Storey(st, st.Name or st.LongName or f"#{st.id()}", M, M[2, 3] * scale))
    if not storeys:
        raise ValueError("The IFC contains no IfcBuildingStorey with a placement.")
    zs = {round(s.elev, 3) for s in storeys}
    if len(zs) == 1 and len(storeys) > 1:
        # All placements identical -> fall back to the Elevation attribute
        alt = [(s.ent.Elevation or 0.0) * scale for s in storeys]
        if len({round(a, 3) for a in alt}) > 1:
            rep.log("Note: storey placements share the same height, using the 'Elevation' attribute.")
            for s, a in zip(storeys, alt):
                s.elev = a
    storeys.sort(key=lambda s: s.elev)
    return storeys


def load_items(model, rep):
    kinds = {}
    for kind, classes in (("barrier", BARRIER_CLASSES), ("fill", FILL_CLASSES), ("slab", SLAB_CLASSES)):
        for cls in classes:
            try:
                for e in model.by_type(cls):
                    kinds.setdefault(e.id(), kind)
            except RuntimeError:
                pass
    elements = [model.by_id(i) for i in kinds]
    rep.log(f"Found {len(elements)} relevant elements - computing geometry ...")

    settings = ifcopenshell.geom.settings()
    try:
        settings.set("use-world-coords", True)
    except Exception:
        settings.set(settings.USE_WORLD_COORDS, True)

    items = []
    if not elements:
        return items
    it = ifcopenshell.geom.iterator(settings, model, max(1, (os.cpu_count() or 2) - 1), include=elements)
    if not it.initialize():
        return items
    n = 0
    while True:
        rep.check()
        shape = it.get()
        kind = kinds.get(shape.id)
        g = shape.geometry
        if kind and len(g.faces):
            v = np.array(g.verts, dtype=float).reshape(-1, 3)
            f = np.array(g.faces, dtype=np.int64).reshape(-1, 3)
            items.append(Item(kind, v, f, float(v[:, 2].min()), float(v[:, 2].max())))
        n += 1
        if n % 20 == 0:
            rep.progress(3 + 62 * it.progress() / 100.0)
            rep.status(f"Computing geometry ... {n}/{len(elements)} elements")
        if not it.next():
            break
    return items


# --------------------------------------------------------------------------
# Writing the IFC
# --------------------------------------------------------------------------
def _body_context(model):
    subs = [c for c in model.by_type("IfcGeometricRepresentationSubContext")
            if c.ContextIdentifier == "Body"]
    if subs:
        return subs[0]
    ctxs = [c for c in model.by_type("IfcGeometricRepresentationContext")
            if not c.is_a("IfcGeometricRepresentationSubContext")]
    for c in ctxs:
        if c.ContextType == "Model":
            return c
    return ctxs[0]


def _local_ring(coords, Minv, z0_file, scale):
    a = np.asarray(coords, dtype=float)[:, :2] / scale
    h = np.column_stack([a, np.full(len(a), z0_file), np.ones(len(a))])
    loc = (Minv @ h.T).T
    return [(float(x), float(y)) for x, y, _, _ in loc]


def _polyline(model, pts):
    return model.create_entity("IfcPolyline", Points=[
        model.create_entity("IfcCartesianPoint", Coordinates=p) for p in pts])


def add_space(model, ctx, oh, storey, Minv, scale, poly, z0_m, height_m, number):
    z0_file = z0_m / scale
    poly = orient(poly, 1.0)
    outer = _polyline(model, _local_ring(poly.exterior.coords, Minv, z0_file, scale))
    if len(poly.interiors):
        inner = [_polyline(model, _local_ring(r.coords, Minv, z0_file, scale)) for r in poly.interiors]
        profile = model.create_entity("IfcArbitraryProfileDefWithVoids", ProfileType="AREA",
                                      OuterCurve=outer, InnerCurves=inner)
    else:
        profile = model.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=outer)

    base_z = float((Minv @ np.array([0.0, 0.0, z0_file, 1.0]))[2])
    solid = model.create_entity(
        "IfcExtrudedAreaSolid", SweptArea=profile,
        Position=model.create_entity("IfcAxis2Placement3D",
                                     Location=model.create_entity("IfcCartesianPoint",
                                                                  Coordinates=(0.0, 0.0, base_z))),
        ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
        Depth=float(height_m / scale))
    rep = model.create_entity("IfcShapeRepresentation", ContextOfItems=ctx,
                              RepresentationIdentifier="Body", RepresentationType="SweptSolid",
                              Items=[solid])
    pds = model.create_entity("IfcProductDefinitionShape", Representations=[rep])
    placement = model.create_entity(
        "IfcLocalPlacement", PlacementRelTo=storey.ObjectPlacement,
        RelativePlacement=model.create_entity(
            "IfcAxis2Placement3D",
            Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))))

    kw = dict(GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, Name=number, LongName=number,
              ObjectPlacement=placement, Representation=pds, CompositionType="ELEMENT")
    if model.schema == "IFC2X3":
        kw["InteriorOrExteriorSpace"] = "INTERNAL"
    else:
        kw["PredefinedType"] = "INTERNAL"
    space = model.create_entity("IfcSpace", **kw)

    props = [
        model.create_entity("IfcPropertySingleValue", Name="Reference",
                            NominalValue=model.create_entity("IfcIdentifier", number)),
        model.create_entity("IfcPropertySingleValue", Name="IsExternal",
                            NominalValue=model.create_entity("IfcBoolean", False)),
    ]
    pset = model.create_entity("IfcPropertySet", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
                               Name="Pset_SpaceCommon", HasProperties=props)
    model.create_entity("IfcRelDefinesByProperties", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
                        RelatedObjects=[space], RelatingPropertyDefinition=pset)
    return space


# --------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------
@dataclass
class Chain:
    base: int
    top: int
    base_poly: Polygon
    last_poly: Polygon


def generate_spaces(p: Params, rep: Reporter) -> dict:
    t0 = time.time()
    if os.path.abspath(p.src) == os.path.abspath(p.dst):
        raise ValueError("The output file must not be the input file.")
    try:
        p.number_format.format(g=1, s="X", n=1)
    except Exception as e:
        raise ValueError(f"Invalid numbering scheme '{p.number_format}': {e}")

    rep.status("Loading IFC file ... (this can take a moment for large files)")
    rep.progress(None)
    model = ifcopenshell.open(p.src)
    scale = ifcopenshell.util.unit.calculate_unit_scale(model)
    rep.log(f"Schema {model.schema}, length unit = {scale} m")
    if model.by_type("IfcSpace"):
        rep.log(f"WARNING: file already contains {len(model.by_type('IfcSpace'))} IfcSpace objects; "
                f"new spaces are added in addition.")

    storeys = read_storeys(model, scale, rep)
    n = len(storeys)
    rep.log(f"{n} storeys: " + ", ".join(f"{s.name} ({s.elev:+.2f} m)" for s in storeys))

    rep.progress(3)
    items = load_items(model, rep)
    barrier_items = [i for i in items if i.kind != "slab"]
    slab_items = [i for i in items if i.kind == "slab"]
    rep.log(f"{len(barrier_items)} barrier elements, {len(slab_items)} slabs with geometry")
    if not barrier_items:
        raise ValueError("No wall/column geometry found.")

    tol = max(p.gap_mm / 2000.0, 0.0005)

    # ---- Analyze storeys
    rooms_by_storey = []
    for k, st in enumerate(storeys):
        rep.check()
        rep.status(f"Analyzing storey {k + 1}/{n}: {st.name} ...")
        barrier = build_barrier(barrier_items, st.elev + p.cut_height, tol)
        rooms = extract_rooms(barrier, tol, p.min_area) if barrier is not None else []
        rooms_by_storey.append(rooms)
        rep.log(f"  {st.name}: {len(rooms)} enclosed rooms")
        rep.progress(65 + 25 * (k + 1) / n)

    # ---- Join multi-storey rooms
    chains, active = [], []
    for k in range(n):
        rep.check()
        rooms = rooms_by_storey[k]
        used, nxt = set(), []
        if p.merge_multistorey and k > 0 and active and rooms:
            fp = slab_footprint(slab_items, storeys[k].elev)
            if fp is None:
                rep.log(f"  No slab geometry at {storeys[k].name} - not joining downwards.")
            else:
                for ch in active:
                    last = ch.last_poly
                    if last.intersection(fp).area / last.area >= p.slab_open_ratio:
                        continue                      # closed slab -> separate room
                    best_iou, best_j = 0.0, None
                    for j, r in enumerate(rooms):
                        if j in used:
                            continue
                        inter = last.intersection(r).area
                        iou = inter / (last.area + r.area - inter)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_j is not None and best_iou >= p.iou_min:
                        ch.top, ch.last_poly = k, rooms[best_j]
                        used.add(best_j)
                        nxt.append(ch)
        for j, r in enumerate(rooms):
            if j not in used:
                ch = Chain(k, k, r, r)
                chains.append(ch)
                nxt.append(ch)
        active = nxt
    multi = [c for c in chains if c.top > c.base]
    rep.log(f"{len(chains)} spaces in total, {len(multi)} of them multi-storey")

    # ---- Heights
    heights = [storeys[k + 1].elev - storeys[k].elev for k in range(n - 1)]
    zc = storeys[-1].elev + p.cut_height
    tops = [i.zmax for i in barrier_items if i.kind == "barrier" and i.zmin <= zc <= i.zmax]
    last_h = (max(tops) - storeys[-1].elev) if tops else 0.0
    if not 2.0 <= last_h <= 6.0:
        last_h = statistics.median(heights) if heights else 3.0

    # ---- Write
    rep.status("Writing IfcSpaces ...")
    oh = model.by_type("IfcOwnerHistory")[0] if model.by_type("IfcOwnerHistory") else None
    ctx = _body_context(model)
    per_storey_spaces = {}
    per_storey_count = {}
    for k, st in enumerate(storeys):
        rep.check()
        group = [c for c in chains if c.base == k]
        group.sort(key=lambda c: (-round(c.base_poly.centroid.y, 1), c.base_poly.centroid.x))
        Minv = np.linalg.inv(st.M)
        for nr, ch in enumerate(group, start=1):
            number = p.number_format.format(g=p.first_storey_no + k, s=st.name, n=nr)
            z_top = storeys[ch.top + 1].elev if ch.top + 1 < n else storeys[ch.top].elev + last_h
            height = max(z_top - st.elev - p.ceiling_deduction, 0.5)
            sp = add_space(model, ctx, oh, st.ent, Minv, scale, ch.base_poly, st.elev, height, number)
            per_storey_spaces.setdefault(k, []).append(sp)
        per_storey_count[st.name] = len(group)
        rep.progress(90 + 8 * (k + 1) / n)

    for k, spaces in per_storey_spaces.items():
        st = storeys[k].ent
        rel = next((r for r in (st.IsDecomposedBy or ()) if r.is_a("IfcRelAggregates")), None)
        if rel is not None:
            rel.RelatedObjects = tuple(rel.RelatedObjects) + tuple(spaces)
        else:
            model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
                                RelatingObject=st, RelatedObjects=spaces)

    rep.status("Saving IFC file ...")
    rep.progress(None)
    model.write(p.dst)
    dt = time.time() - t0
    rep.progress(100)
    rep.status("Done.")
    rep.log(f"Saved: {p.dst} ({dt:.0f} s)")
    return {"spaces": len(chains), "multi": len(multi), "per_storey": per_storey_count,
            "dst": p.dst, "seconds": dt}


# --------------------------------------------------------------------------
# Subprocess entry point (keeps the GUI responsive during long computations)
# --------------------------------------------------------------------------
def _worker_main(params, q):
    rep = Reporter(on_progress=lambda v: q.put(("progress", v)),
                   on_status=lambda s: q.put(("status", s)),
                   on_log=lambda s: q.put(("log", s)))
    try:
        q.put(("done", generate_spaces(params, rep)))
    except Exception:
        q.put(("error", traceback.format_exc()))


# --------------------------------------------------------------------------
# GUI (PySide6)
# --------------------------------------------------------------------------
def run_gui():
    import multiprocessing as mp
    from PySide6 import QtCore, QtWidgets

    ctx = mp.get_context("spawn")

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("IFC Space Generator")
            self.resize(860, 680)
            self.setAcceptDrops(True)
            self._proc = None
            self._queue = None
            self._t0 = 0.0
            self._finished = True

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            lay = QtWidgets.QVBoxLayout(central)

            files = QtWidgets.QGroupBox("Files")
            fl = QtWidgets.QGridLayout(files)
            self.in_edit = QtWidgets.QLineEdit()
            self.in_edit.setPlaceholderText("Choose an IFC file or drag it here ...")
            b_in = QtWidgets.QPushButton("Choose IFC ...")
            b_in.clicked.connect(self.pick_in)
            self.out_edit = QtWidgets.QLineEdit()
            self.out_edit.setPlaceholderText("Output file (default: <name>_spaces.ifc)")
            b_out = QtWidgets.QPushButton("Save as ...")
            b_out.clicked.connect(self.pick_out)
            fl.addWidget(QtWidgets.QLabel("Input:"), 0, 0)
            fl.addWidget(self.in_edit, 0, 1)
            fl.addWidget(b_in, 0, 2)
            fl.addWidget(QtWidgets.QLabel("Output:"), 1, 0)
            fl.addWidget(self.out_edit, 1, 1)
            fl.addWidget(b_out, 1, 2)
            lay.addWidget(files)

            opts = QtWidgets.QGroupBox("Settings")
            of = QtWidgets.QFormLayout(opts)
            d = Params("", "")
            self.cut = QtWidgets.QDoubleSpinBox()
            self.cut.setRange(0.1, 3.0); self.cut.setSingleStep(0.1)
            self.cut.setSuffix(" m"); self.cut.setValue(d.cut_height)
            self.min_area = QtWidgets.QDoubleSpinBox()
            self.min_area.setRange(0.1, 100.0); self.min_area.setSingleStep(0.5)
            self.min_area.setSuffix(" m²"); self.min_area.setValue(d.min_area)
            self.gap = QtWidgets.QSpinBox()
            self.gap.setRange(0, 200); self.gap.setSuffix(" mm"); self.gap.setValue(int(d.gap_mm))
            self.fmt = QtWidgets.QLineEdit(d.number_format)
            self.fmt.setToolTip("{g} = storey number, {s} = storey name, {n} = running number\n"
                                "e.g. {g:02d}.{n:03d} -> 01.012")
            self.first = QtWidgets.QSpinBox()
            self.first.setRange(-20, 99); self.first.setValue(d.first_storey_no)
            self.deduct = QtWidgets.QDoubleSpinBox()
            self.deduct.setRange(0.0, 1.0); self.deduct.setSingleStep(0.05)
            self.deduct.setSuffix(" m"); self.deduct.setValue(d.ceiling_deduction)
            self.merge = QtWidgets.QCheckBox("Join multi-storey rooms (stairwells) into one space")
            self.merge.setChecked(False)
            of.addRow("Cut height above storey level:", self.cut)
            of.addRow("Minimum room area:", self.min_area)
            of.addRow("Close gaps up to:", self.gap)
            of.addRow("Numbering scheme:", self.fmt)
            of.addRow("Number of the lowest storey:", self.first)
            of.addRow("Subtract slab thickness from room height:", self.deduct)
            of.addRow("", self.merge)
            lay.addWidget(opts)

            row = QtWidgets.QHBoxLayout()
            self.btn_start = QtWidgets.QPushButton("Generate spaces")
            self.btn_start.setMinimumHeight(34)
            self.btn_start.clicked.connect(self.start)
            self.btn_cancel = QtWidgets.QPushButton("Cancel")
            self.btn_cancel.setMinimumHeight(34)
            self.btn_cancel.setEnabled(False)
            self.btn_cancel.clicked.connect(self.cancel)
            row.addWidget(self.btn_start, 3)
            row.addWidget(self.btn_cancel, 1)
            lay.addLayout(row)

            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            lay.addWidget(self.log, 1)

            sb = self.statusBar()
            self.elapsed = QtWidgets.QLabel("")
            self.bar = QtWidgets.QProgressBar()
            self.bar.setFixedWidth(260)
            self.bar.setRange(0, 1000)
            self.bar.setValue(0)
            sb.addPermanentWidget(self.elapsed)
            sb.addPermanentWidget(self.bar)
            sb.showMessage("Ready.")

            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(100)
            self.timer.timeout.connect(self._poll)

        # ---------- File selection
        def _set_input(self, path):
            self.in_edit.setText(path)
            base, _ = os.path.splitext(path)
            self.out_edit.setText(base + "_spaces.ifc")

        def pick_in(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open IFC", os.path.dirname(self.in_edit.text()), "IFC (*.ifc);;All files (*)")
            if path:
                self._set_input(path)

        def pick_out(self):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save output", self.out_edit.text(), "IFC (*.ifc)")
            if path:
                self.out_edit.setText(path)

        def dragEnterEvent(self, e):
            if e.mimeData().hasUrls():
                e.acceptProposedAction()

        def dropEvent(self, e):
            for u in e.mimeData().urls():
                if u.toLocalFile().lower().endswith(".ifc"):
                    self._set_input(u.toLocalFile())
                    break

        # ---------- Workflow
        def start(self):
            src = self.in_edit.text().strip()
            if not os.path.isfile(src):
                QtWidgets.QMessageBox.warning(self, "Input", "Please choose an existing IFC file.")
                return
            dst = self.out_edit.text().strip() or os.path.splitext(src)[0] + "_spaces.ifc"
            if os.path.abspath(dst) == os.path.abspath(src):
                QtWidgets.QMessageBox.warning(self, "Output", "The output must not be the input file.")
                return
            try:
                self.fmt.text().format(g=1, s="X", n=1)
            except Exception as ex:
                QtWidgets.QMessageBox.warning(self, "Numbering scheme", f"Invalid: {ex}")
                return
            params = Params(src=src, dst=dst, cut_height=self.cut.value(), min_area=self.min_area.value(),
                            gap_mm=float(self.gap.value()), number_format=self.fmt.text(),
                            first_storey_no=self.first.value(), merge_multistorey=self.merge.isChecked(),
                            ceiling_deduction=self.deduct.value())
            self.log.clear()
            self._queue = ctx.Queue()
            self._proc = ctx.Process(target=_worker_main, args=(params, self._queue), daemon=True)
            self._proc.start()
            self._finished = False
            self._t0 = time.time()
            self._set_running(True)
            self.timer.start()

        def cancel(self):
            if self._proc and self._proc.is_alive():
                self._proc.terminate()
            self._finish("Cancelled.")

        def _set_running(self, running):
            self.btn_start.setEnabled(not running)
            self.btn_cancel.setEnabled(running)

        def _finish(self, message):
            self._finished = True
            self.timer.stop()
            self._set_running(False)
            self.bar.setRange(0, 1000)
            self.statusBar().showMessage(message)

        def _drain(self):
            try:
                while True:
                    kind, payload = self._queue.get_nowait()
                    if kind == "progress":
                        if payload is None:
                            self.bar.setRange(0, 0)
                        else:
                            self.bar.setRange(0, 1000)
                            self.bar.setValue(int(payload * 10))
                    elif kind == "status":
                        self.statusBar().showMessage(payload)
                    elif kind == "log":
                        self.log.appendPlainText(payload)
                    elif kind == "done":
                        self._finish("Done.")
                        self.bar.setValue(1000)
                        per = "\n".join(f"  {k}: {v}" for k, v in payload["per_storey"].items())
                        QtWidgets.QMessageBox.information(
                            self, "Done",
                            f"Created {payload['spaces']} IfcSpaces "
                            f"({payload['multi']} multi-storey).\n\n{per}\n\n{payload['dst']}")
                    elif kind == "error":
                        self._finish("Error.")
                        self.log.appendPlainText(payload)
                        QtWidgets.QMessageBox.critical(self, "Error", payload.strip().splitlines()[-1])
            except queue.Empty:
                pass

        def _poll(self):
            if self._finished:
                return
            s = int(time.time() - self._t0)
            self.elapsed.setText(f"Elapsed {s // 60:02d}:{s % 60:02d}")
            self._drain()
            if not self._finished and self._proc and not self._proc.is_alive():
                self._drain()
                if not self._finished:
                    self._finish("Process ended unexpectedly.")
                    self.log.appendPlainText("The worker process ended unexpectedly "
                                             "(possibly out of memory).")

        def closeEvent(self, e):
            if self._proc and self._proc.is_alive():
                self._proc.terminate()
            e.accept()

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run_cli(a):
    dst = a.output or os.path.splitext(a.src)[0] + "_spaces.ifc"
    params = Params(src=a.src, dst=dst, cut_height=a.cut, min_area=a.min_area, gap_mm=a.gap_mm,
                    number_format=a.format, first_storey_no=a.first_storey,
                    merge_multistorey=not a.no_merge, ceiling_deduction=a.deduct)
    rep = Reporter(on_status=lambda s: print(f"[..] {s}"), on_log=print)
    res = generate_spaces(params, rep)
    print(f"{res['spaces']} spaces, {res['multi']} multi-storey -> {res['dst']}")


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser(description="Generate IfcSpaces from wall geometry (no argument: GUI).")
    ap.add_argument("src", nargs="?", help="input IFC")
    ap.add_argument("-o", "--output")
    ap.add_argument("--cut", type=float, default=1.0, help="cut height above storey level in m")
    ap.add_argument("--min-area", type=float, default=1.0)
    ap.add_argument("--gap-mm", type=float, default=20.0)
    ap.add_argument("--format", default="{g:02d}.{n:03d}")
    ap.add_argument("--first-storey", type=int, default=1)
    ap.add_argument("--deduct", type=float, default=0.0)
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    if a.src:
        run_cli(a)
    else:
        run_gui()


if __name__ == "__main__":
    main()
