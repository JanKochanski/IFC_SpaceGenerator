#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFC Space Generator
===================
Creates IfcSpace objects (with room number and room name) for all fully
enclosed rooms of an IFC file, without requiring any rooms in the source model.
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
   storey, and store the room number in Name, the room name in LongName and the
   number additionally in Pset_SpaceCommon.Reference.

Two ways to work
----------------
* "Generate spaces (automatic)": analysis + export in one go, numbers are
  generated automatically (room name = room number).
* "Analyze and edit rooms": analysis first, then an interactive room editor
  opens. Per storey you see the floor plan with all detected rooms, click a
  room (or a table row) and enter room number and room name in a dialog.
  Rooms can be excluded from the export. The work can be saved as a session
  file (JSON) and continued later without re-running the analysis.

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
    python ifc_space_generator.py model.ifc        # no GUI (CLI, automatic numbering)
    python ifc_space_generator.py -h               # all options
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field

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

SESSION_VERSION = 1


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
# Data model shared by analysis, editor, session files and export
# --------------------------------------------------------------------------
@dataclass
class RoomData:
    uid: int
    base: int                # index of the storey where the space starts
    top: int                 # index of the last storey the space spans
    polys: dict              # storey index -> shapely Polygon (footprint on that storey)
    height: float            # extrusion height in m
    number: str
    name: str
    include: bool = True     # exported or not
    edited: bool = False     # touched by the user in the editor


@dataclass
class StoreyData:
    step_id: int             # STEP id of the IfcBuildingStorey in the source file
    name: str
    elev: float              # meters
    barrier: object = None   # shapely geometry of the wall section (display only)


@dataclass
class Analysis:
    src: str
    schema: str
    storeys: list = field(default_factory=list)
    rooms: list = field(default_factory=list)


def natural_key(text):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def session_to_dict(a: Analysis, params: dict) -> dict:
    def hx(g):
        return None if g is None else shapely.to_wkb(g, hex=True)
    return {
        "version": SESSION_VERSION,
        "src": a.src,
        "schema": a.schema,
        "params": params,
        "storeys": [{"id": s.step_id, "name": s.name, "elev": s.elev, "barrier": hx(s.barrier)}
                    for s in a.storeys],
        "rooms": [{"uid": r.uid, "base": r.base, "top": r.top, "height": r.height,
                   "number": r.number, "name": r.name, "include": r.include, "edited": r.edited,
                   "polys": {str(k): hx(g) for k, g in r.polys.items()}} for r in a.rooms],
    }


def session_from_dict(d: dict):
    if d.get("version") != SESSION_VERSION:
        raise ValueError("Unsupported session file version.")

    def fx(h):
        return None if h is None else shapely.from_wkb(h)
    storeys = [StoreyData(s["id"], s["name"], s["elev"], fx(s["barrier"])) for s in d["storeys"]]
    rooms = [RoomData(r["uid"], r["base"], r["top"], {int(k): fx(h) for k, h in r["polys"].items()},
                      r["height"], r["number"], r["name"], r["include"], r["edited"])
             for r in d["rooms"]]
    return Analysis(d["src"], d["schema"], storeys, rooms), d.get("params", {})


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


def _storey_name(ent):
    return ent.Name or ent.LongName or f"#{ent.id()}"


def read_storeys(model, scale, rep):
    storeys = []
    for st in model.by_type("IfcBuildingStorey"):
        if st.ObjectPlacement is None:
            continue
        M = np.array(ifcopenshell.util.placement.get_local_placement(st.ObjectPlacement), dtype=float)
        storeys.append(Storey(st, _storey_name(st), M, M[2, 3] * scale))
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


def add_space(model, ctx, oh, storey, Minv, scale, poly, z0_m, height_m, number, name):
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

    kw = dict(GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh, Name=number, LongName=name,
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
# Workflow: analysis
# --------------------------------------------------------------------------
@dataclass
class Chain:
    base: int
    top: int
    polys: dict      # storey index -> Polygon


def _analyze_model(p: Params, rep: Reporter):
    """Detect rooms. Returns (Analysis, open ifcopenshell model)."""
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
    storey_data = [StoreyData(s.ent.id(), s.name, s.elev, None) for s in storeys]

    # ---- Analyze storeys
    rooms_by_storey = []
    for k, st in enumerate(storeys):
        rep.check()
        rep.status(f"Analyzing storey {k + 1}/{n}: {st.name} ...")
        barrier = build_barrier(barrier_items, st.elev + p.cut_height, tol)
        rooms = extract_rooms(barrier, tol, p.min_area) if barrier is not None else []
        if barrier is not None:
            storey_data[k].barrier = barrier.simplify(0.01)     # for the editor background
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
                    last = ch.polys[ch.top]
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
                        ch.top = k
                        ch.polys[k] = rooms[best_j]
                        used.add(best_j)
                        nxt.append(ch)
        for j, r in enumerate(rooms):
            if j not in used:
                ch = Chain(k, k, {k: r})
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

    # ---- Default numbering (room name = room number until the user edits it)
    rooms_out = []
    uid = 0
    for k, st in enumerate(storeys):
        group = [c for c in chains if c.base == k]
        group.sort(key=lambda c: (-round(c.polys[k].centroid.y, 1), c.polys[k].centroid.x))
        for nr, ch in enumerate(group, start=1):
            number = p.number_format.format(g=p.first_storey_no + k, s=st.name, n=nr)
            z_top = storeys[ch.top + 1].elev if ch.top + 1 < n else storeys[ch.top].elev + last_h
            height = max(z_top - st.elev - p.ceiling_deduction, 0.5)
            rooms_out.append(RoomData(uid, k, ch.top, ch.polys, height, number, number))
            uid += 1

    return Analysis(p.src, model.schema, storey_data, rooms_out), model


def analyze(p: Params, rep: Reporter) -> Analysis:
    analysis, _model = _analyze_model(p, rep)
    rep.log(f"Analysis finished: {len(analysis.rooms)} rooms.")
    return analysis


# --------------------------------------------------------------------------
# Workflow: export
# --------------------------------------------------------------------------
def write_ifc(analysis: Analysis, dst: str, rep: Reporter, model=None,
              lo: float = 0.0, hi: float = 100.0) -> dict:
    """Write all included rooms of an analysis as IfcSpaces into a copy of the source IFC."""
    t0 = time.time()
    if os.path.abspath(analysis.src) == os.path.abspath(dst):
        raise ValueError("The output file must not be the input file.")
    if model is None:
        rep.status("Loading IFC file ... (this can take a moment for large files)")
        rep.progress(None)
        model = ifcopenshell.open(analysis.src)
    scale = ifcopenshell.util.unit.calculate_unit_scale(model)

    ents, minvs = [], []
    for sd in analysis.storeys:
        try:
            ent = model.by_id(sd.step_id)
        except Exception:
            ent = None
        if ent is None or not ent.is_a("IfcBuildingStorey") or _storey_name(ent) != sd.name:
            raise ValueError(f"Storey '{sd.name}' (#{sd.step_id}) does not match the IFC file. "
                             f"Was the file changed since the analysis/session was created?")
        M = np.array(ifcopenshell.util.placement.get_local_placement(ent.ObjectPlacement), dtype=float)
        ents.append(ent)
        minvs.append(np.linalg.inv(M))

    oh = model.by_type("IfcOwnerHistory")[0] if model.by_type("IfcOwnerHistory") else None
    ctx = _body_context(model)

    rooms = [r for r in analysis.rooms if r.include]
    dups = sorted([num for num, c in Counter(r.number for r in rooms).items() if c > 1], key=natural_key)
    if dups:
        rep.log("WARNING: duplicate room numbers: " + ", ".join(dups))

    rep.status("Writing IfcSpaces ...")
    per_storey_spaces, per_storey_count = {}, {}
    total = len(rooms)
    for i, r in enumerate(rooms):
        rep.check()
        sd = analysis.storeys[r.base]
        sp = add_space(model, ctx, oh, ents[r.base], minvs[r.base], scale, r.polys[r.base],
                       sd.elev, r.height, r.number, r.name)
        per_storey_spaces.setdefault(r.base, []).append(sp)
        per_storey_count[sd.name] = per_storey_count.get(sd.name, 0) + 1
        if i % 20 == 0:
            rep.progress(lo + (hi - lo) * 0.9 * i / max(total, 1))
            rep.status(f"Writing IfcSpaces ... {i}/{total}")

    for k, spaces in per_storey_spaces.items():
        st = ents[k]
        rel = next((x for x in (st.IsDecomposedBy or ()) if x.is_a("IfcRelAggregates")), None)
        if rel is not None:
            rel.RelatedObjects = tuple(rel.RelatedObjects) + tuple(spaces)
        else:
            model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=oh,
                                RelatingObject=st, RelatedObjects=spaces)

    rep.status("Saving IFC file ...")
    rep.progress(None)
    model.write(dst)
    dt = time.time() - t0
    rep.progress(hi)
    rep.status("Done.")
    rep.log(f"Saved: {dst} ({dt:.0f} s)")
    return {"spaces": len(rooms), "multi": sum(1 for r in rooms if r.top > r.base),
            "per_storey": per_storey_count, "dst": dst, "seconds": dt}


def generate_spaces(p: Params, rep: Reporter) -> dict:
    """Automatic mode: analysis + export in one go."""
    t0 = time.time()
    if os.path.abspath(p.src) == os.path.abspath(p.dst):
        raise ValueError("The output file must not be the input file.")
    analysis, model = _analyze_model(p, rep)
    res = write_ifc(analysis, p.dst, rep, model=model, lo=90.0, hi=100.0)
    res["seconds"] = time.time() - t0
    return res


# --------------------------------------------------------------------------
# Subprocess entry point (keeps the GUI responsive during long computations)
# --------------------------------------------------------------------------
def _worker_main(mode, params, analysis, q):
    rep = Reporter(on_progress=lambda v: q.put(("progress", v)),
                   on_status=lambda s: q.put(("status", s)),
                   on_log=lambda s: q.put(("log", s)))
    try:
        if mode == "generate":
            res = generate_spaces(params, rep)
        elif mode == "analyze":
            res = analyze(params, rep)
        elif mode == "export":
            res = write_ifc(analysis, params.dst, rep)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        q.put(("done", res))
    except Exception:
        q.put(("error", traceback.format_exc()))


# --------------------------------------------------------------------------
# GUI: room editor (PySide6 classes are created lazily so the CLI works without Qt)
# --------------------------------------------------------------------------
def _make_editor_class(QtCore, QtGui, QtWidgets):
    Qt = QtCore.Qt

    def geom_to_path(g):
        """shapely geometry -> QPainterPath (y is flipped so that north points up)."""
        path = QtGui.QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        for poly in _polys(g):
            for ring in [poly.exterior, *poly.interiors]:
                pts = [QtCore.QPointF(c[0], -c[1]) for c in ring.coords]
                path.addPolygon(QtGui.QPolygonF(pts))
                path.closeSubpath()
        return path

    class PlanView(QtWidgets.QGraphicsView):
        """Zoom with the mouse wheel, pan with the middle or right mouse button."""

        def __init__(self, scene):
            super().__init__(scene)
            self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(245, 245, 245)))
            self._pan = None

        def wheelEvent(self, e):
            f = 1.0015 ** e.angleDelta().y()
            self.scale(f, f)

        def mousePressEvent(self, e):
            if e.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
                self._pan = e.position().toPoint()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                e.accept()
                return
            super().mousePressEvent(e)

        def mouseMoveEvent(self, e):
            if self._pan is not None:
                pos = e.position().toPoint()
                d = pos - self._pan
                self._pan = pos
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
                e.accept()
                return
            super().mouseMoveEvent(e)

        def mouseReleaseEvent(self, e):
            if self._pan is not None and e.button() in (Qt.MouseButton.MiddleButton,
                                                        Qt.MouseButton.RightButton):
                self._pan = None
                self.unsetCursor()
                e.accept()
                return
            super().mouseReleaseEvent(e)

        def contextMenuEvent(self, e):
            e.accept()      # right button is used for panning

    class RoomItem(QtWidgets.QGraphicsPathItem):
        """Clickable room polygon in the floor plan."""

        def __init__(self, editor, room, k):
            super().__init__(geom_to_path(room.polys[k]))
            self.editor, self.room, self.k = editor, room, k
            self.selected = False
            self.hover = False
            self.setAcceptHoverEvents(True)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.restyle()

        def restyle(self):
            r = self.room
            if self.selected:
                col, pen_col, w = QtGui.QColor(255, 160, 0, 170), QtGui.QColor(210, 90, 0), 3.0
            elif not r.include:
                col, pen_col, w = QtGui.QColor(140, 140, 140, 80), QtGui.QColor(120, 120, 120), 1.5
            elif r.edited:
                col, pen_col, w = QtGui.QColor(70, 175, 95, 120), QtGui.QColor(30, 120, 60), 1.5
            else:
                col, pen_col, w = QtGui.QColor(70, 130, 220, 100), QtGui.QColor(30, 80, 170), 1.5
            if self.hover and not self.selected:
                col.setAlpha(min(255, col.alpha() + 70))
            if self.k != r.base:                      # continuation of a room from a lower storey
                col.setAlpha(200)
            brush = QtGui.QBrush(col)
            if self.k != r.base:
                brush.setStyle(Qt.BrushStyle.BDiagPattern)
            pen = QtGui.QPen(pen_col)
            pen.setWidthF(w)
            pen.setCosmetic(True)
            if not r.include:
                pen.setStyle(Qt.PenStyle.DashLine)
            self.setBrush(brush)
            self.setPen(pen)
            self.setZValue(5 if self.selected else 1)

        def set_selected(self, value):
            self.selected = value
            self.restyle()

        def hoverEnterEvent(self, e):
            self.hover = True
            self.restyle()

        def hoverLeaveEvent(self, e):
            self.hover = False
            self.restyle()

        def mousePressEvent(self, e):
            if e.button() == Qt.MouseButton.LeftButton:
                self.editor.select_room(self.room.uid)
                e.accept()
            else:
                e.ignore()

        def mouseDoubleClickEvent(self, e):
            if e.button() == Qt.MouseButton.LeftButton:
                self.editor.edit_room(self.room.uid)
                e.accept()

    class RoomDialog(QtWidgets.QDialog):
        """Input window for the data of one room."""

        def __init__(self, room, storey_label, taken_numbers, parent=None,
                     same_default=None, include_default=None):
            super().__init__(parent)
            self.setWindowTitle("Room data")
            self.setModal(True)
            self.room, self.taken = room, taken_numbers
            self.ok = False
            self.advance = False

            lay = QtWidgets.QVBoxLayout(self)
            span = "" if room.top == room.base else f"  |  spans {room.top - room.base + 1} storeys"
            lay.addWidget(QtWidgets.QLabel(
                f"<b>Storey:</b> {storey_label}  |  <b>Area:</b> {room.polys[room.base].area:.1f} m²{span}"))

            form = QtWidgets.QFormLayout()
            self.num = QtWidgets.QLineEdit(room.number)
            self.name = QtWidgets.QLineEdit(room.name)
            self.same = QtWidgets.QCheckBox("Use room number as room name")
            # Already edited rooms show their actual state; for all others the choice made
            # in the previous dialog is carried over (if there was one).
            keep_actual = room.edited or same_default is None
            self.same.setChecked(room.name == room.number if keep_actual else same_default)
            self.inc = QtWidgets.QCheckBox("Include this room in the export")
            keep_actual = room.edited or include_default is None
            self.inc.setChecked(room.include if keep_actual else include_default)
            form.addRow("Room number:", self.num)
            form.addRow("Room name:", self.name)
            form.addRow("", self.same)
            form.addRow("", self.inc)
            lay.addLayout(form)

            row = QtWidgets.QHBoxLayout()
            b_ok = QtWidgets.QPushButton("OK")
            b_ok.setDefault(True)
            b_next = QtWidgets.QPushButton("OK and next room")
            b_next.setShortcut("Ctrl+Return")
            b_cancel = QtWidgets.QPushButton("Cancel")
            row.addStretch(1)
            row.addWidget(b_ok)
            row.addWidget(b_next)
            row.addWidget(b_cancel)
            lay.addLayout(row)
            b_ok.clicked.connect(lambda: self._commit(False))
            b_next.clicked.connect(lambda: self._commit(True))
            b_cancel.clicked.connect(self.reject)

            self.num.textChanged.connect(self._sync)
            self.same.toggled.connect(self._sync)
            self._sync()
            self.num.setFocus()
            self.num.selectAll()

        def _sync(self, *_):
            self.name.setEnabled(not self.same.isChecked())
            if self.same.isChecked():
                self.name.setText(self.num.text())

        def _commit(self, advance):
            number = self.num.text().strip()
            if not number:
                QtWidgets.QMessageBox.warning(self, "Room data", "The room number must not be empty.")
                return
            if self.inc.isChecked() and number in self.taken:
                answer = QtWidgets.QMessageBox.question(
                    self, "Duplicate room number",
                    f"The room number '{number}' is already used by another room. Use it anyway?")
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            self.ok = True
            self.advance = advance
            self.accept()

        def values(self):
            number = self.num.text().strip()
            name = number if self.same.isChecked() else self.name.text().strip()
            return number, name, self.inc.isChecked()

    class RoomEditor(QtWidgets.QDialog):
        """Floor plan per storey + room table + input dialog for room number and name."""

        def __init__(self, analysis, params, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Room editor")
            self.resize(1280, 820)
            self.a = analysis
            self.params = params
            self.rooms = {r.uid: r for r in analysis.rooms}
            self.k = 0
            self.sel = None
            self.items, self.labels, self.row_of = {}, {}, {}
            self.dirty = False
            self.wants_export = False
            self.export_dst = None
            self.session_path = None
            self.pref_same = None        # last state of "use room number as room name"
            self.pref_include = None     # last state of "include in export"
            self._first_show = True

            lay = QtWidgets.QVBoxLayout(self)

            top = QtWidgets.QHBoxLayout()
            top.addWidget(QtWidgets.QLabel("Storey:"))
            self.combo = QtWidgets.QComboBox()
            top.addWidget(self.combo, 1)
            self.chk_labels = QtWidgets.QCheckBox("Show labels")
            self.chk_labels.setChecked(True)
            self.chk_labels.toggled.connect(self._toggle_labels)
            b_fit = QtWidgets.QPushButton("Fit view")
            b_fit.clicked.connect(self.fit)
            top.addWidget(self.chk_labels)
            top.addWidget(b_fit)
            lay.addLayout(top)

            self.scene = QtWidgets.QGraphicsScene(self)
            self.view = PlanView(self.scene)

            right = QtWidgets.QWidget()
            rl = QtWidgets.QVBoxLayout(right)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(QtWidgets.QLabel("Rooms on this storey (double-click to edit):"))
            self.table = QtWidgets.QTableWidget(0, 4)
            self.table.setHorizontalHeaderLabels(["Number", "Name", "m²", "Done"])
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
            self.table.itemSelectionChanged.connect(self._on_table_selection)
            self.table.cellDoubleClicked.connect(self._on_table_double_click)
            rl.addWidget(self.table, 1)
            legend = QtWidgets.QLabel(
                "Blue = not yet edited, green = edited, gray dashed = excluded, "
                "hatched = room continues from a lower storey, red number = duplicate.")
            legend.setWordWrap(True)
            rl.addWidget(legend)

            split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
            split.addWidget(self.view)
            split.addWidget(right)
            split.setStretchFactor(0, 4)
            split.setStretchFactor(1, 1)
            split.setSizes([920, 340])
            lay.addWidget(split, 1)

            hint = QtWidgets.QLabel("Click = select  |  Double-click = edit room data  |  "
                                    "Mouse wheel = zoom  |  Right/middle mouse button drag = pan")
            lay.addWidget(hint)

            bottom = QtWidgets.QHBoxLayout()
            self.status = QtWidgets.QLabel("")
            b_save = QtWidgets.QPushButton("Save session ...")
            b_save.clicked.connect(self.save_session)
            b_export = QtWidgets.QPushButton("Export IFC ...")
            b_export.clicked.connect(self.export)
            b_close = QtWidgets.QPushButton("Close")
            b_close.clicked.connect(self.reject)
            bottom.addWidget(self.status, 1)
            bottom.addWidget(b_save)
            bottom.addWidget(b_export)
            bottom.addWidget(b_close)
            lay.addLayout(bottom)

            for k in range(len(self.a.storeys)):
                self.combo.addItem(self._combo_text(k))
            first = next((r.base for r in sorted(self.a.rooms, key=lambda r: r.base)), 0)
            self.combo.setCurrentIndex(first)
            self.combo.currentIndexChanged.connect(self.show_storey)
            self.show_storey(first)

        # ---------- helpers
        def _combo_text(self, k):
            st = self.a.storeys[k]
            own = [r for r in self.a.rooms if r.base == k]
            done = sum(1 for r in own if r.edited)
            return f"{st.name} ({st.elev:+.2f} m) - {done}/{len(own)} edited"

        def _update_combo_texts(self):
            self.combo.blockSignals(True)
            for k in range(len(self.a.storeys)):
                self.combo.setItemText(k, self._combo_text(k))
            self.combo.blockSignals(False)

        def rooms_on(self, k):
            rs = [r for r in self.a.rooms if r.base <= k <= r.top and k in r.polys]
            return sorted(rs, key=lambda r: natural_key(r.number))

        def _label_text(self, r):
            return r.number if (not r.name or r.name == r.number) else f"{r.number}\n{r.name}"

        def _make_label(self, r, k):
            lbl = QtWidgets.QGraphicsSimpleTextItem(self._label_text(r))
            font = QtGui.QFont()
            font.setPointSize(8)
            font.setBold(True)
            lbl.setFont(font)
            lbl.setBrush(QtGui.QBrush(QtGui.QColor(20, 20, 20)))
            lbl.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            lbl.setAcceptedMouseButtons(Qt.MouseButton.NoButton)   # clicks go to the room below
            p = r.polys[k].representative_point()
            lbl.setPos(p.x, -p.y)
            br = lbl.boundingRect()
            lbl.setTransform(QtGui.QTransform.fromTranslate(-br.width() / 2, -br.height() / 2))
            lbl.setZValue(10)
            lbl.setVisible(self.chk_labels.isChecked())
            return lbl

        def _toggle_labels(self, on):
            for lbl in self.labels.values():
                lbl.setVisible(on)

        # ---------- plan
        def show_storey(self, k):
            if k < 0:
                return
            self.k = k
            self.scene.clear()
            self.items.clear()
            self.labels.clear()
            st = self.a.storeys[k]
            if st.barrier is not None and not st.barrier.is_empty:
                walls = self.scene.addPath(geom_to_path(st.barrier), QtGui.QPen(Qt.PenStyle.NoPen),
                                           QtGui.QBrush(QtGui.QColor(70, 70, 70)))
                walls.setZValue(0)
                walls.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            for r in self.rooms_on(k):
                item = RoomItem(self, r, k)
                self.scene.addItem(item)
                self.items[r.uid] = item
                lbl = self._make_label(r, k)
                self.scene.addItem(lbl)
                self.labels[r.uid] = lbl
            rect = self.scene.itemsBoundingRect()
            self.scene.setSceneRect(rect.adjusted(-rect.width(), -rect.height(), rect.width(), rect.height()))
            if self.sel not in self.items:
                self.sel = None
            elif self.sel is not None:
                self.items[self.sel].set_selected(True)
            self.refresh_table()
            self.fit()

        def fit(self):
            rect = self.scene.itemsBoundingRect()
            if not rect.isNull():
                self.view.fitInView(rect.adjusted(-1, -1, 1, 1), Qt.AspectRatioMode.KeepAspectRatio)

        def showEvent(self, e):
            super().showEvent(e)
            if self._first_show:
                self._first_show = False
                QtCore.QTimer.singleShot(0, self.fit)

        # ---------- table
        def refresh_table(self):
            rooms = self.rooms_on(self.k)
            counts = Counter(r.number for r in self.a.rooms if r.include)
            t = self.table
            t.blockSignals(True)
            t.setRowCount(len(rooms))
            self.row_of = {}
            for i, r in enumerate(rooms):
                self.row_of[r.uid] = i
                cells = [r.number, r.name, f"{r.polys[self.k].area:.1f}", "yes" if r.edited else ""]
                for c, text in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(text)
                    if c == 0:
                        it.setData(Qt.ItemDataRole.UserRole, r.uid)
                    if not r.include:
                        it.setForeground(QtGui.QBrush(QtGui.QColor(140, 140, 140)))
                    elif c == 0 and counts[r.number] > 1:
                        it.setForeground(QtGui.QBrush(QtGui.QColor(200, 0, 0)))
                    t.setItem(i, c, it)
            if self.sel in self.row_of:
                t.selectRow(self.row_of[self.sel])
            t.blockSignals(False)
            t.resizeColumnToContents(0)

        def _on_table_selection(self):
            rows = self.table.selectionModel().selectedRows()
            if rows:
                uid = self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
                self.select_room(uid, center=True, sync_table=False)

        def _on_table_double_click(self, row, _col):
            uid = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.edit_room(uid)

        # ---------- selection / editing
        def select_room(self, uid, center=False, sync_table=True):
            old = self.sel
            self.sel = uid
            if old in self.items:
                self.items[old].set_selected(False)
            item = self.items.get(uid)
            if item is not None:
                item.set_selected(True)
                if center:
                    self.view.centerOn(item)
            if sync_table and uid in self.row_of:
                self.table.blockSignals(True)
                self.table.selectRow(self.row_of[uid])
                self.table.blockSignals(False)

        def _next_uid(self, uid):
            order = [r.uid for r in self.rooms_on(self.k)]
            if uid in order and order.index(uid) + 1 < len(order):
                return order[order.index(uid) + 1]
            return None

        def edit_room(self, uid):
            while True:
                room = self.rooms[uid]
                self.select_room(uid, center=True)
                taken = {r.number for r in self.a.rooms if r.uid != uid and r.include}
                dlg = RoomDialog(room, self.a.storeys[room.base].name, taken, self,
                                 same_default=self.pref_same, include_default=self.pref_include)
                dlg.exec()
                if not dlg.ok:
                    return
                self.pref_same = dlg.same.isChecked()
                self.pref_include = dlg.inc.isChecked()
                room.number, room.name, room.include = dlg.values()
                room.edited = True
                self.dirty = True
                self._refresh_after_edit(uid)
                if not dlg.advance:
                    return
                uid = self._next_uid(uid)
                if uid is None:
                    return

        def _refresh_after_edit(self, uid):
            room = self.rooms[uid]
            if uid in self.items:
                self.items[uid].restyle()
                lbl = self.labels[uid]
                lbl.setText(self._label_text(room))
                br = lbl.boundingRect()
                lbl.setTransform(QtGui.QTransform.fromTranslate(-br.width() / 2, -br.height() / 2))
            self.refresh_table()
            self._update_combo_texts()

        # ---------- session / export / close
        def save_session(self):
            default = self.session_path or os.path.splitext(self.a.src)[0] + "_rooms.json"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save session", default,
                                                            "Session (*.json)")
            if not path:
                return False
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(session_to_dict(self.a, self.params), f)
            except Exception as ex:
                QtWidgets.QMessageBox.critical(self, "Save session", str(ex))
                return False
            self.session_path = path
            self.dirty = False
            self.status.setText(f"Session saved: {path}")
            return True

        def export(self):
            if not os.path.isfile(self.a.src):
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Locate the source IFC file", "", "IFC (*.ifc);;All files (*)")
                if not path:
                    return
                self.a.src = path
            included = [r for r in self.a.rooms if r.include]
            dups = [n for n, c in Counter(r.number for r in included).items() if c > 1]
            if dups:
                answer = QtWidgets.QMessageBox.question(
                    self, "Duplicate room numbers",
                    f"{len(dups)} room number(s) are used more than once. Export anyway?")
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            default = self.params.get("dst") or os.path.splitext(self.a.src)[0] + "_spaces.ifc"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export IFC", default, "IFC (*.ifc)")
            if not path:
                return
            if os.path.abspath(path) == os.path.abspath(self.a.src):
                QtWidgets.QMessageBox.warning(self, "Export", "The output must not be the input file.")
                return
            self.export_dst = path
            self.wants_export = True
            self.accept()

        def reject(self):
            if self.dirty:
                B = QtWidgets.QMessageBox.StandardButton
                answer = QtWidgets.QMessageBox.question(
                    self, "Close editor", "Save the session before closing?",
                    B.Save | B.Discard | B.Cancel)
                if answer == B.Cancel:
                    return
                if answer == B.Save and not self.save_session():
                    return
            super().reject()

    return RoomEditor


# --------------------------------------------------------------------------
# GUI: main window (PySide6)
# --------------------------------------------------------------------------
def run_gui():
    import multiprocessing as mp
    from PySide6 import QtCore, QtGui, QtWidgets

    ctx = mp.get_context("spawn")
    RoomEditor = _make_editor_class(QtCore, QtGui, QtWidgets)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("IFC Space Generator")
            self.resize(900, 720)
            self.setAcceptDrops(True)
            self._proc = None
            self._queue = None
            self._t0 = 0.0
            self._finished = True
            self._mode = ""
            self.analysis = None
            self.params_dict = {}

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
            self.merge.setChecked(True)
            of.addRow("Cut height above storey level:", self.cut)
            of.addRow("Minimum room area:", self.min_area)
            of.addRow("Close gaps up to:", self.gap)
            of.addRow("Numbering scheme (initial numbers):", self.fmt)
            of.addRow("Number of the lowest storey:", self.first)
            of.addRow("Subtract slab thickness from room height:", self.deduct)
            of.addRow("", self.merge)
            lay.addWidget(opts)

            row = QtWidgets.QHBoxLayout()
            self.btn_gen = QtWidgets.QPushButton("Generate spaces (automatic)")
            self.btn_gen.setToolTip("Analysis and export in one go, rooms are numbered automatically.")
            self.btn_analyze = QtWidgets.QPushButton("Analyze and edit rooms ...")
            self.btn_analyze.setToolTip("Detect rooms, then edit room numbers and names per storey.")
            self.btn_load = QtWidgets.QPushButton("Load session ...")
            self.btn_reopen = QtWidgets.QPushButton("Reopen editor")
            self.btn_reopen.setEnabled(False)
            self.btn_cancel = QtWidgets.QPushButton("Cancel")
            self.btn_cancel.setEnabled(False)
            for b in (self.btn_gen, self.btn_analyze, self.btn_load, self.btn_reopen, self.btn_cancel):
                b.setMinimumHeight(34)
                row.addWidget(b)
            self.btn_gen.clicked.connect(self.start_generate)
            self.btn_analyze.clicked.connect(self.start_analyze)
            self.btn_load.clicked.connect(self.load_session)
            self.btn_reopen.clicked.connect(self.open_editor)
            self.btn_cancel.clicked.connect(self.cancel)
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

        # ---------- Parameters
        def _collect_params(self, need_dst):
            src = self.in_edit.text().strip()
            if not os.path.isfile(src):
                QtWidgets.QMessageBox.warning(self, "Input", "Please choose an existing IFC file.")
                return None
            dst = self.out_edit.text().strip() or os.path.splitext(src)[0] + "_spaces.ifc"
            if need_dst and os.path.abspath(dst) == os.path.abspath(src):
                QtWidgets.QMessageBox.warning(self, "Output", "The output must not be the input file.")
                return None
            try:
                self.fmt.text().format(g=1, s="X", n=1)
            except Exception as ex:
                QtWidgets.QMessageBox.warning(self, "Numbering scheme", f"Invalid: {ex}")
                return None
            return Params(src=src, dst=dst, cut_height=self.cut.value(), min_area=self.min_area.value(),
                          gap_mm=float(self.gap.value()), number_format=self.fmt.text(),
                          first_storey_no=self.first.value(), merge_multistorey=self.merge.isChecked(),
                          ceiling_deduction=self.deduct.value())

        # ---------- Workflow
        def start_generate(self):
            p = self._collect_params(need_dst=True)
            if p:
                self.log.clear()
                self._run("generate", p, None)

        def start_analyze(self):
            p = self._collect_params(need_dst=False)
            if p:
                if self.analysis is not None and any(r.edited for r in self.analysis.rooms):
                    answer = QtWidgets.QMessageBox.question(
                        self, "Analyze again",
                        "A new analysis replaces the rooms currently in memory, including your edits "
                        "(save a session first if you want to keep them). Continue?")
                    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                        return
                self.log.clear()
                self.params_dict = asdict(p)
                self._run("analyze", p, None)

        def load_session(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load session", os.path.dirname(self.in_edit.text()), "Session (*.json)")
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.analysis, self.params_dict = session_from_dict(json.load(f))
            except Exception as ex:
                QtWidgets.QMessageBox.critical(self, "Load session", f"Could not load the session:\n{ex}")
                return
            if self.analysis.src:
                self.in_edit.setText(self.analysis.src)
            if self.params_dict.get("dst"):
                self.out_edit.setText(self.params_dict["dst"])
            self.log.appendPlainText(f"Session loaded: {path} ({len(self.analysis.rooms)} rooms)")
            self.btn_reopen.setEnabled(True)
            self.open_editor()

        def open_editor(self):
            if self.analysis is None:
                return
            ed = RoomEditor(self.analysis, self.params_dict, self)
            ed.exec()
            if ed.wants_export:
                self.params_dict["dst"] = ed.export_dst
                self.out_edit.setText(ed.export_dst)
                self._run("export", Params(src=self.analysis.src, dst=ed.export_dst), self.analysis)

        def _run(self, mode, params, analysis):
            self._mode = mode
            self._queue = ctx.Queue()
            self._proc = ctx.Process(target=_worker_main, args=(mode, params, analysis, self._queue),
                                     daemon=True)
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
            for b in (self.btn_gen, self.btn_analyze, self.btn_load):
                b.setEnabled(not running)
            self.btn_reopen.setEnabled((not running) and self.analysis is not None)
            self.btn_cancel.setEnabled(running)

        def _finish(self, message):
            self._finished = True
            self.timer.stop()
            self._set_running(False)
            self.bar.setRange(0, 1000)
            self.statusBar().showMessage(message)

        def _on_done(self, payload):
            if self._mode == "analyze":
                self.analysis = payload
                self.btn_reopen.setEnabled(True)
                self.log.appendPlainText(f"{len(payload.rooms)} rooms detected - opening the room editor ...")
                self.open_editor()
            else:
                per = "\n".join(f"  {k}: {v}" for k, v in payload["per_storey"].items())
                QtWidgets.QMessageBox.information(
                    self, "Done",
                    f"Created {payload['spaces']} IfcSpaces "
                    f"({payload['multi']} multi-storey).\n\n{per}\n\n{payload['dst']}")

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
                        # defer so that no dialog opens while this loop still reads the queue
                        QtCore.QTimer.singleShot(0, lambda p=payload: self._on_done(p))
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
