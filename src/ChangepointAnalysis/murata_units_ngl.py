"""Interactive NGL HTML of locked Murata cation–π and equatorial units.

Mirrors ``write_transition_sites_ngl_html``: licorice structure, centroid
spheres + cylinders, click-to-solo sidebar. Units are the Hungarian-locked
contacts:

* cation–π unit *k*: π(pole) pole Py⁺ *i* → Ph *j* and π(equator) eq Py⁺ *k* → Ph *j*
* equator unit *k*: d1 R2 *i* → R3 *j* and d2 CPy *i* → R3 *j*
  (CPy = eq Py⁺ carbon para to the phenylene linker)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .endpoint_features import classify_paper_d1_state
from .murata_criteria import (
    CATION_PI_OPEN_LO,
    _cpy_point,
    _indices_for,
    _role_point,
    _scalar_index,
    angle_at_vertex_deg,
    assign_equatorial_pi_partners,
    cation_pi_unit_indices,
    cation_pi_unit_label,
    equator_unit_indices,
    equator_unit_label,
    lock_cation_pi_sandwiches,
    lock_equator_edges,
    resolve_murata_criteria_atoms,
)

MURATA_ROLE_COLORS: dict[str, str] = {
    "py_pole": "#911eb4",
    "ph": "#f5c542",
    "py_eq": "#42d4f4",
    "r2": "#4363d8",
    "r3": "#f58231",
    "cpy": "#00d4aa",
}

# Distinct scaffold colors for the six GSA monomers (NGL licorice).
MONOMER_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
]

_UNIT_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#800000",
]

_ROLE_ORDER = ("py_pole", "ph", "py_eq", "r2", "r3")


def classify_cation_pi_state(
    distance: float, *, open_lo: float = CATION_PI_OPEN_LO
) -> str:
    if not np.isfinite(distance):
        return "unknown"
    return "open" if float(distance) >= float(open_lo) else "closed"


def sandwiches_from_units_df(
    units_df: pd.DataFrame,
) -> list[tuple[int, int, int]]:
    """Read ``(pole, ph, eq)`` sandwiches; ``eq`` is -1 when ``mon_c`` is missing."""
    out: list[tuple[int, int, int]] = []
    if units_df is None or units_df.empty:
        return out
    ordered = units_df.sort_values(["kind", "edge"], kind="mergesort")
    for _, row in ordered.iterrows():
        if str(row["kind"]).strip().lower() != "cation_pi":
            continue
        eq = row["mon_c"] if "mon_c" in row.index and pd.notna(row["mon_c"]) else -1
        out.append((int(row["mon_a"]), int(row["mon_b"]), int(eq)))
    return out


def edges_from_units_df(
    units_df: pd.DataFrame,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Read locked ``(mon_a, mon_b)`` pairs from a ``*_motif_units.csv``."""
    cation: list[tuple[int, int]] = []
    equator: list[tuple[int, int]] = []
    if units_df is None or units_df.empty:
        return cation, equator
    ordered = units_df.sort_values(["kind", "edge"], kind="mergesort")
    for _, row in ordered.iterrows():
        pair = (int(row["mon_a"]), int(row["mon_b"]))
        kind = str(row["kind"]).strip().lower()
        if kind == "cation_pi":
            cation.append(pair)
        elif kind == "equator":
            equator.append(pair)
    return cation, equator


def _escape_pdb_for_js(pdb_text: str) -> str:
    return pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def _atoms_to_pdb_string(atoms: Any) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        atoms.write(str(tmp_path))
        return tmp_path.read_text(encoding="utf-8", errors="replace")
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _indices_to_ngl(
    universe: Any,
    indices: Sequence[int],
    index_to_pos: dict[int, int],
) -> tuple[list[int], Optional[list[float]]]:
    idx = [int(i) for i in indices]
    if not idx:
        return [], None
    try:
        atoms = universe.atoms[idx]
    except Exception:
        return [], None
    if len(atoms) == 0:
        return [], None
    ngl = sorted(
        {index_to_pos[int(a.index)] for a in atoms if int(a.index) in index_to_pos}
    )
    if not ngl:
        return [], None
    centroid = np.asarray(atoms.positions, dtype=float).mean(axis=0)
    return ngl, [float(centroid[0]), float(centroid[1]), float(centroid[2])]


def _midpoint(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]


def monomer_color(monomer: int) -> str:
    return MONOMER_COLORS[int(monomer) % len(MONOMER_COLORS)]


def _monomer_backbone_groups(
    universe: Any,
    roles_df: pd.DataFrame,
    index_to_pos: dict[int, int],
    *,
    monomer_selections: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """NGL atom groups for per-monomer backbone (full scaffold) coloring."""
    groups: list[dict[str, Any]] = []
    if monomer_selections:
        for i, sel in enumerate(monomer_selections):
            try:
                ag = universe.select_atoms(str(sel))
            except Exception:
                continue
            ngl = sorted(
                {
                    index_to_pos[int(a.index)]
                    for a in ag
                    if int(a.index) in index_to_pos
                }
            )
            if not ngl:
                continue
            groups.append(
                {
                    "label": f"M{i}",
                    "monomer": int(i),
                    "color": monomer_color(i),
                    "atoms": ngl,
                }
            )
        if groups:
            return groups
    monomers = sorted(set(int(m) for m in roles_df["monomer"]))
    for mon in monomers:
        idx: list[int] = []
        sub = roles_df[roles_df["monomer"].astype(int) == int(mon)]
        for role in sub["role"].astype(str):
            try:
                idx.extend(_indices_for(roles_df, mon, role))
            except KeyError:
                continue
        ngl, _ = _indices_to_ngl(universe, idx, index_to_pos)
        if not ngl:
            continue
        groups.append(
            {
                "label": f"M{mon}",
                "monomer": int(mon),
                "color": monomer_color(mon),
                "atoms": ngl,
            }
        )
    return groups


def _cpy_atom_indices(roles_df: pd.DataFrame, monomer: int) -> list[int]:
    """MDA indices of the CPy atom; fall back to the equatorial Py⁺ ring."""
    try:
        return [_scalar_index(roles_df, monomer, "py_eq", "cpy_index")]
    except KeyError:
        return _indices_for(roles_df, monomer, "py_eq")


def _site_groups_for_roles(
    universe: Any,
    roles_df: pd.DataFrame,
    index_to_pos: dict[int, int],
) -> list[dict[str, Any]]:
    from src.EndpointAnalyzer.gsa_site_map import plot_label_for_role

    groups: list[dict[str, Any]] = []
    monomers = sorted(set(int(m) for m in roles_df["monomer"]))
    for role in _ROLE_ORDER:
        ngl: set[int] = set()
        for mon in monomers:
            try:
                idxs = _indices_for(roles_df, mon, role)
            except KeyError:
                continue
            found, _ = _indices_to_ngl(universe, idxs, index_to_pos)
            ngl.update(found)
        if not ngl:
            continue
        groups.append(
            {
                "label": plot_label_for_role(role),
                "role": role,
                "color": MURATA_ROLE_COLORS[role],
                "atoms": sorted(ngl),
            }
        )
    cpy_ngl: set[int] = set()
    for mon in monomers:
        try:
            found, _ = _indices_to_ngl(
                universe, _cpy_atom_indices(roles_df, mon), index_to_pos
            )
        except KeyError:
            continue
        cpy_ngl.update(found)
    if cpy_ngl:
        groups.append(
            {
                "label": "CPy (para to benzene linker)",
                "role": "cpy",
                "color": MURATA_ROLE_COLORS["cpy"],
                "atoms": sorted(cpy_ngl),
            }
        )
    return groups


def _cation_sandwiches(
    pos: np.ndarray,
    roles_df: pd.DataFrame,
    cation_edges: Sequence[tuple[int, ...]],
) -> list[tuple[int, int, int]]:
    if not cation_edges:
        return lock_cation_pi_sandwiches(pos, roles_df)
    first = cation_edges[0]
    if len(first) >= 3 and int(first[2]) >= 0:
        return [(int(a), int(b), int(c)) for a, b, c in cation_edges]
    pairs = [(int(edge[0]), int(edge[1])) for edge in cation_edges]
    return assign_equatorial_pi_partners(pos, roles_df, pairs)


def build_murata_unit_payloads(
    universe: Any,
    roles_df: pd.DataFrame,
    *,
    cation_edges: Sequence[tuple[int, ...]],
    equator_edges: Sequence[tuple[int, int]],
    index_to_pos: dict[int, int],
    cation_pi_open_lo: float = CATION_PI_OPEN_LO,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
) -> list[dict[str, Any]]:
    """Centroid / NGL-index records for locked cation–π and equator units."""
    pos = np.asarray(universe.atoms.positions, dtype=float)
    pairs: list[dict[str, Any]] = []
    color_i = 0
    sandwiches = _cation_sandwiches(pos, roles_df, cation_edges)
    for k, (i, j, eq_m) in enumerate(sandwiches):
        ngl_i, _ = _indices_to_ngl(
            universe, _indices_for(roles_df, i, "py_pole"), index_to_pos
        )
        ngl_j, _ = _indices_to_ngl(
            universe, _indices_for(roles_df, j, "ph"), index_to_pos
        )
        ngl_eq, _ = _indices_to_ngl(
            universe, _indices_for(roles_df, eq_m, "py_eq"), index_to_pos
        )
        pole = np.asarray(pos[_indices_for(roles_df, i, "py_pole")], dtype=float).mean(
            axis=0
        )
        ph = np.asarray(pos[_indices_for(roles_df, j, "ph")], dtype=float).mean(axis=0)
        eq = np.asarray(pos[_indices_for(roles_df, eq_m, "py_eq")], dtype=float).mean(
            axis=0
        )
        dist = float(np.linalg.norm(pole - ph))
        dist_eq = float(np.linalg.norm(eq - ph))
        state = classify_cation_pi_state(dist, open_lo=cation_pi_open_lo)
        state_eq = classify_cation_pi_state(dist_eq, open_lo=cation_pi_open_lo)
        cen_i = [float(pole[0]), float(pole[1]), float(pole[2])]
        cen_j = [float(ph[0]), float(ph[1]), float(ph[2])]
        cen_k = [float(eq[0]), float(eq[1]), float(eq[2])]
        unit_ngl, _ = _indices_to_ngl(
            universe, cation_pi_unit_indices(roles_df, i, j, eq_m), index_to_pos
        )
        pairs.append(
            {
                "rank": color_i + 1,
                "kind": "cation_pi",
                "edge": int(k),
                "mon_a": int(i),
                "mon_b": int(j),
                "mon_c": int(eq_m),
                "label": cation_pi_unit_label(k, i, j, eq_m),
                "distance": dist,
                "distance_eq": dist_eq,
                "angle_deg": angle_at_vertex_deg(pole, ph, eq),
                "state": state,
                "state_eq": state_eq,
                "color": _UNIT_COLORS[color_i % len(_UNIT_COLORS)],
                "color_i": monomer_color(i),
                "color_j": monomer_color(j),
                "color_k": monomer_color(eq_m),
                "atoms_i": ngl_i,
                "atoms_j": ngl_j,
                "atoms_k": ngl_eq,
                "atoms_unit": unit_ngl,
                "centroid_i": cen_i,
                "centroid_j": cen_j,
                "centroid_k": cen_k,
                "midpoint": _midpoint(cen_i, cen_j),
                "midpoint_eq": _midpoint(cen_k, cen_j),
                "labelVisible": state == "open",
            }
        )
        color_i += 1
    for k, (i, j) in enumerate(equator_edges):
        ngl_i, _ = _indices_to_ngl(
            universe, _indices_for(roles_df, i, "r2"), index_to_pos
        )
        ngl_j, _ = _indices_to_ngl(
            universe, _indices_for(roles_df, j, "r3"), index_to_pos
        )
        ngl_cpy, _ = _indices_to_ngl(
            universe, _cpy_atom_indices(roles_df, i), index_to_pos
        )
        r2 = _role_point(pos, roles_df, i, "r2")
        r3 = _role_point(pos, roles_df, j, "r3")
        cpy = _cpy_point(pos, roles_df, i)
        dist = float(np.linalg.norm(r2 - r3))
        dist_d2 = float(np.linalg.norm(cpy - r3))
        state = classify_paper_d1_state(dist, open_lo=open_lo, open_hi=open_hi)
        cen_i = [float(r2[0]), float(r2[1]), float(r2[2])]
        cen_j = [float(r3[0]), float(r3[1]), float(r3[2])]
        cen_k = [float(cpy[0]), float(cpy[1]), float(cpy[2])]
        unit_ngl, _ = _indices_to_ngl(
            universe, equator_unit_indices(roles_df, i, j), index_to_pos
        )
        pairs.append(
            {
                "rank": color_i + 1,
                "kind": "equator",
                "edge": int(k),
                "mon_a": int(i),
                "mon_b": int(j),
                "label": equator_unit_label(k, i, j),
                "distance": dist,
                "distance_d2": dist_d2,
                "angle_deg": angle_at_vertex_deg(r2, r3, cpy),
                "state": state,
                "color": _UNIT_COLORS[color_i % len(_UNIT_COLORS)],
                "color_i": monomer_color(i),
                "color_j": monomer_color(j),
                "color_k": MURATA_ROLE_COLORS["cpy"],
                "atoms_i": ngl_i,
                "atoms_j": ngl_j,
                "atoms_k": ngl_cpy,
                "atoms_unit": unit_ngl,
                "centroid_i": cen_i,
                "centroid_j": cen_j,
                "centroid_k": cen_k,
                "midpoint": _midpoint(cen_i, cen_j),
                "midpoint_d2": _midpoint(cen_k, cen_j),
                "labelVisible": state in {"open", "elongated"},
            }
        )
        color_i += 1
    return pairs


_NGL_UNITS_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; overflow: hidden; font-family: system-ui, sans-serif; }}
  #viewport {{ width: 100vw; height: 100vh; }}
  #sidebar {{
    position: absolute; top: 12px; left: 12px; width: 360px; max-height: calc(100vh - 24px);
    overflow-y: auto; color: #e8e8e8; background: rgba(0,0,0,0.78);
    padding: 12px 14px; border-radius: 10px; backdrop-filter: blur(8px);
    font-size: 12px; line-height: 1.45;
  }}
  #sidebar h2 {{ font-size: 14px; margin-bottom: 6px; font-weight: 600; }}
  #sidebar .meta {{ color: #aaa; font-size: 11px; margin-bottom: 10px; }}
  .section-title {{
    margin: 10px 0 4px; font-size: 11px; letter-spacing: 0.04em;
    text-transform: uppercase; color: #9ad; font-weight: 600;
  }}
  .pair-row {{
    display: flex; gap: 8px; align-items: flex-start; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08); cursor: pointer;
  }}
  .pair-row:hover {{ background: rgba(255,255,255,0.05); }}
  .pair-row.dimmed {{ opacity: 0.35; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; margin-top: 3px; }}
  .pair-body {{ flex: 1; }}
  .pair-title {{ font-weight: 600; }}
  .pair-stats {{ color: #b0b0b0; font-size: 11px; }}
  .legend {{ margin-top: 10px; color: #aaa; font-size: 11px; }}
  .site-legend-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  .btn-row {{ display: flex; gap: 6px; margin: 8px 0 4px; flex-wrap: wrap; }}
  button {{
    flex: 1; min-width: 72px; padding: 5px 8px; border: none; border-radius: 6px;
    background: #2d2d44; color: #ddd; cursor: pointer; font-size: 11px;
  }}
  button:hover {{ background: #3d3d5c; }}
  #controls {{
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    color: #bbb; font-size: 12px; background: rgba(0,0,0,0.45);
    padding: 6px 16px; border-radius: 6px; pointer-events: none;
    backdrop-filter: blur(6px);
  }}
</style>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
</head>
<body>
<div id="viewport"></div>
<div id="sidebar">
  <h2>{title}</h2>
  <div class="meta">{meta_text}</div>
  <div class="btn-row">
    <button id="show-all">Show all</button>
    <button id="hide-all">Hide units</button>
    <button id="show-pole">π(pole)</button>
    <button id="show-eq">π(eq)</button>
    <button id="show-d1">d1</button>
    <button id="show-d2">d2</button>
  </div>
  <div id="pair-list"></div>
  <div class="legend">
    <div style="margin-bottom:6px;"><b>Monomer backbone</b></div>
    {monomer_legend_html}
    <div style="margin:10px 0 6px;"><b>Site colors</b></div>
    {site_legend_html}
    <div style="margin-top:8px;">
      Each monomer scaffold is a distinct licorice color.<br/>
      π(pole) cylinders: red = open (≥6.5 Å), gray = closed.<br/>
      π(eq) cylinders: orange = open (≥6.5 Å), gray = closed.<br/>
      d1 cylinders: green = compact (4.5–5.5 Å), gray = closed (&lt;4.5),
      magenta = elongated (&gt;5.5).<br/>
      d2 cylinders: cyan = CPy–C3 (CPy = para to the Py⁺–benzene linker).<br/>
      Endpoint spheres use monomer backbone colors. Click a unit to solo it.
    </div>
  </div>
</div>
<div id="controls">Scroll to zoom &middot; Click-drag to rotate &middot; Right-drag to pan</div>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  var pairs = {pairs_json};
  var siteGroups = {site_groups_json};
  var monomerGroups = {monomer_groups_json};
  var stage = new NGL.Stage("viewport", {{
    backgroundColor: "#1a1a2e",
    ambientIntensity: 0.35,
    quality: "high"
  }});
  window.addEventListener("resize", function() {{ stage.handleResize(); }});

  var pdbBlob = new Blob([`{pdb_string}`], {{ type: "text/plain" }});
  var shapeComp = null;
  var siteReprs = [];
  var monomerReprs = [];
  var visible = pairs.map(function() {{ return true; }});
  var contactMode = "all";

  function hexToRgb(hex) {{
    var h = String(hex).replace("#", "");
    if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
    return [
      parseInt(h.slice(0, 2), 16) / 255,
      parseInt(h.slice(2, 4), 16) / 255,
      parseInt(h.slice(4, 6), 16) / 255
    ];
  }}

  function cylinderColor(p) {{
    if (p.kind === "cation_pi") {{
      return p.state === "open" ? [0.85, 0.2, 0.2] : [0.55, 0.55, 0.55];
    }}
    if (p.state === "open") return [0.15, 0.75, 0.35];
    if (p.state === "closed") return [0.55, 0.55, 0.55];
    if (p.state === "elongated") return [0.75, 0.2, 0.75];
    return [0.8, 0.8, 0.2];
  }}

  function eqPiColor(p) {{
    var st = p.state_eq || "closed";
    return st === "open" ? [0.95, 0.45, 0.1] : [0.55, 0.55, 0.55];
  }}

  function showPole(p) {{
    return p.kind === "cation_pi" && (contactMode === "all" || contactMode === "pole");
  }}
  function showEqPi(p) {{
    return p.kind === "cation_pi" && p.centroid_k && (contactMode === "all" || contactMode === "eq");
  }}
  function showD1(p) {{
    return p.kind === "equator" && (contactMode === "all" || contactMode === "d1");
  }}
  function showD2(p) {{
    return p.kind === "equator" && p.centroid_k && (contactMode === "all" || contactMode === "d2");
  }}

  function nVisible() {{
    return visible.filter(Boolean).length;
  }}

  function rebuildShapes() {{
    if (shapeComp) {{
      stage.removeComponent(shapeComp);
      shapeComp = null;
    }}
    var shape = new NGL.Shape("murata-units");
    var solo = nVisible() === 1;
    pairs.forEach(function(p, i) {{
      if (!visible[i]) return;
      var labeled = p.labelVisible || solo || contactMode !== "all";
      if (showPole(p)) {{
        var col = cylinderColor(p);
        shape.addSphere(p.centroid_i, hexToRgb(p.color_i || p.color), 0.55);
        shape.addSphere(p.centroid_j, hexToRgb(p.color_j || p.color), 0.55);
        shape.addCylinder(p.centroid_i, p.centroid_j, col, 0.12);
        if (labeled) shape.addText(p.midpoint, col, 1.2, p.distance.toFixed(2));
      }}
      if (showEqPi(p)) {{
        var colEq = eqPiColor(p);
        shape.addSphere(p.centroid_k, hexToRgb(p.color_k || p.color), 0.55);
        shape.addSphere(p.centroid_j, hexToRgb(p.color_j || p.color), 0.50);
        shape.addCylinder(p.centroid_k, p.centroid_j, colEq, 0.12);
        if (labeled && p.distance_eq != null && p.midpoint_eq) {{
          shape.addText(p.midpoint_eq, colEq, 1.2, p.distance_eq.toFixed(2));
        }}
      }}
      if (showD1(p)) {{
        var colD1 = cylinderColor(p);
        shape.addSphere(p.centroid_i, hexToRgb(p.color_i || p.color), 0.55);
        shape.addSphere(p.centroid_j, hexToRgb(p.color_j || p.color), 0.55);
        shape.addCylinder(p.centroid_i, p.centroid_j, colD1, 0.12);
        if (labeled) shape.addText(p.midpoint, colD1, 1.2, p.distance.toFixed(2));
      }}
      if (showD2(p)) {{
        var colD2 = [0.2, 0.65, 0.85];
        shape.addSphere(p.centroid_k, hexToRgb(p.color_k || p.color), 0.50);
        shape.addSphere(p.centroid_j, hexToRgb(p.color_j || p.color), 0.50);
        shape.addCylinder(p.centroid_k, p.centroid_j, colD2, 0.10);
        if (labeled && p.distance_d2 != null && p.midpoint_d2) {{
          shape.addText(p.midpoint_d2, colD2, 1.2, p.distance_d2.toFixed(2));
        }}
      }}
    }});
    shapeComp = stage.addComponentFromObject(shape);
    shapeComp.addRepresentation("buffer");
  }}

  function rebuildMonomerBackbone(comp) {{
    monomerReprs.forEach(function(r) {{
      try {{ comp.removeRepresentation(r); }} catch (e) {{}}
    }});
    monomerReprs = [];
    var keep = null;
    if (nVisible() === 1) {{
      pairs.forEach(function(p, i) {{
        if (visible[i]) keep = {{ a: p.mon_a, b: p.mon_b, c: p.mon_c }};
      }});
    }}
    (monomerGroups || []).forEach(function(g) {{
      if (!g.atoms || !g.atoms.length) return;
      if (keep && g.monomer !== keep.a && g.monomer !== keep.b && g.monomer !== keep.c) return;
      monomerReprs.push(comp.addRepresentation("licorice", {{
        sele: "@" + g.atoms.join(","),
        color: g.color,
        radiusScale: 0.95,
        opacity: 0.88
      }}));
    }});
    if (!(monomerGroups && monomerGroups.length)) {{
      monomerReprs.push(comp.addRepresentation("licorice", {{
        color: "element", radiusScale: 0.9, opacity: 0.55
      }}));
    }}
  }}

  function rebuildSiteHighlight(comp) {{
    siteReprs.forEach(function(r) {{
      try {{ comp.removeRepresentation(r); }} catch (e) {{}}
    }});
    siteReprs = [];
    var solo = nVisible() === 1;
    if (solo) {{
      var extra = {{}};
      pairs.forEach(function(p, i) {{
        if (!visible[i]) return;
        (p.atoms_unit || p.atoms_i.concat(p.atoms_j)).forEach(function(a) {{
          extra[a] = true;
        }});
      }});
      var keys = Object.keys(extra);
      if (keys.length) {{
        siteReprs.push(comp.addRepresentation("ball+stick", {{
          sele: "@" + keys.join(","),
          color: "#f5c542",
          radiusScale: 2.0
        }}));
      }}
      return;
    }}
    // Always show CPy so d2 endpoints are visible on the backbone.
    (siteGroups || []).forEach(function(g) {{
      if (g.role !== "cpy" || !g.atoms || !g.atoms.length) return;
      siteReprs.push(comp.addRepresentation("ball+stick", {{
        sele: "@" + g.atoms.join(","),
        color: g.color,
        radiusScale: 1.6
      }}));
    }});
  }}

  function renderList() {{
    var list = document.getElementById("pair-list");
    list.innerHTML = "";
    var sections = [
      {{kind: "cation_pi", title: "Cation–π  π(pole) Py⁺ i → Ph j · π(eq) Py⁺ k → Ph j"}},
      {{kind: "equator", title: "Equator  d1 R2 i → R3 j · d2 CPy i → R3 j"}}
    ];
    sections.forEach(function(sec) {{
      var heading = document.createElement("div");
      heading.className = "section-title";
      heading.textContent = sec.title;
      list.appendChild(heading);
      pairs.forEach(function(p, i) {{
        if (p.kind !== sec.kind) return;
        var row = document.createElement("div");
        row.className = "pair-row" + (visible[i] ? "" : " dimmed");
        var sw = document.createElement("div");
        sw.className = "swatch";
        sw.style.background = p.color;
        var body = document.createElement("div");
        body.className = "pair-body";
        var title = document.createElement("div");
        title.className = "pair-title";
        title.textContent = p.label;
        var stats = document.createElement("div");
        stats.className = "pair-stats";
        var statsTxt;
        if (p.kind === "cation_pi") {{
          statsTxt = "π(pole) " + p.distance.toFixed(2) + " Å · " + p.state;
          if (p.distance_eq != null) {{
            statsTxt += " · π(eq) " + p.distance_eq.toFixed(2) + " Å";
            if (p.state_eq) statsTxt += " " + p.state_eq;
          }}
          if (p.angle_deg != null) {{
            statsTxt += " · " + p.angle_deg.toFixed(1) + "°";
          }}
        }} else {{
          statsTxt = "d1 " + p.distance.toFixed(2) + " Å · " + p.state;
          if (p.distance_d2 != null) {{
            statsTxt += " · d2 " + p.distance_d2.toFixed(2) + " Å";
          }}
          if (p.angle_deg != null) {{
            statsTxt += " · " + p.angle_deg.toFixed(1) + "°";
          }}
        }}
        stats.textContent = statsTxt;
        body.appendChild(title);
        body.appendChild(stats);
        row.appendChild(sw);
        row.appendChild(body);
        row.addEventListener("click", function() {{
          visible = pairs.map(function(_, j) {{ return j === i; }});
          rebuildShapes();
          rebuildMonomerBackbone(window._comp);
          rebuildSiteHighlight(window._comp);
          renderList();
        }});
        list.appendChild(row);
      }});
    }});
  }}

  function applyVisible(pred, mode) {{
    contactMode = mode || "all";
    visible = pairs.map(pred);
    rebuildShapes();
    rebuildMonomerBackbone(window._comp);
    rebuildSiteHighlight(window._comp);
    renderList();
  }}

  document.getElementById("show-all").addEventListener("click", function() {{
    applyVisible(function() {{ return true; }}, "all");
  }});
  document.getElementById("hide-all").addEventListener("click", function() {{
    applyVisible(function() {{ return false; }}, "all");
  }});
  document.getElementById("show-pole").addEventListener("click", function() {{
    applyVisible(function(p) {{ return p.kind === "cation_pi"; }}, "pole");
  }});
  document.getElementById("show-eq").addEventListener("click", function() {{
    applyVisible(function(p) {{ return p.kind === "cation_pi"; }}, "eq");
  }});
  document.getElementById("show-d1").addEventListener("click", function() {{
    applyVisible(function(p) {{ return p.kind === "equator"; }}, "d1");
  }});
  document.getElementById("show-d2").addEventListener("click", function() {{
    applyVisible(function(p) {{ return p.kind === "equator"; }}, "d2");
  }});

  stage.loadFile(pdbBlob, {{ ext: "pdb", defaultRepresentation: false }}).then(function(comp) {{
    window._comp = comp;
    rebuildMonomerBackbone(comp);
    rebuildSiteHighlight(comp);
    rebuildShapes();
    renderList();
    comp.autoView();
  }});
}});
</script>
</body>
</html>
"""


def _legend_rows(groups: Sequence[dict[str, Any]], empty_html: str) -> str:
    rows = []
    for g in groups:
        rows.append(
            '<div class="site-legend-row">'
            f'<div class="swatch" style="background:{g["color"]};"></div>'
            f'<span>{g["label"]} ({len(g.get("atoms", []))} atoms)</span>'
            "</div>"
        )
    return "\n    ".join(rows) if rows else empty_html


def render_murata_units_html(
    *,
    title: str,
    meta_text: str,
    pdb_string: str,
    pairs: Sequence[dict[str, Any]],
    site_groups: Sequence[dict[str, Any]],
    monomer_groups: Optional[Sequence[dict[str, Any]]] = None,
) -> str:
    site_legend_html = _legend_rows(
        site_groups,
        '<div class="site-legend-row">'
        '<div class="swatch" style="background:#911eb4;"></div><span>Py-pole</span></div>',
    )
    monomer_legend_html = _legend_rows(
        monomer_groups or [],
        '<div class="site-legend-row"><span>No monomer map</span></div>',
    )
    return _NGL_UNITS_HTML.format(
        title=title,
        meta_text=meta_text,
        pdb_string=_escape_pdb_for_js(pdb_string),
        pairs_json=json.dumps(list(pairs), ensure_ascii=False),
        site_groups_json=json.dumps(list(site_groups), ensure_ascii=False),
        monomer_groups_json=json.dumps(list(monomer_groups or []), ensure_ascii=False),
        site_legend_html=site_legend_html,
        monomer_legend_html=monomer_legend_html,
    )


def write_murata_units_ngl_html(
    universe: Any,
    output_path: Path | str,
    *,
    frame: Optional[int] = 0,
    selection: str = "resname MOL",
    n_monomers: int = 6,
    roles_df: Optional[pd.DataFrame] = None,
    monomer_selections: Optional[Sequence[str]] = None,
    units_df: Optional[pd.DataFrame] = None,
    cation_edges: Optional[Sequence[tuple[int, int]]] = None,
    equator_edges: Optional[Sequence[tuple[int, int]]] = None,
    title: Optional[str] = None,
    cation_pi_open_lo: float = CATION_PI_OPEN_LO,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
) -> Path:
    """Write an NGL HTML view of locked cation–π and equator units at ``frame``."""
    from src.utils.gsa_selections import resolve_selections

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(universe.trajectory)
    if frame is None:
        frame = 0
    frame = int(max(0, min(int(frame), n_frames - 1)))
    universe.trajectory[frame]

    try:
        atoms = universe.select_atoms(selection)
    except Exception:
        atoms = universe.atoms
    if len(atoms) == 0:
        atoms = universe.atoms

    resolved_monomer_sels: Optional[Sequence[str]] = (
        list(monomer_selections) if monomer_selections else None
    )
    if roles_df is None:
        sels = resolve_selections(
            universe, gsa_resname="MOL", n_monomers=int(n_monomers), auto_tooth=False
        )
        roles_df = resolve_murata_criteria_atoms(universe, sels.monomer_selections)
        if resolved_monomer_sels is None:
            resolved_monomer_sels = sels.monomer_selections

    csv_cation: list[tuple[int, int]] = []
    csv_equator: list[tuple[int, int]] = []
    csv_sandwiches: list[tuple[int, int, int]] = []
    if units_df is not None:
        csv_cation, csv_equator = edges_from_units_df(units_df)
        csv_sandwiches = sandwiches_from_units_df(units_df)
    pos = np.asarray(universe.atoms.positions, dtype=float)
    if cation_edges is None:
        if csv_sandwiches and all(t[2] >= 0 for t in csv_sandwiches):
            cation_edges = csv_sandwiches
        elif csv_cation:
            cation_edges = csv_cation
        else:
            cation_edges = lock_cation_pi_sandwiches(pos, roles_df)
    if equator_edges is None:
        equator_edges = csv_equator or lock_equator_edges(pos, roles_df)

    pdb_string = _atoms_to_pdb_string(atoms)
    index_to_pos = {int(atom.index): pos_i for pos_i, atom in enumerate(atoms)}
    site_groups = _site_groups_for_roles(universe, roles_df, index_to_pos)
    monomer_groups = _monomer_backbone_groups(
        universe,
        roles_df,
        index_to_pos,
        monomer_selections=resolved_monomer_sels,
    )
    pairs = build_murata_unit_payloads(
        universe,
        roles_df,
        cation_edges=cation_edges,
        equator_edges=equator_edges,
        index_to_pos=index_to_pos,
        cation_pi_open_lo=cation_pi_open_lo,
        open_lo=open_lo,
        open_hi=open_hi,
    )
    if not pairs:
        raise ValueError("No cation–π or equator units to draw")

    n_cation = sum(1 for p in pairs if p["kind"] == "cation_pi")
    n_equator = sum(1 for p in pairs if p["kind"] == "equator")
    n_open = sum(1 for p in pairs if p["kind"] == "cation_pi" and p["state"] == "open")
    n_elong = sum(1 for p in pairs if p["kind"] == "equator" and p["state"] == "elongated")
    if title is None:
        title = "Murata cation–π and equator units"
    meta = (
        f"Frame {frame} &bull; {len(atoms)} atoms &bull; "
        f"{n_cation} cation–π &bull; {n_equator} equator"
        f" &bull; n_open={n_open} &bull; d1 elongated={n_elong}"
    )
    html = render_murata_units_html(
        title=title,
        meta_text=meta,
        pdb_string=pdb_string,
        pairs=pairs,
        site_groups=site_groups,
        monomer_groups=monomer_groups,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
