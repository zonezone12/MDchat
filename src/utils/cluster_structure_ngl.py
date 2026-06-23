"""
Shared helpers for cluster / macrostate structure inspection in NGL HTML.

Extract segment-average coordinates, superpose, and write interactive viewers.
"""

from __future__ import annotations

import glob
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.metastable_states import Segment
from src.utils.run_log import log_event

_TRAJ_EXTENSIONS = (".trj", ".xtc", ".dcd", ".nc", ".crd")

SEGMENT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9",
]

NGL_CLUSTER_HTML_TEMPLATE = """\
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
    position: absolute; top: 12px; left: 12px; width: 280px; max-height: calc(100vh - 24px);
    overflow-y: auto; color: #e8e8e8; background: rgba(0,0,0,0.72);
    padding: 12px 14px; border-radius: 10px; backdrop-filter: blur(8px);
    font-size: 13px; line-height: 1.45;
  }}
  #sidebar h2 {{ font-size: 14px; margin-bottom: 8px; font-weight: 600; }}
  #sidebar .meta {{ color: #aaa; font-size: 12px; margin-bottom: 10px; }}
  .seg-row {{
    display: flex; align-items: center; gap: 8px; padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }}
  .seg-row:last-child {{ border-bottom: none; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }}
  .seg-row label {{ cursor: pointer; flex: 1; }}
  .rmsd {{ color: #888; font-size: 11px; margin-left: 20px; }}
  .btn-row {{ display: flex; gap: 6px; margin: 10px 0; }}
  button {{
    flex: 1; padding: 5px 8px; border: none; border-radius: 6px;
    background: #2d2d44; color: #ddd; cursor: pointer; font-size: 12px;
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
    <button id="hide-all">Hide all</button>
    <button id="solo-first">Solo first</button>
  </div>
  <div id="segment-list"></div>
</div>
<div id="controls">Scroll to zoom &middot; Click-drag to rotate &middot; Right-drag to pan</div>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  var segments = {segments_json};
  var reprStyle = "{repr_style}";
  var radiusScale = {radius_scale};

  var stage = new NGL.Stage("viewport", {{
    backgroundColor: "#1a1a2e",
    ambientIntensity: 0.35,
    quality: "high"
  }});
  window.addEventListener("resize", function() {{ stage.handleResize(); }});

  var components = [];
  var listEl = document.getElementById("segment-list");

  function setVisibility(index, visible) {{
    if (components[index]) {{
      components[index].setVisibility(visible);
    }}
  }}

  function buildCheckboxes() {{
    listEl.innerHTML = "";
    segments.forEach(function(seg, i) {{
      var row = document.createElement("div");
      row.className = "seg-row";
      var swatch = document.createElement("div");
      swatch.className = "swatch";
      swatch.style.background = seg.color;
      var label = document.createElement("label");
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = seg.visible;
      cb.dataset.index = i;
      cb.addEventListener("change", function() {{
        setVisibility(i, cb.checked);
      }});
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + seg.name));
      row.appendChild(swatch);
      row.appendChild(label);
      listEl.appendChild(row);
      if (seg.rmsd_to_ref !== null && seg.rmsd_to_ref !== undefined) {{
        var rmsdEl = document.createElement("div");
        rmsdEl.className = "rmsd";
        rmsdEl.textContent = "RMSD vs ref: " + seg.rmsd_to_ref.toFixed(3) + " \\u00c5";
        listEl.appendChild(rmsdEl);
      }}
    }});
  }}

  document.getElementById("show-all").addEventListener("click", function() {{
    listEl.querySelectorAll("input[type=checkbox]").forEach(function(cb, i) {{
      cb.checked = true;
      setVisibility(i, true);
    }});
  }});
  document.getElementById("hide-all").addEventListener("click", function() {{
    listEl.querySelectorAll("input[type=checkbox]").forEach(function(cb, i) {{
      cb.checked = false;
      setVisibility(i, false);
    }});
  }});
  document.getElementById("solo-first").addEventListener("click", function() {{
    listEl.querySelectorAll("input[type=checkbox]").forEach(function(cb, i) {{
      cb.checked = (i === 0);
      setVisibility(i, i === 0);
    }});
  }});

  Promise.all(segments.map(function(seg) {{
    var blob = new Blob([seg.pdb], {{ type: "text/plain" }});
    return stage.loadFile(blob, {{
      ext: "pdb",
      name: seg.name,
      defaultRepresentation: false
    }}).then(function(comp) {{
      comp.addRepresentation(reprStyle, {{
        color: seg.color,
        radiusScale: radiusScale,
        opacity: 0.92
      }});
      comp.setVisibility(seg.visible);
      return comp;
    }});
  }})).then(function(comps) {{
    components = comps;
    buildCheckboxes();
    if (components.length) {{
      stage.autoView();
    }}
  }});
}});
</script>
</body>
</html>
"""


def expand_trajectory_paths(patterns: list[str]) -> list[Path]:
    """Resolve trajectory paths or globs (same logic as run_imamura_msm)."""
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            p = Path(pat)
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                for ext in ("*.xtc", "*.trj", "*.dcd", "*.nc"):
                    paths.extend(sorted(p.glob(ext)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def trajectory_id_from_path(traj_path: Path) -> str:
    """Match Imamura MSM traj_id convention (parent folder name, else stem)."""
    folder = traj_path.parent.name
    return folder if folder else traj_path.stem


def build_trajectory_map(patterns: list[str]) -> dict[str, Path]:
    """Map traj_id → trajectory path for explicit --trajectories lists."""
    mapping: dict[str, Path] = {}
    for path in expand_trajectory_paths(patterns):
        tid = trajectory_id_from_path(path)
        if tid in mapping:
            raise ValueError(
                f"Duplicate traj_id {tid!r}: {mapping[tid]} and {path}"
            )
        mapping[tid] = path
    return mapping


def _traj_folder_names(traj_id: str, traj_filename: str) -> list[str]:
    candidates = [traj_id]
    suffix = f"_{traj_filename}"
    if traj_id.endswith(suffix):
        candidates.append(traj_id[: -len(suffix)])

    parts = traj_id.split("_")
    if len(parts) >= 3:
        if parts[1].isdigit():
            candidates.append(parts[1])
        if len(parts) >= 4 and parts[2] == "re" and parts[1].isdigit():
            candidates.append(f"{parts[1]}_re")

    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def resolve_trajectory_path(
    traj_id: str,
    *,
    trajectory_dir: Optional[Path] = None,
    trajectory_map: Optional[dict[str, Path]] = None,
    layout: str = "nested",
    traj_filename: str = "mdcrd_v",
) -> Path:
    """Resolve traj_id via explicit map or directory layout."""
    if trajectory_map:
        if traj_id in trajectory_map:
            return trajectory_map[traj_id]
        for path in trajectory_map.values():
            if path.stem == traj_id:
                return path

    if trajectory_dir is None:
        raise FileNotFoundError(
            f"No trajectory for {traj_id!r} (not in map and no --trajectory-dir)"
        )

    tried: list[Path] = []

    def _check(path: Path) -> Optional[Path]:
        tried.append(path)
        return path if path.is_file() else None

    if layout in ("flat", "auto"):
        for ext in _TRAJ_EXTENSIONS:
            found = _check(trajectory_dir / f"{traj_id}{ext}")
            if found:
                return found

    if layout in ("nested", "auto"):
        for folder in _traj_folder_names(traj_id, traj_filename):
            base = trajectory_dir / folder / traj_filename
            found = _check(base)
            if found:
                return found
            for ext in _TRAJ_EXTENSIONS:
                found = _check(Path(f"{base}{ext}"))
                if found:
                    return found

    tried_str = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        f"No trajectory for {traj_id!r} under {trajectory_dir}\n"
        f"Tried:\n  {tried_str}"
    )


def load_universe(
    topology: Path,
    traj_path: Path,
    traj_format: Optional[str],
) -> "mda.Universe":
    import MDAnalysis as mda

    kwargs: dict = {}
    if traj_format and traj_path.suffix == "":
        kwargs["format"] = traj_format
    return mda.Universe(str(topology), str(traj_path), **kwargs)


def infer_frame_step(frames: np.ndarray) -> int:
    """Infer spacing between stored frame indices (handles analysis stride)."""
    if len(frames) < 2:
        return 1
    diffs = np.diff(np.unique(frames))
    diffs = diffs[diffs > 0]
    return int(np.min(diffs)) if len(diffs) else 1


def dwell_segments_from_frame_states(
    df: pd.DataFrame,
    state_label: int,
    *,
    label_col: str = "macro_label",
    min_dwell_frames: int = 1,
    frame_step: Optional[int] = None,
) -> tuple[list[Segment], list[str]]:
    """
    Contiguous dwell periods for one macro/micro state → Segment list.

    Each dwell run becomes one segment with ``start_frame`` inclusive and
    ``end_frame`` exclusive (same convention as changepoint segments).
    """
    required = {"traj_id", "frame", label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"frame_states missing columns: {sorted(missing)}")

    sub = df[df[label_col] == state_label].copy()
    if sub.empty:
        return [], []

    segments: list[Segment] = []
    leaf_labels: list[str] = []

    for traj_id, group in sub.groupby("traj_id", sort=True):
        frames = np.sort(group["frame"].to_numpy(dtype=int))
        if len(frames) == 0:
            continue

        step = frame_step if frame_step is not None else infer_frame_step(frames)
        seg_local_id = 0
        run_start = 0

        def _append_run(start_idx: int, end_idx: int) -> None:
            nonlocal seg_local_id
            start = int(frames[start_idx])
            end = int(frames[end_idx]) + step
            if (end - start) < min_dwell_frames:
                return
            rep = (start + end) // 2
            segments.append(
                Segment(
                    traj_id=str(traj_id),
                    segment_id=seg_local_id,
                    start_frame=start,
                    end_frame=end,
                    rep_frame=rep,
                    rmsd_mean=float("nan"),
                    rg_mean=float("nan"),
                    cluster_label=int(state_label),
                )
            )
            leaf_labels.append(f"{traj_id}:dwell{seg_local_id}")
            seg_local_id += 1

        for i in range(1, len(frames)):
            if frames[i] != frames[i - 1] + step:
                _append_run(run_start, i - 1)
                run_start = i
        _append_run(run_start, len(frames) - 1)

    return segments, leaf_labels


def segment_frame_indices(seg: Segment, *, segment_stride: int) -> list[int]:
    frames = range(seg.start_frame, seg.end_frame, max(1, segment_stride))
    if not frames:
        return [seg.rep_frame]
    return list(frames)


def extract_average_positions_multi_trajectory(
    segments: list[Segment],
    *,
    topology: Path,
    selection: str,
    trajectory_dir: Optional[Path] = None,
    trajectory_map: Optional[dict[str, Path]] = None,
    trajectory_layout: str = "nested",
    trajectory_filename: str = "mdcrd_v",
    trajectory_format: Optional[str] = "TRJ",
    segment_stride: int = 1,
    log_component: str = "cluster_structure_ngl",
) -> np.ndarray:
    """Mean coordinates per segment; one universe load per unique traj_id."""
    if not segments:
        raise ValueError("No segments to extract")
    if trajectory_dir is None and not trajectory_map:
        raise ValueError("Provide trajectory_dir and/or trajectory_map")

    by_traj: dict[str, list[tuple[int, Segment]]] = {}
    for idx, seg in enumerate(segments):
        by_traj.setdefault(seg.traj_id, []).append((idx, seg))

    n_atoms: Optional[int] = None
    positions = np.empty((len(segments), 0, 3), dtype=np.float64)

    for traj_id, indexed_segs in by_traj.items():
        traj_path = resolve_trajectory_path(
            traj_id,
            trajectory_dir=trajectory_dir,
            trajectory_map=trajectory_map,
            layout=trajectory_layout,
            traj_filename=trajectory_filename,
        )
        u = load_universe(topology, traj_path, trajectory_format)
        sel = u.select_atoms(selection)
        if len(sel) == 0:
            raise ValueError(f"Empty selection {selection!r} for {traj_id}")

        if n_atoms is None:
            n_atoms = len(sel)
            positions = np.zeros((len(segments), n_atoms, 3), dtype=np.float64)
        elif len(sel) != n_atoms:
            raise ValueError(
                f"Atom count mismatch for {traj_id}: {len(sel)} vs {n_atoms}"
            )

        for global_idx, seg in indexed_segs:
            frame_indices = segment_frame_indices(seg, segment_stride=segment_stride)
            stack = np.empty((len(frame_indices), n_atoms, 3), dtype=np.float64)
            for j, fr in enumerate(frame_indices):
                u.trajectory[fr]
                stack[j] = sel.positions.copy()
            positions[global_idx] = stack.mean(axis=0)

        log_event(
            "info",
            f"Averaged {len(indexed_segs)} segment(s) from {traj_path.name}",
            component=log_component,
        )

    return positions


def align_coords_to_reference(mobile: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    from MDAnalysis.analysis import align

    R, rmsd = align.rotation_matrix(mobile, ref)
    mobile_com = mobile.mean(axis=0)
    ref_com = ref.mean(axis=0)
    aligned = np.dot(mobile - mobile_com, R.T) + ref_com
    return aligned, float(rmsd)


def align_structures_to_reference(
    positions: np.ndarray,
    *,
    ref_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Superpose all structures onto positions[ref_index]."""
    ref = positions[ref_index]
    aligned = np.empty_like(positions)
    rmsds = np.zeros(positions.shape[0], dtype=np.float64)
    for i in range(positions.shape[0]):
        aligned[i], rmsds[i] = align_coords_to_reference(positions[i], ref)
    rmsds[ref_index] = 0.0
    return aligned, rmsds


def _escape_pdb_for_js(pdb_text: str) -> str:
    return pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def coords_to_pdb_string(template_sel: "mda.core.groups.AtomGroup", coords: np.ndarray) -> str:
    if len(template_sel) != coords.shape[0]:
        raise ValueError(
            f"Coordinate/atom mismatch: {coords.shape[0]} coords vs {len(template_sel)} atoms"
        )
    template_sel = template_sel.copy()
    template_sel.positions = coords
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w", encoding="utf-8") as fh:
        tmp_path = Path(fh.name)
    try:
        template_sel.write(str(tmp_path))
        return tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)


def build_template_universe(
    segments: list[Segment],
    *,
    topology: Path,
    selection: str,
    trajectory_dir: Optional[Path] = None,
    trajectory_map: Optional[dict[str, Path]] = None,
    trajectory_layout: str = "nested",
    trajectory_filename: str = "mdcrd_v",
    trajectory_format: Optional[str] = None,
) -> tuple["mda.Universe", "mda.core.groups.AtomGroup"]:
    if not segments:
        raise ValueError("No segments for template universe")
    first = segments[0]
    traj_path = resolve_trajectory_path(
        first.traj_id,
        trajectory_dir=trajectory_dir,
        trajectory_map=trajectory_map,
        layout=trajectory_layout,
        traj_filename=trajectory_filename,
    )
    u = load_universe(topology, traj_path, trajectory_format)
    sel = u.select_atoms(selection)
    if len(sel) == 0:
        raise ValueError(f"Empty selection: {selection!r}")
    u.trajectory[first.rep_frame]
    return u, sel


def write_ngl_cluster_html(
    output_path: Path,
    *,
    title: str,
    meta_text: str,
    leaf_labels: list[str],
    pdb_strings: list[str],
    rmsds_to_ref: np.ndarray,
    colors: list[str],
    repr_style: str = "licorice",
    radius_scale: float = 1.4,
    default_visible: str = "first",
) -> None:
    segment_payload = []
    for i, (name, pdb, color, rmsd) in enumerate(
        zip(leaf_labels, pdb_strings, rmsds_to_ref)
    ):
        if default_visible == "all":
            visible = True
        elif default_visible == "none":
            visible = False
        else:
            visible = i == 0
        segment_payload.append({
            "name": name,
            "pdb": _escape_pdb_for_js(pdb),
            "color": color,
            "rmsd_to_ref": float(rmsd),
            "visible": visible,
        })

    html = NGL_CLUSTER_HTML_TEMPLATE.format(
        title=title,
        meta_text=meta_text,
        segments_json=json.dumps(segment_payload, ensure_ascii=False),
        repr_style=repr_style,
        radius_scale=radius_scale,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def inspect_cluster_structures(
    segments: list[Segment],
    leaf_labels: list[str],
    *,
    cluster_label: int,
    topology: Path,
    selection: str,
    output_dir: Path,
    group_tag: str,
    trajectory_dir: Optional[Path] = None,
    trajectory_map: Optional[dict[str, Path]] = None,
    trajectory_layout: str = "nested",
    trajectory_filename: str = "mdcrd_v",
    trajectory_format: Optional[str] = "TRJ",
    segment_stride: int = 1,
    repr_style: str = "licorice",
    radius_scale: float = 1.4,
    default_visible: str = "first",
    write_pdb: bool = False,
    ref_index: int = 0,
    log_component: str = "cluster_structure_ngl",
    cluster_prefix: str = "cluster",
) -> dict:
    """Extract, align, and write NGL HTML for a set of segments."""
    if not segments:
        raise ValueError(f"No segments for {cluster_prefix}={cluster_label}")

    positions = extract_average_positions_multi_trajectory(
        segments,
        topology=topology,
        selection=selection,
        trajectory_dir=trajectory_dir,
        trajectory_map=trajectory_map,
        trajectory_layout=trajectory_layout,
        trajectory_filename=trajectory_filename,
        trajectory_format=trajectory_format,
        segment_stride=segment_stride,
        log_component=log_component,
    )

    ref_idx = max(0, min(ref_index, len(segments) - 1))
    aligned, rmsds = align_structures_to_reference(positions, ref_index=ref_idx)

    _, template_sel = build_template_universe(
        segments,
        topology=topology,
        selection=selection,
        trajectory_dir=trajectory_dir,
        trajectory_map=trajectory_map,
        trajectory_layout=trajectory_layout,
        trajectory_filename=trajectory_filename,
        trajectory_format=trajectory_format,
    )

    colors = [SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i in range(len(segments))]
    pdb_strings: list[str] = []
    cluster_dir = output_dir / f"{cluster_prefix}_{cluster_label:02d}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for i, coords in enumerate(aligned):
        pdb_strings.append(coords_to_pdb_string(template_sel, coords))
        if write_pdb:
            pdb_path = cluster_dir / f"{leaf_labels[i].replace(':', '_')}_aligned.pdb"
            pdb_path.write_text(pdb_strings[-1], encoding="utf-8")

    title = f"{cluster_prefix.replace('_', ' ').title()} {cluster_label} — {group_tag}"
    meta = (
        f"{len(segments)} segments &bull; aligned to {leaf_labels[ref_idx]} "
        f"&bull; {selection}"
    )
    html_path = cluster_dir / f"{cluster_prefix}_{cluster_label:02d}_structures.html"
    write_ngl_cluster_html(
        html_path,
        title=title,
        meta_text=meta,
        leaf_labels=leaf_labels,
        pdb_strings=pdb_strings,
        rmsds_to_ref=rmsds,
        colors=colors,
        repr_style=repr_style,
        radius_scale=radius_scale,
        default_visible=default_visible,
    )

    np.savez_compressed(
        cluster_dir / f"{cluster_prefix}_{cluster_label:02d}_aligned.npz",
        positions=aligned,
        raw_average_positions=positions,
        leaf_labels=np.array(leaf_labels, dtype=object),
        rmsd_to_ref=rmsds,
        cluster_label=cluster_label,
        ref_leaf_label=leaf_labels[ref_idx],
    )

    summary_lines = [
        f"{cluster_prefix} {cluster_label} structure inspection",
        f"Context: {group_tag}",
        f"Segments: {len(segments)}",
        f"Alignment reference: {leaf_labels[ref_idx]} (index {ref_idx})",
        f"Selection: {selection}",
        f"Segment frame stride (averaging): {segment_stride}",
        "",
        "Per-segment RMSD to reference after alignment (Å):",
    ]
    for label, rmsd in zip(leaf_labels, rmsds):
        summary_lines.append(f"  {label}: {rmsd:.4f}")
    summary_lines.extend([
        "",
        f"NGL viewer: {html_path.name}",
        "Toggle segments in the sidebar; use Show all / Solo first for quick checks.",
    ])
    (cluster_dir / f"{cluster_prefix}_{cluster_label:02d}_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    log_event(
        "info",
        f"{cluster_prefix}={cluster_label}: {len(segments)} segments -> {html_path}",
        component=log_component,
    )
    return {
        "cluster_label": cluster_label,
        "n_segments": len(segments),
        "html_path": str(html_path),
        "mean_rmsd_to_ref": float(np.mean(rmsds)),
        "max_rmsd_to_ref": float(np.max(rmsds)),
    }
