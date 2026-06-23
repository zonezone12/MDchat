"""
Visual inspection of changepoint cluster structures in NGL.

For a chosen feature group and cluster label, extracts the per-segment
*average* structure (mean coordinates over each segment's frame range),
superposes all averages onto the first segment, and writes a self-contained
NGL HTML viewer with checkboxes to show/hide individual segments.

Example
-------
python scripts/inspect_cluster_structures_ngl.py \\
    --segments-csv output/changepoints/clusters/gsa/segments_clustered.csv \\
    --cluster-label 2 \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectory-dir traj/BMMpM.bak \\
    --trajectory-layout nested \\
    --selection "resname MOL" \\
    --output-dir output/changepoints/cluster_inspection/k2_gsa
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.metastable_states import Segment
from src.utils.run_log import RunContext, log_event, step

_TRAJ_EXTENSIONS = (".trj", ".xtc", ".dcd", ".nc", ".crd")

_SEGMENT_COLORS = [
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


def _resolve_trajectory(
    trajectory_dir: Path,
    traj_id: str,
    *,
    layout: str = "nested",
    traj_filename: str = "mdcrd_v",
) -> Path:
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


def _load_universe(
    topology: Path,
    traj_path: Path,
    traj_format: Optional[str],
) -> "mda.Universe":
    import MDAnalysis as mda

    kwargs: dict = {}
    if traj_format and traj_path.suffix == "":
        kwargs["format"] = traj_format
    return mda.Universe(str(topology), str(traj_path), **kwargs)


def _segments_from_df(df: pd.DataFrame) -> list[Segment]:
    segments: list[Segment] = []
    for _, row in df.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        rep = int(row["rep_frame"]) if "rep_frame" in row and pd.notna(row["rep_frame"]) else (start + end) // 2
        segments.append(
            Segment(
                traj_id=str(row["traj_id"]),
                segment_id=int(row["segment_id"]),
                start_frame=start,
                end_frame=end,
                rep_frame=rep,
                rmsd_mean=float(row.get("assembly_rmsd_to_ref_mean", np.nan))
                if pd.notna(row.get("assembly_rmsd_to_ref_mean"))
                else float("nan"),
                rg_mean=float(row.get("assembly_rg_mean", np.nan))
                if pd.notna(row.get("assembly_rg_mean"))
                else float("nan"),
            )
        )
    return segments


def _segment_frame_indices(seg: Segment, *, segment_stride: int) -> list[int]:
    frames = range(seg.start_frame, seg.end_frame, max(1, segment_stride))
    if not frames:
        return [seg.rep_frame]
    return list(frames)


def extract_average_positions_multi_trajectory(
    segments: list[Segment],
    *,
    topology: Path,
    trajectory_dir: Path,
    selection: str,
    trajectory_layout: str = "nested",
    trajectory_filename: str = "mdcrd_v",
    trajectory_format: Optional[str] = "TRJ",
    segment_stride: int = 1,
) -> np.ndarray:
    """Mean coordinates per segment; one universe load per unique traj_id."""
    if not segments:
        raise ValueError("No segments to extract")

    by_traj: dict[str, list[tuple[int, Segment]]] = {}
    for idx, seg in enumerate(segments):
        by_traj.setdefault(seg.traj_id, []).append((idx, seg))

    n_atoms: Optional[int] = None
    positions = np.empty((len(segments), 0, 3), dtype=np.float64)

    for traj_id, indexed_segs in by_traj.items():
        traj_path = _resolve_trajectory(
            trajectory_dir,
            traj_id,
            layout=trajectory_layout,
            traj_filename=trajectory_filename,
        )
        u = _load_universe(topology, traj_path, trajectory_format)
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
            frame_indices = _segment_frame_indices(seg, segment_stride=segment_stride)
            stack = np.empty((len(frame_indices), n_atoms, 3), dtype=np.float64)
            for j, fr in enumerate(frame_indices):
                u.trajectory[fr]
                stack[j] = sel.positions.copy()
            positions[global_idx] = stack.mean(axis=0)

        log_event(
            "info",
            f"Averaged {len(indexed_segs)} segment(s) from {traj_path.name}",
            component="inspect_cluster_structures_ngl",
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
    trajectory_dir: Path,
    selection: str,
    trajectory_layout: str,
    trajectory_filename: str,
    trajectory_format: Optional[str],
) -> tuple["mda.Universe", "mda.core.groups.AtomGroup"]:
    import MDAnalysis as mda

    if not segments:
        raise ValueError("No segments for template universe")
    first = segments[0]
    traj_path = _resolve_trajectory(
        trajectory_dir,
        first.traj_id,
        layout=trajectory_layout,
        traj_filename=trajectory_filename,
    )
    u = _load_universe(topology, traj_path, trajectory_format)
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
        zip(leaf_labels, pdb_strings, colors, rmsds_to_ref)
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


def resolve_segments_csv(
    segments_csv: Optional[Path],
    *,
    clusters_dir: Optional[Path],
    group: Optional[str],
) -> Path:
    if segments_csv is not None:
        return segments_csv
    if clusters_dir is None or group is None:
        raise ValueError("Provide --segments-csv or both --clusters-dir and --group")
    path = clusters_dir / group / "segments_clustered.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return path


def inspect_cluster(
    df: pd.DataFrame,
    cluster_label: int,
    *,
    topology: Path,
    trajectory_dir: Path,
    selection: str,
    output_dir: Path,
    group: Optional[str],
    trajectory_layout: str,
    trajectory_filename: str,
    trajectory_format: Optional[str],
    segment_stride: int,
    repr_style: str,
    radius_scale: float,
    default_visible: str,
    write_pdb: bool,
    ref_index: int,
) -> dict:
    sub = df[df["cluster_label"] == cluster_label].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No segments with cluster_label={cluster_label}")

    segments = _segments_from_df(sub)
    leaf_labels = sub["leaf_label"].astype(str).tolist() if "leaf_label" in sub.columns else [
        f"{s.traj_id}:seg{s.segment_id}" for s in segments
    ]

    positions = extract_average_positions_multi_trajectory(
        segments,
        topology=topology,
        trajectory_dir=trajectory_dir,
        selection=selection,
        trajectory_layout=trajectory_layout,
        trajectory_filename=trajectory_filename,
        trajectory_format=trajectory_format,
        segment_stride=segment_stride,
    )

    ref_idx = max(0, min(ref_index, len(segments) - 1))
    aligned, rmsds = align_structures_to_reference(positions, ref_index=ref_idx)

    _, template_sel = build_template_universe(
        segments,
        topology=topology,
        trajectory_dir=trajectory_dir,
        selection=selection,
        trajectory_layout=trajectory_layout,
        trajectory_filename=trajectory_filename,
        trajectory_format=trajectory_format,
    )

    colors = [_SEGMENT_COLORS[i % len(_SEGMENT_COLORS)] for i in range(len(segments))]
    pdb_strings: list[str] = []
    cluster_dir = output_dir / f"cluster_{cluster_label:02d}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for i, coords in enumerate(aligned):
        pdb_strings.append(coords_to_pdb_string(template_sel, coords))
        if write_pdb:
            pdb_path = cluster_dir / f"{leaf_labels[i].replace(':', '_')}_aligned.pdb"
            pdb_path.write_text(pdb_strings[-1], encoding="utf-8")

    group_tag = group or "segments"
    title = f"Cluster {cluster_label} — {group_tag}"
    meta = (
        f"{len(segments)} segments &bull; aligned to {leaf_labels[ref_idx]} "
        f"&bull; {selection}"
    )
    html_path = cluster_dir / f"cluster_{cluster_label:02d}_structures.html"
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
        cluster_dir / f"cluster_{cluster_label:02d}_aligned.npz",
        positions=aligned,
        raw_average_positions=positions,
        leaf_labels=np.array(leaf_labels, dtype=object),
        rmsd_to_ref=rmsds,
        cluster_label=cluster_label,
        ref_leaf_label=leaf_labels[ref_idx],
    )

    summary_lines = [
        f"Cluster {cluster_label} structure inspection",
        f"Group: {group_tag}",
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
    (cluster_dir / f"cluster_{cluster_label:02d}_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    log_event(
        "info",
        f"cluster={cluster_label}: {len(segments)} segments -> {html_path}",
        component="inspect_cluster_structures_ngl",
    )
    return {
        "cluster_label": cluster_label,
        "n_segments": len(segments),
        "html_path": str(html_path),
        "mean_rmsd_to_ref": float(np.mean(rmsds)),
        "max_rmsd_to_ref": float(np.max(rmsds)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract segment-average structures for a changepoint cluster, "
            "align them, and write an interactive NGL HTML viewer."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--segments-csv",
        type=Path,
        help="Path to segments_clustered.csv",
    )
    src.add_argument(
        "--clusters-dir",
        type=Path,
        help="Cluster output dir (use with --group)",
    )
    parser.add_argument(
        "--group",
        help="Feature group subdirectory under --clusters-dir (gsa, iodine, ...)",
    )
    parser.add_argument(
        "--cluster-label",
        type=int,
        required=True,
        help="Cluster label to inspect (from segments_clustered.csv)",
    )
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-layout",
        default="nested",
        choices=("nested", "flat", "auto"),
    )
    parser.add_argument("--trajectory-filename", default="mdcrd_v")
    parser.add_argument(
        "--trajectory-format",
        default="TRJ",
        help="MDAnalysis format for extensionless trajectories (empty to auto-detect)",
    )
    parser.add_argument("--selection", default="resname MOL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/changepoints/cluster_inspection"),
    )
    parser.add_argument(
        "--segment-stride",
        type=int,
        default=1,
        help="Subsample frames when averaging within each segment (default: 1)",
    )
    parser.add_argument(
        "--repr-style",
        default="licorice",
        choices=("licorice", "ball+stick", "cartoon", "spacefill", "line"),
    )
    parser.add_argument("--radius-scale", type=float, default=1.4)
    parser.add_argument(
        "--default-visible",
        default="first",
        choices=("first", "all", "none"),
        help="Which segments are visible on load (default: first only)",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=0,
        help="Index of segment used as alignment reference (default: 0)",
    )
    parser.add_argument(
        "--write-pdb",
        action="store_true",
        help="Also write one aligned PDB per segment",
    )
    parser.add_argument(
        "--all-clusters",
        action="store_true",
        help="Process every cluster label in the CSV (ignores single --cluster-label value as filter)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology = Path(args.topology)
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)

    if not topology.is_file():
        print(f"Topology not found: {topology}", file=sys.stderr)
        sys.exit(1)
    if not trajectory_dir.is_dir():
        print(f"Trajectory directory not found: {trajectory_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        segments_csv = resolve_segments_csv(
            args.segments_csv,
            clusters_dir=args.clusters_dir,
            group=args.group,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.clusters_dir and not args.group and args.segments_csv is None:
        print("--group is required with --clusters-dir", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(segments_csv)
    if "cluster_label" not in df.columns:
        print("CSV missing cluster_label column", file=sys.stderr)
        sys.exit(1)

    group = args.group
    if group is None and "group" in df.columns and df["group"].nunique() == 1:
        group = str(df["group"].iloc[0])

    traj_format = args.trajectory_format.strip() or None
    cluster_labels = sorted(df["cluster_label"].unique()) if args.all_clusters else [args.cluster_label]

    with RunContext.from_namespace(args, name="inspect_cluster_structures_ngl"):
        rows: list[dict] = []
        for cl in cluster_labels:
            with step(f"cluster={cl}"):
                try:
                    row = inspect_cluster(
                        df,
                        int(cl),
                        topology=topology,
                        trajectory_dir=trajectory_dir,
                        selection=args.selection,
                        output_dir=output_dir,
                        group=group,
                        trajectory_layout=args.trajectory_layout,
                        trajectory_filename=args.trajectory_filename,
                        trajectory_format=traj_format,
                        segment_stride=max(1, args.segment_stride),
                        repr_style=args.repr_style,
                        radius_scale=args.radius_scale,
                        default_visible=args.default_visible,
                        write_pdb=args.write_pdb,
                        ref_index=args.ref_index,
                    )
                    rows.append(row)
                except ValueError as exc:
                    log_event(
                        "warning",
                        str(exc),
                        component="inspect_cluster_structures_ngl",
                    )

        if rows:
            summary_df = pd.DataFrame(rows)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_dir / "inspection_summary.csv", index=False)

    print("\nOutput files:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()
