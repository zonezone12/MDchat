"""
Export and visualize cluster representative structures from cluster_representatives.csv.

Reads medoid rows produced by cluster_changepoint_segments.py / cluster_inspection,
loads each trajectory at rep_frame, writes PDB files, and builds interactive
NGL.js HTML viewers (one per cluster plus a gallery index).

Designed for HPC or any machine with trajectories on disk.

Example (on HPC)
----------------
python scripts/draw_cluster_representatives.py \\
    --representatives-csv output/changepoints/clusters/BMMpM/gsa/by_k/k_07/cluster_representatives.csv \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectory-dir traj/BMMpM.bak \\
    --trajectory-layout nested \\
    --trajectory-filename mdcrd_v \\
    --trajectory-format TRJ \\
    --selection "resname MOL" \\
    --highlight-selection "resname IOD" \\
    --output-dir output/changepoints/clusters/BMMpM/gsa/by_k/k_07/representative_structures
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.run_log import RunContext, log_event, step

_TRAJ_EXTENSIONS = (".trj", ".xtc", ".dcd", ".nc", ".crd")

NGL_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; overflow: hidden; font-family: system-ui, sans-serif; }}
  #viewport {{ width: 100vw; height: 100vh; }}
  #info {{
    position: absolute; top: 12px; left: 12px; color: #e0e0e0;
    font-size: 13px; background: rgba(0,0,0,0.55);
    padding: 8px 14px; border-radius: 8px; pointer-events: none;
    backdrop-filter: blur(6px);
  }}
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
<div id="info">{info_text}</div>
<div id="controls">Scroll to zoom &middot; Click-drag to rotate &middot; Right-drag to pan</div>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  var stage = new NGL.Stage("viewport", {{
    backgroundColor: "{bg_color}",
    ambientIntensity: 0.3,
    quality: "high"
  }});
  window.addEventListener("resize", function() {{ stage.handleResize(); }});

  var pdbBlob = new Blob([`{pdb_string}`], {{ type: "text/plain" }});
  stage.loadFile(pdbBlob, {{ ext: "pdb", defaultRepresentation: false }}).then(function(comp) {{
    {representation_js}
    comp.autoView();
  }});
}});
</script>
</body>
</html>
"""

GALLERY_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f5f5f7; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 8px; }}
  p.meta {{ color: #555; margin-bottom: 20px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 14px; }}
  th {{ background: #eee; }}
  a {{ color: #0b5fff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Selection: <code>{selection}</code> &middot; {n_clusters} cluster representatives</p>
<table>
  <thead>
    <tr>
      <th>Cluster</th>
      <th>Trajectory</th>
      <th>Segment</th>
      <th>Rep frame</th>
      <th>RMSD mean</th>
      <th>Rg mean</th>
      <th>PDB</th>
      <th>3D view</th>
    </tr>
  </thead>
  <tbody>
{rows}
  </tbody>
</table>
</body>
</html>
"""


def _traj_folder_names(traj_id: str, traj_filename: str) -> list[str]:
    """
    Candidate per-trajectory subfolder names under --trajectory-dir.

  For traj_id ``BMMpM_278943_mdcrd_v`` and filename ``mdcrd_v``, tries:
  - ``BMMpM_278943_mdcrd_v`` (full id)
  - ``BMMpM_278943`` (strip _mdcrd_v)
  - ``278943`` (numeric variant folder, as in HPC .bak layout)
    """
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
    """Map traj_id to an on-disk trajectory file."""
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


def _load_universe(topology: Path, traj_path: Path, traj_format: Optional[str]) -> "mda.Universe":
    import MDAnalysis as mda

    kwargs: dict = {}
    if traj_format and traj_path.suffix == "":
        kwargs["format"] = traj_format
    return mda.Universe(str(topology), str(traj_path), **kwargs)


def _escape_pdb_for_js(pdb_string: str) -> str:
    return pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def _build_ngl_html(
    pdb_string: str,
    *,
    title: str,
    info_text: str,
    style: str,
    highlight_positions: list[int],
    highlight_color: str,
    highlight_repr: str,
    bg_color: str,
) -> str:
    repr_js_lines = [
        f'comp.addRepresentation("{style}", '
        f'{{ color: "element", radiusScale: 1.5 }});',
    ]
    if highlight_positions:
        sele_str = ",".join(str(p) for p in highlight_positions)
        repr_js_lines.append(
            f'comp.addRepresentation("{highlight_repr}", '
            f'{{ sele: "@{sele_str}", color: "{highlight_color}", radiusScale: 2.5 }});'
        )
    return NGL_HTML_TEMPLATE.format(
        title=html.escape(title),
        info_text=info_text,
        pdb_string=_escape_pdb_for_js(pdb_string),
        representation_js="\n    ".join(repr_js_lines),
        bg_color=bg_color,
    )


def _atoms_to_pdb_string(atoms) -> str:
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
    try:
        atoms.write(tmp_path)
        return Path(tmp_path).read_text(encoding="utf-8")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _highlight_positions_in_selection(universe, main_sel, highlight_sel: str) -> list[int]:
    if not highlight_sel:
        return []
    main_atoms = universe.select_atoms(main_sel)
    idx_to_pos = {atom.index: pos for pos, atom in enumerate(main_atoms)}
    try:
        hl_atoms = universe.select_atoms(highlight_sel)
    except Exception:
        return []
    return sorted(
        idx_to_pos[a.index] for a in hl_atoms if a.index in idx_to_pos
    )


def export_representatives(
    reps_df: pd.DataFrame,
    *,
    topology: Path,
    trajectory_dir: Path,
    trajectory_layout: str,
    trajectory_filename: str,
    trajectory_format: Optional[str],
    selection: str,
    output_dir: Path,
    style: str,
    highlight_selection: str,
    highlight_color: str,
    highlight_repr: str,
    bg_color: str,
    export_full_system: bool,
) -> pd.DataFrame:
    """Export PDB + HTML for each cluster representative row."""
    import MDAnalysis as mda

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = output_dir / "pdb"
    html_dir = output_dir / "html"
    pdb_dir.mkdir(exist_ok=True)
    html_dir.mkdir(exist_ok=True)

    required = {"cluster_label", "traj_id", "rep_frame"}
    missing = required - set(reps_df.columns)
    if missing:
        raise ValueError(f"cluster_representatives.csv missing columns: {sorted(missing)}")

    reps_df = reps_df.sort_values("cluster_label").reset_index(drop=True)
    manifest_rows: list[dict] = []
    universe_cache: dict[str, mda.Universe] = {}

    for _, row in reps_df.iterrows():
        cluster = int(row["cluster_label"])
        traj_id = str(row["traj_id"])
        rep_frame = int(row["rep_frame"])
        segment_id = int(row["segment_id"]) if "segment_id" in row and pd.notna(row["segment_id"]) else -1
        leaf_label = str(row.get("leaf_label", f"{traj_id}:seg{segment_id}"))

        if traj_id not in universe_cache:
            traj_path = _resolve_trajectory(
                trajectory_dir,
                traj_id,
                layout=trajectory_layout,
                traj_filename=trajectory_filename,
            )
            universe_cache[traj_id] = _load_universe(
                topology, traj_path, trajectory_format
            )
            log_event(
                "info",
                f"Loaded universe for {traj_id} ({traj_path})",
                component="draw_cluster_representatives",
            )

        u = universe_cache[traj_id]
        n_frames = u.trajectory.n_frames
        if rep_frame < 0 or rep_frame >= n_frames:
            raise ValueError(
                f"rep_frame {rep_frame} out of range for {traj_id} "
                f"(0–{n_frames - 1}); leaf={leaf_label}"
            )

        u.trajectory[rep_frame]
        display_atoms = u.select_atoms(selection)
        if len(display_atoms) == 0:
            raise ValueError(
                f"Selection {selection!r} matched 0 atoms for {traj_id} "
                f"frame {rep_frame}"
            )

        stem = f"cluster_{cluster:02d}"
        pdb_path = pdb_dir / f"{stem}.pdb"
        html_path = html_dir / f"{stem}.html"

        if export_full_system:
            u.atoms.write(str(pdb_dir / f"{stem}_full.pdb"))
        display_atoms.write(str(pdb_path))

        hl_positions = _highlight_positions_in_selection(u, selection, highlight_selection)
        pdb_string = _atoms_to_pdb_string(display_atoms)
        rmsd_mean = row.get("rmsd_mean")
        rg_mean = row.get("rg_mean")
        info_parts = [
            f"Cluster {cluster}",
            html.escape(leaf_label),
            f"frame {rep_frame}",
            f"{len(display_atoms)} atoms",
        ]
        if pd.notna(rmsd_mean):
            info_parts.append(f"RMSD mean {float(rmsd_mean):.2f} Å")
        if pd.notna(rg_mean):
            info_parts.append(f"Rg mean {float(rg_mean):.2f} Å")

        html_content = _build_ngl_html(
            pdb_string,
            title=f"Cluster {cluster} — {leaf_label}",
            info_text=" &bull; ".join(info_parts),
            style=style,
            highlight_positions=hl_positions,
            highlight_color=highlight_color,
            highlight_repr=highlight_repr,
            bg_color=bg_color,
        )
        html_path.write_text(html_content, encoding="utf-8")

        manifest_rows.append({
            "cluster_label": cluster,
            "traj_id": traj_id,
            "segment_id": segment_id,
            "leaf_label": leaf_label,
            "rep_frame": rep_frame,
            "start_frame": int(row["start_frame"]) if "start_frame" in row and pd.notna(row["start_frame"]) else None,
            "end_frame": int(row["end_frame"]) if "end_frame" in row and pd.notna(row["end_frame"]) else None,
            "n_frames": int(row["n_frames"]) if "n_frames" in row and pd.notna(row["n_frames"]) else None,
            "rmsd_mean": float(rmsd_mean) if pd.notna(rmsd_mean) else None,
            "rg_mean": float(rg_mean) if pd.notna(rg_mean) else None,
            "pdb_path": str(pdb_path.relative_to(output_dir)),
            "html_path": str(html_path.relative_to(output_dir)),
        })
        log_event(
            "info",
            f"cluster {cluster}: {leaf_label} frame {rep_frame} -> {pdb_path.name}",
            component="draw_cluster_representatives",
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "representatives_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    gallery_rows = []
    for r in manifest_rows:
        rmsd_cell = f"{r['rmsd_mean']:.3f}" if r["rmsd_mean"] is not None else ""
        rg_cell = f"{r['rg_mean']:.3f}" if r["rg_mean"] is not None else ""
        gallery_rows.append(
            "<tr>"
            f"<td>{r['cluster_label']}</td>"
            f"<td>{html.escape(r['traj_id'])}</td>"
            f"<td>{r['segment_id']}</td>"
            f"<td>{r['rep_frame']}</td>"
            f"<td>{rmsd_cell}</td>"
            f"<td>{rg_cell}</td>"
            f"<td><a href=\"{html.escape(r['pdb_path'])}\">PDB</a></td>"
            f"<td><a href=\"{html.escape(r['html_path'])}\">Open 3D</a></td>"
            "</tr>"
        )

    gallery_path = output_dir / "index.html"
    gallery_path.write_text(
        GALLERY_HTML_TEMPLATE.format(
            title="Cluster representative structures",
            selection=html.escape(selection),
            n_clusters=len(manifest_rows),
            rows="\n".join(gallery_rows),
        ),
        encoding="utf-8",
    )

    return manifest_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PDB and NGL HTML for cluster representative structures."
    )
    parser.add_argument(
        "--representatives-csv",
        required=True,
        help="Path to cluster_representatives.csv",
    )
    parser.add_argument(
        "--topology",
        required=True,
        help="Topology file (e.g. traj/BMMpM_ca.prmtop)",
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Base directory for trajectories (e.g. traj/BMMpM.bak for nested layout)",
    )
    parser.add_argument(
        "--trajectory-layout",
        default="nested",
        choices=("nested", "flat", "auto"),
        help="nested: {trajectory-dir}/{folder}/mdcrd_v; flat: {trajectory-dir}/{traj_id}.trj; "
        "auto: try both (default: nested)",
    )
    parser.add_argument(
        "--trajectory-filename",
        default="mdcrd_v",
        help="Trajectory basename inside each per-run subfolder (default: mdcrd_v)",
    )
    parser.add_argument(
        "--trajectory-format",
        default="TRJ",
        help="MDAnalysis format for extensionless trajectories (default: TRJ; empty to auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <csv_dir>/representative_structures)",
    )
    parser.add_argument(
        "--selection",
        default="resname MOL",
        help="MDAnalysis selection for exported/displayed structure (default: resname MOL)",
    )
    parser.add_argument(
        "--highlight-selection",
        default="resname IOD",
        help="Optional highlight selection (default: resname IOD; empty to disable)",
    )
    parser.add_argument(
        "--style",
        default="licorice",
        choices=[
            "licorice", "ball+stick", "cartoon", "spacefill",
            "surface", "ribbon", "rope", "tube", "line",
        ],
        help="NGL representation for main structure",
    )
    parser.add_argument(
        "--highlight-repr",
        default="ball+stick",
        help="NGL representation for highlighted atoms",
    )
    parser.add_argument(
        "--highlight-color",
        default="purple",
        help="Color for highlighted atoms",
    )
    parser.add_argument(
        "--background",
        default="#1a1a2e",
        help="NGL viewer background color",
    )
    parser.add_argument(
        "--export-full-system",
        action="store_true",
        help="Also write full-system PDB (all atoms) alongside the selection PDB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reps_csv = Path(args.representatives_csv)
    topology = Path(args.topology)
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else reps_csv.parent / "representative_structures"
    )

    if not reps_csv.is_file():
        print(f"Representatives CSV not found: {reps_csv}", file=sys.stderr)
        sys.exit(1)
    if not topology.is_file():
        print(f"Topology not found: {topology}", file=sys.stderr)
        sys.exit(1)
    if not trajectory_dir.is_dir():
        print(f"Trajectory directory not found: {trajectory_dir}", file=sys.stderr)
        sys.exit(1)

    reps_df = pd.read_csv(reps_csv)
    highlight_sel = (args.highlight_selection or "").strip()
    traj_format = (args.trajectory_format or "").strip() or None

    with RunContext.from_namespace(args, name="draw_cluster_representatives"):
        with step("export representatives"):
            manifest_df = export_representatives(
                reps_df,
                topology=topology,
                trajectory_dir=trajectory_dir,
                trajectory_layout=args.trajectory_layout,
                trajectory_filename=args.trajectory_filename,
                trajectory_format=traj_format,
                selection=args.selection,
                output_dir=output_dir,
                style=args.style,
                highlight_selection=highlight_sel,
                highlight_color=args.highlight_color,
                highlight_repr=args.highlight_repr,
                bg_color=args.background,
                export_full_system=args.export_full_system,
            )

    print(f"\nExported {len(manifest_df)} cluster representatives to {output_dir}")
    print(f"  Gallery: {output_dir / 'index.html'}")
    print(f"  Manifest: {output_dir / 'representatives_manifest.csv'}")
    for _, row in manifest_df.iterrows():
        print(f"  cluster {int(row['cluster_label']):02d}: {row['html_path']}")


if __name__ == "__main__":
    main()
