"""3D structure visualization skill using NGL.js (browser) and nglview (Jupyter)."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext

_NGL_REPR_CHOICES = [
    "licorice", "ball+stick", "cartoon", "spacefill",
    "surface", "ribbon", "rope", "tube", "line",
]

_DEFAULT_HIGHLIGHT_COLORS = [
    "red", "blue", "green", "orange", "purple", "cyan",
]

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


class VisualizeStructureSkill(Skill):
    name = "visualize_structure"
    description = (
        "Render a 3D interactive visualization of the molecular structure "
        "at a given trajectory frame. Produces a self-contained HTML file "
        "(opens in any browser) and a reusable nglview Python script for "
        "Jupyter notebooks. Supports custom atom selections, representation "
        "styles, and highlighted regions with distinct colors."
    )
    category = "visualization"
    parameters = [
        Parameter(
            "selection", ParamType.ATOM_SELECTION,
            "MDAnalysis atom selection for the main structure "
            "(e.g., 'protein', 'resname GSA', 'resname MOF', 'all'). "
            "If omitted, uses the session main selection.",
            required=False, default=None,
        ),
        Parameter(
            "frame", ParamType.INTEGER,
            "Trajectory frame index to visualize (0-based).",
            required=False, default=0,
        ),
        Parameter(
            "representation", ParamType.STRING,
            "NGL representation for the main selection.",
            required=False, default="licorice",
            enum_values=_NGL_REPR_CHOICES,
        ),
        Parameter(
            "highlight_selections", ParamType.ARRAY,
            "Additional MDAnalysis atom selections to highlight "
            "(e.g., ['resid 33', 'name I']).",
            required=False, default=[],
            items_type=ParamType.STRING,
        ),
        Parameter(
            "highlight_colors", ParamType.ARRAY,
            "Colors for each highlight (CSS names or hex). "
            "Cycles if fewer colors than selections.",
            required=False, default=_DEFAULT_HIGHLIGHT_COLORS,
            items_type=ParamType.STRING,
        ),
        Parameter(
            "highlight_repr", ParamType.STRING,
            "Representation style for highlighted atoms.",
            required=False, default="ball+stick",
            enum_values=_NGL_REPR_CHOICES,
        ),
        Parameter(
            "background", ParamType.STRING,
            "Viewer background color.",
            required=False, default="white",
            enum_values=["white", "black", "#1a1a2e"],
        ),
        Parameter(
            "filename", ParamType.STRING,
            "Output HTML filename (saved in session output directory).",
            required=False, default="structure_view.html",
        ),
    ]
    requires = ["universe"]
    produces = []

    def execute(self, context: AnalysisContext, **params: Any) -> SkillResult:
        import MDAnalysis as mda

        u = context.get("universe")
        selection = params.get("selection") or context.main_selection
        frame_idx = params.get("frame", 0)
        main_repr = params.get("representation", "licorice")
        highlight_sels = params.get("highlight_selections", []) or []
        highlight_colors = params.get("highlight_colors", _DEFAULT_HIGHLIGHT_COLORS)
        highlight_repr_name = params.get("highlight_repr", "ball+stick")
        bg_color = params.get("background", "white")
        filename = params.get("filename", "structure_view.html")

        n_frames = u.trajectory.n_frames
        if frame_idx < 0 or frame_idx >= n_frames:
            return SkillResult(
                success=False,
                error=f"Frame {frame_idx} out of range (0\u2013{n_frames - 1}).",
                summary=f"Invalid frame index {frame_idx}. "
                        f"Trajectory has {n_frames} frames (0\u2013{n_frames - 1}).",
            )

        u.trajectory[frame_idx]

        try:
            atoms = u.select_atoms(selection)
        except Exception as exc:
            return SkillResult(
                success=False,
                error=f"Invalid selection '{selection}': {exc}",
                summary=f"Atom selection failed: {exc}",
            )

        if len(atoms) == 0:
            return SkillResult(
                success=False,
                error=f"Selection '{selection}' matched 0 atoms at frame {frame_idx}.",
                summary=f"No atoms matched '{selection}'.",
            )

        # --- write PDB to string via temp file ---
        tmp_path = os.path.join(context.output_dir, "_tmp_vis.pdb")
        atoms.write(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as fh:
            pdb_string = fh.read()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        pdb_string = pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

        # --- map MDAnalysis indices to 0-based position in the PDB ---
        idx_to_pos = {atom.index: pos for pos, atom in enumerate(atoms)}

        # --- build NGL.js representation commands ---
        repr_js_lines = [
            f'comp.addRepresentation("{main_repr}", '
            f'{{ color: "element", radiusScale: 1.5 }});',
        ]

        highlight_summaries = []
        for i, hl_sel in enumerate(highlight_sels):
            color = highlight_colors[i % len(highlight_colors)]
            try:
                hl_atoms = u.select_atoms(hl_sel)
            except Exception:
                highlight_summaries.append(f"  - '{hl_sel}': invalid selection, skipped")
                continue

            ngl_positions = sorted(
                idx_to_pos[a.index] for a in hl_atoms if a.index in idx_to_pos
            )
            if not ngl_positions:
                highlight_summaries.append(
                    f"  - '{hl_sel}': no overlap with main selection, skipped"
                )
                continue

            sele_str = ",".join(str(p) for p in ngl_positions)
            repr_js_lines.append(
                f'comp.addRepresentation("{highlight_repr_name}", '
                f'{{ sele: "@{sele_str}", color: "{color}", radiusScale: 2.5 }});'
            )
            highlight_summaries.append(
                f"  - '{hl_sel}': {len(ngl_positions)} atoms in {color}"
            )

        representation_js = "\n    ".join(repr_js_lines)

        title = f"MDChat \u2014 Frame {frame_idx}"
        info_text = (
            f"Frame {frame_idx} / {n_frames - 1} &bull; "
            f"{len(atoms)} atoms &bull; {selection}"
        )

        html = NGL_HTML_TEMPLATE.format(
            title=title,
            info_text=info_text,
            pdb_string=pdb_string,
            representation_js=representation_js,
            bg_color=bg_color,
        )

        out_path = os.path.join(context.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        # --- companion nglview script for Jupyter ---
        script_name = filename.rsplit(".", 1)[0] + "_nglview.py"
        script_path = os.path.join(context.output_dir, script_name)
        _write_nglview_script(
            script_path, context, selection, frame_idx,
            main_repr, highlight_sels, highlight_colors, highlight_repr_name,
        )

        summary_parts = [
            f"3D visualization saved to {out_path} "
            f"(frame {frame_idx}, {len(atoms)} atoms).",
            f"Jupyter nglview script: {script_path}",
        ]
        if highlight_summaries:
            summary_parts.append("Highlights:\n" + "\n".join(highlight_summaries))

        return SkillResult(
            success=True,
            artifacts={
                "structure_html": out_path,
                "nglview_script": script_path,
            },
            summary="\n".join(summary_parts),
        )


def _write_nglview_script(
    path: str,
    context: Any,
    selection: str,
    frame: int,
    main_repr: str,
    highlight_sels: list[str],
    highlight_colors: list[str],
    highlight_repr: str,
) -> None:
    """Write a self-contained nglview Python script for use in Jupyter."""
    topo = context.get("topology_path", "<TOPOLOGY_FILE>")
    traj = context.get("trajectory_path", "<TRAJECTORY_FILE>")

    lines = [
        '"""nglview visualization — generated by MDChat. Run in Jupyter."""',
        "",
        "import MDAnalysis as mda",
        "import nglview as nv",
        "",
        f"u = mda.Universe(r\"{topo}\", r\"{traj}\")",
        f"u.trajectory[{frame}]",
        f"sel = u.select_atoms(\"{selection}\")",
        "view = nv.show_mdanalysis(sel)",
        "",
        f"view.clear_representations()",
        f"view.add_representation(\"{main_repr}\", selection=\"all\", "
        f"color=\"grey\", radiusScale=1.5)",
    ]

    for i, hl_sel in enumerate(highlight_sels):
        color = highlight_colors[i % len(highlight_colors)]
        var = f"hl_{i}"
        lines += [
            "",
            f"# Highlight: {hl_sel}",
            f"{var} = u.select_atoms(\"{hl_sel}\")",
            f"# Map to indices within the displayed atom group",
            f"_idx = [j for j, a in enumerate(sel) if a.index in set({var}.indices)]",
            f"if _idx:",
            f"    _sele = \",\".join(str(j) for j in _idx)",
            f"    view.add_representation(\"{highlight_repr}\", "
            f"selection=f\"@{{_sele}}\", color=\"{color}\", radiusScale=2.5)",
        ]

    lines += ["", "view"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


get_default_registry().register(VisualizeStructureSkill())
