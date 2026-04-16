# Optional plateau on RMSD / Rg / volume + plots

## Keep standalone `detect_motion_plateau`

Retain [`src/mdchat/skills/plateau.py`](src/mdchat/skills/plateau.py) for: retuning thresholds on cached series, cavity-only / GSA-sourced volume without recomputing, and any workflow that only needs plateau.

## Optional plateau on compute skills

Add optional parameters to [`ComputeRMSDSkill` / `ComputeRgSkill`](src/mdchat/skills/metrics.py) and [`ComputeVolumeSkill`](src/mdchat/skills/volume.py) (defaults preserve current behavior). Centralize logic via a small helper (e.g. [`src/mdchat/plateau_helpers.py`](src/mdchat/plateau_helpers.py)) that wraps [`src/PlateauDetection/detect.py`](src/PlateauDetection/detect.py) and sets context keys consistently with the standalone skill.

## Plot visualization: plateau period as shaded background

When plateau information exists, generated plots should **visually mark the plateau region** (e.g. from plateau onset to the end of the series, or the longest low-fluctuation segment) using a **distinct background color** behind the curve so the “steady-state” period is obvious at a glance. Optionally add a short legend label (e.g. “Plateau”).

### Context contract for plotting

Store enough information for any plot skill to align the x-axis with the shaded band:

- **Trajectory-native series (RMSD, Rg):** x is frame index `0 … n-1`; plateau span is `[plateau_start_trajectory_frame, last_frame]` (or derived from `plateau_detection`).
- **Strided volume:** x must match what is plotted: use `volume_frames` for the x-axis when plotting volume vs trajectory frame; shaded region uses the same **x coordinates** as the plateau interval (map plateau sample indices to trajectory frames via existing mapping).

Recommended context keys (names can be finalized in implementation):

- `plateau_plot_x0`, `plateau_plot_x1` — inclusive/exclusive span in **the same units as the plot x-axis** (frame or time), or
- A small dict `plateau_plot_span` with `{ "xmin", "xmax", "x_label_matches" }`.

### Where to implement

| Plot path | File | Notes |
|-----------|------|--------|
| Generic time series | [`PlotTimeseriesSkill`](src/mdchat/skills/plotting.py) | Matplotlib: `ax.axvspan(xmin, xmax, alpha=0.15–0.25, color=..., zorder=0)` **before** `ax.plot(...)` (or set zorder so the line stays above). Add optional param e.g. `highlight_plateau` defaulting to True when span exists in context. |
| Volume | [`Plotter.plot_volume_change`](src/Plotter/Plotter.py) + [`PlotVolumeSkill`](src/mdchat/skills/plotting.py) | Datashader path is pixel-based; simplest robust approach is either a **matplotlib fallback** when plateau shading is requested, or a **second pass** that composites a semi-transparent rectangle. Prefer one clear implementation (matplotlib volume plot with matching styling) if datashader cannot cheaply draw axvspan-equivalent bands. |

Use a single muted color (e.g. light green or light blue) for the plateau band and keep alpha moderate so the line remains readable.

### When plateau is computed together with metrics

If the user enables `detect_plateau` on `compute_rmsd` / `compute_rg` / `compute_volume`, the same response should populate `plateau_plot_*` (or equivalent) so a subsequent `plot_timeseries` / `plot_volume` picks it up automatically. Optionally add `highlight_plateau` on plot skills defaulting to true when span is present.

## Implementation todos (execution phase)

1. Helper module + optional params on compute skills + refactor standalone plateau skill.
2. Set plateau span keys in context whenever plateau runs (standalone or bundled).
3. **Plots:** extend `plot_timeseries` with shaded plateau; extend volume plotting (`Plotter` + `plot_volume` skill) with the same semantics and document x-axis alignment for strided volume.
4. Update skill descriptions for compute, plateau, and plotting.
