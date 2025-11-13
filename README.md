# GSA Nanocube MD Trajectory Analysis Pipeline

## Overview

This repository provides a modular Python pipeline for analyzing molecular dynamics (MD) trajectories of Glycine Sulfate Amide (GSA) nanocubes, built on [MDAnalysis](https://www.mdanalysis.org/). The pipeline focuses on identifying structural dynamics, particularly shrinkage/expansion events and guest molecule interactions, by:

1. **Endpoint Identification**: Using RDKit and a custom `EndpointsFinder` to detect molecular endpoints (e.g., reactive sites or distant atoms) in individual GSA residue fragments.
2. **Geometric Metrics**: Computing distances between endpoints across residues, correlating them with nanocube volume, and detecting key frames for conformational changes.
3. **Frame Selection**: Automatically selecting representative frames based on clustering, change-point detection, and endpoint-derived scores.
4. **GSA-Specific Analysis**: Tailored metrics for nanocube geometry (e.g., face planarity, edge lengths, volume) and guest positioning.

The pipeline is designed for systems like Amber/GROMACS trajectories with explicit hydrogens. It handles optional dependencies gracefully and outputs CSVs, PDBs, and plots for visualization.

### Key Features
- **Endpoint-Based Dynamics**: Tracks inter-residue endpoint distances to quantify expansion/shrinkage.
- **Volume Correlation**: Identifies endpoint pairs driving structural changes (e.g., Pearson correlation >0.6).
- **Crucial Frame Labeling**: Tags frames for guest entry (min distance to center) or extreme volume events.
- **Visualization**: Plots endpoint distances vs. time/volume; saves selected frames as PDBs.
- **Modularity**: Classes for metrics, clustering, selection, GSA analysis, endpoints, and plotting.

### Dependencies
Install via `pip` (core + optional):

```bash
# Core
pip install MDAnalysis numpy pandas scikit-learn matplotlib

# Optional (for advanced clustering/change points)
pip install hdbscan ruptures

# For endpoint analysis (required for full functionality)
pip install rdkit

# Custom module (included)
# Save as endpoints_finder.py in the repo
```

No internet access needed post-install; all analysis is offline.

### Project Structure
```
gsa-md-pipeline/
├── trajectory_analysis.py     # Main pipeline classes (your provided code)
├── endpoints_finder.py        # Endpoint detection module (your provided code)
├── README.md                  # This file
├── workflow.py                # Example workflow script
├── example_workflow.ipynb     # Jupyter notebook (optional, generated below)
└── tests/                     # Unit tests for endpoints (optional)
```

## Quick Start

1. **Prepare Input**:
   - Topology: e.g., `gsa.prmtop` or `gsa.pdb`.
   - Trajectory: e.g., `gsa.nc` or `gsa.dcd`.
   - Residue IDs: Identify GSA residues (e.g., resids 1-24 for 6 faces × 4 residues).

2. **Run Workflow**:
   ```bash
   python workflow.py --topology gsa.prmtop --trajectory gsa.nc --out_prefix gsa_analysis
   ```

3. **Outputs**:
   - `gsa_analysis_gsa_metrics.csv`: Nanocube geometry (volume, planarity, guest dist).
   - `gsa_analysis_endpoints.csv`: Endpoint distances (min/mean/max per pair).
   - `gsa_analysis_correlations.csv`: Endpoint-volume correlations.
   - `gsa_analysis_selected_frames.csv`: Scores and labels for crucial frames.
   - `gsa_analysis_frames/frame_XXXXXX.pdb`: Selected PDBs.
   - Plots: `gsa_analysis_endpoint_distances.png`, `gsa_analysis_volume_correlation.png`.

## Usage

### Core Classes
- **`TrajectoryMetrics`**: RMSD, RMSF, Rg, PCA, strain.
- **`ClusteringAnalysis`**: Frame clustering (HDBSCAN/KMeans), change-point detection.
- **`FrameSelection`**: Scores/selects frames using endpoints + metrics.
- **`GSAnalyzer`**: Nanocube-specific (faces, volume, planarity, guest dist).
- **`EndpointAnalyzer`**: Finds endpoints, computes distances/correlations.
- **`Plotter`**: Time-series and correlation plots.
- **`FileIO`**: Saves PDBs.

### Configuration
- Customize in `workflow.py`: Residue selections, finder params (e.g., ring gap for aromatics), thresholds (e.g., correlation >0.7 for "crucial" pairs).
- For explicit H: Enable `step_back_from_terminals=True` in `EndpointsFinder`.

### Example: Basic Analysis
```python
import MDAnalysis as mda
from trajectory_analysis import GSAnalyzer, EndpointAnalyzer, Plotter
from endpoints_finder import EndpointsFinder

u = mda.Universe("topology.pdb", "trajectory.dcd")
gsa_sel = "resname GSA"

# Auto-detect faces (6 for cube)
face_sels = GSAnalyzer.gsa_auto_faces_by_kmeans(u, gsa_sel=gsa_sel, n_faces=6)

# GSA metrics
gsa_df = GSAnalyzer.gsa_nanocube_metrics(u, face_sels, out_prefix="test")

# Residue endpoints (e.g., resids 1-24)
res_sel_list = [f"resid {i} and {gsa_sel}" for i in range(1, 25)]
finder = EndpointsFinder(step_back_from_terminals=True)  # Handles explicit H
endpoint_df = EndpointAnalyzer.compute_endpoint_metrics(u, res_sel_list, finder)

# Correlations
volume = gsa_df['volume'].values
dists_dict = EndpointAnalyzer.compute_endpoint_distances(u, res_sel_list, finder)
corr_df = EndpointAnalyzer.compute_endpoint_volume_correlation(dists_dict, volume, res_sel_list)

# Plot
Plotter.plot_endpoint_volume_correlation(dists_dict, volume, res_sel_list, corr_df, "test")
```

## Crucial Frame Labeling
- **Shrinkage/Expansion**: Frames where `volume` change > threshold (e.g., ±10% from mean); label via `|Δvolume| / mean > 0.1`.
- **Guest Entry**: Frames where `guest_min_center_dist` < threshold (e.g., 5 Å).
- Integrated in `FrameSelection`: Boost scores for endpoint extremes + volume events.

## Limitations
- Assumes ~24 GSA residues (adjustable).
- RDKit for endpoints: Best for small fragments; fallback to geometric if fails.
- Volume approx. as `edge_mean³`—use convex hull for distorted cubes (future enhancement).
- Tested on Amber; adapt selections for other formats.

## Contributing
- Add tests: `pytest tests/test_endpoints.py`.
- Report issues: E.g., ring endpoint clustering in aromatics.

## License
MIT—feel free to adapt for publications (cite MDAnalysis/RDKit).

---

# Workflow: GSA Nanocube Endpoint-Driven Analysis

This workflow script (`workflow.py`) orchestrates the pipeline. Run with:
```bash
python workflow.py --topology gsa.prmtop --trajectory gsa.nc --gsa_resids "1-24" --guest_sel "resname GUEST" --out_prefix gsa_analysis --max_frames 20
```

It labels crucial frames (e.g., "expansion", "guest_entry") in the output CSV.

```python
#!/usr/bin/env python3
"""
GSA Nanocube MD Analysis Workflow
================================
Step-by-step pipeline: Load trajectory → GSA metrics → Endpoints → Correlations → Frame selection → Outputs.
Usage: python workflow.py [args]
"""

import argparse
import os
import warnings
from typing import List
import numpy as np
import pandas as pd
import MDAnalysis as mda

# Import pipeline (assume in same dir or PYTHONPATH)
from trajectory_analysis import (
    TrajectoryMetrics, ClusteringAnalysis, FrameSelection, GSAnalyzer,
    EndpointAnalyzer, Plotter, FileIO
)
from endpoints_finder import EndpointsFinder

def parse_args():
    parser = argparse.ArgumentParser(description="GSA Nanocube Trajectory Analysis")
    parser.add_argument("--topology", required=True, help="Topology file (e.g., prmtop/pdb)")
    parser.add_argument("--trajectory", required=True, help="Trajectory file (e.g., nc/dcd)")
    parser.add_argument("--gsa_resids", default="1-24", help="GSA residue IDs (e.g., '1-24' or '1 5 10')")
    parser.add_argument("--guest_sel", default=None, help="Guest selection (e.g., 'resname GUEST')")
    parser.add_argument("--out_prefix", default="analysis", help="Output prefix")
    parser.add_argument("--max_frames", type=int, default=20, help="Max selected frames")
    parser.add_argument("--corr_threshold", type=float, default=0.6, help="Min |correlation| for key pairs")
    parser.add_argument("--volume_threshold", type=float, default=0.1, help="Fractional volume change for labeling")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading {args.topology} + {args.trajectory}...")

    # Step 1: Load Universe
    u = mda.Universe(args.topology, args.trajectory)
    gsa_sel = "resname GSA"

    # Parse resids
    if '-' in args.gsa_resids:
        start, end = map(int, args.gsa_resids.split('-'))
        resids = list(range(start, end + 1))
    else:
        resids = list(map(int, args.gsa_resids.split()))
    res_sel_list: List[str] = [f"resid {rid} and {gsa_sel}" for rid in resids]
    print(f"Analyzing {len(res_sel_list)} GSA residues")

    # Step 2: GSA Nanocube Metrics
    face_sels = GSAnalyzer.gsa_auto_faces_by_kmeans(u, gsa_sel=gsa_sel, n_faces=6)
    gsa_df = GSAnalyzer.gsa_nanocube_metrics(
        u, face_sel_list=face_sels, guest_sel=args.guest_sel, out_prefix=args.out_prefix
    )
    volume = gsa_df['volume'].values
    mean_vol = np.nanmean(volume)
    print(f"GS A metrics saved: Mean volume = {mean_vol:.1f} Å³")

    # Step 3: Endpoint Analysis
    finder = EndpointsFinder(
        angle_tol_deg=10.0,
        use_graph_farness=True,
        alpha=0.3,
        ring_min_gap_deg=45.0,  # For aromatic GSA
        step_back_from_terminals=True  # Handles explicit H
    )
    endpoint_df = EndpointAnalyzer.compute_endpoint_metrics(u, res_sel_list, finder)
    dists_dict = EndpointAnalyzer.compute_endpoint_distances(u, res_sel_list, finder)
    endpoint_df.to_csv(f"{args.out_prefix}_endpoints.csv", index=False)
    print("Endpoint metrics saved")

    # Step 4: Correlations & Key Pairs
    corr_df = EndpointAnalyzer.compute_endpoint_volume_correlation(dists_dict, volume, res_sel_list)
    corr_df.to_csv(f"{args.out_prefix}_correlations.csv", index=False)
    key_pairs = EndpointAnalyzer.identify_key_endpoint_pairs_for_expansion(
        dists_dict, volume, res_sel_list,
        variation_threshold=0.05, correlation_threshold=args.corr_threshold, top_n=10
    )
    key_pairs.to_csv(f"{args.out_prefix}_key_pairs.csv", index=False)
    print(f"Correlations saved: {len(key_pairs[key_pairs['abs_correlation'] > args.corr_threshold])} crucial pairs")

    # Step 5: Basic Metrics for Selection
    rmsd = TrajectoryMetrics.compute_rmsd(u, gsa_sel)
    rg = TrajectoryMetrics.radius_of_gyration(u, gsa_sel)
    pcs, _ = TrajectoryMetrics.pca_on_fluctuations(u, gsa_sel, n_components=3)
    strain = TrajectoryMetrics.local_affine_strain_proxy(u, gsa_sel)

    # Clustering & Change Points
    clustering = ClusteringAnalysis()
    _, medoids = clustering.cluster_frames(pcs)
    cpd_idx = clustering.change_points(volume, n_bkps=5)  # Use volume for events

    # Step 6: Score & Select Frames
    scores, score_df = FrameSelection.score_frames(rmsd, rg, pcs, cpd_idx, strain)
    chosen_frames, selected_df = FrameSelection.select_meaningful_frames(
        medoids=medoids,
        cpd_idx=cpd_idx,
        scores=scores,
        pcs=pcs,
        max_frames=args.max_frames,
        endpoint_dists_array=dists_dict,
        endpoint_metrics_df=endpoint_df
    )
    selected_df.to_csv(f"{args.out_prefix}_selected_frames.csv", index=False)

    # Label Crucial Frames
    T = len(volume)
    labels = ['normal'] * T
    delta_vol = np.abs(np.diff(volume, prepend=volume[0]))
    expansion_mask = (delta_vol / mean_vol > args.volume_threshold) & (np.diff(volume) > 0)
    shrinkage_mask = (delta_vol / mean_vol > args.volume_threshold) & (np.diff(volume) < 0)
    guest_entry_mask = np.array([d < 5.0 for d in gsa_df['guest_min_center_dist']]) if args.guest_sel else np.zeros(T, bool)

    labels = np.array(labels)
    labels[1:][expansion_mask] = 'expansion'
    labels[1:][shrinkage_mask] = 'shrinkage'
    labels[guest_entry_mask] = 'guest_entry'  # Overwrite if overlap
    selected_df['label'] = [labels[i] for i in selected_df.index if 'frame' in selected_df.columns else labels[chosen_frames]]

    print(f"Selected {len(chosen_frames)} frames; Crucial: {np.sum(labels != 'normal')} labeled events")

    # Step 7: Outputs
    os.makedirs(f"{args.out_prefix}_frames", exist_ok=True)
    FileIO.save_frames_as_pdb(u, chosen_frames, args.out_prefix, sel=gsa_sel)
    gsa_df.to_csv(f"{args.out_prefix}_gsa_metrics.csv", index=False)

    # Plots
    Plotter.plot_endpoint_distances(dists_dict, res_sel_list, args.out_prefix)
    Plotter.plot_endpoint_volume_correlation(
        dists_dict, volume, res_sel_list, corr_df, args.out_prefix, top_n=5
    )

    print(f"Analysis complete! Outputs in {args.out_prefix}_*")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress non-critical
    main()
```

### Jupyter Notebook Outline (example_workflow.ipynb)
For interactive use, copy this into a notebook:

```python
# Cell 1: Imports & Load
import MDAnalysis as mda
# ... (imports as above)
u = mda.Universe("gsa.prmtop", "gsa.nc")

# Cell 2: GSA Metrics
# (Code from Step 2)

# Cell 3: Endpoints & Distances
# (Code from Step 3; visualize finder on one residue)

# Cell 4: Correlations
# (Code from Step 4; plot top pairs)

# Cell 5: Frame Selection & Labeling
# (Code from Step 6; inspect selected_df)

# Cell 6: Outputs & Plots
# (Code from Step 7)
```
