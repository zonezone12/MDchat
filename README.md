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
We recommend using [mamba](https://mamba.readthedocs.io/) (a faster `conda` alternative) for installing dependencies, as this ensures all packages—including those with compiled C/C++ extensions (like RDKit and MDAnalysis)—are handled robustly.

Install via `mamba` (recommended):

```bash
# Create and activate a new environment (recommended)
mamba create -n gsa-analysis python=3.10
conda activate gsa-analysis

# Core requirements
mamba install mdanalysis numpy pandas scikit-learn matplotlib

# Optional: for advanced clustering and change points
mamba install hdbscan ruptures

# Endpoint analysis (required for full functionality)
mamba install rdkit

# Volume analysis and visualization
mamba install plotly scipy imageio

# All dependencies at once
mamba install mdanalysis numpy pandas scikit-learn matplotlib hdbscan ruptures rdkit plotly scipy imageio
```

If you prefer, you can still use `pip` as shown below (but mamba/conda is recommended for best compatibility):

```bash
pip install MDAnalysis numpy pandas scikit-learn matplotlib hdbscan ruptures rdkit plotly scipy imageio
```

No internet access is needed post-install; all analysis is offline.
```

No internet access needed post-install; all analysis is offline.

### Project Structure
```
MD_analysis/
├── trajectory_deformation_workflow.py  # Main pipeline classes (TrajectoryMetrics, ClusteringAnalysis, etc.)
├── run_trajectory_analysis.py         # Main CLI entry point
├── endpoints_finder.py                # Endpoint detection module
├── volume_analyser.py                 # Volume analysis (target volume, cavity detection)
├── plotly_molecule.py                 # Interactive 3D molecule visualization
├── __init__.py                        # Package initialization
├── README.md                          # This file
├── script_in_hpc.py                   # HPC batch processing script
├── ngl_traj_plot.ipynb                # Jupyter notebook for visualization
├── MDA.py                             # Example/test script
└── tests/                             # Unit tests (optional)
```

## Quick Start

1. **Prepare Input**:
   - Topology: e.g., `gsa.prmtop` or `gsa.pdb`.
   - Trajectory: e.g., `gsa.nc` or `gsa.dcd`.
   - Residue IDs: Identify GSA residues (e.g., resids 1-24 for 6 faces × 4 residues).

2. **Run Workflow**:
   ```bash
   python run_trajectory_analysis.py --top gsa.prmtop --traj gsa.nc --out_prefix gsa_analysis
   ```
   
   For endpoint-based analysis with cube volume correlation:
   ```bash
   python run_trajectory_analysis.py --top gsa.prmtop --traj mdcrd_v --out_prefix gsa_analysis \
     --endpoint_residues "resid 1" "resid 2" "resid 3" ... \
     --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6"
   ```

3. **Outputs**:
   - `gsa_analysis_metrics.csv`: Global trajectory metrics (RMSD, Rg, PCs, strain).
   - `gsa_analysis_scores.csv`: Frame scores with clustering labels and selection flags.
   - `gsa_analysis_endpoint_metrics.csv`: Endpoint-based metrics per residue (if endpoints enabled).
   - `gsa_analysis_endpoint_volume_correlation.csv`: Correlation between endpoint distances and cube volume.
   - `gsa_analysis_key_endpoint_pairs.csv`: Key endpoint pairs driving expansion/shrinkage.
   - `gsa_analysis_frames/frame_XXXXXX.pdb`: Selected representative frames as PDBs.
   - `gsa_analysis_SUMMARY.txt`: Text summary of analysis results.
   - Plots: `gsa_analysis_endpoint_distances.png`, `gsa_analysis_endpoint_volume_correlation.png`.

## Usage

### Core Classes
- **`AlignedTrajectory`**: Trajectory alignment and superposition management.
- **`TrajectoryMetrics`**: RMSD, RMSF, Rg, PCA, strain, contact distances.
- **`ClusteringAnalysis`**: Frame clustering (HDBSCAN/KMeans), change-point detection.
- **`FrameSelection`**: Scores/selects frames using endpoints + metrics.
- **`GSAnalyzer`**: Nanocube-specific (faces, volume, planarity, guest dist).
- **`EndpointAnalyzer`**: Finds endpoints, computes distances/correlations.
- **`VolumeAnalyzer`**: Target volume and cavity volume analysis using voxel grids.
- **`Plotter`**: Time-series and correlation plots.
- **`FileIO`**: Saves PDBs.

### Configuration
- Customize in `run_trajectory_analysis.py` or via CLI arguments: Residue selections, finder params (e.g., ring gap for aromatics), thresholds (e.g., correlation >0.7 for "crucial" pairs).
- For explicit H: Enable `step_back_from_terminals=True` in `EndpointsFinder`.
- Volume analysis: Adjust `spacing` (grid resolution) and `probe_radius` in `VolumeAnalyzer` for different cavity sizes.

### Example: Basic Analysis
```python
import MDAnalysis as mda
from trajectory_deformation_workflow import GSAnalyzer, EndpointAnalyzer, Plotter
from endpoints_finder import EndpointsFinder

u = mda.Universe("topology.pdb", "trajectory.dcd")
gsa_sel = "resname GSA"

# Auto-detect faces (6 for cube)
gsa_analyzer = GSAnalyzer()
face_sels = gsa_analyzer.gsa_auto_faces_by_kmeans(u, gsa_sel=gsa_sel, n_faces=6)

# GSA metrics
gsa_df = gsa_analyzer.gsa_nanocube_metrics(u, face_sels, out_prefix="test")

# Residue endpoints (e.g., resids 1-24)
res_sel_list = [f"resid {i} and {gsa_sel}" for i in range(1, 25)]
finder = EndpointsFinder(step_back_from_terminals=True)  # Handles explicit H
endpoint_analyzer = EndpointAnalyzer()
endpoint_df = endpoint_analyzer.compute_endpoint_metrics(u, res_sel_list, finder)

# Correlations
volume = gsa_df['volume'].values
dists_dict = endpoint_analyzer.compute_endpoint_distances(u, res_sel_list, finder)
corr_df = endpoint_analyzer.compute_endpoint_volume_correlation(dists_dict, volume, res_sel_list)

# Plot
plotter = Plotter()
plotter.plot_endpoint_volume_correlation(dists_dict, volume, res_sel_list, corr_df, "test")
```

### Example: Volume Analysis
```python
from volume_analyser import VolumeAnalyzer
import MDAnalysis as mda

u = mda.Universe("topology.pdb", "trajectory.dcd")

# Initialize volume analyzer
va = VolumeAnalyzer(
    u, 
    selection="not water and not name I and not name Na+",
    spacing=1.0,  # Grid spacing in Å
    probe_radius=1.4  # Water-sized probe
)

# Compute volume for a specific frame
target_vol, cavity_vol, inside_mask, cavities = va.compute_frame(0, return_masks=True)
print(f"Target volume: {target_vol:.2f} Å³")
print(f"Cavity volume: {cavity_vol:.2f} Å³")

# Interactive 3D visualization
fig = va.plot_interactive_3d()
fig.show()

# Generate GIF showing volume computation pipeline
va.make_volume_pipeline_gif(frame_index=0, gif_path="volume_pipeline.gif", atom_stride=5, fps=10)
```

## Crucial Frame Labeling
- **Shrinkage/Expansion**: Frames where `volume` change > threshold (e.g., ±10% from mean); label via `|Δvolume| / mean > 0.1`.
- **Guest Entry**: Frames where `guest_min_center_dist` < threshold (e.g., 5 Å).
- Integrated in `FrameSelection`: Boost scores for endpoint extremes + volume events.

## Limitations
- Assumes ~24 GSA residues (adjustable via selections).
- RDKit for endpoints: Best for small fragments; fallback to geometric if fails.
- Volume computation: `GSAnalyzer` uses `edge_mean³` approximation; `VolumeAnalyzer` provides more accurate voxel-based method.
- Tested on Amber/GROMACS formats; adapt selections for other formats.
- Volume analysis can be memory-intensive for large systems (adjust `spacing` parameter).

## Contributing
- Add tests: `pytest tests/test_endpoints.py`.
- Report issues: E.g., ring endpoint clustering in aromatics.

## License
MIT—feel free to adapt for publications (cite MDAnalysis/RDKit).

---

# Workflow: GSA Nanocube Endpoint-Driven Analysis

The main workflow script (`run_trajectory_analysis.py`) orchestrates the pipeline. Run with:
```bash
python run_trajectory_analysis.py \
  --top gsa.prmtop \
  --traj gsa.nc \
  --out_prefix gsa_analysis \
  --max_frames 20 \
  --endpoint_residues "resid 1" "resid 2" ... "resid 24" \
  --cube_faces "resid 1-4" "resid 5-8" "resid 9-12" "resid 13-16" "resid 17-20" "resid 21-24" \
  --plot_top_correlations 5
```

Key CLI arguments:
- `--top`: Topology file (PDB/PSF/PRMTOP)
- `--traj`: Trajectory file(s) (XTC/DCD/TRR/NC)
- `--out_prefix`: Output prefix for all files
- `--endpoint_residues`: List of residue selections for endpoint analysis
- `--cube_faces`: List of face selections for cube volume computation
- `--max_frames`: Maximum number of frames to save
- `--align_sel`: Selection for trajectory alignment (default: `resid 1`)
- `--cpd_n`: Number of change-points to detect (default: 6)
- `--plot_top_correlations`: Number of top endpoint pairs to plot (default: 5)

The script automatically labels crucial frames and generates comprehensive outputs.

## Output Files

The pipeline generates several output files:

- `{prefix}_metrics.csv`: Global trajectory metrics (RMSD, Rg, PCs, strain, contacts)
- `{prefix}_scores.csv`: Frame scores with clustering labels and selection flags
- `{prefix}_endpoint_metrics.csv`: Endpoint-based metrics per residue (if `--endpoint_residues` provided)
- `{prefix}_endpoint_volume_correlation.csv`: Correlation between endpoint distances and cube volume
- `{prefix}_endpoint_variation.csv`: Variation analysis of endpoint pair distances
- `{prefix}_key_endpoint_pairs.csv`: Key endpoint pairs driving expansion/shrinkage
- `{prefix}_endpoint_distances.png`: Plot of endpoint distances over time
- `{prefix}_endpoint_volume_correlation.png`: Plot of top endpoint-volume correlations
- `{prefix}_SUMMARY.txt`: Text summary of analysis results
- `{prefix}_frames/`: Directory containing selected frames as PDB files

## Programmatic Usage

For programmatic access, import classes directly:

```python
from trajectory_deformation_workflow import (
    AlignedTrajectory, TrajectoryMetrics, ClusteringAnalysis,
    FrameSelection, GSAnalyzer, EndpointAnalyzer, Plotter, FileIO
)
from endpoints_finder import EndpointsFinder
from volume_analyser import VolumeAnalyzer
```

## Advanced Features

### Volume Analysis
The `VolumeAnalyzer` class provides sophisticated volume computation:
- **Target Volume**: Solvent-excluded volume using VDW radii + probe radius
- **Cavity Detection**: Identifies enclosed voids within the structure
- **Interactive Visualization**: 3D Plotly plots with cavity highlighting
- **GIF Generation**: Animated visualization of volume computation pipeline

### Endpoint Analysis
The `EndpointsFinder` class identifies molecular endpoints using:
- Convex hull analysis on 2D projections
- Graph-based topological farness
- Ring-aware endpoint selection for aromatic systems
- Terminal atom handling for explicit hydrogen systems

### Frame Selection Strategy
Frames are selected based on:
1. **Cluster Medoids**: Representative states from HDBSCAN/KMeans clustering
2. **Change-Points**: Transition frames detected via ruptures
3. **Metric Extremes**: Frames with extreme RMSD, Rg, or PC values
4. **Endpoint Dynamics**: Frames with significant endpoint distance changes
5. **Non-redundancy**: Distance threshold in PC space prevents similar frames

## Jupyter Notebooks

See `ngl_traj_plot.ipynb` for interactive trajectory visualization examples. The notebook demonstrates:
- Loading and visualizing trajectories
- Endpoint visualization
- Volume analysis integration
