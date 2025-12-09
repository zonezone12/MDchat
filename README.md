# GSA Nanocube MD Trajectory Analysis Pipeline

## Overview

This repository provides a modular Python pipeline for analyzing molecular dynamics (MD) trajectories of Gear shape amphiphile (GSA) nanocubes, built on [MDAnalysis](https://www.mdanalysis.org/). The pipeline focuses on identifying structural dynamics, particularly shrinkage/expansion events and guest molecule interactions, by:

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

No internet access needed post-install; all analysis is offline.

### Project Structure
```
MD_analysis/
├── src/                                # Modular package structure
│   ├── AlignedTrajectory/             # Trajectory alignment and superposition
│   ├── ClusteringAnalysis/            # Frame clustering (HDBSCAN/KMeans)
│   ├── EndpointAnalyzer/              # Endpoint finding and analysis
│   ├── FileIO/                        # File I/O utilities
│   ├── FrameGatherer/                 # Efficient frame data collection
│   ├── FrameProcessor/                # Frame processing utilities
│   ├── FrameSelection/                # Frame selection and scoring
│   ├── MetricRegistry/                # Metric registration system
│   ├── Plotter/                       # Visualization and plotting
│   ├── task/                          # GSA-specific analysis (GSAnalyzer)
│   ├── TrajectoryIterator/            # Trajectory iteration utilities
│   ├── TrajectoryMetrics/             # RMSD, RMSF, Rg, PCA, strain
│   └── VolumeAnalyzer/                # Volume analysis (voxel-based)
├── run_trajectory_analysis.py         # Main CLI entry point (full workflow)
├── run_volume_endpoint_analysis.py    # Focused script for volume and endpoint analysis only
├── run_volume_endpoint_guest_analysis.py  # Enhanced script with guest entering/exiting analysis and simulation scoring
├── trajectory_deformation_workflow.py # Legacy workflow (deprecated, use src/)
├── endpoints_finder.py                # Legacy endpoint finder (moved to src/EndpointAnalyzer/)
├── volume_analyser.py                 # Legacy volume analyzer (moved to src/VolumeAnalyzer/)
├── gs_analyzer.py                     # Legacy GSA analyzer (moved to src/task/)
├── plotly_molecule.py                 # Interactive 3D molecule visualization
├── __init__.py                        # Package initialization
├── README.md                          # This file
├── script_in_hpc.py                   # HPC batch processing script
├── ngl_traj_plot.ipynb                # Jupyter notebook for visualization
├── MDA.py                             # Example/test script
└── test/                              # Test outputs and examples
```

## Quick Start

1. **Prepare Input**:
   - Topology: e.g., `gsa.prmtop` or `gsa.pdb`.
   - Trajectory: e.g., `gsa.nc` or `gsa.dcd`.
   - Residue IDs: Identify GSA residues (e.g., resids 1-24 for 6 faces × 4 residues).

2. **Run Full Workflow**:
   ```bash
   python run_trajectory_analysis.py --top gsa.prmtop --traj gsa.nc --out_prefix gsa_analysis
   ```
   
   For endpoint-based analysis with cube volume correlation:
   ```bash
   python run_trajectory_analysis.py --top gsa.prmtop --traj mdcrd_v --out_prefix gsa_analysis \
     --endpoint_residues "resid 1" "resid 2" "resid 3" ... \
     --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6"
   ```

3. **Run Focused Volume/Endpoint Analysis** (faster, no clustering/frame selection):
   ```bash
   python run_volume_endpoint_analysis.py \
     --top topology.prmtop \
     --traj trajectory.dcd \
     --out_prefix analysis_output \
     --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
     --cube_faces "resid 1-6" \
     --guest_sel "name I" \
     --plot_top_correlations 10
   ```

4. **Run Enhanced Volume/Endpoint/Guest Analysis** (includes guest entering/exiting tracking and simulation scoring):
   ```bash
   python run_volume_endpoint_guest_analysis.py \
     --top topology.prmtop \
     --traj trajectory.dcd \
     --out_prefix analysis_output \
     --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
     --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
     --guest_sel "name I" \
     --guest_tracking_method distance \
     --plot_top_correlations 10 \
     --n_jobs 4
   ```

5. **Outputs**:
   - `gsa_analysis_metrics.csv`: Global trajectory metrics (RMSD, Rg, PCs, strain).
   - `gsa_analysis_scores.csv`: Frame scores with clustering labels and selection flags.
   - `gsa_analysis_endpoint_metrics.csv`: Endpoint-based metrics per residue (if endpoints enabled).
   - `gsa_analysis_endpoint_volume_correlation.csv`: Correlation between endpoint distances and cube volume.
   - `gsa_analysis_key_endpoint_pairs.csv`: Key endpoint pairs driving expansion/shrinkage.
   - `gsa_analysis_frames/frame_XXXXXX.pdb`: Selected representative frames as PDBs.
   - `gsa_analysis_SUMMARY.txt`: Text summary of analysis results.
   - Plots: `gsa_analysis_endpoint_distances.png`, `gsa_analysis_endpoint_volume_correlation.png`.
   
   **Additional outputs from `run_volume_endpoint_guest_analysis.py`**:
   - `{prefix}_guest_entering_stats.csv`: Guest residence statistics (entry/exit counts, durations).
   - `{prefix}_guest_entering_events.csv`: Detailed entry/exit events with frame numbers and times.
   - `{prefix}_simulation_score.csv`: Overall simulation quality score and component scores.
   - `{prefix}_guest_entry_exit_timeline.gif`: Animated timeline of guest entry/exit events.

## Usage

### Core Classes

All core classes are organized in the `src/` package:

- **`AlignedTrajectory`** (`src.AlignedTrajectory`): Trajectory alignment and superposition management.
- **`TrajectoryMetrics`** (`src.TrajectoryMetrics`): RMSD, RMSF, Rg, PCA, strain, contact distances.
- **`ClusteringAnalysis`** (`src.ClusteringAnalysis`): Frame clustering (HDBSCAN/KMeans), change-point detection.
- **`FrameSelection`** (`src.FrameSelection`): Scores/selects frames using endpoints + metrics.
- **`GSAnalyzer`** (`src.task`): Nanocube-specific (faces, volume, planarity, guest dist).
- **`EndpointAnalyzer`** (`src.EndpointAnalyzer`): Finds endpoints, computes distances/correlations.
- **`EndpointsFinder`** (`src.EndpointAnalyzer`): Endpoint detection algorithm.
- **`VolumeAnalyzer`** (`src.VolumeAnalyzer`): Target volume and cavity volume analysis using voxel grids.
- **`Plotter`** (`src.Plotter`): Time-series and correlation plots.
- **`FileIO`** (`src.FileIO`): Saves PDBs.
- **`FrameGatherer`** (`src.FrameGatherer`): Efficient frame data collection in single trajectory iteration.
- **`TrajectoryIterator`** (`src.TrajectoryIterator`): Trajectory iteration utilities with observer pattern.

### Configuration
- Customize in `run_trajectory_analysis.py` or via CLI arguments: Residue selections, finder params (e.g., ring gap for aromatics), thresholds (e.g., correlation >0.7 for "crucial" pairs).
- For explicit H: Enable `step_back_from_terminals=True` in `EndpointsFinder`.
- Volume analysis: Adjust `spacing` (grid resolution) and `probe_radius` in `VolumeAnalyzer` for different cavity sizes.

### Example: Basic Analysis
```python
import MDAnalysis as mda
from src.task import GSAnalyzer
from src.EndpointAnalyzer import EndpointAnalyzer, EndpointsFinder
from src.Plotter import Plotter

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
from src.VolumeAnalyzer import VolumeAnalyzer
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

Example with all options:
```bash
python run_trajectory_analysis.py \
  --top topology.prmtop \
  --traj trajectory.dcd \
  --out_prefix analysis_output \
  --max_frames 20 \
  --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --guest_sel "I-" \
  --plot_top_correlations 10
```

The script automatically labels crucial frames and generates comprehensive outputs.

---

## Focused Volume and Endpoint Analysis

For cases where you only need endpoint and volume analysis without the full workflow (clustering, frame selection, etc.), use `run_volume_endpoint_analysis.py`. This script is faster and focuses exclusively on:

1. **Endpoint-based analysis** for specified residues
2. **Volume analysis** (cube volume from face selections)
3. **Correlation analysis** between endpoint distances and volume

### Usage

```bash
python run_volume_endpoint_analysis.py \
  --top <topology_file> \
  --traj <trajectory_file(s)> \
  --out_prefix <output_prefix> \
  --endpoint_residues <residue_selections> \
  [--cube_faces <face_selections>] \
  [--guest_sel <guest_selection>] \
  [--plot_top_correlations <N>] \
  [--endpoint_angle_tol <degrees>] \
  [--endpoint_alpha <weight>]
```

### Arguments

- `--top` (required): Topology file (PDB/PSF/PRMTOP/etc.)
- `--traj` (required): One or more trajectory files (XTC/DCD/TRR/etc.)
- `--out_prefix` (required): Prefix for all output files
- `--endpoint_residues` (required): List of residue selection strings for endpoint analysis (e.g., `"resid 1" "resid 2" "resid 3"`)
- `--cube_faces` (optional): List of face selection strings for cube volume computation (e.g., `"resid 1-4" "resid 5-8"`)
- `--guest_sel` (optional): Selection for guest molecule (e.g., `"I-"`)
- `--plot_top_correlations` (optional): Number of top endpoint pairs to plot for volume correlation (default: 5)
- `--endpoint_angle_tol` (optional): Angle tolerance in degrees for endpoint finding (default: 15.0)
- `--endpoint_alpha` (optional): Graph farness weight for endpoint finding (default: 0.2)

### Example

```bash
python run_volume_endpoint_analysis.py \
  --top BMMpM_ca.prmtop \
  --traj BMMpM_mdcrd_v \
  --out_prefix BMMpM_analy \
  --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --guest_sel "I-" \
  --plot_top_correlations 10
```

### Output Files (from `run_volume_endpoint_analysis.py`)

- `{prefix}_endpoint_metrics.csv`: Endpoint-based metrics per residue
- `{prefix}_endpoint_volume_correlation.csv`: Correlation between endpoint distances and cube volume
- `{prefix}_endpoint_variation.csv`: Variation analysis of endpoint pair distances
- `{prefix}_key_endpoint_pairs.csv`: Key endpoint pairs driving expansion/shrinkage (variation_score ≥ 0.1, |correlation| ≥ 0.7)
- `{prefix}_gsa_nanocube.csv`: Cube metrics including volume, edge lengths, planarity
- `{prefix}_endpoint_distances.png`: Plot of endpoint distances over time
- `{prefix}_endpoint_volume_correlation.png`: Plot of top endpoint-volume correlations
- `{prefix}_volume_change.png`: Plot of volume change over frames
- `{prefix}_residue_endpoints.png`: Visualization of residue endpoints

### Key Features

- **Single trajectory iteration**: Uses `FrameGatherer` to efficiently collect coordinates for all selections in one pass
- **Automatic correlation analysis**: Identifies endpoint pairs that correlate with volume changes
- **Variation analysis**: Quantifies how much each endpoint pair distance varies over time
- **Key pair identification**: Automatically finds endpoint pairs most relevant for expansion/shrinkage events
- **Comprehensive plotting**: Generates multiple visualization plots for analysis

---

## Enhanced Volume, Endpoint, and Guest Analysis

For comprehensive analysis including guest molecule tracking and simulation quality scoring, use `run_volume_endpoint_guest_analysis.py`. This enhanced script provides:

1. **Endpoint-based analysis** for specified residues
2. **Volume analysis** (cube volume from face selections) with integrated guest tracking
3. **Correlation analysis** between endpoint distances and volume
4. **Guest entering/exiting analysis** with detailed residence statistics
5. **Simulation quality scoring** based on guest entry, volume dynamics, and correlations
6. **Parallel processing support** with automatic memory management for HPC environments

### Usage for Enhanced Guest Analysis

```bash
python run_volume_endpoint_guest_analysis.py \
  --top <topology_file> \
  --traj <trajectory_file(s)> \
  --out_prefix <output_prefix> \
  --endpoint_residues <residue_selections> \
  [--cube_faces <face_selections>] \
  [--guest_sel <guest_selection>] \
  [--guest_tracking_method {distance,volume}] \
  [--guest_distance_threshold <Å>] \
  [--plot_top_correlations <N>] \
  [--endpoint_angle_tol <degrees>] \
  [--endpoint_alpha <weight>] \
  [--n_jobs <N>] \
  [--use_dask] \
  [--auto_limit_workers]
```

### Arguments for Enhanced Script

- `--top` (required): Topology file (PDB/PSF/PRMTOP/etc.)
- `--traj` (required): One or more trajectory files (XTC/DCD/TRR/etc.)
- `--out_prefix` (required): Prefix for all output files
- `--align_sel` (optional): Selection for trajectory alignment (default: `resid 1-6`)
- `--endpoint_residues` (required): List of residue selection strings for endpoint analysis
- `--cube_faces` (optional): List of face selection strings for cube volume computation
- `--guest_sel` (optional): Selection for guest molecule (e.g., `"name I"` or `"resname IOD"`)
- `--guest_tracking_method` (optional): Method to determine if guest is inside host: `"distance"` (faster) or `"volume"` (more accurate, default: `distance`)
- `--guest_distance_threshold` (optional): Distance threshold in Angstrom for guest entering detection (if None, auto-calculates)
- `--plot_top_correlations` (optional): Number of top endpoint pairs to plot (default: 5)
- `--endpoint_angle_tol` (optional): Angle tolerance in degrees for endpoint finding (default: 15.0)
- `--endpoint_alpha` (optional): Graph farness weight for endpoint finding (default: 0.2)
- `--n_jobs` (optional): Number of parallel jobs. Default: 1 (sequential). Use > 1 for parallel processing. Use -1 to use all CPU cores (WARNING: may cause OOM on large trajectories)
- `--use_dask` (optional): Use Dask Distributed for parallel processing (requires dask installed)
- `--auto_limit_workers` (optional): Automatically limit number of workers based on available memory. Recommended for HPC environments (default: True)

### Example for Enhanced Script

```bash
python run_volume_endpoint_guest_analysis.py \
  --top BMMpM_ca.prmtop \
  --traj BMMpM_mdcrd_v \
  --out_prefix BMMpM_analysis \
  --align_sel "resid 1-6" \
  --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --guest_sel "resname IOD" \
  --guest_tracking_method distance \
  --plot_top_correlations 10 \
  --n_jobs 4 \
  --auto_limit_workers
```

### Output Files (from `run_volume_endpoint_guest_analysis.py`)

- `{prefix}_endpoint_metrics.csv`: Endpoint-based metrics per residue
- `{prefix}_endpoint_volume_correlation.csv`: Correlation between endpoint distances and cube volume
- `{prefix}_endpoint_variation.csv`: Variation analysis of endpoint pair distances
- `{prefix}_key_endpoint_pairs.csv`: Key endpoint pairs driving expansion/shrinkage (variation_score ≥ 0.1, |correlation| ≥ 0.7)
- `{prefix}_gsa_nanocube.csv`: Cube metrics including volume, edge lengths, planarity
- `{prefix}_guest_entering_stats.csv`: Guest residence statistics (first entry frame/time, entry/exit counts, durations)
- `{prefix}_guest_entering_events.csv`: Detailed entry/exit events with frame numbers, times, and guest indices
- `{prefix}_simulation_score.csv`: Overall simulation quality score and component scores (guest entry, volume dynamics, correlation, structural stability)
- `{prefix}_endpoint_distances.png`: Plot of endpoint distances over time
- `{prefix}_endpoint_volume_correlation.png`: Plot of top endpoint-volume correlations
- `{prefix}_volume_change.png`: Plot of volume change over frames
- `{prefix}_residue_endpoints.png`: Visualization of residue endpoints
- `{prefix}_guest_entry_exit_timeline.gif`: Animated timeline of guest entry/exit events

### Enhanced Features

- **Guest entering/exiting tracking**: Automatically detects when guest molecules enter or exit the host structure
- **Residence statistics**: Computes total time inside/outside, entry/exit counts, and stay durations
- **Simulation quality scoring**: Provides overall score (0-1) based on guest entry, volume dynamics, correlations, and structural stability
- **Parallel processing**: Supports multi-core processing with automatic memory management for HPC environments
- **Memory management**: Automatic worker limiting based on available memory to prevent OOM errors
- **Comprehensive visualization**: Generates animated GIFs showing guest entry/exit timeline

---

## Output Files

The pipeline generates several output files:

### From `run_trajectory_analysis.py`:
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

### Additional from `run_volume_endpoint_guest_analysis.py`:
- `{prefix}_guest_entering_stats.csv`: Guest residence statistics
- `{prefix}_guest_entering_events.csv`: Detailed entry/exit events
- `{prefix}_simulation_score.csv`: Simulation quality score and component scores
- `{prefix}_guest_entry_exit_timeline.gif`: Animated timeline visualization

## Programmatic Usage

For programmatic access, import classes directly from the `src` package:

```python
from src.AlignedTrajectory import AlignedTrajectory
from src.TrajectoryMetrics import TrajectoryMetrics
from src.ClusteringAnalysis import ClusteringAnalysis
from src.FrameSelection import FrameSelection
from src.task import GSAnalyzer
from src.EndpointAnalyzer import EndpointAnalyzer, EndpointsFinder
from src.Plotter import Plotter
from src.FileIO import FileIO
from src.VolumeAnalyzer import VolumeAnalyzer
```

Alternatively, you can use the package-level imports (backward compatible):

```python
from MD_analysis import (
    AlignedTrajectory, TrajectoryMetrics, ClusteringAnalysis,
    FrameSelection, GSAnalyzer, EndpointAnalyzer, Plotter, FileIO,
    VolumeAnalyzer, EndpointsFinder
)
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

### Guest Entering/Exiting Analysis
The `GSAnalyzerObserver` class (used in `run_volume_endpoint_guest_analysis.py`) tracks guest molecules:
- **Distance-based tracking**: Faster method using distance from guest to host center
- **Volume-based tracking**: More accurate method checking if guest is inside host volume
- **Automatic threshold calculation**: Computes optimal distance threshold from host structure
- **Residence statistics**: Tracks entry/exit events, durations, and total time inside/outside
- **Timeline visualization**: Generates animated GIFs showing guest position over time

### Simulation Quality Scoring
The `FrameSelection.score_simulation()` method provides:
- **Overall score** (0-1): Weighted combination of component scores
- **Guest entry score**: Based on whether guest entered and residence time
- **Volume dynamics score**: Based on volume change magnitude
- **Correlation score**: Based on endpoint-volume correlation strength
- **Structural stability score**: Based on endpoint distance variation
- **Validation flags**: Indicates if simulation meets minimum quality thresholds

## Jupyter Notebooks

See `ngl_traj_plot.ipynb` for interactive trajectory visualization examples. The notebook demonstrates:
- Loading and visualizing trajectories
- Endpoint visualization
- Volume analysis integration
