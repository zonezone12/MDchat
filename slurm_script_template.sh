#!/bin/bash
#SBATCH --job-name=md_analysis
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=your_partition_name

# Set variables
CUBE="your_cube_name"
TRAJ_ID="your_traj_id"
TRAJ_PATH="path/to/your/trajectory"

# IMPORTANT: When using line continuation with backslash (\),
# ensure there are NO trailing spaces after the backslash.
# The backslash must be the very last character on the line.

python MD_analysis/run_volume_endpoint_guest_analysis.py \
  --top ${CUBE}_ca.prmtop \
  --traj "$TRAJ_PATH" \
  --out_prefix output/${CUBE}_${TRAJ_ID}_test \
  --align_sel "resid 1-6" \
  --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" \
  --guest_sel "name I" \
  --guest_tracking_method distance \
  --plot_top_correlations 10

# Alternative: If you prefer, you can put everything on one line (no backslashes needed):
# python MD_analysis/run_volume_endpoint_guest_analysis.py --top ${CUBE}_ca.prmtop --traj "$TRAJ_PATH" --out_prefix output/${CUBE}_${TRAJ_ID}_test --align_sel "resid 1-6" --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" --guest_sel "name I" --guest_tracking_method distance --plot_top_correlations 10

