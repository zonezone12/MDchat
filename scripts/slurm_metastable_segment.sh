#!/bin/bash
#SBATCH --job-name=meta_seg
#SBATCH --output=slurm_meta_seg_%A_%a.out
#SBATCH --error=slurm_meta_seg_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=your_partition_name

# Per-trajectory segmentation (SLURM array task).
# Submitted by submit_metastable_parallel.sh — do not run directly unless configured.

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must be set}"
: "${TOPOLOGY:?TOPOLOGY must be set}"
: "${WORK_DIR:?WORK_DIR must be set}"
: "${TRAJ_LIST:?TRAJ_LIST must be set}"
: "${SELECTION:?SELECTION must be set}"

cd "${REPO_ROOT}"

TRAJ_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TRAJ_LIST}")
if [[ -z "${TRAJ_FILE}" ]]; then
  echo "No trajectory for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: ${TRAJ_FILE}"

python scripts/segment_single_trajectory.py \
  --topology "${TOPOLOGY}" \
  --trajectory "${TRAJ_FILE}" \
  --work-dir "${WORK_DIR}" \
  --selection "${SELECTION}" \
  --stride "${STRIDE:-1}" \
  --ref-frame "${REF_FRAME:-0}" \
  --signal-metric "${SIGNAL_METRIC:-rmsd}" \
  --method "${RUPTURES_METHOD:-Pelt}" \
  --cost-model "${RUPTURES_COST:-rbf}" \
  --min-segment-frames "${MIN_SEGMENT_FRAMES:-50}" \
  ${PENALTY:+--penalty "${PENALTY}"} \
  ${N_BKPS:+--n-bkps "${N_BKPS}"}
