#!/bin/bash
#SBATCH --job-name=meta_merge
#SBATCH --output=slurm_meta_merge_%j.out
#SBATCH --error=slurm_meta_merge_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=your_partition_name

# Merge segment artifacts and cluster all representatives.
# Runs after the array job completes (dependency set by submit script).

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must be set}"
: "${WORK_DIR:?WORK_DIR must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

cd "${REPO_ROOT}"

python scripts/merge_metastable_states.py \
  --work-dir "${WORK_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --linkage "${LINKAGE:-ward}" \
  --k-max "${K_MAX:-10}" \
  ${RMSD_CUTOFF:+--rmsd-cutoff "${RMSD_CUTOFF}"}
