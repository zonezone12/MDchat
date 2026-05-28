#!/usr/bin/env bash
#
# Submit parallel metastable-state analysis on a SLURM cluster.
#
# Stage 1: SLURM array — one trajectory per task (segment + extract positions)
# Stage 2: single merge job — pairwise RMSD + hierarchical clustering
#
# Usage:
#   ./scripts/submit_metastable_parallel.sh
#
# Or override defaults:
#   TOPOLOGY=traj/BMMpM_ca.prmtop \
#   TRAJ_GLOB="traj/BMMpM_*_mdcrd_v.trj" \
#   OUTPUT_DIR=output/metastable_parallel \
#   PARTITION=gpu \
#   ./scripts/submit_metastable_parallel.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── Config (override via environment) ─────────────────────────────
TOPOLOGY="${TOPOLOGY:-traj/BMMpM_ca.prmtop}"
TRAJ_GLOB="${TRAJ_GLOB:-traj/BMMpM_*_mdcrd_v.trj}"
OUTPUT_DIR="${OUTPUT_DIR:-output/metastable_parallel}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/work}"
SELECTION="${SELECTION:-not water and not resname I and not resname Na+}"

STRIDE="${STRIDE:-10}"
REF_FRAME="${REF_FRAME:-0}"
SIGNAL_METRIC="${SIGNAL_METRIC:-rmsd}"
RUPTURES_METHOD="${RUPTURES_METHOD:-Pelt}"
RUPTURES_COST="${RUPTURES_COST:-rbf}"
MIN_SEGMENT_FRAMES="${MIN_SEGMENT_FRAMES:-50}"
PENALTY="${PENALTY:-}"
N_BKPS="${N_BKPS:-}"

LINKAGE="${LINKAGE:-ward}"
K_MAX="${K_MAX:-10}"
RMSD_CUTOFF="${RMSD_CUTOFF:-}"

PARTITION="${PARTITION:-your_partition_name}"
SEGMENT_TIME="${SEGMENT_TIME:-12:00:00}"
MERGE_TIME="${MERGE_TIME:-02:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
SEGMENT_MEM="${SEGMENT_MEM:-16G}"
MERGE_MEM="${MERGE_MEM:-16G}"

DRY_RUN="${DRY_RUN:-0}"

# ── Build trajectory list ─────────────────────────────────────────
mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"
TRAJ_LIST="${WORK_DIR}/traj_list.txt"

shopt -s nullglob
TRAJ_FILES=(${TRAJ_GLOB})
shopt -u nullglob

if [[ ${#TRAJ_FILES[@]} -eq 0 ]]; then
  echo "No trajectories matched TRAJ_GLOB=${TRAJ_GLOB}" >&2
  exit 1
fi

printf '%s\n' "${TRAJ_FILES[@]}" > "${TRAJ_LIST}"
N_TRAJ=${#TRAJ_FILES[@]}
ARRAY_MAX=$((N_TRAJ - 1))

echo "Repository:  ${REPO_ROOT}"
echo "Topology:    ${TOPOLOGY}"
echo "Trajectories: ${N_TRAJ} (see ${TRAJ_LIST})"
echo "Work dir:    ${WORK_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Selection:   ${SELECTION}"
echo "Partition:   ${PARTITION}"
echo ""

export REPO_ROOT TOPOLOGY WORK_DIR OUTPUT_DIR TRAJ_LIST SELECTION
export STRIDE REF_FRAME SIGNAL_METRIC RUPTURES_METHOD RUPTURES_COST
export MIN_SEGMENT_FRAMES PENALTY N_BKPS LINKAGE K_MAX RMSD_CUTOFF

SEGMENT_SBATCH="${REPO_ROOT}/scripts/slurm_metastable_segment.sh"
MERGE_SBATCH="${REPO_ROOT}/scripts/slurm_metastable_merge.sh"

if [[ ! -f "${TOPOLOGY}" ]]; then
  echo "Topology not found: ${TOPOLOGY}" >&2
  exit 1
fi

SBATCH_SEGMENT=(
  --job-name=meta_seg
  --array="0-${ARRAY_MAX}"
  --output="${OUTPUT_DIR}/slurm_seg_%A_%a.out"
  --error="${OUTPUT_DIR}/slurm_seg_%A_%a.err"
  --time="${SEGMENT_TIME}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${SEGMENT_MEM}"
  --partition="${PARTITION}"
  --export=ALL
)

SBATCH_MERGE=(
  --job-name=meta_merge
  --output="${OUTPUT_DIR}/slurm_merge_%j.out"
  --error="${OUTPUT_DIR}/slurm_merge_%j.err"
  --time="${MERGE_TIME}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MERGE_MEM}"
  --partition="${PARTITION}"
  --export=ALL
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] sbatch ${SBATCH_SEGMENT[*]} ${SEGMENT_SBATCH}"
  echo "[DRY RUN] sbatch --dependency=afterok:<ARRAY_JOB_ID> ${SBATCH_MERGE[*]} ${MERGE_SBATCH}"
  exit 0
fi

if ! command -v sbatch &>/dev/null; then
  echo "sbatch not found. Running locally with GNU parallel / sequential fallback." >&2
  echo "Set USE_LOCAL=1 or install SLURM to use array jobs." >&2

  if [[ "${USE_LOCAL:-0}" == "1" ]]; then
    export REPO_ROOT TOPOLOGY WORK_DIR TRAJ_LIST SELECTION
    export STRIDE REF_FRAME SIGNAL_METRIC RUPTURES_METHOD RUPTURES_COST MIN_SEGMENT_FRAMES
    while IFS= read -r traj; do
      echo "Processing ${traj}..."
      python scripts/segment_single_trajectory.py \
        --topology "${TOPOLOGY}" \
        --trajectory "${traj}" \
        --work-dir "${WORK_DIR}" \
        --selection "${SELECTION}" \
        --stride "${STRIDE}" \
        --ref-frame "${REF_FRAME}" \
        --signal-metric "${SIGNAL_METRIC}" \
        --method "${RUPTURES_METHOD}" \
        --cost-model "${RUPTURES_COST}" \
        --min-segment-frames "${MIN_SEGMENT_FRAMES}" \
        ${PENALTY:+--penalty "${PENALTY}"} \
        ${N_BKPS:+--n-bkps "${N_BKPS}"} &
    done < "${TRAJ_LIST}"
    wait
    python scripts/merge_metastable_states.py \
      --work-dir "${WORK_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --linkage "${LINKAGE}" \
      --k-max "${K_MAX}" \
      ${RMSD_CUTOFF:+--rmsd-cutoff "${RMSD_CUTOFF}"}
    echo "Local run complete: ${OUTPUT_DIR}"
    exit 0
  fi
  exit 1
fi

ARRAY_JOB_ID=$(sbatch --parsable "${SBATCH_SEGMENT[@]}" "${SEGMENT_SBATCH}")
echo "Submitted segment array: ${ARRAY_JOB_ID} (tasks 0-${ARRAY_MAX})"

MERGE_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  "${SBATCH_MERGE[@]}" \
  "${MERGE_SBATCH}")
echo "Submitted merge job:     ${MERGE_JOB_ID} (after array ${ARRAY_JOB_ID})"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  ls ${OUTPUT_DIR}/slurm_*.out"
echo ""
echo "Results (after merge):"
echo "  ${OUTPUT_DIR}/segments_all.csv"
echo "  ${OUTPUT_DIR}/distance_matrix.csv"
echo "  ${OUTPUT_DIR}/dendrogram.png"
echo "  ${OUTPUT_DIR}/cluster_summary.txt"
