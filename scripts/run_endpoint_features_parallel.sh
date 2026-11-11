#!/bin/bash
# Endpoint features for every trajectory of one cube.
#
# One SLURM job, one node, 16 CPUs. Four Python processes run at once, each
# on its own trajectory with --n_jobs 1. The timing test on this filesystem
# put the throughput peak there: 8 at once was no faster, and --n_jobs 16
# inside one file only went from 165s to 142s.
#
# Submit from the directory that contains MDchat/ and ${CUBE}_ca.prmtop:
#   sbatch MDchat/scripts/run_endpoint_features_parallel.sh
#
#SBATCH -p r01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH -J endpoint-BMHpM
#SBATCH -o slurm-endpoint-%j.out

set -uo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

CUBE="${CUBE:-BMHpM}"
TRAJ_DIR="${TRAJ_DIR:-$HOME/traj/$CUBE.bak}"
TOPOLOGY="${TOPOLOGY:-${CUBE}_ca.prmtop}"
SCRIPT="${SCRIPT:-MDchat/scripts/compute_gsa_features.py}"
STRIDE="${STRIDE:-5}"
CONCURRENCY="${CONCURRENCY:-4}"
N_JOBS="${N_JOBS:-1}"
OUT="${OUT:-output/endpoint_changepoints_${CUBE}_hullplus/endpoint_features}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ ! -f "$TOPOLOGY" ]]; then
    echo "Topology not found: $TOPOLOGY" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "Script not found: $SCRIPT" >&2
    exit 1
fi

shopt -s nullglob
mapfile -t FILES < <(printf '%s\n' "$TRAJ_DIR"/*/mdcrd_v | sort)
shopt -u nullglob

if (( ${#FILES[@]} == 0 )); then
    echo "No trajectories matched $TRAJ_DIR/*/mdcrd_v" >&2
    exit 1
fi

mkdir -p "$OUT/logs"
FAILS="$OUT/logs/failures.txt"
: > "$FAILS"

echo "============================================================="
echo "Cube: $CUBE"
echo "Trajectory dir: $TRAJ_DIR"
echo "Files: ${#FILES[@]}"
echo "Concurrency: $CONCURRENCY   n_jobs: $N_JOBS   stride: $STRIDE"
echo "Output: $OUT"
echo "Start time: $(date)"
echo "SLURM job ID: ${SLURM_JOB_ID:-local}"
echo "============================================================="

run_one() {
    local file="$1"
    local replica log
    replica="$(basename "$(dirname "$file")")"
    log="$OUT/logs/${replica}.log"
    echo "[start] $replica $(date)"
    python "$SCRIPT" \
        --topology "$TOPOLOGY" \
        --cube "$CUBE" \
        --trajectories "$file" \
        --gsa-resname MOL \
        --guest-selection "resname IOD" \
        --output-dir "$OUT" \
        --stride "$STRIDE" \
        --endpoints-only \
        --include-site-pairs \
        --n_jobs "$N_JOBS" \
        --format TRJ \
        > "$log" 2>&1
    local rc=$?
    if (( rc == 0 )); then
        echo "[done]  $replica $(date)"
    else
        echo "[fail]  $replica exit=$rc $(date)"
        flock -x "$OUT/logs/failures.lock" printf '%s\t%s\n' "$replica" "$rc" >> "$FAILS"
    fi
    return "$rc"
}

for file in "${FILES[@]}"; do
    while true; do
        running=$(jobs -pr | wc -l)
        if (( running < CONCURRENCY )); then
            break
        fi
        sleep 2
    done
    run_one "$file" &
done
wait || true

n_fail=0
if [[ -s "$FAILS" ]]; then
    n_fail=$(wc -l < "$FAILS")
fi
echo "============================================================="
echo "Finished at $(date)"
echo "Files: ${#FILES[@]}    failures: $n_fail"
echo "Output: $OUT"
if (( n_fail > 0 )); then
    echo "Failed replicas listed in $FAILS"
    exit 1
fi
echo "============================================================="
