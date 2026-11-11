#!/bin/bash
# Time endpoint-feature reads under file-level parallelism.
#
# Same 16-core node as the production job. Each cell uses a different set of
# trajectory files so a later cell is not faster just because /nas3 cached the
# earlier read.
#
#   single_n1   one file, --n_jobs 1     (baseline, one reader)
#   single_n16  one file, --n_jobs 16    (current inner-file pool)
#   files_c4    8 files, 4 at a time, --n_jobs 1
#   files_c8    8 files, 8 at a time, --n_jobs 1
#   files_c16   16 files, 16 at a time, --n_jobs 1
#
# Read output/test_io_parallel/summary.txt when the job finishes.
# mean_file_s / single_n1 is the I/O collision signal:
#   near 1.0  -> extra readers did not stall each other
#   1.5 or more -> that many simultaneous mdcrd reads is too many
#
# Submit from the directory that contains MDchat/ and BMHpM_ca.prmtop:
#   sbatch MDchat/scripts/test_endpoint_io_parallel.sh
#
# Shorter run (drops the 16-reader cell):
#   sbatch --export=ALL,RUN_C16=0 MDchat/scripts/test_endpoint_io_parallel.sh
#
#SBATCH -p r01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 06:00:00
#SBATCH -J endpoint-io-test
#SBATCH -o slurm-io-test-%j.out

set -uo pipefail

# One BLAS thread per process so 16 Python processes do not look slow
# because each one spawned its own thread pool.
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
N_PER_CELL="${N_PER_CELL:-8}"
RUN_C16="${RUN_C16:-1}"
OUT="${OUT:-output/test_io_parallel}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$OUT/logs"

TIMINGS="$OUT/timings.tsv"
CELLS="$OUT/cells.tsv"
printf 'mode\tconcurrency\tn_jobs\tfile\telapsed_s\texit_code\n' > "$TIMINGS"
printf 'mode\tconcurrency\tn_jobs\tn_files\twall_s\n' > "$CELLS"

if [[ ! -f "$TOPOLOGY" ]]; then
    echo "Topology not found: $TOPOLOGY (submit from the directory that contains it)." >&2
    exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "Script not found: $SCRIPT" >&2
    exit 1
fi

shopt -s nullglob
mapfile -t ALL_FILES < <(printf '%s\n' "$TRAJ_DIR"/*/mdcrd_v | sort)
shopt -u nullglob
N_ALL="${#ALL_FILES[@]}"

need=$((2 + N_PER_CELL + N_PER_CELL))
if [[ "$RUN_C16" == "1" ]]; then
    need=$((need + 16))
fi
if (( N_ALL < need )); then
    echo "Found $N_ALL trajectory files under $TRAJ_DIR; this test needs $need." >&2
    echo "Lower N_PER_CELL or set RUN_C16=0." >&2
    exit 1
fi

echo "============================================================="
echo "Endpoint I/O parallelism test"
echo "Start time: $(date)"
echo "SLURM job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Topology: $TOPOLOGY"
echo "Trajectory dir: $TRAJ_DIR"
echo "Files available: $N_ALL"
echo "Stride: $STRIDE"
echo "Files per 4-wide and 8-wide cell: $N_PER_CELL"
echo "Output: $OUT"
echo "============================================================="

next=0

take_files() {
    local n="$1"
    local start="$next"
    local end=$((start + n))
    if (( end > N_ALL )); then
        echo "Not enough files (asked $n from index $start, have $N_ALL)." >&2
        exit 1
    fi
    TAKEN=("${ALL_FILES[@]:start:n}")
    next=$end
}

run_one() {
    local mode="$1"
    local concurrency="$2"
    local n_jobs="$3"
    local file="$4"
    local replica log start end elapsed rc

    replica="$(basename "$(dirname "$file")")"
    log="$OUT/logs/${mode}_${replica}.log"
    start=$(date +%s)
    python "$SCRIPT" \
        --topology "$TOPOLOGY" \
        --cube "$CUBE" \
        --trajectories "$file" \
        --gsa-resname MOL \
        --guest-selection "resname IOD" \
        --output-dir "$OUT/$mode" \
        --stride "$STRIDE" \
        --endpoints-only \
        --include-site-pairs \
        --n_jobs "$n_jobs" \
        --format TRJ \
        > "$log" 2>&1
    rc=$?
    end=$(date +%s)
    elapsed=$((end - start))
    flock -x "$OUT/timings.lock" printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$concurrency" "$n_jobs" "$file" "$elapsed" "$rc" >> "$TIMINGS"
    echo "[$mode] $replica ${elapsed}s exit=$rc"
}

run_cell() {
    local mode="$1"
    local concurrency="$2"
    local n_jobs="$3"
    shift 3
    local files=("$@")
    local start end wall

    mkdir -p "$OUT/$mode"
    echo "----- $mode  concurrency=$concurrency  n_jobs=$n_jobs  files=${#files[@]}  $(date) -----"
    start=$(date +%s)
    for file in "${files[@]}"; do
        while true; do
            local running
            running=$(jobs -pr | wc -l)
            if (( running < concurrency )); then
                break
            fi
            sleep 2
        done
        run_one "$mode" "$concurrency" "$n_jobs" "$file" &
    done
    wait || true
    end=$(date +%s)
    wall=$((end - start))
    flock -x "$OUT/cells.lock" printf '%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$concurrency" "$n_jobs" "${#files[@]}" "$wall" >> "$CELLS"
    echo "----- $mode wall ${wall}s -----"
}

take_files 1
run_cell single_n1 1 1 "${TAKEN[@]}"

take_files 1
run_cell single_n16 1 16 "${TAKEN[@]}"

take_files "$N_PER_CELL"
run_cell files_c4 4 1 "${TAKEN[@]}"

take_files "$N_PER_CELL"
run_cell files_c8 8 1 "${TAKEN[@]}"

if [[ "$RUN_C16" == "1" ]]; then
    take_files 16
    run_cell files_c16 16 1 "${TAKEN[@]}"
fi

python - "$TIMINGS" "$CELLS" "$OUT/summary.txt" << 'PY'
import sys
from pathlib import Path

timings_path, cells_path, summary_path = sys.argv[1:]

def load(path):
    lines = Path(path).read_text().splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        rows.append(dict(zip(header, line.split("\t"))))
    return rows

timings = load(timings_path)
cells = load(cells_path)

baseline = [float(r["elapsed_s"]) for r in timings if r["mode"] == "single_n1"]
base = baseline[0] if baseline else None

lines = []
lines.append("mode        conc  n_jobs  files  wall_s  mean_file_s  files/hour  vs_single_n1  failures")
for cell in cells:
    mode = cell["mode"]
    rows = [r for r in timings if r["mode"] == mode]
    elapsed = [float(r["elapsed_s"]) for r in rows]
    fails = sum(int(r["exit_code"]) != 0 for r in rows)
    mean = sum(elapsed) / len(elapsed) if elapsed else float("nan")
    wall = float(cell["wall_s"])
    n = int(cell["n_files"])
    per_hour = (n / wall * 3600.0) if wall else float("nan")
    ratio = (mean / base) if base else float("nan")
    lines.append(
        f"{mode:<11} {cell['concurrency']:>4}  {cell['n_jobs']:>6}  {n:>5}  "
        f"{wall:>6.0f}  {mean:>11.0f}  {per_hour:>10.1f}  {ratio:>12.2f}  {fails:>8}"
    )

lines.append("")
if base:
    lines.append(f"single_n1 baseline: {base:.0f}s for one file, one reader.")
    lines.append("vs_single_n1 = mean file time / that baseline.")
    lines.append("Near 1.0 means those simultaneous readers did not stall each other.")
    lines.append("1.5 or higher means cut the concurrency back to the last row that stayed near 1.")
single16 = [float(r["elapsed_s"]) for r in timings if r["mode"] == "single_n16"]
if base and single16:
    lines.append(
        f"Inner pool: n_jobs=16 took {single16[0]:.0f}s vs n_jobs=1 {base:.0f}s "
        f"on one file (ratio {single16[0] / base:.2f})."
    )
text = "\n".join(lines) + "\n"
Path(summary_path).write_text(text)
print(text)
PY

echo "Finished at $(date)"
echo "Summary: $OUT/summary.txt"
echo "============================================================="
