#!/bin/bash
#SBATCH --job-name=svar2-scale
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/bench-pr140-reader-workers
# Slurm's default cwd is the *submission* directory, not this script's location,
# but every `python -m scripts.bench_svar2.<module>` below needs cwd = repo root
# to resolve as a package. Without this, resubmitting from anywhere but $WT dies
# on the first command with ModuleNotFoundError.
cd "$WT"
JD="${CLAUDE_JOB_DIR:?set CLAUDE_JOB_DIR}"/tmp
PX="pixi run --manifest-path $WT/pixi.toml"
CELLS=1400000000

echo "=== node $(hostname) nproc=$(nproc) allocated=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))') ==="

# Rebuild the Rust extension FIRST. `pixi run` does not rebuild it, so without
# this the sweep silently measures whatever .so happens to be installed --
# including one predating the PendingGauge counting change, whose
# `pending_highwater` means something different. Twenty-four hours of data
# that answers the question about the wrong binary is worse than no data, and
# `set -e` makes a build failure abort in the first minute rather than at the
# end. CARGO_TARGET_DIR must live off NFS or the linker bus-errors.
export CARGO_TARGET_DIR="$JD/cargo-target"
export PATH="$HOME/.cargo/bin:$PATH"
$PX maturin develop --release
echo "--- built $(find "$WT/python/genoray" -name '*.so' -printf '%p %TY-%Tm-%Td %TH:%TM\n') ---"

mkdir -p "$JD/corpora" "$JD/plans" "$JD/out"

# --- corpora -----------------------------------------------------------------
for S in 250 1000 4000 16000 64000 250000 500000; do
  OUT="$JD/corpora/s${S}.vcf.gz"
  [ -f "$OUT" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$S" --cells-budget "$CELLS" --procs 16 --bgzip-threads 8
done

for C in 2 8 22; do
  OUT="$JD/corpora/s4000_c${C}.vcf.gz"
  [ -f "$OUT" ] && continue
  CONTIGS=$($PX python -c "print(','.join(f'chr{i}' for i in range(1, $C + 1)))")
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples 4000 --cells-budget "$CELLS" --contigs "$CONTIGS" \
    --procs 16 --bgzip-threads 8
done
cp "$JD/corpora/s4000.manifest.json" "$JD/corpora/s4000_c1.manifest.json"

# Hold-out: off-grid in S, unfitted in F, 2x the fitted cell count. Shape
# comes from plans/build_plans.py's HOLDOUT dict (the single source of
# truth) rather than being repeated here, so this corpus cannot silently
# drift out of sync with the plan point built against it.
read -r HOLDOUT_S HOLDOUT_V HOLDOUT_F < <($PX python -c "
from scripts.bench_svar2.plans.build_plans import HOLDOUT
print(HOLDOUT['samples'], HOLDOUT['variants'], ','.join(HOLDOUT['format_fields']))
")
[ -f "$JD/corpora/holdout.vcf.gz" ] || $PX python -m scripts.bench_svar2.scale_corpus \
  --out "$JD/corpora/holdout.vcf.gz" --samples "$HOLDOUT_S" --variants "$HOLDOUT_V" \
  --format-fields "$HOLDOUT_F" --procs 16 --bgzip-threads 8

# V-linearity ladder (C2): fixed small S, V varies alone. Shape comes from
# plans/build_plans.py's VLINEAR_SAMPLES/VLINEAR_VARIANTS -- same
# single-source-of-truth reasoning as the hold-out above. This is the ladder
# the spec calls "the only reason 500,000 samples is reachable": every other
# scale point extrapolates variant count off the line these four points fit.
read -r VLIN_S < <($PX python -c "
from scripts.bench_svar2.plans.build_plans import VLINEAR_SAMPLES
print(VLINEAR_SAMPLES)
")
for V in $($PX python -c "
from scripts.bench_svar2.plans.build_plans import VLINEAR_VARIANTS
print(' '.join(str(v) for v in VLINEAR_VARIANTS))
"); do
  OUT="$JD/corpora/vlinear_v${V}.vcf.gz"
  [ -f "$OUT" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$VLIN_S" --variants "$V" --procs 16 --bgzip-threads 8
done

# --- plans and sweeps --------------------------------------------------------
$PX python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$JD/corpora" --out-dir "$JD/plans" --threads 48

for NAME in scale contig holdout vlinear; do
  $PX python -m scripts.bench_svar2.sweep \
    --plan "$JD/plans/$NAME.json" \
    --results "$JD/out/$NAME.ndjson" \
    --outdir "$JD/out/$NAME"
done

echo "=== ALL DONE ==="
