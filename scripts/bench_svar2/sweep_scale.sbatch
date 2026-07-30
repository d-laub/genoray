#!/bin/bash
#SBATCH --job-name=svar2-scale
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
# 24h was self-imposed and too tight: the partition allows 14 days, and the
# s500000 ladder alone projects to ~18h on top of ~6h for everything below it.
# A wall-clock kill mid-point is recoverable (see the resume note below) but
# costs a queue round-trip for nothing.
#SBATCH --time=72:00:00
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

# `-@ THREADS` is the conversion's TOTAL thread budget and reader workers are
# carved out of it (probe.py `_build_cmd` passes `--threads`, `_build_env` sets
# GENORAY_READER_WORKERS). So threads and cpus-per-task must move together:
# hardcoding 48 here while lowering `--cpus-per-task` would oversubscribe the
# allocation, and the reader knee -- which is precisely the reader-vs-executor
# balance -- would be measured against a pool the job does not own. Deriving it
# from the actual affinity mask makes `--cpus-per-task` the single knob.
#
# NOTE: changing --cpus-per-task changes the REGIME, not just the cost. At 48
# threads the top of the ladder (KNEE_WORKERS max = 11) leaves 37 threads for
# the executor; at 16 it leaves 5, which starves the consumer and moves the
# knee. Results measured at different widths answer different questions and
# must not be pooled (see regression.py, which keys baselines by worker count).
#
# --mem=64G is measured, not guessed: peak MaxRSS across the ladder is ~3 GB
# (RAM grows ~215 MB/worker), and the deliberate scale probes bound ADDRESS
# SPACE via RLIMIT_AS=60 GB, which BitGrid3's calloc never makes resident.
THREADS=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))')

echo "=== node $(hostname) nproc=$(nproc) allocated=$THREADS ==="

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
  [ -f "${OUT%.vcf.gz}.manifest.json" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$S" --cells-budget "$CELLS" --procs 16 --bgzip-threads 8
done

for C in 2 8 22; do
  OUT="$JD/corpora/s4000_c${C}.vcf.gz"
  [ -f "${OUT%.vcf.gz}.manifest.json" ] && continue
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
[ -f "$JD/corpora/holdout.manifest.json" ] || $PX python -m scripts.bench_svar2.scale_corpus \
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
  [ -f "${OUT%.vcf.gz}.manifest.json" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$VLIN_S" --variants "$V" --procs 16 --bgzip-threads 8
done

# --- plans and sweeps --------------------------------------------------------
$PX python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$JD/corpora" --out-dir "$JD/plans" --threads "$THREADS"

# Order is CHEAPEST-AND-MOST-LOAD-BEARING FIRST, deliberately.
#
# `scale`'s s500000 ladder dominates the whole sweep: measured phase-1 time at
# w=1 and fixed cell count runs 41s (S=4k), 44s (16k), 113s (64k), 1134s
# (250k) -- an accelerating curve whose log-log slope has already reached 1.69,
# so the seven remaining 500k points are ~18h of the budget on their own.
#
# Running that first is what nearly cost this study its answer. `vlinear` is
# ~10 minutes and fits the V-law; the V-law is the ONLY reason 500,000 samples
# is reachable at all, so a timeout inside `scale` would have left the model
# unable to produce ANY verdict despite ~24h of measurement. `holdout` (the
# out-of-sample check) and `contig` are likewise hours cheaper than `scale`.
#
# Safe to reorder because `run_sweep` resumes: `pending_points` skips any
# point already fsynced to the results ndjson, so a resubmit continues rather
# than repeats. That also means point_id must stay stable -- it hashes every
# SweepPoint field including `threads` and `reps`, so changing
# --cpus-per-task invalidates resume as well as the regime.
for NAME in vlinear contig holdout scale; do
  $PX python -m scripts.bench_svar2.sweep \
    --plan "$JD/plans/$NAME.json" \
    --results "$JD/out/$NAME.ndjson" \
    --outdir "$JD/out/$NAME"
done

echo "=== ALL DONE ==="
