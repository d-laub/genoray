#!/bin/bash
#SBATCH --job-name=svar2-scale
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/bench-pr140-reader-workers
JD="${CLAUDE_JOB_DIR:?set CLAUDE_JOB_DIR}"/tmp
PX="pixi run --manifest-path $WT/pixi.toml"
CELLS=1400000000

echo "=== node $(hostname) nproc=$(nproc) allocated=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))') ==="

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

# Hold-out: off-grid in S, unfitted in F, 2x the fitted cell count.
[ -f "$JD/corpora/holdout.vcf.gz" ] || $PX python -m scripts.bench_svar2.scale_corpus \
  --out "$JD/corpora/holdout.vcf.gz" --samples 100000 --variants 28000 \
  --format-fields DP,GQ,AD --procs 16 --bgzip-threads 8

# --- plans and sweeps --------------------------------------------------------
$PX python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$JD/corpora" --out-dir "$JD/plans" --threads 48

for NAME in scale contig holdout; do
  $PX python -m scripts.bench_svar2.sweep \
    --plan "$JD/plans/$NAME.json" \
    --results "$JD/out/$NAME.ndjson" \
    --outdir "$JD/out/$NAME"
done

echo "=== ALL DONE ==="
