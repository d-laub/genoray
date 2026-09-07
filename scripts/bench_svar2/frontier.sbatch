#!/bin/bash
#SBATCH --job-name=genoray-frontier
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# MUST be first: $CLAUDE_JOB_DIR points at node-local scratch on the
# SUBMITTING node and dangles here, and helpers read the variable themselves.
unset CLAUDE_JOB_DIR

SCRATCH=/local/$USER/genoray-frontier
mkdir -p "$SCRATCH"
# Per-ARM target dir. A SHARED CARGO_TARGET_DIR can hand a second source tree
# the first tree's artifact byte-for-byte, which silently turns an A/B into
# an A/A.
export CARGO_TARGET_DIR=/local/$USER/genoray-target-${ARM:?set ARM=main or ARM=branch}
echo "node=$(hostname) arm=$ARM scratch=$SCRATCH"

CORPUS="$SCRATCH/wide.vcf.gz"
if [ ! -f "$CORPUS" ]; then
  pixi run python -m scripts.bench_svar2.scale_corpus \
    --out "$CORPUS" \
    --samples 500000 \
    --variants 200000 \
    --contigs chr21 \
    --procs 32 \
    --bgzip-threads 8
fi

pixi run python -m scripts.bench_svar2.frontier_points \
  --manifest "$SCRATCH/wide.manifest.json" \
  --out "$SCRATCH/plan.json"

# ${ARM:?} forces the caller to name the arm: sweep.py resumes on
# (point_id, code_id), so two arms sharing a results file is CORRECT and
# informative, but two arms sharing an --outdir would race on bench.svar.
pixi run python -m scripts.bench_svar2.sweep \
  --plan "$SCRATCH/plan.json" \
  --results "$SCRATCH/frontier.ndjson" \
  --outdir "$SCRATCH/out-${ARM:?set ARM=main or ARM=branch}"
