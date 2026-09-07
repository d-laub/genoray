#!/bin/bash
#SBATCH --job-name=genoray-frontier
# REQUIRED: this cluster declares no default partition, so a job without
# `--partition` never starts -- it sits PENDING with reason
# `(PartitionConfig)` indefinitely rather than failing at submit time.
#SBATCH --partition=carter-compute
#SBATCH --nodes=1
#SBATCH --exclusive
# 48h, not 8h. Corpus generation dominates this job and runs at TWO
# workers, not 32: `scale_corpus.plan_blocks` caps `procs` so at most
# POOL_MEMORY_BUDGET (32 GB) of blocks are live, and at 500,000 samples a
# 2,000-variant GT-only block is ~12 GB, so 32 -> 2. Verified by calling
# plan_blocks(per_contig=200_000, n_contigs=1, n_samples=500_000,
# n_format=0, procs=32) -> (2000, 2). Generation is also encode-bound
# (~120% CPU despite rayon), so 8h was not a safe budget. The partition
# allows 14 days; an over-long limit costs nothing but queue priority.
#SBATCH --time=2-00:00:00
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
  # NOTE: `--procs 32` is a request, not a guarantee. `plan_blocks` caps it
  # to 2 at this shape (see the --time comment above). Left at 32 so the
  # intent survives if that budget is ever raised.
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
