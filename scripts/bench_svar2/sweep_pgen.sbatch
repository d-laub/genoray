#!/bin/bash
#SBATCH --job-name=svar2-pgen
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --nodelist=carter-cn-04
# 24h (the brief's literal value) was measured, before submitting, to be at
# real risk of insufficiency: three direct probe generations on this cluster
# gave a corpus-generation rate of ~6.15e-7 s/cell, essentially SINGLE-CORE
# (`vcfixture bulk`'s own --threads defaults to "all available cores" but
# user+sys time tracked real time within ~4% across all three probes, so the
# 48-cpu allocation buys generation nothing). The seven PGEN_LADDERS corpora
# (including the S=128,000 rung added for the RAM-law refit, 128,000 x
# 250,000 = 3.2e10 cells ~= 5.5h on its own) sum to 9.5e10 cells, i.e. ~16.2h
# for generation ALONE before any conversion runs -- and that is the exact
# failure mode `sweep_scale.sbatch` already hit once (24h -> 72h, see its own
# header comment) for a comparable VCF corpus ladder. Matching that precedent
# rather than re-discovering it the expensive way; see task-9-report.md
# Phase A for the full arithmetic. The chunk-size axis added alongside the
# new rung (`PGEN_CHUNK_SIZE_AXIS`) reuses existing corpora and adds no
# generation cost.
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# --nodelist is pinned, not optional: node speed on this cluster varies by
# 2.08x, so an unpinned sweep cannot support any cross-point comparison.
WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/svar2-pgen-budget-planner
# Slurm's default cwd is the SUBMISSION directory, so every
# `python -m scripts.bench_svar2.<mod>` below needs this to resolve as a package.
cd "$WT"
PX="pixi run --manifest-path $WT/pixi.toml"

echo "=== commit $(git rev-parse HEAD) ==="

# NOT $CLAUDE_JOB_DIR/tmp: /tmp inside a Slurm job is a PRIVATE slurmtmpfs bind
# mount, so a job writing there dies with exit 53 and no log. /local/$USER is
# node-local disk, visible from the compute node and not cleaned between jobs.
#
# Unset the var, do not merely avoid it here: `--export=ALL` carries it in, and
# modules we invoke (probe.py) read it directly and would silently override this
# choice with a path that is node-local to the SUBMITTING host. Stating the
# policy in this script is not enough to enforce it.
unset CLAUDE_JOB_DIR
SCRATCH="/local/$USER/pgen-sweep"
mkdir -p "$SCRATCH"/{corpora,plans,out}
echo "=== disk headroom on /local/$USER ==="
df -h "/local/$USER"

export VCFIXTURE_BIN="${VCFIXTURE_BIN:?set VCFIXTURE_BIN to the vcfixture bulk binary}"
echo "=== VCFIXTURE_BIN=$VCFIXTURE_BIN plink2=$(command -v plink2 || echo MISSING) ==="

# Threads must track --cpus-per-task: hardcoding 48 while lowering the
# allocation would oversubscribe it and measure against a pool the job does
# not own. Note that changing --cpus-per-task changes the REGIME, not just the
# cost -- results at different widths must not be pooled.
THREADS=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))')
echo "=== node $(hostname) nproc=$(nproc) allocated=$THREADS ==="

# Rebuild the extension FIRST. `pixi run` does NOT rebuild it, so without this
# the sweep silently measures whatever .so is installed -- possibly one
# predating the planner change entirely. `set -e` makes a build failure abort
# in the first minute rather than after hours of useless data.
export CARGO_TARGET_DIR="$SCRATCH/cargo-target"
export PATH="$HOME/.cargo/bin:$PATH"
$PX maturin develop --release
echo "--- built $(find "$WT/python/genoray" -name '*.so' -printf '%p %TY-%Tm-%Td %TH:%TM\n') ---"

# --- corpora (generation is compute-heavy; it belongs on the allocation) ------
# Cached on the full spec + GENERATOR_VERSION (see pgen_corpus.generate), so a
# resubmit after a wall-clock kill skips every corpus already on disk instead
# of regenerating it. Timed per-corpus so a hung/slow generation is visible in
# the log rather than discovered only at the 72h wall.
echo "=== disk headroom before corpus generation ==="
df -h "/local/$USER"
time $PX python -c "
from pathlib import Path
from scripts.bench_svar2.pgen_corpus import PgenCorpusSpec, generate
from scripts.bench_svar2.plans.build_plans import PGEN_LADDERS
contigs = tuple(f'chr{i}' for i in range(1, 23))
out = Path('$SCRATCH/corpora')
for s, vs in PGEN_LADDERS:
    for v in vs:
        m = generate(PgenCorpusSpec(s, v, contigs, seed=42), out)
        print(m.path, m.compressed_bytes, flush=True)
"
echo "=== disk headroom after corpus generation ==="
df -h "/local/$USER"
du -sh "$SCRATCH/corpora"

# --- plans and sweep ---------------------------------------------------------
$PX python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$SCRATCH/corpora" --out-dir "$SCRATCH/plans" --threads "$THREADS"

$PX python -m scripts.bench_svar2.sweep \
  --plan "$SCRATCH/plans/pgen.json" \
  --results "$SCRATCH/out/pgen.ndjson" \
  --outdir "$SCRATCH/out/pgen"

echo "=== ALL DONE (node $(hostname)) ==="
