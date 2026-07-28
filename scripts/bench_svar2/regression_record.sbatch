#!/bin/bash
#SBATCH --job-name=svar2-regression-record
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

# Re-record the fast-tier baselines on a DEDICATED allocation.
#
# Record baselines through this script, not by running
# `pixi run bench-regression-record` on a login/head node. The tier's corpus is
# ~140 KB, so its wall time is almost entirely interpreter startup plus
# whatever else is on the box: the same 3 points measured 68/54/119 s on a
# contended head node and, on a *different* day at higher load, 198/250/302 s
# for a corpus 10x SMALLER. Baselines recorded that way encode the load
# average, not the code. maxrss_mb is far steadier, but it is the metric that
# actually gates, so it is the one worth recording somewhere reproducible.
#
# --cpus-per-task=8 is deliberate and load-bearing: `regression.py` derives
# `threads` from `os.sched_getaffinity(0)`, so the allocation width becomes a
# recorded parameter of the baseline. 8 matches the affinity a dev box here
# gets, which keeps a locally-run `pixi run bench-regression` comparable to
# the committed numbers.

set -euo pipefail

WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/bench-pr140-reader-workers
# Slurm's cwd is the submission directory, not this script's location, and
# `python -m scripts.bench_svar2.<module>` needs cwd = repo root to resolve.
cd "$WT"
JD="${CLAUDE_JOB_DIR:?set CLAUDE_JOB_DIR}"/tmp
PX="pixi run --manifest-path $WT/pixi.toml"

# A fresh workdir so the recorded numbers can never come from a corpus left
# behind by an earlier CORPUS (`corpus_is_current` guards this too; this is
# belt-and-braces, and it makes the cold-start timing below honest).
WORKDIR="$JD/bench_reg_sbatch_${SLURM_JOB_ID:-manual}"
rm -rf "$WORKDIR"

echo "=== node $(hostname) nproc=$(nproc) allocated=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))') ==="
echo "=== load $(cat /proc/loadavg) ==="

# Cold: includes corpus generation. This is what a first run costs.
echo "=== RECORD (cold, includes corpus generation) ==="
time $PX python -m scripts.bench_svar2.regression --record --workdir "$WORKDIR"

# Warm: corpus reused, which is the steady-state `pixi run bench-regression`.
echo "=== CHECK (warm, corpus reused) ==="
time $PX python -m scripts.bench_svar2.regression --workdir "$WORKDIR"

echo "=== recorded baselines ==="
cat "$WT/scripts/bench_svar2/baselines/regression.json"
echo "=== ALL DONE ==="
