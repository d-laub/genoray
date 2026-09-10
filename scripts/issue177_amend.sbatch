#!/bin/bash
#SBATCH --job-name=genoray-177-amend
#SBATCH --partition=carter-compute
#SBATCH --nodes=1
#SBATCH --exclude=carter-cn-03
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
unset CLAUDE_JOB_DIR
unset RUSTC_WRAPPER

cd /carter/users/dlaub/projects/genoray/.claude/worktrees/issue-177-by-scan-blocked
export CARGO_TARGET_DIR=/local/$USER/genoray-177-target
mkdir -p "$CARGO_TARGET_DIR"
echo "=== node: $(hostname) ==="

git add src/rvk.rs scripts/
if git diff --cached --quiet; then
  echo "(nothing staged)"
  exit 0
fi
git commit -m "docs(rvk): replace the projected speedup with the measured one

The comment cited 63x/59x/39x as \"measured\". They were not: they came from an
operation-count model I wrote while designing the change. The A/B (job 1074195,
both arms in one allocation on cn-02, a separate target dir per arm, binary
hashes confirmed distinct) puts the real figure ~10x lower:

    S = 535,662    AF 0.01%: 6.3x   0.1%: 5.9x   1%: 3.7x
    S =  16,000    AF 0.01%: 7.4x   0.1%: 6.1x   1%: 3.0x

Output is byte-identical across every point -- the digests match pairwise.

The model counted instructions; both arms are bound by memory LATENCY. Window
reads do fall 64-fold, but consecutive variants within a block are still
\`columns\` bits apart, so each read is its own cache line: the change buys 64x
fewer misses, not 64x cheaper ones, and O(calls) emission remains underneath.

The per-column-test alternative's 45x/13x/1.5x is likewise a model, not a
measurement -- that variant was never built. Labelled as such now.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L9ceFDd39dibEKLy5aA4KV"

git log --oneline -3
git push
echo "=== AMEND DONE ==="
