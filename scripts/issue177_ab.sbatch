#!/bin/bash
#SBATCH --job-name=genoray-177-ab
# REQUIRED: this cluster declares no default partition, so a job without
# `--partition` never starts -- it sits PENDING with reason
# `(PartitionConfig)` indefinitely rather than failing at submit time.
#SBATCH --partition=carter-compute
#SBATCH --nodes=1
# Pinned. Node speed varies 2.08x across this partition (cn-03 151.9 s vs
# cn-04 73.2 s on one job), so an A/B whose arms could land on different
# nodes measures the node, not the change. Both arms run in THIS job.
#SBATCH --nodelist=carter-cn-03
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# MUST be first: $CLAUDE_JOB_DIR points at node-local scratch on the
# SUBMITTING node and dangles here, and helpers read the variable themselves.
unset CLAUDE_JOB_DIR

REPO=/carter/users/dlaub/projects/genoray
BRANCH_TREE="$REPO/.claude/worktrees/issue-177-by-scan-blocked"
BASE_TREE="$REPO/.claude/worktrees/issue-177-baseline"

# One target dir PER ARM. A SHARED CARGO_TARGET_DIR can hand a second source
# tree the first tree's artifact byte-for-byte, silently turning an A/B into
# an A/A. The sha comparison below is the check that this actually held.
BRANCH_TARGET=/local/$USER/genoray-177-target
BASE_TARGET=/local/$USER/genoray-177-target-base

echo "node=$(hostname) cores=$(nproc)"

build_arm() {
  local tree=$1 target=$2 label=$3
  echo "=== building $label: $tree -> $target ==="
  cd "$tree"
  CARGO_TARGET_DIR="$target" pixi run -- cargo build --release \
    --no-default-features --features conversion --bin bench_dense2sparse 2>&1 | tail -5
  sha256sum "$target/release/bench_dense2sparse"
}

build_arm "$BASE_TREE" "$BASE_TARGET" base
build_arm "$BRANCH_TREE" "$BRANCH_TARGET" branch

BASE_SHA=$(sha256sum "$BASE_TARGET/release/bench_dense2sparse" | cut -d' ' -f1)
BRANCH_SHA=$(sha256sum "$BRANCH_TARGET/release/bench_dense2sparse" | cut -d' ' -f1)
if [ "$BASE_SHA" = "$BRANCH_SHA" ]; then
  echo "FATAL: the two arms produced an IDENTICAL binary ($BASE_SHA)."
  echo "This is an A/A, not an A/B. Refusing to report numbers."
  exit 1
fi
echo "binaries differ: base=$BASE_SHA branch=$BRANCH_SHA"

# The old loop's cost is V*S*P bit tests regardless of density, so the base arm
# should come out flat across --af while the branch arm scales with rarity.
# That flatness is itself a check that the arms are what they claim to be.
echo "=== points ==="
for S in 16000 535662; do
  for AF in 0.0001 0.001 0.01; do
    for ARM in base branch; do
      case $ARM in
        base)   BIN="$BASE_TARGET/release/bench_dense2sparse" ;;
        branch) BIN="$BRANCH_TARGET/release/bench_dense2sparse" ;;
      esac
      echo -n "arm=$ARM "
      "$BIN" --samples "$S" --variants 2000 --af "$AF" --reps 3 2>/dev/null
    done
  done
done

echo "=== AB DONE ==="
