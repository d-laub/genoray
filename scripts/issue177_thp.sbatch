#!/bin/bash
#SBATCH --job-name=genoray-177-thp
#SBATCH --partition=carter-compute
#SBATCH --nodes=1
# PINNED to cn-02 -- the node the published #177 A/B ran on. The THP-ON arms
# here are a CONTROL: they must reproduce 4.36 s (base) and 0.696 s (branch) at
# S=535,662 / AF=0.01 %. On any other node (this partition varies 2.08x) that
# control is worthless and a THP effect could not be separated from a node
# effect.
#SBATCH --nodelist=carter-cn-02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# $CLAUDE_JOB_DIR points at node-local scratch on the SUBMITTING node and
# dangles here; helpers read the variable themselves, so unset it.
unset CLAUDE_JOB_DIR
# sccache is a per-node daemon shared with every other job on the node and has
# already failed one build here with "Failed to read response header".
unset RUSTC_WRAPPER

REPO=/carter/users/dlaub/projects/genoray
BRANCH_TREE="$REPO/.claude/worktrees/issue-177-by-scan-blocked"
BASE_TREE="$REPO/.claude/worktrees/issue-177-baseline"
BRANCH_TARGET=/local/$USER/genoray-177-target
BASE_TARGET=/local/$USER/genoray-177-target-base

echo "node=$(hostname) cores=$(nproc)"
echo "system THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"

NOTHP=/local/$USER/nothp
cc -O2 -o "$NOTHP" "$BRANCH_TREE/scripts/nothp.c"
echo "built $NOTHP"

build_arm() {
  local tree=$1 target=$2 label=$3
  echo "=== building $label ==="
  cd "$tree"
  CARGO_TARGET_DIR="$target" pixi run -- cargo build --release \
    --no-default-features --features conversion --bin bench_dense2sparse 2>&1 | tail -3
}
build_arm "$BASE_TREE" "$BASE_TARGET" base
build_arm "$BRANCH_TREE" "$BRANCH_TARGET" branch

BASE_BIN="$BASE_TARGET/release/bench_dense2sparse"
BRANCH_BIN="$BRANCH_TARGET/release/bench_dense2sparse"
BASE_SHA=$(sha256sum "$BASE_BIN" | cut -d' ' -f1)
BRANCH_SHA=$(sha256sum "$BRANCH_BIN" | cut -d' ' -f1)
if [ "$BASE_SHA" = "$BRANCH_SHA" ]; then
  echo "FATAL: identical binaries ($BASE_SHA) -- this is an A/A. Refusing."
  exit 1
fi
echo "binaries differ: base=$BASE_SHA branch=$BRANCH_SHA"

PERF=""
if command -v perf >/dev/null 2>&1; then
  if perf stat -e dTLB-load-misses -x, /bin/true >/dev/null 2>&1; then
    PERF="perf stat -e dTLB-load-misses,dTLB-loads,page-faults -x,"
    echo "perf available; counting dTLB misses"
  else
    echo "perf present but the dTLB event is not readable here; timings only"
  fi
else
  echo "no perf; timings only"
fi

# One rep for the THP-off base arm would still be minutes if the TLB
# hypothesis holds, so reps stay at 3 (best-of) but V stays at 2000 to match
# the published A/B exactly.
run_point() {
  local arm=$1 thp=$2 s=$3 af=$4
  local bin cmd
  case $arm in
    base)   bin="$BASE_BIN" ;;
    branch) bin="$BRANCH_BIN" ;;
  esac
  if [ "$thp" = off ]; then cmd="$NOTHP $bin"; else cmd="$bin"; fi
  echo "--- arm=$arm thp=$thp S=$s af=$af ---"
  # shellcheck disable=SC2086
  $PERF $cmd --samples "$s" --variants 2000 --af "$af" --reps 3 2>&1 \
    | sed "s/^/arm=$arm thp=$thp | /"
}

for AF in 0.0001 0.01; do
  for THP in on off; do
    for ARM in base branch; do
      run_point "$ARM" "$THP" 535662 "$AF"
    done
  done
done

echo "=== THP EXPERIMENT DONE ==="
