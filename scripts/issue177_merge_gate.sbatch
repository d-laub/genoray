#!/bin/bash
#SBATCH --job-name=genoray-177-merge-gate
#SBATCH --partition=carter-compute
#SBATCH --nodes=1
#SBATCH --exclude=carter-cn-03
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
unset CLAUDE_JOB_DIR
unset RUSTC_WRAPPER

WORKTREE=/carter/users/dlaub/projects/genoray/.claude/worktrees/issue-177-by-scan-blocked
cd "$WORKTREE"
export CARGO_TARGET_DIR=/local/$USER/genoray-177-target
mkdir -p "$CARGO_TARGET_DIR"
echo "=== node: $(hostname) cores: $(nproc) ==="

echo "=== cargo fmt --check ==="
pixi run -- cargo fmt --check 2>&1

echo "=== cargo clippy --all-targets ==="
pixi run -- cargo clippy --no-default-features --features conversion --all-targets -- -D warnings 2>&1 | tail -8

echo "=== cargo check --no-default-features (query-only core) ==="
pixi run -- cargo check --no-default-features 2>&1 | tail -3

echo "=== cargo check --all-targets (lint env; what the prek hook runs) ==="
pixi run -e lint cargo check --all-targets 2>&1 | tail -3

# KEEP `conversion` on: without it the executor, orchestrator and rvk's
# emission loops are not compiled and the run vacuously passes a smaller suite.
echo "=== cargo test --no-default-features --features conversion ==="
pixi run -- cargo test --no-default-features --features conversion 2>&1 | tail -40

echo "=== ALL GATES PASSED ==="
