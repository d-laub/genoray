#!/usr/bin/env bash
# scripts/profile/cargo_asm.sh <function-path>
# Inspect hot inner-loop codegen (inlining/vectorization). Requires cargo-show-asm.
set -euo pipefail
export CARGO_TARGET_DIR=/tmp/genoray-target-$$
cargo asm --no-default-features --features conversion "$1"
