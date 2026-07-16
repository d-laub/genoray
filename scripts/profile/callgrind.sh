#!/usr/bin/env bash
# scripts/profile/callgrind.sh <src> <out-dir> <ref>
# Deterministic instruction-count / cache A/B on a SMALL input (serialized).
set -euo pipefail
src=$1 out=$2 ref=$3
valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
  pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" 1
callgrind_annotate callgrind.out | head -60
