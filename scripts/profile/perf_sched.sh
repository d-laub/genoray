#!/usr/bin/env bash
# scripts/profile/perf_sched.sh <src> <out-dir> <ref> <threads>
# Off-CPU / scheduling: exposes collector serialization + channel stalls.
set -euo pipefail
src=$1 out=$2 ref=$3 threads=$4
perf sched record -o perf.sched.data \
  -- pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" "$threads"
perf sched latency -i perf.sched.data | head -40
