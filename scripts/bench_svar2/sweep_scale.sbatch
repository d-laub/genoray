#!/bin/bash
#SBATCH --job-name=svar2-scale
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
# 24h was self-imposed and too tight: the partition allows 14 days, and the
# s500000 ladder alone projects to ~18h on top of ~6h for everything below it.
# A wall-clock kill mid-point is recoverable (see the resume note below) but
# costs a queue round-trip for nothing.
#SBATCH --time=72:00:00
# PIN THE NODE for any run whose rows will be compared or pooled. Measured
# spread on byte-identical work is 2.08x (151.9s on one node vs 73.2s on
# another), and unpinned rows have already produced published-then-retracted
# findings. Submit with:  sbatch --nodelist=carter-cn-04 sweep_scale.sbatch
#SBATCH --output=%x_%j.log

set -euo pipefail

# Repo root DERIVED, not hardcoded. This previously pinned a specific worktree
# (`bench-pr140-reader-workers`) that no longer exists, so resubmitting from a
# different tree silently measured the wrong checkout -- or died on the first
# command. Slurm's default cwd is the SUBMISSION directory, and every
# `python -m scripts.bench_svar2.<module>` below needs cwd = repo root to
# resolve as a package.
WT=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
cd "$WT"

# NOT $CLAUDE_JOB_DIR/tmp. The harness symlinks that path to node-local disk on
# the SUBMITTING node, and `sbatch --export=ALL` carries the variable to
# whatever node runs the job, where it is a DANGLING symlink: mkdir of a child
# raises FileNotFoundError and the parents=True retry then raises FileExistsError
# on the symlink itself, naming neither the symlink nor the node. That killed
# job 13351680 after 6:57:53 with all corpora already built. `probe.py` reads
# the variable directly, so it must be unset here, not merely avoided.
unset CLAUDE_JOB_DIR
SCRATCH="/local/$USER/scale-sweep"
mkdir -p "$SCRATCH"
JD="$SCRATCH"
PX="pixi run --manifest-path $WT/pixi.toml"

# Which sweep families this submission measures. Defaults to the historical list
# so existing studies are unaffected. Override to measure a subset, e.g. the
# RAM-law refit (issue #158), which needs only its own family:
#   sbatch --export=ALL,SWEEP_FAMILIES=vcf_ram --nodelist=<node> sweep_scale.sbatch
# Corpus generation below is gated on the same variable: generating the scale
# ladder's corpora costs hours and is pure waste for a run that will not sweep
# them.
SWEEP_FAMILIES="${SWEEP_FAMILIES:-vlinear vlinear2 contig concurrency holdout scale}"

# True when $1 appears in $SWEEP_FAMILIES.
wants() { case " $SWEEP_FAMILIES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

CELLS=1400000000

# `-@ THREADS` is the conversion's TOTAL thread budget and reader workers are
# carved out of it (probe.py `_build_cmd` passes `--threads`, `_build_env` sets
# GENORAY_READER_WORKERS). So threads and cpus-per-task must move together:
# hardcoding 48 here while lowering `--cpus-per-task` would oversubscribe the
# allocation, and the reader knee -- which is precisely the reader-vs-executor
# balance -- would be measured against a pool the job does not own. Deriving it
# from the actual affinity mask makes `--cpus-per-task` the single knob.
#
# NOTE: changing --cpus-per-task changes the REGIME, not just the cost. At 48
# threads the top of the ladder (KNEE_WORKERS max = 11) leaves 37 threads for
# the executor; at 16 it leaves 5, which starves the consumer and moves the
# knee. Results measured at different widths answer different questions and
# must not be pooled (see regression.py, which keys baselines by worker count).
#
# --mem=64G is measured, not guessed: peak MaxRSS across the ladder is ~3 GB
# (RAM grows ~215 MB/worker), and the deliberate scale probes bound ADDRESS
# SPACE via RLIMIT_AS=60 GB, which BitGrid3's calloc never makes resident.
THREADS=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))')

echo "=== node $(hostname) nproc=$(nproc) allocated=$THREADS ==="

# Rebuild the Rust extension FIRST. `pixi run` does not rebuild it, so without
# this the sweep silently measures whatever .so happens to be installed --
# including one predating the PendingGauge counting change, whose
# `pending_highwater` means something different. Twenty-four hours of data
# that answers the question about the wrong binary is worse than no data, and
# `set -e` makes a build failure abort in the first minute rather than at the
# end. CARGO_TARGET_DIR must live off NFS or the linker bus-errors.
export CARGO_TARGET_DIR="$JD/cargo-target"
export PATH="$HOME/.cargo/bin:$PATH"
$PX maturin develop --release
echo "--- built $(find "$WT/python/genoray" -name '*.so' -printf '%p %TY-%Tm-%Td %TH:%TM\n') ---"

mkdir -p "$JD/corpora" "$JD/plans" "$JD/out"

# The legacy scale/contig/holdout/vlinear/vlinear2/concurrency corpora are
# expensive to generate (hours) and only feed those six families -- skip the
# whole section when none of them is selected (e.g. a vcf_ram-only run).
if wants scale || wants contig || wants holdout || wants vlinear || wants vlinear2 || wants concurrency; then

# --- corpora -----------------------------------------------------------------
for S in 250 1000 4000 16000 64000 250000 500000; do
  OUT="$JD/corpora/s${S}.vcf.gz"
  [ -f "${OUT%.vcf.gz}.manifest.json" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$S" --cells-budget "$CELLS" --procs 16 --bgzip-threads 8
done

for C in 2 8 22; do
  OUT="$JD/corpora/s4000_c${C}.vcf.gz"
  [ -f "${OUT%.vcf.gz}.manifest.json" ] && continue
  CONTIGS=$($PX python -c "print(','.join(f'chr{i}' for i in range(1, $C + 1)))")
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples 4000 --cells-budget "$CELLS" --contigs "$CONTIGS" \
    --procs 16 --bgzip-threads 8
done
cp "$JD/corpora/s4000.manifest.json" "$JD/corpora/s4000_c1.manifest.json"

# Hold-out: off-grid in S, unfitted in F, 2x the fitted cell count. Shape
# comes from plans/build_plans.py's HOLDOUT dict (the single source of
# truth) rather than being repeated here, so this corpus cannot silently
# drift out of sync with the plan point built against it.
read -r HOLDOUT_S HOLDOUT_V HOLDOUT_F < <($PX python -c "
from scripts.bench_svar2.plans.build_plans import HOLDOUT
print(HOLDOUT['samples'], HOLDOUT['variants'], ','.join(HOLDOUT['format_fields']))
")
[ -f "$JD/corpora/holdout.manifest.json" ] || $PX python -m scripts.bench_svar2.scale_corpus \
  --out "$JD/corpora/holdout.vcf.gz" --samples "$HOLDOUT_S" --variants "$HOLDOUT_V" \
  --format-fields "$HOLDOUT_F" --procs 16 --bgzip-threads 8

# V-linearity ladder (C2): fixed small S, V varies alone. Shape comes from
# plans/build_plans.py's VLINEAR_SAMPLES/VLINEAR_VARIANTS -- same
# single-source-of-truth reasoning as the hold-out above. This is the ladder
# the spec calls "the only reason 500,000 samples is reachable": every other
# scale point extrapolates variant count off the line these four points fit.
read -r VLIN_S < <($PX python -c "
from scripts.bench_svar2.plans.build_plans import VLINEAR_SAMPLES
print(VLINEAR_SAMPLES)
")
for V in $($PX python -c "
from scripts.bench_svar2.plans.build_plans import VLINEAR_VARIANTS
print(' '.join(str(v) for v in VLINEAR_VARIANTS))
"); do
  OUT="$JD/corpora/vlinear_v${V}.vcf.gz"
  [ -f "${OUT%.vcf.gz}.manifest.json" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$VLIN_S" --variants "$V" --procs 16 --bgzip-threads 8
done

# Second V-linearity ladder (cohort-beta fix, commit 407a56c): same shape as
# above but at a DIFFERENT fixed S (VLINEAR2_SAMPLES), so `fit_cohort_beta_
# from_ladders` has two cohort sizes to form a ratio from instead of the
# design pinning beta ~ 1. This block was missing entirely until now, which
# left `vlinear2.json`/`.ndjson` permanently empty even though `model.py`
# already knew how to consume them -- the plan existed, nothing ever filled
# it.
read -r VLIN2_S < <($PX python -c "
from scripts.bench_svar2.plans.build_plans import VLINEAR2_SAMPLES
print(VLINEAR2_SAMPLES)
")
for V in $($PX python -c "
from scripts.bench_svar2.plans.build_plans import VLINEAR2_VARIANTS
print(' '.join(str(v) for v in VLINEAR2_VARIANTS))
"); do
  OUT="$JD/corpora/vlinear2_v${V}.vcf.gz"
  [ -f "${OUT%.vcf.gz}.manifest.json" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$VLIN2_S" --variants "$V" --procs 16 --bgzip-threads 8
done

fi

if wants vcf_ram; then

# --- VCF RAM-law corpora (issue #158) ----------------------------------------
echo "=== disk headroom before RAM-law corpora ==="
df -h "/local/$USER"
time $PX python -c "
from pathlib import Path
from scripts.bench_svar2.vcf_corpus import VcfCorpusSpec, generate
from scripts.bench_svar2.plans.build_plans import (
    CELLS_BUDGET, VCF_BIGCHUNK, VCF_CONTIGS, VCF_CROSSED, VCF_NCHUNKS,
)
out = Path('$SCRATCH/corpora')
shapes = {(s, CELLS_BUDGET // s) for s in VCF_CROSSED}
shapes |= {(s, v) for s, vs in VCF_NCHUNKS.items() for v in vs}
shapes.add((VCF_BIGCHUNK['samples'], VCF_BIGCHUNK['variants']))
for s, v in sorted(shapes):
    m = generate(VcfCorpusSpec(s, v, VCF_CONTIGS, seed=42), out)
    print(m.path, m.compressed_bytes, m.generator_cli_version, flush=True)
"
echo "=== disk headroom after RAM-law corpora ==="
df -h "/local/$USER"
du -sh "$SCRATCH/corpora"

fi

# --- plans and sweeps --------------------------------------------------------
$PX python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$JD/corpora" --out-dir "$JD/plans" --threads "$THREADS"

# Order is CHEAPEST-AND-MOST-LOAD-BEARING FIRST, deliberately.
#
# `scale`'s s500000 ladder dominates the whole sweep: measured phase-1 time at
# w=1 and fixed cell count runs 41s (S=4k), 44s (16k), 113s (64k), 1134s
# (250k) -- an accelerating curve whose log-log slope has already reached 1.69,
# so the seven remaining 500k points are ~18h of the budget on their own.
#
# Running that first is what nearly cost this study its answer. `vlinear` is
# ~10 minutes and fits the V-law; the V-law is the ONLY reason 500,000 samples
# is reachable at all, so a timeout inside `scale` would have left the model
# unable to produce ANY verdict despite ~24h of measurement. `vlinear2` (the
# second V-ladder, needed so `fit_cohort_beta_from_ladders` has two cohort
# sizes to form a ratio from) is the same shape and cost class as `vlinear`,
# so it sits right next to it. `holdout` (the out-of-sample check), `contig`,
# and `concurrency` (the cc={1,8,15,22} probe of the tuned load-balancing
# planner's chosen contig concurrency, reusing `contig`'s own s4000_c22
# corpus -- no separate corpus-generation step needed) are likewise hours
# cheaper than `scale`.
#
# Safe to reorder because `run_sweep` resumes: `pending_points` skips any
# point already fsynced to the results ndjson, so a resubmit continues rather
# than repeats. That also means point_id must stay stable -- it hashes every
# SweepPoint field including `threads` and `reps`, so changing
# --cpus-per-task invalidates resume as well as the regime.
for NAME in $SWEEP_FAMILIES; do
  $PX python -m scripts.bench_svar2.sweep \
    --plan "$JD/plans/$NAME.json" \
    --results "$JD/out/$NAME.ndjson" \
    --outdir "$JD/out/$NAME"

  # Row count vs plan count. The 2026-08-07 contamination was invisible in every
  # summary its run printed -- exit 0, all points ok, matching row count -- because
  # stale rows counted as recorded. What did NOT match was how many points the run
  # actually MEASURED. Fail loudly here, not in a fit weeks later.
  $PX python - "$JD/plans/$NAME.json" "$JD/out/$NAME.ndjson" <<'PYEOF'
import json, sys
plan, results = sys.argv[1], sys.argv[2]
n_plan = len(json.load(open(plan)))
rows = [json.loads(line) for line in open(results) if line.strip()]
ids = {r["point_id"] for r in rows}
print(f"plan points: {n_plan}   result rows: {len(rows)}   unique ids: {len(ids)}")
bad = [r["point_id"] for r in rows if not r.get("ok") and r.get("oom_at_rss_mb") is None]
if bad:
    # A point that failed for a reason other than a deliberate OOM probe is
    # not a measurement. This used to be a print-and-continue, which is why a
    # run where ALL 48 rows failed on the same "symbolic ALT out of scope"
    # error (job 13355460) still printed "OK: every planned point measured"
    # and exited 0 -- the guard checked row COUNT, never row VALIDITY. An OOM
    # row (`oom_at_rss_mb` set) is excluded above and must keep passing.
    sys.exit(
        f"ABORT: {len(bad)} of {len(rows)} point(s) failed (not OOM "
        f"deliverables): {bad[:5]}"
    )
if len(ids) != n_plan:
    sys.exit(f"ABORT: {len(ids)} distinct points in results, plan has {n_plan}")
nodes = {r.get("node") for r in rows}
if len(nodes) > 1:
    sys.exit(f"ABORT: rows span multiple nodes {nodes} -- not comparable")
missing = [r["point_id"] for r in rows if r.get("concurrent_chroms_used") is None]
if missing:
    # Was a WARNING while src/logging.rs stripped the field from every
    # event (issue #162). Now that the planner's choice actually reaches
    # the row, an unobserved cc is a defect: `_ram_rows` drops such rows,
    # so a law would be fitted on a subset excluding the planner-chosen
    # configurations.
    sys.exit(
        f"ABORT: {len(missing)} of {len(rows)} row(s) never reported a "
        f"realised concurrent_chroms: {missing[:5]}"
    )
print(f"OK: every planned point measured, single node {nodes}")
PYEOF
done

echo "=== ALL DONE ==="
