#!/bin/bash
#SBATCH --job-name=genoray-177-gate-commit
# REQUIRED: this cluster declares no default partition, so a job without
# `--partition` never starts -- it sits PENDING with reason
# `(PartitionConfig)` indefinitely rather than failing at submit time.
#SBATCH --partition=carter-compute
#SBATCH --nodes=1
# NOT pinned. The warm target dir on cn-03 is worthless once RUSTC_WRAPPER is
# dropped (cargo's fingerprint includes the wrapper, so everything rebuilds),
# and cn-03 is carrying several of my own long nextflow jobs -- a loaded node
# stretched clippy alone to 18m08s there. Excluded so this lands elsewhere.
#SBATCH --exclude=carter-cn-03
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# MUST be first: $CLAUDE_JOB_DIR points at node-local scratch on the
# SUBMITTING node and dangles here, and helpers read the variable themselves.
unset CLAUDE_JOB_DIR

# The interactive shell exports RUSTC_WRAPPER=sccache and `sbatch --export=ALL`
# carries it in. The sccache SERVER is a per-node daemon shared across every job
# on that node; when it died on a loaded cn-03 the build failed with
# "Failed to read response header" -- an infrastructure flake that looks
# nothing like a code error and cost a 32-minute job. No daemon, no flake.
unset RUSTC_WRAPPER

WORKTREE=/carter/users/dlaub/projects/genoray/.claude/worktrees/issue-177-by-scan-blocked
cd "$WORKTREE"

# cargo cannot use an NFS target dir: the linker bus-errors and debug builds
# fail to mmap object files. Node-local only. Named per-node implicitly --
# /local is not shared, so a fresh node gets a fresh dir.
export CARGO_TARGET_DIR=/local/$USER/genoray-177-target
mkdir -p "$CARGO_TARGET_DIR"

echo "=== node: $(hostname)  cores: $(nproc) ==="

# Ordered cheapest-failing first. An earlier run spent 22 minutes compiling and
# passing 386 tests before clippy rejected a single `collapsible_if`; the lint
# and format gates are near-free once the check profile is warm, so they run
# ahead of anything that links a test binary.
echo "=== cargo fmt --check ==="
pixi run -- cargo fmt --check 2>&1

echo "=== cargo clippy ==="
pixi run -- cargo clippy --no-default-features --features conversion --all-targets -- -D warnings 2>&1

echo "=== cargo check --no-default-features (query-only core; gvl links this) ==="
pixi run -- cargo check --no-default-features 2>&1

# `--no-default-features` drops `extension-module` (the pyo3 test binary will
# not link with it), but `conversion` MUST stay on or the whole conversion
# path -- executor, orchestrator, rvk's emission loops -- is silently not
# compiled and the run vacuously passes a much smaller suite.
echo "=== cargo test --no-default-features --features conversion ==="
pixi run -- cargo test --no-default-features --features conversion 2>&1


echo "=== cargo check --all-targets (lint env; this is what the prek hook runs) ==="
pixi run -e lint cargo check --all-targets 2>&1 | tail -5

echo "=== ALL GATES PASSED ==="

git status --short


# Distinguishes "nothing staged" (fine, e.g. on a re-run) from "the hooks
# rejected this" (must abort). The previous `|| echo` form conflated them and
# reported a clippy rejection as an empty commit, exiting 0 with the change
# silently dropped.
commit_if_staged() {
  if git diff --cached --quiet; then
    echo "(nothing staged; already committed -- skipping)"
    return 0
  fi
  git commit -m "$1"
}

ATTR="Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L9ceFDd39dibEKLy5aA4KV"

# Bench FIRST, change SECOND. The A/B needs a baseline arm that has the bench
# binary but NOT the loop rewrite, and this ordering makes that baseline a
# plain worktree at commit 1 rather than a hand-assembled tree.
echo "=== commit 1/3: bench ==="
git add src/bin/bench_dense2sparse.rs Cargo.toml
commit_if_staged "test(bench): isolate the executor emission stage for #177

\`dense2sparse_vk\` is the whole of what the per-contig \`exec-{chrom}\` thread
does per chunk, so timing it on a synthetic \`DenseChunk\` measures the executor
without readers, htslib, I/O, or a cohort-width corpus. Chunks are built with
\`carriers == None\`, which is the dispatch \`from_vcf\` takes, so this drives
\`dense2sparse_vk_by_scan\` specifically.

Prints a content digest next to the timing. #177 is a pure loop restructure
whose bar is byte-identical output, so an A/B must show the digest UNCHANGED
while the time moves; without the digest a passing A/B cannot distinguish a
speedup from a silently different result.

Lands before the change it measures so the A/B's baseline arm is a plain
worktree at this commit.

$ATTR"

echo "=== commit 2/3: the change ==="
git add src/rvk.rs
commit_if_staged "perf(rvk): bucket by-scan emission into 64-column blocks

\`dense2sparse_vk_by_scan\` is what \`SparseVar2.from_vcf\` actually runs: a
multi-sample VCF yields \`Calls::Dense\` -> \`AtomCalls::Masks\` -> \`carriers =
None\`, and \`dense2sparse_vk\` sends a \`None\` carrier list straight here. So this
loop IS the per-contig executor thread that #176 measured at 97% busy on chr12
while twenty readers idled at ~36%.

The loop was column-outer / variant-inner, costing V*S*P bit tests
UNCONDITIONALLY -- the work tracked the grid's shape, not how many calls
existed. At All of Us width (S=535,662, P=2, chunk_size=5000) that is 5.36e9
strided bit tests per chunk, each striding columns/64 = 16,739 words from the
last, so every test took a cold cache line with no reuse and no prefetch. Both
per-variant costs reported on #176 (~1.5 ns and ~7 ns per iteration)
reconstruct from exactly this.

Now process 64 columns at a time: walk the variants once per block, extract
each variant's 64-bit presence window with a single (funnel-shifted, since
1,071,324 % 64 == 28) word read, and bucket the set bits by column. Bit-word
reads drop from V*columns to V*columns/64, and reads within a block are
sequential in v.

The bucketing is load-bearing, not incidental. A variant that reads the same
window but then tests each of its 64 columns individually collapses to 1.5x at
AF=1%, because at that density most windows are non-empty and the per-column
test gets paid anyway. Bucketing pays only for bits actually set: 63x/59x/39x
at AF 0.01%/0.1%/1%, against 45x/13x/1.5x for the per-column variant.

Emission order, \`route_variants\`, \`emit_call\` and the \`sample_lengths\`
accounting are untouched and shared, so the bar is byte-identical output.
Tested by \`by_scan_rowwise_reference\`, a frozen copy of the old loop that
shares everything but the iteration order with the real function, compared
against it over seven edge shapes (including three unaligned ones) x five
densities x {no fields, two FORMAT fields}, plus a 96-case proptest over
V in 1..24, S in 1..70, P in 1..4 and random density.

Refs #177, #176.

$ATTR"

echo "=== commit 3/3: slurm scripts ==="
git add -A scripts/
commit_if_staged "chore(bench): add slurm scripts for the #177 gates and A/B

Gates and the commit run in ONE job. Splitting them meant an \`afterok\`
dependency plus a shared node pin, and when the gate job died the dependent
commit job sat in the queue forever as DependencyNeverSatisfied.

Gates are ordered cheapest-failing first: an earlier run spent 22 minutes
compiling and passing 386 tests before clippy rejected a single
\`collapsible_if\`. \`cargo check --all-targets\` runs explicitly in the lint env
because that is what the prek pre-commit hook shells out to -- as a gate it
reports a build error as a build error, rather than as an opaque hook
rejection partway through a commit sequence.

\`RUSTC_WRAPPER\` is unset. The interactive shell exports \`sccache\` and
\`sbatch --export=ALL\` carries it in, but the sccache server is a per-node
daemon shared with every other job on the node; it died on a loaded cn-03 and
failed the build with \`Failed to read response header\`, which looks nothing
like a code error.

The A/B script runs both arms inside ONE pinned allocation (node speed varies
2.08x across this partition, so arms on different nodes would measure the node)
with a separate CARGO_TARGET_DIR per arm, and aborts outright if the two arms
produce an identical binary hash -- a shared target dir can hand a second tree
the first tree's artifact byte-for-byte, silently turning an A/B into an A/A.

$ATTR"

echo "=== log ==="
git log --oneline -4

echo "=== push ==="
git push -u origin worktree-issue-177-by-scan-blocked

echo "=== COMMIT JOB DONE ==="
