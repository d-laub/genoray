# SVAR2 `from_vcf` Livelock — Repro & Diagnose (Phases 0–1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private-data-free repro of the issue #135 `from_vcf` concurrent-chromosome livelock, instrument the pipeline to localize the stall, and produce a written root-cause finding — the gate to re-planning the Phase 2 fix.

**Architecture:** This is a diagnostic **spike**, not a feature build. We (0) generate a synthetic multi-contig, multi-sample BCF with a `VAF` FORMAT field via the `vcfixture-rs 0.3.0 bulk` CLI; (1) reproduce the livelock deterministically on this box by driving `SparseVar2.from_vcf` with `threads` high enough to force ≥2 concurrent chromosomes; (2) add env-gated heartbeat instrumentation to the sharded reader (`shard_exec.rs`) and the encode→write handoff (`executor.rs`), and fix the misleading `read=0%` telemetry; (3) run the instrumented repro and write the root cause. Phase 2 (the fix) and Phase 3 (throughput) are **out of scope for this plan** and re-planned after the finding.

**Tech Stack:** Rust (genoray_core, pyo3/maturin), Python 3.10 (pixi `default` env), `vcfixture` 0.3.0 CLI (crates.io), `bcftools` 1.23 (pixi), `pytest`.

## Global Constraints

- **`available_cores` == the `threads` argument** (`src/lib.rs:183`), not physical cores. `concurrent_chroms = min((threads-1)/6, n_chroms)` (`src/budget.rs:31,47-48`). To force the issue's exact regime (`5 concurrent, htslib=2, processing_threads=1`) use **≥5 contigs + `threads=32`**. This box is 8-core; the 30 logical pipeline threads oversubscribe, which is fine (and may aggravate the livelock).
- **Fixtures + results live in `$CLAUDE_JOB_DIR/tmp`** — `/tmp` is reaped mid-session and would silently delete a cohort.
- **`CARGO_TARGET_DIR=/tmp/genoray-target-svar2`** for every `cargo`/`maturin` invocation — the worktree is on NFS and an NFS `target/` bus-errors `cargo test` (debug) and the lint hooks. (Build artifacts are transient, so `/tmp` reaping is harmless for them; only *data* goes in `$CLAUDE_JOB_DIR/tmp`.)
- **`cargo test` / `cargo check` MUST pass `--no-default-features`** — otherwise the pyo3 test binary fails to link (`undefined symbol: _Py_Dealloc`).
- **`maturin develop --release` before any Python-level repro run** — `pixi run test`/`pytest` does *not* rebuild the Rust `.so`; a stale `.so` silently runs old code. Release `.so` ≈ 4 MB, debug ≈ 79 MB (sanity-check the size).
- **All commands run in the pixi `default` env:** `pixi run bash -lc '...'`.
- **Do not commit generated BCF/PGEN fixtures** (large, regenerable) — commit the *generator*, not its output. Add `scripts/from_vcf_livelock/*.bcf` etc. to `.gitignore` if generated in-tree; prefer `$CLAUDE_JOB_DIR/tmp` output paths.
- **No public API change is expected.** If any name reachable from `import genoray` without underscores changes, `skills/genoray-api/SKILL.md` must be updated in the same commit (repo policy). This plan should not trigger it.

---

### Task 1: Install `vcfixture bulk` and confirm it can emit a `VAF Number=A Float` FORMAT field

**Files:**
- Create: `scripts/from_vcf_livelock/README.md`

**Interfaces:**
- Produces: a validated `vcfixture bulk` invocation (documented in the README) that Task 2 wraps; the name of the profile/preset used and whether a post-processing VAF-injection fallback is needed.

- [ ] **Step 1: Install the CLI off-NFS**

```bash
pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-svar2 cargo install vcfixture --version 0.3.0 --features cli --root /tmp/vcfixture-cli --locked'
export PATH=/tmp/vcfixture-cli/bin:$PATH
/tmp/vcfixture-cli/bin/vcfixture --version
```
Expected: prints `vcfixture 0.3.0` (or compatible).

- [ ] **Step 2: Inspect the bulk subcommand and available profiles/presets**

```bash
/tmp/vcfixture-cli/bin/vcfixture bulk --help
```
Read the fitted-vs-dialed split and payload presets. Record in the README: which profile is somatic-like (or whether only `germline-1kgp` exists) and how to declare a `VAF` FORMAT field (`Number=A, Type=Float`) — via a preset, a dialed field flag, or not at all.

- [ ] **Step 3: Generate a tiny probe BCF (2 contigs, few samples) and inspect it**

```bash
mkdir -p "$CLAUDE_JOB_DIR/tmp/probe"
/tmp/vcfixture-cli/bin/vcfixture bulk --samples 50 --contigs chr1,chr2 \
    --target-size 5MB --seed 0 -o "$CLAUDE_JOB_DIR/tmp/probe/probe.bcf"   # adjust flags per --help
pixi run bash -lc 'bcftools view -h "$CLAUDE_JOB_DIR/tmp/probe/probe.bcf" | grep -E "FORMAT|contig"'
pixi run bash -lc 'bcftools index -s "$CLAUDE_JOB_DIR/tmp/probe/probe.bcf"'   # lists per-contig record counts
```
Expected: header shows ≥2 `##contig` lines and a `##FORMAT=<ID=GT...`. Check whether a `VAF` (`Number=A,Type=Float`) FORMAT line is present.

- [ ] **Step 4: If `VAF` is not natively emittable, establish the fallback and document it**

If no somatic/`VAF` preset exists, the fallback is to inject a synthetic `VAF` per ALT after generation:
```bash
# Sketch: add a Number=A Float VAF drawn per-record; verify it round-trips.
pixi run bash -lc 'bcftools view "$CLAUDE_JOB_DIR/tmp/probe/probe.bcf" | head'
```
Document the exact chosen mechanism (native preset vs. post-process) in `scripts/from_vcf_livelock/README.md`, including the final command that yields a BCF whose header carries `##FORMAT=<ID=VAF,Number=A,Type=Float,...>`.

- [ ] **Step 5: Verify the fixture path end-to-end and commit the README**

```bash
pixi run bash -lc 'bcftools view -h "$CLAUDE_JOB_DIR/tmp/probe/probe.bcf" | grep -E "ID=VAF.*Number=A.*Type=Float"'
```
Expected: one matching line (from native emission or the fallback).

```bash
git add scripts/from_vcf_livelock/README.md
git commit -m "docs(svar2): document vcfixture bulk repro-fixture generation for #135"
```

---

### Task 2: Repro cohort generator script

**Files:**
- Create: `scripts/from_vcf_livelock/generate_repro.py`
- Modify: `scripts/from_vcf_livelock/README.md` (add the generator usage)

**Interfaces:**
- Consumes: the validated `vcfixture bulk` command from Task 1.
- Produces: `generate_repro.py --out <dir> --samples N --contigs C --target-size S --seed K` → writes `<dir>/cohort.bcf` + `<dir>/cohort.bcf.csi` with ≥5 contigs and a `VAF Number=A Float` FORMAT field; prints the output path. Task 3 consumes `cohort.bcf`.

- [ ] **Step 1: Write the failing test**

`tests/test_from_vcf_livelock_fixture.py`:
```python
import subprocess
import os
from pathlib import Path

def test_generate_repro_emits_multicontig_vaf_bcf(tmp_path: Path):
    out = tmp_path / "cohort"
    subprocess.run(
        ["python", "scripts/from_vcf_livelock/generate_repro.py",
         "--out", str(out), "--samples", "40",
         "--contigs", "chr1,chr2,chr3,chr4,chr5,chr6",
         "--target-size", "8MB", "--seed", "0"],
        check=True, env={**os.environ},
    )
    bcf = out / "cohort.bcf"
    assert bcf.exists() and (out / "cohort.bcf.csi").exists()
    hdr = subprocess.run(["bcftools", "view", "-h", str(bcf)],
                         capture_output=True, text=True, check=True).stdout
    assert hdr.count("##contig=") >= 5
    assert "ID=VAF" in hdr and "Number=A" in hdr and "Type=Float" in hdr
    contigs = subprocess.run(["bcftools", "index", "-s", str(bcf)],
                             capture_output=True, text=True, check=True).stdout
    assert len([l for l in contigs.splitlines() if l.strip()]) >= 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run pytest tests/test_from_vcf_livelock_fixture.py -v
```
Expected: FAIL (`generate_repro.py` does not exist).

- [ ] **Step 3: Write `generate_repro.py`**

Wrap the Task-1 command. Resolve the `vcfixture` binary from `PATH` or `/tmp/vcfixture-cli/bin`. If Task 1 chose the post-process fallback, apply it here so the emitted `cohort.bcf` always carries `VAF`. Ensure `bcftools index -c` (CSI) is written. Use `argparse` with `--out/--samples/--contigs/--target-size/--seed`. Print the final BCF path.

- [ ] **Step 4: Run test to verify it passes**

```bash
pixi run pytest tests/test_from_vcf_livelock_fixture.py -v
```
Expected: PASS.

- [ ] **Step 5: Generate the real repro cohort (persisted, not committed)**

```bash
pixi run bash -lc 'python scripts/from_vcf_livelock/generate_repro.py \
    --out "$CLAUDE_JOB_DIR/tmp/repro" --samples 2000 \
    --contigs chr1,chr2,chr3,chr4,chr5,chr6 --target-size 500MB --seed 0'
pixi run bash -lc 'bcftools index -s "$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf"'
```
Expected: ≥5 contigs, ~2000 samples. (Scale `--samples`/`--target-size` up only if Task 3 does not reproduce the livelock.)

- [ ] **Step 6: Commit**

```bash
git add scripts/from_vcf_livelock/generate_repro.py scripts/from_vcf_livelock/README.md tests/test_from_vcf_livelock_fixture.py
git commit -m "feat(svar2): synthetic multi-contig VAF cohort generator for the #135 repro"
```

---

### Task 3: Reproduce the livelock (control passes, multi-concurrent stalls)

**Files:**
- Create: `scripts/from_vcf_livelock/repro.py`
- Create: `tests/test_from_vcf_livelock.py`

**Interfaces:**
- Consumes: `cohort.bcf` from Task 2 (`$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf`).
- Produces: `repro.py --bcf <path> --out <dir> --threads T --timeout S` → runs `SparseVar2.from_vcf` under a watchdog, prints the `Pipeline Config` line and whether any chunk committed before timeout; exit 0 if it completed, 124 if it timed out (livelock). Task 4/5 reuse this.

- [ ] **Step 1: Build the current `.so`**

```bash
pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-svar2 maturin develop --release'
pixi run bash -lc 'ls -la python/genoray/_core*.so'
```
Expected: a ~4 MB release `.so` (not ~79 MB debug).

- [ ] **Step 2: Write `repro.py`**

Runs the conversion in a child process with a wall-clock timeout, capturing stdout (telemetry). Control call and failing call differ only in `threads`:
```python
# scripts/from_vcf_livelock/repro.py (essential shape)
import argparse, subprocess, sys, textwrap
def run(bcf, out, threads, timeout):
    code = textwrap.dedent(f"""
        from genoray import SparseVar2, FormatField
        SparseVar2.from_vcf({out!r}, {bcf!r}, no_reference=True,
            format_fields=[FormatField("VAF")], threads={threads},
            chunk_size=5000, overwrite=True)
    """)
    try:
        subprocess.run([sys.executable, "-c", code], timeout=timeout, check=True)
        return 0
    except subprocess.TimeoutExpired:
        return 124
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bcf", required=True); p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, required=True); p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args(); sys.exit(run(a.bcf, a.out, a.threads, a.timeout))
```

- [ ] **Step 3: Reproduce — control completes, multi-concurrent stalls**

```bash
# Control: threads=6 -> (6-1)/6 = 0 -> low-end -> concurrent_chroms=1. Must COMPLETE.
pixi run bash -lc 'python scripts/from_vcf_livelock/repro.py \
    --bcf "$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf" --out "$CLAUDE_JOB_DIR/tmp/out_ctrl" \
    --threads 6 --timeout 600; echo "control exit=$?"'
# Failing: threads=32 -> concurrent_chroms=5, processing_threads=1 (issue regime). Expect TIMEOUT.
pixi run bash -lc 'python scripts/from_vcf_livelock/repro.py \
    --bcf "$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf" --out "$CLAUDE_JOB_DIR/tmp/out_ll" \
    --threads 32 --timeout 300; echo "multi exit=$?"'
```
Expected: `control exit=0`; `multi exit=124`. Confirm the failing run printed `Pipeline Config: 5 concurrent chromosomes ... 1 reader-side processing/shard threads` and that `tx_dense/tx_sparse/tx_long` stayed at `0/...` with `cw=0% lw=0%`.

**If `multi` does NOT time out:** scale up `--samples`/`--target-size` in Task 2 Step 5 and/or lengthen per-chunk work (`chunk_size` is already 5000) until it reproduces, and record the smallest reproducing size in the README. Also capture the `Pipeline Config` line for the size that does reproduce.

- [ ] **Step 4: Write the regression test documenting both regimes**

`tests/test_from_vcf_livelock.py`:
```python
import os, subprocess, sys, pytest
from pathlib import Path

BCF = Path(os.environ.get("REPRO_BCF", "")) # set to $CLAUDE_JOB_DIR/tmp/repro/cohort.bcf

def _repro(threads, out, timeout):
    return subprocess.run([sys.executable, "scripts/from_vcf_livelock/repro.py",
        "--bcf", str(BCF), "--out", str(out), "--threads", str(threads),
        "--timeout", str(timeout)]).returncode

@pytest.mark.skipif(not BCF.exists(), reason="set REPRO_BCF to a generated cohort.bcf")
def test_single_concurrent_completes(tmp_path):
    assert _repro(6, tmp_path / "ctrl", 600) == 0

@pytest.mark.skipif(not BCF.exists(), reason="set REPRO_BCF to a generated cohort.bcf")
@pytest.mark.xfail(reason="#135 livelock: >=2 concurrent chromosomes never commit a chunk", strict=True)
def test_multi_concurrent_completes(tmp_path):
    # Flips from xfail to pass once the livelock is fixed (Phase 2).
    assert _repro(32, tmp_path / "multi", 300) == 0
```

- [ ] **Step 5: Run the tests**

```bash
pixi run bash -lc 'REPRO_BCF="$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf" pytest tests/test_from_vcf_livelock.py -v'
```
Expected: `test_single_concurrent_completes` PASS; `test_multi_concurrent_completes` XFAIL (strict). The livelock is now captured as an executable regression.

- [ ] **Step 6: Commit**

```bash
git add scripts/from_vcf_livelock/repro.py tests/test_from_vcf_livelock.py
git commit -m "test(svar2): reproduce the #135 from_vcf concurrent-chromosome livelock"
```

---

### Task 4: Env-gated heartbeat instrumentation + fix the misleading `read=0%` telemetry

**Files:**
- Modify: `src/shard_exec.rs` (near `:217` read_next_chunk return; `:288`/`:305` forward-to-`tx_dense`)
- Modify: `src/executor.rs` (near `:34` `dense2sparse_vk` enter/exit; `:47` `tx_sparse.send`)
- Modify: `src/orchestrator.rs` (log `workers` and `shards.len()` alongside the existing config line)
- Modify: `src/monitor.rs` (sample `shard-worker-*` threads, not the parked `read-*` thread)
- Test: `cargo test --no-default-features` + a manual instrumented repro run

**Interfaces:**
- Consumes: repro from Task 3.
- Produces: when `GENORAY_TRACE=1` is set, stderr heartbeats: `[trace {chrom}] reader: chunk N assembled (shard s)`, `[trace {chrom}] exec: dense2sparse enter/exit chunk N`, `[trace {chrom}] exec: tx_sparse.send chunk N`, plus a one-time `[plan {chrom}] workers=W shards=S`. Task 5 reads these.

- [ ] **Step 1: Add a tiny trace helper (gated, zero-cost when off)**

In a shared module (e.g. top of `src/shard_exec.rs` or a new `src/trace.rs` if cleaner), add:
```rust
#[inline]
pub(crate) fn traced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GENORAY_TRACE").is_some())
}
macro_rules! trace_ll { ($($a:tt)*) => { if crate::shard_exec::traced() { eprintln!($($a)*); } } }
```
(Adjust module path so `executor.rs`/`orchestrator.rs` can call it.)

- [ ] **Step 2: Instrument the reader assembly and forward points (`shard_exec.rs`)**

At the `read_next_chunk` return (~`:217`) emit `trace_ll!("[trace {chrom}] reader: shard {i} assembled chunk rows={n}")`; at the two `tx_dense` forward sites (~`:288`, `:305`) emit `trace_ll!("[trace {chrom}] reader: forwarded ordinal {ord} to tx_dense")`. Pass `chrom` through (already available in `process_chromosome`; thread it into `shard_exec::run` if not present).

- [ ] **Step 3: Instrument the encode→write handoff (`executor.rs`)**

Around `:34` bracket the `dense2sparse_vk` call with enter/exit traces including the chunk index; after `:47` `tx_sparse.send`, emit `trace_ll!("[trace {chrom}] exec: sent SparseChunk {k}")`. This is the single sharpest seam: it distinguishes "reader never delivered chunk 0" from "exec stuck in dense2sparse" from "exec sent but cw never received".

- [ ] **Step 4: Log the concurrency regime and de-lie the telemetry**

In `orchestrator.rs` where `shards`/`workers` are known, print once: `[plan {chrom}] workers={workers} shards={shards}`. In `monitor.rs` (`:102-105`), additionally resolve and aggregate CPU for threads whose `comm` starts with `shard-worker-` under the owning chrom, and label it `read*` (or add a `shard` column) so a busy reader no longer shows `read=0%`.

- [ ] **Step 5: Build and unit-check**

```bash
pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-svar2 cargo test --no-default-features -q 2>&1 | tail -20'
pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-svar2 cargo clippy --no-default-features 2>&1 | tail -5'
```
Expected: tests pass; no new clippy errors. (Trace is behind `GENORAY_TRACE`, so default behavior is unchanged.)

- [ ] **Step 6: Rebuild the `.so` and smoke-test the trace on the control run**

```bash
pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-svar2 maturin develop --release'
pixi run bash -lc 'GENORAY_TRACE=1 python scripts/from_vcf_livelock/repro.py \
    --bcf "$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf" --out "$CLAUDE_JOB_DIR/tmp/out_trace_ctrl" \
    --threads 6 --timeout 600 2>&1 | grep -E "\[trace|\[plan" | head'
```
Expected (control, 1 concurrent): reader/exec/tx_sparse heartbeats advance and the run completes — proves the trace fires on the working path.

- [ ] **Step 7: Commit**

```bash
git add src/shard_exec.rs src/executor.rs src/orchestrator.rs src/monitor.rs
git commit -m "feat(svar2): GENORAY_TRACE heartbeats + shard-worker CPU sampling for #135 diagnosis"
```

---

### Task 5: Capture evidence in both regimes and write the root-cause finding

**Files:**
- Create: `docs/superpowers/specs/2026-07-20-svar2-from-vcf-livelock-diagnosis.md`

**Interfaces:**
- Consumes: instrumented `.so` (Task 4) + repro (Task 3).
- Produces: the written root cause + evidence — the gate to re-plan Phase 2. No code.

> **Note on execution:** this task is analysis/synthesis, not mechanical implementation — interpreting the heartbeat evidence to localize the stall. Per project convention it should be done by the orchestrating (Opus-tier) session or a review-level agent, not a weak implementer.

- [ ] **Step 1: Capture the issue-faithful regime (5 concurrent, processing_threads=1)**

```bash
pixi run bash -lc 'GENORAY_TRACE=1 python scripts/from_vcf_livelock/repro.py \
    --bcf "$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf" --out "$CLAUDE_JOB_DIR/tmp/out_ll5" \
    --threads 32 --timeout 300 2>&1 | tee "$CLAUDE_JOB_DIR/tmp/trace_5concurrent.log"'
```
Read the tail: does any lane emit `reader: forwarded ordinal 0`? Does `exec: dense2sparse enter` for chunk 0 ever appear? Does `exec: sent SparseChunk 0` ever appear? Note where the last heartbeat per lane stops.

- [ ] **Step 2: Capture the `workers≥2` regime (few contigs, exercises ReorderBuffer HOL)**

```bash
# 2 contigs + threads=32 -> concurrent=2, processing_threads=7, so workers=7 -> HOL path is live.
pixi run bash -lc 'python scripts/from_vcf_livelock/generate_repro.py \
    --out "$CLAUDE_JOB_DIR/tmp/repro2" --samples 2000 --contigs chr1,chr2 --target-size 300MB --seed 0'
pixi run bash -lc 'GENORAY_TRACE=1 python scripts/from_vcf_livelock/repro.py \
    --bcf "$CLAUDE_JOB_DIR/tmp/repro2/cohort.bcf" --out "$CLAUDE_JOB_DIR/tmp/out_ll2" \
    --threads 32 --timeout 300 2>&1 | tee "$CLAUDE_JOB_DIR/tmp/trace_2concurrent.log"'
```
Compare: does the stall signature differ between `processing_threads=1` (Step 1) and `processing_threads=7` (this step)? If the `workers=1` regime stalls but the reader *does* assemble chunks, hypothesis (a) reader-contention / oversubscription is favored over (b) ReorderBuffer HOL.

- [ ] **Step 3: Confirm or refute reader-contention with a single-reader cross-check**

```bash
# Force the NON-sharded reader by converting one explicit sub-region (single range => 1 shard),
# still with threads=32 (5 concurrent chroms) if >=5 regions across contigs are given.
# If a single-shard-per-contig run makes progress while the 4-shard tiling does not,
# the stall is in the sharded reader / concurrent IndexedReader contention.
pixi run bash -lc 'GENORAY_TRACE=1 python - <<PY 2>&1 | tail -30
from genoray import SparseVar2, FormatField
SparseVar2.from_vcf("$CLAUDE_JOB_DIR/tmp/out_region", "$CLAUDE_JOB_DIR/tmp/repro/cohort.bcf",
    no_reference=True, format_fields=[FormatField("VAF")], threads=32, chunk_size=5000,
    regions=["chr1:1-1000000","chr2:1-1000000","chr3:1-1000000","chr4:1-1000000","chr5:1-1000000"],
    overwrite=True)
PY'
```
Expected: informative either way — completes (→ sharded/whole-contig path is implicated) or stalls (→ contention is independent of sharding, points at concurrent readers / htslib threads / rayon-join).

- [ ] **Step 4: Write the diagnosis document**

`docs/superpowers/specs/2026-07-20-svar2-from-vcf-livelock-diagnosis.md` containing: the exact repro command + cohort dimensions + smallest reproducing size; the `Pipeline Config`/`[plan ...]` lines for each regime; per-lane last-heartbeat evidence from Steps 1–3; the **root cause** (which stage stalls and the mechanism); and which of hypotheses (a) reader contention, (b) ReorderBuffer HOL, (c) thread oversubscription — or a newly identified mechanism — the evidence supports. End with a **recommended Phase 2 fix direction** and the byte-identical-output acceptance test the fix must satisfy (differential vs the `threads=6` control output).

- [ ] **Step 5: Commit and STOP for re-plan**

```bash
git add docs/superpowers/specs/2026-07-20-svar2-from-vcf-livelock-diagnosis.md
git commit -m "docs(svar2): root-cause diagnosis of the #135 from_vcf livelock"
```
This is the Phase 1 gate. **Do not begin Phase 2** — return the finding so the fix is planned against evidence.

---

## Self-Review

**Spec coverage:** Phase 0 (repro harness) → Tasks 1–2. Phase 1 (instrument & diagnose) → Tasks 3–5. `vcfixture bulk` VAF open item → Task 1 Step 4 (explicit fallback). Misleading `read=0%` telemetry → Task 4 Step 4. Issue-faithful regime (5 concurrent, processing=1) → Task 3 Step 3 / Task 5 Step 1. `workers≥2` HOL regime → Task 5 Step 2. Byte-identical acceptance criterion → Task 5 Step 4. Build/measure discipline → Global Constraints, applied per task. Phases 2–3 are intentionally deferred (re-planned after Task 5), consistent with the approved spec.

**Placeholder scan:** No `TBD`/`TODO`. Task 1 flags configurable `vcfixture bulk` flags ("adjust per `--help`") because the exact flag names are unknowable until the CLI is inspected — this is an inspection step with a concrete verification (header grep), not a hand-wave. Task 3 Step 3 has an explicit "if it does not reproduce, scale up" branch with a concrete knob.

**Type/name consistency:** `generate_repro.py` args (`--out/--samples/--contigs/--target-size/--seed`) are consistent across Tasks 1–3 and 5. `repro.py` contract (exit 0 = completed, 124 = timeout) is consistent across Tasks 3–5. `GENORAY_TRACE` env gate and heartbeat prefixes (`[trace ...]`, `[plan ...]`) are consistent across Tasks 4–5. Output paths all under `$CLAUDE_JOB_DIR/tmp`.
