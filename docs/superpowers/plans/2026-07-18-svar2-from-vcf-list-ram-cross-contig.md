# `from_vcf_list` peak-RAM (#120): cross-contig ratchet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `SparseVar2.from_vcf_list` peak RSS below 64 GB for the 7,089-sample somatic WGS cohort by eliminating the cross-contig memory ratchet, keeping the output store byte-identical.

**Architecture:** This is a **measurement-first** optimization (performant-py-rust). Phase A builds a trustworthy harness (fast single-sample cohort generator + per-contig-high-water instrumentation). Phase B measures the current baseline and **localizes** the cross-contig retention (memray/heaptrack) — that measurement *decides* the fix. Phase C applies the localized lever(s) — most likely `malloc_trim` / arena-cap at the contig boundary, since the contig loop already drops all Rust state per iteration — re-measuring after each change and keeping only wins that stay byte-identical.

**Tech Stack:** Rust (pyo3 extension, `src/orchestrator.rs`, `libc`), Python 3.10+ (`genoray/_svar2.py`, bench scripts), pixi, maturin, bcftools 1.22, memray, `/usr/bin/time -v`.

## Global Constraints

- **Output must stay byte-identical** to the current `from_vcf_list` store on every change — verified by the existing small-data differential oracle (`from_vcf` vs `from_vcf_list`, genotypes + DP/VAF exact). This is the correctness gate for every Phase-C task.
- **Target: peak RSS < 64 GB at N=7,089** (24 contigs, F≈7). Success KPI = per-contig MaxRSS high-water mark **flat across contigs** (no ratchet).
- **Do not regress wall-clock** materially (CPU is already linear, PR #121). Measure wall cost of any allocator change; keep only if net-acceptable.
- **Bulk cohort is speed/memory ONLY** — no VcfBuilder, no GroundTruth. Correctness lives on the small-data oracle.
- **Never guess the fix before Phase B localizes it.** A blank measurement slot means stop and measure.
- **Bench/build hygiene:** export a local `CARGO_TARGET_DIR` (NFS `target/` bus-errors on debug/dhat mmap); `maturin develop --release` before any Python-level perf check (`pixi run test` does NOT rebuild the `.so`); park cohorts/results in `$CLAUDE_JOB_DIR/tmp` (`/tmp` is reaped); Rust tests run under `cargo test --no-default-features` and filter by `--test <file>`, not bare name.
- **Public-API rule:** if any public name reachable from `import genoray` (without underscores) changes, update `skills/genoray-api/SKILL.md` in the same PR. (`max_mem` already exists and is documented — only touch SKILL.md if you add/rename a kwarg.)
- **Commits:** Conventional Commits; never edit `CHANGELOG.md` (commitizen owns it); never bump the version by hand.

## File Structure

- `scripts/bench_from_vcf_list/generate_cohort.py` — cohort generator; **modify** to parallelize the per-file `bgzip`+`index`.
- `scripts/bench_from_vcf_list/test_generate_cohort.py` — generator tests; **add** a serial-vs-parallel identity test.
- `scripts/bench_from_vcf_list/run_bench.py` — bench driver; **modify** to capture the per-contig high-water mark + arena-heap count and sweep N.
- `scripts/bench_from_vcf_list/test_run_bench.py` — **create**; unit-test the new log parsers on captured sample output (no subprocess).
- `src/orchestrator.rs` — `run_vcf_list` contig loop (the `==> Processing {chrom}` / `Cohort Processing Complete.` boundary at lines ~1044-1069); **modify** to release memory at the contig boundary.
- `Cargo.toml` — **modify** to add `libc` (for `malloc_trim`).
- `docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md` — **create**; the measured baseline + localization report (the artifact Phase B produces and Phase C is judged against).
- `genoray/_svar2.py` — read-only reference for lever-4 verification (no code change expected).

## Parallelization note (per user workflow rules)

- **Tasks 1 and 2 are independent** (generator vs. bench driver, different files) → dispatch in parallel via `dispatching-parallel-agents`, Sonnet-or-weaker implementers.
- **Tasks 3 → 4 → 5 → 6 → 7 → 8 are strictly sequential** — each is gated on the previous measurement. Do NOT parallelize them; a fix dispatched before its localization is a methodology violation.

---

## Task 1: Parallelize the cohort generator's per-file bgzip+index

**Files:**
- Modify: `scripts/bench_from_vcf_list/generate_cohort.py` (the per-file write/bgzip/index loop, currently lines ~152-183)
- Test: `scripts/bench_from_vcf_list/test_generate_cohort.py`

**Interfaces:**
- Consumes: nothing (leaf task).
- Produces: `generate_cohort(out_dir, n_files, n_variants, *, contigs=None, contig_len=None, shared_frac=0.1, indel_frac=0.1, seed=0, format_fields=(), jobs=None) -> Path` — **same signature plus one new optional kwarg `jobs: int | None`** (max parallel worker processes; `None` → `os.cpu_count()`). Output files, manifest, and byte content are **unchanged** from the serial version.

**Why:** Generating 7,089 WGS-scale single-sample VCFs is dominated by one `bgzip` + one `bcftools index` subprocess *per file, serially*. Parallelizing across cores is the whole "bulk generation" ask; nothing about variant content changes.

- [ ] **Step 1: Write the failing test** (serial and parallel produce identical bytes + manifest)

Add to `scripts/bench_from_vcf_list/test_generate_cohort.py`:

```python
import gzip
from generate_cohort import generate_cohort


def _decompressed_records(manifest: Path) -> list[bytes]:
    """Raw VCF record bytes (post-bgzip, order-preserving) for every file in a manifest."""
    out = []
    for gz in Path(manifest).read_text().split():
        with gzip.open(gz, "rb") as fh:
            out.append(b"".join(l for l in fh if not l.startswith(b"##fileformat")))
    return out


def test_parallel_generation_is_byte_identical_to_serial(tmp_path: Path) -> None:
    kw = dict(
        n_files=12, n_variants=15, contigs=["1", "2"], contig_len=200_000,
        shared_frac=0.1, indel_frac=0.1, seed=0, format_fields=["VAF", "DP"],
    )
    m1 = generate_cohort(tmp_path / "serial", jobs=1, **kw)
    m8 = generate_cohort(tmp_path / "par", jobs=8, **kw)
    names1 = [Path(p).name for p in m1.read_text().split()]
    names8 = [Path(p).name for p in m8.read_text().split()]
    assert names1 == names8, "manifest order/names diverged under parallelism"
    assert _decompressed_records(m1) == _decompressed_records(m8), "record bytes diverged"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo
cd scripts/bench_from_vcf_list
pixi run pytest test_generate_cohort.py::test_parallel_generation_is_byte_identical_to_serial -v
```
Expected: FAIL — `generate_cohort() got an unexpected keyword argument 'jobs'`.

- [ ] **Step 3: Implement parallel bgzip+index**

Refactor so per-file *content* is built exactly as today (deterministic, seed-driven), but the write→`bgzip`→`bcftools index`→`unlink(plain)` step for each file runs in a `concurrent.futures.ProcessPoolExecutor(max_workers=jobs or os.cpu_count())`. Each worker receives `(i, out_dir, sample, header, body_lines)` (all plain data — no rng objects across the process boundary) and returns the `.vcf.gz` path. Collect returned paths, **re-sort into original `i` order**, then write the manifest in that order so output is deterministic regardless of completion order. Keep `jobs=1` as an in-process fast path (no pool) so the existing tests and small callers are unaffected. Add `jobs: int | None = None` to the signature and the argparse (`--jobs`, default `None`).

- [ ] **Step 4: Run tests to verify they pass** (new test + all existing generator tests)

```bash
cd scripts/bench_from_vcf_list
pixi run pytest test_generate_cohort.py -v
```
Expected: PASS — including `test_union_grows_linearly_with_n`, `test_generated_union_matches_estimate_and_grows_linearly`, `test_format_fields_are_emitted`, `test_multiple_contigs_present`, and the new identity test.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_from_vcf_list/generate_cohort.py scripts/bench_from_vcf_list/test_generate_cohort.py
git commit -m "perf(bench): parallelize from_vcf_list cohort generation across cores"
```

---

## Task 2: Capture per-contig high-water mark + arena stats in the bench driver

**Files:**
- Modify: `scripts/bench_from_vcf_list/run_bench.py`
- Test: `scripts/bench_from_vcf_list/test_run_bench.py` (create)

**Interfaces:**
- Consumes: nothing (leaf task; operates on captured process output).
- Produces: two pure parsers importable by tests and used by `main`:
  - `parse_per_contig_highwater(stderr: str, rss_samples: list[tuple[float, int]]) -> dict[str, int]` — map each contig label (from the `==> Processing {chrom}` banner lines the pipeline prints) to the MaxRSS-in-KB high-water mark observed up to the *next* contig banner.
  - `parse_arena_heaps(status_or_smaps: str) -> int` — count of glibc 64 MB arena heaps (from a captured `/proc/self/smaps`-derived summary), returning `0` if absent.
  - New CSV columns appended to the existing row: `per_contig_highwater_json` (JSON string of the dict) and `arena_heaps`.

**Why:** The ratchet — MaxRSS rising per contig — is the KPI. Total MaxRSS alone can't show whether a fix flattened the per-contig curve. The driver must sample RSS over time and bucket it by the contig banner the pipeline already prints.

- [ ] **Step 1: Write the failing tests** (pure parsers on captured text — no subprocess)

Create `scripts/bench_from_vcf_list/test_run_bench.py`:

```python
from run_bench import parse_per_contig_highwater, parse_arena_heaps

# Two contig banners; rss_samples are (elapsed_s, rss_kb) sampled during the run.
_STDERR = "==> Processing 1\n(work)\n==> Processing 2\n(work)\nCohort Processing Complete.\n"
_BANNER_TIMES = {"1": 0.0, "2": 5.0}  # the driver stamps each banner's arrival time


def test_per_contig_highwater_buckets_by_banner():
    rss = [(0.5, 41_000_000), (2.0, 41_000_000), (5.5, 80_000_000), (9.0, 80_000_000)]
    hw = parse_per_contig_highwater(_STDERR, rss, banner_times=_BANNER_TIMES)
    assert hw == {"1": 41_000_000, "2": 80_000_000}


def test_arena_heaps_counts_64mb_heaps():
    smaps = "Size:  65536 kB\n...\nSize:  65536 kB\n...\nSize:   4 kB\n"
    assert parse_arena_heaps(smaps) == 2


def test_arena_heaps_absent_is_zero():
    assert parse_arena_heaps("") == 0
```

(If the exact `parse_per_contig_highwater` signature you implement differs — e.g. you fold `banner_times` into the sampler — update this test to match; the load-bearing assertion is "per-contig high-water is bucketed correctly.")

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd scripts/bench_from_vcf_list
pixi run pytest test_run_bench.py -v
```
Expected: FAIL — `ImportError: cannot import name 'parse_per_contig_highwater'`.

- [ ] **Step 3: Implement the sampler + parsers**

In `run_bench.py`: when `--profiler time`, run the child under `/usr/bin/time -v` as today, but ALSO launch a lightweight RSS sampler thread that reads the child PID's `/proc/<pid>/status` `VmRSS` every ~2 s, appending `(elapsed_s, rss_kb)`; simultaneously tail the child's stdout for `==> Processing {chrom}` banners, stamping each banner's arrival `elapsed_s` into `banner_times`. Implement the two pure parsers above (`parse_per_contig_highwater`, `parse_arena_heaps`) and call them after the child exits. At the final contig, read the child's last `/proc/<pid>/smaps_rollup` (or a one-shot `smaps`) *just before it exits* is not reliable — instead sample arena-heap count from the periodic `/proc/<pid>/smaps` reads and keep the max. Append `per_contig_highwater_json` and `arena_heaps` to the CSV row. Add a `--rss-sample-interval` arg (default `2.0`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd scripts/bench_from_vcf_list
pixi run pytest test_run_bench.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_from_vcf_list/run_bench.py scripts/bench_from_vcf_list/test_run_bench.py
git commit -m "feat(bench): record per-contig RSS high-water + arena-heap count"
```

---

## Task 3: Establish the trustworthy baseline sweep (measurement, no code change)

**Files:**
- Create: `docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md`

**Interfaces:**
- Consumes: Task 1 generator (`--jobs`), Task 2 bench driver (per-contig high-water).
- Produces: a committed baseline report — per-N MaxRSS, the per-contig high-water curve, arena-heap counts, and the extrapolation to N=7,089 — that Phase C is measured against.

**Why:** The only real number (283 GiB) predates part of the current tree and the N^0.8 figure was F=2 synthetic. Nothing is optimized until there is a current-code baseline on this harness.

- [ ] **Step 1: Build the extension and generate the sweep cohorts**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo
pixi run maturin develop --release
COHORTS=$CLAUDE_JOB_DIR/tmp/cohorts
for N in 500 1000 2000 4000; do
  pixi run python scripts/bench_from_vcf_list/generate_cohort.py $COHORTS/n$N \
    --n-files $N --n-variants 300 \
    $(for c in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y; do echo --contig $c; done) \
    --format-field VAF --format-field DP --format-field AD --format-field GQ \
    --format-field PL --format-field MQ --format-field SB --jobs $(nproc)
done
```
(7 FORMAT fields ≈ the Hartwig F=7 regime; 24 contigs; `n-variants` per contig per file kept modest so the sweep runs in minutes, not hours — the ratchet shows at these sizes per the issue's N=1000-4000 data.)

- [ ] **Step 2: Run the baseline sweep**

```bash
RESULTS=$CLAUDE_JOB_DIR/tmp/baseline.csv
for N in 500 1000 2000 4000; do
  pixi run python scripts/bench_from_vcf_list/run_bench.py \
    --manifest $COHORTS/n$N/manifest.txt --out $CLAUDE_JOB_DIR/tmp/out_n$N.svar2 \
    --no-reference --threads 1 \
    --format-field VAF --format-field DP --format-field AD --format-field GQ \
    --format-field PL --format-field MQ --format-field SB \
    --results $RESULTS
done
```
Expected: 4 rows; `maxrss_kb` rising with N; `per_contig_highwater_json` showing a per-contig ratchet (high-water climbing across contigs 1→24).

- [ ] **Step 3: Fit and extrapolate**

Fit MaxRSS vs N (log-log exponent) and the mean per-contig increment; extrapolate peak to N=7,089. Record whether the extrapolation exceeds 64 GB (it is expected to).

- [ ] **Step 4: Write the baseline report**

Create the baseline doc with: the sweep table (N, MaxRSS, per-contig high-water curve, arena_heaps, wall_s), the exponent fit, the N=7,089 extrapolation, and a one-line statement of the gap to 64 GB. This is the number every Phase-C change is compared against.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md
git commit -m "docs(bench): record from_vcf_list peak-RAM baseline sweep (pre-fix)"
```

---

## Task 4: Localize the cross-contig retention (measurement — decides Task 5)

**Files:**
- Modify: `docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md` (append the localization section)

**Interfaces:**
- Consumes: Task 3 cohorts.
- Produces: a recorded verdict — **is the ratchet glibc arenas (freed-but-not-returned), retained Rust state, or on-disk staging read back?** — that selects the Task 5 lever.

**Why:** performant-py-rust: do not guess the fix. The contig loop (`orchestrator.rs:1044-1068`) already drops all Rust state per iteration (`process_chromosome` returns only `u64`), so arenas are the leading hypothesis — but confirm before coding.

- [ ] **Step 1: Heap-vs-RSS split at contig boundaries**

Run the `bench_from_vcf_list` Rust binary (or the Python path under `memray run`) on the N=2000 cohort and capture, at ≥2 contig boundaries: process RSS (`/proc/self/status` VmRSS), glibc `malloc_info()` (arena count + total_free vs. system bytes), and peak *live* heap (dhat or memray). Expected signature of the arena hypothesis: RSS high, `malloc_info` `<system>` bytes ≫ live heap, arena count rising per contig, live heap flat.

- [ ] **Step 2: Confirm what (if anything) Rust retains across the boundary**

Instrument `run_vcf_list` to log RSS + arena count immediately after each `process_chromosome` returns (before the next iteration). If RSS does not fall after a contig completes and live heap is flat, the retention is allocator-side (→ Task 5 = `malloc_trim`/arena-cap). If live heap itself ratchets, find the retained Rust owner (→ Task 5 = explicit drop). Record which.

- [ ] **Step 3: Write the verdict**

Append to the baseline report: the per-boundary heap-vs-RSS table and a one-line verdict naming the lever Task 5 must apply. **Do not proceed to Task 5 until this verdict is written.**

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md
git commit -m "docs(bench): localize the from_vcf_list cross-contig RSS ratchet"
```

---

## Task 5: Release memory at the contig boundary (the fix — gated on Task 4)

**Files:**
- Modify: `Cargo.toml` (add `libc`)
- Modify: `src/orchestrator.rs` (`run_vcf_list` contig loop)
- Test: `tests/test_svar2_from_vcf_list.py` (byte-identical multi-contig oracle — reuse/extend existing)

**Interfaces:**
- Consumes: Task 4 verdict.
- Produces: `run_vcf_list` returns freed memory to the OS after each contig; per-contig MaxRSS high-water stops ratcheting. No public-API change.

**Why (default path = arenas, per the leading hypothesis):** glibc keeps freed 64 MB arena heaps mapped. With the thread explosion gone (`VCF_LIST_HTSLIB_THREADS = 0`, 1 processing thread), `malloc_trim(0)` at the contig boundary is now cheap and safe (the old "73% slower" result was 4,000-thread arena-lock contention). If Task 4 instead found retained Rust state, replace the `malloc_trim` call with the explicit `drop(...)` of the named owner and keep the byte-identical gate identical.

- [ ] **Step 1: Write the failing test** (multi-contig store is byte-identical AND high-water is flat)

Extend the existing byte-identical oracle in `tests/test_svar2_from_vcf_list.py` with a small **multi-contig** cohort (≥3 contigs, F≥2) asserting `from_vcf_list` == `from_vcf` store bytes. (The flatness KPI is a bench measurement in Step 4, not a unit assertion — the unit test guards correctness only.) Confirm the assertion name/pattern matches the file's existing `_assert_stores_equal`-style helper.

- [ ] **Step 2: Run it on the UNPATCHED code to verify it passes** (guards against regressions)

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo
pixi run maturin develop --release
pixi run pytest tests/test_svar2_from_vcf_list.py -v -k multi_contig
```
Expected: PASS (this is a characterization test — it must pass before AND after the fix).

- [ ] **Step 3: Add `libc` and call `malloc_trim(0)` at the contig boundary**

In `Cargo.toml` `[dependencies]`: `libc = "0.2"`. In `run_vcf_list`, after `total_dropped += dropped;` inside the contig loop (orchestrator.rs ~1067), add a Linux-gated trim:

```rust
// glibc keeps freed per-contig arena heaps mapped, so RSS ratchets ~15-40 GB/contig
// across the 24-contig somatic cohort (issue #120). With htslib threads at 0 and one
// processing thread, malloc_trim is cheap here (no arena-lock contention) and returns
// the freed heaps to the OS between contigs. glibc-only; a no-op elsewhere.
#[cfg(target_os = "linux")]
// SAFETY: malloc_trim takes no ownership and only releases free top-of-heap memory.
unsafe {
    libc::malloc_trim(0);
}
```

- [ ] **Step 4: Re-measure — byte-identical AND ratchet flattened**

```bash
pixi run maturin develop --release
pixi run pytest tests/test_svar2_from_vcf_list.py -v          # byte-identical gate
# re-run the Task-3 sweep into a NEW results file and compare per_contig_highwater_json:
pixi run python scripts/bench_from_vcf_list/run_bench.py \
  --manifest $CLAUDE_JOB_DIR/tmp/cohorts/n2000/manifest.txt \
  --out $CLAUDE_JOB_DIR/tmp/out_fix_n2000.svar2 --no-reference --threads 1 \
  --format-field VAF --format-field DP --format-field AD --format-field GQ \
  --format-field PL --format-field MQ --format-field SB \
  --results $CLAUDE_JOB_DIR/tmp/fix.csv
```
Expected: tests PASS (byte-identical preserved); per-contig high-water **flat** (later contigs no longer set new high-water marks) and total MaxRSS down materially vs. the Task-3 baseline. If flat but total still > budget, that is the signal for Task 7 (stream the gather). Record wall-clock delta.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/orchestrator.rs tests/test_svar2_from_vcf_list.py
git commit -m "perf(svar2): trim glibc arenas between contigs to stop the from_vcf_list RSS ratchet (#120)"
```

---

## Task 6: Evaluate `MALLOC_ARENA_MAX` as a documented complement (gated on Task 5 result)

**Files:**
- Modify: `docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md` (append)
- Modify (docs only, if warranted): the `from_vcf_list` docstring in `python/genoray/_svar2.py`

**Interfaces:** Consumes Task 5. Produces a measured recommendation (env var value + measured RSS/wall trade-off), documented — **not** set from the library.

**Why:** The prior finding that `MALLOC_ARENA_MAX` is "73% slower" was at 4,000 threads; with 1 processing thread it may now be a free RSS win. But it is an environment concern — never `mallopt` from the library, never default it.

- [ ] **Step 1: Measure with and without the cap**

```bash
for AM in "" 2 4; do
  MALLOC_ARENA_MAX=$AM pixi run python scripts/bench_from_vcf_list/run_bench.py \
    --manifest $CLAUDE_JOB_DIR/tmp/cohorts/n2000/manifest.txt \
    --out $CLAUDE_JOB_DIR/tmp/out_am${AM:-def}.svar2 --no-reference --threads 1 \
    --format-field VAF --format-field DP --format-field AD --format-field GQ \
    --format-field PL --format-field MQ --format-field SB \
    --results $CLAUDE_JOB_DIR/tmp/arena.csv
done
```
Expected: a small table of (arena_max → MaxRSS, wall_s). If a cap helps RSS with acceptable wall cost *on top of* Task 5, document it as a recommended env var for very large cohorts.

- [ ] **Step 2: Record the recommendation (docs only)**

Append the trade-off table to the baseline report; if net-positive, add a one-paragraph note to the `from_vcf_list` docstring pointing users at `MALLOC_ARENA_MAX` for extreme N. If the API surface (docstring) changes, mirror it in `skills/genoray-api/SKILL.md`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md python/genoray/_svar2.py skills/genoray-api/SKILL.md
git commit -m "docs(svar2): record MALLOC_ARENA_MAX trade-off for large from_vcf_list cohorts (#120)"
```

---

## Task 7: Stream the per-contig gather — CONDITIONAL (only if single-contig peak still > 64 GB)

**Files:**
- Modify: `src/orchestrator.rs` / `src/chunk_assembler.rs` (window the per-contig gather flush)
- Test: `tests/test_svar2_from_vcf_list.py` (byte-identical gate, reused)

**Interfaces:** Consumes Tasks 5-6. Produces a per-contig peak bounded by window size, not contig size.

**Why:** After the ratchet is flat, peak = max single-contig cost (~41 GB for chr1 at N=7,089). If that alone still exceeds 64 GB, the contig gather must flush in windows so peak is bounded by `chunk_size`/window, not the whole contig. **Skip this task entirely if Task 5's flattened peak is already < 64 GB.**

- [ ] **Step 1: Decision gate**

From the Task-5 re-measure + extrapolation: is the flattened max single-contig peak at N=7,089 below 64 GB? If yes, mark this task **N/A**, note it in the baseline report, and stop. If no, proceed.

- [ ] **Step 2: Write the failing test**

Byte-identical multi-contig oracle at a chunk/window size small enough to force ≥2 windows per contig, asserting store bytes unchanged vs. a single-window run. (Reuse the Task-5 helper.)

- [ ] **Step 3: Implement windowed flush**

Bound the in-memory gather to `chunk_size` variants: assemble, route (`rvk`), and flush each window's staged output before reading the next, so live gather memory is O(window × N_dense), not O(contig × N). Keep the on-disk layout — hence the store bytes — identical.

- [ ] **Step 4: Re-measure (byte-identical + single-contig peak bounded)**

```bash
pixi run maturin develop --release
pixi run pytest tests/test_svar2_from_vcf_list.py -v
# sweep chunk_size at fixed N=2000; peak should track window size, not contig size
```
Expected: PASS; MaxRSS scales with `chunk_size`, and the N=7,089 extrapolation is < 64 GB.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.rs src/chunk_assembler.rs tests/test_svar2_from_vcf_list.py
git commit -m "perf(svar2): window the from_vcf_list per-contig gather to bound peak RSS (#120)"
```

---

## Task 8: Full-scale validation, `max_mem` verification, and close-out

**Files:**
- Modify: `docs/superpowers/plans/2026-07-18-svar2-from-vcf-list-ram-cross-contig-baseline.md` (final results)
- Modify: `skills/genoray-api/SKILL.md` (only if any public kwarg changed)

**Interfaces:** Consumes all prior tasks. Produces the #120 close-out evidence: peak RSS < 64 GB at N=7,089 (or the honest gap + next lever).

- [ ] **Step 1: Verify `max_mem` bites (lever 4 is already shipped)**

At N=2000, run with `max_mem="8GiB"` vs default and confirm the CSV MaxRSS drops (or record that peak is ratchet/reader-dominated and `max_mem` alone cannot cap it — a finding either way). No code change expected.

- [ ] **Step 2: Full-scale run at N=7,089**

Generate the full cohort (`--jobs $(nproc)`) and run one measured conversion. Record peak MaxRSS and the per-contig high-water curve.
Expected: peak RSS < 64 GB. If between 64 and the old ~283 GiB but not under target, record the residual and the recommended next lever (Task 7 if skipped, or reader-baseline reduction).

- [ ] **Step 3: Update the #120 peak-RAM model + report**

Finalize the baseline/results report: before/after table, the corrected peak-RAM model (`O(n_files) baseline + max_contig` after the fix vs. `+ Σ_contigs(retained)` before), and the wall-clock delta. Post the summary to #120.

- [ ] **Step 4: Full test + lint gate**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo
pixi run test
pixi run bash -lc 'cargo test --no-default-features'
ruff check genoray tests && ruff format --check genoray tests
```
Expected: all green.

- [ ] **Step 5: Commit + push + draft PR**

```bash
git add -A && git commit -m "docs(svar2): finalize from_vcf_list peak-RAM results and close #120"
git push -u origin worktree-svar2-from-vcf-list-ram-close-120
gh pr create --draft --title "perf(svar2): close #120 — from_vcf_list peak RAM (cross-contig ratchet)" \
  --body "$(cat <<'EOF'
Closes #120. Eliminates the cross-contig RSS ratchet in `from_vcf_list` so peak
memory is O(max single contig) instead of O(sum over contigs). Byte-identical
store throughout; correctness on the existing small-data differential oracle.
Full before/after in docs/superpowers/plans/2026-07-18-...-baseline.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review (against the spec)

- **Spec coverage:** §4 lever 1 → Tasks 4-5; lever 2 → Tasks 5-6; lever 3 → Task 7 (conditional); lever 4 → Task 8 Step 1 (verify, already shipped). §5.1 oracle → Task 5 gate. §5.2 generator → Task 1. §5.3 bench → Tasks 2-3. §6 measure-first loop → Tasks 3-4 before 5. §7 SKILL.md → Tasks 6/8. All covered.
- **Placeholders:** none — every code step shows the code or the exact command + expected output. Measurement-gated tasks (4-7) name the concrete candidate implementation and the decision criterion rather than a vague "optimize".
- **Type consistency:** `generate_cohort(..., jobs=...)` used consistently (Tasks 1, 3); `parse_per_contig_highwater` / `parse_arena_heaps` names match between Task 2 test and impl; `per_contig_highwater_json` / `arena_heaps` CSV columns consistent across Tasks 2-3-5.
