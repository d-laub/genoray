# SVAR2 Region And Sub-Contig VCF Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native `SparseVar2.from_vcf(regions=..., samples=...)`, then split one large contig into deterministic padded indexed work units so parse, genotype decode, and internal normalization can all run in parallel.

**Architecture:** Ship this as a PR stack. PR 1 adds the serial indexed API foundation and CLI parity while preserving the existing full-file path. PR 2 introduces padded sub-contig work units with deterministic ownership so each worker fetches, atomizes, left-aligns, and emits only its owned normalized atoms. PR 3 wires bounded ordered merge, structured progress integration, stress tests, and benchmarks.

**Tech Stack:** Python 3.10+, Rust, PyO3, rust-htslib `IndexedReader`, rayon, crossbeam-channel, Polars, cyvcf2, Cyclopts, pytest, cargo tests with `--no-default-features --features conversion`.

## Global Constraints

- Current `origin/main` has no sub-contig parallelism for `SparseVar2.from_vcf`: `VcfRecordSource::new` still calls `fetch(rid, 0, None)`.
- Current internal normalization (`atomize_record`, `validate_ref`, `left_align`) runs in one reader thread per contig; dlaub is correct that norm is not sub-contig-parallelized.
- Region string input follows existing Genoray convention: `chrom:start-end` is 1-based inclusive and converts to 0-based half-open. BED, tuples, and frames are 0-based half-open.
- `threads` remains a total process budget. Do not multiply it independently across contig readers, BGZF threads, packing pools, and merge writers.
- Existing no-argument behavior must be byte-compatible: `regions=None`, `samples=None`, `threads=1` follows the current full-contig conversion path.
- PR #113 already owns structured progress. Do not duplicate it; integrate with it after dlaub merges, or keep progress changes isolated in a follow-up PR.
- CI gate for each PR: use the exact cargo filters listed in that task, focused pytest, `pixi run pytest tests -m "not network"` before final PR readiness, and `pixi run prek run --all-files` before push.

---

## File Structure

| File | Responsibility | Planned Change |
| --- | --- | --- |
| `python/genoray/_svar2.py` | Public `SparseVar2` API and VCF metadata discovery | Add `regions`, `samples`, `merge_overlapping`, `regions_overlap`; normalize inputs using v1 helpers; pass selected samples and per-contig ranges to Rust. |
| `python/genoray/_cli/__main__.py` | `genoray write` CLI | Add `--regions/-r`, `--regions-file/-R`, `--samples/-s`, `--samples-file/-S` for SVAR2 VCF/BCF writes. |
| `python/genoray/_cli/_view_helpers.py` | CLI parsing helpers | Reuse `parse_regions_arg` for write-region lists. |
| `src/lib.rs` | PyO3 conversion entrypoint and thread budget dispatch | Extend `run_conversion_pipeline` to accept region ranges for PR 1; later accept shard work units and bounded merge config. |
| `src/orchestrator.rs` | Conversion pipeline wiring | Add region-aware VCF source specs in PR 1; add range-shard worker orchestration in PR 2. |
| `src/vcf_reader.rs` | Indexed VCF/BCF record source | PR 1: fetch one or more intervals serially. PR 2: create worker-safe range readers that fetch padded intervals. |
| `src/chunk_assembler.rs` | Atomize/left-align/reorder/pack | PR 2: expose a worker-safe normalized atom batch path or split shared normalization into a new module. |
| `src/normalize.rs` | Atomization and left-alignment | Keep rules unchanged; add boundary ownership tests around left-shifted indels. |
| `tests/test_svar2_from_vcf.py` | Python API tests | Add regions/samples tests, overlap dedupe, unknown sample/contig errors, and no-argument compatibility. |
| `tests/cli/test_write_cli.py` | CLI tests | Add write-time `--regions`/`--samples` coverage. |
| `tests/test_*e2e.rs` | Rust conversion tests | Add serial interval fetch, padded shard ownership, deterministic merge, and worker-failure cases. |
| `docs/source/svar.md` and `docs/roadmap/svar-2.md` | User docs and architecture notes | Document coordinate conventions, sample order, thread budget, and benchmark results. |

---

### Task 1: Serial Region/Sample Foundation (PR 1)

**Files:**
- Modify: `python/genoray/_svar2.py`
- Modify: `src/lib.rs`
- Modify: `src/orchestrator.rs`
- Modify: `src/vcf_reader.rs`
- Test: `tests/test_svar2_from_vcf.py`

**Interfaces:**
- Consumes: `genoray._svar._regions._normalize_regions`, `_normalize_samples`, and `_resolve_kept_rows`.
- Produces: `SparseVar2.from_vcf(..., regions=None, samples=None, merge_overlapping=False, regions_overlap="pos")`.
- Produces Rust input: `Vec<(String, u32, u32)>` region ranges, where empty means full contig.

- [ ] **Step 1: Add failing Python tests**

Add tests that build the existing tiny VCF and assert:

```python
def test_from_vcf_regions_restricts_conversion(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "regioned"
    SparseVar2.from_vcf(out, vcf, ref, regions="chr1:1-4", threads=1)
    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]
    counts = sv.var_counts("chr1", [0], [40], samples=["S0"])
    assert int(counts.sum()) == 1
```

Also add tests for `samples=["S1"]` preserving output sample metadata, unknown samples raising `ValueError`, and `regions=None` still matching the current full conversion.

- [ ] **Step 2: Run tests to verify failure**

Run: `pixi run pytest tests/test_svar2_from_vcf.py -k 'regions or samples' -q`

Expected: fail with `TypeError: SparseVar2.from_vcf() got an unexpected keyword argument 'regions'`.

- [ ] **Step 3: Add Python argument normalization**

In `SparseVar2.from_vcf`, add keyword-only parameters:

```python
regions: str | tuple[str, int, int] | Path | object | None = None,
samples: str | Sequence[str] | Path | None = None,
merge_overlapping: bool = False,
regions_overlap: Literal["pos", "record", "variant"] = "pos",
```

Resolve samples with `_normalize_samples` when provided. Resolve regions with `_normalize_regions` against `ContigNormalizer(v.seqnames)`, then coalesce with `_resolve_kept_rows` for validation and dedupe. For PR 1, pass per-contig intervals to Rust and preserve contigs in VCF header order after filtering out contigs with no selected variants.

- [ ] **Step 4: Add serial interval fetch in Rust**

Extend `SourceSpec::Vcf` with `regions: Vec<(u32, u32)>`. Update `VcfRecordSource::new` to accept a `Vec<(u32, u32)>`; if empty, keep the current `fetch(rid, 0, None)`. If non-empty, sort/coalesce ranges and fetch them sequentially. `next_record` should advance to the next interval when the current fetch returns EOF.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pixi run pytest tests/test_svar2_from_vcf.py -q
pixi run bash -lc 'cargo test --no-default-features --features conversion vcf_reader'
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2.py src/lib.rs src/orchestrator.rs src/vcf_reader.rs tests/test_svar2_from_vcf.py
git commit -m "feat(svar2): add serial regions and samples to from_vcf"
```

---

### Task 2: CLI Parity And Docs (PR 1)

**Files:**
- Modify: `python/genoray/_cli/__main__.py`
- Modify: `tests/cli/test_write_cli.py`
- Modify: `docs/source/svar.md`

**Interfaces:**
- Consumes: Task 1 `SparseVar2.from_vcf(..., regions=..., samples=...)`.
- Produces: `genoray write SOURCE OUT --regions/-r`, `--regions-file/-R`, `--samples/-s`, `--samples-file/-S`.

- [ ] **Step 1: Add failing CLI tests**

Add tests that run:

```bash
python -m genoray._cli write in.vcf.gz out.svar2 --reference ref.fa --regions chr1:1-4 --samples S1 --threads 1
```

Then assert `SparseVar2(out).available_samples == ["S1"]` and the store contains only the expected selected variant.

- [ ] **Step 2: Implement CLI flags**

Mirror the existing `view_svar2` parsing style. For `--regions`, call `parse_regions_arg`; for `--regions-file`, pass the `Path`; for `--samples`, split comma-separated names; for `--samples-file`, pass the `Path`. Reject both inline and file variants together.

- [ ] **Step 3: Run focused CLI/docs tests**

Run:

```bash
pixi run pytest tests/cli/test_write_cli.py -q
pixi run pytest tests/test_svar2_from_vcf.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add python/genoray/_cli/__main__.py tests/cli/test_write_cli.py docs/source/svar.md
git commit -m "feat(cli): expose regions and samples for SVAR2 VCF writes"
```

---

### Task 3: Padded Sub-Contig Work Units (PR 2)

**Files:**
- Modify: `src/vcf_reader.rs`
- Modify: `src/chunk_assembler.rs`
- Modify: `src/orchestrator.rs`
- Test: `tests/test_svar2_from_vcf.py`
- Test: `tests/test_left_align_e2e.rs`

**Interfaces:**
- Consumes: Task 1 per-contig normalized intervals.
- Produces: `VcfShard { fetch_start, fetch_end, own_start, own_end, ordinal, record_count }`.
- Produces ownership rule: keep a normalized atom iff `own_start <= atom.pos < own_end`.

- [ ] **Step 1: Add boundary tests**

Create a VCF fixture where an insertion/deletion left-aligns from the padded fetch interval into a neighboring ownership interval. Assert `threads=1` and `threads=4` produce the same logical decoded variants and no duplicates.

- [x] **Step 2: Build shard planner**

Perform a lightweight indexed position scan without decoding sample fields, then
split each requested interval after approximately equal numbers of source records.
Never split equal-POS records. Treat `chunk_size` as a maximum record count and use
`ceil(total_records / worker_budget)` when smaller so available workers stay busy.
Clamp `normalize::L_MAX` padding to the caller's requested-region edges, assign
stable ordinals, and retain each shard's exact source-record count.

- [ ] **Step 3: Move normalization into worker batches**

Factor `ChunkAssembler::decompose_record` logic into a worker-callable function that returns normalized atom metadata plus genotype/field payloads. Workers run fetch, GT decode, atomization, validation, left-alignment, and ownership filtering before sending ordered batches.

- [ ] **Step 4: Add ordered bounded merge**

Merge worker batches by `(chrom ordinal, normalized pos, seq tie-breaker)` into the existing dense chunk assembly. Use bounded channels so completed batches cannot accumulate without limit.

- [ ] **Step 5: Run deterministic tests**

Run:

```bash
pixi run pytest tests/test_svar2_from_vcf.py -k 'regions or boundary or deterministic' -q
pixi run bash -lc 'cargo test --no-default-features --features conversion left_align_e2e'
```

Expected: all pass repeatedly.

- [ ] **Step 6: Commit**

```bash
git add src/vcf_reader.rs src/chunk_assembler.rs src/orchestrator.rs tests/test_svar2_from_vcf.py tests/test_left_align_e2e.rs
git commit -m "perf(svar2): shard VCF conversion within contigs"
```

---

### Task 4: Thread Budget, Progress, And Failure Handling (PR 3)

**Files:**
- Modify: `src/budget.rs`
- Modify: `src/lib.rs`
- Modify: `src/orchestrator.rs`
- Modify after PR #113 merges: `python/genoray/_progress.py`, `python/genoray/_svar2.py`
- Test: `tests/test_svar2_progress.py`
- Test: `tests/test_svar2_errors.py`

**Interfaces:**
- Consumes: Task 3 shard planner and PR #113 progress observer contract.
- Produces: a single process-wide reader/normalizer/BGZF budget and one aggregated progress stream.

- [x] **Step 1: Add thread-budget tests**

Assert that `threads=32` on one contig allocates multiple VCF shards but does not allocate `32` HTSlib threads per shard. Assert total planned worker plus BGZF plus packing threads is bounded by `threads`.

- [x] **Step 2: Implement budget allocator**

Extend `ThreadPlan` with a per-contig `shard_workers` budget. The position scan uses
the serial path's HTSlib pool; shard workers then replace those HTSlib threads and
decompress synchronously, so worker plus pipeline threads stay within the process
budget. Bound normalization batches by cohort width (about 64 MiB of GT indices per
reader, capped at 1,024 records).

- [ ] **Step 3: Integrate progress**

After PR #113 is merged, emit one aggregate conversion progress stream with decoded, normalized, written, completed work units, active reader count, and final native finalization state.

- [ ] **Step 4: Add worker failure cleanup tests**

Inject a missing contig and a REF mismatch inside one shard. Assert sibling workers stop, output is removed or restored according to existing atomic-output behavior, and the error includes the shard region.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pixi run pytest tests/test_svar2_progress.py tests/test_svar2_errors.py -q
pixi run bash -lc 'cargo test --no-default-features --features conversion budget'
pixi run bash -lc 'cargo test --no-default-features --features conversion orchestrator'
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/budget.rs src/lib.rs src/orchestrator.rs python/genoray/_progress.py python/genoray/_svar2.py tests/test_svar2_progress.py tests/test_svar2_errors.py
git commit -m "feat(svar2): aggregate progress and bounded worker failures"
```

---

### Task 5: Benchmarks And PR Readiness

**Files:**
- Create or modify: `scripts/svar2_region_parallel_bench.py`
- Modify: `docs/roadmap/svar-2.md`
- Modify: PR descriptions

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: benchmark table for 1, 2, 4, 8, 16, 32 threads and a high-core run on a wide VCF.

- [ ] **Step 1: Add benchmark script**

Benchmark full conversion and region-restricted conversion separately. Report read/decode, normalize, pack, merge/write, finalization, wall time, and peak RSS where available.

- [ ] **Step 2: Run full verification**

Run:

```bash
pixi run pytest tests -m "not network"
pixi run bash -lc 'cargo test --no-default-features --features conversion'
pixi run prek run --all-files
```

Expected: all pass.

- [ ] **Step 3: Open PRs**

Open PR 1 after Tasks 1-2. Open PR 2 after Task 3. Open PR 3 after Task 4-5. Request dlaub review on each PR and state that only dlaub can merge.

- [ ] **Step 4: Monitor and fix CI**

Use GitHub Actions logs for any failing checks. Fix failures on the same branch, push, and wait until every check is green.
