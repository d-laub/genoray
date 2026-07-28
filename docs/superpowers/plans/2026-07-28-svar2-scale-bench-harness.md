# SVAR2 Scale-Bench Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallel execution:** Tasks are grouped into waves below. Within a wave, tasks touch disjoint files and MUST be dispatched concurrently using superpowers:dispatching-parallel-agents. Use Sonnet or weaker for implementers; reserve Opus for review and for fixing a critical implementer failure.

**Goal:** Build a benchmark harness that predicts biobank-scale (500,000-sample) VCF→SVAR2 conversion behaviour from affordable measurements, and decides whether the reader-budget needs a static cap, a formula in cohort size, or a byte-bounded global pool.

**Architecture:** A Python harness under `scripts/bench_svar2/` splits into five single-responsibility modules around one shared record schema: corpus generation, single-run probing, sweep execution, law fitting, and a fast regression tier. Three trace-level Rust instrumentation additions make the currently-invisible reorder-buffer backlog measurable and let a benchmark override contig concurrency. Nothing on a default code path changes behaviour.

**Tech Stack:** Python 3.10+ (numpy, stdlib multiprocessing/dataclasses/statistics — no new dependencies), Rust (crossbeam-channel, tracing, std::sync::atomic), pixi, pytest, bgzip/tabix from htslib, Slurm.

## Global Constraints

Every task's requirements implicitly include this section.

- **Rust tests MUST use `--no-default-features --features conversion`.** Plain `cargo test` fails to link the pyo3 test binary with `undefined symbol: _Py_Dealloc`.
- **`export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp/cargo-target` before any cargo command.** The repo lives on NFS; an NFS `target/` bus-errors on debug builds and on prek's cargo hooks.
- **Run `maturin develop --release` before any Python-level verification of a Rust change.** `pixi run test` does NOT rebuild the extension module, so Python tests silently run stale `.so` code.
- **Bench data and temp files go under `$CLAUDE_JOB_DIR/tmp`, never `/tmp`.** `/tmp` is reaped on this cluster and has silently destroyed corpora mid-run.
- **All commits follow Conventional Commits.** Never edit `CHANGELOG.md` — commitizen owns it.
- **No public API changes in this plan.** Everything added is under `scripts/` or is trace-level Rust. `skills/genoray-api/SKILL.md` therefore needs no update; if a task finds itself changing a name reachable from `import genoray` without an underscore, stop and escalate.
- **Multiprocessing worker functions MUST live in a module imported by name**, and entry points MUST be run as `python -m scripts.bench_svar2.<module>` from the repo root. Python 3.14 on Linux defaults to the `forkserver` start method; a pool worker in a `spec_from_file_location`-loaded module dies with `ModuleNotFoundError` → `BrokenProcessPool`.
- **Slurm CPU pinning MUST use `os.sched_getaffinity(0)`,** never `0-(N-1)`. Slurm hands out non-contiguous CPU ids and `taskset -c 0-15` fails with "Invalid argument".
- **Ploidy is 2 throughout.** `_DENSE_CHUNK_TARGET_BYTES = 256 * 1024 * 1024` and `_auto_chunk_size`'s 25,000 clamp are the production reference values.

## File Structure

| path | responsibility |
|---|---|
| `scripts/bench_svar2/__init__.py` | package marker so `python -m` and pool workers resolve by name |
| `scripts/bench_svar2/records.py` | shared frozen dataclasses + JSON codecs. The interface contract all other modules consume |
| `scripts/bench_svar2/scale_corpus.py` | deterministic seeded VCF generation, parallel formatting, manifest emission |
| `scripts/bench_svar2/probe.py` | exactly one instrumented conversion run → one `ProbeRecord` |
| `scripts/bench_svar2/sweep.py` | executes a plan of points, resumable NDJSON, RSS ceiling |
| `scripts/bench_svar2/model.py` | pure law fitting, hypothesis verdict, extrapolation. No I/O beyond reading NDJSON |
| `scripts/bench_svar2/regression.py` | fast tier against committed baselines |
| `scripts/bench_svar2/plans/*.json` | the sweep plans (scale, contig, holdout) |
| `scripts/bench_svar2/baselines/regression.json` | committed regression baselines |
| `scripts/bench_svar2/README.md` | findings + usage (moved from `bench_sharded_vcf/`) |
| `src/monitor.rs` | + `PendingGauge`, + `pending`/`pending_bytes` trace fields |
| `src/types.rs` | + `DenseChunk::approx_bytes()` |
| `src/shard_exec.rs` | + gauge updates in the collector, + per-unit completion timing |
| `src/orchestrator.rs` | + `GENORAY_CONCURRENT_CHROMS` hook, + gauge wiring |
| `tests/bench/test_records.py` | round-trip codecs |
| `tests/bench/test_model.py` | law fitting against planted synthetic laws |
| `tests/bench/test_corpus_smoke.py` | tiny end-to-end generate→convert smoke |
| `pyproject.toml:61-64` | + `bench` marker |
| `pixi.toml:69-74` | + `bench-regression` task |

## Execution Waves

| wave | tasks | may run concurrently |
|---|---|---|
| 1 | Task 1, Task 2 | yes — disjoint (Python skeleton vs Rust) |
| 2 | Task 3, Task 4 | yes — disjoint (`scale_corpus.py` vs `model.py`) |
| 3 | Task 5 | no — needs Task 2's trace fields and Task 1's schema |
| 4 | Task 6 | no — needs Task 5 |
| 5 | Task 7, Task 8 | yes — disjoint (regression tier vs plans/README) |

---

### Task 1: Harness package skeleton and shared record schema

Creates the interface contract every later Python task consumes. Also relocates the existing PR #140 harness, since it now characterizes the conversion pipeline rather than only sharding.

**Files:**
- Create: `scripts/bench_svar2/__init__.py`
- Create: `scripts/bench_svar2/records.py`
- Create: `tests/bench/test_records.py`
- Modify: `pyproject.toml:61-64` (add `bench` marker)
- Move: `scripts/bench_sharded_vcf/{gen_vcf.py,bench_sharded.py,sweep.sbatch,sweep2.sbatch,README.md}` → `scripts/bench_svar2/legacy_pr140/`

**Interfaces:**
- Consumes: nothing.
- Produces: `CorpusManifest`, `SweepPoint`, `ProbeRecord`, `VLaw`, `CostLaw`, `RamLaw`, `Verdict`; `to_json(obj) -> str`, `from_json(cls, s)`, `read_ndjson(path, cls) -> list`, `append_ndjson(path, obj) -> None`.

- [ ] **Step 1: Move the existing harness and add the package marker**

```bash
cd /carter/users/dlaub/projects/genoray/.claude/worktrees/bench-pr140-reader-workers
mkdir -p scripts/bench_svar2/legacy_pr140 tests/bench
git mv scripts/bench_sharded_vcf/gen_vcf.py       scripts/bench_svar2/legacy_pr140/gen_vcf.py
git mv scripts/bench_sharded_vcf/bench_sharded.py scripts/bench_svar2/legacy_pr140/bench_sharded.py
git mv scripts/bench_sharded_vcf/sweep.sbatch     scripts/bench_svar2/legacy_pr140/sweep.sbatch
git mv scripts/bench_sharded_vcf/sweep2.sbatch    scripts/bench_svar2/legacy_pr140/sweep2.sbatch
git mv scripts/bench_sharded_vcf/README.md        scripts/bench_svar2/legacy_pr140/README.md
rmdir scripts/bench_sharded_vcf
printf '"""Benchmark harness for the SVAR2 conversion pipeline."""\n' > scripts/bench_svar2/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/bench/test_records.py`:

```python
import pytest

from scripts.bench_svar2.records import (
    CorpusManifest,
    ProbeRecord,
    SweepPoint,
    append_ndjson,
    from_json,
    read_ndjson,
    to_json,
)

pytestmark = pytest.mark.bench


def _manifest() -> CorpusManifest:
    return CorpusManifest(
        path="corpus.vcf.gz",
        samples=1000,
        variants=100_000,
        contigs=("chr22",),
        format_fields=(),
        ploidy=2,
        cells=100_000_000,
        compressed_bytes=30_000_000,
        seed=7,
        generator_version=1,
    )


def _record(point_id: str = "p0") -> ProbeRecord:
    return ProbeRecord(
        point_id=point_id,
        ok=True,
        wall_s=10.2,
        phase1_s=8.1,
        cpu_s=30.0,
        maxrss_mb=512.0,
        digest="abc123",
        dense_cap=6,
        dense_occupancy=(0, 1, 5),
        cpu_shard_pct=(100.0, 360.0),
        cpu_exec_pct=(60.0, 50.0),
        pending_highwater=3,
        pending_bytes_highwater=78_643_200,
        shard_unit_secs=(1.5, 2.5),
        oom_at_rss_mb=None,
        error=None,
    )


def test_manifest_round_trips():
    m = _manifest()
    assert from_json(CorpusManifest, to_json(m)) == m


def test_tuple_fields_survive_round_trip():
    """JSON has no tuples; the codec must restore them or equality breaks."""
    r = from_json(ProbeRecord, to_json(_record()))
    assert r.dense_occupancy == (0, 1, 5)
    assert isinstance(r.dense_occupancy, tuple)


def test_optional_fields_round_trip():
    r = _record()
    failed = ProbeRecord(**{**r.__dict__, "ok": False, "oom_at_rss_mb": 64_000.0, "error": "OOM"})
    assert from_json(ProbeRecord, to_json(failed)).oom_at_rss_mb == 64_000.0


def test_ndjson_append_and_read(tmp_path):
    p = tmp_path / "results.ndjson"
    append_ndjson(p, _record("p0"))
    append_ndjson(p, _record("p1"))
    got = read_ndjson(p, ProbeRecord)
    assert [g.point_id for g in got] == ["p0", "p1"]


def test_read_ndjson_missing_file_is_empty(tmp_path):
    """Resumption reads before the first write; that must not raise."""
    assert read_ndjson(tmp_path / "nope.ndjson", ProbeRecord) == []


def test_sweep_point_id_is_deterministic():
    a = SweepPoint(corpus="c.json", reader_workers=3, concurrent_chroms=None,
                   shard_htslib=0, overshard=4, chunk_size=25_000, threads=16, reps=3)
    b = SweepPoint(corpus="c.json", reader_workers=3, concurrent_chroms=None,
                   shard_htslib=0, overshard=4, chunk_size=25_000, threads=16, reps=3)
    c = SweepPoint(corpus="c.json", reader_workers=5, concurrent_chroms=None,
                   shard_htslib=0, overshard=4, chunk_size=25_000, threads=16, reps=3)
    assert a.point_id == b.point_id
    assert a.point_id != c.point_id
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pixi run pytest tests/bench/test_records.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.bench_svar2.records'`

- [ ] **Step 4: Add the `bench` marker so the test is collectable**

In `pyproject.toml`, extend the `markers` list at line 61:

```toml
markers = [
    "network: requires network access (deselect with '-m not network')",
    "sigprofiler: requires SigProfilerMatrixGenerator (run in the 'sigprofiler' pixi env)",
    "bench: benchmark-harness tests (deselect with '-m not bench')",
]
```

- [ ] **Step 5: Write the implementation**

Create `scripts/bench_svar2/records.py`:

```python
"""Shared record schema for the SVAR2 scale-bench harness.

Every other module in this package reads or writes these types. They are frozen
so a record cannot be mutated after a run is recorded, and the JSON codec
restores tuples explicitly because JSON has no tuple type -- without that,
round-tripped records compare unequal and resumption silently re-runs points.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CorpusManifest:
    """Shape of one generated corpus. Written next to the .vcf.gz.

    Consumers read shape from here rather than parsing filenames, so a corpus
    can be renamed or relocated without breaking a sweep.
    """

    path: str
    samples: int
    variants: int
    contigs: tuple[str, ...]
    format_fields: tuple[str, ...]
    ploidy: int
    cells: int
    compressed_bytes: int
    seed: int
    generator_version: int

    @property
    def chunk_bytes(self) -> int:
        """Analytic bytes of one dense chunk at `chunk_size` variants.

        Mirrors `_auto_chunk_size`'s cost model: packed presence grid plus
        staged FORMAT values. Callers multiply by chunk_size.
        """
        grid = (self.samples * self.ploidy) // 8
        fmt = len(self.format_fields) * self.samples * 4
        return grid + fmt


@dataclass(frozen=True)
class SweepPoint:
    """One configuration to measure. `point_id` is content-derived so a
    resumed sweep can skip points it already recorded."""

    corpus: str
    reader_workers: int
    concurrent_chroms: int | None
    shard_htslib: int
    overshard: int
    chunk_size: int
    threads: int
    reps: int
    rss_ceiling_mb: int | None = None

    @property
    def point_id(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ProbeRecord:
    """Result of one instrumented conversion run.

    `ok=False` with `oom_at_rss_mb` set is a legitimate datum, not an error --
    demonstrating that the current chunk_size OOMs at scale is a deliverable.
    """

    point_id: str
    ok: bool
    wall_s: float
    phase1_s: float
    cpu_s: float
    maxrss_mb: float
    digest: str
    dense_cap: int
    dense_occupancy: tuple[int, ...]
    cpu_shard_pct: tuple[float, ...]
    cpu_exec_pct: tuple[float, ...]
    pending_highwater: int
    pending_bytes_highwater: int
    shard_unit_secs: tuple[float, ...]
    oom_at_rss_mb: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class VLaw:
    """phase1_s ~ intercept + slope * variants."""

    slope_s_per_variant: float
    intercept_s: float
    r2: float
    n_points: int
    max_extrapolation_factor: float


@dataclass(frozen=True)
class CostLaw:
    """cost(S) = alpha * S**beta, fitted on logs."""

    name: str
    alpha: float
    beta: float
    beta_ci95: tuple[float, float]
    n_points: int


@dataclass(frozen=True)
class RamLaw:
    """peak_rss_mb ~ base_mb + kappa * (workers + pending_hw) * chunk_bytes."""

    base_mb: float
    kappa: float
    r2: float
    n_points: int


@dataclass(frozen=True)
class Verdict:
    hypothesis: str  # "H1" | "H2" | "H3" | "none"
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)


# --- codecs -----------------------------------------------------------------

def _tuple_fields(cls: type) -> dict[str, type]:
    """Field names whose annotation is a tuple type. JSON round-trips them as
    lists, so they must be re-coerced or frozen-dataclass equality fails."""
    hints = typing.get_type_hints(cls)
    out = {}
    for f in dataclasses.fields(cls):
        origin = typing.get_origin(hints[f.name])
        if origin is tuple:
            args = typing.get_args(hints[f.name])
            out[f.name] = args[0] if args else str
    return out


def to_json(obj: Any) -> str:
    return json.dumps(dataclasses.asdict(obj), sort_keys=True)


def from_json(cls: type[T], s: str) -> T:
    raw = json.loads(s)
    coerce = _tuple_fields(cls)
    for name, elem in coerce.items():
        if name in raw and raw[name] is not None:
            raw[name] = tuple(elem(v) for v in raw[name])
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def append_ndjson(path: Path, obj: Any) -> None:
    """Append one record and fsync. A preempted Slurm job must not lose the
    point it just finished paying for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(to_json(obj) + "\n")
        fh.flush()
        import os

        os.fsync(fh.fileno())


def read_ndjson(path: Path, cls: type[T]) -> list[T]:
    if not Path(path).exists():
        return []
    return [
        from_json(cls, line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pixi run pytest tests/bench/test_records.py -v`
Expected: PASS — 6 passed

- [ ] **Step 7: Commit**

```bash
git add scripts/bench_svar2 tests/bench/test_records.py pyproject.toml
git rm -r --cached scripts/bench_sharded_vcf 2>/dev/null || true
git commit -m "test(svar2): shared record schema for the scale-bench harness"
```

---

### Task 2: Rust instrumentation — pending gauge, shard timings, concurrency hook

Three additions, all trace-level, none changing default behaviour. Grouped into one task because they share `orchestrator.rs` and would otherwise conflict.

The reorder-buffer backlog is the load-bearing one: `shard_exec.rs`'s `pending: HashMap<(usize, usize), DenseChunk>` is unbounded, and hypothesis H3 cannot be tested without measuring it.

**Files:**
- Modify: `src/types.rs` (add `DenseChunk::approx_bytes`, `BitGrid3::approx_bytes`, `StagedColumn::approx_bytes`)
- Modify: `src/monitor.rs:85-107` (add `PendingGauge`, new `spawn_sampler` parameter, two trace fields)
- Modify: `src/shard_exec.rs:180-183, 326` (add gauge parameter, update on insert/remove, time each unit)
- Modify: `src/orchestrator.rs:358-370, 423, 506, 696` (create and thread the gauge; add `GENORAY_CONCURRENT_CHROMS`)

**Interfaces:**
- Consumes: nothing.
- Produces: trace fields `pending=<usize>`, `pending_bytes=<u64>` on the `genoray::monitor` target's `pipeline sampler` event; trace event `target: "genoray::monitor", unit_ordinal=<usize>, unit_secs=<f64>, "shard unit done"`; env var `GENORAY_CONCURRENT_CHROMS`.

- [ ] **Step 1: Write the failing test**

Append to `src/monitor.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::PendingGauge;
    use std::sync::atomic::Ordering;

    #[test]
    fn gauge_records_len_highwater_not_current() {
        let g = PendingGauge::default();
        g.observe(3, 300);
        g.observe(7, 700);
        g.observe(1, 100);
        assert_eq!(g.len_highwater.load(Ordering::Relaxed), 7);
        assert_eq!(g.bytes_highwater.load(Ordering::Relaxed), 700);
    }

    #[test]
    fn gauge_starts_at_zero() {
        let g = PendingGauge::default();
        assert_eq!(g.len_highwater.load(Ordering::Relaxed), 0);
        assert_eq!(g.bytes_highwater.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn gauge_tracks_bytes_independently_of_len() {
        // A single very large chunk must raise the byte high-water even though
        // the length high-water is already higher from an earlier tick.
        let g = PendingGauge::default();
        g.observe(9, 90);
        g.observe(1, 5_000);
        assert_eq!(g.len_highwater.load(Ordering::Relaxed), 9);
        assert_eq!(g.bytes_highwater.load(Ordering::Relaxed), 5_000);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp/cargo-target
cargo test --no-default-features --features conversion monitor::tests
```
Expected: FAIL — `cannot find type PendingGauge in this scope`

- [ ] **Step 3: Add `PendingGauge` to `src/monitor.rs`**

Insert above `pub fn spawn_sampler` (line 85), and add `use std::sync::atomic::AtomicU64;` plus `AtomicUsize` to the existing atomic imports:

```rust
/// High-water gauge for the shard collector's reorder backlog.
///
/// `shard_exec`'s `pending` map is unbounded by construction: a fast shard's
/// chunks accumulate while the reorder head waits on a slow one. That backlog
/// is a second peak-RSS term alongside the in-flight `workers * chunk_bytes`,
/// and it is otherwise invisible -- nothing else observes the map.
///
/// High-water rather than instantaneous: the sampler ticks on a multi-second
/// interval and would routinely miss the transient peak that actually sets
/// peak RSS.
#[derive(Debug, Default)]
pub struct PendingGauge {
    pub len_highwater: AtomicUsize,
    pub bytes_highwater: AtomicU64,
}

impl PendingGauge {
    /// Record the collector's current backlog. Monotonic in both dimensions;
    /// `Relaxed` is sufficient because these are diagnostics with no
    /// happens-before obligation to any other state.
    pub fn observe(&self, len: usize, bytes: u64) {
        self.len_highwater.fetch_max(len, Ordering::Relaxed);
        self.bytes_highwater.fetch_max(bytes, Ordering::Relaxed);
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test --no-default-features --features conversion monitor::tests
```
Expected: PASS — 3 passed

- [ ] **Step 5: Add byte accounting to `src/types.rs`**

Add to `impl BitGrid3`:

```rust
    /// Heap bytes held by the packed grid.
    pub fn approx_bytes(&self) -> u64 {
        (self.words.len() * std::mem::size_of::<u64>()) as u64
    }
```

Add to `impl StagedColumn`:

```rust
    /// Heap bytes held by this staged column.
    pub fn approx_bytes(&self) -> u64 {
        match self {
            StagedColumn::Int(v) => (v.len() * std::mem::size_of::<i32>()) as u64,
            StagedColumn::Float(v) => (v.len() * std::mem::size_of::<f32>()) as u64,
        }
    }
```

Add a new `impl DenseChunk` block (or extend the existing one):

```rust
impl DenseChunk {
    /// Approximate heap bytes held by this chunk.
    ///
    /// Used only by the benchmark gauge, so it counts the dominant allocations
    /// (the packed grid, the variant metadata vectors, and staged field
    /// columns) and ignores per-Vec overhead.
    pub fn approx_bytes(&self) -> u64 {
        let meta = (self.pos.len() * 4
            + self.global_idx.len() * 4
            + self.ilens.len() * 4
            + self.alt.len()
            + self.alt_offsets.len() * 4) as u64;
        let staged: u64 = self
            .info_staged
            .iter()
            .chain(self.format_staged.iter())
            .map(|c| c.approx_bytes())
            .sum();
        meta + self.genos.approx_bytes() + staged
    }
}
```

- [ ] **Step 6: Thread the gauge through `spawn_sampler`**

In `src/monitor.rs`, add a parameter after `shard_worker_tids`:

```rust
    shard_worker_tids: Arc<Mutex<Vec<i32>>>,
    // Reorder-backlog high-water for THIS chrom, updated by the shard
    // collector. Stays zero on the single-reader fallback path.
    pending_gauge: Arc<PendingGauge>,
```

and add two fields to the `tracing::trace!` call, immediately after the `long_cap` line:

```rust
                    pending = pending_gauge.len_highwater.load(Ordering::Relaxed),
                    pending_bytes = pending_gauge.bytes_highwater.load(Ordering::Relaxed),
```

- [ ] **Step 7: Update the collector in `src/shard_exec.rs`**

Add a parameter to `pub fn run` after `worker_tids`:

```rust
    worker_tids: &Mutex<Vec<i32>>,
    pending_gauge: &crate::monitor::PendingGauge,
```

Track a running byte total beside the map (recomputing the sum each tick would
be O(n) per chunk). Replace the `let mut pending: HashMap<...> = HashMap::new();`
declaration at line 326 with:

```rust
        let mut pending: HashMap<(usize, usize), DenseChunk> = HashMap::new();
        // Running total of `pending`'s chunk bytes, maintained incrementally so
        // the gauge costs O(1) per insert/remove rather than O(|pending|).
        let mut pending_bytes: u64 = 0;
```

At every `pending.insert(key, chunk)` site, immediately before the insert:

```rust
                    pending_bytes += chunk.approx_bytes();
```

and immediately after the insert:

```rust
                    pending_gauge.observe(pending.len(), pending_bytes);
```

At every `pending.remove(&key)` site, after taking the chunk:

```rust
                    pending_bytes = pending_bytes.saturating_sub(chunk.approx_bytes());
```

- [ ] **Step 8: Add per-unit completion timing in `src/shard_exec.rs`**

Inside the worker loop, immediately before the unit's work begins:

```rust
                    let unit_start = std::time::Instant::now();
```

and immediately before the `tx_res.send(Msg::Done { ... })` call:

```rust
                        // Per-unit wall time. Shard skew is what distinguishes
                        // "too few readers" from "readers unevenly loaded", and
                        // it cannot be inferred from aggregate CPU.
                        tracing::trace!(
                            target: "genoray::monitor",
                            unit_ordinal = unit.ordinal,
                            unit_secs = unit_start.elapsed().as_secs_f64(),
                            "shard unit done"
                        );
```

- [ ] **Step 9: Wire the gauge and add the concurrency hook in `src/orchestrator.rs`**

After line 358 (`let shard_worker_tids: ... = Arc::new(Mutex::new(Vec::new()));`):

```rust
    let pending_gauge: Arc<crate::monitor::PendingGauge> =
        Arc::new(crate::monitor::PendingGauge::default());
```

Pass `Arc::clone(&pending_gauge)` as the new final argument to
`monitor::spawn_sampler` (line 364-370). Clone it into the per-chrom closure
beside `shard_worker_tids` (line 382), and pass `&pending_gauge` as the new
final argument at both `shard_exec::run` call sites (lines 506 and 696).

For the concurrency hook, alongside the existing bench hooks in the
`SourceSpec::Vcf` sharded branch (~line 423), override the planner's value:

```rust
    // BENCH-ONLY: override the planner's contig concurrency. Required to hold
    // TOTAL reader workers constant while varying how they are partitioned
    // across contigs -- `GENORAY_READER_WORKERS` alone cannot separate "too few
    // readers" from "readers on the wrong contig".
    let concurrent_chroms = bench_env("GENORAY_CONCURRENT_CHROMS")
        .unwrap_or(plan.concurrent_chroms)
        .max(1);
```

and use `concurrent_chroms` in place of `plan.concurrent_chroms` for the rest of
that branch.

- [ ] **Step 10: Verify the full Rust suite is green**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp/cargo-target
cargo fmt
cargo clippy --no-default-features --features conversion -- -D warnings
cargo test --no-default-features --features conversion
cargo check --no-default-features
```
Expected: fmt clean, clippy clean, all tests pass (310 unit + e2e suites), `check-core` builds. The final `cargo check --no-default-features` guards the query-core build that GenVarLoader links against — CI only exercises the conversion-on build.

- [ ] **Step 11: Verify the new trace fields actually appear**

```bash
maturin develop --release
GENORAY_LOG="genoray::monitor=trace" GENORAY_SAMPLE_INTERVAL=1 \
  python -m genoray._cli write vcf tests/data/*.vcf.gz "$CLAUDE_JOB_DIR/tmp/probe.svar" \
  --no-reference --overwrite --log-level info 2>&1 | grep -o "pending=[0-9]*" | head -3
```
Expected: at least one `pending=<n>` line. An empty result means the gauge is not wired to the sampler.

- [ ] **Step 12: Commit**

```bash
git add src/monitor.rs src/types.rs src/shard_exec.rs src/orchestrator.rs
git commit -m "feat(svar2): trace-level gauges for reorder backlog and shard skew"
```

---

### Task 3: Corpus generator

**Files:**
- Create: `scripts/bench_svar2/scale_corpus.py`
- Create: `tests/bench/test_corpus_smoke.py`

**Interfaces:**
- Consumes: `records.CorpusManifest`, `records.to_json`.
- Produces: `generate(out: Path, samples: int, variants: int, contigs: Sequence[str], format_fields: Sequence[str], seed: int, procs: int, bgzip_threads: int) -> CorpusManifest`; `size_corpus(samples: int, cells_budget: int) -> tuple[int, int]` returning `(variants, chunk_size)`; module-level pool worker `_format_block`.

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_corpus_smoke.py`:

```python
import subprocess

import pytest

from scripts.bench_svar2.records import CorpusManifest
from scripts.bench_svar2.scale_corpus import generate, size_corpus

pytestmark = pytest.mark.bench


def test_size_corpus_respects_production_chunk_clamp():
    """_auto_chunk_size never exceeds 25_000; measuring above it would
    characterize a regime production cannot reach."""
    variants, chunk_size = size_corpus(samples=250, cells_budget=1_400_000_000)
    assert variants == 5_600_000
    assert chunk_size == 25_000


def test_size_corpus_gives_at_least_32_chunks_at_large_S():
    variants, chunk_size = size_corpus(samples=500_000, cells_budget=1_400_000_000)
    assert variants == 2_800
    assert chunk_size == 87
    assert variants // chunk_size >= 32


def test_size_corpus_floors_chunk_size_at_64():
    _, chunk_size = size_corpus(samples=1_000_000, cells_budget=1_000_000)
    assert chunk_size == 64


def test_generate_is_deterministic(tmp_path):
    a = generate(tmp_path / "a.vcf.gz", samples=8, variants=200, contigs=["chr22"],
                 format_fields=(), seed=42, procs=2, bgzip_threads=1)
    b = generate(tmp_path / "b.vcf.gz", samples=8, variants=200, contigs=["chr22"],
                 format_fields=(), seed=42, procs=2, bgzip_threads=1)
    assert (tmp_path / "a.vcf.gz").read_bytes() == (tmp_path / "b.vcf.gz").read_bytes()
    assert a.cells == b.cells == 8 * 200


def test_generate_record_count_matches_manifest(tmp_path):
    """A truncated corpus must not silently yield fast, bogus timings."""
    m = generate(tmp_path / "c.vcf.gz", samples=4, variants=500, contigs=["chr21", "chr22"],
                 format_fields=(), seed=1, procs=2, bgzip_threads=1)
    n = int(subprocess.run(["bcftools", "index", "-n", str(tmp_path / "c.vcf.gz")],
                           capture_output=True, text=True, check=True).stdout.strip())
    assert n == m.variants


def test_generate_with_format_fields(tmp_path):
    m = generate(tmp_path / "d.vcf.gz", samples=4, variants=100, contigs=["chr22"],
                 format_fields=("DP", "GQ", "AD"), seed=3, procs=1, bgzip_threads=1)
    assert m.format_fields == ("DP", "GQ", "AD")
    hdr = subprocess.run(["bcftools", "view", "-h", str(tmp_path / "d.vcf.gz")],
                         capture_output=True, text=True, check=True).stdout
    for f in ("DP", "GQ", "AD"):
        assert f'##FORMAT=<ID={f},' in hdr


def test_generate_writes_manifest(tmp_path):
    generate(tmp_path / "e.vcf.gz", samples=4, variants=100, contigs=["chr22"],
             format_fields=(), seed=5, procs=1, bgzip_threads=1)
    assert (tmp_path / "e.manifest.json").exists()


def test_positions_are_sorted_and_unique(tmp_path):
    generate(tmp_path / "f.vcf.gz", samples=2, variants=1000, contigs=["chr22"],
             format_fields=(), seed=9, procs=2, bgzip_threads=1)
    out = subprocess.run(["bcftools", "query", "-f", "%POS\n", str(tmp_path / "f.vcf.gz")],
                         capture_output=True, text=True, check=True).stdout.split()
    pos = [int(x) for x in out]
    assert pos == sorted(pos)
    assert len(set(pos)) == len(pos)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/bench/test_corpus_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.bench_svar2.scale_corpus'`

- [ ] **Step 3: Write the implementation**

Create `scripts/bench_svar2/scale_corpus.py`:

```python
"""Deterministic synthetic VCF generation for the SVAR2 scale bench.

Generation cost is linear in genotype cells (~1.2M cells/s/process measured),
which is the binding constraint on how far up the sample axis the sweep can
reach. Record blocks are formatted in a process pool and streamed IN ORDER into
a single `bgzip` stdin: parallel formatting, one compression pass, one valid
BGZF stream, no temp files and bounded memory.

Run as a module so pool workers resolve `_format_block` by name under the
forkserver start method (Python 3.14's Linux default):

    python -m scripts.bench_svar2.scale_corpus --out corpus.vcf.gz --samples 1000 ...
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np

from scripts.bench_svar2.records import CorpusManifest, to_json

GENERATOR_VERSION = 1
PLOIDY = 2
BASES = np.array(["A", "C", "G", "T"])
GT_TOKENS = np.array(["0|0", "0|1", "1|0", "1|1"])
GT_WEIGHTS = np.array([0.72, 0.12, 0.12, 0.04])
BLOCK_VARIANTS = 2_000
DEFAULT_CONTIG_LEN = 50_818_468  # GRCh38 chr22
# Production reference: `_auto_chunk_size` clamps at 25_000 and `from_vcf`
# hardcodes exactly that.
MAX_CHUNK_SIZE = 25_000
MIN_CHUNK_SIZE = 64
MIN_CHUNKS = 32


def size_corpus(samples: int, cells_budget: int) -> tuple[int, int]:
    """Variants and chunk size for one scale point.

    Two constraints bind against each other: generation cost is linear in
    cells, and steady state needs enough chunks that fill/drain is a small
    fraction of the run. Fixing the cell budget and flooring the chunk count at
    32 resolves both. The upper clamp keeps small cohorts inside the regime
    production can actually reach.
    """
    variants = max(1, cells_budget // max(samples, 1))
    chunk_size = min(MAX_CHUNK_SIZE, max(MIN_CHUNK_SIZE, variants // MIN_CHUNKS))
    return variants, chunk_size


def _format_block(
    task: tuple[str, int, int],
    *,
    n_samples: int,
    n_format: int,
    seed: int,
) -> bytes:
    """Format one block of records. Module-level and keyword-bound via
    `partial` so forkserver workers can import it by name."""
    contig, block_index, n = task
    # Derive a per-block seed so output is independent of pool scheduling
    # order -- otherwise `procs` would change the bytes and break determinism.
    rng = np.random.default_rng([seed, hash(contig) % (2**31), block_index])
    pos = _block_positions(contig, block_index, n, seed)
    ref = rng.choice(BASES, size=n)
    alt_offset = rng.integers(1, 4, size=n)
    alt = BASES[(np.searchsorted(BASES, ref) + alt_offset) % 4]

    gts = rng.choice(GT_TOKENS, size=(n, n_samples), p=GT_WEIGHTS)
    if n_format:
        dp = rng.integers(1, 100, size=(n, n_samples))
        gq = rng.integers(1, 100, size=(n, n_samples))
        ad = rng.integers(0, 50, size=(n, n_samples))
        cells = np.char.add(np.char.add(gts, ":"), dp.astype(str))
        cells = np.char.add(np.char.add(cells, ":"), gq.astype(str))
        cells = np.char.add(np.char.add(cells, ":"), ad.astype(str))
        cells = np.char.add(np.char.add(cells, ","), (ad * 2).astype(str))
        fmt_key = "GT:DP:GQ:AD"
    else:
        cells = gts
        fmt_key = "GT"

    lines = []
    for i in range(n):
        lines.append(
            f"{contig}\t{pos[i]}\t.\t{ref[i]}\t{alt[i]}\t.\tPASS\t.\t{fmt_key}\t"
            + "\t".join(cells[i])
        )
    return ("\n".join(lines) + "\n").encode()


def _block_positions(contig: str, block_index: int, n: int, seed: int) -> np.ndarray:
    """Sorted unique positions for a block, in a disjoint stripe per block.

    Striping keeps blocks globally sorted without a cross-block sort, which is
    what lets blocks be formatted independently and still concatenate into a
    valid tabix-indexable VCF.
    """
    stride = DEFAULT_CONTIG_LEN // max(BLOCK_VARIANTS, n)
    lo = block_index * n * stride + 1
    rng = np.random.default_rng([seed, hash(contig) % (2**31), block_index, 99])
    offs = np.sort(rng.choice(np.arange(n * stride), size=n, replace=False))
    return lo + offs


def generate(
    out: Path,
    samples: int,
    variants: int,
    contigs: Sequence[str],
    format_fields: Sequence[str],
    seed: int,
    procs: int = 8,
    bgzip_threads: int = 4,
) -> CorpusManifest:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    contigs = list(contigs)
    n_format = len(format_fields)

    header = ["##fileformat=VCFv4.2"]
    for c in contigs:
        header.append(f"##contig=<ID={c},length={DEFAULT_CONTIG_LEN}>")
    header.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    if n_format:
        header.append('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">')
        header.append('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">')
        header.append('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic Depths">')
    sample_names = [f"S{i:06d}" for i in range(samples)]
    header.append(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names)
    )

    per_contig = variants // len(contigs)
    total = per_contig * len(contigs)
    tasks: list[tuple[str, int, int]] = []
    for c in contigs:
        remaining = per_contig
        bi = 0
        while remaining > 0:
            n = min(BLOCK_VARIANTS, remaining)
            tasks.append((c, bi, n))
            remaining -= n
            bi += 1

    with out.open("wb") as sink:
        bg = subprocess.Popen(
            ["bgzip", "-c", "-@", str(bgzip_threads)], stdin=subprocess.PIPE, stdout=sink
        )
        assert bg.stdin is not None
        bg.stdin.write(("\n".join(header) + "\n").encode())
        worker = partial(
            _format_block, n_samples=samples, n_format=n_format, seed=seed
        )
        if procs > 1:
            with mp.Pool(procs) as pool:
                # imap (not imap_unordered) -- VCF records must stay sorted.
                for blob in pool.imap(worker, tasks, chunksize=1):
                    bg.stdin.write(blob)
        else:
            for t in tasks:
                bg.stdin.write(worker(t))
        bg.stdin.close()
        if bg.wait() != 0:
            raise RuntimeError("bgzip failed")

    subprocess.run(["tabix", "-f", "-p", "vcf", str(out)], check=True)

    indexed = int(
        subprocess.run(
            ["bcftools", "index", "-n", str(out)], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    if indexed != total:
        raise RuntimeError(
            f"corpus truncated: index reports {indexed} records, expected {total}"
        )

    manifest = CorpusManifest(
        path=str(out),
        samples=samples,
        variants=total,
        contigs=tuple(contigs),
        format_fields=tuple(format_fields),
        ploidy=PLOIDY,
        cells=samples * total,
        compressed_bytes=out.stat().st_size,
        seed=seed,
        generator_version=GENERATOR_VERSION,
    )
    out.with_suffix("").with_suffix(".manifest.json").write_text(to_json(manifest))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--variants", type=int)
    p.add_argument("--cells-budget", type=int, default=1_400_000_000)
    p.add_argument("--contigs", type=str, default="chr22")
    p.add_argument("--format-fields", type=str, default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--bgzip-threads", type=int, default=4)
    a = p.parse_args()

    variants = a.variants
    if variants is None:
        variants, _ = size_corpus(a.samples, a.cells_budget)
    fields = tuple(f for f in a.format_fields.split(",") if f)
    m = generate(a.out, a.samples, variants, a.contigs.split(","), fields,
                 a.seed, a.procs, a.bgzip_threads)
    print(f"wrote {m.path}: {m.variants} variants x {m.samples} samples "
          f"= {m.cells} cells, {m.compressed_bytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/bench/test_corpus_smoke.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Verify the pool path works under the module entry point**

```bash
pixi run python -m scripts.bench_svar2.scale_corpus \
  --out "$CLAUDE_JOB_DIR/tmp/smoke.vcf.gz" --samples 100 --variants 5000 --procs 4
```
Expected: prints `wrote ...: 5000 variants x 100 samples = 500000 cells`. A `BrokenProcessPool` here means `_format_block` is not importable by name.

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_svar2/scale_corpus.py tests/bench/test_corpus_smoke.py
git commit -m "test(svar2): deterministic scale-corpus generator"
```

---

### Task 4: Law fitting and hypothesis verdict

Pure module: reads records, returns laws. No subprocesses, no I/O beyond NDJSON. This is where the spec's falsifiable criteria become code.

**Files:**
- Create: `scripts/bench_svar2/model.py`
- Create: `tests/bench/test_model.py`

**Interfaces:**
- Consumes: `records.{ProbeRecord, CorpusManifest, VLaw, CostLaw, RamLaw, Verdict}`.
- Produces: `fit_v_law(points) -> VLaw`; `fit_cost_law(name, samples, costs) -> CostLaw`; `fit_ram_law(rows) -> RamLaw`; `knee_from_probe(cpu_shard_pct, cpu_exec_pct) -> int`; `decide(knees, read_law, exec_law, rows) -> Verdict`; `extrapolate(v_law, read_law, exec_law, ram_law, samples, variants, chunk_size, workers, format_fields) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_model.py`:

```python
import math

import pytest

from scripts.bench_svar2.model import (
    decide,
    extrapolate,
    fit_cost_law,
    fit_ram_law,
    fit_v_law,
    knee_from_probe,
)

pytestmark = pytest.mark.bench


def test_v_law_recovers_planted_line():
    variants = [25_000, 50_000, 100_000, 200_000]
    walls = [1.0 + 1e-4 * v for v in variants]
    law = fit_v_law(list(zip(variants, walls)))
    assert law.r2 > 0.999
    assert math.isclose(law.slope_s_per_variant, 1e-4, rel_tol=1e-6)
    assert math.isclose(law.intercept_s, 1.0, abs_tol=1e-6)


def test_v_law_reports_low_r2_on_nonlinear_data():
    """If V-linearity fails, every downstream extrapolation is invalid and the
    harness must be able to say so rather than report a number."""
    variants = [25_000, 50_000, 100_000, 200_000]
    walls = [1e-8 * v**1.6 for v in variants]
    assert fit_v_law(list(zip(variants, walls))).r2 < 0.98


def test_cost_law_recovers_planted_exponent():
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    costs = [3.5 * s**0.8 for s in samples]
    law = fit_cost_law("read", samples, costs)
    assert math.isclose(law.beta, 0.8, rel_tol=1e-6)
    assert law.beta_ci95[0] < 0.8 < law.beta_ci95[1]


def test_knee_is_ratio_of_read_to_exec_cost():
    # 360% shard CPU against 60% exec CPU is a 6:1 cost ratio.
    assert knee_from_probe((360.0, 360.0), (60.0, 60.0)) == 6


def test_knee_floors_at_one():
    assert knee_from_probe((10.0,), (100.0,)) == 1


def test_ram_law_recovers_planted_slope():
    rows = [
        # (workers, pending_highwater, chunk_bytes, peak_rss_mb)
        (w, 0, 25_000_000, 100.0 + 3.0 * w * 25_000_000 / 1e6)
        for w in (1, 3, 5, 7, 11)
    ]
    law = fit_ram_law(rows)
    assert math.isclose(law.kappa, 3.0, rel_tol=1e-6)
    assert math.isclose(law.base_mb, 100.0, abs_tol=1e-6)


def test_decide_picks_h1_when_knee_is_flat():
    knees = {250: 5, 1_000: 5, 4_000: 5, 16_000: 6, 500_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, rows=[(5, 0, 1, 1.0)])
    assert v.hypothesis == "H1"


def test_decide_picks_h2_when_knee_trends():
    knees = {250: 3, 1_000: 5, 4_000: 7, 16_000: 11, 64_000: 17}
    samples = [250, 1_000, 4_000, 16_000, 64_000]
    read = fit_cost_law("read", samples, [3.0 * s**0.9 for s in samples])
    exec_ = fit_cost_law("exec", samples, [3.0 * s**0.5 for s in samples])
    v = decide(knees, read, exec_, rows=[(5, 0, 1, 1.0)])
    assert v.hypothesis == "H2"


def test_decide_picks_h3_when_pending_backlog_is_material():
    """Pending >= workers/2 means bytes, not worker count, set peak RSS."""
    knees = {250: 5, 1_000: 5, 4_000: 5}
    read = fit_cost_law("read", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", [250, 1_000, 4_000], [1.0, 1.0, 1.0])
    v = decide(knees, read, exec_, rows=[(8, 6, 1, 1.0)])
    assert v.hypothesis == "H3"


def test_decide_returns_none_when_nothing_is_supported():
    """Ambiguous data must not silently default to a hypothesis."""
    knees = {250: 3, 1_000: 9, 4_000: 2}
    samples = [250, 1_000, 4_000]
    read = fit_cost_law("read", samples, [1.0, 5.0, 0.5])
    exec_ = fit_cost_law("exec", samples, [1.0, 0.4, 3.0])
    v = decide(knees, read, exec_, rows=[(8, 0, 1, 1.0)])
    assert v.hypothesis == "none"


def test_extrapolate_flags_the_current_default_as_over_budget():
    """chunk_size=25_000 at 500k samples is ~3.1 GB of packed grid per chunk."""
    v_law = fit_v_law([(25_000, 1.0 + 1e-4 * 25_000), (200_000, 1.0 + 1e-4 * 200_000)])
    samples = [250, 1_000, 4_000]
    read = fit_cost_law("read", samples, [1.0, 1.0, 1.0])
    exec_ = fit_cost_law("exec", samples, [1.0, 1.0, 1.0])
    ram = fit_ram_law([(w, 0, 25_000_000, 100.0 + 3.0 * w * 25.0) for w in (1, 3, 5)])
    out = extrapolate(v_law, read, exec_, ram, samples=500_000, variants=1_000_000_000,
                      chunk_size=25_000, workers=1, format_fields=0)
    assert out["chunk_bytes"] > 3e9
    assert out["predicted_peak_rss_mb"] > 9_000
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/bench/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.bench_svar2.model'`

- [ ] **Step 3: Write the implementation**

Create `scripts/bench_svar2/model.py`:

```python
"""Fit the three scaling laws and return a falsifiable hypothesis verdict.

Pure: every function takes numbers and returns dataclasses. That is what makes
the verdict testable against planted synthetic laws rather than only against
cluster runs.

No scipy. With 5-7 scale points a normal approximation to the 95% CI is
materially wrong (t(5)=2.571 vs z=1.96), so a small Student-t table is
inlined instead of taking a dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from scripts.bench_svar2.records import CostLaw, RamLaw, VLaw, Verdict

# Two-tailed t at alpha=0.05, indexed by degrees of freedom.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
}
PLOIDY = 2
# Spec thresholds. Changing these changes the verdict, so they are named.
H1_KNEE_TOLERANCE = 1        # w* varies by less than +/-1 across the S range
H3_PENDING_FRACTION = 0.5    # pending_hw >= workers/2 makes bytes the invariant
V_LAW_MIN_R2 = 0.98


def _t95(df: int) -> float:
    if df <= 0:
        return float("inf")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def _linfit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float, float]:
    """Least squares. Returns (slope, intercept, r2, slope_stderr)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(xa)
    slope, intercept = np.polyfit(xa, ya, 1)
    pred = slope * xa + intercept
    ss_res = float(((ya - pred) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if n > 2 and ss_res > 0:
        sxx = float(((xa - xa.mean()) ** 2).sum())
        stderr = float(np.sqrt(ss_res / (n - 2) / sxx)) if sxx > 0 else 0.0
    else:
        stderr = 0.0
    return float(slope), float(intercept), r2, stderr


def fit_v_law(points: Sequence[tuple[float, float]]) -> VLaw:
    """points: (variants, phase1_s). Wall must be linear in variant count for
    the extrapolation to 10^9 variants to mean anything."""
    v = [p[0] for p in points]
    t = [p[1] for p in points]
    slope, intercept, r2, _ = _linfit(v, t)
    return VLaw(
        slope_s_per_variant=slope,
        intercept_s=intercept,
        r2=r2,
        n_points=len(points),
        max_extrapolation_factor=1e9 / max(v),
    )


def fit_cost_law(name: str, samples: Sequence[float], costs: Sequence[float]) -> CostLaw:
    """cost(S) = alpha * S**beta, fitted on logs."""
    beta, log_alpha, _, stderr = _linfit(np.log(samples), np.log(costs))
    half = _t95(len(samples) - 2) * stderr
    return CostLaw(
        name=name,
        alpha=float(np.exp(log_alpha)),
        beta=beta,
        beta_ci95=(beta - half, beta + half),
        n_points=len(samples),
    )


def fit_ram_law(rows: Sequence[tuple[int, int, int, float]]) -> RamLaw:
    """rows: (workers, pending_highwater, chunk_bytes, peak_rss_mb).

    peak_rss ~ base + kappa * (workers + pending_hw) * chunk_bytes. kappa is the
    observed overhead multiple over the analytic chunk size: a DenseChunk holds
    more than its packed grid.
    """
    x = [(w + p) * cb / 1e6 for (w, p, cb, _) in rows]
    y = [r[3] for r in rows]
    kappa, base, r2, _ = _linfit(x, y)
    return RamLaw(base_mb=base, kappa=kappa, r2=r2, n_points=len(rows))


def knee_from_probe(
    cpu_shard_pct: Sequence[float], cpu_exec_pct: Sequence[float]
) -> int:
    """Predicted knee from a single w=1 run.

    At w=1 the shard aggregate is one reader's cost and cpu_exec is the
    executor's, so their ratio is how many readers it takes to saturate the
    serial executor. Ticks where either is zero are startup/teardown and are
    dropped.
    """
    pairs = [(s, e) for s, e in zip(cpu_shard_pct, cpu_exec_pct) if s > 0 and e > 0]
    if not pairs:
        return 1
    shard = float(np.median([p[0] for p in pairs]))
    exec_ = float(np.median([p[1] for p in pairs]))
    if exec_ <= 0:
        return 1
    return max(1, int(np.ceil(shard / exec_)))


def decide(
    knees: dict[int, int],
    read_law: CostLaw,
    exec_law: CostLaw,
    rows: Sequence[tuple[int, int, int, float]],
    contig_counterfactual: tuple[float, float] | None = None,
) -> Verdict:
    """Apply the spec's falsifiable criteria, in H3-first order.

    H3 supersedes H1/H2: if bytes rather than worker count set peak RSS, or if
    the multi-contig regression is a partitioning artifact, a byte-bounded
    global pool needs no knee prediction at all.
    """
    evidence: dict[str, object] = {
        "knees": dict(knees),
        "beta_read": read_law.beta,
        "beta_exec": exec_law.beta,
    }

    max_pending_frac = max(
        (p / w for (w, p, _, _) in rows if w > 0), default=0.0
    )
    evidence["max_pending_fraction"] = max_pending_frac
    if max_pending_frac >= H3_PENDING_FRACTION:
        return Verdict("H3", (
            f"reorder backlog reached {max_pending_frac:.2f} x workers, so in-flight "
            "bytes rather than worker count set peak RSS"), evidence)

    if contig_counterfactual is not None:
        a, b = contig_counterfactual
        delta = abs(a - b) / max(min(a, b), 1e-9)
        evidence["contig_partition_delta"] = delta
        if delta > 0.15:
            return Verdict("H3", (
                f"same total readers split differently across contigs differ by "
                f"{delta:.0%} wall time: the multi-contig regression is a "
                "partitioning artifact"), evidence)

    diff_lo = read_law.beta_ci95[0] - exec_law.beta_ci95[1]
    diff_hi = read_law.beta_ci95[1] - exec_law.beta_ci95[0]
    evidence["beta_diff_ci95"] = (diff_lo, diff_hi)

    values = list(knees.values())
    spread = (max(values) - min(values)) if values else 0
    evidence["knee_spread"] = spread

    if spread <= H1_KNEE_TOLERANCE:
        return Verdict("H1", (
            f"w* varies by {spread} across the full sample range: a static cap "
            "suffices, no autotuner needed"), evidence)

    if diff_lo > 0 or diff_hi < 0:
        return Verdict("H2", (
            f"95% CI of (beta_read - beta_exec) = ({diff_lo:.3f}, {diff_hi:.3f}) "
            "excludes zero: the cost ratio genuinely trends with cohort size"), evidence)

    return Verdict("none", (
        f"w* spread is {spread} (> {H1_KNEE_TOLERANCE}, so not H1) but the "
        f"beta difference CI ({diff_lo:.3f}, {diff_hi:.3f}) includes zero (so not "
        "H2), and the backlog is immaterial (so not H3). Collect more points."), evidence)


def extrapolate(
    v_law: VLaw,
    read_law: CostLaw,
    exec_law: CostLaw,
    ram_law: RamLaw,
    samples: int,
    variants: int,
    chunk_size: int,
    workers: int,
    format_fields: int,
) -> dict[str, float]:
    """Project wall and peak RSS at a target regime.

    `v_law_r2` and `extrapolation_factor` ride along on every projection: the
    V-law is fitted over an 8x range and stretched to 10^9, which is the
    least-supported step in the chain and must not read as better evidenced
    than it is.
    """
    grid = (samples * PLOIDY) // 8
    fmt = format_fields * samples * 4
    chunk_bytes = chunk_size * (grid + fmt)
    predicted_wall = v_law.intercept_s + v_law.slope_s_per_variant * variants
    predicted_rss = ram_law.base_mb + ram_law.kappa * workers * chunk_bytes / 1e6
    return {
        "chunk_bytes": float(chunk_bytes),
        "predicted_wall_s": float(predicted_wall),
        "predicted_peak_rss_mb": float(predicted_rss),
        "predicted_knee": float(
            max(1, np.ceil(
                (read_law.alpha * samples**read_law.beta)
                / max(exec_law.alpha * samples**exec_law.beta, 1e-12)
            ))
        ),
        "v_law_r2": v_law.r2,
        "v_law_ok": float(v_law.r2 >= V_LAW_MIN_R2),
        "extrapolation_factor": variants / max(v_law.n_points, 1),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/bench/test_model.py -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_svar2/model.py tests/bench/test_model.py
git commit -m "test(svar2): scaling-law fitter and hypothesis verdict"
```

---

### Task 5: Instrumented single-run probe

**Files:**
- Create: `scripts/bench_svar2/probe.py`

**Interfaces:**
- Consumes: `records.{CorpusManifest, SweepPoint, ProbeRecord}`; the Task 2 trace fields `pending=`, `pending_bytes=`, `unit_secs=`.
- Produces: `run_point(point: SweepPoint, manifest: CorpusManifest, outdir: Path) -> ProbeRecord`; `digest(store: Path) -> str`; `parse_trace(text: str) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_probe.py`:

```python
import pytest

from scripts.bench_svar2.probe import parse_trace

pytestmark = pytest.mark.bench

SAMPLE = """
2026-07-28 INFO done: 1000 kept, 0 excluded (8.15s)
TRACE genoray::monitor: pipeline sampler chrom=chr22 elapsed_s=1 dense=0 dense_cap=6 sparse=0 sparse_cap=4 long=0 long_cap=2 pending=0 pending_bytes=0 cpu_read=0% cpu_shard=100% cpu_exec=60% cpu_cw=5% cpu_lw=1%
TRACE genoray::monitor: pipeline sampler chrom=chr22 elapsed_s=2 dense=5 dense_cap=6 sparse=1 sparse_cap=4 long=0 long_cap=2 pending=3 pending_bytes=78643200 cpu_read=0% cpu_shard=360% cpu_exec=50% cpu_cw=5% cpu_lw=1%
TRACE genoray::monitor: shard unit done unit_ordinal=0 unit_secs=1.5
TRACE genoray::monitor: shard unit done unit_ordinal=1 unit_secs=2.5
2026-07-28 INFO done: 500 kept, 0 excluded (2.00s)
"""


def test_parses_phase1_as_sum_of_per_contig_spans():
    assert parse_trace(SAMPLE)["phase1_s"] == pytest.approx(10.15)


def test_parses_dense_occupancy_and_cap():
    t = parse_trace(SAMPLE)
    assert t["dense_occupancy"] == (0, 5)
    assert t["dense_cap"] == 6


def test_parses_cpu_percentages_stripping_the_sign():
    t = parse_trace(SAMPLE)
    assert t["cpu_shard_pct"] == (100.0, 360.0)
    assert t["cpu_exec_pct"] == (60.0, 50.0)


def test_parses_pending_highwater_as_max_not_last():
    t = parse_trace(SAMPLE)
    assert t["pending_highwater"] == 3
    assert t["pending_bytes_highwater"] == 78_643_200


def test_parses_shard_unit_times():
    assert parse_trace(SAMPLE)["shard_unit_secs"] == (1.5, 2.5)


def test_handles_na_cpu_columns():
    """cpu_shard reads n/a on the single-reader fallback path."""
    line = ("TRACE genoray::monitor: pipeline sampler chrom=chr1 elapsed_s=1 dense=0 "
            "dense_cap=6 sparse=0 sparse_cap=4 long=0 long_cap=2 pending=0 pending_bytes=0 "
            "cpu_read=0% cpu_shard=n/a cpu_exec=10% cpu_cw=0% cpu_lw=0%")
    t = parse_trace(line)
    assert t["cpu_shard_pct"] == ()
    assert t["cpu_exec_pct"] == (10.0,)


def test_empty_input_yields_zeroed_trace():
    t = parse_trace("")
    assert t["phase1_s"] == 0.0
    assert t["pending_highwater"] == 0
    assert t["dense_occupancy"] == ()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/bench/test_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.bench_svar2.probe'`

- [ ] **Step 3: Write the implementation**

Create `scripts/bench_svar2/probe.py`:

```python
"""Run exactly one instrumented conversion and return one ProbeRecord.

The only module in the harness that shells out. Everything it learns comes from
two places: `os.wait4` rusage, and the `genoray::monitor` trace stream.
"""

from __future__ import annotations

import hashlib
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.bench_svar2.records import CorpusManifest, ProbeRecord, SweepPoint

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# "done: 1000 kept, 0 excluded (8.15s)" -- the per-contig phase-1 span. This is
# the ONLY span reader_workers can move; the rayon merge tail that follows is
# reader-independent, so total wall understates the reader-side effect.
RE_PHASE1 = re.compile(r"done:.*?\(([0-9.]+)s\)")
RE_SAMPLER = re.compile(r"pipeline sampler .*")
RE_UNIT = re.compile(r"shard unit done .*?unit_secs=([0-9.]+)")


def _field(line: str, key: str) -> str | None:
    m = re.search(rf"\b{key}=([^\s]+)", line)
    return m.group(1) if m else None


def parse_trace(text: str) -> dict:
    plain = ANSI.sub("", text)
    phase1 = sum(float(x) for x in RE_PHASE1.findall(plain))

    dense, shard, execp = [], [], []
    dense_cap = 0
    pending_hw = 0
    pending_bytes_hw = 0
    for line in RE_SAMPLER.findall(plain):
        d = _field(line, "dense")
        if d is not None:
            dense.append(int(d))
        c = _field(line, "dense_cap")
        if c is not None:
            dense_cap = max(dense_cap, int(c))
        p = _field(line, "pending")
        if p is not None:
            pending_hw = max(pending_hw, int(p))
        pb = _field(line, "pending_bytes")
        if pb is not None:
            pending_bytes_hw = max(pending_bytes_hw, int(pb))
        for key, sink in (("cpu_shard", shard), ("cpu_exec", execp)):
            v = _field(line, key)
            # `n/a` on the single-reader fallback path -- skip, do not zero,
            # or the median in `knee_from_probe` gets dragged down.
            if v is not None and v != "n/a":
                sink.append(float(v.rstrip("%")))

    return {
        "phase1_s": phase1,
        "dense_occupancy": tuple(dense),
        "dense_cap": dense_cap,
        "cpu_shard_pct": tuple(shard),
        "cpu_exec_pct": tuple(execp),
        "pending_highwater": pending_hw,
        "pending_bytes_highwater": pending_bytes_hw,
        "shard_unit_secs": tuple(float(x) for x in RE_UNIT.findall(plain)),
    }


def digest(store: Path) -> str:
    """Order-independent hash of every file in the .svar store -- the
    correctness oracle. Sharding is byte-identical, so this must not move
    across any configuration."""
    h = hashlib.sha256()
    for p in sorted(store.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _preexec(rss_ceiling_mb: int | None):
    if rss_ceiling_mb is None:
        return None

    def _limit() -> None:
        cap = rss_ceiling_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    return _limit


def run_point(
    point: SweepPoint, manifest: CorpusManifest, outdir: Path, warm: bool = True
) -> ProbeRecord:
    store = outdir / "bench.svar"
    env = dict(os.environ) | {
        "GENORAY_READER_WORKERS": str(point.reader_workers),
        "GENORAY_SHARD_HTSLIB": str(point.shard_htslib),
        "GENORAY_OVERSHARD": str(point.overshard),
        "GENORAY_LOG": "genoray::monitor=trace",
        "GENORAY_SAMPLE_INTERVAL": "1",
    }
    if point.concurrent_chroms is not None:
        env["GENORAY_CONCURRENT_CHROMS"] = str(point.concurrent_chroms)

    cmd = [
        sys.executable, "-m", "genoray._cli", "write", "vcf",
        manifest.path, str(store),
        "--no-reference", "--log-level", "info", "--overwrite",
        "-@", str(point.threads),
        "--chunk-size", str(point.chunk_size),
    ]

    best: ProbeRecord | None = None
    for rep in range(point.reps + (1 if warm else 0)):
        if store.exists():
            shutil.rmtree(store)
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=_preexec(point.rss_ceiling_mb),
        )
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - t0
        out = (proc.stdout.read().decode() if proc.stdout else "")
        err = (proc.stderr.read().decode() if proc.stderr else "")
        if warm and rep == 0:
            continue  # page-cache warm-up; measure inflate+parse CPU, not first-touch IO

        maxrss_mb = ru.ru_maxrss / 1024.0
        if status != 0:
            oom = maxrss_mb if point.rss_ceiling_mb else None
            # An OOM at a known ceiling is a legitimate datum: proving the
            # current chunk_size cannot survive biobank scale is a deliverable.
            return ProbeRecord(
                point_id=point.point_id, ok=False, wall_s=wall, phase1_s=0.0,
                cpu_s=ru.ru_utime + ru.ru_stime, maxrss_mb=maxrss_mb, digest="",
                dense_cap=0, dense_occupancy=(), cpu_shard_pct=(), cpu_exec_pct=(),
                pending_highwater=0, pending_bytes_highwater=0, shard_unit_secs=(),
                oom_at_rss_mb=oom, error=err[-2000:],
            )

        t = parse_trace(out + err)
        rec = ProbeRecord(
            point_id=point.point_id, ok=True, wall_s=wall,
            phase1_s=t["phase1_s"], cpu_s=ru.ru_utime + ru.ru_stime,
            maxrss_mb=maxrss_mb, digest=digest(store),
            dense_cap=t["dense_cap"], dense_occupancy=t["dense_occupancy"],
            cpu_shard_pct=t["cpu_shard_pct"], cpu_exec_pct=t["cpu_exec_pct"],
            pending_highwater=t["pending_highwater"],
            pending_bytes_highwater=t["pending_bytes_highwater"],
            shard_unit_secs=t["shard_unit_secs"],
        )
        # Min-of-N on wall time; the cluster is shared, so the minimum is the
        # least contended estimate.
        if best is None or rec.wall_s < best.wall_s:
            best = rec

    assert best is not None
    return best
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/bench/test_probe.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Confirm the `--chunk-size` flag behaves as the probe assumes**

The flag already exists (`python/genoray/_cli/__main__.py:114`, forwarded at
line 221 with `chunk_size if chunk_size is not None else 25_000`). Verify it is
surfaced and accepts the small values the large-`S` points need:

```bash
pixi run python -m genoray._cli write vcf --help | grep -- --chunk-size
pixi run python -m genoray._cli write vcf "$CLAUDE_JOB_DIR/tmp/smoke.vcf.gz" \
  "$CLAUDE_JOB_DIR/tmp/smoke.svar" --no-reference --overwrite --chunk-size 64
```
Expected: the flag is listed, and the conversion succeeds at `--chunk-size 64`.
A failure here means the large-`S` scale points cannot be measured at all, since
`size_corpus` floors chunk size at 64.

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_svar2/probe.py tests/bench/test_probe.py
git commit -m "test(svar2): instrumented single-run conversion probe"
```

---

### Task 6: Resumable sweep driver

**Files:**
- Create: `scripts/bench_svar2/sweep.py`
- Create: `tests/bench/test_sweep.py`

**Interfaces:**
- Consumes: `records.{SweepPoint, ProbeRecord, CorpusManifest, read_ndjson, append_ndjson, from_json}`; `probe.run_point`.
- Produces: `load_plan(path) -> list[SweepPoint]`; `pending_points(plan, results_path) -> list[SweepPoint]`; `run_sweep(plan_path, results_path, outdir, runner=probe.run_point) -> list[ProbeRecord]`; `check_oracle(records) -> str | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_sweep.py`:

```python
import json

import pytest

from scripts.bench_svar2.records import ProbeRecord, SweepPoint, append_ndjson
from scripts.bench_svar2.sweep import check_oracle, load_plan, pending_points, run_sweep

pytestmark = pytest.mark.bench


def _plan_file(tmp_path, n=3):
    pts = [
        {"corpus": str(tmp_path / "c.manifest.json"), "reader_workers": w,
         "concurrent_chroms": None, "shard_htslib": 0, "overshard": 4,
         "chunk_size": 25_000, "threads": 16, "reps": 1}
        for w in range(1, n + 1)
    ]
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(pts))
    return p


def _rec(pid: str, digest: str = "aaa", ok: bool = True) -> ProbeRecord:
    return ProbeRecord(
        point_id=pid, ok=ok, wall_s=1.0, phase1_s=1.0, cpu_s=1.0, maxrss_mb=1.0,
        digest=digest, dense_cap=6, dense_occupancy=(0,), cpu_shard_pct=(100.0,),
        cpu_exec_pct=(50.0,), pending_highwater=0, pending_bytes_highwater=0,
        shard_unit_secs=(1.0,),
    )


def test_load_plan_returns_sweep_points(tmp_path):
    pts = load_plan(_plan_file(tmp_path))
    assert len(pts) == 3
    assert all(isinstance(p, SweepPoint) for p in pts)


def test_pending_skips_already_recorded_points(tmp_path):
    plan = load_plan(_plan_file(tmp_path))
    results = tmp_path / "r.ndjson"
    append_ndjson(results, _rec(plan[0].point_id))
    remaining = pending_points(plan, results)
    assert [p.point_id for p in remaining] == [plan[1].point_id, plan[2].point_id]


def test_pending_returns_all_when_no_results_yet(tmp_path):
    plan = load_plan(_plan_file(tmp_path))
    assert len(pending_points(plan, tmp_path / "absent.ndjson")) == 3


def test_run_sweep_is_resumable(tmp_path):
    """A preempted Slurm job must resume, not restart."""
    plan_path = _plan_file(tmp_path)
    results = tmp_path / "r.ndjson"
    calls = []

    def runner(point, manifest, outdir, warm=True):
        calls.append(point.point_id)
        return _rec(point.point_id)

    (tmp_path / "c.manifest.json").write_text(json.dumps({
        "path": str(tmp_path / "c.vcf.gz"), "samples": 10, "variants": 100,
        "contigs": ["chr22"], "format_fields": [], "ploidy": 2, "cells": 1000,
        "compressed_bytes": 10, "seed": 1, "generator_version": 1}))

    run_sweep(plan_path, results, tmp_path / "out", runner=runner)
    assert len(calls) == 3
    calls.clear()
    run_sweep(plan_path, results, tmp_path / "out", runner=runner)
    assert calls == []


def test_check_oracle_flags_a_digest_mismatch():
    recs = [_rec("a", "aaa"), _rec("b", "bbb")]
    assert check_oracle(recs) is not None


def test_check_oracle_passes_when_all_digests_agree():
    assert check_oracle([_rec("a", "aaa"), _rec("b", "aaa")]) is None


def test_check_oracle_ignores_failed_runs():
    """An OOM datum has no digest and must not be read as a mismatch."""
    assert check_oracle([_rec("a", "aaa"), _rec("b", "", ok=False)]) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/bench/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.bench_svar2.sweep'`

- [ ] **Step 3: Write the implementation**

Create `scripts/bench_svar2/sweep.py`:

```python
"""Execute a plan of sweep points, resumably.

Holds no domain knowledge -- only execution, resumption and the oracle check.
A full sweep is an overnight job on a shared, preemptible cluster, so every
finished point is durably appended before the next one starts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from scripts.bench_svar2.records import (
    CorpusManifest,
    ProbeRecord,
    SweepPoint,
    append_ndjson,
    from_json,
    read_ndjson,
)


def load_plan(path: Path) -> list[SweepPoint]:
    raw = json.loads(Path(path).read_text())
    return [SweepPoint(**entry) for entry in raw]


def pending_points(plan: Sequence[SweepPoint], results_path: Path) -> list[SweepPoint]:
    done = {r.point_id for r in read_ndjson(results_path, ProbeRecord)}
    return [p for p in plan if p.point_id not in done]


def check_oracle(records: Sequence[ProbeRecord]) -> str | None:
    """Every successful configuration must produce a byte-identical store.
    Returns an error message, or None when all digests agree."""
    digests = {r.digest for r in records if r.ok and r.digest}
    if len(digests) > 1:
        return f"digest mismatch across configurations: {sorted(digests)}"
    return None


def run_sweep(
    plan_path: Path,
    results_path: Path,
    outdir: Path,
    runner: Callable[..., ProbeRecord] | None = None,
) -> list[ProbeRecord]:
    if runner is None:
        from scripts.bench_svar2.probe import run_point as runner  # noqa: PLW0127

    plan = load_plan(plan_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, CorpusManifest] = {}
    for point in pending_points(plan, results_path):
        if point.corpus not in manifests:
            manifests[point.corpus] = from_json(
                CorpusManifest, Path(point.corpus).read_text()
            )
        rec = runner(point, manifests[point.corpus], outdir)
        append_ndjson(results_path, rec)
        status = "OOM" if rec.oom_at_rss_mb else ("ok" if rec.ok else "FAIL")
        print(
            f"w={point.reader_workers:>3} cs={point.chunk_size:>6} "
            f"| wall {rec.wall_s:7.2f}s | phase1 {rec.phase1_s:6.2f}s "
            f"| rss {rec.maxrss_mb:7.0f}MB | pending_hw {rec.pending_highwater:>3} "
            f"| {status}",
            flush=True,
        )

    records = read_ndjson(results_path, ProbeRecord)
    problem = check_oracle(records)
    if problem:
        raise RuntimeError(problem)
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    recs = run_sweep(a.plan, a.results, a.outdir)
    print(f"{len(recs)} points recorded to {a.results}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/bench/test_sweep.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_svar2/sweep.py tests/bench/test_sweep.py
git commit -m "test(svar2): resumable sweep driver with oracle enforcement"
```

---

### Task 7: Fast regression tier

**Files:**
- Create: `scripts/bench_svar2/regression.py`
- Create: `scripts/bench_svar2/baselines/regression.json`
- Modify: `pixi.toml:69-74` (add `bench-regression` task)

**Interfaces:**
- Consumes: `scale_corpus.generate`, `probe.run_point`, `records.*`.
- Produces: `check(records, baselines, tolerance=0.25) -> list[str]`; `main()` exiting non-zero on regression.

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_regression.py`:

```python
import pytest

from scripts.bench_svar2.records import ProbeRecord
from scripts.bench_svar2.regression import check

pytestmark = pytest.mark.bench


def _rec(pid: str, wall: float, rss: float = 100.0) -> ProbeRecord:
    return ProbeRecord(
        point_id=pid, ok=True, wall_s=wall, phase1_s=wall, cpu_s=wall,
        maxrss_mb=rss, digest="aaa", dense_cap=6, dense_occupancy=(0,),
        cpu_shard_pct=(100.0,), cpu_exec_pct=(50.0,), pending_highwater=0,
        pending_bytes_highwater=0, shard_unit_secs=(1.0,),
    )


BASE = {"p0": {"wall_s": 10.0, "maxrss_mb": 100.0}}


def test_within_tolerance_reports_nothing():
    assert check([_rec("p0", 11.0)], BASE, tolerance=0.25) == []


def test_wall_regression_is_reported():
    problems = check([_rec("p0", 14.0)], BASE, tolerance=0.25)
    assert len(problems) == 1
    assert "wall_s" in problems[0]


def test_rss_regression_is_reported():
    problems = check([_rec("p0", 10.0, rss=200.0)], BASE, tolerance=0.25)
    assert any("maxrss_mb" in p for p in problems)


def test_improvement_is_not_a_regression():
    assert check([_rec("p0", 4.0, rss=10.0)], BASE, tolerance=0.25) == []


def test_missing_baseline_is_reported_not_silently_passed():
    problems = check([_rec("unknown", 10.0)], BASE, tolerance=0.25)
    assert any("no baseline" in p for p in problems)


def test_failed_run_is_always_a_regression():
    bad = ProbeRecord(**{**_rec("p0", 10.0).__dict__, "ok": False, "error": "boom"})
    assert check([bad], BASE, tolerance=0.25) != []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/bench/test_regression.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.bench_svar2.regression'`

- [ ] **Step 3: Write the implementation**

Create `scripts/bench_svar2/regression.py`:

```python
"""Fast regression tier: tiny corpora, committed baselines, ~2 minutes.

Guards the small-scale behaviour the cluster sweeps are too expensive to
re-run. Baselines are wall time and peak RSS at a handful of worker counts, and
a regression is a one-sided band -- getting faster is never a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts.bench_svar2.records import ProbeRecord, SweepPoint
from scripts.bench_svar2.scale_corpus import generate

BASELINE_PATH = Path(__file__).parent / "baselines" / "regression.json"
# Small enough to run in ~2 minutes on a laptop-class allocation.
CORPUS = {"samples": 200, "variants": 20_000, "contigs": ["chr22"], "seed": 1234}
WORKERS = (1, 3, 7)
DEFAULT_TOLERANCE = 0.25


def check(
    records: Sequence[ProbeRecord],
    baselines: dict[str, dict[str, float]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[str]:
    problems: list[str] = []
    for r in records:
        if not r.ok:
            problems.append(f"{r.point_id}: run failed ({r.error})")
            continue
        base = baselines.get(r.point_id)
        if base is None:
            problems.append(
                f"{r.point_id}: no baseline recorded -- regenerate with --record"
            )
            continue
        for metric in ("wall_s", "maxrss_mb"):
            got = getattr(r, metric)
            want = base[metric]
            if got > want * (1 + tolerance):
                problems.append(
                    f"{r.point_id}: {metric} regressed {got:.1f} vs baseline "
                    f"{want:.1f} (+{100 * (got / want - 1):.0f}%)"
                )
    return problems


def _points(manifest_path: Path, threads: int) -> list[SweepPoint]:
    return [
        SweepPoint(
            corpus=str(manifest_path), reader_workers=w, concurrent_chroms=None,
            shard_htslib=0, overshard=4, chunk_size=25_000, threads=threads, reps=2,
        )
        for w in WORKERS
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true", help="write baselines instead of checking")
    p.add_argument("--workdir", type=Path,
                   default=Path(os.environ.get("CLAUDE_JOB_DIR", ".")) / "tmp" / "bench_reg")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    a = p.parse_args()

    from scripts.bench_svar2.probe import run_point
    from scripts.bench_svar2.records import from_json, to_json
    from scripts.bench_svar2.records import CorpusManifest

    a.workdir.mkdir(parents=True, exist_ok=True)
    vcf = a.workdir / "reg.vcf.gz"
    if not vcf.exists():
        generate(vcf, format_fields=(), procs=4, bgzip_threads=2, **CORPUS)
    manifest_path = vcf.with_suffix("").with_suffix(".manifest.json")
    manifest = from_json(CorpusManifest, manifest_path.read_text())

    threads = len(os.sched_getaffinity(0))
    records = [run_point(pt, manifest, a.workdir) for pt in _points(manifest_path, threads)]

    if a.record:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(
            {r.point_id: {"wall_s": r.wall_s, "maxrss_mb": r.maxrss_mb} for r in records},
            indent=2, sort_keys=True))
        print(f"recorded {len(records)} baselines to {BASELINE_PATH}")
        return

    problems = check(records, json.loads(BASELINE_PATH.read_text()), a.tolerance)
    for msg in problems:
        print(f"REGRESSION: {msg}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"{len(records)} points within {a.tolerance:.0%} of baseline")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run pytest tests/bench/test_regression.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Add the pixi task**

In `pixi.toml`, under `[tasks]` (after line 74):

```toml
bench-regression = "python -m scripts.bench_svar2.regression"
bench-regression-record = "python -m scripts.bench_svar2.regression --record"
```

- [ ] **Step 6: Record the initial baselines and verify the check passes**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp/cargo-target
maturin develop --release
pixi run bench-regression-record
pixi run bench-regression
```
Expected: the record run writes `baselines/regression.json` with 3 entries; the check run prints `3 points within 25% of baseline` and exits 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/bench_svar2/regression.py scripts/bench_svar2/baselines tests/bench/test_regression.py pixi.toml
git commit -m "test(svar2): fast regression tier with committed baselines"
```

---

### Task 8: Sweep plans, Slurm driver, and README

The plans encode the spec's scale points. This task produces no new logic — it wires the harness into runnable artifacts.

**Files:**
- Create: `scripts/bench_svar2/plans/build_plans.py`
- Create: `scripts/bench_svar2/sweep_scale.sbatch`
- Create: `scripts/bench_svar2/README.md`

**Interfaces:**
- Consumes: `records.SweepPoint`, `scale_corpus.size_corpus`.
- Produces: `plans/scale.json`, `plans/contig.json`, `plans/holdout.json` (generated, not committed).

- [ ] **Step 1: Write the plan builder**

Create `scripts/bench_svar2/plans/build_plans.py`:

```python
"""Generate the sweep plans from the spec's scale points.

Generated rather than committed so a change to `size_corpus`'s rule cannot
silently disagree with a stale JSON file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from scripts.bench_svar2.records import SweepPoint
from scripts.bench_svar2.scale_corpus import size_corpus

CELLS_BUDGET = 1_400_000_000
SCALE_SAMPLES = (250, 1_000, 4_000, 16_000, 64_000, 250_000, 500_000)
# Validate the predicted knee against a real sweep at only three points.
KNEE_VALIDATION_SAMPLES = (250, 16_000, 500_000)
KNEE_WORKERS = (1, 2, 3, 5, 7, 11)
CONTIG_COUNTS = (1, 2, 8, 22)
HOLDOUT = {"samples": 100_000, "variants": 28_000, "format_fields": ("DP", "GQ", "AD")}


def _point(corpus: Path, workers: int, chunk_size: int, threads: int,
           concurrent: int | None = None) -> SweepPoint:
    return SweepPoint(
        corpus=str(corpus), reader_workers=workers, concurrent_chroms=concurrent,
        shard_htslib=0, overshard=4, chunk_size=chunk_size, threads=threads, reps=3,
        rss_ceiling_mb=60_000,
    )


def build(corpus_dir: Path, threads: int) -> dict[str, list[SweepPoint]]:
    scale, contig, holdout = [], [], []

    for s in SCALE_SAMPLES:
        _, cs = size_corpus(s, CELLS_BUDGET)
        corpus = corpus_dir / f"s{s}.manifest.json"
        # One w=1 run per point predicts the knee; only three points get swept.
        scale.append(_point(corpus, 1, cs, threads))
        if s in KNEE_VALIDATION_SAMPLES:
            for w in KNEE_WORKERS:
                if w != 1:
                    scale.append(_point(corpus, w, cs, threads))

    # Contig axis at fixed cohort: hold TOTAL readers constant (12) and vary the
    # split, which is what separates "too few readers" from "wrong contig".
    _, cs = size_corpus(4_000, CELLS_BUDGET)
    for c in CONTIG_COUNTS:
        corpus = corpus_dir / f"s4000_c{c}.manifest.json"
        for concurrent in (1, min(c, 4)):
            workers = max(1, 12 // concurrent)
            contig.append(_point(corpus, workers, cs, threads, concurrent=concurrent))

    holdout.append(_point(corpus_dir / "holdout.manifest.json", 1, 2_000, threads))

    return {"scale": scale, "contig": contig, "holdout": holdout}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=48)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, points in build(a.corpus_dir, a.threads).items():
        path = a.out_dir / f"{name}.json"
        path.write_text(json.dumps([dataclasses.asdict(pt) for pt in points], indent=2))
        print(f"{path}: {len(points)} points")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the plans build**

```bash
pixi run python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$CLAUDE_JOB_DIR/tmp/corpora" --out-dir "$CLAUDE_JOB_DIR/tmp/plans"
```
Expected: prints three lines; `scale.json` has 7 + 3*5 = 22 points.

- [ ] **Step 3: Write the Slurm driver**

Create `scripts/bench_svar2/sweep_scale.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=svar2-scale
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/bench-pr140-reader-workers
JD="${CLAUDE_JOB_DIR:?set CLAUDE_JOB_DIR}"/tmp
PX="pixi run --manifest-path $WT/pixi.toml"
CELLS=1400000000

echo "=== node $(hostname) nproc=$(nproc) allocated=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))') ==="

mkdir -p "$JD/corpora" "$JD/plans" "$JD/out"

# --- corpora -----------------------------------------------------------------
for S in 250 1000 4000 16000 64000 250000 500000; do
  OUT="$JD/corpora/s${S}.vcf.gz"
  [ -f "$OUT" ] && { echo "have $OUT"; continue; }
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples "$S" --cells-budget "$CELLS" --procs 16 --bgzip-threads 8
done

for C in 2 8 22; do
  OUT="$JD/corpora/s4000_c${C}.vcf.gz"
  [ -f "$OUT" ] && continue
  CONTIGS=$($PX python -c "print(','.join(f'chr{i}' for i in range(1, $C + 1)))")
  $PX python -m scripts.bench_svar2.scale_corpus \
    --out "$OUT" --samples 4000 --cells-budget "$CELLS" --contigs "$CONTIGS" \
    --procs 16 --bgzip-threads 8
done
cp "$JD/corpora/s4000.manifest.json" "$JD/corpora/s4000_c1.manifest.json"

# Hold-out: off-grid in S, unfitted in F, 2x the fitted cell count.
[ -f "$JD/corpora/holdout.vcf.gz" ] || $PX python -m scripts.bench_svar2.scale_corpus \
  --out "$JD/corpora/holdout.vcf.gz" --samples 100000 --variants 28000 \
  --format-fields DP,GQ,AD --procs 16 --bgzip-threads 8

# --- plans and sweeps --------------------------------------------------------
$PX python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$JD/corpora" --out-dir "$JD/plans" --threads 48

for NAME in scale contig holdout; do
  $PX python -m scripts.bench_svar2.sweep \
    --plan "$JD/plans/$NAME.json" \
    --results "$JD/out/$NAME.ndjson" \
    --outdir "$JD/out/$NAME"
done

echo "=== ALL DONE ==="
```

- [ ] **Step 4: Verify the script parses and the corpus commands are well-formed**

```bash
bash -n scripts/bench_svar2/sweep_scale.sbatch
```
Expected: no output (syntax OK). Do not submit it as part of this task — submission is a separate, deliberate act.

- [ ] **Step 5: Write the README**

Create `scripts/bench_svar2/README.md`:

```markdown
# SVAR2 conversion bench harness

Characterizes `SparseVar2.from_vcf`'s conversion pipeline from small cohorts to
biobank scale, and decides what a reader-budget autotuner should key on.

Design: `docs/superpowers/specs/2026-07-28-svar2-scale-bench-harness-design.md`

## Layout

| file | purpose |
|---|---|
| `records.py` | shared frozen record schema + NDJSON codecs |
| `scale_corpus.py` | deterministic seeded corpus generation |
| `probe.py` | one instrumented conversion run |
| `sweep.py` | resumable plan execution + digest oracle |
| `model.py` | law fitting, hypothesis verdict, extrapolation |
| `regression.py` | fast tier against committed baselines |
| `plans/build_plans.py` | generates the sweep plans from the spec's scale points |
| `sweep_scale.sbatch` | the overnight cluster job |
| `legacy_pr140/` | the original PR #140 sharded-reader harness and its findings |

## Running

```bash
pixi run bench-regression            # ~2 min, guards against regressions
sbatch scripts/bench_svar2/sweep_scale.sbatch   # overnight, full scale sweep
```

Corpora are regenerated from seeds, never committed: a 500,000-sample point is
~420 MB and its seed is 8 bytes.

## Gotchas

- Corpus and result files go under `$CLAUDE_JOB_DIR/tmp`. `/tmp` is reaped on
  this cluster and has destroyed corpora mid-run.
- Slurm hands out non-contiguous CPU ids; pin with `os.sched_getaffinity(0)`,
  never `taskset -c 0-15`.
- `pixi run test` does not rebuild the Rust extension. Run
  `maturin develop --release` before any Python-level verification of a Rust
  change, or you are timing stale code.
- Rust tests need `--no-default-features --features conversion`, and
  `CARGO_TARGET_DIR` must point off NFS.

## Prior findings

`legacy_pr140/README.md` records the PR #140 review: the knee sits at w≈3-7 and
moves with cohort size, not core count, because only the shard readers are
parallel — `executor::run_compute_engine` is a serial loop. That is the result
this harness extends to biobank scale.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_svar2/plans scripts/bench_svar2/sweep_scale.sbatch scripts/bench_svar2/README.md
git commit -m "test(svar2): scale sweep plans, Slurm driver, and harness README"
```

---

## Self-Review

**Spec coverage:**

| spec requirement | task |
|---|---|
| V-linearity verified, R² reported, invalid below 0.98 | 4 (`fit_v_law`, `V_LAW_MIN_R2`) |
| Knee predicted from one w=1 run, validated at 3 points | 4 (`knee_from_probe`), 8 (`KNEE_VALIDATION_SAMPLES`) |
| RAM law with in-flight and reorder-skew terms | 2 (gauge), 4 (`fit_ram_law`) |
| `pending` map made measurable | 2 |
| Per-shard completion times | 2 |
| `GENORAY_CONCURRENT_CHROMS` hook | 2 |
| Corpus sizing rule with 25,000 clamp and ≥32 chunks | 3 (`size_corpus`) |
| Parallel in-order generation into one BGZF stream | 3 |
| Hold-out point S=100k, F=3, V=28,000 | 8 |
| Digest oracle on every point, mismatch is hard failure | 5, 6 |
| Resumable NDJSON | 6 |
| OOM recorded as a datum | 5 (`rss_ceiling_mb` → `oom_at_rss_mb`) |
| Record counts asserted post-index | 3 |
| Falsifiable H1/H2/H3 criteria + "none" | 4 (`decide`) |
| Extrapolation with three policies | 4 (`extrapolate`) |
| Fast regression tier + pixi task | 7 |
| Directory rename, README moved | 1, 8 |
| pytest `bench` marker | 1 |

No gaps.

**Placeholder scan:** every step carries runnable code or an exact command. Task 5 Step 5 is a conditional (`--chunk-size` may not exist on the CLI) but states exactly what to do in both branches rather than deferring.

**Type consistency:** `ProbeRecord` field names are identical across Tasks 1, 5, 6 and 7. `parse_trace`'s returned keys match `ProbeRecord`'s constructor arguments exactly. `fit_ram_law` takes `(workers, pending_highwater, chunk_bytes, peak_rss_mb)` in Tasks 4's implementation, tests, and `decide`'s `rows` parameter. `size_corpus` returns `(variants, chunk_size)` in Tasks 3 and 8. `SweepPoint.point_id` is a property in Task 1 and consumed as such in Tasks 5, 6 and 7.

**Scope:** eight tasks, five waves, one subsystem. No decomposition needed.
