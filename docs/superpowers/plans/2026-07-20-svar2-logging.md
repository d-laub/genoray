# SVAR2 Write Logging & Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the SVAR2 write path structured `tracing`-based logging plus a `rich` progress bar (terminal + Jupyter) with a compact non-TTY heartbeat fallback, configurable via `progress`/`log_level` on the Python API and CLI.

**Architecture:** The Rust core emits `tracing` events and per-chunk progress into a single bounded `crossbeam` channel. Python spawns one drain thread (before entering Rust, which runs under `py.detach` with the GIL released) that reads events via a pyo3 `PyEventReceiver` and renders them with `rich` — a live `Progress` bar when interactive, compact heartbeat lines otherwise. A custom `tracing_subscriber` Layer converts `tracing` events into channel messages; a fallback stderr formatter (driven by `GENORAY_LOG`) covers pure-Rust runs.

**Tech Stack:** Rust (`tracing`, `tracing-subscriber`, `crossbeam-channel`, `pyo3` 0.29, edition 2024), Python 3.10+ (`rich`), `cyclopts` CLI, `maturin`/`pixi` build.

## Global Constraints

- **Rust edition 2024; pyo3 `0.29`** (`multiple-pymethods`). New pyclasses/pyfunctions register in the `#[pymodule]` at `src/lib.rs:1108` and MUST be `#[cfg(feature = "conversion")]` gated (mirroring the other conversion pyfunctions), so the no-conversion query-core build still compiles.
- **GIL release**: heavy work runs inside `py.detach(|| …)`. Rust worker threads push events with no GIL. The Python `recv` binding MUST release the GIL while blocking (`py.detach`).
- **Rust tests**: run `pixi run bash -lc 'cargo test --no-default-features --test <file>'` — the pyo3 test binary otherwise fails to link (`undefined symbol: _Py_Dealloc`), and a bare name filter silently passes 0 tests. Conversion-gated code (`#[cfg(feature = "conversion")]`) can't be unit-tested under `--no-default-features`; put those tests behind `#[cfg(feature = "conversion")]` and exercise them from Python instead.
- **Extension rebuild**: `pixi run test` does NOT rebuild the Rust `.so`. Run `pixi run bash -lc 'maturin develop --release'` before any Python-level verification of Rust changes (debug `.so` ≈79 MB, release ≈4 MB).
- **NFS linker**: if `cargo`/`prek` bus-errors on `target/`, `export CARGO_TARGET_DIR=/tmp/genoray-target-$USER` first.
- **Public API**: `progress` (now functional) and the new `log_level` kwarg on every write entry point, plus the CLI `--progress`/`--log-level` flags, are public — the same PR MUST update `skills/genoray-api/SKILL.md` (Task 14).
- **Output bytes unchanged**: `.svar` output must remain byte-identical; the existing conversion round-trip tests must still pass.
- **Conventional Commits**: every commit uses `feat:`/`fix:`/`refactor:`/`docs:`/`test:` etc. Do not edit `CHANGELOG.md` or bump versions by hand.
- **Log-level vocabulary** (fixed across Rust + Python + CLI): `"off" | "warning" | "info" | "debug"`. Default `"info"`. `progress` default `False`.
- **Event vocabulary** (fixed): `ContigStart { chrom, total: Option<u64> }`, `Progress { chrom, delta: u64 }`, `ContigDone { chrom, kept: u64, excluded: u64, elapsed_ms: u64 }`, `Log { level, chrom: Option<String>, message, target }`.

---

## File Structure

- **Create `src/logging.rs`** — `Event` enum, `EventSink` (buffered progress counter + `crossbeam::Sender<Event>`), the tracing bridge `Layer`, and `install_subscriber`/scoped-subscriber helpers.
- **Create `python/genoray/_logging.py`** — `log_level` → level string mapping, `GENORAY_LOG` handling, the `ProgressRenderer` (rich `Progress` vs. heartbeat), and `write_reporting(...)` context manager that owns the drain thread and the `PyEventReceiver`.
- **Modify `src/lib.rs`** — add `PyEventReceiver` `#[pyclass]` + a `new_event_channel()` pyfunction returning `(sender_token, receiver)`; add `log_level: String` / receiver params to the five conversion pyfunctions; register everything in the `#[pymodule]`.
- **Modify `src/orchestrator.rs`** — thread `EventSink` into `process_chromosome`; emit `ContigStart`/`ContigDone`; convert milestone `println!` to `tracing`.
- **Modify `src/executor.rs`** — increment the `EventSink` progress counter per record and flush `Progress` coarsely.
- **Modify `src/writer.rs`** — convert its two `println!` to `tracing::debug!`.
- **Modify `src/chunk_assembler.rs`** — emit `warn!`/`debug!` for excluded & normalized atoms.
- **Modify `src/normalize.rs` + contig-resolution call sites** — emit contig-name-resolution events from `ContigNormalizer`.
- **Modify `src/monitor.rs`** — demote the sampler’s `eprintln!`/`println!` to `tracing::trace!` under `target: "genoray::monitor"`.
- **Modify `python/genoray/_svar2.py`** — add `progress`/`log_level` to all write methods; wrap each `_core.*` call in `write_reporting(...)`.
- **Modify `python/genoray/_cli/__main__.py`** — add `--progress`/`--log-level` to the `write` subcommands.
- **Modify `Cargo.toml`** — add `tracing`, `tracing-subscriber`.
- **Modify `pyproject.toml`** — add `rich` as a direct dependency.
- **Modify `skills/genoray-api/SKILL.md`** — document the new kwargs/flags.
- **Create `tests/test_logging.py`** — Python-level end-to-end tests.
- **Create `src/logging_tests.rs`** (or `#[cfg(test)] mod tests` in `logging.rs`) — Rust unit tests for the non-gated pieces.

---

## Task 1: Add dependencies

**Files:**
- Modify: `Cargo.toml` (`[dependencies]`)
- Modify: `pyproject.toml` (`[project].dependencies`)

**Interfaces:**
- Consumes: nothing.
- Produces: `tracing`, `tracing-subscriber` crates available in Rust; `rich` importable in Python.

- [ ] **Step 1: Add Rust deps**

In `Cargo.toml` under `[dependencies]` (near `crossbeam-channel = "0.5.15"`):

```toml
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"] }
```

- [ ] **Step 2: Add Python dep**

In `pyproject.toml`, add to `[project].dependencies` (keep list alphomatically consistent with the existing entries):

```toml
"rich>=13",
```

- [ ] **Step 3: Verify both resolve/build**

Run: `pixi run bash -lc 'cargo check --no-default-features'`
Expected: compiles clean (new crates downloaded, no code using them yet).

Run: `pixi run bash -lc 'python -c "import rich; print(rich.__version__)"'`
Expected: prints a version ≥ 13.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock pyproject.toml
git commit -m "build: add tracing/tracing-subscriber (Rust) and rich (Python) deps"
```

---

## Task 2: Rust `Event` + `EventSink`

**Files:**
- Create: `src/logging.rs`
- Modify: `src/lib.rs` (add `mod logging;` near the other `mod` declarations, ~top of file)

**Interfaces:**
- Consumes: `crossbeam_channel`.
- Produces:
  - `pub enum Event { ContigStart { chrom: String, total: Option<u64> }, Progress { chrom: String, delta: u64 }, ContigDone { chrom: String, kept: u64, excluded: u64, elapsed_ms: u64 }, Log { level: LogLevel, chrom: Option<String>, message: String, target: String } }`
  - `pub enum LogLevel { Warning, Info, Debug }` with `impl LogLevel { pub fn as_str(&self) -> &'static str }`
  - `#[derive(Clone)] pub struct EventSink { … }` with:
    - `pub fn new(tx: crossbeam_channel::Sender<Event>, flush_every: u64) -> Self`
    - `pub fn disabled() -> Self` (no channel; all methods are no-ops — used when Python passed no receiver)
    - `pub fn contig_start(&self, chrom: &str, total: Option<u64>)`
    - `pub fn tick(&self, chrom: &str, n: u64)` (buffers; emits a `Progress` every `flush_every`)
    - `pub fn flush(&self, chrom: &str)` (emit any buffered remainder)
    - `pub fn contig_done(&self, chrom: &str, kept: u64, excluded: u64, elapsed_ms: u64)`
    - `pub fn send_log(&self, level: LogLevel, chrom: Option<&str>, target: &str, message: String)`

- [ ] **Step 1: Write the failing test**

Create `src/logging.rs` with a test module at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    #[test]
    fn tick_flushes_on_threshold_and_remainder() {
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 100);
        sink.contig_start("chr1", Some(250));
        sink.tick("chr1", 60);
        sink.tick("chr1", 60); // crosses 100 -> emits Progress{delta:120}
        sink.flush("chr1"); // remainder 0 already flushed? remainder=20 -> emit 20
        sink.contig_done("chr1", 230, 20, 1234);

        let evs: Vec<Event> = rx.try_iter().collect();
        assert!(matches!(evs[0], Event::ContigStart { total: Some(250), .. }));
        // one Progress of 120, then remainder 20
        let deltas: Vec<u64> = evs.iter().filter_map(|e| match e {
            Event::Progress { delta, .. } => Some(*delta),
            _ => None,
        }).collect();
        assert_eq!(deltas.iter().sum::<u64>(), 120 + 20);
        assert!(matches!(evs.last().unwrap(),
            Event::ContigDone { kept: 230, excluded: 20, elapsed_ms: 1234, .. }));
    }

    #[test]
    fn disabled_sink_is_silent() {
        let sink = EventSink::disabled();
        sink.tick("chr1", 10);
        sink.contig_done("chr1", 1, 0, 1); // must not panic
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features --lib logging::'`
Expected: FAIL (types/methods not defined).

- [ ] **Step 3: Implement `Event`, `LogLevel`, `EventSink`**

Top of `src/logging.rs`:

```rust
//! SVAR2 write-path logging & progress events bridged to Python.
use crossbeam_channel::Sender;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel { Warning, Info, Debug }

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self { LogLevel::Warning => "warning", LogLevel::Info => "info", LogLevel::Debug => "debug" }
    }
}

#[derive(Debug)]
pub enum Event {
    ContigStart { chrom: String, total: Option<u64> },
    Progress { chrom: String, delta: u64 },
    ContigDone { chrom: String, kept: u64, excluded: u64, elapsed_ms: u64 },
    Log { level: LogLevel, chrom: Option<String>, message: String, target: String },
}

#[derive(Clone)]
pub struct EventSink {
    inner: Option<Arc<SinkInner>>,
}

struct SinkInner {
    tx: Sender<Event>,
    pending: AtomicU64,
    flush_every: u64,
}

impl EventSink {
    pub fn new(tx: Sender<Event>, flush_every: u64) -> Self {
        EventSink { inner: Some(Arc::new(SinkInner {
            tx, pending: AtomicU64::new(0), flush_every: flush_every.max(1),
        })) }
    }
    pub fn disabled() -> Self { EventSink { inner: None } }

    pub fn contig_start(&self, chrom: &str, total: Option<u64>) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::ContigStart { chrom: chrom.to_string(), total });
        }
    }

    pub fn tick(&self, chrom: &str, n: u64) {
        if let Some(i) = &self.inner {
            let prev = i.pending.fetch_add(n, Ordering::Relaxed) + n;
            if prev >= i.flush_every {
                // Take whatever is currently buffered and emit it.
                let take = i.pending.swap(0, Ordering::Relaxed);
                if take > 0 {
                    let _ = i.tx.send(Event::Progress { chrom: chrom.to_string(), delta: take });
                }
            }
        }
    }

    pub fn flush(&self, chrom: &str) {
        if let Some(i) = &self.inner {
            let take = i.pending.swap(0, Ordering::Relaxed);
            if take > 0 {
                let _ = i.tx.send(Event::Progress { chrom: chrom.to_string(), delta: take });
            }
        }
    }

    pub fn contig_done(&self, chrom: &str, kept: u64, excluded: u64, elapsed_ms: u64) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::ContigDone {
                chrom: chrom.to_string(), kept, excluded, elapsed_ms,
            });
        }
    }

    pub fn send_log(&self, level: LogLevel, chrom: Option<&str>, target: &str, message: String) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::Log {
                level, chrom: chrom.map(|s| s.to_string()), message, target: target.to_string(),
            });
        }
    }
}
```

Add `mod logging;` in `src/lib.rs` alongside the other module declarations.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features --lib logging::'`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/logging.rs src/lib.rs
git commit -m "feat(logging): add Event enum and buffered EventSink"
```

---

## Task 3: tracing bridge Layer + subscriber install

**Files:**
- Modify: `src/logging.rs`

**Interfaces:**
- Consumes: `EventSink`, `tracing`, `tracing_subscriber`.
- Produces:
  - `pub fn level_from_str(s: &str) -> Option<tracing::level_filters::LevelFilter>` mapping `"off"/"warning"/"info"/"debug"`.
  - `pub struct ChannelLayer { sink: EventSink }` implementing `tracing_subscriber::Layer<S>` that converts each event into `EventSink::send_log`, reading the event's `message` field and its `chrom` field if present.
  - `pub fn with_channel_subscriber<R>(sink: EventSink, level: &str, f: impl FnOnce() -> R) -> R` — sets a **scoped** subscriber (`tracing::subscriber::with_default`) for the duration of `f`, so concurrent/repeated writes don't fight a global default.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/logging.rs`:

```rust
#[test]
fn channel_layer_routes_events_at_level() {
    use crossbeam_channel::unbounded;
    let (tx, rx) = unbounded();
    let sink = EventSink::new(tx, 1);
    with_channel_subscriber(sink, "info", || {
        tracing::info!(chrom = "chr1", "excluded 12 records");
        tracing::debug!(chrom = "chr1", "chr1:100 REF mismatch"); // filtered out at info
    });
    let logs: Vec<(String, String)> = rx.try_iter().filter_map(|e| match e {
        Event::Log { chrom, message, .. } => Some((chrom.unwrap_or_default(), message)),
        _ => None,
    }).collect();
    assert_eq!(logs.len(), 1);
    assert_eq!(logs[0].0, "chr1");
    assert!(logs[0].1.contains("excluded 12"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run bash -lc 'cargo test --no-default-features --lib logging::channel_layer'`
Expected: FAIL (`with_channel_subscriber` undefined).

- [ ] **Step 3: Implement the Layer + install helpers**

Add to `src/logging.rs`:

```rust
use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing::field::{Field, Visit};

pub fn level_from_str(s: &str) -> Option<LevelFilter> {
    match s {
        "off" => Some(LevelFilter::OFF),
        "warning" => Some(LevelFilter::WARN),
        "info" => Some(LevelFilter::INFO),
        "debug" => Some(LevelFilter::DEBUG),
        _ => None,
    }
}

fn to_log_level(l: &tracing::Level) -> LogLevel {
    match *l {
        tracing::Level::ERROR | tracing::Level::WARN => LogLevel::Warning,
        tracing::Level::INFO => LogLevel::Info,
        _ => LogLevel::Debug,
    }
}

#[derive(Default)]
struct FieldGrab { message: String, chrom: Option<String> }

impl Visit for FieldGrab {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "chrom" { self.chrom = Some(value.to_string()); }
        else if field.name() == "message" { self.message = value.to_string(); }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let v = format!("{value:?}");
        if field.name() == "chrom" { self.chrom = Some(v.trim_matches('"').to_string()); }
        else if field.name() == "message" { self.message = v; }
    }
}

pub struct ChannelLayer { sink: EventSink }
impl ChannelLayer { pub fn new(sink: EventSink) -> Self { Self { sink } } }

impl<S: tracing::Subscriber> Layer<S> for ChannelLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut g = FieldGrab::default();
        event.record(&mut g);
        self.sink.send_log(
            to_log_level(event.metadata().level()),
            g.chrom.as_deref(),
            event.metadata().target(),
            g.message,
        );
    }
}

pub fn with_channel_subscriber<R>(sink: EventSink, level: &str, f: impl FnOnce() -> R) -> R {
    let filter = level_from_str(level).unwrap_or(LevelFilter::INFO);
    let subscriber = tracing_subscriber::registry()
        .with(ChannelLayer::new(sink).with_filter(filter));
    tracing::subscriber::with_default(subscriber, f)
}
```

- [ ] **Step 4: Add the pure-Rust fmt fallback (GENORAY_LOG)**

For runs with no Python consumer (the `bench_from_vcf_list` binary, `cargo`
runs), tracing should write compact stderr lines gated by `GENORAY_LOG`
(`RUST_LOG`-style directives). Add to `src/logging.rs`:

```rust
use tracing_subscriber::EnvFilter;
use std::sync::Once;

static FMT_INIT: Once = Once::new();

/// Install a global stderr fmt subscriber driven by `GENORAY_LOG`, at most once
/// per process. Called by pure-Rust entry points (bench bin) — NOT by the
/// Python pipeline, which uses `with_channel_subscriber` instead.
pub fn install_fmt_fallback() {
    FMT_INIT.call_once(|| {
        let filter = EnvFilter::try_from_env("GENORAY_LOG")
            .unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .compact()
            .with_writer(std::io::stderr)
            .try_init();
    });
}
```

Call `crate::logging::install_fmt_fallback();` at the top of
`src/bin/bench_from_vcf_list.rs`’s `main` so bench runs get stderr logs.

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run bash -lc 'cargo test --no-default-features --lib logging::'`
Expected: PASS (all logging tests). Then `pixi run bash -lc 'cargo build --bin bench_from_vcf_list'` — expected: compiles.

- [ ] **Step 6: Commit**

```bash
git add src/logging.rs src/bin/bench_from_vcf_list.rs
git commit -m "feat(logging): tracing ChannelLayer, scoped subscriber, GENORAY_LOG fmt fallback"
```

---

## Task 4: `PyEventReceiver` + `new_event_channel` binding

**Files:**
- Modify: `src/lib.rs` (add `#[pyclass]` + `#[pyfunction]`, register in `#[pymodule]` at `src/lib.rs:1108`)

**Interfaces:**
- Consumes: `logging::{Event, EventSink}`.
- Produces (Python-visible under `genoray._core`):
  - `new_event_channel(flush_every: int) -> tuple[int, PyEventReceiver]` — returns an opaque `sender_token` (a boxed `EventSink` pointer id) and a receiver. *(Simpler alternative used here: `new_event_channel` returns just the `PyEventReceiver`, and the sink is created Rust-side inside each pipeline call from a receiver handle — see note.)*
  - `PyEventReceiver.recv_timeout(millis: int) -> tuple | None` — releases the GIL while blocking; returns a decoded event tuple or `None` on timeout; raises `StopIteration` when the channel is disconnected.

**Note on wiring:** To avoid passing raw pointers across FFI, the channel is created *inside* each conversion pyfunction: the pyfunction accepts a `log_level: String` plus a mutable `PyEventReceiver` to publish into. Concretely, `PyEventReceiver` owns `(Sender<Event>, Receiver<Event>)`; the pyfunction clones the `Sender` into an `EventSink`. This keeps all channel ownership in the Rust object the Python side already holds.

- [ ] **Step 1: Write the failing test (Python smoke)**

Create `tests/test_logging.py`:

```python
import genoray._core as core

def test_event_channel_roundtrip():
    rx = core.PyEventReceiver(flush_every=1)
    # No producer yet: recv_timeout returns None promptly.
    assert rx.recv_timeout(1) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run bash -lc 'maturin develop --release && pytest tests/test_logging.py::test_event_channel_roundtrip -v'`
Expected: FAIL (`PyEventReceiver` missing).

- [ ] **Step 3: Implement the pyclass**

Add to `src/lib.rs` (gated with the conversion feature, near the other pyclasses):

```rust
#[cfg(feature = "conversion")]
#[pyclass]
pub struct PyEventReceiver {
    tx: crossbeam_channel::Sender<crate::logging::Event>,
    rx: crossbeam_channel::Receiver<crate::logging::Event>,
}

#[cfg(feature = "conversion")]
#[pymethods]
impl PyEventReceiver {
    #[new]
    #[pyo3(signature = (flush_every = 25_000))]
    fn new(flush_every: u64) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let _ = flush_every; // flush_every consumed when the pipeline builds its EventSink
        PyEventReceiver { tx, rx }
    }

    /// Block up to `millis` for the next event. GIL is released while parked.
    /// Returns a decoded tuple, or None on timeout. Raises StopIteration when
    /// all senders have dropped (channel disconnected).
    fn recv_timeout(&self, py: Python, millis: u64) -> PyResult<Option<Py<PyAny>>> {
        use crossbeam_channel::RecvTimeoutError;
        let res = py.detach(|| self.rx.recv_timeout(std::time::Duration::from_millis(millis)));
        match res {
            Ok(ev) => Ok(Some(encode_event(py, ev)?)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                Err(pyo3::exceptions::PyStopIteration::new_err("event channel closed"))
            }
        }
    }
}

#[cfg(feature = "conversion")]
impl PyEventReceiver {
    /// Build an EventSink that publishes into this receiver's channel.
    pub fn sink(&self, flush_every: u64) -> crate::logging::EventSink {
        crate::logging::EventSink::new(self.tx.clone(), flush_every)
    }
    /// Drop the internal keep-alive sender so the drain side sees disconnect
    /// once all pipeline senders drop. Called at end of each pyfunction.
    pub fn tx_clone(&self) -> crossbeam_channel::Sender<crate::logging::Event> { self.tx.clone() }
}

#[cfg(feature = "conversion")]
fn encode_event(py: Python, ev: crate::logging::Event) -> PyResult<Py<PyAny>> {
    use crate::logging::Event::*;
    let t = match ev {
        ContigStart { chrom, total } => ("contig_start", chrom, total, py.None(), py.None()).into_pyobject(py)?.into_any().unbind(),
        Progress { chrom, delta } => ("progress", chrom, Some(delta), py.None(), py.None()).into_pyobject(py)?.into_any().unbind(),
        ContigDone { chrom, kept, excluded, elapsed_ms } => {
            ("contig_done", chrom, Some(kept), Some(excluded), Some(elapsed_ms)).into_pyobject(py)?.into_any().unbind()
        }
        Log { level, chrom, message, target } => {
            ("log", level.as_str().to_string(), chrom, message, target).into_pyobject(py)?.into_any().unbind()
        }
    };
    Ok(t)
}
```

Register in the `#[pymodule]` (with the other `add_class` lines around `src/lib.rs:1124`):

```rust
m.add_class::<PyEventReceiver>()?;
```

> Implementer note: the exact `into_pyobject` shape may need small adjustments for pyo3 0.29 tuple conversion (heterogeneous `Option<u64>`). If the mixed-tuple conversion is awkward, build a `PyTuple` explicitly via `PyTuple::new(py, [...])` with each element converted through `.into_pyobject(py)?`. The Python side only relies on the tuple *layout* documented above: `(tag, str, int|None, int|None|str, int|None|str)`.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run bash -lc 'maturin develop --release && pytest tests/test_logging.py::test_event_channel_roundtrip -v'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs tests/test_logging.py
git commit -m "feat(logging): PyEventReceiver binding with GIL-releasing recv_timeout"
```

---

## Task 5: Thread `EventSink` through the pipeline + emit progress

**Files:**
- Modify: `src/orchestrator.rs` (`process_chromosome` at `:261`, `run_conversion_pipeline` driver, `run_*_conversion_pipeline` cohort loops, milestone `println!`s)
- Modify: `src/executor.rs` (per-record tick)
- Modify: `src/writer.rs` (`:58`, `:89` `println!` → `tracing::debug!`)
- Modify: `src/lib.rs` (add `log_level`/receiver params to the 5 pyfunctions — see Task 11/12 for Python wiring; here add the Rust params + build the sink)

**Interfaces:**
- Consumes: `logging::{EventSink, LogLevel, with_channel_subscriber}`, `PyEventReceiver::sink`.
- Produces: `process_chromosome(..., sink: &crate::logging::EventSink)` gains a trailing `sink` parameter; each pyfunction gains `receiver: Option<&PyEventReceiver>` and `log_level: String`, wraps the `py.detach` body in `with_channel_subscriber`, and builds an `EventSink` (or `EventSink::disabled()` when `receiver` is `None`).

- [ ] **Step 1: Add `sink` param + emit ContigStart/Done (compile-first, then test via Python in Task 11)**

In `src/orchestrator.rs`, change the signature at `:261`:

```rust
pub fn process_chromosome(
    source: SourceSpec,
    fasta_path: Option<&str>,
    chrom: &str,
    base_out_dir: &str,
    samples: &[&str],
    chunk_size: usize,
    ploidy: usize,
    long_allele_capacity: usize,
    skip_out_of_scope: bool,
    check_ref: crate::normalize::CheckRef,
    processing_threads: usize,
    signatures: bool,
    fields: &[crate::field::FieldSpec],
    sink: &crate::logging::EventSink,
) -> Result<u64, ConversionError> {
```

At the top of the function body, after `chrom` is known, add:

```rust
let contig_started = std::time::Instant::now();
sink.contig_start(chrom, None); // streaming: total unknown
```

Just before the successful return (where the per-chrom result/`dropped` count is known), add:

```rust
sink.flush(chrom);
// `kept` = records emitted; `excluded` = ref-excluded + dropped. Use the
// counters already tracked in this function (ref_excluded, dropped). If a
// single `kept` counter is not already present, sum chunk variant counts as
// they are produced (add a local `let mut kept_total = 0u64;` incremented in
// the executor callback — see Step 2).
sink.contig_done(
    chrom,
    kept_total,
    ref_excluded_total + dropped,
    contig_started.elapsed().as_millis() as u64,
);
```

- [ ] **Step 2: Emit per-record progress in the executor**

In `src/executor.rs`, locate the loop that consumes decoded records/atoms from the reader (the stage that feeds `DenseChunk`s). Thread the `&EventSink` in and call, once per record (or once per produced chunk with the chunk’s variant count):

```rust
sink.tick(chrom, n_variants_in_this_unit as u64);
kept_total += n_variants_in_this_unit as u64;
```

If per-record is too granular, prefer per-chunk: emit `sink.tick(chrom, chunk_len)` right after a `DenseChunk` is assembled. The `EventSink` buffers and only forwards a `Progress` every `flush_every` records, so per-record calls are cheap.

- [ ] **Step 3: Replace milestone `println!` with tracing**

In `src/orchestrator.rs` convert (keep the message text, add `chrom` where available):

```rust
// was: println!("[{}] Phase 1 Complete. ...", chrom);
tracing::debug!(chrom = %chrom, "Phase 1 complete; triggering in-memory merge");
// was: println!("==> Processing {}", chrom);
tracing::info!(chrom = %chrom, "processing contig");
// was: println!("Cohort Processing Complete.");
tracing::info!("cohort processing complete");
// was: the Pipeline Config / thread-notice println!s
tracing::info!(threads = processing_threads, "pipeline configured");
```

In `src/writer.rs`:

```rust
// :58  -> tracing::debug!("writer thread: all chunks committed");
// :89  -> tracing::debug!(chrom = %chrom_label, "long-allele writer: buffers committed");
```

In `report_ref_excluded` (orchestrator), replace the `println!` with:

```rust
if ref_excluded > 0 {
    tracing::info!(chrom = %chrom, excluded = ref_excluded,
        "check_ref=x: excluded records whose REF disagreed with the reference FASTA");
}
```

- [ ] **Step 4: Update all `process_chromosome` call sites + pyfunctions**

In `src/lib.rs`, for each of `run_conversion_pipeline`, `run_pgen_conversion_pipeline`, `run_vcf_list_conversion_pipeline`, `run_svar1_conversion_pipeline`:

1. Add params to the `#[pyo3(signature = (...))]` and the fn signature: `log_level: String` and `receiver: Option<Py<PyEventReceiver>>` (accept `None`).
2. Before `py.detach`, build the sink and borrow a `&str` level:

```rust
let sink = match &receiver {
    Some(r) => r.borrow(py).sink(chunk_size as u64),
    None => crate::logging::EventSink::disabled(),
};
let level = log_level.clone();
```

3. Wrap the existing `py.detach(|| { ... })` body so tracing is active:

```rust
let results = py.detach(|| {
    crate::logging::with_channel_subscriber(sink.clone(), &level, || {
        // ... existing body, passing &sink into process_chromosome(...) ...
    })
});
```

4. Pass `&sink` as the new trailing arg at each `process_chromosome(...)` call.

- [ ] **Step 5: Compile**

Run: `pixi run bash -lc 'cargo check'` (default features / conversion on)
Expected: compiles. Then `pixi run bash -lc 'cargo check --no-default-features'` — expected: compiles (the `#[cfg(feature="conversion")]` gates keep `PyEventReceiver` out of the query-core build).

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator.rs src/executor.rs src/writer.rs src/lib.rs
git commit -m "feat(logging): emit contig/progress events and route milestones through tracing"
```

---

## Task 6: Edge-case events — excluded & normalized atoms

**Files:**
- Modify: `src/chunk_assembler.rs` (the `apply_check_ref` / `RefDecision::Exclude` path ~`:309`, and the `left_align`/atomize path ~`:335`)

**Interfaces:**
- Consumes: `tracing`.
- Produces: `debug!` per-event and (via the summary in Task 5) `info!` totals. The assembler must have access to `chrom`; if it does not already, pass `chrom: &str` down from `process_chromosome`.

- [ ] **Step 1: Emit per-excluded-record debug event**

In the `RefDecision::Exclude(e)` arm (`src/chunk_assembler.rs:311`), before continuing, add:

```rust
tracing::debug!(chrom = %chrom, pos = pos, detail = %e,
    "excluded record: REF disagrees with reference");
```

- [ ] **Step 2: Emit per-normalization debug event**

Where `left_align` changes an atom’s position/allele (`:335`), add (only when the alignment actually moved anything — compare pre/post pos or allele):

```rust
if aligned.pos != atom.pos {
    tracing::debug!(chrom = %chrom, from = atom.pos, to = aligned.pos,
        "left-aligned indel");
}
```

Accumulate a per-contig `normalized_total` counter and, in `process_chromosome` right before `contig_done`, emit the summary:

```rust
if normalized_total > 0 {
    tracing::info!(chrom = %chrom, normalized = normalized_total, "left-aligned indels");
}
```

- [ ] **Step 3: Verify build + no byte change**

Run: `pixi run bash -lc 'cargo check && maturin develop --release'`
Then run an existing conversion round-trip test:
Run: `pixi run pytest tests/ -k "roundtrip or from_vcf" -q`
Expected: PASS (output bytes unchanged; events are side-channel only).

- [ ] **Step 4: Commit**

```bash
git add src/chunk_assembler.rs src/orchestrator.rs
git commit -m "feat(logging): per-event debug + summary info for excluded/normalized variants"
```

---

## Task 7: Contig-name resolution events

**Files:**
- Modify: `src/normalize.rs` and/or the `ContigNormalizer` call sites (search: `grep -rn "ContigNormalizer\|normalize_contig\|resolve" src/*.rs python/genoray/_utils.py`)

**Interfaces:**
- Consumes: `tracing`.
- Produces: a one-time `info!` per resolved contig where the queried name differs from the stored/file name, plus a `debug!` per occurrence.

- [ ] **Step 1: Locate the resolution point**

Run: `grep -rn "ContigNormalizer\|fn normalize\|chr_prefix\|strip_chr\|add_chr" src/*.rs`
Identify where a user/query contig name is mapped to a file contig name (the point that today resolves `'1' ↔ 'chr1'` silently).

- [ ] **Step 2: Emit resolution events**

At that mapping point, when `input_name != resolved_name`:

```rust
tracing::info!(requested = %input_name, resolved = %resolved_name,
    "contig name resolved via normalization");
```

If the mapping lives on the Python side (`_utils.ContigNormalizer`), emit through Python `logging` there instead and mirror the message; but prefer the Rust site if the write path resolves names in Rust. (Genoray links this from gvl — do NOT rename `ContigNormalizer` or move it; only add logging.)

- [ ] **Step 3: Build + smoke**

Run: `pixi run bash -lc 'cargo check && maturin develop --release'`
Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(logging): report contig names resolved via normalization"
```

---

## Task 8: Demote `monitor.rs` sampler to tracing

**Files:**
- Modify: `src/monitor.rs`

**Interfaces:**
- Consumes: `tracing`.
- Produces: the sampler’s periodic output becomes `tracing::trace!(target: "genoray::monitor", …)` instead of `eprintln!`/`println!`.

- [ ] **Step 1: Convert the sampler print**

In `src/monitor.rs`, replace the periodic `eprintln!`/`println!` that prints channel fill + CPU% with:

```rust
tracing::trace!(
    target: "genoray::monitor",
    chrom = %chrom,
    dense = tx_dense.len(), sparse = tx_sparse.len(), long = tx_long.len(),
    "pipeline sampler"
);
```

(Keep the CPU% fields as additional structured fields where available.)

- [ ] **Step 2: Build**

Run: `pixi run bash -lc 'cargo check'`
Expected: compiles. The sampler now only surfaces when `GENORAY_LOG=genoray::monitor=trace` (pure-Rust) or `log_level="debug"` won’t show it (trace < debug) — it is intentionally opt-in via the env var only.

- [ ] **Step 3: Commit**

```bash
git add src/monitor.rs
git commit -m "refactor(logging): demote pipeline sampler to trace target"
```

---

## Task 9: Python renderer (rich + heartbeat) against synthetic events

**Files:**
- Create: `python/genoray/_logging.py`
- Modify: `tests/test_logging.py`

**Interfaces:**
- Consumes: `rich`.
- Produces:
  - `LOG_LEVELS = ("off", "warning", "info", "debug")`
  - `class ProgressRenderer:` with `__init__(self, console: rich.console.Console, show_bar: bool)`, `handle(self, event: tuple) -> None`, `close(self) -> None`. `event` is the tuple layout from Task 4.
  - The renderer shows a live `rich.progress.Progress` when `show_bar and console.is_terminal (or Jupyter)`, else prints throttled heartbeat lines (only when `show_bar`) and always prints `log`/`contig_done` summary lines.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_logging.py`:

```python
import io
from rich.console import Console
from genoray._logging import ProgressRenderer

def _events():
    return [
        ("contig_start", "chr1", None, None, None),
        ("progress", "chr1", 100, None, None),
        ("log", "info", "chr1", "excluded 12 records (check_ref=x)", "genoray"),
        ("contig_done", "chr1", 230, 20, 1234),
    ]

def test_heartbeat_non_tty_summary_lines():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    r = ProgressRenderer(console, show_bar=True)
    for e in _events():
        r.handle(e)
    r.close()
    out = buf.getvalue()
    assert "excluded 12 records" in out
    assert "chr1 done" in out
    assert "230" in out and "20" in out

def test_progress_false_suppresses_percent_lines():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    r = ProgressRenderer(console, show_bar=False)
    for e in _events():
        r.handle(e)
    r.close()
    out = buf.getvalue()
    # summaries still present, but no "%" throttled progress line
    assert "chr1 done" in out
    assert "%" not in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_logging.py::test_heartbeat_non_tty_summary_lines -v`
Expected: FAIL (`_logging` missing).

- [ ] **Step 3: Implement the renderer**

Create `python/genoray/_logging.py`:

```python
from __future__ import annotations

import time
from typing import Literal

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

LOG_LEVELS = ("off", "warning", "info", "debug")
LogLevel = Literal["off", "warning", "info", "debug"]

_HEARTBEAT_SECS = 5.0  # min seconds between throttled % lines per contig


class ProgressRenderer:
    """Render SVAR2 write events. Single-consumer: only the drain thread calls in."""

    def __init__(self, console: Console, show_bar: bool) -> None:
        self.console = console
        self.show_bar = show_bar
        self._live = bool(show_bar) and (console.is_terminal or console.is_jupyter)
        self._progress: Progress | None = None
        self._tasks: dict[str, int] = {}
        self._done: dict[str, int] = {}
        self._totals: dict[str, int | None] = {}
        self._last_beat: dict[str, float] = {}
        if self._live:
            self._progress = Progress(
                TextColumn("[bold blue]{task.fields[chrom]}"),
                BarColumn(),
                TextColumn("{task.completed:,} var"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
            self._progress.start()

    def handle(self, event: tuple) -> None:
        tag = event[0]
        if tag == "contig_start":
            _, chrom, total, _, _ = event
            self._totals[chrom] = total
            self._done[chrom] = 0
            self._last_beat[chrom] = 0.0
            if self._progress is not None:
                self._tasks[chrom] = self._progress.add_task(
                    "", chrom=chrom, total=total
                )
        elif tag == "progress":
            _, chrom, delta, _, _ = event
            self._done[chrom] = self._done.get(chrom, 0) + int(delta)
            if self._progress is not None:
                self._progress.update(self._tasks[chrom], advance=int(delta))
            elif self.show_bar:
                self._maybe_beat(chrom)
        elif tag == "contig_done":
            _, chrom, kept, excluded, elapsed_ms = event
            secs = int(elapsed_ms) / 1000.0
            if self._progress is not None and chrom in self._tasks:
                self._progress.update(
                    self._tasks[chrom], completed=int(kept), total=int(kept)
                )
            self.console.print(
                f"[green][svar2][/green] {chrom} done: "
                f"{int(kept):,} kept, {int(excluded):,} excluded ({secs:.1f}s)"
            )
        elif tag == "log":
            _, level, chrom, message, _target = event
            style = {"warning": "yellow", "info": "cyan", "debug": "dim"}.get(level, "")
            prefix = f"[svar2] {chrom}: " if chrom else "[svar2] "
            self.console.print(f"[{style}]{prefix}{message}[/{style}]" if style else f"{prefix}{message}")

    def _maybe_beat(self, chrom: str) -> None:
        now = time.monotonic()
        if now - self._last_beat.get(chrom, 0.0) < _HEARTBEAT_SECS:
            return
        self._last_beat[chrom] = now
        done = self._done.get(chrom, 0)
        total = self._totals.get(chrom)
        if total:
            pct = 100.0 * done / total
            self.console.print(f"[svar2] {chrom} {pct:4.0f}% ({done:,}/{total:,}) ...")
        else:
            self.console.print(f"[svar2] {chrom} {done:,} variants ...")

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_logging.py -k "heartbeat or progress_false" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_logging.py tests/test_logging.py
git commit -m "feat(logging): rich ProgressRenderer with heartbeat fallback"
```

---

## Task 10: Drain thread + `write_reporting` context manager

**Files:**
- Modify: `python/genoray/_logging.py`

**Interfaces:**
- Consumes: `genoray._core.PyEventReceiver`, `ProgressRenderer`, `LOG_LEVELS`.
- Produces:
  - `def resolve_log_level(log_level: str) -> str` — validates against `LOG_LEVELS`, applies `GENORAY_LOG` env override precedence (env wins if it names a bare level; otherwise the arg).
  - `@contextmanager def write_reporting(progress: bool, log_level: str) -> Iterator[PyEventReceiver | None]` — if both `progress` is False and `log_level == "off"`, yields `None` (no receiver, zero overhead). Otherwise creates a `PyEventReceiver`, spawns a daemon drain thread that loops `recv_timeout(100)` → `renderer.handle(...)` until `StopIteration`, yields the receiver, and on exit joins the thread and calls `renderer.close()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_logging.py`:

```python
import threading
from genoray._logging import resolve_log_level, write_reporting

def test_resolve_log_level_validates_and_env(monkeypatch):
    assert resolve_log_level("info") == "info"
    monkeypatch.setenv("GENORAY_LOG", "debug")
    assert resolve_log_level("info") == "debug"
    monkeypatch.delenv("GENORAY_LOG", raising=False)
    import pytest
    with pytest.raises(ValueError):
        resolve_log_level("loud")

def test_write_reporting_disabled_yields_none():
    with write_reporting(progress=False, log_level="off") as rx:
        assert rx is None

def test_write_reporting_drains_and_joins():
    n_threads_before = threading.active_count()
    with write_reporting(progress=False, log_level="info") as rx:
        assert rx is not None
        # Simulate a producer finishing immediately by not sending anything;
        # context exit must drop the sender and join the drain thread.
    # After exit, no leaked drain thread.
    assert threading.active_count() == n_threads_before
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_logging.py -k "resolve_log_level or write_reporting" -v`
Expected: FAIL (`resolve_log_level`/`write_reporting` missing).

- [ ] **Step 3: Implement**

Append to `python/genoray/_logging.py`:

```python
import os
import threading
from contextlib import contextmanager
from typing import Iterator


def resolve_log_level(log_level: str) -> str:
    if log_level not in LOG_LEVELS:
        raise ValueError(f"log_level must be one of {LOG_LEVELS}; got {log_level!r}")
    env = os.environ.get("GENORAY_LOG", "").strip().lower()
    if env in LOG_LEVELS:
        return env
    return log_level


@contextmanager
def write_reporting(progress: bool, log_level: str) -> Iterator[object | None]:
    level = resolve_log_level(log_level)
    if not progress and level == "off":
        yield None
        return

    from genoray import _core

    console = Console()
    renderer = ProgressRenderer(console, show_bar=progress)
    rx = _core.PyEventReceiver()
    stop = threading.Event()

    def _drain() -> None:
        while not stop.is_set():
            try:
                ev = rx.recv_timeout(100)
            except StopIteration:
                break
            if ev is not None:
                try:
                    renderer.handle(ev)
                except Exception:
                    pass  # never let rendering crash the write
        # drain any straggling events after disconnect
        while True:
            try:
                ev = rx.recv_timeout(0)
            except StopIteration:
                break
            if ev is None:
                break
            try:
                renderer.handle(ev)
            except Exception:
                pass

    t = threading.Thread(target=_drain, name="genoray-log-drain", daemon=True)
    t.start()
    try:
        yield rx
    finally:
        stop.set()
        # Dropping the Rust-side pipeline senders triggers StopIteration in the
        # drain loop; the receiver's own internal sender is released when `rx`
        # is garbage-collected. Give the drain a bounded join.
        t.join(timeout=5.0)
        renderer.close()
```

> Implementer note: for the drain thread to observe `StopIteration`, all `Sender`s must drop. The pipeline’s `EventSink` senders drop when the Rust call returns. `PyEventReceiver` also holds a keep-alive `tx`; ensure the drain’s post-loop uses `recv_timeout(0)` and the `stop` event so join is bounded even if that keep-alive sender is still alive. (A cleaner alternative the implementer may choose: give `PyEventReceiver` a `close()` method that drops its internal `tx`, called in `finally` before `join`.)

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run pytest tests/test_logging.py -k "resolve_log_level or write_reporting" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_logging.py tests/test_logging.py
git commit -m "feat(logging): drain-thread write_reporting context manager"
```

---

## Task 11: Wire `progress`/`log_level` into `from_vcf`

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_vcf` signature `:611` and its `_core.run_conversion_pipeline` call `:775`)
- Modify: `src/lib.rs` (`run_conversion_pipeline` signature — add `log_level`, `receiver`)
- Modify: `tests/test_logging.py`

**Interfaces:**
- Consumes: `write_reporting`, `resolve_log_level`, `_core.run_conversion_pipeline(..., log_level, receiver)`.
- Produces: `from_vcf(..., progress: bool = False, log_level: str = "info")` end-to-end.

- [ ] **Step 1: Write the failing end-to-end test**

The svar2 class is `SparseVar2` (exported as `from genoray import SparseVar2`).
Fixtures are built **inline** with `samtools`/`bgzip` exactly like
`tests/test_svar2_from_vcf.py` (there are no static `tests/data/*.fa` files).
Add to `tests/test_logging.py`:

```python
import subprocess
from pathlib import Path

from genoray import SparseVar2

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _tiny_vcf(d: Path) -> tuple[Path, Path]:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz, ref


def test_from_vcf_emits_summary(tmp_path, capsys):
    src, ref = _tiny_vcf(tmp_path)
    out = tmp_path / "out.svar"
    SparseVar2.from_vcf(out, src, ref, progress=False, log_level="info")
    captured = capsys.readouterr()
    assert "done" in captured.out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run pytest tests/test_logging.py::test_from_vcf_emits_summary -v`
Expected: FAIL (unexpected `progress`/`log_level` kwargs, or no summary printed).

- [ ] **Step 3: Add Rust params to `run_conversion_pipeline`**

In `src/lib.rs`, extend the `#[pyo3(signature = (...))]` and fn args for `run_conversion_pipeline` with `log_level: String` and `receiver: Option<Py<PyEventReceiver>>` (append at the end; update the Python call in Step 4 to match positionally or use keywords). Build the sink + wrap `py.detach` per Task 5 Step 4. (If Task 5 already added these to all five pyfunctions, this step is just confirming `run_conversion_pipeline` is done.)

- [ ] **Step 4: Wire the Python side**

In `python/genoray/_svar2.py`, add to `from_vcf`’s keyword-only args:

```python
        progress: bool = False,
        log_level: str = "info",
```

Replace the `return _core.run_conversion_pipeline(...)` at `:775` with:

```python
        from ._logging import write_reporting

        with write_reporting(progress, log_level) as rx:
            return _core.run_conversion_pipeline(
                str(source),
                reference_path,
                contigs,
                str(out),
                selected_samples,
                chunk_size,
                ploidy,
                threads,
                long_allele_capacity,
                skip_out_of_scope,
                signatures,
                info,
                format_,
                check_ref,
                region_ranges,
                regions_overlap,
                log_level,   # new
                rx,          # new: PyEventReceiver | None
            )
```

Update the `from_vcf` docstring: replace the "currently a no-op" progress note with the real behavior and document `log_level`.

- [ ] **Step 5: Build + run**

Run: `pixi run bash -lc 'maturin develop --release && pytest tests/test_logging.py::test_from_vcf_emits_summary -v'`
Expected: PASS.

- [ ] **Step 6: Regression — output unchanged**

Run: `pixi run pytest tests/ -k "from_vcf" -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_svar2.py src/lib.rs tests/test_logging.py
git commit -m "feat(svar2): functional progress/log_level on from_vcf"
```

---

## Task 12: Wire remaining entry points

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_pgen` `:795`/`:1092`, `from_vcf_list` `:1115`/`:1378`, `from_svar1` `:1399`/`:1576`, `write_view` `:464`/`:595`)
- Modify: `src/lib.rs` (`run_pgen_conversion_pipeline`, `run_vcf_list_conversion_pipeline`, `run_svar1_conversion_pipeline`, `run_slice_view` — add `log_level`, `receiver` if not already added in Task 5)

**Interfaces:**
- Consumes/Produces: same pattern as Task 11, applied to the four remaining methods. Each gains `progress: bool = False, log_level: str = "info"` and wraps its `_core.*` call in `write_reporting`.

- [ ] **Step 1: Repeat the Task 11 pattern for each method**

For `from_pgen`, `from_vcf_list`, `from_svar1`, `write_view`:
1. Add `progress`/`log_level` keyword-only args (with the same defaults).
2. Import and wrap: `with write_reporting(progress, log_level) as rx:` around the `_core.*` call, passing `log_level` and `rx` as the two new trailing args.
3. Add the two params to the corresponding pyfunction in `src/lib.rs`.
4. Update each docstring (replace no-op progress language; document `log_level`).

For `write_view`/`run_slice_view`: the slicer is concurrent per contig with no per-record stream. Emit only `ContigStart(total=None)` → `ContigDone(kept, excluded=0, elapsed)` per sliced contig (no per-record `tick`), plus any `warn!` for skipped regions. This keeps `write_view`’s progress coarse-grained (one line per contig).

- [ ] **Step 2: Add per-method end-to-end tests**

For each `SparseVar2` method, add a test mirroring `test_from_vcf_emits_summary`,
building fixtures inline the way the matching canonical test does:
- `from_pgen` → see `tests/test_svar2_from_pgen.py` (uses a `.pgen` built via `plink2`/helpers there).
- `from_vcf_list` → see `tests/test_svar2_from_vcf_list.py` (list of bgzipped VCFs).
- `from_svar1` → see `tests/test_svar2_from_svar1.py`.
- `write_view` → see `tests/test_svar2_write_view.py` (writes an svar2, then slices it).

Reuse each file's fixture helpers (import them or copy the small builder). Each
test asserts a summary line appears in `capsys.readouterr().out` and that the
return value/output matches a `progress=False, log_level="off"` baseline.

- [ ] **Step 3: Build + run**

Run: `pixi run bash -lc 'maturin develop --release && pytest tests/test_logging.py -q'`
Expected: PASS.

Run: `pixi run pytest tests/ -q`
Expected: PASS (full suite; output byte-identical).

- [ ] **Step 4: Commit**

```bash
git add python/genoray/_svar2.py src/lib.rs tests/test_logging.py
git commit -m "feat(svar2): functional progress/log_level on from_pgen/from_vcf_list/from_svar1/write_view"
```

---

## Task 13: CLI flags

**Files:**
- Modify: `python/genoray/_cli/__main__.py` (the `write vcf`/`write pgen`/`write svar1` subcommands, and any `write` view command)

**Interfaces:**
- Consumes: the new `progress`/`log_level` kwargs on the `SparseVar.from_*` methods.
- Produces: `--progress/--no-progress` and `--log-level {off,warning,info,debug}` on each `write` subcommand, forwarded to the method call.

- [ ] **Step 1: Add flags to each write subcommand**

For each `write_*` function (e.g. `write_vcf` at `:104`), add keyword-only params:

```python
    progress: Annotated[bool, Parameter(name="--progress", negative="--no-progress")] = False,
    log_level: Annotated[
        Literal["off", "warning", "info", "debug"], Parameter(name="--log-level")
    ] = "info",
```

Forward them in the `SparseVar.from_*(...)` call inside each subcommand:

```python
        progress=progress,
        log_level=log_level,
```

- [ ] **Step 2: Manual smoke test**

Run: `pixi run bash -lc 'genoray write vcf --help'`
Expected: help shows `--progress/--no-progress` and `--log-level`.

Build a tiny fixture, then run the CLI on it:

```bash
pixi run bash -lc '
  d=$(mktemp -d); printf ">chr1\nACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT\n" > $d/ref.fa
  samtools faidx $d/ref.fa
  printf "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=40>\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"GT\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n" | tr -s " " "\t" > $d/in.vcf
  bgzip -c $d/in.vcf > $d/in.vcf.gz; bcftools index $d/in.vcf.gz
  genoray write vcf $d/in.vcf.gz $d/out.svar --reference $d/ref.fa --log-level info'
```

Expected: prints a `[svar2] … done: …` summary line; exit 0.

- [ ] **Step 3: Commit**

```bash
git add python/genoray/_cli/__main__.py
git commit -m "feat(cli): --progress/--log-level on write subcommands"
```

---

## Task 14: Docs — SKILL.md + docstrings

**Files:**
- Modify: `skills/genoray-api/SKILL.md`
- Verify: docstrings in `python/genoray/_svar2.py` (already updated per-method in Tasks 11–12)

**Interfaces:**
- Consumes: the final public surface.
- Produces: documented `progress`, `log_level`, and CLI flags.

- [ ] **Step 1: Document the new kwargs/flags**

In `skills/genoray-api/SKILL.md`, in the SparseVar write section, add `progress: bool = False` (now functional) and `log_level: Literal["off","warning","info","debug"] = "info"` to every write entry point’s documented signature, with a one-paragraph description of: bar in terminal/Jupyter, heartbeat lines non-TTY, per-event edge cases at `debug`, and the `GENORAY_LOG` env escape hatch. Add the CLI `--progress/--no-progress` and `--log-level` flags to the CLI section.

- [ ] **Step 2: Verify no stale "no-op" text remains**

Run: `grep -rn "no-op\|currently a no-op" python/genoray/ skills/`
Expected: no matches referencing `progress`.

- [ ] **Step 3: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(skill): document progress/log_level on SVAR2 write + CLI"
```

---

## Task 15: Full verification + PR

**Files:** none (verification only)

- [ ] **Step 1: Rust tests**

Run: `pixi run bash -lc 'cargo test --no-default-features --lib logging::'`
Expected: PASS (all logging unit tests).

Run: `pixi run bash -lc 'cargo check --no-default-features'` (query-core build)
Expected: compiles (conversion-gated `PyEventReceiver` excluded cleanly).

- [ ] **Step 2: Python suite (release ext)**

Run: `pixi run bash -lc 'maturin develop --release && pytest tests/ -q'`
Expected: PASS, including the new `tests/test_logging.py` and all pre-existing conversion round-trip tests (byte-identical output).

- [ ] **Step 3: Lint**

Run: `pixi run bash -lc 'ruff check genoray tests python && ruff format --check python tests'`
Run: `pixi run bash -lc 'cargo fmt --check && cargo clippy --no-default-features'`
Expected: clean.

- [ ] **Step 4: Manual interactive check**

In a real terminal and (optionally) a Jupyter kernel, run a `from_vcf(..., progress=True, log_level="info")` on a medium fixture; confirm a live bar renders in the terminal and (in Jupyter) a widget-style bar, and that `info` summary lines appear above the bar without corrupting it.

- [ ] **Step 5: Open the PR**

```bash
git push -u origin worktree-svar2-logging-spec
gh pr create --draft --title "feat: SVAR2 write logging & progress (tracing + rich)" \
  --body "Implements docs/superpowers/specs/2026-07-20-svar2-logging-design.md. See docs/superpowers/plans/2026-07-20-svar2-logging.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Parallelization notes (for subagent-driven execution)

- **Sequential spine:** Task 1 → 2 → 3 → 4 → 5 gate everything Rust-side; Task 11 gates 12/13.
- **Parallelizable once Task 4 lands:**
  - Rust edge-case events (Tasks 6, 7, 8) are independent of each other and of the Python renderer.
  - Python renderer + drain (Tasks 9, 10) are testable against synthetic event tuples **without** the Rust side and can proceed in parallel with Tasks 5–8.
- **Converge** at Task 11 (first real end-to-end wiring), then fan out Task 12’s four methods in parallel, with 13/14 after.
- Per project convention: use Sonnet (or weaker) implementers for these tasks; reserve Opus for review and any critical-failure fixes.

## Open implementation risks (flag to reviewer)

1. **pyo3 0.29 mixed-tuple encoding** (Task 4): heterogeneous `Option<u64>` tuples may need explicit `PyTuple::new`. Layout is fixed; encoding mechanism is the implementer’s choice.
2. **Channel disconnect / drain join** (Task 10): the cleanest shutdown is a `PyEventReceiver.close()` that drops its internal keep-alive `tx`; add it if the bounded `join` proves flaky.
3. **`kept_total` sourcing** (Task 5): confirm whether `process_chromosome` already tracks a kept count; if not, accumulate in the executor tick callback as shown.
4. **Contig-resolution site** (Task 7): confirm resolution happens in Rust vs. Python `_utils.ContigNormalizer`; log at whichever side actually resolves, without renaming the type (gvl imports it).
