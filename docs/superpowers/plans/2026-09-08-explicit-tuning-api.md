# Explicit Tuning API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** Tasks 1–4 are mutually independent and touch disjoint files —
> dispatch them together using superpowers:dispatching-parallel-agents. Tasks 5–8
> are a serial chain. Tasks 9–11 are again mutually independent. Wave boundaries
> are marked in the task list below.

**Goal:** Replace genoray's `GENORAY_*` environment-variable configuration
channel with an explicit `genoray.Tuning` object plus CLI flags, and align
`log_level` with current Python level names.

**Architecture:** A frozen Python dataclass (`genoray.Tuning`) carries six
scheduling knobs. It crosses the FFI as one argument per pyo3 entry point via a
`#[derive(FromPyObject)]` struct, is resolved once in `lib.rs` beside the
concurrency plan into a `ResolvedTuning`, and flows down as data. Every
`std::env::var` read in the conversion path is deleted, so the `pipeline config`
banner — which gains a `<field>_src=explicit|planner` tag per knob — is correct
by construction rather than by maintenance.

**Tech Stack:** Rust (pyo3 0.2x, tracing/tracing-subscriber, rayon), Python 3.10+
(dataclasses, cyclopts CLI), pixi for env/tasks, pytest, prek pre-commit hooks.

**Spec:** `docs/superpowers/specs/2026-09-08-explicit-tuning-api-design.md`

## Global Constraints

- **Worktree:** all work happens in
  `/carter/users/dlaub/projects/genoray/.claude/worktrees/issue-explicit-tuning-api`
  on branch `worktree-issue-explicit-tuning-api`. Do not `cd` to the main
  checkout.
- **`export CARGO_TARGET_DIR=/local/$USER/genoray-target` (or another non-NFS
  path) before ANY `cargo` command or any `git commit` whose hooks run cargo.**
  The repo's `target/` is on NFS, where cargo bus-errors while mmapping object
  files. This is not optional; it is the difference between a green run and a
  spurious SIGBUS.
- **Rust tests:** `cargo test --no-default-features --features conversion`.
  Dropping `extension-module` is required or the pyo3 test binary will not link;
  keeping `conversion` is required or you silently skip the entire conversion
  path (341 tests vs 189).
- **Python tests:** `pixi run test`, NOT `pixi run pytest` — the latter skips
  fixture generation and produces ~277 bogus `FileNotFoundError`s that look like
  a broken branch. To run one file: `pixi run test tests/test_tuning.py`.
- **`pixi run test` does NOT rebuild the Rust `.so`.** After any Rust change,
  run `maturin develop --release` before running Python tests, or you will be
  testing stale code. Verify the `.so` mtime changed.
- **Foreground only.** Do not background long `cargo`/`maturin` runs and report
  early; run them to completion in the foreground.
- **Conventional Commits**, enforced by a `commitizen` pre-commit hook. This work
  is breaking: use `feat!:` and a `BREAKING CHANGE:` footer where the public
  surface changes. **Never edit `CHANGELOG.md` or the version by hand** — CI's
  release workflow owns both.
- **Public API changes MUST update `skills/genoray-api/SKILL.md` in the same
  work** (Task 11). This is a repo rule, not a nicety.
- **Six `Tuning` fields, exactly**: `concurrent_chroms`, `reader_workers`,
  `overshard`, `dense_cap`, `merge_threads`, `sample_interval`. There is
  deliberately **no `shard_htslib_threads`** — PR #174 removed that hook and made
  it the constant `SHARDED_VCF_HTSLIB_THREADS_PER_READER = 0`.
- **Accepted `log_level` values**: `"off"`, `"critical"`, `"error"`,
  `"warning"`, `"info"`, `"debug"` (case-insensitive) plus integers.
  `"warn"` is REJECTED (removed from Python in 3.13). `"critical"` is an alias
  for `"error"`.
- **Canonical level ranks**: `off=0, error=1, warning=2, info=3, debug=4`.

---

## File Structure

**Created**

| File | Responsibility |
|---|---|
| `python/genoray/_tuning.py` | The `Tuning` dataclass, its validation, and the per-backend applicability matrix |
| `python/genoray/_cli/_tuning_flags.py` | One cyclopts flag group shared by the four write commands |
| `src/tuning.rs` | `TuningIn` (FFI input), `ResolvedTuning` (planner output), defaults, provenance tags |
| `tests/test_tuning.py` | `Tuning` validation + applicability |
| `tests/test_log_levels.py` | Level name/int parsing table |
| `tests/test_no_env_vars.py` | Regression guard: no `std::env::var` may return to `src/` |
| `tests/test_pipeline_config_banner.py` | Banner reports every field with the right `_src` tag |

**Modified**

| File | Change |
|---|---|
| `python/genoray/__init__.py` | Export `Tuning` |
| `python/genoray/_logging.py` | `resolve_log_level` → `parse_log_level`; drop `GENORAY_LOG` |
| `python/genoray/_svar2.py` | `tuning=` on 4 methods; `log_filter=` on 5; drop flat `reader_workers=` |
| `python/genoray/_cli/__main__.py` | New flags; wire the shared flag group |
| `src/logging.rs` | 5-level ranks; reloadable fmt filter; `with_channel_subscriber` + `install_fmt_fallback` signatures |
| `src/lib.rs` | 5 entry points take `tuning`/`log_filter`; resolution; 3 unified banners; drop `mod trace` |
| `src/orchestrator.rs` | Delete `bench_env` & friends; thread `ResolvedTuning`; drop `trace_ll!` |
| `src/monitor.rs` | `spawn_sampler` takes the interval; delete `sample_interval_secs` |
| `src/executor.rs`, `src/shard_exec.rs` | Drop `trace_ll!` call sites |
| `src/bin/bench_from_vcf_list.rs` | `--log-filter` flag |
| 5 test files, 5 bench scripts | Migrate off env vars (Tasks 10, 11) |
| `skills/genoray-api/SKILL.md`, `docs/source/svar.md`, `scripts/bench_svar2/README.md` | Docs (Task 11) |

**Deleted**: `src/trace.rs`.

---

## WAVE A — Tasks 1–4, dispatch in parallel (disjoint files)

### Task 1: The `Tuning` dataclass

**Files:**
- Create: `python/genoray/_tuning.py`
- Create: `tests/test_tuning.py`
- Modify: `python/genoray/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `genoray.Tuning` — frozen, kw-only, slots dataclass; six `int | None` fields
    defaulting to `None`.
  - `Tuning._check_backend(backend: str) -> None` — raises `ValueError` if a
    non-`None` field does not apply to `backend`. `backend` is one of
    `"vcf" | "pgen" | "vcf_list" | "svar1"`.
  - `Tuning._as_ffi() -> dict[str, int | None]` — the six fields as a plain dict,
    for handing to `_core`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tuning.py`:

```python
from __future__ import annotations

import dataclasses

import pytest

from genoray import Tuning


def test_defaults_are_all_none():
    t = Tuning()
    assert t._as_ffi() == {
        "concurrent_chroms": None,
        "reader_workers": None,
        "overshard": None,
        "dense_cap": None,
        "merge_threads": None,
        "sample_interval": None,
    }


def test_frozen_and_kw_only():
    t = Tuning(reader_workers=20)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.reader_workers = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        Tuning(2)  # type: ignore[misc]  # positional args rejected


def test_replace_works():
    t = Tuning(reader_workers=20, overshard=40)
    assert dataclasses.replace(t, overshard=4).overshard == 4


@pytest.mark.parametrize(
    "field",
    ["concurrent_chroms", "reader_workers", "overshard", "dense_cap", "merge_threads"],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_below_one_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        Tuning(**{field: bad})


def test_sample_interval_zero_allowed_negative_rejected():
    assert Tuning(sample_interval=0).sample_interval == 0
    with pytest.raises(ValueError, match="sample_interval"):
        Tuning(sample_interval=-1)


# The applicability matrix, straight from the spec's section A table.
@pytest.mark.parametrize(
    ("backend", "field"),
    [
        ("pgen", "reader_workers"),
        ("pgen", "overshard"),
        ("vcf_list", "concurrent_chroms"),
        ("vcf_list", "reader_workers"),
        ("vcf_list", "overshard"),
        ("svar1", "reader_workers"),
        ("svar1", "overshard"),
    ],
)
def test_inapplicable_field_rejected_naming_field_and_backend(backend, field):
    t = Tuning(**{field: 2})
    with pytest.raises(ValueError) as excinfo:
        t._check_backend(backend)
    msg = str(excinfo.value)
    assert field in msg
    assert backend in msg


@pytest.mark.parametrize(
    ("backend", "field"),
    [
        ("vcf", "reader_workers"),
        ("vcf", "overshard"),
        ("vcf", "concurrent_chroms"),
        ("pgen", "concurrent_chroms"),
        ("svar1", "concurrent_chroms"),
        ("vcf_list", "dense_cap"),
        ("vcf_list", "merge_threads"),
        ("vcf_list", "sample_interval"),
    ],
)
def test_applicable_field_accepted(backend, field):
    Tuning(**{field: 2})._check_backend(backend)  # must not raise


def test_none_fields_never_rejected():
    for backend in ("vcf", "pgen", "vcf_list", "svar1"):
        Tuning()._check_backend(backend)


def test_unknown_backend_is_a_programming_error():
    with pytest.raises(KeyError):
        Tuning()._check_backend("nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run test tests/test_tuning.py`
Expected: FAIL — `ImportError: cannot import name 'Tuning' from 'genoray'`.

- [ ] **Step 3: Write the implementation**

Create `python/genoray/_tuning.py`:

```python
"""Explicit scheduling knobs for SVAR2 conversion.

Replaces the `GENORAY_*` environment variables. Every field is `None` by
default, meaning "let the planner derive it" -- the values a caller does not set
are chosen exactly as they are today.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

Backend = Literal["vcf", "pgen", "vcf_list", "svar1"]

# Which knobs each conversion backend can actually use. A field set on a backend
# that cannot use it raises rather than being silently ignored -- silent
# ignoring is the failure mode this whole module exists to remove.
#
# `reader_workers`/`overshard` are sharded-VCF only: `from_pgen` pins P=1
# (pgenlib holds the GIL through decode, so sub-contig sharding is pure
# overhead) and neither `from_vcf_list` nor `from_svar1` shards within a contig.
# `concurrent_chroms` is unavailable on `from_vcf_list` because that pipeline
# walks contigs sequentially by design (`orchestrator::run_vcf_list`).
_ALWAYS = frozenset({"dense_cap", "merge_threads", "sample_interval"})
_APPLICABLE: dict[str, frozenset[str]] = {
    "vcf": _ALWAYS | {"concurrent_chroms", "reader_workers", "overshard"},
    "pgen": _ALWAYS | {"concurrent_chroms"},
    "vcf_list": _ALWAYS,
    "svar1": _ALWAYS | {"concurrent_chroms"},
}

# Smallest meaningful value per field. `sample_interval` is the one field where
# 0 is meaningful: it disables the monitor sampler.
_MINIMUM: dict[str, int] = {
    "concurrent_chroms": 1,
    "reader_workers": 1,
    "overshard": 1,
    "dense_cap": 1,
    "merge_threads": 1,
    "sample_interval": 0,
}


@dataclass(frozen=True, kw_only=True, slots=True)
class Tuning:
    """Scheduling knobs for a `SparseVar2` write.

    Every field defaults to `None`, which means the planner derives it. A value
    you do set is honoured or refused -- never silently shrunk -- and is
    reported in the `pipeline config` log line tagged `explicit`.

    Not every knob applies to every backend; see the table below. Setting one
    that does not apply raises `ValueError`.

    | field | `from_vcf` | `from_pgen` | `from_vcf_list` | `from_svar1` |
    |---|---|---|---|---|
    | `concurrent_chroms` | yes | yes | no | yes |
    | `reader_workers` | yes | no | no | no |
    | `overshard` | yes | no | no | no |
    | `dense_cap` | yes | yes | yes | yes |
    | `merge_threads` | yes | yes | yes | yes |
    | `sample_interval` | yes | yes | yes | yes |

    Attributes:
        concurrent_chroms: contigs converted concurrently. Honoured, or refused
            with `InsufficientMemory` when it does not fit `max_mem`.
        reader_workers: independent indexed shard readers per concurrent contig.
        overshard: work units per reader, decoupling unit size from reader
            count. Only consulted when a contig has no exact record count.
        dense_cap: depth of the dense-chunk channel between reader and executor.
        merge_threads: gather threads for the per-contig var_key merge tail.
        sample_interval: monitor sampling cadence in seconds; 0 disables it.
    """

    concurrent_chroms: int | None = None
    reader_workers: int | None = None
    overshard: int | None = None
    dense_cap: int | None = None
    merge_threads: int | None = None
    sample_interval: int | None = None

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"{f.name} must be None or an int; got {value!r}"
                )
            minimum = _MINIMUM[f.name]
            if value < minimum:
                raise ValueError(
                    f"{f.name} must be None (let the planner choose) or an "
                    f"integer >= {minimum}; got {value!r}"
                )

    def _check_backend(self, backend: str) -> None:
        """Raise if any set field cannot be used by `backend`."""
        allowed = _APPLICABLE[backend]  # KeyError on an unknown backend: a bug
        for f in fields(self):
            if getattr(self, f.name) is None or f.name in allowed:
                continue
            raise ValueError(
                f"tuning.{f.name} does not apply to the {backend!r} backend "
                f"(applicable knobs: {', '.join(sorted(allowed))}). Leave it "
                f"as None."
            )

    def _as_ffi(self) -> dict[str, int | None]:
        """The six fields as a plain dict for `_core`."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
```

- [ ] **Step 4: Export it**

In `python/genoray/__init__.py`, add `"Tuning"` to `__all__` (keep the list
alphabetically sorted, as it already is) and add the import next to the other
eager imports:

```python
from ._tuning import Tuning
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run test tests/test_tuning.py`
Expected: PASS, all parametrizations.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_tuning.py python/genoray/__init__.py tests/test_tuning.py
git commit -m "feat(tuning): add the Tuning dataclass and its applicability matrix"
```

---

### Task 2: Python-convention log levels

**Files:**
- Modify: `python/genoray/_logging.py:12-14` (the `LOG_LEVELS`/`LogLevel` block)
  and `:96-105` (`resolve_log_level`)
- Create: `tests/test_log_levels.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `genoray._logging.parse_log_level(level: str | int) -> str` — returns one of
    the four canonical names Rust understands: `"off" | "error" | "warning" |
    "info" | "debug"`. Raises `ValueError` on anything else.
  - `genoray._logging.LOG_LEVELS` — the accepted string spellings, now
    `("off", "critical", "error", "warning", "info", "debug")`.
  - `LogLevel` — the matching `Literal`.
  - `write_reporting(progress: bool, log_level: str | int)` keeps its name and
    arity; it now calls `parse_log_level` instead of `resolve_log_level`.

Note: `resolve_log_level` is deleted, not kept as an alias — it is private
(underscore module) and its only behaviour difference was the env var.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_log_levels.py`:

```python
from __future__ import annotations

import logging

import pytest

from genoray._logging import LOG_LEVELS, parse_log_level


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("off", "off"),
        ("debug", "debug"),
        ("info", "info"),
        ("warning", "warning"),
        ("error", "error"),
        # "critical" is an alias: tracing has no CRITICAL level.
        ("critical", "error"),
        # Case-insensitive, like logging.getLevelName's own inputs.
        ("DEBUG", "debug"),
        ("Warning", "warning"),
        ("CRITICAL", "error"),
    ],
)
def test_named_levels(given, expected):
    assert parse_log_level(given) == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (logging.DEBUG, "debug"),
        (logging.INFO, "info"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "error"),
        (0, "off"),  # logging.NOTSET: no parent to inherit from -> silence
        # Between named levels, round UP to the more severe one: a logger set
        # to 25 suppresses INFO(20) and admits WARNING(30).
        (25, "warning"),
        (11, "info"),
        (35, "error"),
        (99, "error"),
    ],
)
def test_integer_levels(given, expected):
    assert parse_log_level(given) == expected


def test_warn_is_rejected_as_deprecated():
    with pytest.raises(ValueError) as excinfo:
        parse_log_level("warn")
    # The message must name the accepted set, since "warn" is what tracing
    # calls this level and is a natural thing to type.
    assert "warning" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["", "verbose", "trace", "none", "quiet"])
def test_unknown_names_rejected(bad):
    with pytest.raises(ValueError):
        parse_log_level(bad)


def test_negative_int_rejected():
    with pytest.raises(ValueError):
        parse_log_level(-1)


def test_accepted_spellings_are_exported():
    assert set(LOG_LEVELS) == {
        "off",
        "critical",
        "error",
        "warning",
        "info",
        "debug",
    }
    # Every advertised spelling must actually parse.
    for name in LOG_LEVELS:
        parse_log_level(name)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run test tests/test_log_levels.py`
Expected: FAIL — `ImportError: cannot import name 'parse_log_level'`.

- [ ] **Step 3: Replace the level block in `python/genoray/_logging.py`**

Replace the module-level constants (currently lines 12-14):

```python
LOG_LEVELS = ("off", "critical", "error", "warning", "info", "debug")
LogLevel = Literal["off", "critical", "error", "warning", "info", "debug"]

# What Rust's `level_rank` understands. "critical" is an accepted spelling but
# not a distinct rank: `tracing` has no CRITICAL level, and no genoray call site
# emits `error!` or `warn!` today, so it gates the same (currently empty) set as
# "error".
_CANONICAL: dict[str, str] = {
    "off": "off",
    "critical": "error",
    "error": "error",
    "warning": "warning",
    "info": "info",
    "debug": "debug",
}

# Python `logging` integer thresholds, ascending. An integer that falls between
# two named levels rounds UP to the more severe one, matching Python's own
# gate: a logger set to 25 suppresses INFO (20) and admits WARNING (30).
_INT_LEVELS: tuple[tuple[int, str], ...] = (
    (10, "debug"),
    (20, "info"),
    (30, "warning"),
    (40, "error"),
)
```

Then delete `resolve_log_level` (currently lines 96-105) and put this in its
place:

```python
def parse_log_level(log_level: str | int) -> str:
    """Normalize a Python-convention level to the name Rust's gate understands.

    Accepts the names in `LOG_LEVELS` (case-insensitive) and `logging` integer
    constants. Returns one of "off", "error", "warning", "info", "debug".

    `"warn"` is deliberately NOT accepted: `logging.warn()` was removed in
    Python 3.13, so it is a deprecated spelling rather than a convention.
    """
    if isinstance(log_level, bool):  # bool is an int; nobody means this
        raise ValueError(f"log_level must be a str or int; got {log_level!r}")
    if isinstance(log_level, int):
        if log_level < 0:
            raise ValueError(
                f"log_level as an int must be >= 0 (logging.NOTSET); "
                f"got {log_level!r}"
            )
        if log_level == 0:
            # logging.NOTSET means "inherit from the parent logger"; there is no
            # parent here, so the only coherent reading is silence.
            return "off"
        for threshold, name in _INT_LEVELS:
            if log_level <= threshold:
                return name
        return "error"  # >= CRITICAL
    if isinstance(log_level, str):
        canonical = _CANONICAL.get(log_level.strip().lower())
        if canonical is not None:
            return canonical
    raise ValueError(
        f"log_level must be one of {LOG_LEVELS} (case-insensitive) or a "
        f"logging level int; got {log_level!r}"
    )
```

- [ ] **Step 4: Point `write_reporting` at it**

In `write_reporting` (currently `level = resolve_log_level(log_level)`), change
the one call:

```python
    level = parse_log_level(log_level)
```

and widen its parameter annotation to `log_level: str | int`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run test tests/test_log_levels.py`
Expected: PASS.

- [ ] **Step 6: Check nothing else called the deleted function**

Run: `rg -n 'resolve_log_level' python/ tests/ scripts/`
Expected: no hits outside `tests/test_logging.py` (which Task 10 migrates).
If any *other* caller appears, update it to `parse_log_level` now.

- [ ] **Step 7: Commit**

```bash
git add python/genoray/_logging.py tests/test_log_levels.py
git commit -m "feat(logging)!: accept Python-convention log level names and ints

BREAKING CHANGE: log_level no longer consults GENORAY_LOG. It now accepts
critical/error alongside the existing off/warning/info/debug, plus logging
integer constants. \"warn\" is rejected as a deprecated Python spelling."
```

---

### Task 3: Reloadable stderr filter and 5-level ranks in Rust

**Files:**
- Modify: `src/logging.rs:228-252` (ranks), `:301-320` (`ensure_global_subscriber`),
  `:333-358` (`with_channel_subscriber`), `:366-368` (`install_fmt_fallback`)
- Modify: `src/bin/bench_from_vcf_list.rs:48`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `logging::with_channel_subscriber<R>(sink: EventSink, level: &str, filter: Option<&str>, f: impl FnOnce() -> R) -> R`
    — one new third parameter, before the closure.
  - `logging::install_fmt_fallback(filter: Option<&str>)`.
  - Ranks: `off=0, error=1, warning=2, info=3, debug=4`.

- [ ] **Step 1: Write the failing tests**

Add to the existing `#[cfg(test)] mod tests` at the bottom of `src/logging.rs`:

```rust
    #[test]
    fn level_ranks_split_error_from_warning() {
        assert_eq!(level_rank("off"), 0);
        assert_eq!(level_rank("error"), 1);
        assert_eq!(level_rank("warning"), 2);
        assert_eq!(level_rank("info"), 3);
        assert_eq!(level_rank("debug"), 4);
        // Unknown spellings fall back to info, as before -- Python has already
        // validated and canonicalized by the time a level reaches here.
        assert_eq!(level_rank("nonsense"), 3);
    }

    #[test]
    fn event_ranks_match_level_ranks() {
        assert_eq!(event_rank(&tracing::Level::ERROR), 1);
        assert_eq!(event_rank(&tracing::Level::WARN), 2);
        assert_eq!(event_rank(&tracing::Level::INFO), 3);
        assert_eq!(event_rank(&tracing::Level::DEBUG), 4);
        assert_eq!(event_rank(&tracing::Level::TRACE), 4);
    }

    #[test]
    fn a_warning_is_dropped_at_error_level_but_kept_at_warning() {
        // Rank ordering is the whole contract of the channel gate: an event is
        // dropped when its rank exceeds the active level's rank.
        assert!(event_rank(&tracing::Level::WARN) > level_rank("error"));
        assert!(event_rank(&tracing::Level::WARN) <= level_rank("warning"));
        assert!(event_rank(&tracing::Level::INFO) > level_rank("warning"));
    }

    #[test]
    fn setting_a_filter_twice_is_accepted() {
        // The reload handle must survive repeated set/restore cycles: this is
        // what a second `from_vcf` call in one process does.
        install_fmt_fallback(Some("genoray=debug"));
        install_fmt_fallback(Some("genoray::monitor=trace"));
        install_fmt_fallback(None);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
cargo test --no-default-features --features conversion logging::
```
Expected: FAIL to compile — `install_fmt_fallback` takes 0 arguments; the rank
assertions disagree with the current 4-rank table.

- [ ] **Step 3: Update the rank tables**

In `src/logging.rs`, replace `level_rank` and `event_rank`:

```rust
fn level_rank(s: &str) -> u8 {
    match s {
        "off" => 0,
        "error" => 1,
        "warning" => 2,
        "info" => 3,
        "debug" => 4,
        // Python canonicalizes before this point (`parse_log_level`), so an
        // unknown spelling here means a pure-Rust caller; default to info.
        _ => 3,
    }
}

fn event_rank(l: &tracing::Level) -> u8 {
    match *l {
        tracing::Level::ERROR => 1,
        tracing::Level::WARN => 2,
        tracing::Level::INFO => 3,
        // TRACE is treated as debug for channel gating, but TRACE events never
        // reach `on_event`: the channel layer is filtered to max DEBUG at
        // install time, so the monitor sampler's trace-level output stays
        // reachable only through the stderr fmt layer's filter.
        tracing::Level::DEBUG | tracing::Level::TRACE => 4,
    }
}
```

Also update the `CURRENT_LEVEL` initializer comment and value — `info` is now
rank 3:

```rust
static CURRENT_LEVEL: AtomicU8 = AtomicU8::new(3); // info
```

- [ ] **Step 4: Make the fmt filter reloadable**

Replace `ensure_global_subscriber` (currently line 301). The handle's full
generic type is painful to name, so store a type-erased setter closure instead:

```rust
/// Type-erased setter for the stderr fmt layer's `EnvFilter`, published by
/// `ensure_global_subscriber`. Naming `reload::Handle`'s generic parameter
/// would mean spelling out the whole `Layered<..>` subscriber type; a boxed
/// closure that captures the handle sidesteps that entirely.
static SET_FMT_FILTER: OnceLock<Box<dyn Fn(EnvFilter) + Send + Sync>> = OnceLock::new();

fn ensure_global_subscriber() {
    INSTALL.call_once(|| {
        // Channel layer is filtered to max DEBUG so `on_event` ever sees debug
        // events at all; the real per-write gate is `CURRENT_LEVEL` inside
        // `on_event`. TRACE stays out of the channel entirely.
        let channel = ChannelLayer::new().with_filter(LevelFilter::DEBUG);
        // The stderr fmt layer is ALWAYS installed, initially filtered to
        // "off" (which costs nothing), so that a later `log_filter=` can turn
        // it on. Reading the filter once at install time -- as the
        // `GENORAY_LOG` version did -- would fix it for the life of the
        // process and make a per-call filter impossible.
        let (filter, handle) = tracing_subscriber::reload::Layer::new(off_filter());
        let fmt = tracing_subscriber::fmt::layer()
            .with_target(true)
            .compact()
            .with_writer(std::io::stderr)
            .with_filter(filter);
        let subscriber = tracing_subscriber::registry().with(channel).with(fmt);
        let _ = SET_FMT_FILTER.set(Box::new(move |f| {
            let _ = handle.reload(f);
        }));
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

/// A filter that admits nothing.
fn off_filter() -> EnvFilter {
    EnvFilter::new("off")
}

/// Apply `directives` to the stderr fmt layer, or silence it when `None`.
/// An unparseable directive string silences the layer rather than panicking:
/// a bad filter must not take down a conversion that is otherwise fine.
fn set_fmt_filter(directives: Option<&str>) {
    ensure_global_subscriber();
    let filter = match directives {
        Some(d) => EnvFilter::try_new(d).unwrap_or_else(|_| off_filter()),
        None => off_filter(),
    };
    if let Some(set) = SET_FMT_FILTER.get() {
        set(filter);
    }
}
```

Add the imports this needs at the top of the file: `std::sync::OnceLock` and
`tracing_subscriber::EnvFilter` (the latter is already used at the old line 308;
keep whichever import form the file already has).

- [ ] **Step 5: Thread the filter through the two entry points**

`with_channel_subscriber` gains a `filter` parameter and restores the previous
filter on exit, exactly as it already does for sink and level:

```rust
pub fn with_channel_subscriber<R>(
    sink: EventSink,
    level: &str,
    filter: Option<&str>,
    f: impl FnOnce() -> R,
) -> R {
    ensure_global_subscriber();
    set_fmt_filter(filter);
    let prev_level = CURRENT_LEVEL.swap(level_rank(level), Ordering::Relaxed);
    let prev_sink = {
        let mut g = CURRENT_SINK.lock().unwrap();
        g.replace(sink)
    };

    struct Restore {
        prev_level: u8,
        prev_sink: Option<EventSink>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT_LEVEL.store(self.prev_level, Ordering::Relaxed);
            *CURRENT_SINK.lock().unwrap() = self.prev_sink.take();
            // The fmt filter is restored to silence rather than to a saved
            // previous value: it is only ever set by a caller of this function
            // or of `install_fmt_fallback`, both of which set it explicitly on
            // entry, so there is no ambient value to preserve.
            set_fmt_filter(None);
        }
    }
    let _restore = Restore {
        prev_level,
        prev_sink,
    };

    f()
}
```

And `install_fmt_fallback`:

```rust
/// Install the global tracing subscriber and point its stderr fmt layer at
/// `filter`, for a pure-Rust entry point (the bench bin). NOT used by the
/// Python pipeline, which goes through `with_channel_subscriber`. Since
/// `CURRENT_SINK` stays `None` outside that scope, only the fmt layer fires
/// here.
pub fn install_fmt_fallback(filter: Option<&str>) {
    set_fmt_filter(filter);
}
```

- [ ] **Step 6: Update the bench binary**

In `src/bin/bench_from_vcf_list.rs`, add a `--log-filter <directives>` argument
to whatever argument loop the file already uses (match the surrounding style),
defaulting to `None`, and pass it through:

```rust
    genoray_core::logging::install_fmt_fallback(log_filter.as_deref());
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
cargo test --no-default-features --features conversion logging::
```
Expected: PASS.

Then confirm the whole crate still builds both ways — the query-core build has
no CI coverage and has broken before:

```bash
cargo check --no-default-features --features conversion
cargo check --no-default-features
```
Expected: both clean.

- [ ] **Step 8: Commit**

```bash
git add src/logging.rs src/bin/bench_from_vcf_list.rs
git commit -m "feat(logging)!: reloadable stderr filter and a five-rank level gate

BREAKING CHANGE: with_channel_subscriber and install_fmt_fallback take a
filter argument; the stderr layer no longer reads GENORAY_LOG."
```

---

### Task 4: `src/tuning.rs` — the FFI and resolved types

**Files:**
- Create: `src/tuning.rs`
- Modify: `src/lib.rs` (add `mod tuning;` beside the other module declarations —
  this is the ONLY line Task 4 may touch in `lib.rs`, so it does not collide
  with the parallel tasks)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `tuning::TuningIn` — `#[derive(FromPyObject)]`, six `Option<usize>` fields
    named exactly as the Python dataclass's.
  - `tuning::ResolvedTuning` — six concrete `usize` fields plus
    `requested: TuningIn`.
  - `ResolvedTuning::resolve(requested: TuningIn, concurrent_chroms: usize, reader_workers: usize) -> ResolvedTuning`
  - `ResolvedTuning::with_merge_threads(self, planner_default: usize) -> Self`
  - Six `*_src(&self) -> &'static str` accessors returning `"explicit"` or
    `"planner"`.

- [ ] **Step 1: Write the failing tests**

Create `src/tuning.rs` containing ONLY the test module first, so the failure is
about missing items rather than a missing file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_fields_take_planner_defaults() {
        let r = ResolvedTuning::resolve(TuningIn::default(), 4, 6)
            .with_merge_threads(12);
        assert_eq!(r.concurrent_chroms, 4);
        assert_eq!(r.reader_workers, 6);
        assert_eq!(r.overshard, crate::orchestrator::OVERSHARD_FACTOR);
        assert_eq!(r.dense_cap, crate::orchestrator::VCF_LIST_DENSE_CHANNEL_CAP);
        assert_eq!(r.merge_threads, 12);
        assert_eq!(r.sample_interval, DEFAULT_SAMPLE_INTERVAL_SECS);
    }

    #[test]
    fn unset_fields_report_the_planner_as_their_source() {
        let r = ResolvedTuning::resolve(TuningIn::default(), 4, 6)
            .with_merge_threads(12);
        assert_eq!(r.concurrent_chroms_src(), "planner");
        assert_eq!(r.reader_workers_src(), "planner");
        assert_eq!(r.overshard_src(), "planner");
        assert_eq!(r.dense_cap_src(), "planner");
        assert_eq!(r.merge_threads_src(), "planner");
        assert_eq!(r.sample_interval_src(), "planner");
    }

    #[test]
    fn explicit_fields_win_and_report_themselves() {
        let requested = TuningIn {
            overshard: Some(40),
            dense_cap: Some(24),
            merge_threads: Some(2),
            sample_interval: Some(0),
            ..TuningIn::default()
        };
        let r = ResolvedTuning::resolve(requested, 4, 6).with_merge_threads(12);
        assert_eq!(r.overshard, 40);
        assert_eq!(r.dense_cap, 24);
        assert_eq!(r.merge_threads, 2);
        assert_eq!(r.sample_interval, 0);
        assert_eq!(r.overshard_src(), "explicit");
        assert_eq!(r.dense_cap_src(), "explicit");
        assert_eq!(r.merge_threads_src(), "explicit");
        // 0 is a real, meaningful request here -- it disables the sampler --
        // so it must NOT be mistaken for "unset".
        assert_eq!(r.sample_interval_src(), "explicit");
    }

    #[test]
    fn concurrency_and_workers_come_from_the_planner_but_report_the_request() {
        // The planner has already honoured-or-refused these two by the time
        // `resolve` runs, so the resolved value is the planner's either way;
        // only the SOURCE tag distinguishes them.
        let requested = TuningIn {
            concurrent_chroms: Some(2),
            reader_workers: Some(20),
            ..TuningIn::default()
        };
        let r = ResolvedTuning::resolve(requested, 2, 20).with_merge_threads(12);
        assert_eq!(r.concurrent_chroms, 2);
        assert_eq!(r.reader_workers, 20);
        assert_eq!(r.concurrent_chroms_src(), "explicit");
        assert_eq!(r.reader_workers_src(), "explicit");
    }

    #[test]
    fn with_merge_threads_does_not_override_an_explicit_request() {
        let requested = TuningIn {
            merge_threads: Some(2),
            ..TuningIn::default()
        };
        let r = ResolvedTuning::resolve(requested, 4, 6).with_merge_threads(99);
        assert_eq!(r.merge_threads, 2);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
cargo test --no-default-features --features conversion tuning::
```
Expected: FAIL — the module is not declared, and `ResolvedTuning`/`TuningIn` do
not exist.

- [ ] **Step 3: Write the implementation**

Put this ABOVE the test module in `src/tuning.rs`:

```rust
//! Explicit scheduling knobs, replacing the `GENORAY_*` environment channel.
//!
//! Two types, with a deliberate split:
//!
//! - [`TuningIn`] is what Python sends: every field optional, `None` meaning
//!   "planner's choice". It is the caller's REQUEST.
//! - [`ResolvedTuning`] is what the pipeline runs on: every field concrete. It
//!   keeps the original request so the `pipeline config` log line can say, per
//!   field, whether the value came from the caller or the planner -- the
//!   distinction that a `GENORAY_*` override could never report.

use pyo3::FromPyObject;

/// Monitor sampling cadence when the caller does not choose one. Matches the
/// old `GENORAY_SAMPLE_INTERVAL` default.
pub const DEFAULT_SAMPLE_INTERVAL_SECS: usize = 5;

/// A caller's tuning request, extracted from the Python `Tuning` dataclass by
/// attribute name. Field names MUST stay in lockstep with
/// `python/genoray/_tuning.py`.
#[derive(FromPyObject, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TuningIn {
    pub concurrent_chroms: Option<usize>,
    pub reader_workers: Option<usize>,
    pub overshard: Option<usize>,
    pub dense_cap: Option<usize>,
    pub merge_threads: Option<usize>,
    pub sample_interval: Option<usize>,
}

/// The concrete values this run uses, plus the request they came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedTuning {
    pub concurrent_chroms: usize,
    pub reader_workers: usize,
    pub overshard: usize,
    pub dense_cap: usize,
    pub merge_threads: usize,
    pub sample_interval: usize,
    /// The caller's original request, kept so `*_src` can report provenance
    /// without a parallel bool per field.
    pub requested: TuningIn,
}

fn tag(explicit: bool) -> &'static str {
    if explicit {
        "explicit"
    } else {
        "planner"
    }
}

impl ResolvedTuning {
    /// Build from the planner's `(cc, w)` decision plus the caller's request.
    ///
    /// `concurrent_chroms` and `reader_workers` are passed in rather than read
    /// from `requested` because the planner has already honoured-or-refused
    /// them against `max_mem`: the value that actually runs is the planner's
    /// output, and `requested` only supplies the source tag.
    ///
    /// `merge_threads` cannot be defaulted here -- its planner default is
    /// `processing_threads`, which is computed from `concurrent_chroms`
    /// downstream -- so it is left at 0 until [`Self::with_merge_threads`].
    pub fn resolve(
        requested: TuningIn,
        concurrent_chroms: usize,
        reader_workers: usize,
    ) -> Self {
        Self {
            concurrent_chroms,
            reader_workers,
            overshard: requested
                .overshard
                .unwrap_or(crate::orchestrator::OVERSHARD_FACTOR)
                .max(1),
            dense_cap: requested
                .dense_cap
                .unwrap_or(crate::orchestrator::VCF_LIST_DENSE_CHANNEL_CAP)
                .max(1),
            merge_threads: 0,
            sample_interval: requested
                .sample_interval
                .unwrap_or(DEFAULT_SAMPLE_INTERVAL_SECS),
            requested,
        }
    }

    /// Fill `merge_threads` from the planner's `processing_threads`, unless the
    /// caller asked for a specific value.
    pub fn with_merge_threads(mut self, planner_default: usize) -> Self {
        self.merge_threads = self
            .requested
            .merge_threads
            .unwrap_or(planner_default)
            .max(1);
        self
    }

    pub fn concurrent_chroms_src(&self) -> &'static str {
        tag(self.requested.concurrent_chroms.is_some())
    }
    pub fn reader_workers_src(&self) -> &'static str {
        tag(self.requested.reader_workers.is_some())
    }
    pub fn overshard_src(&self) -> &'static str {
        tag(self.requested.overshard.is_some())
    }
    pub fn dense_cap_src(&self) -> &'static str {
        tag(self.requested.dense_cap.is_some())
    }
    pub fn merge_threads_src(&self) -> &'static str {
        tag(self.requested.merge_threads.is_some())
    }
    pub fn sample_interval_src(&self) -> &'static str {
        tag(self.requested.sample_interval.is_some())
    }
}
```

- [ ] **Step 4: Declare the module**

In `src/lib.rs`, add `mod tuning;` beside the other `mod` declarations. It must
be reachable from the conversion feature; if the neighbouring modules are gated
(e.g. `#[cfg(feature = "conversion")]`), match that gating exactly.

Then check `VCF_LIST_DENSE_CHANNEL_CAP` and `OVERSHARD_FACTOR` are visible from
`tuning.rs`. Both live in `orchestrator.rs`; `OVERSHARD_FACTOR` is
`pub(crate)` and `VCF_LIST_DENSE_CHANNEL_CAP` is exported to Python, so both
should already be reachable. If `OVERSHARD_FACTOR`'s visibility is too narrow,
widen it to `pub(crate)` — do not copy the value.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
cargo test --no-default-features --features conversion tuning::
```
Expected: PASS, 5 tests.

- [ ] **Step 6: Commit**

```bash
git add src/tuning.rs src/lib.rs
git commit -m "feat(tuning): add TuningIn and ResolvedTuning with provenance tags"
```

---

## WAVE B — Tasks 5–8, strictly serial (each depends on the last)

### Task 5: Thread `ResolvedTuning` through the orchestrator; delete the env reads

**Files:**
- Modify: `src/orchestrator.rs` — delete `bench_env` (`:65`),
  `bench_concurrent_chroms` (`:79`), `bench_env_reader_workers` (`:90`),
  `bench_overshard` (`:97`); `process_chromosome` (`:381`) gains a parameter;
  the `dense_cap` read (`:456`) and `merge_threads` read (`:1130`) become field
  reads
- Modify: `src/monitor.rs:181` (delete `sample_interval_secs`), `:309`
  (`spawn_sampler` gains a parameter)
- Delete: `src/trace.rs`
- Modify: `src/lib.rs` (drop `mod trace;`), `src/executor.rs`,
  `src/shard_exec.rs` (drop `trace_ll!` call sites)

**Interfaces:**
- Consumes: `tuning::ResolvedTuning` (Task 4).
- Produces:
  - `orchestrator::process_chromosome(..., tuning: crate::tuning::ResolvedTuning, ...)`
    — the new parameter goes immediately after `processing_threads`, keeping the
    existing argument order otherwise.
  - `monitor::spawn_sampler(chrom, tx_dense, tx_sparse, tx_long, stop, probes, interval_secs: u64)`
    — `interval_secs` appended last.
  - No `bench_*` functions and no `std::env::var` remain in these files.

- [ ] **Step 1: Write the failing test**

Create `tests/test_no_env_vars.py` — the regression guard that stops the channel
growing back:

```python
"""The GENORAY_* configuration channel is gone; keep it gone.

Every knob is a Python or CLI argument. An environment read is invisible to the
`pipeline config` banner, cannot be validated, and silently wins over an
explicit argument -- the exact combination that cost a downstream user a day of
misdiagnosis on genoray 4.0.1 (PR #174).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# `svar1_reader.rs` uses `std::env::temp_dir()` for scratch paths, which is not
# configuration. Nothing else in src/ may read the environment.
_ALLOWED = {"src/svar1_reader.rs"}

_ENV_READ = re.compile(r"std::env::var|env::var_os|std::env::var_os")


def _rg(pattern: str, *paths: str) -> list[str]:
    proc = subprocess.run(
        ["rg", "--no-heading", "--line-number", pattern, *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # 1 == no matches
        pytest.fail(f"rg failed: {proc.stderr}")
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def test_no_environment_reads_in_rust_sources():
    offenders = [
        line
        for line in _rg(r"std::env::var|env::var_os", "src")
        if not any(line.startswith(allowed) for allowed in _ALLOWED)
    ]
    assert offenders == [], (
        "environment reads found in src/; every knob must be an explicit "
        "argument:\n" + "\n".join(offenders)
    )


def test_no_genoray_env_var_names_anywhere_in_the_package():
    offenders = _rg(r"GENORAY_[A-Z_]+", "src", "python")
    assert offenders == [], (
        "GENORAY_* names found in shipped code:\n" + "\n".join(offenders)
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run test tests/test_no_env_vars.py`
Expected: FAIL — both assertions list the remaining `orchestrator.rs`,
`monitor.rs`, `trace.rs` and `_logging.py` hits.

- [ ] **Step 3: Delete `src/trace.rs` and its call sites**

```bash
rm src/trace.rs
rg -n 'trace_ll!|crate::trace|mod trace' src/
```

Remove `mod trace;` from `src/lib.rs` and delete every `trace_ll!(...)`
statement the grep lists (they are standalone statements in `orchestrator.rs`,
`executor.rs` and `shard_exec.rs`; deleting the whole statement is correct — do
not replace them with `tracing::trace!`, which would change what the monitor
emits). Re-run the grep until it is empty.

- [ ] **Step 4: Delete the `bench_*` helpers**

In `src/orchestrator.rs`, delete `bench_env`, `bench_concurrent_chroms`,
`bench_env_reader_workers` and `bench_overshard` outright, along with their doc
comments. Keep `OVERSHARD_FACTOR` and `VCF_LIST_DENSE_CHANNEL_CAP` — they are
now the planner defaults consumed by `ResolvedTuning::resolve`.

- [ ] **Step 5: Give `process_chromosome` the resolved tuning**

Add the parameter immediately after `processing_threads`:

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
    tuning: crate::tuning::ResolvedTuning,
    signatures: bool,
    fields: &[crate::field::FieldSpec],
    sink: &crate::logging::EventSink,
) -> Result<u64, ConversionError> {
```

Replace the `dense_cap` read (was line 456):

```rust
    let (tx_dense, rx_dense) = bounded::<crate::types::DenseChunk>(tuning.dense_cap);
```

Replace the `merge_threads` read (was line 1130):

```rust
    let merge_threads = tuning.merge_threads;
```

Delete both of the BENCH-ONLY comment blocks that explained the env vars, and
replace the `SourceSpec::Vcf` branch's `overshard`/`reader_workers` env lookups
(if any remain after #174) with `tuning.overshard` / the `reader_workers` the
`SourceSpec::Vcf` variant already carries.

- [ ] **Step 6: Give the sampler its interval**

In `src/monitor.rs`, delete `sample_interval_secs` entirely and add a parameter:

```rust
pub fn spawn_sampler(
    chrom: String,
    tx_dense: Sender<DenseChunk>,
    tx_sparse: Sender<SparseChunk>,
    tx_long: Sender<Vec<u8>>,
    stop: Arc<AtomicBool>,
    probes: PipelineProbes,
    interval_secs: u64,
) -> thread::JoinHandle<()> {
```

and inside, replace `let interval_secs = sample_interval_secs();` with nothing —
the parameter shadows it. The existing `if interval_secs == 0 { return; }`
disable path stays exactly as it is.

Update the call in `process_chromosome` to pass
`tuning.sample_interval as u64`.

- [ ] **Step 7: Fix the four `process_chromosome` call sites**

They are at `src/lib.rs:378`, `:668`, `:1379` and `src/orchestrator.rs:1398`.
Each must pass a `ResolvedTuning`. For now — Task 6 does the real resolution —
thread through a value built at the top of each caller:

```rust
    let tuning = crate::tuning::ResolvedTuning::resolve(
        crate::tuning::TuningIn::default(),
        concurrent_chroms,
        reader_workers,
    )
    .with_merge_threads(processing_threads);
```

In `orchestrator::run_vcf_list`, which has no `reader_workers`, pass `1`.

- [ ] **Step 8: Build and test**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
cargo test --no-default-features --features conversion
cargo check --no-default-features
```
Expected: all tests pass (341-ish), both builds clean.

- [ ] **Step 9: Rebuild the extension and run the guard test**

```bash
maturin develop --release
pixi run test tests/test_no_env_vars.py
```
Expected: `test_no_environment_reads_in_rust_sources` PASSES.
`test_no_genoray_env_var_names_anywhere_in_the_package` still FAILS on
`python/genoray/_logging.py` only if Task 2 has not landed; if Tasks 1-4 were
merged first, it passes too.

- [ ] **Step 10: Commit**

```bash
git add -A src tests/test_no_env_vars.py
git commit -m "refactor(svar2)!: thread ResolvedTuning through the orchestrator

Deletes bench_env and friends, src/trace.rs, and the GENORAY_DENSE_CAP,
GENORAY_MERGE_THREADS, GENORAY_SAMPLE_INTERVAL and GENORAY_TRACE reads.

BREAKING CHANGE: GENORAY_DENSE_CAP, GENORAY_MERGE_THREADS,
GENORAY_SAMPLE_INTERVAL and GENORAY_TRACE no longer do anything."
```

---

### Task 6: Wire the pyo3 entry points and unify the banners

**Files:**
- Modify: `src/lib.rs` — `run_conversion_pipeline` (`:153`),
  `run_pgen_conversion_pipeline` (`:459`), `run_slice_view` (`:827`),
  `run_vcf_list_conversion_pipeline` (`:1179`),
  `run_svar1_conversion_pipeline` (`:1270`); the three banner blocks
  (`:328-347`, `:625-632`, `:1364-1368`)
- Create: `tests/test_pipeline_config_banner.py`

**Interfaces:**
- Consumes: `tuning::ResolvedTuning`, `logging::with_channel_subscriber`'s new
  arity.
- Produces: each pyo3 entry point takes two new trailing keyword arguments
  before `receiver`:
  - `tuning: Option<crate::tuning::TuningIn>` (absent on `run_slice_view`)
  - `log_filter: Option<String>` (on all five)

  and drops `reader_workers: Option<usize>` from `run_conversion_pipeline`
  (it now arrives inside `tuning`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_config_banner.py`:

```python
"""The `pipeline config` line must report every knob and where it came from.

On genoray 4.0.1 this line printed the planner's `reader_workers` while an
environment variable had set a different value, and a downstream operator spent
a day chasing the difference. With the environment channel gone the line is
correct by construction -- these tests hold it to reporting provenance too.
"""

from __future__ import annotations

import re

import pytest

from genoray import SparseVar2, Tuning

pytest_plugins = ()


def _banner(caplog_text: str) -> dict[str, str]:
    """Parse the flat `key=value` pairs out of the `pipeline config` line."""
    for line in caplog_text.splitlines():
        if "pipeline config" not in line:
            continue
        return dict(re.findall(r"(\w+)=([^\s]+)", line))
    raise AssertionError(f"no `pipeline config` line found in:\n{caplog_text}")


def test_explicit_values_are_tagged_explicit(tmp_path, small_vcf, capsys):
    out = tmp_path / "explicit.svar"
    SparseVar2.from_vcf(
        out,
        small_vcf,
        no_reference=True,
        tuning=Tuning(reader_workers=2, overshard=3, dense_cap=7),
        log_filter="genoray=info",
    )
    fields = _banner(capsys.readouterr().err)
    assert fields["reader_workers"] == "2"
    assert fields["reader_workers_src"] == "explicit"
    assert fields["overshard"] == "3"
    assert fields["overshard_src"] == "explicit"
    assert fields["dense_cap"] == "7"
    assert fields["dense_cap_src"] == "explicit"


def test_unset_values_are_tagged_planner(tmp_path, small_vcf, capsys):
    out = tmp_path / "planner.svar"
    SparseVar2.from_vcf(
        out, small_vcf, no_reference=True, log_filter="genoray=info"
    )
    fields = _banner(capsys.readouterr().err)
    for knob in (
        "concurrent_chroms",
        "reader_workers",
        "overshard",
        "dense_cap",
        "merge_threads",
        "sample_interval",
    ):
        assert fields[f"{knob}_src"] == "planner", knob


def test_banner_omits_knobs_the_backend_cannot_use(tmp_path, small_pgen, capsys):
    out = tmp_path / "pgen.svar"
    SparseVar2.from_pgen(
        out, small_pgen, no_reference=True, log_filter="genoray=info"
    )
    fields = _banner(capsys.readouterr().err)
    # from_pgen pins P=1 and never shards within a contig, so advertising these
    # would be advertising an inert knob.
    assert "overshard" not in fields
    assert "reader_workers_src" not in fields
    assert fields["concurrent_chroms_src"] == "planner"
```

Reuse whatever small-VCF and small-PGEN fixtures the neighbouring conversion
tests already use — check `tests/test_svar2_from_vcf.py` and
`tests/conftest.py` for their names and import them rather than building new
ones. If the fixtures are module-local, move them to `conftest.py` as part of
this task.

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run test tests/test_pipeline_config_banner.py`
Expected: FAIL — `from_vcf() got an unexpected keyword argument 'tuning'`.

- [ ] **Step 3: Add the parameters to `run_conversion_pipeline`**

In its `#[pyo3(signature = (...))]` attribute and its Rust parameter list,
replace `reader_workers: Option<usize>` with nothing and add, immediately before
`receiver`:

```rust
    tuning: Option<crate::tuning::TuningIn>,
    log_filter: Option<String>,
```

with signature defaults `tuning = None, log_filter = None`.

- [ ] **Step 4: Resolve, and rewrite the VCF banner**

Replace the `requested_workers` line (was `lib.rs:279`) and the block that
follows it:

```rust
            let requested = tuning.unwrap_or_default();

            let sharded = crate::budget::plan_sharded(crate::budget::PlanInputs {
                usable_cores: available_cores.saturating_sub(1).max(1),
                n_contigs: chroms.len(),
                n_samples: samples.len(),
                chunk_bytes,
                max_mem_bytes,
                reader_workers: requested.reader_workers,
                ram: crate::budget::RamLaw::VCF,
            });
```

then, where `concurrent_chroms`/`overshard` were derived from the `bench_*`
helpers:

```rust
            // An explicit `concurrent_chroms` is honoured; `plan_sharded` has
            // already refused it against `max_mem` if it did not fit.
            let concurrent_chroms = requested
                .concurrent_chroms
                .unwrap_or(sharded.concurrent_chroms)
                .max(1);
            let reader_workers = sharded.reader_workers;
            let resolved = crate::tuning::ResolvedTuning::resolve(
                requested,
                concurrent_chroms,
                reader_workers,
            );
            let overshard = resolved.overshard;
```

and after `processing_threads` is computed:

```rust
            let resolved = resolved.with_merge_threads(processing_threads);
```

Then extend the `tracing::info!` banner with the six knobs and their tags,
keeping every field it already has:

```rust
            tracing::info!(
                concurrent_chroms = resolved.concurrent_chroms,
                concurrent_chroms_src = resolved.concurrent_chroms_src(),
                reader_workers = resolved.reader_workers,
                reader_workers_src = resolved.reader_workers_src(),
                overshard = resolved.overshard,
                overshard_src = resolved.overshard_src(),
                dense_cap = resolved.dense_cap,
                dense_cap_src = resolved.dense_cap_src(),
                merge_threads = resolved.merge_threads,
                merge_threads_src = resolved.merge_threads_src(),
                sample_interval = resolved.sample_interval,
                sample_interval_src = resolved.sample_interval_src(),
                htslib_threads,
                monolithic_reader_active,
                exact_counts = costs.exact_counts,
                planned_units,
                pending_budget_mb = pending_budget_bytes as f64 / 1e6,
                sharded_vcf_active,
                processing_threads,
                "pipeline config"
            );
```

Pass `resolved` to the `process_chromosome` call at `:378`, replacing the
placeholder Task 5 added.

- [ ] **Step 5: Do the same for PGEN, keeping the cap falsifiable**

In `run_pgen_conversion_pipeline`, add the same two parameters. The
`PGEN_MAX_CONCURRENT` clamp must apply to the DERIVE path only — an explicit
request is honoured past it and warned about, because clamping the escape hatch
would make the constant unfalsifiable (this is why the env var was applied after
the cap):

```rust
            let requested = tuning.unwrap_or_default();
            let planned = sharded
                .concurrent_chroms
                .min(crate::budget::PGEN_MAX_CONCURRENT);
            let concurrent_chroms = match requested.concurrent_chroms {
                Some(cc) => {
                    let cc = cc.max(1);
                    if cc > crate::budget::PGEN_MAX_CONCURRENT {
                        tracing::warn!(
                            concurrent_chroms = cc,
                            measured_knee = crate::budget::PGEN_MAX_CONCURRENT,
                            "explicit concurrent_chroms exceeds the measured \
                             PGEN concurrency knee; honouring it so the knee \
                             stays falsifiable, but peak RSS is unmodelled above it"
                        );
                    }
                    cc
                }
                None => planned,
            };
```

Then build `resolved` the same way and extend the `pipeline config (PGEN)`
banner with the four applicable knobs (`concurrent_chroms`, `dense_cap`,
`merge_threads`, `sample_interval`) and their `_src` tags. Do NOT emit
`overshard` or `reader_workers_src` — this backend cannot use them.

- [ ] **Step 6: Do the same for SVAR1 and vcf-list**

`run_svar1_conversion_pipeline`: add both parameters; honour an explicit
`concurrent_chroms`; extend the `pipeline config (SVAR1)` banner with
`concurrent_chroms`, `dense_cap`, `merge_threads`, `sample_interval` and tags.

`run_vcf_list_conversion_pipeline`: add both parameters. It has no
`concurrent_chroms` (the pipeline is sequential), so build the resolved value
with `ResolvedTuning::resolve(requested, 1, 1)` and pass it into
`orchestrator::run_vcf_list`, which forwards it to `process_chromosome`. It has
no banner today; add one:

```rust
            tracing::info!(
                dense_cap = resolved.dense_cap,
                dense_cap_src = resolved.dense_cap_src(),
                merge_threads = resolved.merge_threads,
                merge_threads_src = resolved.merge_threads_src(),
                sample_interval = resolved.sample_interval,
                sample_interval_src = resolved.sample_interval_src(),
                "pipeline config (VCF list)"
            );
```

`run_slice_view`: add `log_filter: Option<String>` only — no `tuning`.

- [ ] **Step 7: Update every `with_channel_subscriber` call**

All five entry points call it. Each gains the filter argument:

```rust
        crate::logging::with_channel_subscriber(sink.clone(), &level, log_filter.as_deref(), || {
```

- [ ] **Step 8: Build, rebuild the extension, and test**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
cargo test --no-default-features --features conversion
cargo check --no-default-features
maturin develop --release
```
Expected: Rust tests pass; both builds clean; the `.so` mtime changes.

Python tests will still fail here — `_svar2.py` does not pass `tuning` yet.
That is Task 7. Do not "fix" it by editing `_svar2.py` in this task.

- [ ] **Step 9: Commit**

```bash
git add src/lib.rs tests/test_pipeline_config_banner.py
git commit -m "feat(svar2)!: take Tuning at the FFI and report knob provenance

Every pipeline config line now reports each applicable knob with a
<field>_src tag saying whether the caller or the planner chose it.

BREAKING CHANGE: GENORAY_CONCURRENT_CHROMS, GENORAY_READER_WORKERS and
GENORAY_OVERSHARD no longer do anything; run_conversion_pipeline takes
tuning= instead of reader_workers=."
```

---

### Task 7: Wire the Python `SparseVar2` methods

**Files:**
- Modify: `python/genoray/_svar2.py` — `from_vcf` (`:645`), `from_pgen`
  (`:907`), `from_vcf_list` (`:1311`), `from_svar1` (`:1723`), and the
  slice/view method at `:490`

**Interfaces:**
- Consumes: `Tuning` (Task 1), `parse_log_level` (Task 2), the new pyo3
  signatures (Task 6).
- Produces: `tuning: Tuning | None = None` on the four conversion methods;
  `log_filter: str | None = None` on all five; `reader_workers` removed from
  `from_vcf`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tuning.py`:

```python
def test_from_vcf_rejects_the_old_flat_reader_workers_kwarg(tmp_path, small_vcf):
    from genoray import SparseVar2

    with pytest.raises(TypeError, match="reader_workers"):
        SparseVar2.from_vcf(
            tmp_path / "x.svar", small_vcf, no_reference=True, reader_workers=2
        )


def test_from_pgen_rejects_a_vcf_only_knob(tmp_path, small_pgen):
    from genoray import SparseVar2, Tuning

    with pytest.raises(ValueError, match="overshard"):
        SparseVar2.from_pgen(
            tmp_path / "x.svar",
            small_pgen,
            no_reference=True,
            tuning=Tuning(overshard=4),
        )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run test tests/test_tuning.py -k reader_workers or vcf_only`
Expected: FAIL — `from_vcf` still accepts `reader_workers`.

- [ ] **Step 3: Update `from_vcf`**

Remove `reader_workers: int | None = None` from the signature and add, after
`progress`:

```python
        tuning: Tuning | None = None,
        log_level: str | int = "info",
        log_filter: str | None = None,
```

(Widen `log_level`'s annotation from the old four-value `Literal` to `str | int`
on all five methods; `parse_log_level` is what validates it.)

Delete the hand-rolled `reader_workers` validation block (was `:781-785`) — the
dataclass does it now — and add, near the top of the body:

```python
        tuning = tuning if tuning is not None else Tuning()
        tuning._check_backend("vcf")
```

Then pass it across the FFI:

```python
        with write_reporting(progress, log_level) as (rx, level):
            return _core.run_conversion_pipeline(
                ...,
                tuning=tuning._as_ffi(),
                log_filter=log_filter,
                log_level=level,
                receiver=rx,
            )
```

Note `_as_ffi()` returns a dict; pyo3's `FromPyObject` derive reads named
attributes, so pass the `Tuning` object itself if the derive is attribute-based.
Verify which one Task 4's `TuningIn` extracts — if `cargo test` fails with a
`TypeError` about attribute access, pass `tuning` rather than `tuning._as_ffi()`
and delete `_as_ffi`.

Update the docstring: document `tuning`, `log_filter`, and the new `log_level`
values; delete the `GENORAY_LOG` sentences (they appear at `:549`, `:729`,
`:1044`, `:1538`, `:1811`).

- [ ] **Step 4: Update the other four methods**

Same shape, with the backend name changed: `from_pgen` → `_check_backend("pgen")`,
`from_vcf_list` → `"vcf_list"`, `from_svar1` → `"svar1"`. The slice/view method
gets `log_filter` and the widened `log_level` only — no `tuning`.

- [ ] **Step 5: Run the whole Python suite**

```bash
pixi run test
```
Expected: `tests/test_tuning.py`, `tests/test_log_levels.py` and
`tests/test_pipeline_config_banner.py` all pass. The five files Task 8 migrates
still fail — that is expected and is Task 8's job.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2.py tests/test_tuning.py
git commit -m "feat(svar2)!: take tuning= and log_filter= on the write methods

BREAKING CHANGE: from_vcf's reader_workers= argument moved into
Tuning(reader_workers=...); log_level no longer consults GENORAY_LOG."
```

---

### Task 8: Migrate the tests that configured through the environment

**Files:**
- Modify: `tests/test_svar2_schedule_invariance.py` (9 env uses),
  `tests/test_logging.py` (8), `tests/test_svar2_pgen_schedule_invariance.py`
  (1), `tests/test_svar2_from_vcf.py` (1), `tests/bench/test_probe.py` (2)

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: a green `pixi run test`.

- [ ] **Step 1: Migrate the schedule-invariance helpers**

In `tests/test_svar2_schedule_invariance.py`, replace `_convert`:

```python
def _convert(vcf, out, cc, w):
    SparseVar2.from_vcf(
        out,
        vcf,
        no_reference=True,
        chunk_size=CHUNK_SIZE,
        tuning=Tuning(concurrent_chroms=cc, reader_workers=w),
    )
    return _oracle.store_digest(out)
```

Delete the six-line comment above it explaining why the values had to be set on
`os.environ` — the reason is gone with the mechanism. Drop the now-unused
`monkeypatch` parameter from `_convert` and from every call site.

Do the same in `tests/test_svar2_pgen_schedule_invariance.py`, using
`Tuning(concurrent_chroms=cc)` — `reader_workers` does not apply to PGEN, and
passing it now raises.

Two more tests in the same file drive the environment directly:

`test_digest_is_invariant_across_frontier_granularities` (`:152`) opens with
three `monkeypatch.delenv`/`setenv` calls. Replace them with
`tuning=Tuning(concurrent_chroms=1, reader_workers=1)` on the `from_vcf` call
and drop the `monkeypatch` parameter. Its docstring paragraph explaining that
`GENORAY_OVERSHARD` is inert on the exact-counts tier stays true — reword it to
`Tuning(overshard=)` rather than deleting it, since that inertness is exactly
why the test drives granularity through `chunk_size`.

`test_explicit_reader_workers_matches_the_derived_default` (`:207`) exists to
prove the public `reader_workers=` argument agrees byte-for-byte with the env
var at the same value. With the env var gone there is no second channel to
agree with, so **delete this test outright**. Its stated purpose is now covered
by `test_from_vcf_reader_workers_reaches_the_planner` in
`tests/test_svar2_from_vcf.py` (which its own docstring already names as the
test that actually proves reachability) plus
`tests/test_pipeline_config_banner.py` from Task 6. Do not keep a
`Tuning(reader_workers=6)`-vs-`Tuning(reader_workers=6)` version: comparing a
call to itself proves nothing.

- [ ] **Step 2: Migrate `tests/test_logging.py`**

Delete the two `GENORAY_LOG` monkeypatch fixtures (`:57`, `:294`, `:304`) and
the regression test at `:252-270` that asserted the env var overrode the
argument and reached the channel gate — that coupling is deliberately gone.
Replace it with the equivalent argument-based assertion:

```python
def test_log_level_argument_reaches_the_channel_gate(tmp_path, small_vcf, capsys):
    """A debug-level line must reach the Python renderer when asked for.

    The old version of this test drove the level through GENORAY_LOG, which
    also silently overrode the argument. The argument is now the only channel.
    """
    out = tmp_path / "lvl.svar"
    SparseVar2.from_vcf(
        out, small_vcf, no_reference=True, progress=True, log_level="debug"
    )
    assert "[svar2]" in capsys.readouterr().out
```

- [ ] **Step 3: Migrate the remaining two**

`tests/test_svar2_from_vcf.py` and `tests/bench/test_probe.py`: replace their
env usage with `tuning=Tuning(...)` / `log_filter=` arguments. `test_probe.py`'s
docstring references `GENORAY_LOG=genoray::monitor=trace` with
`GENORAY_SAMPLE_INTERVAL=1`; that becomes
`log_filter="genoray::monitor=trace"` with `Tuning(sample_interval=1)`.

- [ ] **Step 4: Run the full suite**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
maturin develop --release
pixi run test
```
Expected: green. Confirm `tests/test_no_env_vars.py` passes both assertions now.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: drive scheduling and logging through arguments, not the environment"
```

---

## WAVE C — Tasks 9–11, dispatch in parallel (disjoint files)

### Task 9: CLI flags

**Files:**
- Create: `python/genoray/_cli/_tuning_flags.py`
- Modify: `python/genoray/_cli/__main__.py` — `write_vcf` (`:86`), `write_pgen`
  (`:268`), `write_from_svar1` (`:411`), `view_svar2` (`:651`)
- Create: `tests/cli/test_tuning_flags.py`

**Interfaces:**
- Consumes: `Tuning` (Task 1).
- Produces: `_cli._tuning_flags.TuningFlags` — a cyclopts parameter group
  dataclass with `to_tuning() -> Tuning`.

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_tuning_flags.py`:

```python
from __future__ import annotations

import subprocess
import sys

import pytest

from genoray import Tuning
from genoray._cli._tuning_flags import TuningFlags


def test_flags_convert_to_a_tuning():
    flags = TuningFlags(reader_workers=20, overshard=40, sample_interval=0)
    assert flags.to_tuning() == Tuning(
        reader_workers=20, overshard=40, sample_interval=0
    )


def test_unset_flags_stay_none():
    assert TuningFlags().to_tuning() == Tuning()


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "genoray", *args],
        capture_output=True,
        text=True,
    )


def test_vcf_only_flags_are_absent_from_the_pgen_command():
    # Rejected by argument parsing, not at runtime: a flag the backend cannot
    # use should not be spellable.
    proc = _cli("write", "pgen", "--help")
    assert "--overshard" not in proc.stdout
    assert "--reader-workers" not in proc.stdout


def test_shared_flags_are_present_on_every_write_command():
    for sub in ("vcf", "pgen"):
        out = _cli("write", sub, "--help").stdout
        for flag in ("--dense-cap", "--merge-threads", "--sample-interval",
                     "--log-filter"):
            assert flag in out, (sub, flag)


def test_vcf_command_has_the_sharded_flags():
    out = _cli("write", "vcf", "--help").stdout
    assert "--reader-workers" in out
    assert "--overshard" in out
    assert "--concurrent-chroms" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run test tests/cli/test_tuning_flags.py`
Expected: FAIL — `No module named 'genoray._cli._tuning_flags'`.

- [ ] **Step 3: Write the shared flag group**

Create `python/genoray/_cli/_tuning_flags.py`:

```python
"""One tuning flag group, shared by the write commands.

Six knobs across four commands is 24 parameter declarations if each command
spells its own; this is the single place they live. Each command composes only
the subset its backend can use -- a flag the backend would ignore is not
spellable rather than rejected at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import Parameter

from .._tuning import Tuning


@dataclass
class TuningFlags:
    """Scheduling knobs shared by every write command."""

    dense_cap: Annotated[int | None, Parameter(name="--dense-cap")] = None
    merge_threads: Annotated[int | None, Parameter(name="--merge-threads")] = None
    sample_interval: Annotated[
        int | None, Parameter(name="--sample-interval")
    ] = None
    concurrent_chroms: Annotated[
        int | None, Parameter(name="--concurrent-chroms")
    ] = None
    reader_workers: Annotated[int | None, Parameter(name="--reader-workers")] = None
    overshard: Annotated[int | None, Parameter(name="--overshard")] = None

    def to_tuning(self) -> Tuning:
        return Tuning(
            concurrent_chroms=self.concurrent_chroms,
            reader_workers=self.reader_workers,
            overshard=self.overshard,
            dense_cap=self.dense_cap,
            merge_threads=self.merge_threads,
            sample_interval=self.sample_interval,
        )
```

Check how the repo's cyclopts version composes a parameter group — look at how
`write_vcf` already groups its options. If cyclopts here does not support a
dataclass parameter group, fall back to a module-level dict of
`Annotated[...]` aliases and spell the subset per command; keep the
`to_tuning()` conversion in this module either way, so the mapping lives in one
place.

- [ ] **Step 4: Wire the four commands**

In `write_vcf`, remove the standalone `reader_workers` parameter (it is in the
group now), add the group, add `--log-filter`, and build the `Tuning`:

```python
    tuning = flags.to_tuning()
```

passing `tuning=tuning, log_filter=log_filter` into `SparseVar2.from_vcf`.

`write_vcf` already raises a helpful error when `--reader-workers` is passed
with multi-file input (`:214-219`). Generalize it: for the vcf-list branch, call
`tuning._check_backend("vcf_list")` and let its message do the work, then delete
the hand-rolled check.

`write_pgen`, `write_from_svar1`: same, with the subset each backend supports
and the matching `_check_backend` call. `view_svar2` gets `--log-filter` and the
widened `--log-level` only.

- [ ] **Step 5: Run the tests**

Run: `pixi run test tests/cli/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_cli tests/cli/test_tuning_flags.py
git commit -m "feat(cli): expose the tuning knobs and --log-filter as flags"
```

---

### Task 10: Migrate the benchmark scripts

**Files:**
- Modify: `scripts/bench_svar2/probe.py` (7 env uses),
  `scripts/bench_svar2/plans/build_plans.py` (2),
  `scripts/bench_svar2/records.py` (1),
  `scripts/bench_svar2/sweep_scale.sbatch` (1),
  `scripts/bench_svar2/ab_builds.py` (9)
- Modify: `scripts/bench_svar2/legacy_pr140/README.md` (header note only)

**Interfaces:**
- Consumes: the CLI flags (Task 9) and `Tuning` (Task 1).
- Produces: no `GENORAY_*` reference in any live bench script.

- [ ] **Step 1: Invert `ab_builds.py`'s scrubbing into an assertion**

`ab_builds.py` currently strips a `BENCH_ENV_VARS` list from the child
environment so a sweep hook cannot contaminate an A/B run. With the channel
gone there is nothing to strip — but the integrity check is still worth having,
inverted:

```python
# Every knob is now an explicit argument, so there is nothing to scrub -- but a
# stray GENORAY_* in the environment would mean the channel has grown back and
# this run's configuration is not what the command line says it is.
_STALE = sorted(k for k in os.environ if k.startswith("GENORAY_"))
if _STALE:
    raise SystemExit(
        f"GENORAY_* variables are set but no longer do anything: {_STALE}. "
        "Pass the knobs as CLI flags instead."
    )
```

Delete the `BENCH_ENV_VARS` list and the `env=` filtering that used it.

- [ ] **Step 2: Convert the sweep hooks to flags**

In `probe.py`, `build_plans.py`, `records.py` and `sweep_scale.sbatch`, replace
each `GENORAY_X=v` environment assignment with the corresponding flag on the
`genoray write` command line (`--reader-workers`, `--concurrent-chroms`,
`--overshard`, `--dense-cap`, `--merge-threads`, `--sample-interval`,
`--log-filter`). Where a script calls the Python API rather than the CLI, pass
`tuning=Tuning(...)`.

`probe.py` parses the `pipeline config` line; its parser reads flat
`key=value` pairs, so the new `<field>_src` fields need no parser change. Add
them to whatever record it emits so a sweep row records provenance.

- [ ] **Step 3: Note the legacy directory**

Add to the top of `scripts/bench_svar2/legacy_pr140/README.md`:

```markdown
> **Superseded.** These scripts drive the pre-5.0 `GENORAY_*` environment
> interface, which no longer exists. They are kept for provenance of the PR #140
> measurements and will not run against current genoray. The equivalent knobs
> are now `genoray write vcf --reader-workers/--overshard/...` and
> `genoray.Tuning`.
```

Do not rewrite the legacy scripts themselves.

- [ ] **Step 4: Verify**

```bash
rg -n 'GENORAY_' scripts/ --glob '!legacy_pr140/**'
```
Expected: only the `ab_builds.py` assertion's `"GENORAY_"` prefix string.

Then smoke-test one script end to end on a small fixture — `probe.py` against a
test VCF is enough to prove the flags reach the pipeline.

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "chore(bench): drive sweeps through CLI flags instead of GENORAY_*"
```

---

### Task 11: Documentation

**Files:**
- Modify: `skills/genoray-api/SKILL.md` (`:423-428`, `:1317`)
- Modify: `docs/source/svar.md`
- Modify: `scripts/bench_svar2/README.md:63`
- Modify: `docs/roadmap/svar-2.md` (add an M17 entry)

**Interfaces:**
- Consumes: the finished public surface.
- Produces: documentation that matches it.

- [ ] **Step 1: Update the skill**

`skills/genoray-api/SKILL.md` is mandatory for any public-name change. Replace
the `GENORAY_LOG` paragraphs at `:423-428` and `:1317` with:

- the `Tuning` dataclass, its six fields, and the applicability table copied
  from `python/genoray/_tuning.py`'s docstring;
- the accepted `log_level` values including the integer forms, the `"critical"`
  alias, and the explicit note that `"warn"` is rejected;
- `log_filter` and one worked example
  (`log_filter="genoray::monitor=trace"`);
- the statement that no environment variable configures genoray.

- [ ] **Step 2: Update the user docs**

`docs/source/svar.md`: same content, prose form, plus a short "migrating from
`GENORAY_*`" table mapping each removed variable to its replacement:

| removed | replacement |
|---|---|
| `GENORAY_CONCURRENT_CHROMS` | `Tuning(concurrent_chroms=)` / `--concurrent-chroms` |
| `GENORAY_READER_WORKERS` | `Tuning(reader_workers=)` / `--reader-workers` |
| `GENORAY_OVERSHARD` | `Tuning(overshard=)` / `--overshard` |
| `GENORAY_DENSE_CAP` | `Tuning(dense_cap=)` / `--dense-cap` |
| `GENORAY_MERGE_THREADS` | `Tuning(merge_threads=)` / `--merge-threads` |
| `GENORAY_SAMPLE_INTERVAL` | `Tuning(sample_interval=)` / `--sample-interval` |
| `GENORAY_TRACE` | `log_filter="genoray=trace"` |
| `GENORAY_LOG` | `log_level=` (levels) or `log_filter=` (directives) |

- [ ] **Step 3: Update the bench README**

`scripts/bench_svar2/README.md:63` describes reader-worker behaviour in terms of
`GENORAY_READER_WORKERS`. Reword to the flag, and document that every sweep row
should now record the `<field>_src` tags.

- [ ] **Step 4: Record the milestone**

Add an `M17` entry to `docs/roadmap/svar-2.md` recording what shipped: the
`Tuning` object, Python-convention levels, `log_filter`, the removal of all
eight variables, and the `tests/test_no_env_vars.py` guard that keeps them gone.
Note explicitly that M16's public-API note is superseded, the same way M16
superseded M15's.

- [ ] **Step 5: Verify no stale references remain**

```bash
rg -n 'GENORAY_' docs/ skills/ --glob '!docs/superpowers/**'
```
Expected: only the migration table in `docs/source/svar.md` and the roadmap
entry. `docs/superpowers/**` is historical record and is excluded on purpose —
do not rewrite past plans and specs.

- [ ] **Step 6: Commit**

```bash
git add docs skills
git commit -m "docs: document Tuning, log_filter, and the GENORAY_* removal"
```

---

## Final verification

- [ ] **Full Rust suite**

```bash
export CARGO_TARGET_DIR=/local/$USER/genoray-target
cargo test --no-default-features --features conversion
cargo check --no-default-features
```
Expected: all pass, both builds clean.

- [ ] **Fresh extension + full Python suite**

```bash
maturin develop --release
pixi run test
```
Expected: green. Confirm the `.so` mtime changed before believing the result.

- [ ] **The guard actually guards**

Temporarily add `let _ = std::env::var("GENORAY_NOPE");` to `src/orchestrator.rs`,
run `pixi run test tests/test_no_env_vars.py`, confirm it FAILS, then remove the
line. A guard test that cannot fail is not a guard.

- [ ] **Byte-identity across a tuning sweep**

Run the schedule-invariance tests specifically — they are the oracle that proves
these knobs change scheduling and nothing else:

```bash
pixi run test tests/test_svar2_schedule_invariance.py tests/test_svar2_pgen_schedule_invariance.py
```
Expected: identical digests across every `(cc, w)` combination.

- [ ] **Open the PR**

Title: `feat!: explicit tuning API, no environment variables`.
Body must state: the eight removed variables and their replacements; that this
is a major bump; that `reader_workers=` moved into `Tuning` and was never
released; and that `log_filter` is the replacement for
`GENORAY_LOG=genoray::monitor=trace`, which a downstream All-of-Us pipeline uses
in production (PR #174 comment).

---

## Self-Review

**Spec coverage.** Section A (`Tuning`, FFI shape, applicability) → Tasks 1, 4,
6, 7. Section B (honour-or-refuse, PGEN falsifiability, single resolution point,
banner provenance) → Tasks 5, 6. Section C (Python levels, `critical` alias,
`warn` rejection, `log_filter`, reload layer, `trace.rs` deletion) → Tasks 2, 3,
5. Section D (CLI) → Task 9. Section E (migration, five new tests, docs) →
Tasks 5, 8, 10, 11. Public API impact → the PR body in Final verification.

**Known gap, deliberate.** The spec's "byte-identity across a `Tuning` sweep"
reuses #174's existing schedule-invariance oracle rather than adding a new
sweep; Task 8 migrates it and Final verification runs it. No new test file is
warranted for a property an existing oracle already checks.

**Type consistency.** `Tuning` field names are identical in
`python/genoray/_tuning.py`, `src/tuning.rs`'s `TuningIn`, and the CLI's
`TuningFlags`. `parse_log_level` returns exactly the five names `level_rank`
matches on. `ResolvedTuning::resolve` + `with_merge_threads` is the only
construction path, used identically in all four entry points.

**One open implementation question, flagged in place.** Task 7 Step 3 notes that
pyo3's `FromPyObject` derive reads named *attributes*, so passing the `Tuning`
object directly may be correct and `_as_ffi()` redundant. The task tells the
implementer how to tell which, and to delete `_as_ffi` if so. This is a
five-minute empirical check, not a design decision.
