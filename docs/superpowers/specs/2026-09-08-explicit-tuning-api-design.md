# Explicit tuning: a `Tuning` object, Python-convention log levels, no environment

Status: approved design, not yet implemented.
Supersedes the `GENORAY_*` configuration channel entirely.

## Problem

Every scheduling knob in the SVAR2 conversion pipeline is configured through an
environment variable. Eight of them exist on `main` (nine read sites —
`GENORAY_LOG` is read twice):

| variable | read at | public equivalent today |
|---|---|---|
| `GENORAY_CONCURRENT_CHROMS` | `orchestrator.rs:80` via `bench_concurrent_chroms` | none |
| `GENORAY_READER_WORKERS` | `orchestrator.rs:91` via `bench_env_reader_workers` | `reader_workers=` (#174) — **env wins** |
| `GENORAY_OVERSHARD` | `orchestrator.rs:98` via `bench_overshard` | none |
| `GENORAY_DENSE_CAP` | `orchestrator.rs:456` | none |
| `GENORAY_MERGE_THREADS` | `orchestrator.rs:1130` | none |
| `GENORAY_SAMPLE_INTERVAL` | `monitor.rs:181` | none |
| `GENORAY_TRACE` | `trace.rs:17` | none |
| `GENORAY_LOG` | `logging.rs:308` **and** `_logging.py:102` | `log_level=` — **env wins** |

Three things are wrong with this.

**It is invisible.** An operator running the conversion inside a Nextflow
pipeline cannot see, from the run's own output, which values took effect. This
is not hypothetical: on PR #174, a downstream user reported that the 4.0.1
`pipeline config` banner printed `reader_workers=3` after `GENORAY_READER_WORKERS`
had set it to 20, and that the mismatch cost them a day of misdiagnosis. #174
partly fixed that specific case by hoisting the `reader_workers` resolution up
into `lib.rs` beside the public argument, so it now reaches the banner. The
general defect stands: `overshard`, `dense_cap`, `merge_threads` and
`sample_interval` are still resolved deep inside `process_chromosome`, and none
of them appears in the banner at all.

**It is a second, competing configuration channel.** Where a Python argument and
an environment variable both exist (`reader_workers`, `log_level`), the
environment silently wins. A caller who passes `reader_workers=8` and gets 20
has no way to find out from the API.

**It is untyped and unvalidated.** `bench_env` is
`std::env::var(key).ok()?.parse::<usize>().ok()` — an unparseable value is
indistinguishable from an unset one, and a typo in the variable name is silently
a no-op.

Only some of these are labelled BENCH-ONLY. `GENORAY_LOG` is documented public
behaviour in `skills/genoray-api/SKILL.md` and is in production use.

## Design

Configuration becomes exclusively Python and CLI. Every `GENORAY_*` read is
deleted, along with `bench_env`, `bench_concurrent_chroms`,
`bench_env_reader_workers`, `bench_overshard` and the whole of `src/trace.rs`.

### A. `genoray.Tuning`

New `python/genoray/_tuning.py`, exported from `genoray`:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class Tuning:
    concurrent_chroms: int | None = None
    reader_workers: int | None = None
    overshard: int | None = None
    dense_cap: int | None = None
    merge_threads: int | None = None
    sample_interval: int | None = None
```

`None` means "the planner derives it" — the current default behaviour, unchanged.
`__post_init__` raises a `ValueError` naming the field, so Rust never receives an
invalid value. Every field rejects values below 1, except `sample_interval`,
which rejects values below 0: `0` is meaningful there and disables the monitor
sampler, matching what `GENORAY_SAMPLE_INTERVAL=0` does today
(`monitor.rs:181`).

`reader_workers` **moves into `Tuning`** from the flat kwarg #174 added. That
kwarg is unreleased (`main` is 4.0.2 + #174; 4.0.2 shipped before the merge), so
this needs no deprecation shim. `threads` and `log_level` stay flat: both are
released public API on five methods.

There is deliberately **no `shard_htslib_threads` field**. #174 removed
`GENORAY_SHARD_HTSLIB` and made per-shard HTSlib threads the constant
`SHARDED_VCF_HTSLIB_THREADS_PER_READER = 0`, because sharded readers decompress
inline and never allocate that pool. Resurrecting the knob would re-add a
degree of freedom that change deliberately removed.

**Crossing the FFI.** `Tuning` is a Python dataclass, not a `#[pyclass]`. `_core`
ships no `.pyi` stubs, so a Python-owned type is what makes the knobs typed,
docstringed, and `dataclasses.replace`-able. Rust receives it as one argument per
entry point through a `#[derive(FromPyObject)]` struct that reads the named
attributes:

```rust
#[derive(FromPyObject, Debug, Clone, Copy, Default)]
pub struct TuningIn {
    concurrent_chroms: Option<usize>,
    reader_workers: Option<usize>,
    overshard: Option<usize>,
    dense_cap: Option<usize>,
    merge_threads: Option<usize>,
    sample_interval: Option<usize>,
}
```

**Applicability.** Not every knob reaches every backend. All four conversion
pipelines funnel through `orchestrator::process_chromosome` (`run_vcf_list`
included, at `orchestrator.rs:1398`), which owns the dense channel, the sampler
and the merge tail; only contig concurrency differs, and `run_slice_view` shares
none of it.

| field | `from_vcf` | `from_pgen` | `from_vcf_list` | `from_svar1` |
|---|---|---|---|---|
| `concurrent_chroms` | yes | yes | **no** — sequential by design | yes |
| `reader_workers` | yes | **no** — `P=1` pinned | no | no |
| `overshard` | yes | no | no | no |
| `dense_cap` | yes | yes | yes | yes |
| `merge_threads` | yes | yes | yes | yes |
| `sample_interval` | yes | yes | yes | yes |

A field set to a non-`None` value on a method that cannot use it raises
`ValueError` at the Python boundary, naming the field and the backend. Silently
ignoring it is the failure mode this whole change exists to remove. The slice/view
method takes no `tuning` at all — only the logging arguments in section C.

### B. Resolution: honour or refuse, and a banner that says so

#174 established the rule for `reader_workers`: an explicit request is honoured or
refused, never silently shrunk, because a caller who asked for 24 and got 3 has no
way to find out. That generalizes to every `Tuning` field, with one deliberate
exception.

**Refusal is reserved for the memory budget.** `plan_sharded`'s contract already
says this: concurrency may come down to fit `max_mem`, and an infeasible explicit
request returns `PlanError::InsufficientMemory`. An explicit `concurrent_chroms`
that does not fit `max_mem` is refused the same way.

**The PGEN concurrency cap stays falsifiable.** `lib.rs:619` clamps the PGEN plan
to `PGEN_MAX_CONCURRENT = 8`, the measured knee, and its comment states that the
override is applied *after* the cap on purpose — clamping the override would make
the constant unfalsifiable, and the escape hatch exists so a maintainer can
re-measure past 8. That requirement survives the env var: an explicit
`concurrent_chroms` on the PGEN path is **honoured past the cap** and logged at
`warning` noting it exceeds the measured knee. The cap still applies to the derive
path. Refusing here would trade one unfalsifiable constant for another.

**Resolution moves to one place.** All six values are resolved in `lib.rs`
alongside the plan, before any thread spawns, and passed down as data. The late
reads inside `process_chromosome` (`orchestrator.rs:456`, `:1130`) and
`monitor.rs:181` are deleted.

**The banner reports every field and its source.** `pipeline config` currently
logs `concurrent_chroms`, `reader_workers` and `processing_threads`. It gains the
remaining resolved fields, each tagged `explicit` or `planner`:

```
pipeline config concurrent_chroms=2(explicit) reader_workers=20(explicit)
  overshard=40(explicit) dense_cap=6(planner) merge_threads=12(planner)
  sample_interval=5(planner) processing_threads=12
```

With no channel able to override a value after it is printed, the line is correct
by construction rather than by maintenance. This is the specific request made on
PR #174 by the operator who lost a day to the 4.0.1 mismatch.

### C. Logging

**Levels follow current Python convention.** `log_level` accepts,
case-insensitively:

```
"off" | "critical" | "error" | "warning" | "info" | "debug"
```

plus `logging` integer constants (`logging.CRITICAL`/`ERROR`/`WARNING`/`INFO`/
`DEBUG` = 50/40/30/20/10). `0` maps to `"off"`: `logging.NOTSET` means "inherit
from the parent logger" in Python, and there is no parent here, so the only
coherent reading of "no level" for a library gate is silence. An integer that
falls between the named levels rounds **up** to the next more-severe one, which
is what Python's own gate does: a logger set to `25` suppresses `INFO` (20) and
admits `WARNING` (30), so `25` must gate like `warning`. This matters because
`logging.getEffectiveLevel()` is a natural thing to pass and can return such a
value. The four values accepted today
(`off`, `warning`, `info`, `debug`) all remain valid, so this is a pure extension
with no break.

Channel ranks become `off=0, error=1, warning=2, info=3, debug=4`, which splits
`event_rank`'s current `ERROR | WARN => 1` collapse in `logging.rs:240`.

Two aliasing decisions, on one rule — **accept every current Python level name,
reject deprecated ones**:

- `"warn"` is **rejected**. `logging.warn()` was removed in Python 3.13, so
  accepting it is an extra spelling, not a convention. The `ValueError` names the
  accepted set, which is the right answer for someone typing it out of `tracing`
  habit (where the level genuinely is `warn`).
- `"critical"` is **accepted as a documented alias for `"error"`**. `tracing` has
  no CRITICAL level, and no genoray call site emits `warn!` or `error!` today —
  only `info!`, `debug!` and `trace!` — so both gate a currently-empty set.
  Accepting `40` while rejecting `50` on the integer path would be the odd rule,
  and the string form should match the integer form.

**`log_filter` replaces what `GENORAY_LOG` could do and `log_level` cannot.** New
flat argument `log_filter: str | None`, taking `tracing` directive syntax
(`"genoray::monitor=trace"`), driving the stderr fmt layer. It is a separate
argument from `log_level` because the two feed different sinks: `log_level` gates
the structured channel that `ProgressRenderer` renders in Python, while
`log_filter` gates raw stderr and is the only route to `trace` — the level the
monitor sampler emits at. Today's coupling, where `GENORAY_LOG` also forces the
channel level (`_logging.py:102`, guarded by `tests/test_logging.py:252`), was a
workaround for one variable serving two sinks; it disappears with the variable.

**Mechanically, the subscriber must become reconfigurable.**
`ensure_global_subscriber` (`logging.rs:301`) is a `Once` that calls
`EnvFilter::try_from_env("GENORAY_LOG")`, so today the filter is fixed at
first-write for the life of the process and a per-call filter could not take
effect. Wrap the `EnvFilter` in `tracing_subscriber::reload::Layer`, install it
unconditionally with an initial `off` filter (free when off), keep the handle in a
static, and have `with_channel_subscriber` set and restore it RAII-style beside
`CURRENT_SINK`/`CURRENT_LEVEL`. The existing documented caveat applies unchanged:
these are process-global slots, so two concurrent writes in one process are
last-writer-wins.

**`src/trace.rs` is deleted outright**, along with `trace_ll!` and its call sites.
Those raw `eprintln!` heartbeats predate the tracing and monitor layers and
duplicate them; `log_filter="genoray=trace"` reaches the same seams through the
subscriber, with targets and timestamps.

### D. CLI

Named flags on the commands each knob applies to, per the section A matrix:
`--concurrent-chroms`, `--reader-workers` (already present), `--overshard`,
`--dense-cap`, `--merge-threads`, `--sample-interval`, and `--log-filter`.
`--log-level` gains the section C names.

A flag that does not apply to a command is **not defined on that command**, so
`genoray write pgen --overshard 4` fails in argument parsing rather than at
runtime.

One shared cyclopts helper builds the `Tuning` from the parsed flags, so the four
write commands do not each redeclare six parameters and their docstrings.

### E. Validation

**Migration.** Twelve files read `GENORAY_*`:

- Tests (5): `test_svar2_schedule_invariance.py` (9 uses),
  `test_logging.py` (8), `test_svar2_pgen_schedule_invariance.py`,
  `test_svar2_from_vcf.py`, `tests/bench/test_probe.py`. All move to keyword
  arguments. Their `monkeypatch` comments explaining that the value *must* be set
  on `os.environ` because it is read inside the reader thread
  (`test_svar2_schedule_invariance.py:80-86`) are deleted along with the reason.
- Bench scripts: `probe.py`, `plans/build_plans.py`, `records.py`,
  `sweep_scale.sbatch` move to CLI flags. `ab_builds.py`'s `BENCH_ENV_VARS` scrub
  list becomes an assertion that no `GENORAY_*` variable exists at all — with the
  channel gone, scrubbing it is dead code but asserting its absence is a live
  integrity check. `legacy_pr140/*` gets a header note recording that it targets
  the pre-removal interface and no longer runs.

**New tests.**

1. `Tuning` field validation: below-minimum values, and each field rejected on
   each backend that cannot use it (the section A matrix, as a table test).
2. Banner source labelling: parse the `pipeline config` line and assert
   `explicit` vs `planner` for each field. #174's harness already parses this
   line.
3. **An environment-regression guard**: assert `src/` contains no `std::env::var`
   or `env::var_os` outside `svar1_reader.rs`'s temp-directory use. This is what
   stops the channel growing back one convenience hook at a time.
4. Byte-identity across a `Tuning` sweep, reusing #174's schedule-invariance
   oracle — the tuning values must change scheduling and nothing else.
5. The level table: every accepted string and integer maps to the expected rank;
   `"warn"` raises; `"critical"` and `"error"` produce the same rank.

**Docs.** `skills/genoray-api/SKILL.md` (mandated by `CLAUDE.md` for any public
name change), `docs/source/svar.md`, and `scripts/bench_svar2/README.md`.

## Public API impact

Breaking, and therefore a **major bump**. Commits carry `feat!:` / `BREAKING
CHANGE:` footers and the release workflow cuts the version — do not edit
`CHANGELOG.md` or the version by hand.

- **Added**: `genoray.Tuning`; `tuning=` on `from_vcf`, `from_pgen`,
  `from_vcf_list`, `from_svar1`; `log_filter=` on those plus the slice/view
  method; new `log_level` values and integer forms; the CLI flags in section D.
- **Removed**: all eight `GENORAY_*` variables. `GENORAY_LOG` is the one with
  documented public behaviour and a production consumer; its replacement is
  `--log-filter` / `log_filter=`.
- **Moved**: `reader_workers=` from a flat kwarg into `Tuning` (unreleased; no
  shim).
- **Unchanged**: `threads`, `max_mem`, `chunk_size`, `progress`, and every
  default. A caller who sets nothing gets today's behaviour.

## Deferred

- **#170** executor parallelism — untouched here.
- **#172** `Msg::Err` / `wake_all` error-path coverage — untouched.
- **#173** `memory_fits` over-charges PGEN — untouched; this change does not
  refit any `RamLaw` coefficient.
- The four unmeasured frontier constants (`W_TARGET`, `MERGE_RESERVE_DIV`,
  `UNITS_TARGET_CHUNKS`, `PENDING_BUDGET_CHUNKS`) stay compile-time constants.
  Promoting them to `Tuning` fields is a separate question that should follow the
  measurement #174 deferred, not precede it.
