# SVAR2 write logging & progress — design

Date: 2026-07-20
Status: approved (design), pending implementation plan
Scope: logging and progress reporting for the SVAR2 write/conversion path

## Problem

The SVAR2 conversion path (VCF/PGEN/VCF-list/SVAR1 → `.svar`) currently reports
progress and edge cases through scattered, unconditional `println!` calls in the
Rust core (`orchestrator.rs`, `writer.rs`) that go straight to stdout. There is
no way to silence them, no level control, no progress bar, and edge-case data
(excluded variants, normalization changes, contig-name resolution) is collapsed
to bare counters with little diagnostic detail. The Python `write` methods accept
a `progress: bool` argument that is documented as a no-op (`_svar2.py`).

We want:

1. Structured logging in the Rust core via the `tracing` crate.
2. Configuration from the Python API and the `genoray` CLI.
3. Pretty progress bars in an interactive terminal **and** Jupyter.
4. In non-interactive contexts (log file, SLURM/CI, redirected stdout), progress
   that stays clearly evident without producing massive log files.
5. Edge-case reporting focused on: excluded variants, variant-normalization
   stats, and contig-name mismatches resolved via `ContigNormalizer`.

## Goals / non-goals

**Goals**
- One coherent logging + progress system for all SVAR2 write entry points.
- Rust `tracing` owns structured logging and edge-case events.
- Progress bars rendered in Python via `rich` (works in terminal and Jupyter).
- Compact heartbeat lines when output is not an interactive TTY.
- Per-event edge-case detail at `debug`; per-contig summaries at `info`.
- Replace the ad-hoc `println!` calls and demote the `monitor.rs` sampler.

**Non-goals**
- No logging changes to the SVAR2 *read/query* path (only the write path).
- No new dependency on `indicatif` or any Rust-side TTY renderer (ruled out: it
  cannot render in Jupyter, which is not a TTY).
- No structured/JSON log formatter in this iteration (heartbeat lines only for
  non-TTY). A JSON formatter is a possible follow-up, noted in Out of scope.
- No change to output byte format of `.svar` files.

## Chosen approach (and the alternatives rejected)

- **Progress rendering**: Rust emits progress *events*; **Python renders with
  `rich`**. Rejected: Rust-native `indicatif` (breaks/degrades in Jupyter, which
  is not a TTY) and a hybrid two-renderer path (two code paths to maintain).
  `rich` is already an indirect dependency via `cyclopts`; it will be promoted to
  a direct dependency since we now rely on it at runtime.
- **Transport**: **a single `crossbeam` channel** carries both progress and log
  events, rather than a separate hot atomic-counter lane. Chunks are already the
  coarse unit of work, so one `Progress` message per chunk is negligible. This
  keeps one code path and one ordering guarantee. (`crossbeam-channel` is already
  a dependency.)
- **Edge-case granularity**: **per-event at `debug`, summaries at `info`**. Level
  filtering happens in the Rust subscriber, so per-event events cost nothing when
  the level does not admit `debug`.

## Architecture & data flow

```
Rust write (inside py.detach(...), many pipeline/rayon threads)
  ├─ tracing::info!/debug!/warn!(...)  ──┐  custom tracing Layer
  │                                      ├──►  serialize into Event
  └─ progress emit at each chunk commit ─┘        │
                                                  ▼
                                    crossbeam::Sender<Event>
                                                  │  (GIL released during write)
   ┌──────────────────────────────────────────────┘
   ▼
Python drain thread (spawned before entering Rust)
   recv() releases the GIL while parked on the channel
   routes each Event:
     ContigStart/Progress/ContigDone → rich Progress bar (interactive TTY/Jupyter)
                                        └ or compact heartbeat line (non-TTY)
     Log                              → rich console.print above the live bar
```

### GIL and threading model

- Every write entry point runs its heavy work inside `py.detach(|| ...)` (GIL
  released) across pipeline/rayon threads. Rust worker threads push `Event`s onto
  the `crossbeam::Sender` with no GIL involvement.
- Python's `write()` spawns **one** drain thread *before* calling into Rust. The
  drain thread calls a pyo3-exposed `Receiver.recv()` (or `recv_timeout`) that
  performs `py.detach()` while blocking, so it never contends with the main thread
  (which is parked inside the Rust conversion with the GIL released).
- The drain thread is the **only** writer to the `rich` `Progress`/`Console`
  object, so there are no `rich` thread-safety concerns.
- Shutdown: when the Rust call returns, Python drops/closes the sender handle,
  which unblocks the drain thread's `recv()` with a disconnect; Python then joins
  it. On an exception from Rust, the same teardown runs in a `finally`.

### Event protocol

A Rust enum, serialized across the channel and decoded in Python:

- `ContigStart { chrom: String, total: Option<u64> }` — `total` is `Some` when the
  variant count is known ahead of time (e.g. PGEN via the `.gvi` index), else
  `None` → the bar runs as an indeterminate count-up with rate.
- `Progress { chrom: String, delta: u64 }` — emitted at each chunk commit.
- `ContigDone { chrom: String, kept: u64, excluded: u64, elapsed_ms: u64 }`.
- `Log { level: Level, chrom: Option<String>, message: String, fields: ... }` —
  produced by the tracing bridge Layer from `tracing` events.

The exact serialization (a `#[pyclass]` event object vs. a small tagged tuple)
is an implementation detail for the plan; the semantics above are fixed.

## Rust side

- **Dependencies**: add `tracing` and `tracing-subscriber` to `Cargo.toml` under
  default features. These do not interact with the `abi3` wheel feature, so lint
  hooks and the Rust test suite are unaffected (consistent with the existing
  build notes in the project CLAUDE.md).
- **Instrumentation**: replace the scattered `println!` in `orchestrator.rs` and
  `writer.rs` (pipeline-config banner, `==> Processing {chrom}`, `Phase 1
  Complete`, `report_ref_excluded`, `Cohort Processing Complete`, thread notices)
  with `tracing` events at appropriate levels (`info` for milestones, `debug` for
  detail). Add spans per contig so events carry `chrom` context.
- **Progress emit**: at each chunk commit in the writer/orchestrator, send
  `Progress { chrom, delta }`. Emit `ContigStart` at the start of each contig
  (with `total` when known) and `ContigDone` at the boundary with kept/excluded/
  elapsed.
- **Edge-case events** (the focus areas):
  - **Excluded variants** (`check_ref=x`, out-of-scope skips): `info`/`warn`
    per-contig summary with counts by reason; `debug` per-event with
    `chrom:pos REF=…≠ref` detail. The exclusion detail already exists as
    `RefDecision::Exclude(detail)` / `ref_excluded` in `chunk_assembler.rs` and
    the `report_ref_excluded` counter — route it through `tracing` instead of a
    bare counter.
  - **Normalization stats**: per-contig counts of left-aligned / atomized /
    trimmed records at `info`; per-event normalization changes at `debug`.
  - **Contig-name resolution**: when `ContigNormalizer` maps a queried/records
    contig name to a differently-spelled file contig (`'1' → 'chr1'`), emit a
    one-time `info` per resolved contig and a `debug` per occurrence. This
    requires surfacing the resolution decision from `ContigNormalizer` at the
    call sites that currently resolve silently.
- **Two subscriber sinks**, selected at write-init:
  - *Bridge Layer* — active when Python has registered a channel for this write;
    converts `tracing` events into `Log` events and pushes them to the channel.
    Its max level is set from the Python `log_level` argument.
  - *Fallback fmt Layer* — for pure-Rust runs with no Python consumer (the
    `bench_from_vcf_list` binary, `cargo` runs). Compact single-line formatter to
    stderr, gated by a `GENORAY_LOG` env var (`RUST_LOG`-style directives via
    `EnvFilter`). This is also the power-user escape hatch from Python.
  - Subscriber registration must be **per-write and non-global-conflicting**:
    use a scoped/local subscriber (e.g. `tracing::subscriber::with_default` or a
    reloadable layer) so concurrent or repeated writes in one process don't fight
    over the global default. The plan must verify behavior when `write` is called
    more than once in a process (common in tests and notebooks).
- **Boy-scout cleanups**:
  - Fold the `monitor.rs` sampler (bounded-channel fill levels + per-thread CPU%)
    into `trace!`-level events under a dedicated target (e.g.
    `target: "genoray::monitor"`), so it is opt-in via verbosity rather than an
    always-on stderr printer.
  - Remove the now-dead `progress` no-op comment once the argument is wired up.

## Python side

- **Dependency**: promote `rich` to a direct dependency in `pyproject.toml`.
- **Renderer selection** (in `write`):
  - Interactive (a real TTY, or Jupyter detected via `rich`'s console): a `rich`
    `Progress` with one task per contig (or a single overall task that relabels
    per contig), showing description, bar, count, and rate. Indeterminate when
    `total` is `None`.
  - Non-interactive (`Console(...).is_terminal` false, e.g. redirected/SLURM/CI):
    **compact heartbeat lines** instead of a live bar, e.g.
    ```
    [svar2] chr1  52% (2,010/3,900) ...
    [svar2] chr1 done: 3,880 kept, 20 excluded (14.2s)
    ```
    The throttled `%` progress line (every N% or T seconds, whichever is coarser,
    so volume stays bounded regardless of cohort size) is emitted **only when
    `progress=True`**. Independently of `progress`, `ContigDone` and other
    `info`-level summaries print one line each per `log_level`. So the matrix is:
    `progress=True` non-TTY → throttled `%` lines + summaries; `progress=False`
    → summaries only (no `%` lines); `progress=True` TTY/Jupyter → live bar.
  - Log events (`ContigDone` summaries, edge-case `info`/`warn`, and `debug`
    per-event when enabled) are printed through the same `rich` `Console` that
    owns the live `Progress`, so lines render *above* the bar without corrupting
    it (`rich` redraws the bar beneath printed output).
- **Wiring**: thread `progress` and `log_level` through all write entry points —
  `from_vcf`, `from_pgen`, `from_vcf_list`, `from_svar1`, and `write_view` — into
  the corresponding `_core` pyfunction calls
  (`run_conversion_pipeline`, `run_pgen_conversion_pipeline`,
  `run_vcf_list_conversion_pipeline`, `run_svar1_conversion_pipeline`,
  `run_slice_view`). The pyfunctions gain a channel/level parameter and register
  the bridge subscriber for the duration of the call.

## Configuration surface

- **Python API** (all write entry points):
  - `progress: bool = False` — now functional (controls the bar). Default stays
    `False` for backward compatibility; when `True` and non-interactive, falls
    back to heartbeat lines.
  - `log_level: Literal["off", "warning", "info", "debug"] = "info"` — controls
    `tracing` verbosity. `info` shows per-contig summaries and milestones;
    `debug` adds per-event edge-case detail; `warning` shows only warnings;
    `off` silences all log events (progress bar unaffected).
- **CLI** (`genoray write vcf|pgen|svar1`):
  - `--progress / --no-progress` (default off, mirroring the API).
  - `--log-level {off,warning,info,debug}` (default `info`).
- **Env escape hatch**: `GENORAY_LOG` (`RUST_LOG`-style directives, e.g.
  `GENORAY_LOG=genoray::monitor=trace,info`). When set, it takes precedence for
  the fallback fmt sink and can raise verbosity of specific targets for power
  users / bug reports.

Rationale: `progress` and `log_level` are orthogonal and familiar; the env var
covers targeted deep debugging without widening the API.

## Error handling

- The drain thread must never crash the write: exceptions during rendering are
  caught, and rendering degrades to plain prints; the underlying conversion is
  authoritative.
- If the Rust call raises, Python tears down the channel/drain thread in a
  `finally` and re-raises the original error.
- If `rich` is somehow unavailable at runtime (shouldn't happen once it is a
  direct dep), `progress=True` degrades to heartbeat lines rather than erroring.
- Channel backpressure: the channel is bounded with a modest capacity; if the
  drain thread stalls, producers should not block the conversion — dropping or
  coalescing `Progress` events under pressure is acceptable (progress is
  advisory), but `Log` and `ContigStart/Done` events must not be dropped. The
  plan will specify the exact bound and drop policy.

## Testing

- **Rust**: unit tests that the bridge Layer emits the expected `Event`s for a
  small synthetic conversion (contig start/progress/done, an excluded record, a
  normalized record, a contig-name resolution). Follow the repo convention:
  `cargo test --no-default-features` (the pyo3 test binary otherwise fails to
  link), and use `--test <file>` rather than a bare name filter.
- **Python**: tests that `write(..., progress=False, log_level="info")` produces
  the expected summary lines (capture the `rich` `Console` via `record=True` /
  `capsys`), that `log_level="off"` is silent, that `log_level="debug"` includes
  per-event lines, and that non-TTY output uses heartbeat lines (force
  `Console(force_terminal=False)`). Assert the drain thread is always joined
  (no leak) including on the error path.
- **No-regression**: the write output bytes are unchanged; run the existing
  conversion round-trip tests to confirm byte-identical `.svar` output.
- Note: `pixi run test` does not rebuild the Rust extension — run
  `maturin develop --release` before Python-level verification of Rust changes.

## Backward compatibility

- `progress: bool = False` keeps its default and signature; behavior changes only
  from "no-op" to "functional", which is non-breaking.
- `log_level` is a new keyword-only argument with a default, so existing calls are
  unaffected.
- The old unconditional `println!` output is removed; any downstream tooling that
  scraped that stdout is unsupported and not a compatibility surface.

## Public API / SKILL.md

`progress` (now functional) and the new `log_level` keyword on every write entry
point, plus the two new CLI flags, are **public**. Per the project CLAUDE.md, the
same PR that implements this MUST update `skills/genoray-api/SKILL.md` to document
`progress`, `log_level`, and the CLI `--progress` / `--log-level` flags.

## Out of scope (possible follow-ups)

- A structured JSON-lines log formatter (selectable) on top of the heartbeat
  transport, for machine consumption in pipelines/dashboards.
- Extending the same logging system to the SVAR2 read/query path.
- Progress/logging for `PGEN`/`VCF` dense readers beyond the existing `tqdm`
  usage in `_vcf.py`.
