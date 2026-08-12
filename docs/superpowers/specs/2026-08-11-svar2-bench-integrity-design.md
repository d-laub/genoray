# SVAR2 bench-harness measurement integrity

Design spec closing issues **#159**, **#162**, and **#151**.

Date: 2026-08-11. Base: `803aad1` (merge of PR #163).

## Why these three together

Each one lets the harness report a number it did not measure, without raising:

| Issue | The gap | Status |
|---|---|---|
| #159 | A row can describe *different code* than the run that claims it. | Fired. Cost a retracted, pushed, announced `RamLaw::PGEN` re-fit. |
| #162 | A row can silently omit the regressor that decides whether it is fittable at all. | Latent. Held off only by a hand-applied workaround nothing enforces. |
| #151 | A sweep family can abort hours in on a corpus nobody generates. | Latent. Held off only by the corpora happening to exist on one machine. |

The unifying fix: **make the gap between "reported" and "measured" representable
in the data, and fail loudly when it opens.** Every change below either records
a fact that was previously unrecoverable, or converts a silent drop into an
abort.

None of these changes alter a measurement. No law is re-fitted and no sweep is
re-run as part of this work.

---

## 1. #162 — carry structured fields through the log channel

### Root cause

The defect is in the library, not the bench script. `ChannelLayer::on_event`
(`src/logging.rs:234-255`) constructs a `FieldGrab` visitor that retains only
two fields:

```rust
struct FieldGrab {
    message: String,
    chrom: Option<String>,
}
```

Every other field on a `tracing` event is dropped before the event becomes an
`Event::Log` and crosses the channel to Python. This affects *every*
`tracing::info!(k = v, …, "msg")` in the codebase, not just the bench path: a
user running `genoray write vcf --log-level info` sees a line reading
`pipeline config` with none of the config in it.

For the bench harness specifically, `probe.py`'s `RE_PIPELINE_CONFIG` parses
`key=value` pairs off that line to recover the concurrency the planner actually
chose. There is nothing left to parse, so
`ProbeRecord.concurrent_chroms_used` is `None` on every row produced through
the Python CLI path — confirmed on all 48 rows of the committed 2026-08-11 VCF
crossed sweep.

The stderr fmt layer *does* render fields, but `probe.py:190` sets
`GENORAY_LOG=genoray::monitor=trace`, enabling only the `genoray::monitor`
target; the `pipeline config` event's target is plain `genoray` (emitted from
`src/lib.rs:326` / `:601` / `:1338`). So neither route delivers the field.

### Fix

Widen `FieldGrab` to collect all fields other than `message` and `chrom`, in
visit order, and render them onto the message as ` k=v` pairs — the same
compact shape `tracing_subscriber`'s own fmt layer emits, and the shape
`probe.py::_field`'s `\bkey=([^\s]+)` regex already parses.

`Event::Log`'s shape, the pyo3 boundary, and `python/genoray/_logging.py` are
all untouched.

Sketch:

```rust
#[derive(Default)]
struct FieldGrab {
    message: String,
    chrom: Option<String>,
    fields: Vec<(String, String)>,   // everything else, in visit order
}

impl FieldGrab {
    /// `"<message> k1=v1 k2=v2"`, or just `"<message>"` when no other
    /// fields were recorded.
    fn render(&self) -> String { … }
}
```

**The two existing methods are the only ones that need changing.** `Visit`'s
typed recorders — `record_i64`, `record_u64`, `record_bool` — all default to
forwarding to `record_debug` (tracing-core 0.1.36, `src/field.rs:291-318`), so
`concurrent_chroms`, `reader_workers` and `htslib_threads` already arrive at
`FieldGrab::record_debug` today. They are discarded **by name, not by type**:
the `if/else if` chain simply has no branch for them. Adding typed
`record_u64`/`record_i64` impls is unnecessary; giving both existing methods a
final `else` that pushes onto `fields` is sufficient and complete.

Keep the `record_str` override rather than collapsing to `record_debug` alone.
It exists because `Debug` on a `&str` renders `"chr1"` *with* quotes, and the
same reasoning applies to the new fields: string-valued fields should render
`k=v`, not `k="v"`, to match the fmt layer and to keep `_field`'s
`([^\s]+)` capture clean.

### Two rejected alternatives, and why

**Have `probe.py` widen `GENORAY_LOG` to enable the `genoray` target**
(the issue's second suggestion). `GENORAY_LOG` is doubly overloaded: Rust reads
it as a tracing `EnvFilter` (`logging.rs:283`) while Python reads it as a
four-value log-level override (`_logging.py:102`, documented at
`skills/genoray-api/SKILL.md:423`). `probe.py`'s current
`genoray::monitor=trace` works *only* because it is not one of those four
values and so falls through the Python override untouched. Building the fix on
that coincidence is worse than fixing the visitor, and it leaves every
non-bench consumer still losing fields.

**Add a `fields: Vec<(String, String)>` arm to `Event::Log`.** The Python side
unpacks a fixed 5-tuple and prints `message` (`_logging.py:71-77`); a
structured list would force a rendering decision in Python duplicating what
tracing already does in Rust. Rendering once, in Rust, keeps one format and one
parser.

### Tests

1. **Rust unit test over the visitor.**
   `info!(concurrent_chroms = 8, reader_workers = 4, "pipeline config")`
   renders `pipeline config concurrent_chroms=8 reader_workers=4`. Field order
   is `tracing`'s declaration order, stable per call site, so the assertion can
   be exact. Include a no-extra-fields case asserting the message is unchanged
   (no trailing space).

2. **Python subprocess regression** — the test #162 explicitly asks for, and
   the one today's suite lacks. Drive the real `genoray._cli write vcf`
   subprocess over a small fixture through `probe.run_point`, with
   `SweepPoint.concurrent_chroms` left **unpinned**, and assert
   `rec.concurrent_chroms_used is not None`.

   The existing `tests/bench/test_probe.py:106-126` feeds synthetic fmt-layer
   text straight to `parse_trace`; it passes today against a path that has
   never worked end to end. Marking it clearly as a format test, and adding the
   channel-path test beside it, is the point.

### Remove the workaround scaffolding

This is the part that actually closes the issue — the code currently *documents
itself as broken* in four places, and one guard treats the broken state as
normal.

- `scripts/bench_svar2/sweep_scale.sbatch:255`: the
  `WARNING: N row(s) never reported a realised cc` becomes an **ABORT**. After
  the fix, an unobserved cc is a defect, not the expected condition.
- The same guard block currently runs only under `if [ "$NAME" = "vcf_ram" ]`.
  Hoist it to run for **every** family. Nothing about it is `vcf_ram`-specific,
  and the families it skips are the ones whose data has already been shipped
  into laws.
- `scripts/bench_svar2/model.py::_ram_rows` drops rows with unobserved cc
  **silently**. That silence is half the harm named in #162: a law can be
  certified on a subset that excludes exactly the planner-chosen production
  configurations. Make it report the number and ids dropped.
- Strip the now-false explanatory comments at `records.py:127-152`,
  `probe.py:36-60`, and `probe.py:325-335`, replacing them with a short note
  that the field is parsed from the child's `pipeline config` line.

---

## 2. #159 — code identity in the resume key

### Root cause

`sweep.py:30-32` resumes on `point_id` alone:

```python
done = {r.point_id for r in read_ndjson(results_path, ProbeRecord)}
return [p for p in plan if p.point_id not in done]
```

`SweepPoint.point_id` (`records.py:84-87`) is a SHA256 over the *configuration*
only. Two runs of one configuration against different code are indistinguishable
to it — precisely the case a benchmark exists to distinguish.

On PR #154, job 13351684 reported `18 points recorded` and completed in 18
minutes. **Six points were measured.** The other twelve were served from the
2026-08-05 run's `pgen.ndjson`, still sitting in `/local/dlaub/pgen-sweep/out/`
on the pinned node, and described the old unbounded reader the branch existed to
replace. A `RamLaw::PGEN` re-fit was computed from the mixture, reviewed,
committed, pushed and announced before review caught it (reverted in `51a1a9c`).

Note that the `--nodelist` pinning which preserves an expensive corpus cache is
the same pinning that preserves stale results. The benefit and the hazard arrive
together, so "just don't pin" is not available as a mitigation.

### Fix

**(a) Provenance per row.** Two fields on `ProbeRecord`, defaulted `""` exactly
as `node` is, so records written before they existed still load:

- `code_id: str = ""` — sha256 of the built `_core` extension `.so`.
  **The artifact, not the git commit.** This repository's documented trap is
  that `pixi run test` does not rebuild the extension, so a commit can advance
  while the measured binary does not; hashing the artifact catches that
  direction as well as the ordinary one, and distinguishes A/B builds too.
  `ab_builds.py:80-92` already carries the hashing idiom to follow. Computed
  once per `run_sweep`, not per point.
- `run_id: str = ""` — `$SLURM_JOB_ID` when set, else a `uuid4`, stamped once
  per `run_sweep` invocation. Makes a mixed file self-partitioning after the
  fact, which is what would have let the 2026-08-07 audit answer "which rows
  came from which job" without `cmp`-ing against an older file.

**(b) Resume on `(point_id, code_id)`:**

```python
done = {(r.point_id, r.code_id) for r in read_ndjson(results_path, ProbeRecord)}
return [p for p in plan if (p.point_id, current_code_id) not in done]
```

Rows predating the field carry `code_id=""`, which never equals a real hash, so
they are re-measured. That is the correct default: **fail toward measuring.**

**(c) Honest summary.** `run_sweep` tracks measured and reused counts; `main()`
prints them separately in place of `f"{len(recs)} points recorded"`. The
incident's run printed `18 points recorded` for six measurements; the honest
line would have read `6 measured, 12 reused (18 rows in file)`.

### What is deliberately not changed

`SweepPoint` is untouched, so `point_id` is stable and every committed sweep's
data still loads and still fits. Adding fields to `ProbeRecord` changes the
written schema but not the read schema — `from_json` drops unknown keys and
defaults missing ones.

Rotating the output file by default (the issue's third suggestion) is **not**
adopted: it discards the resume behaviour that a preemptible overnight sweep
depends on, to fix a problem the resume *key* can fix directly.

### Tests

In `tests/bench/test_sweep.py`:

- a row whose `code_id` differs from the current build is re-queued;
- a row whose `code_id` matches is skipped;
- a row with `code_id=""` (pre-existing data) is re-queued;
- the summary reports measured and reused counts separately.

---

## 3. #151 — generate `holdout_f0`, and fail fast on any missing corpus

### Root cause

`plans/build_plans.py:522-528` builds a plan point against
`corpus_dir / "holdout_f0.manifest.json"`, but `sweep_scale.sbatch` has a
generation block for `holdout.vcf.gz` only. On a machine where
`corpora/holdout_f0.*` does not already exist, the `holdout` family raises
`FileNotFoundError`; under `set -euo pipefail` that aborts the whole job,
potentially many hours in.

This is the same mechanism found and fixed for `vlinear2` in #141;
`holdout_f0` was out of that PR's scope.

### Fix

**(a)** Add a generation block to `sweep_scale.sbatch` mirroring the `holdout`
one, reading shape back from `build_plans.HOLDOUT_F0` rather than repeating the
numbers — the single-source-of-truth pattern the neighbouring `holdout`,
`vlinear` and `vlinear2` blocks already use. `HOLDOUT_F0["format_fields"]` is
the empty tuple, so the block must pass no `--format-fields` (or an empty
string the corpus generator accepts) rather than reusing `holdout`'s
comma-join unchanged.

**(b)** Add a **pre-flight corpus check** to `run_sweep`. Manifests are loaded
lazily inside the point loop (`sweep.py:80-85`), which is why a missing corpus
surfaces hours in rather than at second zero. Before running any point, verify
every distinct `point.corpus` path exists and raise once, naming **all** missing
manifests, not just the first.

(b) is what generalizes: it turns the next plan point added without a corpus
block into a 0-second complete error message instead of a repeat of this issue.

### Tests

In `tests/bench/test_sweep.py`: a plan naming a nonexistent manifest fails
before the runner is called even once (assert against a runner that raises if
invoked), and the error names every missing path rather than one.

---

## 4. Scope

### In

Issues #159, #162, #151, complete, with the regression tests each names.

### Out, deliberately

- **Refitting any law.** These changes alter no measurement. The shipped
  `RamLaw::VCF` and `RamLaw::PGEN` envelopes stand unchanged.
- **Re-running any sweep.** Committed sweep data keeps its meaning and gains
  `code_id=""` — "provenance unknown" — which is honest and is exactly what
  those rows are.
- **#157, #156, #152, #153** — the FORMAT-path RAM theme, a separate spec.
- **#125** (`from_vcf_shards`) — a feature project with its own cycle.
- **#139** (`get_record_info` drops INFO) — VCF path, unrelated.

## 5. Verification

| Half | Command |
|---|---|
| Rust | `cargo test --no-default-features --features conversion` |
| Python | `pixi run pytest tests/bench/` |

`--no-default-features` alone compiles neither the executor nor the
orchestrator and runs a reduced suite; `--features conversion` is required.
Set `CARGO_TARGET_DIR` off NFS (`$CLAUDE_JOB_DIR/tmp/...`) or the linker
bus-errors on this cluster.

Run `pixi run maturin develop --release` **before** the Python suite: `pixi run
test` does not rebuild the extension, and the new subprocess probe test would
otherwise measure a stale `.so` — the same trap `code_id` exists to detect.

No cluster allocation is needed. All three changes are testable locally.

## 6. Public surface

No importable name changes, so most of `skills/genoray-api/SKILL.md` is
unaffected.

One user-visible change: `--log-level info` (and `log_level="info"`) output
gains ` key=value` suffixes on structured lines. Add a sentence to the
`log_level` block at `skills/genoray-api/SKILL.md:415-427` recording that
structured log lines render their fields.
