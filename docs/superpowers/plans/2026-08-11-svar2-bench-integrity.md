# SVAR2 Bench-Harness Measurement Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close #162, #159 and #151 so the SVAR2 bench harness can no longer report a number it did not measure without raising.

**Architecture:** Three independent defects, one theme. #162 is a Rust fix in the `tracing`→Python log bridge (structured fields are stripped, so the planner's realised concurrency never reaches a sweep row) plus removal of the workaround scaffolding that treats the broken state as normal. #159 adds artifact-level provenance (`code_id`, `run_id`) to `ProbeRecord` and resumes on `(point_id, code_id)` so a code change re-measures instead of serving stale rows. #151 adds the missing `holdout_f0` corpus block plus a `run_sweep` pre-flight so any missing corpus fails at second zero instead of hours in.

**Tech Stack:** Rust (`tracing`, `tracing-subscriber`, `crossbeam-channel`, pyo3 0.29), Python 3 (pytest, dataclasses), bash (Slurm sbatch), pixi.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-11-svar2-bench-integrity-design.md`. Read it before starting.
- **Worktree:** work in `/carter/users/dlaub/projects/genoray/.claude/worktrees/svar2-bench-integrity`, branch `worktree-svar2-bench-integrity`. Do NOT `cd` to the main checkout.
- **`export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target` before ANY cargo or git command.** The repo lives on NFS; cargo's default `target/` bus-errors the linker there, and pre-commit hooks run cargo.
- **Rust tests are `cargo test --no-default-features --features conversion`.** Dropping `extension-module` is required or the pyo3 test binary will not link; keeping `conversion` is required or the whole conversion path is silently skipped (bare `--no-default-features` runs 189 tests instead of 341).
- **Run `pixi run maturin develop --release` before any Python test that shells out to `genoray._cli`.** `pixi run test` does NOT rebuild the Rust extension — a stale `.so` makes Task 2's test fail (or, worse, a future test pass) against code that is not the code under review.
- **No measurement changes.** No law is re-fitted, no sweep re-run, no `SweepPoint` field added or reordered (that would change every `point_id`).
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `perf:`). Commit messages end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- **Never run long cargo/maturin builds in the background and return early.** Run them in the foreground and wait.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/logging.rs` | `FieldGrab` visitor + `ChannelLayer::on_event`: render structured fields onto the message | 1 |
| `tests/bench/test_probe.py` | Probe regression: the realised cc survives the real subprocess path | 2 |
| `scripts/bench_svar2/records.py` | `ProbeRecord`: drop the false "always None" comment; add `code_id`, `run_id` | 2, 4 |
| `scripts/bench_svar2/probe.py` | Drop the false "always None" comments | 2 |
| `scripts/bench_svar2/model.py` | `_ram_rows`: report dropped rows instead of dropping silently | 3 |
| `scripts/bench_svar2/sweep_scale.sbatch` | Guard runs for every family and aborts on unobserved cc; `holdout_f0` corpus block | 3, 5 |
| `scripts/bench_svar2/sweep.py` | `build_code_id`, `(point_id, code_id)` resume, honest summary, corpus pre-flight | 4, 5 |
| `tests/bench/test_sweep.py` | Resume-key and pre-flight tests | 4, 5 |
| `tests/bench/test_model.py` | `_ram_rows` drop-reporting test | 3 |
| `skills/genoray-api/SKILL.md` | Record that structured log lines render their fields | 1 |

## Parallelism

Dispatch with **superpowers:dispatching-parallel-agents** driving **superpowers:subagent-driven-development**. Use Sonnet or weaker for implementers; reserve Opus for review and for fixing a critical implementer failure.

| Wave | Tasks | Why they can share a wave |
|---|---|---|
| A | **1**, **4** | Task 1 is Rust-only (`src/logging.rs`); Task 4 is Python-only (`records.py`, `sweep.py`, `test_sweep.py`). No shared file. |
| B | **2**, **3** | Both need Task 1 merged. Task 2 touches `test_probe.py` + comments in `records.py`/`probe.py`; Task 3 touches `model.py` + `test_model.py` + the sbatch *guard* block. No shared file. |
| C | **5** | Touches `sweep.py`/`test_sweep.py` (after Task 4) and the sbatch *corpora* block (after Task 3's sbatch edit). Sequenced to avoid a same-file conflict. |

Task 2 also edits the `concurrent_chroms_used` comment in `records.py` while Task 4 adds fields to the same dataclass. **Task 4 lands first (Wave A); Task 2's implementer must rebase onto it.** The edits are in different regions (comment block vs. new fields) but the file is shared.

---

### Task 1: Carry structured tracing fields through the log channel (#162, Rust)

**Files:**
- Modify: `src/logging.rs:157-179` (`FieldGrab` + its `Visit` impl), `src/logging.rs:234-255` (`ChannelLayer::on_event`)
- Modify: `skills/genoray-api/SKILL.md:415-427` (the `log_level` bullet)
- Test: `src/logging.rs` `mod tests` (inline, ~line 340)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Event::Log.message` now renders as `"<message> k1=v1 k2=v2"` for any event carrying fields beyond `message`/`chrom`. Task 2 depends on this: `probe.py::_field`'s `\bkey=([^\s]+)` regex must find `concurrent_chroms=<int>` on the `pipeline config` line.

**Background the implementer needs:** `ChannelLayer::on_event` builds a `FieldGrab` and sends only `g.message` across the channel to Python. Integer fields already *reach* the visitor — `Visit`'s `record_i64`/`record_u64`/`record_bool` all default to forwarding to `record_debug` (tracing-core 0.1.36 `src/field.rs:291-318`) — they are discarded **by name**, because the `if/else if` chain has no branch for them. So **do not add typed `record_u64`/`record_i64` impls**; add a final `else` to the two existing methods.

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/logging.rs`, next to `channel_layer_routes_events_at_level`:

```rust
    /// #162 regression: `FieldGrab` used to forward only `message` and
    /// `chrom`, so `tracing::info!(concurrent_chroms = 8, ..., "pipeline
    /// config")` reached Python as the bare string "pipeline config" --
    /// every structured field silently dropped. That made
    /// `ProbeRecord.concurrent_chroms_used` `None` on all 48 rows of the
    /// committed 2026-08-11 VCF crossed sweep, and it also meant a user
    /// running `--log-level info` saw a config line with no config in it.
    #[test]
    fn channel_layer_carries_structured_fields_onto_the_message() {
        use crossbeam_channel::unbounded;
        let _guard = TEST_LOCK.lock().unwrap();
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 1);
        // Same reason as `channel_layer_routes_events_at_level`: CURRENT_SINK
        // is process-global, so filter to this module's own target or a
        // concurrently running foreign test leaks events into `rx`.
        let own_target = module_path!();
        with_channel_subscriber(sink, "info", || {
            tracing::info!(
                concurrent_chroms = 8usize,
                reader_workers = 4usize,
                "pipeline config"
            );
            tracing::info!(chrom = "chr1", "excluded 12 records");
        });
        let msgs: Vec<String> = rx
            .try_iter()
            .filter_map(|e| match e {
                Event::Log {
                    message, target, ..
                } if target == own_target => Some(message),
                _ => None,
            })
            .collect();
        assert_eq!(msgs.len(), 2);
        // Fields render in tracing's declaration order, which is stable per
        // call site, so this can be asserted exactly.
        assert_eq!(
            msgs[0],
            "pipeline config concurrent_chroms=8 reader_workers=4"
        );
        // An event carrying nothing beyond message/chrom is unchanged: no
        // trailing space, and `chrom` is NOT duplicated into the message
        // (it already has its own `Event::Log` field).
        assert_eq!(msgs[1], "excluded 12 records");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
cargo test --no-default-features --features conversion \
  channel_layer_carries_structured_fields -- --nocapture
```

Expected: FAIL. `msgs[0]` is `"pipeline config"`, missing ` concurrent_chroms=8 reader_workers=4`.

- [ ] **Step 3: Widen `FieldGrab`**

Replace `src/logging.rs:157-179` entirely with:

```rust
#[derive(Default)]
struct FieldGrab {
    message: String,
    chrom: Option<String>,
    // Every field that is neither `message` nor `chrom`, in visit order.
    fields: Vec<(String, String)>,
}

impl FieldGrab {
    /// `"<message> k1=v1 k2=v2"` -- the same compact `key=value` shape
    /// `tracing_subscriber`'s own fmt layer renders, and the shape
    /// `scripts/bench_svar2/probe.py::_field` parses with `\bkey=([^\s]+)`.
    ///
    /// Returns the message unchanged (no trailing space) when the event
    /// carried nothing beyond `message`/`chrom`.
    fn render(&self) -> String {
        if self.fields.is_empty() {
            return self.message.clone();
        }
        let mut s = self.message.clone();
        for (k, v) in &self.fields {
            s.push(' ');
            s.push_str(k);
            s.push('=');
            s.push_str(v);
        }
        s
    }
}

impl Visit for FieldGrab {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "chrom" => self.chrom = Some(value.to_string()),
            "message" => self.message = value.to_string(),
            // Recorded UNQUOTED. This override exists precisely because
            // `Debug` on a `&str` renders `"chr1"` with quotes, which would
            // both diverge from the fmt layer's rendering and break
            // `probe.py::_field`'s `([^\s]+)` capture.
            name => self.fields.push((name.to_string(), value.to_string())),
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let v = format!("{value:?}");
        match field.name() {
            "chrom" => self.chrom = Some(v.trim_matches('"').to_string()),
            "message" => self.message = v,
            // Integers, bools and every other typed field arrive here:
            // `Visit::record_i64`/`record_u64`/`record_bool` all default to
            // forwarding to `record_debug`, so no typed impls are needed.
            name => self.fields.push((name.to_string(), v)),
        }
    }
}
```

- [ ] **Step 4: Send the rendered message**

In `ChannelLayer::on_event` (`src/logging.rs:234-255`), change the `sink.send_log` call's last argument from `g.message` to `g.render()`:

```rust
            sink.send_log(
                to_log_level(lvl),
                g.chrom.as_deref(),
                event.metadata().target(),
                g.render(),
            );
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
cargo test --no-default-features --features conversion \
  channel_layer_carries_structured_fields -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Run the full Rust suite**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
cargo test --no-default-features --features conversion 2>&1 | tail -30
```

Expected: 341+ tests pass. Any test asserting on an exact log message may now see appended fields — if one fails, that is a real behaviour change to reconcile, not a flake. Report it rather than loosening the assertion.

- [ ] **Step 7: Document the user-visible change**

In `skills/genoray-api/SKILL.md`, in the `log_level` bullet (~line 415-427), append one sentence after the sentence ending `...and contig-name resolution against the reference when it differs from the source's own spelling)`:

```markdown
  Structured log lines render their fields inline as ` key=value` pairs
  after the message (e.g. `pipeline config concurrent_chroms=8
  reader_workers=4`), matching what `GENORAY_LOG`'s stderr layer emits.
```

- [ ] **Step 8: Commit**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
git add src/logging.rs skills/genoray-api/SKILL.md
git commit -m "fix(svar2): carry structured tracing fields through the log channel

FieldGrab forwarded only \`message\` and \`chrom\`, so every structured field
on a tracing event was dropped before the event crossed the channel to
Python. \`--log-level info\` printed \`pipeline config\` with none of the config
in it, and \`ProbeRecord.concurrent_chroms_used\` was None on all 48 rows of
the committed 2026-08-11 VCF crossed sweep.

The typed recorders (record_i64/record_u64/record_bool) already default to
forwarding to record_debug, so the fields arrived and were discarded by
NAME, not by type; the fix is a final else branch on the two existing
methods plus rendering them onto the message.

Closes part of #162.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Pin the realised cc through the real subprocess path (#162, Python)

**Depends on:** Task 1 (merged) and Task 4 (merged — shares `records.py`).

**Files:**
- Modify: `tests/bench/test_probe.py:100-130`
- Modify: `scripts/bench_svar2/records.py:127-152` (comment only)
- Modify: `scripts/bench_svar2/probe.py:36-60` and `:325-335` (comments only)

**Interfaces:**
- Consumes: Task 1's rendered `key=value` message; `probe.run_point(point, manifest, outdir, warm=True) -> ProbeRecord`; `scale_corpus.generate(out, samples, variants, contigs, format_fields, seed, procs=8, bgzip_threads=4) -> CorpusManifest`.
- Produces: nothing later tasks consume.

**Background:** `tests/bench/test_probe.py:106-126` feeds synthetic fmt-layer text straight to `parse_trace`. It passed for months against a channel path that never worked end to end. That test is a *format* test and should stay; what is missing is a test of the actual path `run_point` uses.

- [ ] **Step 1: Build the extension**

```bash
pixi run maturin develop --release
```

Required: `pixi run test` does not rebuild the `.so`, and this task's test shells out to `genoray._cli`. Run in the foreground; it takes several minutes.

- [ ] **Step 2: Write the failing test**

Add to `tests/bench/test_probe.py`. Check the file's existing imports and add only what is missing:

```python
from pathlib import Path

from scripts.bench_svar2 import scale_corpus
from scripts.bench_svar2.probe import run_point
from scripts.bench_svar2.records import SweepPoint


def test_run_point_records_the_realised_concurrent_chroms(tmp_path: Path):
    """#162 regression, exercising the REAL Python channel path.

    The `parse_trace` tests above feed synthetic fmt-layer TEXT straight to
    the parser. They passed throughout the period in which the channel path
    carried no structured fields at all, so they could not have caught this.
    This test drives the actual `genoray._cli` subprocess `run_point` runs,
    with `concurrent_chroms` deliberately UNPINNED so the planner chooses and
    this field is the only record of its choice.

    Requires a freshly built extension: `pixi run maturin develop --release`.
    `pixi run test` does not rebuild it.
    """
    manifest = scale_corpus.generate(
        tmp_path / "tiny.vcf.gz",
        samples=8,
        variants=200,
        contigs=["chr22"],
        format_fields=(),
        seed=42,
        procs=1,
        bgzip_threads=1,
    )
    point = SweepPoint(
        corpus=str(tmp_path / "tiny.manifest.json"),
        reader_workers=1,
        concurrent_chroms=None,  # unpinned: the planner decides
        shard_htslib=0,
        overshard=4,
        chunk_size=64,
        threads=2,
        reps=1,
    )

    rec = run_point(point, manifest, tmp_path / "out", warm=False)

    assert rec.ok, rec.error
    assert rec.concurrent_chroms_used is not None, (
        "the planner's realised concurrent_chroms did not survive the "
        "channel path -- see issue #162"
    )
    assert rec.concurrent_chroms_used >= 1
```

- [ ] **Step 3: Run the test to verify it fails on pre-Task-1 code**

```bash
git stash push -u -m "task2-wip-$(date +%s)"   # capture the SHA from `git stash list --format='%H %gs'`
git checkout <commit-before-task-1> -- src/logging.rs
pixi run maturin develop --release
pixi run pytest tests/bench/test_probe.py::test_run_point_records_the_realised_concurrent_chroms -v
```

Expected: FAIL on `assert rec.concurrent_chroms_used is not None`.

Then restore: `git checkout HEAD -- src/logging.rs && pixi run maturin develop --release`, and `git stash apply <sha>` followed by dropping that entry (re-find its `stash@{n}` by the unique tag first). **Never bare `git stash pop`** — the stash stack is shared with other worktrees and sessions.

If this verification is too costly, it is acceptable to skip it and note in the commit that the failure mode was verified analytically against Task 1's own Rust test. Do not silently skip it.

- [ ] **Step 4: Run the test against current code**

```bash
pixi run pytest tests/bench/test_probe.py::test_run_point_records_the_realised_concurrent_chroms -v
```

Expected: PASS.

- [ ] **Step 5: Delete the now-false scaffolding comments**

Three blocks assert the field is permanently broken. All are now wrong.

In `scripts/bench_svar2/records.py`, replace the comment block above `concurrent_chroms_used` (`:127-152`, everything from `# The concurrency the planner ACTUALLY dispatched.` down to the field) with:

```python
    # The concurrency the planner ACTUALLY dispatched, parsed from the
    # child's `pipeline config` tracing line. `SweepPoint.concurrent_chroms`
    # is only the REQUEST and is None whenever the point let the planner
    # choose. None here still means UNOBSERVED, never 1 -- `_ram_rows` drops
    # such rows rather than inventing a value, and `sweep_scale.sbatch`
    # aborts on them.
    #
    # Defaulted for the same reason `node` is: records written before this
    # field existed must still load.
    concurrent_chroms_used: int | None = None
```

In `scripts/bench_svar2/probe.py`, replace the comment above `RE_PIPELINE_CONFIG` (`:36-60`) with:

```python
# `tracing::info!(concurrent_chroms, ..., "pipeline config")` in src/lib.rs --
# emitted by BOTH backends (the PGEN one is "pipeline config (PGEN)"), so this
# pattern must stay loose enough to match either. This is the only record of
# the concurrency the planner actually chose: `SweepPoint.concurrent_chroms`
# is only the REQUEST, and is None whenever a point lets the planner decide.
# The fields ride the message across the channel (see `FieldGrab::render` in
# src/logging.rs -- issue #162); `test_run_point_records_the_realised_
# concurrent_chroms` pins the end-to-end path.
```

In `scripts/bench_svar2/probe.py:325-335` (`run_point`'s docstring), delete the sentences claiming `GENORAY_LOG=genoray::monitor=trace` prevents the config line from being parsed. Keep the rest of the docstring.

- [ ] **Step 6: Run the bench suite**

```bash
pixi run pytest tests/bench/ -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
git add tests/bench/test_probe.py scripts/bench_svar2/records.py scripts/bench_svar2/probe.py
git commit -m "test(bench): pin the realised concurrent_chroms end to end

The existing parse_trace tests feed synthetic fmt-layer text to the parser
and passed throughout the period the channel path carried no fields at all.
This drives the real genoray._cli subprocess with concurrent_chroms
unpinned and asserts the planner's choice is recorded.

Also deletes the three comment blocks documenting the field as permanently
None, which Task 1 made false.

Closes #162.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Make an unobserved cc an abort, for every family (#162, guards)

**Depends on:** Task 1 (merged).

**Files:**
- Modify: `scripts/bench_svar2/sweep_scale.sbatch:227-265` (the guard block)
- Modify: `scripts/bench_svar2/model.py` (`_ram_rows`)
- Test: `tests/bench/test_model.py`

**Interfaces:**
- Consumes: `ProbeRecord.concurrent_chroms_used` (now populated, Task 1).
- Produces: `_ram_rows(sweep, ...)` keeps its existing signature and return type; it gains a `warnings.warn` side effect when it drops rows.

**Background:** the sbatch guard is wrapped in `if [ "$NAME" = "vcf_ram" ]; then`, so the six other families — whose data has already been fitted into shipped laws — run unguarded. Its cc check is a `print("WARNING: ...")` that does not fail the job. And `_ram_rows` drops unobserved-cc rows **silently**, which is the half of #162 that lets a law be certified on a subset excluding exactly the planner-chosen production configurations.

- [ ] **Step 1: Write the failing test**

Add to `tests/bench/test_model.py`, beside the existing `_row`/`_sweep_of` helpers (reuse them; read them first — `_row(concurrent_chroms=None, concurrent_chroms_used=None)` already exists at `:766`):

```python
def test_ram_rows_reports_the_rows_it_drops(recwarn):
    """A silently dropped row lets a law be certified on a subset that
    excludes exactly the planner-chosen production configurations -- the
    under-prediction / OOM direction (issue #162). Dropping is correct;
    dropping in silence is not."""
    sweep = _sweep_of(
        [
            _row(concurrent_chroms=None, concurrent_chroms_used=None),
            _row(concurrent_chroms=4, concurrent_chroms_used=4),
        ]
    )
    with pytest.warns(UserWarning, match="unobserved concurrent_chroms"):
        rows = _ram_rows(sweep)
    # the observable behaviour is unchanged: the bad row is still dropped.
    assert len(rows) == 1
```

Ensure `pytest` is imported in the file (it is).

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run pytest tests/bench/test_model.py::test_ram_rows_reports_the_rows_it_drops -v
```

Expected: FAIL with `DID NOT WARN`.

- [ ] **Step 3: Report the drops in `_ram_rows`**

In `scripts/bench_svar2/model.py`, `_ram_rows` is at `:1023-1063`. Add `import warnings` at the top of the file if it is not already there, then make three edits.

Initialise the accumulator beside `rows` (`:1031`):

```python
    rows: list[RamRow] = []
    dropped: list[str] = []
```

Record the id at the drop site — replace `:1051-1052`:

```python
            if cc is None:
                dropped.append(r.point_id)
                continue
```

And warn once before returning — replace `:1063`'s bare `return rows` with:

```python
    if dropped:
        # Dropping is correct (see above); dropping in SILENCE is not. A law
        # fitted on the survivors excludes exactly the planner-chosen
        # production configurations, which is the under-prediction / OOM
        # direction -- issue #162.
        shown = ", ".join(dropped[:5]) + (" ..." if len(dropped) > 5 else "")
        warnings.warn(
            f"_ram_rows dropped {len(dropped)} row(s) with unobserved "
            f"concurrent_chroms (neither pinned on the point nor reported by "
            f"the probe): {shown}. The law is being fitted WITHOUT them.",
            UserWarning,
            stacklevel=2,
        )
    return rows
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pixi run pytest tests/bench/test_model.py -v 2>&1 | tail -15
```

Expected: the new test passes and no existing `test_model.py` test regresses. If an existing test now emits this warning and the suite is configured to error on warnings, assert the warning there rather than suppressing it globally.

- [ ] **Step 5: Hoist the sbatch guard to every family and abort on unobserved cc**

In `scripts/bench_svar2/sweep_scale.sbatch`, the guard currently sits inside `if [ "$NAME" = "vcf_ram" ]; then ... fi` within the `for NAME in $SWEEP_FAMILIES` loop. Two changes:

1. Delete the `if [ "$NAME" = "vcf_ram" ]; then` / `fi` wrapper so the block runs for every family, and change the two hardcoded `vcf_ram` paths to `$NAME`:

```bash
    $PX python - "$JD/plans/$NAME.json" "$JD/out/$NAME.ndjson" <<'PYEOF'
```

2. Change the cc check from a warning to an abort:

```python
missing = [r["point_id"] for r in rows if r.get("concurrent_chroms_used") is None]
if missing:
    # Was a WARNING while src/logging.rs stripped the field from every
    # event (issue #162). Now that the planner's choice actually reaches
    # the row, an unobserved cc is a defect: `_ram_rows` drops such rows,
    # so a law would be fitted on a subset excluding the planner-chosen
    # configurations.
    sys.exit(
        f"ABORT: {len(missing)} of {len(rows)} row(s) never reported a "
        f"realised concurrent_chroms: {missing[:5]}"
    )
```

Keep the existing `bad`, `len(ids) != n_plan` and multi-node checks exactly as they are.

- [ ] **Step 6: Syntax-check the sbatch and its embedded Python**

```bash
bash -n scripts/bench_svar2/sweep_scale.sbatch && echo "bash syntax OK"
```

Then confirm the heredoc's Python parses by extracting it — the guard is a `<<'PYEOF'` block, so a typo in it would only surface hours into a real job:

```bash
awk "/<<'PYEOF'/{f=1;next} /^PYEOF/{f=0} f" scripts/bench_svar2/sweep_scale.sbatch > /carter/users/dlaub/.claude/jobs/797354e0/tmp/guard.py
python3 -m py_compile /carter/users/dlaub/.claude/jobs/797354e0/tmp/guard.py && echo "guard python OK"
```

Expected: both print OK.

- [ ] **Step 7: Commit**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
git add scripts/bench_svar2/sweep_scale.sbatch scripts/bench_svar2/model.py tests/bench/test_model.py
git commit -m "fix(bench): abort on an unobserved concurrent_chroms, for every family

The row-validity guard ran only for vcf_ram, leaving the six families whose
data was already fitted into shipped laws unguarded, and its cc check was a
non-fatal warning because the field could never be populated (#162). Now
that it can be, an unobserved cc is a defect: _ram_rows drops those rows, so
a law would be fitted on a subset that excludes exactly the planner-chosen
production configurations.

_ram_rows now reports what it drops instead of dropping in silence.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Code identity and provenance in the resume key (#159)

**Files:**
- Modify: `scripts/bench_svar2/records.py:90-152` (`ProbeRecord`)
- Modify: `scripts/bench_svar2/sweep.py:25-125`
- Test: `tests/bench/test_sweep.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, for Task 5 and for Task 2's rebase:
  - `ProbeRecord.code_id: str = ""` and `ProbeRecord.run_id: str = ""`
  - `sweep.build_code_id() -> str`
  - `sweep.pending_points(plan: Sequence[SweepPoint], results_path: Path, code_id: str) -> list[SweepPoint]` — **`code_id` is a required third parameter**
  - `sweep.SweepResult` frozen dataclass with `records: list[ProbeRecord]`, `measured: int`, `reused: int`
  - `run_sweep(...) -> SweepResult` (was `list[ProbeRecord]`)

**Background:** on PR #154, job 13351684 printed `18 points recorded` and completed in 18 minutes. Six points were measured; twelve were served from a 2026-08-05 run's ndjson still sitting on the pinned node, describing the *old unbounded reader* the branch existed to replace. A `RamLaw::PGEN` re-fit was computed from the mixture, committed, pushed and announced before review caught it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/bench/test_sweep.py`. Note the existing `_rec` helper at `:46` — extend it with a `code_id` parameter rather than writing a second helper:

```python
def _rec(
    pid: str, digest: str = "aaa", ok: bool = True, code_id: str = ""
) -> ProbeRecord:
    return ProbeRecord(
        point_id=pid,
        ok=ok,
        wall_s=1.0,
        phase1_s=1.0,
        cpu_s=1.0,
        maxrss_mb=1.0,
        digest=digest,
        dense_cap=6,
        dense_occupancy=(0,),
        cpu_shard_pct=(100.0,),
        cpu_exec_pct=(50.0,),
        pending_highwater=0,
        pending_bytes_highwater=0,
        shard_unit_secs=(1.0,),
        code_id=code_id,
    )


def test_pending_requeues_a_point_measured_by_different_code(tmp_path):
    """#159: `point_id` hashes the CONFIGURATION only, so two runs of one
    configuration against different code are indistinguishable to it --
    exactly the case a benchmark exists to distinguish. On PR #154 that
    served 12 rows describing the old reader as measurements of the new
    one."""
    plan = load_plan(_plan_file(tmp_path))
    results = tmp_path / "r.ndjson"
    append_ndjson(results, _rec(plan[0].point_id, code_id="oldbuild00000000"))
    remaining = pending_points(plan, results, "newbuild11111111")
    assert [p.point_id for p in remaining] == [p.point_id for p in plan]


def test_pending_skips_a_point_measured_by_the_same_code(tmp_path):
    plan = load_plan(_plan_file(tmp_path))
    results = tmp_path / "r.ndjson"
    append_ndjson(results, _rec(plan[0].point_id, code_id="samebuild0000000"))
    remaining = pending_points(plan, results, "samebuild0000000")
    assert [p.point_id for p in remaining] == [plan[1].point_id, plan[2].point_id]


def test_pending_requeues_rows_written_before_code_id_existed(tmp_path):
    """Pre-existing rows carry code_id="" and are re-measured. Failing
    toward MEASURING is the correct default for a provenance gap."""
    plan = load_plan(_plan_file(tmp_path))
    results = tmp_path / "r.ndjson"
    append_ndjson(results, _rec(plan[0].point_id, code_id=""))
    assert len(pending_points(plan, results, "anybuild00000000")) == 3


def test_build_code_id_is_stable_and_nonempty():
    """Hashes the ARTIFACT, so it must be reproducible within one process."""
    assert build_code_id() == build_code_id()
    assert len(build_code_id()) == 16
```

And a summary test, reusing the manifest-writing idiom already in `test_run_sweep_is_resumable`:

```python
def test_run_sweep_reports_measured_and_reused_separately(tmp_path):
    """The contaminated run printed `18 points recorded` for 6
    measurements. The row count is the size of the output FILE, not work
    performed (#159)."""
    plan_path = _plan_file(tmp_path)
    results = tmp_path / "r.ndjson"

    def runner(point, manifest, outdir, warm=True):
        return _rec(point.point_id)

    (tmp_path / "c.manifest.json").write_text(
        json.dumps(
            {
                "path": str(tmp_path / "c.vcf.gz"),
                "samples": 10,
                "variants": 100,
                "contigs": ["chr22"],
                "format_fields": [],
                "ploidy": 2,
                "cells": 1000,
                "compressed_bytes": 10,
                "seed": 1,
                "generator_version": 1,
            }
        )
    )

    first = run_sweep(plan_path, results, tmp_path / "out", runner=runner)
    assert (first.measured, first.reused) == (3, 0)

    second = run_sweep(plan_path, results, tmp_path / "out", runner=runner)
    assert (second.measured, second.reused) == (0, 3)
    assert len(second.records) == 3
```

Update the import line at `test_sweep.py:12` to include `build_code_id`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run pytest tests/bench/test_sweep.py -v 2>&1 | tail -25
```

Expected: FAIL — `ImportError` on `build_code_id`, and `pending_points()` taking 2 positional arguments.

- [ ] **Step 3: Add the provenance fields to `ProbeRecord`**

In `scripts/bench_svar2/records.py`, after `concurrent_chroms_used`:

```python
    # SHA256 (16 hex) of the built `_core` extension the measurement ran
    # against -- the ARTIFACT, not the git commit. `pixi run test` does not
    # rebuild the extension, so a commit can advance while the measured
    # binary does not; hashing the .so catches that direction as well as the
    # ordinary one, and distinguishes two A/B builds at one commit.
    #
    # NOT part of `point_id` (that hashes the SweepPoint, which has no code
    # identity and must stay stable across this change), but it IS half of
    # the resume key -- see `sweep.pending_points`. On PR #154, resuming on
    # `point_id` alone served 12 rows measured against the old unbounded
    # reader as measurements of its replacement; the re-fit computed from
    # the mixture was committed, pushed and announced before review caught
    # it (issue #159).
    #
    # Defaulted to "" so pre-existing rows still load. "" never equals a
    # real hash, so those rows are RE-MEASURED rather than trusted --
    # failing toward measuring is the correct default for a provenance gap.
    code_id: str = ""
    # One id per `run_sweep` invocation (`$SLURM_JOB_ID` when set, else a
    # random hex). Makes a mixed results file self-partitioning after the
    # fact, which is what would have let the 2026-08-07 audit attribute rows
    # to jobs without `cmp`-ing against an older file.
    run_id: str = ""
```

- [ ] **Step 4: Implement `build_code_id`, the resume key, and the honest summary**

In `scripts/bench_svar2/sweep.py`, add imports (`dataclasses`, `hashlib`, `os`, `uuid`, and `dataclass` from `dataclasses`) and:

```python
def build_code_id() -> str:
    """SHA256 (16 hex) of the built `_core` extension this process loaded.

    The ARTIFACT, not the git commit: `pixi run test` does not rebuild the
    extension, so a commit can advance while the measured binary does not.
    `probe.run_point` launches the child with `sys.executable`, so the child
    loads this same extension.
    """
    import genoray._core as _core

    return hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest()[:16]


@dataclass(frozen=True)
class SweepResult:
    """`measured` and `reused` are reported separately because the row count
    is the size of the output FILE, not work performed: the contaminated run
    on PR #154 printed `18 points recorded` for 6 measurements (#159)."""

    records: list[ProbeRecord]
    measured: int
    reused: int
```

Change `pending_points` to take the code id and match on the pair:

```python
def pending_points(
    plan: Sequence[SweepPoint], results_path: Path, code_id: str
) -> list[SweepPoint]:
    """Points still to measure AGAINST THIS BUILD.

    Keyed on `(point_id, code_id)`, not `point_id` alone: `point_id` hashes
    the configuration only, so it cannot distinguish two runs of one
    configuration against different code -- which is the one distinction a
    benchmark exists to make (issue #159).
    """
    done = {(r.point_id, r.code_id) for r in read_ndjson(results_path, ProbeRecord)}
    return [p for p in plan if (p.point_id, code_id) not in done]
```

In `run_sweep`, stamp every record centrally so injected test runners need not know about provenance, and return the counts:

```python
    code_id = build_code_id()
    run_id = os.environ.get("SLURM_JOB_ID") or uuid.uuid4().hex[:16]

    pending = pending_points(plan, results_path, code_id)
    reused = len(plan) - len(pending)
    measured = 0

    manifests: dict[str, CorpusManifest] = {}
    for point in pending:
        ...
        rec = runner(point, manifests[point.corpus], outdir)
        # Stamped here, not in `run_point`: one site, and an injected test
        # runner does not have to know about provenance.
        rec = dataclasses.replace(rec, code_id=code_id, run_id=run_id)
        append_ndjson(results_path, rec)
        measured += 1
        ...
```

and at the end:

```python
    records = read_ndjson(results_path, ProbeRecord)
    problem = check_oracle(records, plan)
    if problem:
        raise RuntimeError(problem)
    return SweepResult(records=records, measured=measured, reused=reused)
```

Note `run_sweep` raises `RuntimeError` mid-loop on an oracle failure; that path returns nothing, unchanged.

- [ ] **Step 5: Make the summary line honest**

In `main()`, replace `print(f"{len(recs)} points recorded to {a.results}")`:

```python
    res = run_sweep(a.plan, a.results, a.outdir)
    print(
        f"{res.measured} measured, {res.reused} reused "
        f"({len(res.records)} rows in {a.results})"
    )
```

Check the rest of `main()` for other uses of `recs` and update them to `res.records`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pixi run pytest tests/bench/test_sweep.py tests/bench/test_records.py -v 2>&1 | tail -25
```

Expected: PASS. `test_pending_skips_already_recorded_points` and `test_pending_returns_all_when_no_results_yet` still call `pending_points` with two arguments — update them to pass a code id (use `""` for the "no results" case and the same non-empty id on both sides for the "skips" case, whose intent is that a matching point is skipped).

- [ ] **Step 7: Run the full bench suite**

```bash
pixi run pytest tests/bench/ -v 2>&1 | tail -20
```

Expected: all pass. `build_code_id` imports `genoray._core`, so the extension must be built; if it is not, run `pixi run maturin develop --release` first.

- [ ] **Step 8: Commit**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
git add scripts/bench_svar2/records.py scripts/bench_svar2/sweep.py tests/bench/test_sweep.py
git commit -m "fix(bench): key sweep resume on (point_id, code_id)

point_id is a SHA256 over the SweepPoint configuration alone, so two runs of
one configuration against different code are indistinguishable to it. On PR
#154 that served 12 rows measured against the old unbounded reader as
measurements of its replacement; the RamLaw::PGEN re-fit computed from the
mixture was committed, pushed and announced before review caught it.

ProbeRecord gains code_id (sha256 of the built _core .so -- the artifact,
not the git commit, because pixi run test does not rebuild it) and run_id.
Pre-existing rows carry code_id=\"\" and are re-measured: failing toward
measuring is the correct default for a provenance gap.

run_sweep now reports measured and reused counts separately; the
contaminated run printed \"18 points recorded\" for 6 measurements.

Closes #159.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Generate `holdout_f0`, and fail fast on any missing corpus (#151)

**Depends on:** Task 4 (merged — shares `sweep.py`/`test_sweep.py`) and Task 3 (merged — shares `sweep_scale.sbatch`).

**Files:**
- Modify: `scripts/bench_svar2/sweep.py` (`run_sweep`)
- Modify: `scripts/bench_svar2/sweep_scale.sbatch` (corpora section, after the `holdout` block ~`:113-124`)
- Test: `tests/bench/test_sweep.py`

**Interfaces:**
- Consumes: `SweepResult`, `pending_points(plan, results_path, code_id)` from Task 4.
- Produces: `sweep.check_corpora(points: Sequence[SweepPoint]) -> None`, raising `FileNotFoundError` naming every missing manifest.

**Background:** `plans/build_plans.py:520-528` dispatches a plan point against `corpus_dir / "holdout_f0.manifest.json"`, but the sbatch generates `holdout.vcf.gz` only. Under `set -euo pipefail` the resulting `FileNotFoundError` aborts the whole job, potentially many hours in. `HOLDOUT_F0 = {"samples": 100_000, "variants": 28_000, "format_fields": ()}` — the empty `format_fields` means the block must omit `--format-fields` (whose argparse default is already `""`), not reuse `holdout`'s comma-join.

- [ ] **Step 1: Write the failing test**

Add to `tests/bench/test_sweep.py`:

```python
def test_run_sweep_fails_before_measuring_when_a_corpus_is_missing(tmp_path):
    """#151: manifests load lazily inside the point loop, so a plan point
    whose corpus nobody generates surfaces hours into an overnight job. Fail
    at second zero instead, naming EVERY missing manifest rather than the
    first."""
    pts = [
        {
            "corpus": str(tmp_path / f"absent_{i}.manifest.json"),
            "reader_workers": 1,
            "concurrent_chroms": None,
            "shard_htslib": 0,
            "overshard": 4,
            "chunk_size": 25_000,
            "threads": 16,
            "reps": 1,
        }
        for i in range(2)
    ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(pts))

    def runner(point, manifest, outdir, warm=True):
        raise AssertionError("no point may run when a corpus is missing")

    with pytest.raises(FileNotFoundError) as exc:
        run_sweep(plan_path, tmp_path / "r.ndjson", tmp_path / "out", runner=runner)

    msg = str(exc.value)
    assert "absent_0.manifest.json" in msg
    assert "absent_1.manifest.json" in msg
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run pytest tests/bench/test_sweep.py::test_run_sweep_fails_before_measuring_when_a_corpus_is_missing -v
```

Expected: FAIL — the runner's `AssertionError` fires (or a bare `FileNotFoundError` naming only the first corpus) instead of the complete pre-flight error.

- [ ] **Step 3: Implement the pre-flight**

In `scripts/bench_svar2/sweep.py`:

```python
def check_corpora(points: Sequence[SweepPoint]) -> None:
    """Raise naming EVERY plan point whose corpus manifest is absent.

    `run_sweep` loads manifests lazily inside the point loop, so a plan that
    names a corpus nobody generates fails hours into an overnight job -- and
    under `set -euo pipefail` that aborts the whole sbatch (issue #151, and
    #141 before it for `vlinear2`). Reporting all of them at once means one
    generation pass fixes the run, rather than one per resubmit.
    """
    missing = sorted({p.corpus for p in points if not Path(p.corpus).exists()})
    if missing:
        listed = "\n".join(f"  {m}" for m in missing)
        raise FileNotFoundError(
            f"{len(missing)} corpus manifest(s) named by the plan do not "
            f"exist:\n{listed}"
        )
```

Call it in `run_sweep` on the **pending** points, immediately after `pending_points` and before the loop:

```python
    pending = pending_points(plan, results_path, code_id)
    # Pending only, not the whole plan: a fully-resumed sweep whose corpora
    # were since cleaned up has nothing left to read and must not fail.
    check_corpora(pending)
    reused = len(plan) - len(pending)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pixi run pytest tests/bench/test_sweep.py -v 2>&1 | tail -20
```

Expected: all pass, including Task 4's resume tests (whose `_plan_file` writes a real `c.manifest.json` in the two `run_sweep` tests — confirm both still create it before calling `run_sweep`; `test_load_plan_returns_sweep_points` and the `pending_points` tests never call `run_sweep`, so they are unaffected).

- [ ] **Step 5: Add the `holdout_f0` corpus block**

In `scripts/bench_svar2/sweep_scale.sbatch`, directly after the existing `holdout` generation block (which ends with the `--format-fields "$HOLDOUT_F"` line, ~`:124`):

```bash
# Hold-out at F=0: same single-source-of-truth pattern as the block above.
# `build_plans.py` dispatches a plan point against this corpus but nothing
# generated it, so `holdout` aborted the whole job on any machine where it
# did not already exist (issue #151 -- the same gap #141 fixed for
# vlinear2). HOLDOUT_F0["format_fields"] is empty, so `--format-fields` is
# deliberately omitted: its argparse default is already "".
read -r HOLDOUT_F0_S HOLDOUT_F0_V < <($PX python -c "
from scripts.bench_svar2.plans.build_plans import HOLDOUT_F0
print(HOLDOUT_F0['samples'], HOLDOUT_F0['variants'])
")
[ -f "$JD/corpora/holdout_f0.manifest.json" ] || $PX python -m scripts.bench_svar2.scale_corpus \
  --out "$JD/corpora/holdout_f0.vcf.gz" --samples "$HOLDOUT_F0_S" \
  --variants "$HOLDOUT_F0_V" --procs 16 --bgzip-threads 8
```

Note it must sit **inside** the `if wants scale || wants contig || wants holdout || ...` guard (`:93`) that wraps the legacy corpora, alongside the `holdout` block it mirrors.

- [ ] **Step 6: Verify the shape read-back and the sbatch syntax**

```bash
bash -n scripts/bench_svar2/sweep_scale.sbatch && echo "bash syntax OK"
pixi run python -c "
from scripts.bench_svar2.plans.build_plans import HOLDOUT_F0
print(HOLDOUT_F0['samples'], HOLDOUT_F0['variants'])
"
```

Expected: `bash syntax OK`, then `100000 28000`. Do **not** run the generation itself — a 100,000-sample corpus is a cluster-scale job, out of scope here.

- [ ] **Step 7: Run the full bench suite**

```bash
pixi run pytest tests/bench/ -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
git add scripts/bench_svar2/sweep.py scripts/bench_svar2/sweep_scale.sbatch tests/bench/test_sweep.py
git commit -m "fix(bench): generate holdout_f0 and pre-flight every plan corpus

build_plans.py dispatches a plan point against holdout_f0 but sweep_scale
.sbatch generated holdout only, so the holdout family raised FileNotFound
on any machine where the corpus did not already exist -- aborting the whole
job under set -euo pipefail, potentially hours in. Same gap #141 fixed for
vlinear2.

run_sweep now verifies every pending point's corpus manifest before running
any point, naming all missing ones at once, so the next plan point added
without a corpus block fails at second zero instead of repeating this.

Closes #151.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Final verification (after all five tasks)

- [ ] **Rust suite**

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/.claude/jobs/797354e0/tmp/cargo-target
cargo test --no-default-features --features conversion 2>&1 | tail -20
```

- [ ] **Rebuild the extension, then the Python suite**

```bash
pixi run maturin develop --release
pixi run pytest tests/bench/ -v 2>&1 | tail -20
pixi run pytest -m "not network" 2>&1 | tail -20
```

- [ ] **Confirm committed sweep data still loads and still fits.** The `code_id` addition must not invalidate any committed results:

```bash
pixi run python -c "
from pathlib import Path
from scripts.bench_svar2.records import ProbeRecord
from scripts.bench_svar2.sweep import read_ndjson
p = Path('docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed-data/vcf_ram.ndjson')
rows = read_ndjson(p, ProbeRecord)
print(f'{len(rows)} rows loaded; code_id values: {sorted({r.code_id for r in rows})}')
assert len(rows) == 48
"
```

Expected: `48 rows loaded; code_id values: ['']`. Pre-existing rows load with the empty default — the read schema is unchanged.

- [ ] **Open a PR** against `main` from `worktree-svar2-bench-integrity`, body closing #162, #159 and #151, noting that no law was re-fitted and no sweep re-run.

## Notes for the reviewer

- **The one thing worth checking hardest** is Task 1's rendering, because it changes user-visible CLI output for every structured log line in the codebase, not just `pipeline config`. Confirm no existing Rust test asserts an exact message that now gains a field suffix.
- **`pending_points` gained a required third parameter** rather than an optional one, deliberately: an optional `code_id` that a caller forgets reproduces #159 exactly.
- **Task 2 Step 3 is the only step permitted to be skipped**, and only with an explicit note in the commit.
