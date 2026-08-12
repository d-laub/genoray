# `RamLaw::VCF` Envelope Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** Tasks are grouped into waves below. Within a wave, tasks touch
> disjoint files and MUST be dispatched concurrently using
> superpowers:dispatching-parallel-agents. Use **Sonnet or weaker** for
> implementers; reserve stronger models for review and for fixing a critical
> implementer failure.
>
> **Wave 1:** Tasks 1, 2, 3 (parallel) — `probe.py`+`records.py`, `sweep_scale.sbatch`, corpus generation
> **Wave 2:** Tasks 4, 5 (parallel) — `model.py`+`fit_ram.py`, `build_plans.py`
> **Wave 3:** Task 6 (alone) — offline form check, gates the functional form
> **Wave 4:** Task 7 (alone) — cluster measurement
> **Wave 5:** Tasks 8, 9 (sequential) — ship the law, close the issue

**Goal:** Refit `RamLaw::VCF` as an envelope against a crossed sweep that varies `concurrent_chroms`, so both conversion backends' memory laws are fitted the way `plan_sharded` consumes them, and close #158 before the next release.

**Architecture:** The fitting machinery (`fit_ram_law`'s envelope LP, `RamRow.concurrent_chroms`, `_ram_rows`' drop rule) already exists and is backend-generic — this is mostly a measurement design. Work splits into harness changes that need no cluster (record the realised `cc`, generate VCF corpora through `vcfixture bulk`, add crossed plan families, add a reproducible fit CLI), one offline experiment on already-committed PGEN data that decides the functional form, then one pinned sweep and the refit itself.

**Tech Stack:** Rust (pyo3, `src/budget.rs`), Python 3.10+ (numpy, scipy LP), pixi, Slurm, `vcfixture bulk` CLI ≥ v0.5.0, plink2, bcftools/tabix.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-11-ramlaw-vcf-envelope-design.md`. Read it before starting any task.
- **Conventional Commits** on every commit (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `refactor:`). Scope with `(svar2)`, `(bench)`, or `(specs)` as the existing history does.
- **Never edit `CHANGELOG.md`** and never bump the version by hand — `commitizen` in the release workflow owns both.
- **`RamLaw` is load-bearing in production.** Change a law only alongside a refit that says so, and record that refit's gate result and `n` in the constant's doc comment.
- **A law is an UPPER BOUND, not a prediction.** Never re-introduce OLS-plus-CI-padding. `r2` is descriptive only and must never be used as the shipping criterion.
- **Rust tests:** `cargo test --no-default-features --features conversion` (473 passing at plan time). Bare `--no-default-features` silently skips the conversion path; dropping `--no-default-features` fails to link the pyo3 test binary.
- **Python tests:** `pixi run test`.
- **`export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target`** before any cargo or pre-commit invocation — an NFS `target/` bus-errors the linker.
- **`pixi run test` does NOT rebuild the Rust extension.** Run `maturin develop --release` before any Python-level verification of a Rust change.
- **vcfixture-rs ≥ v0.5.0** for corpus generation. v0.5.0 is a breaking output change: same seed produces different bytes than v0.4.0. Never pool corpora across that boundary.
- **Never run compute-heavy work on the login node** — `sbatch` it. Never detach a process (`nohup`, `setsid`, trailing `&`); use the harness's background flag so it stays tracked.
- **Inside an `sbatch` script, `unset CLAUDE_JOB_DIR` and use `/local/$USER`.** That variable points at the *submitting* node's scratch and dangles anywhere else.

---

## Wave 1

### Task 1: Record the realised `concurrent_chroms` on every probe record

Today a sweep point that does not pin `concurrent_chroms` is unfittable — `_ram_rows` drops it, which cost 12 of the 58 PGEN crossed rows. Both backends already log the value; nothing captures it.

**Files:**
- Modify: `scripts/bench_svar2/records.py:106-120` (add field to `ProbeRecord`)
- Modify: `scripts/bench_svar2/probe.py:33-35` (regex), `:42-84` (`parse_trace`), `:350-360` and `:371-383` (record construction)
- Test: `tests/bench/test_probe.py`, `tests/bench/test_records.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ProbeRecord.concurrent_chroms_used: int | None` (default `None`), and `parse_trace(text)["concurrent_chroms_used"]: int | None`. Task 4 reads both.

- [ ] **Step 1: Write the failing test for the trace parser**

Add to `tests/bench/test_probe.py`:

```python
def test_parse_trace_extracts_realised_concurrent_chroms():
    # Both backends emit a "pipeline config" line; the VCF one is bare, the
    # PGEN one is suffixed. tracing's fmt layer prints the message first, then
    # the fields, so `concurrent_chroms=` follows the message text.
    vcf = (
        "2026-08-11T00:00:00Z  INFO genoray: pipeline config "
        "concurrent_chroms=4 htslib_threads=2 monolithic_reader_active=8 "
        "reader_workers=3 sharded_vcf_active=16 processing_threads=31\n"
    )
    assert parse_trace(vcf)["concurrent_chroms_used"] == 4

    pgen = (
        "2026-08-11T00:00:00Z  INFO genoray: pipeline config (PGEN) "
        "concurrent_chroms=8 reader_workers=1 processing_threads=31\n"
    )
    assert parse_trace(pgen)["concurrent_chroms_used"] == 8


def test_parse_trace_reports_unknown_concurrent_chroms_as_none():
    # A run whose log level suppressed the line, or a crash before planning.
    # None means UNOBSERVED. It must never be coded as 1 -- that is exactly
    # what produced a 41 MB per-contig estimate against a measured 89.67
    # (issue #158).
    assert parse_trace("no config line here\n")["concurrent_chroms_used"] is None
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
pixi run pytest tests/bench/test_probe.py::test_parse_trace_extracts_realised_concurrent_chroms -v
```
Expected: FAIL with `KeyError: 'concurrent_chroms_used'`.

- [ ] **Step 3: Implement the parse**

In `scripts/bench_svar2/probe.py`, add next to the other regexes (after `RE_UNIT`, line 35):

```python
# `tracing::info!(concurrent_chroms, ..., "pipeline config")` in src/lib.rs --
# emitted by BOTH backends (the PGEN one is "pipeline config (PGEN)"), so this
# pattern must stay loose enough to match either. This is the ONLY record of
# the concurrency the planner actually chose: `SweepPoint.concurrent_chroms` is
# the REQUEST, and is None whenever the point let the planner decide.
RE_PIPELINE_CONFIG = re.compile(r"pipeline config.*")
```

In `parse_trace`, after the sampler loop and before the `return`:

```python
    cc_used: int | None = None
    for line in RE_PIPELINE_CONFIG.findall(plain):
        v = _field(line, "concurrent_chroms")
        if v is not None:
            cc_used = int(v)
```

and add to the returned dict:

```python
        "concurrent_chroms_used": cc_used,
```

- [ ] **Step 4: Run the parser tests**

```bash
pixi run pytest tests/bench/test_probe.py -k concurrent_chroms -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Write the failing test for the record field**

Add to `tests/bench/test_records.py`:

```python
def test_probe_record_round_trips_concurrent_chroms_used():
    rec = ProbeRecord(
        point_id="abc",
        ok=True,
        wall_s=1.0,
        phase1_s=1.0,
        cpu_s=1.0,
        maxrss_mb=100.0,
        digest="d",
        dense_cap=1,
        dense_occupancy=(),
        cpu_shard_pct=(),
        cpu_exec_pct=(),
        pending_highwater=0,
        pending_bytes_highwater=0,
        shard_unit_secs=(),
        concurrent_chroms_used=8,
    )
    assert from_json(ProbeRecord, to_json(rec)) == rec


def test_probe_record_without_concurrent_chroms_used_still_loads():
    # Records written before the field existed must keep loading, the same
    # reason `node` carries a default. `from_json` drops unknown keys but
    # cannot invent missing ones, so the default is what makes this work.
    payload = {
        "point_id": "abc",
        "ok": True,
        "wall_s": 1.0,
        "phase1_s": 1.0,
        "cpu_s": 1.0,
        "maxrss_mb": 100.0,
        "digest": "d",
        "dense_cap": 1,
        "dense_occupancy": [],
        "cpu_shard_pct": [],
        "cpu_exec_pct": [],
        "pending_highwater": 0,
        "pending_bytes_highwater": 0,
        "shard_unit_secs": [],
    }
    assert from_json(ProbeRecord, json.dumps(payload)).concurrent_chroms_used is None
```

Note the codec shape: `from_json` is a module-level function taking a class and a
JSON **string** (`from_json(ProbeRecord, to_json(rec))`), not a classmethod on the
dataclass. `test_records.py` already imports `from_json`, `to_json` and
`ProbeRecord`; add `json` if it is not already imported.

- [ ] **Step 6: Run it to make sure it fails**

```bash
pixi run pytest tests/bench/test_records.py -k concurrent_chroms_used -v
```
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument`.

- [ ] **Step 7: Add the field**

In `scripts/bench_svar2/records.py`, immediately after the `node: str = ""` field of `ProbeRecord`:

```python
    # The concurrency the planner ACTUALLY dispatched, parsed from the child's
    # `pipeline config` line. `SweepPoint.concurrent_chroms` is only the
    # REQUEST and is None whenever the point let the planner choose, which
    # made every such point unfittable -- 12 of the 58 rows in the 2026-08-08
    # PGEN crossed sweep (issue #158). None still means UNOBSERVED, never 1.
    #
    # Defaulted for the same reason `node` is: records written before this
    # field existed must still load.
    concurrent_chroms_used: int | None = None
```

- [ ] **Step 8: Populate it in both record-construction paths**

In `scripts/bench_svar2/probe.py`, the failure path near line 350 constructs a `ProbeRecord` with zeroed telemetry — add `concurrent_chroms_used=None` there. The success path near line 371 reads from the parsed dict — add:

```python
                concurrent_chroms_used=t["concurrent_chroms_used"],
```

- [ ] **Step 9: Run the full bench test suite**

```bash
pixi run pytest tests/bench/test_probe.py tests/bench/test_records.py -v
```
Expected: PASS, no regressions.

- [ ] **Step 10: Commit**

```bash
git add scripts/bench_svar2/probe.py scripts/bench_svar2/records.py tests/bench/test_probe.py tests/bench/test_records.py
git commit -m "feat(bench): record the concurrency the planner actually dispatched

\`SweepPoint.concurrent_chroms\` is the REQUEST and is None whenever a point
lets the planner choose, so \`_ram_rows\` had to drop those rows entirely --
12 of the 58 in the 2026-08-08 PGEN crossed sweep. Both backends already log
the realised value on their \`pipeline config\` line; this captures it.

None still means UNOBSERVED. Coding an unknown cc as 1 is what produced a
41 MB per-contig estimate against a directly measured 89.67.

Refs #158"
```

---

### Task 2: Make `sweep_scale.sbatch` survivable

This driver still carries the bug that killed job 13351680 after 6h57m with every corpus already generated, plus a hardcoded worktree path that no longer exists. Task 7 cannot run until this is fixed.

**Files:**
- Modify: `scripts/bench_svar2/sweep_scale.sbatch:15-21` (worktree path, scratch dir), `:41` (`--nodelist` note)

**Interfaces:**
- Consumes: nothing.
- Produces: an sbatch script whose `$SCRATCH` is node-local and whose repo root is derived, not hardcoded. Task 7 runs it.

- [ ] **Step 1: Replace the worktree and scratch resolution**

Replace lines 15-21 of `scripts/bench_svar2/sweep_scale.sbatch` (the `WT=...`, `cd "$WT"`, `JD=...` block) with:

```bash
# Repo root DERIVED, not hardcoded. This previously pinned a specific worktree
# (`bench-pr140-reader-workers`) that no longer exists, so resubmitting from a
# different tree silently measured the wrong checkout -- or died on the first
# command. Slurm's default cwd is the SUBMISSION directory, and every
# `python -m scripts.bench_svar2.<module>` below needs cwd = repo root to
# resolve as a package.
WT=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
cd "$WT"

# NOT $CLAUDE_JOB_DIR/tmp. The harness symlinks that path to node-local disk on
# the SUBMITTING node, and `sbatch --export=ALL` carries the variable to
# whatever node runs the job, where it is a DANGLING symlink: mkdir of a child
# raises FileNotFoundError and the parents=True retry then raises FileExistsError
# on the symlink itself, naming neither the symlink nor the node. That killed
# job 13351680 after 6:57:53 with all corpora already built. `probe.py` reads
# the variable directly, so it must be unset here, not merely avoided.
unset CLAUDE_JOB_DIR
SCRATCH="/local/$USER/scale-sweep"
mkdir -p "$SCRATCH"
JD="$SCRATCH"
```

- [ ] **Step 2: Add the node-pinning requirement**

Immediately below the `#SBATCH --time=72:00:00` line, add:

```bash
# PIN THE NODE for any run whose rows will be compared or pooled. Measured
# spread on byte-identical work is 2.08x (151.9s on one node vs 73.2s on
# another), and unpinned rows have already produced published-then-retracted
# findings. Submit with:  sbatch --nodelist=carter-cn-04 sweep_scale.sbatch
```

- [ ] **Step 3: Verify the script parses and the paths resolve**

```bash
bash -n scripts/bench_svar2/sweep_scale.sbatch && echo "syntax OK"
rg -n 'CLAUDE_JOB_DIR|/local/\$USER|rev-parse' scripts/bench_svar2/sweep_scale.sbatch
```
Expected: `syntax OK`, and the only `CLAUDE_JOB_DIR` hit is the `unset` line plus its comment.

- [ ] **Step 4: Confirm the sibling driver still agrees**

```bash
rg -n 'unset CLAUDE_JOB_DIR|SCRATCH=' scripts/bench_svar2/sweep_pgen.sbatch scripts/bench_svar2/sweep_scale.sbatch
```
Expected: both scripts now `unset CLAUDE_JOB_DIR` and both define `SCRATCH` under `/local/$USER`.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_svar2/sweep_scale.sbatch
git commit -m "fix(bench): stop sweep_scale.sbatch inheriting another node's scratch

\`JD=\"\${CLAUDE_JOB_DIR:?}\"/tmp\` is a dangling symlink on any node but the
submitting one. sweep_pgen.sbatch got the \`unset\` + /local/\$USER fix in
59665aa after that trap killed job 13351680 at 6:57:53 with every corpus
already generated; this driver never did.

Also derives the repo root from \$SLURM_SUBMIT_DIR instead of hardcoding a
worktree that no longer exists, and documents the --nodelist requirement."
```

---

### Task 3: Generate VCF corpora through `vcfixture bulk`

`scale_corpus.py` uses numpy plus a process pool; `pgen_corpus.py` is the module that shells to `vcfixture bulk`. The RAM-law VCF corpus is the PGEN pipeline stopped one step early, so the CLI wrapper moves to a shared module both corpus generators use.

**Files:**
- Create: `scripts/bench_svar2/vcfixture_cli.py`
- Create: `scripts/bench_svar2/vcf_corpus.py`
- Modify: `scripts/bench_svar2/pgen_corpus.py:45` (`PROFILE`), `:62-87` (`resolve_vcfixture`), `:140-162` (the `vcfixture bulk` subprocess call)
- Modify: `scripts/bench_svar2/records.py` (`CorpusManifest`: add `generator_cli_version`)
- Test: `tests/bench/test_vcf_corpus.py` (new), `tests/bench/test_pgen_corpus.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `vcfixture_cli.resolve_vcfixture() -> Path`, `vcfixture_cli.PROFILE: Path`, `vcfixture_cli.cli_version() -> str`, `vcfixture_cli.bulk(samples, variants, contigs, seed, fmt, out) -> None`
  - `vcf_corpus.VcfCorpusSpec(samples: int, variants: int, contigs: tuple[str, ...], seed: int)`
  - `vcf_corpus.corpus_stem(spec) -> str` returning `f"vcfx_s{samples}_v{variants}"`
  - `vcf_corpus.generate(spec, outdir: Path) -> CorpusManifest`
  - `CorpusManifest.generator_cli_version: str` (default `""`)

  Task 5 imports `corpus_stem` to name the corpora its plan points reference.

- [ ] **Step 1: Verify the CLI can emit bgzf VCF directly**

```bash
VCFIXTURE_BIN=${VCFIXTURE_BIN:-$(command -v vcfixture)} ; echo "$VCFIXTURE_BIN"
"$VCFIXTURE_BIN" --version
"$VCFIXTURE_BIN" bulk --help | rg -n 'format|vcf|bcf|-o'
```
Expected: version ≥ 0.5.0, and `--format` accepting a `vcf` value that writes bgzf. If the binary is absent, install it with `cargo install vcfixture --features cli` or set `VCFIXTURE_BIN`.

If `--format vcf` is **not** available, stop and report before writing any code. The fallback is generating BCF and converting with `bcftools view -Oz`, which changes only this task's `generate`. But taking that branch adds a requirement to Task 7: re-measure 2–3 crossed points on a bgzf-VCF copy of the same corpus and assert `maxrss_mb` agrees within the 63 MB reproducibility floor. BCF input is known to delete 23–41% of reader CPU, and RSS parity between the two input formats must be demonstrated rather than assumed before a law fitted on one is applied to the other.

- [ ] **Step 2: Write the failing test for the shared CLI wrapper**

Create `tests/bench/test_vcf_corpus.py`:

```python
import json
import shutil
from pathlib import Path

import pytest

from scripts.bench_svar2 import vcfixture_cli
from scripts.bench_svar2.vcf_corpus import VcfCorpusSpec, corpus_stem, generate

# The bulk CLI is a Rust binary, separate from the PyPI `vcfixture` package
# (which ships no console script). Its absence must SKIP, not fail, or CI
# breaks on a machine that never had it.
_HAVE_CLI = shutil.which("vcfixture") is not None or bool(
    __import__("os").environ.get("VCFIXTURE_BIN")
)
requires_cli = pytest.mark.skipif(not _HAVE_CLI, reason="vcfixture bulk CLI not found")


def test_corpus_stem_is_shape_derived():
    spec = VcfCorpusSpec(samples=4000, variants=350000, contigs=("chr1",), seed=42)
    assert corpus_stem(spec) == "vcfx_s4000_v350000"


@requires_cli
def test_cli_version_is_reported():
    v = vcfixture_cli.cli_version()
    assert v and v[0].isdigit(), f"expected a version string, got {v!r}"


@requires_cli
def test_generate_writes_indexed_bgzf_vcf_and_a_manifest(tmp_path: Path):
    spec = VcfCorpusSpec(
        samples=8, variants=44, contigs=("chr1", "chr2"), seed=42
    )
    m = generate(spec, tmp_path)

    vcf = Path(m.path)
    assert vcf.name == "vcfx_s8_v44.vcf.gz"
    assert vcf.exists() and vcf.stat().st_size > 0
    # The sharded reader needs an index; without one from_vcf cannot shard.
    assert (vcf.parent / f"{vcf.name}.tbi").exists() or (
        vcf.parent / f"{vcf.name}.csi"
    ).exists()

    assert m.samples == 8
    assert m.variants == 44
    assert m.cells == 8 * 44
    assert m.format_fields == ()
    assert m.ploidy == 2
    # v0.5.0 changed generated bytes for a given seed. A manifest that cannot
    # tell v0.4.0 corpora from v0.5.0 ones invites pooling incompatible data.
    assert m.generator_cli_version


@requires_cli
def test_generate_is_cached_on_the_cli_version(tmp_path: Path):
    spec = VcfCorpusSpec(samples=8, variants=44, contigs=("chr1",), seed=42)
    first = generate(spec, tmp_path)
    mtime = Path(first.path).stat().st_mtime_ns

    second = generate(spec, tmp_path)
    assert second == first
    assert Path(second.path).stat().st_mtime_ns == mtime, "should not regenerate"

    # Poison the recorded CLI version: the corpus must be regenerated, not reused.
    manifest_path = tmp_path / "vcfx_s8_v44.manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["_key"]["cli_version"] = "0.0.0-not-a-real-version"
    manifest_path.write_text(json.dumps(payload))

    third = generate(spec, tmp_path)
    assert Path(third.path).stat().st_mtime_ns != mtime, "should have regenerated"
```

- [ ] **Step 3: Run it to make sure it fails**

```bash
pixi run pytest tests/bench/test_vcf_corpus.py -v
```
Expected: FAIL at import (`ModuleNotFoundError: scripts.bench_svar2.vcfixture_cli`).

- [ ] **Step 4: Extract the shared CLI wrapper**

Create `scripts/bench_svar2/vcfixture_cli.py`:

```python
"""Shared access to the `vcfixture bulk` Rust CLI.

Extracted from `pgen_corpus.py` when `vcf_corpus.py` began needing the same
binary: the VCF corpus is the PGEN pipeline stopped one step early (bulk emits
the variant file; only the PGEN path then runs plink2 over it), so resolution,
profile choice, and argument construction belong in one place.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

PROFILE = Path(__file__).parent / "profiles" / "germline-1kgp-varskew.json"


def resolve_vcfixture() -> Path:
    """Locate the `vcfixture` bulk CLI.

    This is NOT the PyPI `vcfixture` package pinned in pixi.toml -- that ships
    no console script (no entry_points.txt, no bin/vcfixture). The bulk
    generator is a separate Rust binary on its own version line. Shelling out
    to a bare `vcfixture` therefore passes on a dev box that happens to have it
    built and fails in CI with FileNotFoundError, so resolution is explicit and
    the error says how to fix it.
    """
    env = os.environ.get("VCFIXTURE_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"VCFIXTURE_BIN={env} is not a file")
    found = shutil.which("vcfixture")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "vcfixture bulk CLI not found. It is a Rust binary, separate from the "
        "PyPI `vcfixture` package (which has no CLI). Install it with "
        "`cargo install vcfixture --features cli`, or point VCFIXTURE_BIN at "
        "an existing build."
    )


def cli_version() -> str:
    """The bulk CLI's own version, recorded in every manifest.

    v0.5.0 is an explicit BREAKING output change -- "generated output for a
    given seed differs from v0.4.0 ... existing corpora must be regenerated" --
    so a manifest that cannot distinguish the two invites silently pooling
    incompatible corpora into one fit. Cached corpora key on this string.
    """
    out = subprocess.run(
        [str(resolve_vcfixture()), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # `clap` prints "vcfixture 0.5.0"; keep only the version token.
    return out.split()[-1] if out else ""


def bulk(
    *,
    samples: int,
    variants: int,
    contigs: Sequence[str],
    seed: int,
    fmt: str,
    out: Path,
) -> None:
    """Run `vcfixture bulk` for one corpus. `fmt` is "vcf" or "bcf"."""
    subprocess.run(
        [
            str(resolve_vcfixture()),
            "bulk",
            "--profile",
            str(PROFILE),
            "--samples",
            str(samples),
            "--contigs",
            ",".join(contigs),
            "--records",
            str(variants),
            "--payload",
            "gt-only",
            "--format",
            fmt,
            "--seed",
            str(seed),
            "-o",
            str(out),
        ],
        check=True,
    )
```

- [ ] **Step 5: Point `pgen_corpus.py` at the shared wrapper**

In `scripts/bench_svar2/pgen_corpus.py`, delete the local `PROFILE` assignment (line 45) and the whole `resolve_vcfixture` function (lines 62-87), and replace the inline `subprocess.run([...vcfixture...])` block (lines 140-162) with:

```python
    bulk(
        samples=spec.samples,
        variants=spec.variants,
        contigs=spec.contigs,
        seed=spec.seed,
        fmt="bcf",
        out=bcf,
    )
```

Add near the other imports:

```python
# Re-exported so existing callers (sweep_pgen.sbatch, tests) keep importing
# `resolve_vcfixture` and `PROFILE` from here.
from scripts.bench_svar2.vcfixture_cli import (  # noqa: F401
    PROFILE,
    bulk,
    cli_version,
    resolve_vcfixture,
)
```

Remove any now-unused `os`/`shutil` imports that `ruff check` flags.

- [ ] **Step 6: Add `generator_cli_version` to the manifest**

In `scripts/bench_svar2/records.py`, add to `CorpusManifest` after `generator_version`:

```python
    # The `vcfixture bulk` CLI version this corpus came from, for corpora that
    # used it (empty for `scale_corpus.py`'s own generator). Separate from
    # `generator_version`, which versions THIS harness's generation logic:
    # v0.5.0 of the CLI changed generated bytes for a given seed, so the two
    # move independently and pooling across either is wrong.
    generator_cli_version: str = ""
```

- [ ] **Step 7: Write the VCF corpus generator**

Create `scripts/bench_svar2/vcf_corpus.py`:

```python
"""Generate bgzf-VCF corpora for the RAM-law sweep via `vcfixture bulk`.

The same generator and profile `pgen_corpus.py` uses, stopped one step before
plink2. Sharing it retires the caveat on `RamLaw::PGEN` that the two backends'
laws are not comparable because their corpora came from different generators.

NOT a replacement for `scale_corpus.py`, which still owns the hold-out, both
V-linearity ladders, the FORMAT-field corpora and `size_corpus`'s chunk-size
derivation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.bench_svar2.records import CorpusManifest, from_json, to_json
from scripts.bench_svar2.vcfixture_cli import PROFILE, bulk, cli_version

# Versions THIS module's generation logic (layout, indexing, manifest shape) --
# not the CLI's output bytes, which `generator_cli_version` records separately.
GENERATOR_VERSION = 1


@dataclass(frozen=True)
class VcfCorpusSpec:
    samples: int
    variants: int
    contigs: tuple[str, ...]
    seed: int


def corpus_stem(spec: VcfCorpusSpec) -> str:
    """Filename stem for a corpus of this shape.

    Corpora land FLAT in one directory because `model._load_manifests` globs
    `*.manifest.json` non-recursively and keys by FILENAME -- a per-corpus
    subdirectory would make every manifest `corpus.manifest.json` and they
    would collide. The `vcfx_` prefix keeps these distinct from
    `scale_corpus.py`'s `s{N}` stems in that same flat namespace.
    """
    return f"vcfx_s{spec.samples}_v{spec.variants}"


def generate(spec: VcfCorpusSpec, outdir: Path) -> CorpusManifest:
    """Generate (or reuse) a bgzf-VCF corpus in `outdir`.

    Cached on the full spec plus GENERATOR_VERSION plus the profile's content
    hash plus the CLI version: `vcfixture --seed` is byte-reproducible
    regardless of thread count WITHIN a CLI major version, so a corpus is
    reproducible and there is no reason to pay for it twice -- but v0.5.0
    changed those bytes, so the version belongs in the key.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    stem = corpus_stem(spec)
    manifest_path = outdir / f"{stem}.manifest.json"
    version = cli_version()
    profile_hash = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    # Round-tripped through JSON before comparing: `dataclasses.asdict(spec)`
    # holds `contigs` as a tuple while the cached copy loaded off disk holds a
    # list, and comparing those directly never matches -- the corpus would be
    # silently regenerated on every call.
    key = json.loads(
        json.dumps(
            {
                **dataclasses.asdict(spec),
                "generator_version": GENERATOR_VERSION,
                "profile_hash": profile_hash,
                "cli_version": version,
            }
        )
    )

    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached.get("_key") == key and Path(cached["path"]).exists():
            payload = {k: v for k, v in cached.items() if k != "_key"}
            return from_json(CorpusManifest, json.dumps(payload))

    vcf = outdir / f"{stem}.vcf.gz"
    bulk(
        samples=spec.samples,
        variants=spec.variants,
        contigs=spec.contigs,
        seed=spec.seed,
        fmt="vcf",
        out=vcf,
    )
    # The sharded reader seeks per shard, so an index is not optional.
    subprocess.run(["tabix", "-f", "-p", "vcf", str(vcf)], check=True)

    manifest = CorpusManifest(
        path=str(vcf),
        samples=spec.samples,
        variants=spec.variants,
        contigs=spec.contigs,
        format_fields=(),
        ploidy=2,
        cells=spec.samples * spec.variants,
        compressed_bytes=vcf.stat().st_size,
        seed=spec.seed,
        generator_version=GENERATOR_VERSION,
        generator_cli_version=version,
    )
    payload = json.loads(to_json(manifest))
    payload["_key"] = key
    manifest_path.write_text(json.dumps(payload, indent=1) + "\n")
    return manifest
```

- [ ] **Step 8: Run the tests**

```bash
pixi run pytest tests/bench/test_vcf_corpus.py tests/bench/test_pgen_corpus.py -v
```
Expected: PASS. Without the CLI on PATH the `@requires_cli` tests SKIP and `test_corpus_stem_is_shape_derived` still passes — verify the skip reason names the binary rather than erroring.

- [ ] **Step 9: Run the whole bench suite for import fallout**

```bash
pixi run pytest tests/bench/ -q
```
Expected: PASS. `pgen_corpus`'s re-export keeps existing importers working.

- [ ] **Step 10: Commit**

```bash
git add scripts/bench_svar2/vcfixture_cli.py scripts/bench_svar2/vcf_corpus.py scripts/bench_svar2/pgen_corpus.py scripts/bench_svar2/records.py tests/bench/test_vcf_corpus.py
git commit -m "feat(bench): generate VCF RAM-law corpora with vcfixture bulk

The VCF corpus is the PGEN pipeline stopped one step early, so the CLI wrapper
moves to a shared \`vcfixture_cli\` module and \`vcf_corpus\` emits indexed
bgzf VCF from the same profile. Sharing the generator retires the caveat that
RamLaw::VCF and RamLaw::PGEN are not comparable because their corpora came
from different generators.

Manifests now record the CLI version: v0.5.0 changed generated bytes for a
given seed, so corpora must not be pooled across that boundary.

scale_corpus.py is untouched -- it still owns the hold-out, both V-linearity
ladders and the FORMAT-field corpora.

Refs #158"
```

---

## Wave 2

### Task 4: Feed every sweep into the RAM fit, and give the fit a CLI

Two gaps. `_ram_rows` is called with `scale, contig, holdout, vlinear, vlinear2` — it never sees `concurrency`, the one existing VCF family that varies `cc` at fixed `(S, chunk_size)`, which is precisely why `RamLaw::VCF` carries `per_contig_mb = 0.0`. And fitting has no entry point: the PGEN law was fitted ad hoc, so neither Phase 0 nor Phase 3 is reproducible.

**Files:**
- Modify: `scripts/bench_svar2/model.py:762` (`_SWEEP_NAMES`), `:990-1027` (`_ram_rows`), `:1200` (call site)
- Create: `scripts/bench_svar2/fit_ram.py`
- Test: `tests/bench/test_model.py`, `tests/bench/test_fit_ram.py` (new)

**Interfaces:**
- Consumes: `ProbeRecord.concurrent_chroms_used` (Task 1).
- Produces: `python -m scripts.bench_svar2.fit_ram --results DIR --plans DIR --manifests DIR --backend {vcf,pgen} [--margin FLOAT]` printing fitted coefficients and the gate table. Task 6 and Task 8 both run it.

- [ ] **Step 1: Write the failing test for the realised-cc fallback**

Add to `tests/bench/test_model.py`:

```python
def test_ram_rows_fall_back_to_the_realised_concurrent_chroms():
    # A point that let the planner choose (`concurrent_chroms=None`) used to be
    # dropped outright. If the RECORD observed the realised value, the row is
    # fittable and must be kept -- at the observed value, never at 1.
    rows = _ram_rows(
        _sweep_of(
            [
                _row(concurrent_chroms=None, concurrent_chroms_used=8),
            ]
        )
    )
    assert [r.concurrent_chroms for r in rows] == [8]


def test_ram_rows_still_drop_rows_where_cc_was_never_observed():
    # Neither pinned nor parsed: UNOBSERVED. Coding it as 1 produced a pooled
    # per-contig estimate of 41 MB against a directly measured 89.67 (#158).
    rows = _ram_rows(
        _sweep_of([_row(concurrent_chroms=None, concurrent_chroms_used=None)])
    )
    assert rows == []


def test_ram_rows_prefer_the_pinned_value_over_the_realised_one():
    # The pinned value is what the plan asked for and what `point_id` hashes.
    # If the two disagree the plan is what the sweep is indexed by.
    rows = _ram_rows(
        _sweep_of([_row(concurrent_chroms=4, concurrent_chroms_used=8)])
    )
    assert [r.concurrent_chroms for r in rows] == [4]
```

Reuse the file's existing `_sweep_of` helper. If its row factory does not already accept `concurrent_chroms` / `concurrent_chroms_used`, extend it to thread both through to the `SweepPoint` and `ProbeRecord` it builds, defaulting to today's behaviour so existing tests are unaffected.

- [ ] **Step 2: Run it to make sure it fails**

```bash
pixi run pytest tests/bench/test_model.py -k ram_rows -v
```
Expected: FAIL — the fallback test yields `[]` because `_ram_rows` drops on `pt.concurrent_chroms is None`.

- [ ] **Step 3: Implement the fallback**

In `scripts/bench_svar2/model.py`, replace the drop block inside `_ram_rows` (the `if pt.concurrent_chroms is None: continue` and its comment) with:

```python
            # `cc` must be the value production ACTUALLY ran at, because
            # `plan_sharded` multiplies the per-contig bracket by it. Prefer the
            # PINNED request (it is what `point_id` hashes and what the plan is
            # indexed by), fall back to the value the child reported on its
            # `pipeline config` line, and drop only when NEITHER exists.
            #
            # Dropping, not defaulting: an unknown `cc` coded as 1 is what
            # produced a pooled per-contig estimate of 41 MB against a directly
            # measured 89.67 (issue #158).
            cc = pt.concurrent_chroms
            if cc is None:
                cc = r.concurrent_chroms_used
            if cc is None:
                continue
```

and pass `concurrent_chroms=cc` when constructing the `RamRow`.

- [ ] **Step 4: Run the tests**

```bash
pixi run pytest tests/bench/test_model.py -k ram_rows -v
```
Expected: PASS (all three new tests plus the pre-existing ones).

- [ ] **Step 5: Wire the missing sweeps into the fit**

In `scripts/bench_svar2/model.py`, extend `_SWEEP_NAMES` (line 762) to:

```python
_SWEEP_NAMES = (
    "scale",
    "contig",
    "holdout",
    "vlinear",
    "vlinear2",
    "concurrency",
    "vcf_ram",
    "pgen",
)
```

and change the `_ram_rows` call site (line 1200) to:

```python
    # `concurrency` and `vcf_ram` are the families that vary `concurrent_chroms`
    # at fixed (S, chunk_size), which is the ONLY way `per_contig_mb` becomes
    # identifiable. Omitting `concurrency` here is why `RamLaw::VCF` shipped
    # with that term at 0.0 even after the axis existed.
    ram_rows = _ram_rows(
        scale, contig, holdout, vlinear, vlinear2, sweeps["concurrency"],
        sweeps["vcf_ram"],
    )
```

`load_sweep` never raises on a missing file — it records the absence in `.excluded` — so naming families that a given results directory does not contain is safe.

- [ ] **Step 6: Write the failing test for the fit CLI**

Create `tests/bench/test_fit_ram.py`:

```python
import json
from pathlib import Path

from scripts.bench_svar2.fit_ram import gate_report
from scripts.bench_svar2.model import RamRow, fit_ram_law


def test_gate_report_flags_an_underpredicted_point():
    # A law is an UPPER BOUND: the gate is over-prediction at EVERY point,
    # evaluated the way plan_sharded evaluates it. One point below the line
    # fails the whole law, however good the fit looks elsewhere.
    rows = [
        RamRow(workers=1, pending=0, chunk_bytes=10_000_000, samples=4_000,
               peak_rss_mb=1_000.0, concurrent_chroms=1),
        RamRow(workers=1, pending=0, chunk_bytes=10_000_000, samples=4_000,
               peak_rss_mb=1_000_000.0, concurrent_chroms=1),
    ]
    law = fit_ram_law(rows[:1], margin=1.0)
    report = gate_report(law, rows)
    assert not report["passes"]
    assert report["n_under"] == 1


def test_gate_report_passes_a_true_envelope():
    rows = [
        RamRow(workers=1, pending=0, chunk_bytes=10_000_000, samples=4_000,
               peak_rss_mb=1_000.0, concurrent_chroms=1),
        RamRow(workers=1, pending=0, chunk_bytes=20_000_000, samples=32_000,
               peak_rss_mb=2_000.0, concurrent_chroms=4),
    ]
    law = fit_ram_law(rows, margin=1.25)
    report = gate_report(law, rows)
    assert report["passes"], report
    assert report["n_under"] == 0
    assert report["worst_ratio"] >= report["min_ratio"] >= 1.25 - 1e-9
```

- [ ] **Step 7: Run it to make sure it fails**

```bash
pixi run pytest tests/bench/test_fit_ram.py -v
```
Expected: FAIL (`ModuleNotFoundError: scripts.bench_svar2.fit_ram`).

- [ ] **Step 8: Write the fit CLI**

Create `scripts/bench_svar2/fit_ram.py`:

```python
"""Fit and gate one backend's `RamLaw` from a sweep's results.

The PGEN law was fitted ad hoc, which is why its refits are hard to reproduce
and why a contaminated dataset once reached a shipped constant. One entry point
means the fit that is quoted in a doc comment is the fit anyone can re-run.

    python -m scripts.bench_svar2.fit_ram \\
        --results $SCRATCH/out --plans $SCRATCH/plans \\
        --manifests $SCRATCH/corpora --backend vcf
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts.bench_svar2.model import (  # `_ram_rows` is private to the package,
    _load_manifests,  # not to this module -- fit_ram is model.py's CLI face.
    _ram_rows,
    RamRow,  # NB: RamRow lives in model.py; only RamLaw is in records.py.
    fit_ram_law,
    load_sweep,
)
from scripts.bench_svar2.records import RamLaw

# Which sweep families carry each backend's RAM rows. `concurrency` and
# `vcf_ram` are the families that vary `concurrent_chroms` at fixed
# (S, chunk_size) and so are the only ones that identify `per_contig_mb`.
BACKEND_SWEEPS: dict[str, tuple[str, ...]] = {
    "vcf": ("scale", "contig", "holdout", "vlinear", "vlinear2", "concurrency",
            "vcf_ram"),
    "pgen": ("pgen",),
}


def predict_mb(law: RamLaw, row: RamRow) -> float:
    """Exactly what `budget.rs:plan_sharded` computes for this row.

    Mirrored here rather than approximated: a gate that scores the law against
    a DIFFERENT equation than the consumer evaluates is how a cc-blind fit
    passed review in the first place (issue #158).
    """
    bracket = law.per_contig_mb + law.kappa * (
        (row.workers + row.pending) * row.chunk_bytes / 1e6
    )
    return law.base_mb + law.per_sample_mb * row.samples + row.concurrent_chroms * bracket


def gate_report(law: RamLaw, rows: Sequence[RamRow]) -> dict:
    """Over-prediction ratios for every row. `passes` is the shipping gate."""
    ratios = [predict_mb(law, r) / r.peak_rss_mb for r in rows]
    under = [r for r, ratio in zip(rows, ratios) if ratio < 1.0]
    return {
        "n": len(rows),
        "passes": not under,
        "n_under": len(under),
        "worst_ratio": max(ratios),
        "mean_ratio": sum(ratios) / len(ratios),
        "min_ratio": min(ratios),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True, help="dir of <name>.ndjson")
    p.add_argument("--plans", type=Path, required=True, help="dir of <name>.json")
    p.add_argument("--manifests", type=Path, required=True, help="dir of *.manifest.json")
    p.add_argument("--backend", choices=sorted(BACKEND_SWEEPS), required=True)
    p.add_argument(
        "--margin",
        type=float,
        default=1.25,
        help="safety factor: every point must be over-predicted by at least this",
    )
    a = p.parse_args()

    manifests = _load_manifests(a.manifests)
    sweeps = [
        load_sweep(name, a.results, a.plans, manifests)
        for name in BACKEND_SWEEPS[a.backend]
    ]
    for name, s in zip(BACKEND_SWEEPS[a.backend], sweeps):
        print(f"{name}: {len(s.records)} usable record(s)")
        for msg in s.excluded:
            print(f"  - {msg}")

    rows = _ram_rows(*sweeps)
    print(f"\nfittable rows (cc observed): {len(rows)}")
    if len(rows) < 2:
        raise SystemExit("ABORT: need >= 2 rows with an observed concurrent_chroms")

    law = fit_ram_law(rows, margin=a.margin)
    report = gate_report(law, rows)

    print(f"\nRamLaw::{a.backend.upper()} (margin {a.margin}):")
    print(f"    base_mb:       {law.base_mb!r},")
    print(f"    per_sample_mb: {law.per_sample_mb!r},")
    print(f"    per_contig_mb: {law.per_contig_mb!r},")
    print(f"    kappa:         {law.kappa!r},")
    print(f"\ngate: n={report['n']} passes={report['passes']} "
          f"under={report['n_under']} worst={report['worst_ratio']:.4f}x "
          f"mean={report['mean_ratio']:.4f}x min={report['min_ratio']:.4f}x")
    print(f"(descriptive only, NOT the criterion) r2={law.r2:.4f}")
    if not report["passes"]:
        raise SystemExit(
            f"ABORT: law under-predicts {report['n_under']} point(s). A law is a "
            "BOUND -- do not ship this."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Run the tests**

```bash
pixi run pytest tests/bench/test_fit_ram.py tests/bench/test_model.py -q
```
Expected: PASS.

- [ ] **Step 10: Verify the CLI reproduces the shipped PGEN law**

```bash
pixi run python -m scripts.bench_svar2.fit_ram \
  --results docs/superpowers/plans/results/2026-08-08-pgen-ram-law-crossed-data \
  --plans docs/superpowers/plans/results/2026-08-08-pgen-ram-law-crossed-data \
  --manifests docs/superpowers/plans/results/2026-08-08-pgen-ram-law-crossed-data/manifests \
  --backend pgen --margin 1.25
```
Expected: `base_mb` ≈ 2696.79, `per_sample_mb` ≈ 0.0157515, `per_contig_mb` ≈ 209.870, `kappa` ≈ 2.38477, gate `passes=True` with `worst≈2.4189x`, `min≈1.2500x`. These are `RamLaw::PGEN`'s shipped values — a mismatch means the CLI is not fitting the same quantity the shipped law came from, and must be resolved before Task 6 trusts it. If the committed directory's plan/results filenames do not match `<name>.ndjson` / `<name>.json` for family `pgen`, symlink or copy them into a scratch dir with the expected names rather than loosening `load_sweep`.

- [ ] **Step 11: Commit**

```bash
git add scripts/bench_svar2/model.py scripts/bench_svar2/fit_ram.py tests/bench/test_model.py tests/bench/test_fit_ram.py
git commit -m "feat(bench): fit RAM laws from a CLI and stop dropping observed-cc rows

Two gaps behind #158's remaining half.

\`_ram_rows\` never saw the \`concurrency\` family -- the one existing VCF axis
that varies concurrent_chroms at fixed (S, chunk_size) -- which is why
RamLaw::VCF still carries per_contig_mb = 0.0 after that axis landed. It also
dropped every point that let the planner choose, even when the child reported
the realised value; it now prefers the pinned request and falls back to the
parsed one, dropping only when neither exists.

Fitting had no entry point at all: the PGEN law was fitted ad hoc, which is
how a contaminated dataset once reached a shipped constant. \`fit_ram\` fits
and gates one backend reproducibly, scoring the law with plan_sharded's own
equation so the fit and the consumer cannot drift apart again.

Refs #158"
```

---

### Task 5: Add the crossed VCF plan families

**Files:**
- Modify: `scripts/bench_svar2/plans/build_plans.py` (constants near line 37, `build()` near line 460, the returned dict near line 515)
- Test: `tests/bench/test_build_plans.py`

**Interfaces:**
- Consumes: `vcf_corpus.corpus_stem` (Task 3).
- Produces: a `"vcf_ram"` family in `build()`'s returned dict (52 points), and the module constants `VCF_CROSSED`, `VCF_NCHUNKS`, `VCF_BIGCHUNK`, `VCF_CROSSED_CC`, `VCF_NCHUNKS_CC`, `VCF_CONTIGS`. Task 7's sbatch generates corpora from these constants; Task 8 fits the resulting rows.

- [ ] **Step 1: Write the failing tests**

Add to `tests/bench/test_build_plans.py`:

```python
def test_vcf_ram_family_has_the_planned_point_count():
    plans = build(Path("/corpora"), threads=48)
    # 36 crossed (3 widths x 3 chunk sizes x 4 cc) + 12 n_chunks
    # (2 widths x 3 V rungs x 2 cc) + 4 big-chunk (2 chunk sizes x 2 cc).
    assert len(plans["vcf_ram"]) == 52


def test_vcf_ram_points_never_exceed_one_chunk_per_contig():
    # `model._resident_chunk_size` clamps chunk_size by TOTAL V, not per-contig
    # V. On a 22-contig corpus a point above V/22 would be fitted against a
    # chunk up to 22x larger than anything ever resident, because BitGrid3's
    # calloc pages are not resident until written.
    plans = build(Path("/corpora"), threads=48)
    for pt in plans["vcf_ram"]:
        variants = int(Path(pt.corpus).name.split("_v")[1].split(".")[0])
        assert pt.chunk_size <= variants // len(VCF_CONTIGS), (
            f"{pt.corpus} chunk_size={pt.chunk_size} exceeds "
            f"{variants // len(VCF_CONTIGS)} variants per contig"
        )


def test_vcf_crossed_chunk_sizes_land_on_the_same_megabytes_at_every_width():
    # chunk_MB = (S*ploidy/8) * chunk_size and V = CELLS_BUDGET / S, so
    # {V/88, V/44, V/22} is ~4/8/16 MB at EVERY cohort width. That uniformity
    # is the point of expressing them as fractions of V rather than literals.
    for s, chunk_sizes in VCF_CROSSED.items():
        v = CELLS_BUDGET // s
        mbs = [(s * 2 // 8) * cs / 1e6 for cs in chunk_sizes]
        assert mbs == pytest.approx([3.98, 7.95, 15.9], rel=0.02), (s, mbs)


def test_vcf_bigchunk_reaches_a_hundred_megabyte_chunk():
    # The whole reason this corpus exists: without it kappa is measured only to
    # 15.9 MB and extrapolated ~16x to production's 256 MiB chunks.
    s, v = VCF_BIGCHUNK["samples"], VCF_BIGCHUNK["variants"]
    biggest = max(VCF_BIGCHUNK["chunk_sizes"])
    assert (s * 2 // 8) * biggest / 1e6 == pytest.approx(100.0, rel=0.01)
    assert biggest <= v // len(VCF_CONTIGS)


def test_every_vcf_ram_point_id_is_unique():
    plans = build(Path("/corpora"), threads=48)
    ids = [pt.point_id for pt in plans["vcf_ram"]]
    assert len(ids) == len(set(ids))
```

Add `VCF_CONTIGS`, `VCF_CROSSED`, `VCF_BIGCHUNK`, `CELLS_BUDGET` and `build` to the file's imports, plus `pytest` if absent.

- [ ] **Step 2: Run them to make sure they fail**

```bash
pixi run pytest tests/bench/test_build_plans.py -k vcf_ram -v
```
Expected: FAIL with `ImportError` on `VCF_CONTIGS`.

- [ ] **Step 3: Add the axis constants**

In `scripts/bench_svar2/plans/build_plans.py`, after the existing `CONCURRENCY_*` constants (around line 38):

```python
# --- VCF RAM-law crossed design (issue #158) ---------------------------------
# The PGEN crossed grid does not port directly: VCF corpora are TEXT, so the
# cell budget binds. Per-variant chunk bytes are `samples*ploidy/8` and the
# >=1-chunk-per-contig invariant caps chunk_size at V/n_contigs, so
#
#     max chunk_MB = cells / (4 * n_contigs * 1e6)
#
# which at CELLS_BUDGET over 22 contigs is 15.9 MB -- INDEPENDENT of cohort
# width. The `cc` lever arm wants many contigs and the `chunk_MB` lever arm
# wants few, and at a fixed cell budget they are in direct conflict. Hence the
# constant-cells grid below for cc, plus ONE oversized corpus (VCF_BIGCHUNK)
# whose only job is to carry kappa out to ~100 MB, cutting the extrapolation to
# production's 256 MiB chunks from ~16x to ~2.7x.
VCF_CONTIGS = tuple(f"chr{i}" for i in range(1, 23))

# Cohort widths for the crossed grid. V is derived (CELLS_BUDGET // S) so every
# corpus costs the same to generate.
VCF_CROSSED_SAMPLES = (4_000, 32_000, 128_000)

# cc=16 exceeds any production clamp and is reachable only through the
# bench-only GENORAY_CONCURRENT_CHROMS override. It is here for LEVER ARM on
# the per-contig term and sits OUTSIDE the production domain -- the same role
# and the same caveat as the PGEN law's cc=16 rows.
VCF_CROSSED_CC = (1, 4, 8, 16)

# {V/88, V/44, V/22} -> ~4/8/16 MB at every width, the largest sitting exactly
# on the per-contig cap. Fractions of V, not literals, so the grid stays
# uniform if CELLS_BUDGET ever moves.
VCF_CROSSED: dict[int, tuple[int, ...]] = {
    s: tuple((CELLS_BUDGET // s) // d for d in (88, 44, 22))
    for s in VCF_CROSSED_SAMPLES
}

# n_chunks at CONSTANT chunk_bytes -- the orthogonal lever that decides whether
# a per-chunk term is real or merely a reparameterisation of kappa. chunk_size
# is pinned at what the SMALLEST V rung permits, so chunk_bytes is identical
# across a row's rungs and only V (hence the chunk count) moves.
VCF_NCHUNKS_SAMPLES = (4_000, 32_000)
VCF_NCHUNKS_CC = (1, 8)
VCF_NCHUNKS: dict[int, tuple[int, ...]] = {
    s: tuple((CELLS_BUDGET // s) * m // 2 for m in (1, 2, 4))
    for s in VCF_NCHUNKS_SAMPLES
}

# One oversized corpus, ~7.9x CELLS_BUDGET, purely for kappa's lever arm.
# 1.1e10 cells at S=32,000 gives V=343,750 and 15,625 variants per contig, so
# chunk_size 12,500 is 100 MB and still clears the per-contig cap.
VCF_BIGCHUNK = {
    "samples": 32_000,
    "variants": 343_750,
    "chunk_sizes": (3_125, 12_500),
    "cc": (1, 8),
}
```

- [ ] **Step 4: Build the family**

In `build()`, add `vcf_ram` to the tuple of empty lists at the top of the function, then insert before the return (after the PGEN axes):

```python
    # --- VCF RAM-law axes (issue #158) --------------------------------------
    # `w=1` throughout: reader_workers is a separate, already-fitted axis and
    # varying it here would confound the per-contig term these points exist to
    # identify.
    for s_c, chunk_sizes in VCF_CROSSED.items():
        v_c = CELLS_BUDGET // s_c
        corpus = corpus_dir / f"{vcf_stem(s_c, v_c)}.manifest.json"
        for cs in chunk_sizes:
            for cc in VCF_CROSSED_CC:
                vcf_ram.append(_point(corpus, 1, cs, threads, concurrent=cc))

    for s_n, variants in VCF_NCHUNKS.items():
        pinned = min(variants) // len(VCF_CONTIGS)
        for v_n in variants:
            corpus = corpus_dir / f"{vcf_stem(s_n, v_n)}.manifest.json"
            for cc in VCF_NCHUNKS_CC:
                vcf_ram.append(_point(corpus, 1, pinned, threads, concurrent=cc))

    s_b, v_b = VCF_BIGCHUNK["samples"], VCF_BIGCHUNK["variants"]
    corpus_b = corpus_dir / f"{vcf_stem(s_b, v_b)}.manifest.json"
    for cs in VCF_BIGCHUNK["chunk_sizes"]:
        for cc in VCF_BIGCHUNK["cc"]:
            vcf_ram.append(_point(corpus_b, 1, cs, threads, concurrent=cc))

    # Order-preserving dedupe on the full identity, the same way the PGEN axes
    # are deduped: the n_chunks ladder's middle rung IS the crossed grid's
    # corpus, so the two axes can legitimately request the same configuration.
    seen_vcf: set[str] = set()
    vcf_ram = [p for p in vcf_ram if not (p.point_id in seen_vcf or seen_vcf.add(p.point_id))]
```

Add `"vcf_ram": vcf_ram,` to the returned dict, and near the top of the file:

```python
from scripts.bench_svar2.vcf_corpus import VcfCorpusSpec, corpus_stem as _vcf_corpus_stem


def vcf_stem(samples: int, variants: int) -> str:
    """Corpus stem for a VCF RAM-law point, via the generator's own namer so
    the plan and the corpus cannot drift apart."""
    return _vcf_corpus_stem(
        VcfCorpusSpec(samples=samples, variants=variants, contigs=VCF_CONTIGS, seed=42)
    )
```

- [ ] **Step 5: Add the per-contig assertion at plan-build time**

Immediately after the dedupe, still inside `build()`:

```python
    # A point above V/n_contigs would be fitted against a chunk that is never
    # resident (see `model._resident_chunk_size`, which clamps by TOTAL V).
    # Assert at BUILD time: a plan that cannot be fitted correctly must not
    # reach a node and consume hours first.
    for pt in vcf_ram:
        v_pt = int(Path(pt.corpus).name.split("_v")[1].split(".")[0])
        per_contig = v_pt // len(VCF_CONTIGS)
        if pt.chunk_size > per_contig:
            raise ValueError(
                f"{Path(pt.corpus).name}: chunk_size={pt.chunk_size:,} exceeds "
                f"{per_contig:,} variants per contig, so the fitted chunk_bytes "
                "would price memory that is never resident"
            )
```

- [ ] **Step 6: Run the tests**

```bash
pixi run pytest tests/bench/test_build_plans.py -v
```
Expected: PASS, including the pre-existing `test_every_point_id_is_unique`.

- [ ] **Step 7: Print the plan to eyeball it**

```bash
pixi run python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir /tmp/does-not-exist --out-dir "$CLAUDE_JOB_DIR/tmp/plans" --threads 48
```
Expected: a line `.../vcf_ram.json: 52 points` alongside the existing families.

- [ ] **Step 8: Commit**

```bash
git add scripts/bench_svar2/plans/build_plans.py tests/bench/test_build_plans.py
git commit -m "feat(bench): add the crossed VCF RAM-law axes

VCF corpora are text, so the cell budget binds where PGEN's did not: max
chunk_MB is cells/(4*n_contigs*1e6) = 15.9 MB at CELLS_BUDGET over 22 contigs,
independent of cohort width. The cc lever arm wants many contigs and the
chunk_MB lever arm wants few, so this pairs a constant-cells crossed grid with
one oversized corpus that carries kappa out to ~100 MB.

Also asserts chunk_size <= V/n_contigs at BUILD time: model._resident_chunk_size
clamps by total V, so a point above the per-contig cap would price a chunk that
is never resident -- and it must fail before consuming a node, not after.

Refs #158"
```

---

## Wave 3

### Task 6: Decide the functional form offline, before booking a node

The PGEN crossed data is committed, so the one interaction it pointed at can be tested at zero cluster cost — and must be, because the VCF sweep should be fitted against the final form once rather than twice.

**Files:**
- Modify: `scripts/bench_svar2/model.py` (`fit_ram_law`: optional interaction regressor), `scripts/bench_svar2/records.py` (`RamLaw`: optional field)
- Create: `docs/superpowers/plans/results/2026-08-11-ram-law-form-check.md`
- Test: `tests/bench/test_model.py`

**Interfaces:**
- Consumes: `fit_ram.gate_report`, `fit_ram.predict_mb` (Task 4).
- Produces: a GO/NO-GO decision on the five-coefficient form, recorded in the results doc. If GO, `RamLaw.per_contig_per_sample_mb: float = 0.0` exists in both the Python dataclass and (Task 8) `src/budget.rs`.

- [ ] **Step 1: Write the failing test for the optional regressor**

Add to `tests/bench/test_model.py`:

```python
def test_fit_ram_law_without_interaction_leaves_the_term_at_zero():
    rows = [
        RamRow(workers=1, pending=0, chunk_bytes=10_000_000, samples=4_000,
               peak_rss_mb=1_000.0, concurrent_chroms=1),
        RamRow(workers=1, pending=0, chunk_bytes=10_000_000, samples=32_000,
               peak_rss_mb=3_000.0, concurrent_chroms=4),
    ]
    law = fit_ram_law(rows, margin=1.0)
    assert law.per_contig_per_sample_mb == 0.0


def test_fit_ram_law_with_interaction_is_never_looser_than_without():
    # The interaction form NESTS the current one (set the extra coefficient to
    # zero), so its optimal worst-case ratio can only be <= the simpler form's.
    # If it ever comes out larger the LP is misconstructed, not the data.
    rows = [
        RamRow(workers=1, pending=0, chunk_bytes=10_000_000, samples=4_000,
               peak_rss_mb=1_000.0, concurrent_chroms=1),
        RamRow(workers=1, pending=0, chunk_bytes=10_000_000, samples=32_000,
               peak_rss_mb=3_000.0, concurrent_chroms=4),
        RamRow(workers=1, pending=0, chunk_bytes=20_000_000, samples=128_000,
               peak_rss_mb=9_000.0, concurrent_chroms=8),
    ]
    plain = fit_ram_law(rows, margin=1.0)
    inter = fit_ram_law(rows, margin=1.0, interaction=True)
    assert inter.worst_ratio <= plain.worst_ratio + 1e-9
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
pixi run pytest tests/bench/test_model.py -k interaction -v
```
Expected: FAIL — `RamLaw` has no `per_contig_per_sample_mb` and `fit_ram_law` has no `interaction` parameter.

- [ ] **Step 3: Add the optional field and regressor**

In `scripts/bench_svar2/records.py`, add to `RamLaw` after `per_contig_mb`:

```python
    # Per-contig cost that scales with cohort width. The 2026-08-08 PGEN
    # crossed sweep measured the per-contig slope at 83.7 MB (S=4,000), 263 MB
    # (S=32,000) and 301 MB (S=128,000) -- a bracket that grows with S, which a
    # single additive `per_contig_mb` cannot express. 0.0 means "not fitted",
    # which is the form the law had before this field existed.
    per_contig_per_sample_mb: float = 0.0
```

In `scripts/bench_svar2/model.py`, change the signature to:

```python
def fit_ram_law(
    rows: Sequence[RamRow], margin: float = 1.25, interaction: bool = False
) -> RamLaw:
```

In the design-matrix block, immediately after the `per_contig` column is appended:

```python
    if contig_identifiable:
        names.append("per_contig")
        cols.append(cc)
        # Optional per-contig term that scales with cohort width. The
        # 2026-08-08 PGEN crossed sweep measured the per-contig slope at
        # 83.7 MB (S=4,000), 263 MB (S=32,000) and 301 MB (S=128,000): the
        # bracket GROWS with S, which a single additive `per_contig` cannot
        # express. This form NESTS the simpler one (the coefficient can go to
        # zero), so its optimal worst-case ratio is never larger.
        #
        # Off by default. Whether it ships is a recorded decision, not a
        # modelling preference -- see
        # docs/superpowers/plans/results/2026-08-11-ram-law-form-check.md.
        # A term multiplying S is extrapolated ~3.9x to reach S=500,000, which
        # is exactly the kind of reach that made the `n_chunks` term unsafe.
        if interaction and cohort_identifiable:
            names.append("per_contig_per_sample")
            cols.append(cc * samples)
```

and extend the returned `RamLaw` with:

```python
        per_contig_per_sample_mb=coef.get("per_contig_per_sample", 0.0),
```

The LP itself needs no change: it already builds `a` from `cols`, sizes the
variable vector from `a.shape`, and bounds every coefficient non-negative.

- [ ] **Step 4: Run the tests**

```bash
pixi run pytest tests/bench/test_model.py -k "interaction or ram_law" -v
```
Expected: PASS.

- [ ] **Step 5: Fit both forms against the committed PGEN data**

```bash
D=docs/superpowers/plans/results/2026-08-08-pgen-ram-law-crossed-data
pixi run python -m scripts.bench_svar2.fit_ram \
  --results $D --plans $D --manifests $D/manifests --backend pgen --margin 1.25
```

Then re-run with the interaction form. If `fit_ram.py` does not yet expose it, add a `--interaction` flag that threads straight through to `fit_ram_law(rows, margin=..., interaction=...)` — one argparse line and one keyword, no other logic.

Record for BOTH forms: `n`, every coefficient, `passes`, `worst_ratio`, `mean_ratio`, `min_ratio`, and the largest ratio between a coefficient's applied value and its measured domain.

- [ ] **Step 6: Apply the pre-registered decision rule**

Adopt the interaction form **only if both** hold:

1. worst-case `t` improves by **≥20%** against the current form on the same rows; **and**
2. **no** coefficient must be extrapolated more than ~2× beyond its measured domain. The measured domain here is S ∈ {4,000, 32,000, 128,000} against a production target of 500,000 — so a term multiplying S is already ~3.9× extrapolated and clause 2 is a real constraint, not a formality.

This rule is fixed in advance deliberately. The `n_chunks` term reached R² 1.0000 in-sample and was still correctly refused, because applying it 300× beyond its measured range took the S=500,000 projection from 65.3 GiB to 160.7 GiB.

- [ ] **Step 7: Write the results doc**

Create `docs/superpowers/plans/results/2026-08-11-ram-law-form-check.md` containing: the two fits in full; the decision rule as stated above; the verdict (GO or NO-GO) with the two clause outcomes evaluated explicitly; and, if NO-GO, the sentence that goes into the follow-up issue. State plainly that this used already-committed data and cost no measurement time.

- [ ] **Step 8: If NO-GO, revert the regressor to dormant**

If the rule does not fire, leave `interaction` in the fitter and `per_contig_per_sample_mb` at `0.0` — a tested, unused option costs nothing and documents what was tried — but do **not** add the field to `src/budget.rs`. Task 8 then ships the four-coefficient form.

If it does fire, Task 8 ships five coefficients and must **also** refit `RamLaw::PGEN` from this same committed data, since `plan_sharded`'s bracket changes for both backends.

- [ ] **Step 9: Commit**

```bash
git add scripts/bench_svar2/model.py scripts/bench_svar2/records.py tests/bench/test_model.py docs/superpowers/plans/results/2026-08-11-ram-law-form-check.md
git commit -m "test(bench): decide the RAM-law functional form on committed data

The 2026-08-08 PGEN crossed sweep measured a per-contig slope that GROWS with
cohort width (83.7 -> 263 -> 301 MB), which a single additive per_contig_mb
cannot express. Tests that interaction offline against the already-committed
46 rows, under a decision rule fixed BEFORE the numbers were seen: adopt only
on a >=20% worst-case improvement AND no coefficient extrapolated >~2x beyond
its measured domain.

The second clause is the n_chunks lesson -- that term reached R^2 1.0000
in-sample and was still correctly refused.

Verdict and both fits in
docs/superpowers/plans/results/2026-08-11-ram-law-form-check.md.

Refs #158"
```

---

## Wave 4

### Task 7: Measure the VCF crossed sweep

**Files:**
- Modify: `scripts/bench_svar2/sweep_scale.sbatch` (corpus generation + the `vcf_ram` sweep and its guard)
- Create: `docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed-data/` (committed after the run)

**Interfaces:**
- Consumes: Tasks 2, 3, 5, 6.
- Produces: `vcf_ram.ndjson` plus the corpora's manifests, both committed. Task 8 fits them.

- [ ] **Step 1: Add corpus generation for the RAM-law shapes**

In `scripts/bench_svar2/sweep_scale.sbatch`, after the existing corpus loops, add:

```bash
# --- VCF RAM-law corpora (issue #158) ----------------------------------------
echo "=== disk headroom before RAM-law corpora ==="
df -h "/local/$USER"
time $PX python -c "
from pathlib import Path
from scripts.bench_svar2.vcf_corpus import VcfCorpusSpec, generate
from scripts.bench_svar2.plans.build_plans import (
    CELLS_BUDGET, VCF_BIGCHUNK, VCF_CONTIGS, VCF_CROSSED, VCF_NCHUNKS,
)
out = Path('$SCRATCH/corpora')
shapes = {(s, CELLS_BUDGET // s) for s in VCF_CROSSED}
shapes |= {(s, v) for s, vs in VCF_NCHUNKS.items() for v in vs}
shapes.add((VCF_BIGCHUNK['samples'], VCF_BIGCHUNK['variants']))
for s, v in sorted(shapes):
    m = generate(VcfCorpusSpec(s, v, VCF_CONTIGS, seed=42), out)
    print(m.path, m.compressed_bytes, m.generator_cli_version, flush=True)
"
echo "=== disk headroom after RAM-law corpora ==="
df -h "/local/$USER"
du -sh "$SCRATCH/corpora"
```

- [ ] **Step 2: Add the sweep and the row-count guard**

After the plan-build step, add:

```bash
$PX python -m scripts.bench_svar2.sweep \
  --plan "$SCRATCH/plans/vcf_ram.json" \
  --results "$SCRATCH/out/vcf_ram.ndjson" \
  --outdir "$SCRATCH/out/vcf_ram"

# Row count vs plan count. The 2026-08-07 contamination was invisible in every
# summary its run printed -- exit 0, all points ok, matching row count -- because
# stale rows counted as recorded. What did NOT match was how many points the run
# actually MEASURED. Fail loudly here, not in a fit weeks later.
$PX python - "$SCRATCH/plans/vcf_ram.json" "$SCRATCH/out/vcf_ram.ndjson" <<'PYEOF'
import json, sys
plan, results = sys.argv[1], sys.argv[2]
n_plan = len(json.load(open(plan)))
rows = [json.loads(line) for line in open(results) if line.strip()]
ids = {r["point_id"] for r in rows}
print(f"plan points: {n_plan}   result rows: {len(rows)}   unique ids: {len(ids)}")
bad = [r["point_id"] for r in rows if not r.get("ok") and r.get("oom_at_rss_mb") is None]
if bad:
    print(f"FAILED points (not OOM deliverables): {bad}")
if len(ids) != n_plan:
    sys.exit(f"ABORT: {len(ids)} distinct points in results, plan has {n_plan}")
nodes = {r.get("node") for r in rows}
if len(nodes) > 1:
    sys.exit(f"ABORT: rows span multiple nodes {nodes} -- not comparable")
missing = [r["point_id"] for r in rows if r.get("concurrent_chroms_used") is None]
if missing:
    print(f"WARNING: {len(missing)} row(s) never reported a realised cc")
print(f"OK: every planned point measured, single node {nodes}")
PYEOF
```

- [ ] **Step 3: Submit, pinned to one node**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
sbatch --nodelist=carter-cn-04 scripts/bench_svar2/sweep_scale.sbatch
```

Do not run any of this on the login node. Check progress with `squeue -u $USER` and the job's `%x_%j.log`; do not poll in a tight loop.

- [ ] **Step 4: Verify the run before trusting a single number**

Confirm from the log: `COMPLETED 0:0`; the guard printed `OK: every planned point measured, single node {...}`; a single node name; every corpus reporting the same `generator_cli_version` ≥ 0.5.0; and the `maturin develop --release` line naming a freshly-built `.so`. A run that fails any of these is data about the harness, not about memory.

- [ ] **Step 5: Commit the data**

```bash
D=docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed-data
mkdir -p "$D/manifests"
cp /local/$USER/scale-sweep/out/vcf_ram.ndjson "$D/"
cp /local/$USER/scale-sweep/plans/vcf_ram.json "$D/"
cp /local/$USER/scale-sweep/corpora/vcfx_*.manifest.json "$D/manifests/"
git add "$D"
git commit -m "test(bench): commit the VCF RAM-law crossed sweep data

Job <JOBID>, <node>, <N>/<N> points measured on one node. vcfixture bulk
v<VERSION>. Refs #158"
```

Fill the job id, node, point count and CLI version from the run — they are the provenance the fit's doc comment cites.

---

## Wave 5

### Task 8: Fit and ship `RamLaw::VCF`

**Files:**
- Modify: `src/budget.rs:143-155` (`RamLaw::VCF`), `:789-803` (add the VCF usable-law test)
- Modify: `skills/genoray-api/SKILL.md:381-387`, `:466-468`
- Create: `docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed.md`

**Interfaces:**
- Consumes: Task 7's committed data, Task 6's form decision.
- Produces: shipped coefficients. Task 9 closes the issue against them.

- [ ] **Step 1: Fit and gate**

```bash
D=docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed-data
pixi run python -m scripts.bench_svar2.fit_ram \
  --results $D --plans $D --manifests $D/manifests --backend vcf --margin 1.25
```
Expected: `passes=True`. If it exits non-zero the law under-predicts somewhere and must not be shipped — re-check that the rows carry the `cc` the sweep pinned before touching the margin.

- [ ] **Step 2: Record the margin sensitivity**

Re-run at `--margin 1.00`, `1.25`, `1.50` and `2.00`, keeping each `worst_ratio`. The shipped margin is a **chosen** safety factor and the table is what makes that choice reviewable rather than inherited.

- [ ] **Step 3: Ship the coefficients**

Replace `RamLaw::VCF` in `src/budget.rs` with the fitted values, and rewrite its doc comment to state: the fit date, job id, node, `n`, and results-doc path; that it is an **envelope, not least squares**; the gate outcome (over-predicts all `n` points, worst/mean/min); that the margin is a chosen factor with the sensitivity table from Step 2; the validity domain (S range, `chunk_MB` range, `cc` range, `reader_workers`, contig count, `multiallelic_rate`, no FORMAT fields); that `cc = 16` rows sit outside the production domain; and how far `per_sample_mb` is extrapolated to reach S=500,000. Delete the stale sentence claiming `per_contig_mb` is `0.0` because the sweep never varied `concurrent_chroms`, and the "NOT comparable with `RamLaw::VCF`" caveat in `RamLaw::PGEN`'s comment — both corpora now come from `vcfixture bulk`.

- [ ] **Step 4: Add the usable-law guard**

In `src/budget.rs`'s test module, alongside `ram_law_pgen_is_a_usable_law`:

```rust
#[test]
fn ram_law_vcf_is_a_usable_law() {
    // Guards against a placeholder shipping: a zero kappa would make the
    // memory bound vacuous and silently restore unbounded planning.
    assert!(RamLaw::VCF.kappa > 0.0, "kappa must be positive");
    assert!(RamLaw::VCF.base_mb > 0.0, "baseline must be positive");
    assert!(RamLaw::VCF.per_sample_mb >= 0.0);
    // Measured by the 2026-08-11 crossed sweep, which varied concurrent_chroms
    // at fixed (S, chunk_size) for the first time on this backend. A refit that
    // drops it back to zero has silently discarded the per-contig staging cost
    // and restored the under-prediction #158 was opened about.
    assert!(
        RamLaw::VCF.per_contig_mb > 0.0,
        "VCF's per-contig term is measured, not optional"
    );
}
```

- [ ] **Step 5: Run the Rust tests**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion
```
Expected: 474 passing (473 plus the new guard), 0 failing. Any test carrying a hand-derived `RamLaw::VCF` number must be **re-derived**, never hand-patched to match — adjusting an expected value instead of its model is how a test silently stops asserting what it names.

- [ ] **Step 6: Rebuild and run the Python tests**

```bash
pixi run maturin develop --release
pixi run test
```
Expected: no regressions against the pre-change baseline.

- [ ] **Step 7: Update the public floor documentation**

In `skills/genoray-api/SKILL.md`, update both `max_mem` floor passages (near lines 381-387 and 466-468) to the new VCF baseline and per-contig term. The repo's rule is explicit that public-facing text tracks these numbers.

- [ ] **Step 8: Write the results doc**

Create `docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed.md` scoring the sweep against the issue's pre-registered criteria: the gate outcome; whether `per_contig_mb` is additive across the three cohort widths or varies beyond its CI; whether the `n_chunks` term survives at constant `chunk_bytes`; residual σ against the 63 MB floor; and the margin sensitivity table. Say plainly what the data cannot identify.

- [ ] **Step 9: Commit**

```bash
git add src/budget.rs skills/genoray-api/SKILL.md docs/superpowers/plans/results/2026-08-11-vcf-ram-law-crossed.md
git commit -m "perf(svar2): fit RamLaw::VCF as an envelope with a measured per-contig term

RamLaw::VCF was an OLS mean fit, cc-blind while plan_sharded multiplies the
per-contig bracket by cc, and carried per_contig_mb = 0.0 -- leaving the
~128 MiB ChunkAssembler staging allocation unpriced per live contig, which is
roughly 1 GB unaccounted at cc=8 against a 932 MB baseline. That is an
under-prediction, the OOM direction.

Refitted as the same envelope LP RamLaw::PGEN uses, against the 2026-08-11
crossed sweep. Gate: over-predicts all <N> measured points as plan_sharded
evaluates them, worst <W>x, mean <M>x, min <MIN>x at the chosen 1.25 margin.

Both laws now come from vcfixture bulk corpora, so the caveat that they are not
comparable coefficient-by-coefficient is retired.

Closes #158"
```

Fill `<N>`, `<W>`, `<M>`, `<MIN>` from Step 1's output before committing.

---

### Task 9: Close the issue and prepare the release

- [ ] **Step 1: Post the closing comment on #158**

Summarise, in the style of the issue's existing comments: what shipped, the gate numbers, what is deliberately NOT shipped, and what moved to a follow-up.

- [ ] **Step 2: Open the follow-up issue**

Title it for the mechanism question, not the symptom: the law bounds RSS without describing it, residual σ stays ~9× the 63 MB reproducibility floor, and coefficients vary with `S` and `cc` on both backends. Link Task 6's form-check results doc and state that an upper-bound law is safe regardless — this is about the form, not about safety.

- [ ] **Step 3: Verify the whole tree**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion
cargo check --no-default-features
pixi run maturin develop --release
pixi run test
```
Expected: all green. `cargo check --no-default-features` covers the query-core build that CI once had no coverage for.

- [ ] **Step 4: Open the PR**

```bash
gh pr create --draft --title "fix(svar2): fit RamLaw::VCF as an envelope (#158)" --body "..."
```
Body: the problem, the measurement design, the gate result, the form decision from Task 6, and explicit links to both results docs.

- [ ] **Step 5: Hand off the release**

Do not tag or bump anything by hand. Confirm the merged history carries Conventional Commit subjects that `commitizen` can turn into the changelog section, and say so in the handoff.

---

## Notes for the implementer

- **Verify before claiming.** Every "Expected:" line above is a command whose output must be read. `pixi run test` passing against a stale `.so` is the single most common false green in this repo.
- **Do not background long builds.** Run `cargo test` and `maturin develop` in the foreground; a subagent that backgrounds them and returns early reports success it never observed.
- **A shared `CARGO_TARGET_DIR` can fake an A/B.** If you ever compare two builds, demand a "Compiling …" line naming the scratch path and a changed `.so` hash.
- **`test-rust <arg>` filters by test NAME, not file.** A non-matching argument vacuously passes zero tests. Use `--test <file>`.
