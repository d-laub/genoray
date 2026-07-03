# Atomic sibling-tmp writes for `.svar` and `.gvi`

**Date:** 2026-06-30
**Type:** robustness fix (patch)
**Semver:** patch (`fix:` commits) — `2.14.0` → `2.14.1`. No public API name or
semantic change.

---

## Problem

`genoray` writes its outputs **in place**, with no crash safety:

1. **`.svar` directory writes** — `SparseVar.from_vcf` and `SparseVar.from_pgen`
   (the `genoray write` CLI) `out.mkdir(...)` up front, write `metadata.json`,
   then `_concat_data(out, ...)` concatenates the final arrays directly into
   `out`. `SparseVar.write_view` (the `genoray view` CLI) `rmtree`s + `mkdir`s
   `output` and then writes in place. If the process dies mid-write, `out` is
   left partially written / corrupt, and under `overwrite=True` an existing
   output directory has already been destroyed.

2. **`.gvi` single-file index writes** — `VCF._write_gvi_index`
   (`_vcf.py:1215`) does `...collect().write_ipc(self._index_path(), ...)` and
   PGEN's `_write_index` (`_pgen.py:1259`) does `...sink_ipc(index_path)`, both
   writing the `.gvi` file in place. A crash mid-write leaves a truncated `.gvi`
   that later loads as a corrupt or partial index.

3. The intermediate per-contig chunk staging in `from_vcf`/`from_pgen` uses
   `TemporaryDirectory()` with no `dir=`, so it lands in `$TMPDIR`. When
   `$TMPDIR` is on a different filesystem than the output, `_concat_data` copies
   bytes across devices, and `$TMPDIR` must have room for all chunks.

There is currently no atomic-rename path anywhere: a grep for
`rename`/`replace`/`tempfile`/`atomic` in `genoray/*.py` (other than the
chunk `TemporaryDirectory`) returns nothing.

## Goals

- Every durable output (`.svar` directory, `.gvi` file) is written to a
  **sibling** temp path and then **atomically renamed** into place by default.
- A crash mid-write never leaves a partial/corrupt output; an existing output
  being overwritten is preserved until the new one is complete.
- Output bytes, schema, dtypes, and coordinate/missing-value conventions are
  **byte-identical** to today.
- Sibling staging (same parent dir ⇒ same filesystem) makes `os.replace` truly
  atomic and removes cross-device copies during concat.
- No new public kwargs — the behavior is unconditional ("by default"), keeping
  this a patch.

## Non-goals

- New user-facing options to configure the temp location.
- Changing the fail-fast ordering/guards from PR #75.
- Making the `.svar`-internal `index.arrow` writes individually atomic — they
  live inside the staging directory and inherit directory-level atomicity.
- Cross-process write locking. (`filelock` is a dependency but out of scope.)

## Design

### New shared helpers in `genoray/_utils.py`

Two context managers so every write site uses the same crash-safe pattern.

**`atomic_write_path(dest)`** — single files (`.gvi`):

```python
@contextmanager
def atomic_write_path(dest: Path) -> Iterator[Path]:
    dest = Path(dest)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp)
    try:
        yield tmp                 # caller writes to `tmp`
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    else:
        os.replace(tmp, dest)     # atomic; same filesystem guaranteed
```

`mkstemp` creates an empty placeholder file in `dest.parent`; polars
`write_ipc`/`sink_ipc` overwrite it. The caller writes to the yielded `tmp`
path. On clean exit the tmp is atomically renamed onto `dest`; on any exception
the tmp is removed and `dest` is left untouched.

**`atomic_write_dir(dest)`** — `.svar` directories, backup-then-swap:

```python
@contextmanager
def atomic_write_dir(dest: Path) -> Iterator[Path]:
    dest = Path(dest)
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"))
    try:
        yield staging             # caller writes the whole output into `staging`
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # success → swap into place
    if dest.exists():
        backup = _unique_sibling(dest, suffix=".old")     # non-existent name
        os.replace(dest, backup)          # atomic: move old aside
        try:
            os.replace(staging, dest)     # atomic: put new in place
        except BaseException:
            os.replace(backup, dest)      # roll back
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)
    else:
        os.replace(staging, dest)
```

`_unique_sibling(dest, suffix)` returns a not-yet-existing sibling path (e.g.
`f".{dest.name}{suffix}.{os.getpid()}"`, incrementing a counter if it exists)
so the backup rename targets a fresh name — `os.replace` of a directory onto an
*existing* directory is not portable (fails on Windows; only succeeds onto an
empty dir on POSIX). Moving the old dir aside first means `dest` does not exist
when the new dir is renamed in, so the swap is atomic on every platform. On
failure of the final rename, the old dir is rolled back; the backup is always
removed at the end.

`dest.parent` must exist before either helper runs (`mkstemp`/`mkdtemp` with
`dir=` require it). Callers ensure this with `dest.parent.mkdir(parents=True,
exist_ok=True)`.

The `overwrite` flag is **not** consulted by `atomic_write_dir`: callers keep
their existing fail-fast `FileExistsError` guard, so by the time the context
manager runs, either `dest` does not exist or `overwrite=True` was passed. The
helper simply swaps whatever it finds.

### Site changes

| Site | Change |
|---|---|
| `VCF._write_gvi_index` (`_vcf.py:1215`) | wrap the `write_ipc` in `with atomic_write_path(self._index_path()) as tmp:` and write to `tmp` |
| PGEN `_write_index` (`_pgen.py:1253`) | wrap the `sink_ipc` in `with atomic_write_path(index_path) as tmp:` and sink to `tmp` |
| `SparseVar.from_vcf` (`_svar.py:1033`+) | remove `out.mkdir`; add `out.parent.mkdir(parents=True, exist_ok=True)`; wrap the write body in `with atomic_write_dir(out) as staging:`; redirect every `out` write target (`metadata.json`, `cls._index_path(out)`, `_concat_data(out, ...)`, `_subset_var_idxs_and_recompute_af(out, ...)`) to `staging`; change the chunk `TemporaryDirectory()` to `TemporaryDirectory(dir=out.parent)` |
| `SparseVar.from_pgen` (`_svar.py`) | same as `from_vcf`: `atomic_write_dir(out)`, redirect targets to `staging`, sibling chunk `TemporaryDirectory(dir=out.parent)` |
| `SparseVar.write_view` Band C | replace the `rmtree(output)` + `mkdir(output)` with `with atomic_write_dir(output) as staging:`; redirect all Band-C write targets to `staging` |

The `vcf._write_gvi_index()` call inside `from_vcf` (`_svar.py:1046`) writes the
**source VCF's** own `.gvi` (a sibling of the VCF file, not of `out`); it
becomes atomic via the `_write_gvi_index` change itself.

The existing PR #75 fail-fast guards in `write_view` Band A (`output` exists &&
`!overwrite` → `FileExistsError`; `output == source` → `ValueError`; mutcat /
reference / samples / regions validation) and the `from_vcf`/`from_pgen`
`FileExistsError` checks all run **before** the `atomic_write_dir` context is
entered, so nothing on disk is touched until they pass.

### Observable behavior

- Output bytes, schema, dtypes, and conventions are byte-identical to today.
- During a write a transient hidden sibling (`.<name>.tmp…`, and for overwrite a
  brief `.<name>.old.<pid>`) appears in the output's parent directory; both are
  cleaned up on success and on failure.
- A crash mid-write leaves no partial output; an existing output being
  overwritten is preserved intact until the new one is complete.

## Error handling

- Any exception raised by the caller's write code propagates unchanged after the
  staging tmp/file is cleaned up (`rmtree`/`unlink`); `dest` is untouched.
- If the final `os.replace(staging, dest)` fails in the overwrite path, the old
  directory is rolled back via `os.replace(backup, dest)` before re-raising.
- `shutil.rmtree(..., ignore_errors=True)` is used for backup/staging cleanup so
  a cleanup failure never masks the original outcome.

## Testing

**Helper unit tests (`tests/`):**
- `atomic_write_path`: writing then clean exit replaces `dest` atomically; the
  tmp lands in `dest.parent` (sibling); an exception in the body removes the tmp
  and leaves a pre-existing `dest` byte-unchanged.
- `atomic_write_dir`: clean exit swaps staging → `dest`; staging is a sibling;
  an exception in the body removes staging and leaves a pre-existing `dest`
  byte-unchanged; overwrite swaps and removes the backup; a forced failure of
  the final rename rolls back to the old dir.

**Integration tests:**
- `from_vcf` / `from_pgen` / `write_view` produce a dir-digest byte-identical to
  the current implementation (reuse the `_dir_digest` helper from
  `tests/test_svar_write_view.py`).
- An injected failure partway through the write (e.g. monkeypatch `_concat_data`
  / a field write to raise): a pre-existing `out` (overwrite) is preserved, no
  partial new `out` exists, and no leftover `.<name>.tmp…` sibling remains.
- `.gvi` build (`VCF._write_gvi_index`, PGEN `_write_index`): a forced failure
  leaves any prior `.gvi` untouched and no partial `.gvi`; a clean build is
  loadable and equal to the current output.

## Documentation

- `skills/genoray-api/SKILL.md`: add a one-line note that `.svar`/`.gvi` writes
  are atomic (written to a sibling tmp and renamed into place; a crash never
  leaves a partial output, and a transient `.<name>.tmp…` sibling may appear
  during a write). No public name/semantic change, so this is the only doc edit.
