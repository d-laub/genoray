"""Tuning flag groups, shared by the write commands.

`Tuning` has six knobs, but not every backend can use all six (see
`_APPLICABLE` in `genoray._tuning`). A flag group is one dataclass per
applicability tier, mirroring that table, so a knob a backend cannot use is
not spellable on that command's CLI at all -- rejected at argument parsing,
not at runtime:

- `_ALWAYS` (`dense_cap`, `merge_threads`, `sample_interval`) is common to
  every backend and lives on `TuningFlags`, the base.
- `write pgen` / `write svar1` add `concurrent_chroms` -- `ConcurrentTuningFlags`.
- `write vcf` adds `concurrent_chroms`, `reader_workers`, `overshard` --
  `VcfTuningFlags`. (`write vcf`'s multi-file/vcf-list branch narrows this
  further at runtime via `Tuning._check_backend("vcf_list")`, since one
  command serves both the `vcf` and `vcf_list` backends.)

`to_tuning()` is defined once, on the base, over `dataclasses.fields(self)` so
a subclass's extra fields flow through without a second copy of the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Annotated

from cyclopts import Parameter

from .._tuning import Tuning


@dataclass
class TuningFlags:
    """Scheduling knobs applicable to every write command."""

    dense_cap: Annotated[int | None, Parameter(name="--dense-cap")] = None
    merge_threads: Annotated[int | None, Parameter(name="--merge-threads")] = None
    sample_interval: Annotated[int | None, Parameter(name="--sample-interval")] = None

    def to_tuning(self) -> Tuning:
        return Tuning(**{f.name: getattr(self, f.name) for f in fields(self)})


@dataclass
class ConcurrentTuningFlags(TuningFlags):
    """Adds `concurrent_chroms`.

    For backends that convert contigs concurrently but don't shard within a
    contig (`from_pgen`, `from_svar1`).
    """

    concurrent_chroms: Annotated[int | None, Parameter(name="--concurrent-chroms")] = (
        None
    )


@dataclass
class VcfTuningFlags(ConcurrentTuningFlags):
    """Adds `reader_workers`/`overshard`, the sharded-VCF-only knobs."""

    reader_workers: Annotated[int | None, Parameter(name="--reader-workers")] = None
    overshard: Annotated[int | None, Parameter(name="--overshard")] = None
