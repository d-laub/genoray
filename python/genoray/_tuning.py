"""Explicit scheduling knobs for SVAR2 conversion.

Replaces the `GENORAY_*` environment variables. Every field is `None` by
default, meaning "let the planner derive it" -- the values a caller does not set
are chosen exactly as they are today.
"""

from __future__ import annotations

import operator
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
_APPLICABLE: dict[Backend, frozenset[str]] = {
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
            # `bool` is an `int` and defines `__index__`, so it survives both
            # checks below; nobody means `reader_workers=True`.
            if isinstance(value, bool):
                raise ValueError(f"{f.name} must be None or an int; got {value!r}")
            try:
                # `operator.index` rather than `isinstance(value, int)`: this is a
                # numpy-centric library and a caller who computed a knob from an
                # array gets an `np.int64`, which is not an `int`. Coercing here
                # also keeps every field a plain int at the FFI seam, where pyo3
                # extracts `TuningIn` by attribute from this dataclass instance.
                coerced = operator.index(value)
            except TypeError:
                raise ValueError(
                    f"{f.name} must be None or an int; got {value!r}"
                ) from None
            minimum = _MINIMUM[f.name]
            if coerced < minimum:
                raise ValueError(
                    f"{f.name} must be None (let the planner choose) or an "
                    f"integer >= {minimum}; got {value!r}"
                )
            if coerced is not value:  # frozen, so go around __setattr__
                object.__setattr__(self, f.name, coerced)

    def _check_backend(self, backend: Backend) -> None:
        """Raise if any set field cannot be used by `backend`."""
        allowed = _APPLICABLE[backend]  # KeyError on an unknown backend: a bug
        offenders = [
            f.name
            for f in fields(self)
            if getattr(self, f.name) is not None and f.name not in allowed
        ]
        if not offenders:
            return
        # Report every offender at once: a caller who set two inapplicable knobs
        # should not have to discover them one round-trip at a time.
        names = ", ".join(f"tuning.{n}" for n in offenders)
        verb, pronoun = ("do", "them") if len(offenders) > 1 else ("does", "it")
        raise ValueError(
            f"{names} {verb} not apply to the {backend!r} backend "
            f"(applicable knobs: {', '.join(sorted(allowed))}). "
            f"Leave {pronoun} as None."
        )
