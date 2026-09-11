"""Grouped arguments for the Rust conversion pipeline entry points.

`_core.run_conversion_pipeline` reached 20 user-facing parameters one addition at
a time, and most of them were positionally interchangeable by type -- including
two identically typed field manifests, where a transposition would have routed
every INFO field into the FORMAT section and vice versa, silently (#153).

These mirror the `FromPyObject` structs in `src/pipeline_args.rs`, following the
pattern `Tuning` already established: fields cross the boundary **by name**, so
no argument order exists to get wrong, and `kw_only=True` means they cannot be
built positionally on this side either.

Internal. Every field is what the Rust side expects, already normalized -- these
are a transport shape, not a place to put defaults or validation, both of which
stay in the `SparseVar2.from_*` classmethods that build them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FieldTuple = tuple[str, str, str, "str | None", "float | None"]
"""One `_svar2_fields.py` manifest entry, as `field::parse_manifest` reads it."""


@dataclass(frozen=True, kw_only=True, slots=True)
class FieldSpec:
    """Which INFO and FORMAT fields to extract during the write.

    Attributes:
        info: INFO field manifest entries.
        format: FORMAT field manifest entries.
    """

    info: list[FieldTuple] = field(default_factory=list)
    format: list[FieldTuple] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True, slots=True)
class RegionSpec:
    """What to read: which contigs, which samples, which coordinate ranges.

    Attributes:
        chroms: Contigs to convert, in the caller's order.
        samples: Selected sample names.
        region_ranges: `(chrom, start, end)` triples, 0-based half-open.
        regions_overlap: Overlap mode; parsed and validated Rust-side, up front.
    """

    chroms: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    region_ranges: list[tuple[str, int, int]] = field(default_factory=list)
    regions_overlap: str = "pos"


@dataclass(frozen=True, kw_only=True, slots=True)
class PlanSettings:
    """The budget the planner works within.

    Distinct from `Tuning`, which overrides the planner's *decisions*: these are
    the constraints it plans against.

    Attributes:
        chunk_size: Variants per dense chunk.
        max_threads: Thread ceiling, or None to use detected parallelism.
        long_allele_capacity: Long-allele arena size in bytes.
        max_mem_bytes: Memory budget in bytes, or None to plan by core count.
    """

    chunk_size: int = 25_000
    max_threads: int | None = None
    long_allele_capacity: int = 8_388_608
    max_mem_bytes: int | None = None
