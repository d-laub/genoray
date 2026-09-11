"""_auto_chunk_size must budget what a chunk actually costs. The FORMAT term is
n_format_fields * n_samples * 12 bytes per variant -- 336x the bit grid at F=7 -- so
ignoring it makes the memory budget meaningless (issue #120). The 12 is 4 B staged
plus the 8 B of raw buffer the reader retains for the chunk's lifetime; charging the
staged 4 alone under-counted the live set by 3x (issue #156)."""

import pytest

from genoray._svar2 import (
    _DENSE_CHUNK_TARGET_BYTES,
    _FORMAT_BYTES_PER_SAMPLE,
    _RETAINED_FORMAT_BYTES,
    _auto_chunk_size,
)


def test_chunk_size_shrinks_when_format_fields_are_requested() -> None:
    no_fields = _auto_chunk_size(7089, 2, n_format_fields=0)
    with_fields = _auto_chunk_size(7089, 2, n_format_fields=7)
    assert with_fields < no_fields, (
        "F=7 makes a chunk ~112x bigger; the budget must react"
    )


def test_chunk_size_respects_an_explicit_budget() -> None:
    small = _auto_chunk_size(7089, 2, n_format_fields=7, max_mem=256 * 1024**2)
    big = _auto_chunk_size(7089, 2, n_format_fields=7, max_mem=4 * 1024**3)
    assert small < big
    # 256 MiB / (7 fields * 7089 samples * 12 B + 7089*2/8 B) per variant.
    # The 12 is `_FORMAT_BYTES_PER_SAMPLE`, not a literal: a hardcoded 4 here is
    # what let the staged-only under-count (#156) sit unnoticed in a test that
    # reads like it is checking the budget.
    assert small == (256 * 1024**2) // (
        7 * 7089 * _FORMAT_BYTES_PER_SAMPLE + 7089 * 2 // 8
    )


def test_zero_fields_matches_the_historical_default() -> None:
    # No fields requested => unchanged behaviour for every existing caller.
    assert _auto_chunk_size(2, 2, n_format_fields=0) == 25_000


@pytest.mark.parametrize("n_samples", [7089, 128_000, 500_000, 2_000_000])
@pytest.mark.parametrize("n_format_fields", [0, 7])
def test_chunk_never_exceeds_the_budget_it_was_given(
    n_samples: int, n_format_fields: int
) -> None:
    """The invariant the docstring claims. The old `max(1024, ...)` floor broke it
    exactly where it mattered: at S=2,000,000 the budget wants 536 variants and
    got 1024, i.e. a ~512 MB chunk against a 256 MiB target."""
    cs = _auto_chunk_size(n_samples, 2, n_format_fields=n_format_fields)
    per_variant = (
        n_samples * 2
    ) // 8 + n_format_fields * n_samples * _FORMAT_BYTES_PER_SAMPLE
    assert cs * per_variant <= _DENSE_CHUNK_TARGET_BYTES


def test_budget_covers_the_raw_format_buffer_the_reader_retains() -> None:
    """#156. The reader holds the record's raw FORMAT buffer -- one f64 per
    (sample, field) -- for as long as the chunk's metadata is live, so the live
    set is `chunk_size * n_samples * n_fields * 8` B. Charging only the staged
    4 B sized `chunk_size` against a third of the truth: at S=128,000 with one
    field the budget afforded 493 variants, whose retention alone was ~505 MB
    against a 256 MiB target (and ~3.5 GB before the buffer was flattened)."""
    n_samples, n_fields = 128_000, 1
    cs = _auto_chunk_size(n_samples, 2, n_format_fields=n_fields)
    retained = cs * n_samples * n_fields * _RETAINED_FORMAT_BYTES
    assert retained <= _DENSE_CHUNK_TARGET_BYTES


def test_a_budget_too_small_for_one_variant_still_makes_progress() -> None:
    assert _auto_chunk_size(10_000_000, 2, n_format_fields=7, max_mem=1024) == 1


def test_a_tiny_budget_warns_rather_than_silently_ignoring_itself() -> None:
    # Replaces test_chunk_size_never_goes_below_the_floor. The old floor did not
    # protect anything -- it silently returned a chunk 2x the budget.
    with pytest.warns(UserWarning, match="per dense chunk"):
        cs = _auto_chunk_size(10_000_000, 2, n_format_fields=7, max_mem=1024)
    assert cs == 1
