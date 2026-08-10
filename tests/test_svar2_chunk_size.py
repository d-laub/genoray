"""_auto_chunk_size must budget what a chunk actually costs. The staged FORMAT term is
n_format_fields * n_samples * 4 bytes per variant -- 112x the bit grid at F=7 -- so
ignoring it makes the memory budget meaningless (issue #120)."""

import pytest

from genoray._svar2 import (
    _DENSE_CHUNK_TARGET_BYTES,
    _STAGED_FORMAT_BYTES,
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
    # 256 MiB / (7 fields * 7089 samples * 4 B + 7089*2/8 B) per variant
    assert small == (256 * 1024**2) // (7 * 7089 * 4 + 7089 * 2 // 8)


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
    ) // 8 + n_format_fields * n_samples * _STAGED_FORMAT_BYTES
    assert cs * per_variant <= _DENSE_CHUNK_TARGET_BYTES


def test_a_budget_too_small_for_one_variant_still_makes_progress() -> None:
    assert _auto_chunk_size(10_000_000, 2, n_format_fields=7, max_mem=1024) == 1


def test_a_tiny_budget_warns_rather_than_silently_ignoring_itself() -> None:
    # Replaces test_chunk_size_never_goes_below_the_floor. The old floor did not
    # protect anything -- it silently returned a chunk 2x the budget.
    with pytest.warns(UserWarning, match="per dense chunk"):
        cs = _auto_chunk_size(10_000_000, 2, n_format_fields=7, max_mem=1024)
    assert cs == 1
