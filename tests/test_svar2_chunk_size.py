"""_auto_chunk_size must budget what a chunk actually costs. The staged FORMAT term is
n_format_fields * n_samples * 4 bytes per variant -- 112x the bit grid at F=7 -- so
ignoring it makes the memory budget meaningless (issue #120)."""

from genoray._svar2 import _auto_chunk_size


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
    assert small == max(1024, (256 * 1024**2) // (7 * 7089 * 4 + 7089 * 2 // 8))


def test_chunk_size_never_goes_below_the_floor() -> None:
    assert _auto_chunk_size(10_000_000, 2, n_format_fields=7, max_mem=1024) == 1024


def test_zero_fields_matches_the_historical_default() -> None:
    # No fields requested => unchanged behaviour for every existing caller.
    assert _auto_chunk_size(2, 2, n_format_fields=0) == 25_000
