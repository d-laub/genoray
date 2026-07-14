"""TEMPORARY. Proves the slicer-backed reroute=True matches the pipeline-backed
one before the pipeline is deleted (see the unify-routing plan, Task 8/9).
Deleted together with run_view_pipeline in Task 9.

Calls both `_core` entry points directly (`run_view_pipeline`, the path being
deleted, and `run_slice_view(..., reroute=True, ...)`, its replacement) so this
test exercises the exact same production code `SparseVar2.write_view` does,
without going through the Python wrapper.
"""

from pathlib import Path

import numpy as np
import pytest

from genoray import SparseVar2, _core

# Single-contig fixture (`svar2_store`, session-scoped, see conftest.py):
# variants at 0-based POS 2 (SNP), 6 (insertion), 11..14 (deletion). Two
# regions, with a boundary landing exactly on the deletion's POS (11), so the
# three `regions_overlap` modes genuinely select different variant SETS for
# region 1 (`record` keeps the boundary variant, `pos`/`variant` prune it) --
# not just relabeling the same variants, real coverage of the overlap-mode
# dispatch in both backends.
REGIONS = [("chr1", 0, 11), ("chr1", 20, 40)]

# `None` below is resolved to "every available sample" (mirrors
# `SparseVar2.write_view`'s own `_normalize_samples(None, ...)` semantics) --
# both `_core` entry points require a concrete, non-empty `Vec<String>`, they
# have no "all samples" sentinel of their own.
SAMPLE_PARAMS = [["S0"], ["S0", "S1"], None]


def _routing_of(sv: SparseVar2, contig: str) -> dict[tuple[int, int], bool]:
    """Per-variant routing for `contig`: ``{(pos, key): is_dense}``.

    Built from `overlap_batch`'s two-channel result (`vk_pos`/`vk_key` for
    var_key-routed variants, `dense_pos`/`dense_key` for dense-routed ones),
    which is search-order/sort-based and therefore deterministic and directly
    joinable across two independently-built stores.

    Deliberately NOT `_core.svar2_variant_stats`: it also reports `src_dense`
    per variant, but groups internally by `(pos, key)` in a plain
    `std::collections::HashMap` and returns only the aggregated values -- no
    `pos`/`key` arrays at all, and (because of `HashMap`'s per-instance random
    seed) no order that is stable even across two calls in the same process.
    Without a join key there would be no honest way to line up "variant i in
    the old store" with "variant i in the new store".
    """
    d = sv._overlap_batch(contig, [(0, 2**31 - 1)])
    routing: dict[tuple[int, int], bool] = {
        (int(p), int(k)): False for p, k in zip(d["vk_pos"], d["vk_key"])
    }
    routing.update(
        {(int(p), int(k)): True for p, k in zip(d["dense_pos"], d["dense_key"])}
    )
    return routing


def _store_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _assert_decoded_genotypes_equal(
    old: SparseVar2, new: SparseVar2, contig: str
) -> None:
    ra = old.decode(contig, [(0, 2**31 - 1)])
    rb = new.decode(contig, [(0, 2**31 - 1)])
    assert ra.shape == rb.shape
    for key in ("pos", "ilen", "allele"):
        np.testing.assert_array_equal(
            np.asarray(ra[key].data), np.asarray(rb[key].data)
        )
        np.testing.assert_array_equal(ra[key].lengths, rb[key].lengths)


@pytest.mark.parametrize("samples", SAMPLE_PARAMS, ids=["S0", "S0-S1", "all"])
@pytest.mark.parametrize("overlap", ["pos", "record", "variant"])
def test_new_reroute_matches_old_pipeline(svar2_store, samples, overlap, tmp_path):
    src = SparseVar2(svar2_store)
    contig = src.contigs[0]
    sample_names = list(samples) if samples is not None else list(src.available_samples)

    old_dir = tmp_path / "old.svar2"
    new_dir = tmp_path / "new.svar2"

    _core.run_view_pipeline(  # the path being deleted
        str(svar2_store),
        str(old_dir),
        src.contigs,
        sample_names,
        REGIONS,
        overlap,
        False,  # merge_overlapping
        [],  # fields
        None,  # reference
        None,  # max_threads
        False,  # overwrite
    )
    _core.run_slice_view(  # the replacement, reroute=True
        str(svar2_store),
        str(new_dir),
        src.contigs,
        sample_names,
        REGIONS,
        overlap,
        False,  # merge_overlapping
        [],  # fields
        None,  # reference
        True,  # reroute
        None,  # max_threads
        False,  # overwrite
    )

    old = SparseVar2(old_dir)
    new = SparseVar2(new_dir)
    assert old.available_samples == new.available_samples == sample_names

    # 1. Same variant set + same decoded genotypes.
    _assert_decoded_genotypes_equal(old, new, contig)
    # 2. Same per-variant routing -- the whole reason reroute=True exists.
    assert _routing_of(new, contig) == _routing_of(old, contig)
    # 3. No size regression (LUT compaction may make the new one SMALLER).
    assert _store_size(new_dir) <= _store_size(old_dir)
