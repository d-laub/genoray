from pathlib import Path

import pytest

from scripts.bench_svar2.plans.build_plans import (
    HOLDOUT,
    PROD_CHUNK_SIZE,
    SCALE_SAMPLES,
    VLINEAR_CHUNK_SIZE,
    VLINEAR_VARIANTS,
    _chunk_size_for,
    build,
)
from scripts.bench_svar2.scale_corpus import size_corpus

pytestmark = pytest.mark.bench


def _build():
    return build(Path("/corpora"), threads=8)


def test_every_point_id_is_unique():
    """A duplicate point_id wastes hours of cluster time re-measuring the same
    config under two names -- a prior review round already caught one."""
    plans = _build()
    all_ids = [pt.point_id for points in plans.values() for pt in points]
    assert len(all_ids) == len(set(all_ids))


def test_vlinear_plan_has_one_point_per_variant_count():
    plans = _build()
    assert len(plans["vlinear"]) == len(VLINEAR_VARIANTS)


def test_vlinear_varies_only_variants_via_the_corpus_path():
    """Everything on the SweepPoint itself -- chunk_size, reader_workers,
    threads -- must be identical across the ladder; only the corpus (which
    encodes V) may differ, or V is not the only thing varying."""
    plans = _build()
    pts = plans["vlinear"]
    for pt in pts:
        assert pt.chunk_size == VLINEAR_CHUNK_SIZE
        assert pt.reader_workers == 1
        assert pt.concurrent_chroms is None
    corpora = {pt.corpus for pt in pts}
    assert len(corpora) == len(pts)
    non_corpus_fields = {
        (
            pt.reader_workers,
            pt.concurrent_chroms,
            pt.shard_htslib,
            pt.overshard,
            pt.chunk_size,
            pt.threads,
            pt.reps,
            pt.rss_ceiling_mb,
        )
        for pt in pts
    }
    assert len(non_corpus_fields) == 1


def test_vlinear_chunk_size_clears_min_chunks_floor_at_smallest_v():
    assert min(VLINEAR_VARIANTS) // VLINEAR_CHUNK_SIZE >= 32


def test_chunk_size_for_matches_size_corpus_at_the_same_variant_count():
    """`_chunk_size_for` must be the same rule `size_corpus` applies, not a
    parallel implementation that can drift."""
    _, expected = size_corpus(samples=1, cells_budget=28_000)
    assert _chunk_size_for(28_000) == expected


def test_holdout_chunk_size_is_derived_not_a_bare_literal():
    plans = _build()
    assert plans["holdout"][0].chunk_size == _chunk_size_for(HOLDOUT["variants"])
    assert plans["holdout"][0].chunk_size != 2_000  # the old unrelated literal


def test_scale_plan_has_exactly_one_production_default_point_per_s():
    """Every S corpus gets exactly one (reader_workers=1, chunk_size=
    PROD_CHUNK_SIZE) point -- either the base w=1 point itself, when
    size_corpus's own clamp already lands on PROD_CHUNK_SIZE (S=250, 1_000),
    or a dedicated point added for I5. Two such points for the same corpus
    would collide point_ids and silently waste a sweep slot."""
    plans = _build()
    prod_points = [
        pt
        for pt in plans["scale"]
        if pt.chunk_size == PROD_CHUNK_SIZE and pt.reader_workers == 1
    ]
    corpora = [pt.corpus for pt in prod_points]
    assert len(corpora) == len(SCALE_SAMPLES)
    assert len(set(corpora)) == len(SCALE_SAMPLES)
    for pt in prod_points:
        assert pt.rss_ceiling_mb == 60_000


def test_scale_plan_adds_a_dedicated_point_only_where_size_corpus_is_not_clamped():
    plans = _build()
    unclamped_s = [
        s for s in SCALE_SAMPLES if size_corpus(s, 1_400_000_000)[1] != PROD_CHUNK_SIZE
    ]
    dedicated = [
        pt
        for pt in plans["scale"]
        if pt.chunk_size == PROD_CHUNK_SIZE
        and pt.reader_workers == 1
        and pt.corpus not in {f"/corpora/s{s}.manifest.json" for s in (250, 1_000)}
    ]
    assert len(dedicated) == len(unclamped_s) == 5


def test_all_four_plans_are_produced():
    assert set(_build().keys()) == {"scale", "contig", "holdout", "vlinear"}
