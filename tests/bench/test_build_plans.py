from pathlib import Path

import pytest

from scripts.bench_svar2.plans.build_plans import (
    HOLDOUT,
    OOM_PROBE_CEILING_MB,
    PROD_CHUNK_SIZE,
    SCALE_SAMPLES,
    VLINEAR_CHUNK_SIZE,
    VLINEAR_SAMPLES,
    VLINEAR_VARIANTS,
    VLINEAR2_SAMPLES,
    VLINEAR2_VARIANTS,
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


def test_vlinear_is_fitted_at_the_chunk_size_extrapolate_predicts():
    """The V-law is only usable at the chunk size it was fitted under.

    The first ladder was fitted at chunk_size=781 (derived from the smallest V)
    while `extrapolate` targets PROD_CHUNK_SIZE=25_000. Because phase-1 cost at
    S=250 is dominated by per-CHUNK overhead rather than per-variant work, that
    slope was a per-chunk artifact: it predicted 740.5s at V=5_600_000 where
    62.9s was measured (11.8x high), which became a 2820% hold-out error.
    """
    assert VLINEAR_CHUNK_SIZE == PROD_CHUNK_SIZE
    plans = _build()
    for pt in plans["vlinear"]:
        assert pt.chunk_size == PROD_CHUNK_SIZE


def test_vlinear_ladder_sits_in_the_many_chunk_regime():
    """Every rung must be deep enough into chunk-count that per-chunk fixed cost
    cannot dominate the fitted slope -- the failure that falsified the first
    ladder, whose per-variant cost was still falling monotonically at its top
    rung (3.04e-4 -> 1.61e-4 s/variant) and fell a further 14x by V=5.6e6."""
    for v in VLINEAR_VARIANTS:
        assert v // VLINEAR_CHUNK_SIZE >= 32, (
            f"V={v:,} is only {v // VLINEAR_CHUNK_SIZE} chunks at "
            f"chunk_size={VLINEAR_CHUNK_SIZE:,}"
        )


def test_vlinear_top_rung_bounds_the_extrapolation_stretch_to_1e9():
    """`extrapolate`'s target is V=1e9, and the V-law's credibility is the ratio
    of that to the largest V actually measured. The original ladder topped out
    at 200_000 -- a 5000x stretch, which the model itself flagged. Keep the top
    rung within a stretch the harness can defend."""
    assert 1_000_000_000 / max(VLINEAR_VARIANTS) <= 200


def test_vlinear_rungs_are_affordable_at_the_fixed_cohort_size():
    """The ladder is only cheap because S is pinned small; the top rung is the
    binding cost. Generation cost is linear in cells = S*V, and the largest
    scale corpus in this harness is 1.4e9 cells, so no rung may exceed it."""
    from scripts.bench_svar2.plans.build_plans import VLINEAR_SAMPLES

    assert max(VLINEAR_VARIANTS) * VLINEAR_SAMPLES <= 1_400_000_000


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


def test_only_the_oom_probe_points_carry_a_ceiling():
    """`probe.py:_build_env` sets `MALLOC_ARENA_MAX=1` on any point with an
    `rss_ceiling_mb`, which is not the production allocator. Pinning glibc to
    one arena while running `-@ 48` with up to 11 reader workers was measured
    at 73% slower in an earlier multithreaded conversion regime, so it must
    not touch the points the laws are fitted from -- only the dedicated
    production-chunk_size points, which exist to find the OOM.

    The clamped-S points (S=250, 1_000, where size_corpus already lands on
    PROD_CHUNK_SIZE) stay ceiling-free: they double as law-fitting points, and
    a 25_000-variant chunk at S=250 is 1.5 MB of grid, so there is no OOM
    there to probe.
    """
    plans = _build()
    ceilinged = [pt for points in plans.values() for pt in points if pt.rss_ceiling_mb]
    dedicated = {
        f"/corpora/s{s}.manifest.json"
        for s in SCALE_SAMPLES
        if size_corpus(s, 1_400_000_000)[1] != PROD_CHUNK_SIZE
    }
    assert {pt.corpus for pt in ceilinged} == dedicated
    for pt in ceilinged:
        assert pt.rss_ceiling_mb == OOM_PROBE_CEILING_MB
        assert pt.chunk_size == PROD_CHUNK_SIZE
        assert pt.reader_workers == 1

    # Everything the laws are fitted from runs on the production allocator.
    law_points = [pt for name in ("contig", "holdout", "vlinear") for pt in plans[name]]
    law_points += [pt for pt in plans["scale"] if pt.chunk_size != PROD_CHUNK_SIZE]
    assert law_points
    assert all(pt.rss_ceiling_mb is None for pt in law_points)


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


def test_all_seven_plans_are_produced():
    assert set(_build().keys()) == {
        "scale",
        "contig",
        "holdout",
        "vlinear",
        "vlinear2",
        "concurrency",
        "pgen",
    }


def test_concurrency_plan_spans_the_measured_corners(tmp_path):
    """The contig axis compared cc=1,w=12 against cc=4,w=3 and found 2.99x.
    The new planner can reach cc=15; the sweep must cover that, or it cannot
    tell whether the planner's choice is the good one."""
    plans = build(tmp_path, threads=48)
    cc_values = {
        p.concurrent_chroms for p in plans["concurrency"] if p.concurrent_chroms
    }
    assert max(cc_values) >= 15
    assert 1 in cc_values


def test_the_two_v_ladders_sit_at_different_cohort_sizes():
    """The whole point of the second ladder is a second cohort size. If both
    ladders ever collapsed to one S, `fit_cohort_beta_from_ladders` would
    return None and beta would silently fall back to the constant-cells fit
    that forces beta ~ 1."""
    assert VLINEAR_SAMPLES != VLINEAR2_SAMPLES


def test_neither_v_ladder_sits_at_the_holdout_cohort_size():
    """A ladder at the hold-out's S would make the hold-out an interpolation
    inside the fitted data, so the 25% gate would go quiet for the wrong
    reason -- it would be re-checking V-linearity, which the ladder's own R^2
    already reports, instead of testing the composed S,V extrapolation."""
    assert HOLDOUT["samples"] not in (VLINEAR_SAMPLES, VLINEAR2_SAMPLES)


def test_the_holdout_cohort_is_bracketed_by_the_two_ladders():
    """Strictly between the ladders, so the hold-out interpolates in S rather
    than extrapolating past both -- the composed prediction is then tested
    where the cohort law is best supported, not off the end of it."""
    lo, hi = sorted((VLINEAR_SAMPLES, VLINEAR2_SAMPLES))
    assert lo < HOLDOUT["samples"] < hi


def test_the_second_v_ladder_pins_chunk_size_across_its_rungs():
    """V must be the only axis moving. A per-rung chunk size would co-vary
    with V and re-introduce the per-chunk-cost confound that once made the
    V-law over-predict by 11.8x."""
    pts = [p for p in _build()["vlinear2"] if "vlinear2_v" in p.corpus]
    assert len(pts) == len(VLINEAR2_VARIANTS)
    assert len({p.chunk_size for p in pts}) == 1
    assert all(p.reader_workers == 1 for p in pts)


def _shape_of(point) -> tuple[int, int]:
    """(samples, variants) encoded in the corpus path, e.g. `.../s4000_v250000/`."""
    import re

    m = re.search(r"s(\d+)_v(\d+)", point.corpus)
    assert m, f"corpus path does not encode its shape: {point.corpus}"
    return int(m.group(1)), int(m.group(2))


def test_pgen_family_has_two_v_ladders_at_different_sample_counts():
    """A ladder that holds S*V constant forces the cohort exponent to ~1
    arithmetically -- it cannot identify beta no matter how many points it
    has. Two V-ladders at DIFFERENT S is the minimum that can."""
    from scripts.bench_svar2.plans.build_plans import build

    plans = build(Path("/tmp/corpora"), threads=48)
    pgen = plans["pgen"]
    assert pgen, "pgen family must not be empty"
    assert all(p.backend == "pgen" for p in pgen)

    by_samples: dict[int, set[int]] = {}
    for p in pgen:
        s, v = _shape_of(p)
        by_samples.setdefault(s, set()).add(v)
    ladders = [s for s, vs in by_samples.items() if len(vs) >= 2]
    assert len(ladders) >= 2, (
        f"need >=2 V-ladders at different S to identify the cohort exponent; "
        f"got {by_samples}"
    )


def test_pgen_matrix_can_identify_kappa():
    """kappa's CI spanned zero (SE 7.44, 95% CI [-9.99, +23.68]) because
    `_chunk_size_for` depends only on V and both ladders swept the same three
    V values, so `chunk_size` and `S*chunk_size` stayed correlated with S.
    Varying chunk_size at a FIXED (S, V) decorrelates them; a third cohort
    width tightens `per_sample_mb`'s extrapolation to the 500,000-sample
    target."""
    plans = build(Path("/tmp/corpora"), threads=48)
    pgen = plans["pgen"]

    chunk_sizes_by_shape: dict[tuple[int, int], set[int]] = {}
    widths: set[int] = set()
    for p in pgen:
        s, v = _shape_of(p)
        chunk_sizes_by_shape.setdefault((s, v), set()).add(p.chunk_size)
        widths.add(s)

    assert max(len(cs) for cs in chunk_sizes_by_shape.values()) >= 3, (
        "need >=3 distinct chunk_size values at some fixed (S, V) to "
        f"decorrelate chunk_size from S; got {chunk_sizes_by_shape}"
    )
    assert len(widths) >= 3, f"need >=3 distinct cohort widths; got {widths}"

    for p in pgen:
        _, v = _shape_of(p)
        chunks_per_contig = v / 22 / p.chunk_size
        assert chunks_per_contig >= 1.0, (
            f"{p.corpus} chunk_size={p.chunk_size} leaves "
            f"{chunks_per_contig:.2f} chunks/contig < 1 -- BitGrid3::zeros "
            "reserves the full chunk_size and truncates after EOF, so a "
            "partial chunk breaks the linearity kappa measures"
        )


def test_pgen_concurrency_axis_holds_workers_at_one():
    """PGEN pins P=1 (sub-contig sharding disabled), so a reader_workers
    axis would measure nothing."""
    from scripts.bench_svar2.plans.build_plans import build

    for p in build(Path("/tmp/corpora"), threads=48)["pgen"]:
        assert p.reader_workers == 1


def test_every_point_corpus_is_a_manifest_path():
    """`sweep.py:run_sweep` reads `point.corpus` as UTF-8 text and decodes it
    as a `CorpusManifest` -- it is a `sweep.py`-wide contract that every
    family's `corpus` names a `*.manifest.json`, never the underlying data
    file. The pgen family once pointed its points at the binary `.pgen`
    directly, which raised `UnicodeDecodeError` 4.5 hours into a real Slurm
    sweep. This constrains every family, not just pgen, so the next family
    added hits the same trap here instead of at runtime."""
    plans = build(Path("/tmp/corpora"), threads=48)
    offenders = [
        pt.corpus
        for points in plans.values()
        for pt in points
        if not pt.corpus.endswith(".manifest.json")
    ]
    assert not offenders, f"corpus paths not pointing at a manifest: {offenders}"


def test_pgen_crosses_concurrency_with_chunk_size_at_more_than_one_width():
    """The 2026-08-07 sweep varied `concurrent_chroms` at exactly ONE
    (S, V, chunk_size) corner, so nothing could say whether the per-contig RAM
    cost is additive or interacts with cohort width or chunk size. Fitting it
    as additive anyway is part of why that refit reached only R^2 0.7698
    against a measured reproducibility floor of 63 MB (issue #158)."""
    pgen = build(Path("/corpora"), threads=48)["pgen"]

    crossed: dict[int, set[int]] = {}
    for p in pgen:
        if p.concurrent_chroms is None:
            continue
        s, _ = _shape_of(p)
        crossed.setdefault(s, set())
    for s in list(crossed):
        by_cs: dict[int, set[int]] = {}
        for p in pgen:
            if p.concurrent_chroms is None or _shape_of(p)[0] != s:
                continue
            by_cs.setdefault(p.chunk_size, set()).add(p.concurrent_chroms)
        crossed[s] = {cs for cs, ccs in by_cs.items() if len(ccs) >= 2}

    widths = [s for s, cs in crossed.items() if len(cs) >= 2]
    assert len(widths) >= 2, (
        "need >=2 cohort widths that each vary concurrent_chroms at >=2 "
        f"chunk_sizes, or the per-contig term cannot be tested for "
        f"additivity; got {crossed}"
    )


def test_pgen_varies_n_chunks_at_constant_chunk_bytes():
    """`PGEN_CHUNK_SIZE_AXIS` moves chunk_bytes UP and n_chunks DOWN as exact
    reciprocals, so it cannot distinguish a per-chunk residency cost from a
    per-cycle allocator ratchet. Measured peak RSS is non-monotone in chunk
    bytes -- at S=4,000 the 3.1 -> 7.8 MB step DROPS RSS by 586 MB, 9.3x the
    63 MB noise floor -- which no `kappa * (w+p) * chunk_bytes` term can
    produce at any kappa. Separating them needs V varied at PINNED
    chunk_size."""
    pgen = build(Path("/corpora"), threads=48)["pgen"]

    # (samples, chunk_size) -> distinct variant counts. chunk_bytes is a
    # function of exactly those two, so a group with >=3 variant counts varies
    # n_chunks with chunk_bytes held exactly constant.
    by_fixed: dict[tuple[int, int], set[int]] = {}
    for p in pgen:
        s, v = _shape_of(p)
        by_fixed.setdefault((s, p.chunk_size), set()).add(v)

    usable = {k: vs for k, vs in by_fixed.items() if len(vs) >= 3}
    assert usable, (
        "no (samples, chunk_size) group carries >=3 variant counts, so "
        "n_chunks never moves independently of chunk_bytes; got "
        f"{ {k: sorted(v) for k, v in by_fixed.items()} }"
    )
    for (s, cs), vs in usable.items():
        spread = max(vs) / min(vs)
        assert spread >= 3.0, (
            f"S={s} chunk_size={cs} spans only {spread:.1f}x in V; too short a "
            "lever to separate a ratchet term from kappa"
        )
