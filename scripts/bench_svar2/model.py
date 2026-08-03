"""Fit the three scaling laws and return a falsifiable hypothesis verdict.

Pure: every function takes numbers and returns dataclasses. That is what makes
the verdict testable against planted synthetic laws rather than only against
cluster runs.

No scipy. With 5-7 scale points a normal approximation to the 95% CI is
materially wrong (t(5)=2.571 vs z=1.96), so a small Student-t table is
inlined instead of taking a dependency.
"""

from __future__ import annotations

import argparse
import sys
import typing
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scripts.bench_svar2.records import (
    CorpusManifest,
    CostLaw,
    ProbeRecord,
    RamLaw,
    SweepPoint,
    VLaw,
    Verdict,
    from_json,
    read_ndjson,
)

# Two-tailed t at alpha=0.05, indexed by degrees of freedom.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}
PLOIDY = 2
# Spec thresholds. Changing these changes the verdict, so they are named.
H1_KNEE_TOLERANCE = 1  # w* varies by less than +/-1 across the S range
# Minimum number of DISTINCT cohort sizes that must yield a knee before "w*
# barely varies" is allowed to mean anything. H1's whole claim is that the
# knee is FLAT ACROSS THE SAMPLE RANGE, and a spread computed over one or two
# cohort sizes cannot witness that: a single knee has spread 0 by definition,
# so an unguarded gate returns a confident "a static cap suffices, no
# autotuner needed" from the one datum that is equally consistent with every
# hypothesis. Two points are barely better -- they can only ever show a
# difference, never a trend. The scale plan supplies 7 cohort sizes, so this
# floor binds only when most of the sweep failed to produce usable ticks,
# which is exactly when a confident verdict is least warranted.
H1_MIN_KNEE_POINTS = 3
V_LAW_MIN_R2 = 0.98
HOLDOUT_ERROR_GATE = 0.25  # spec: >25% predicted-vs-actual error is a model failure
# Share of measured peak RSS the reorder backlog must account for before H3(a)
# fires -- see `decide` for why the gate is a BYTE share and not the spec's
# literal `pending_highwater >= w/2` count.
#
# 0.25 is deliberately the same number as `HOLDOUT_ERROR_GATE`: that gate is
# already this module's definition of "a term this size is not a modelling
# detail, it is the model", and a backlog worth a quarter of peak RSS cannot
# be bounded by capping `w` -- it has to be bounded in bytes, which is exactly
# what H3 claims. One notion of "material" for the whole module.
H3_BACKLOG_RSS_FRACTION = 0.25
# fit_v_law's max_extrapolation_factor is always stated relative to this many
# variants (the biobank target). Keeping it as one constant lets `extrapolate`
# invert it exactly instead of re-guessing what "1e9" in records.py means.
_V_LAW_TARGET_VARIANTS = 1e9


def _t95(df: int) -> float:
    if df <= 0:
        return float("inf")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def _linfit(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float, float, float]:
    """Least squares. Returns (slope, intercept, r2, slope_stderr)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(xa)
    slope, intercept = np.polyfit(xa, ya, 1)
    pred = slope * xa + intercept
    ss_res = float(((ya - pred) ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if n > 2 and ss_res > 0:
        sxx = float(((xa - xa.mean()) ** 2).sum())
        stderr = float(np.sqrt(ss_res / (n - 2) / sxx)) if sxx > 0 else float("inf")
    else:
        # Degenerate: too few points to estimate a residual variance (n <= 2
        # always fits exactly), or a residual of exactly zero even with more
        # points. Either way there is no information to bound the true slope
        # from. Report unbounded uncertainty rather than stderr=0.0 -- a zero
        # stderr collapses beta_ci95 to a single point, which previously made
        # `decide` report H2 on a difference of any size, including noise.
        stderr = float("inf")
    return float(slope), float(intercept), r2, stderr


def fit_v_law(points: Sequence[tuple[float, float]]) -> VLaw:
    """points: (variants, phase1_s). Wall must be linear in variant count for
    the extrapolation to 10^9 variants to mean anything."""
    v = [p[0] for p in points]
    t = [p[1] for p in points]
    slope, intercept, r2, _ = _linfit(v, t)
    return VLaw(
        slope_s_per_variant=slope,
        intercept_s=intercept,
        r2=r2,
        n_points=len(points),
        max_extrapolation_factor=_V_LAW_TARGET_VARIANTS / max(v),
    )


def fit_cost_law(
    name: str, samples: Sequence[float], costs: Sequence[float]
) -> CostLaw:
    """cost(S) = alpha * S**beta, fitted on logs."""
    beta, log_alpha, _, stderr = _linfit(np.log(samples), np.log(costs))
    half = _t95(len(samples) - 2) * stderr
    return CostLaw(
        name=name,
        alpha=float(np.exp(log_alpha)),
        beta=beta,
        beta_ci95=(beta - half, beta + half),
        n_points=len(samples),
    )


class RamRow(typing.NamedTuple):
    """One observation the RAM law is fitted from.

    Named rather than a bare tuple because the payload is four ints and a
    float: swapping `chunk_bytes` and `samples` positionally still fits, just
    wrongly, and no test would necessarily catch it. Fields are read by name at
    every consumer (`fit_ram_law`, `decide`).
    """

    workers: int
    pending: int
    chunk_bytes: int
    samples: int
    peak_rss_mb: float


def fit_ram_law(rows: Sequence[RamRow]) -> RamLaw:
    """peak_rss ~ base + per_sample_mb * samples
    + kappa * (workers + pending_hw) * chunk_bytes.

    Two regressors, not one. `kappa` is the observed overhead multiple over the
    analytic chunk size (a DenseChunk holds more than its packed grid), and
    `per_sample_mb` is the cohort-sized term: per-sample accumulation buffers
    that exist whether or not any chunk is in flight. Fitting only the chunk
    term forces the cohort cost into the intercept, where it cannot vary with S
    -- that is what held the fit at R^2=0.057 across a real 39-point sweep.

    Solved as ordinary least squares over both regressors simultaneously rather
    than by fitting one and regressing the other on the residual: the two are
    correlated in any real sweep (bigger cohorts get smaller chunk_size), so
    sequential fitting assigns shared variance to whichever goes first.
    """
    chunk = np.array(
        [(r.workers + r.pending) * r.chunk_bytes / 1e6 for r in rows], dtype=float
    )
    samples = np.array([float(r.samples) for r in rows], dtype=float)
    y = np.array([r.peak_rss_mb for r in rows], dtype=float)

    # A sweep at a SINGLE cohort size makes the `samples` column a constant
    # multiple of the intercept column. The two are then unidentifiable, and
    # least squares happily returns a minimum-norm split of the intercept
    # between them -- a `per_sample_mb` that is pure arithmetic artifact.
    # `extrapolate` multiplies that coefficient by the target cohort (500,000),
    # so an artifact here becomes hundreds of GB of projected RSS. Drop the
    # regressor instead and let `base_mb` own the constant, which is what a
    # one-cohort sweep can actually support.
    cohort_identifiable = bool(samples.std() > 0)
    cols = (
        [np.ones(len(rows)), samples, chunk]
        if cohort_identifiable
        else [
            np.ones(len(rows)),
            chunk,
        ]
    )
    a = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    if cohort_identifiable:
        base, per_sample, kappa = (float(c) for c in coef)
    else:
        base, kappa = (float(c) for c in coef)
        per_sample = 0.0
    pred = a @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return RamLaw(
        base_mb=base,
        per_sample_mb=per_sample,
        kappa=kappa,
        r2=r2,
        n_points=len(rows),
    )


def _median_costs(
    cpu_shard_pct: Sequence[float], cpu_exec_pct: Sequence[float]
) -> tuple[float, float] | None:
    """Median (c_read, c_exec) from one w=1 probe's per-tick CPU samples.

    At w=1 the shard aggregate is one reader's cost and cpu_exec is the
    executor's, so these medians ARE `c_read(S)` and `c_exec(S)` from the
    spec's cost laws, not just an internal detail of the knee ratio. Ticks
    where either is zero are startup/teardown and are dropped. Returns None
    when nothing survives the filter (e.g. the single-reader fallback path
    reports no shard/exec ticks at all).
    """
    pairs = [(s, e) for s, e in zip(cpu_shard_pct, cpu_exec_pct) if s > 0 and e > 0]
    if not pairs:
        return None
    shard = float(np.median([p[0] for p in pairs]))
    exec_ = float(np.median([p[1] for p in pairs]))
    return shard, exec_


def knee_from_probe(
    cpu_shard_pct: Sequence[float], cpu_exec_pct: Sequence[float]
) -> int:
    """Predicted knee from a single w=1 run: ceil(c_read / c_exec)."""
    costs = _median_costs(cpu_shard_pct, cpu_exec_pct)
    if costs is None:
        return 1
    shard, exec_ = costs
    if exec_ <= 0:
        return 1
    return max(1, int(np.ceil(shard / exec_)))


def decide(
    knees: dict[int, int],
    read_law: CostLaw,
    exec_law: CostLaw,
    rows: Sequence[RamRow],
    ram_law: RamLaw | None,
    contig_counterfactual: tuple[float, float] | None = None,
) -> Verdict:
    """Apply the spec's falsifiable criteria, in H3-first order.

    H3 supersedes H1/H2: if bytes rather than worker count set peak RSS, or if
    the multi-contig regression is a partitioning artifact, a byte-bounded
    global pool needs no knee prediction at all.

    H3(a) IS NOT THE SPEC'S LITERAL `pending_highwater >= w/2`. That criterion
    cannot discriminate, because `pending_highwater` grows with `w` for purely
    structural reasons: `ReorderBuffer::push` (src/shard_exec.rs) releases a
    chunk on arrival only when its `ordinal == head`, so with `w` workers
    pulling `w` units concurrently, the `w - 1` units ahead of the head keep
    every chunk they produce in `PendingBacklog::map` until the head unit's
    `Done`. Roughly `(w - 1) * chunks_per_unit` chunks are therefore resident
    at all times even with perfectly balanced readers and zero skew. Measured,
    not hypothetical: a 12-unit, `w=3`, `overshard=4` probe log in this repo
    sustains `pending=5` for the whole run, i.e. `pending/w = 1.67`, which
    clears `w/2` outright -- and the scale plan sweeps `w in {2,3,5,7,11}` and
    the contig plan `w in {12,6,3}`, so EVERY row of every planned sweep would
    trip a count-based gate. A gate that always fires is not evidence.

    What H3 actually claims is that in-flight BYTES, not worker count, set peak
    RSS. So test that directly: take the backlog's modelled contribution to
    peak RSS from the fitted RAM law (`kappa * pending_hw * chunk_bytes`, the
    same `kappa` and the same per-row `chunk_bytes` that law was fitted on) and
    require it to be at least `H3_BACKLOG_RSS_FRACTION` of the peak RSS
    actually measured on that row.

    This stays REACHABLE precisely where the hypothesis is live: the structural
    `w - 1` backlog is immaterial when a chunk is small (every small-`S` row of
    the sweep, where `chunk_bytes` is single-digit MB) and dominant when it is
    not (at `S=500_000` with `from_vcf`'s hardcoded `chunk_size=25_000` one
    chunk alone is ~3.1 GB, so even a one-chunk backlog is many times the whole
    process footprint). That is the same axis H3 argues about -- bytes -- and
    it is the axis a static worker cap cannot control.

    What this gate does NOT claim is that the backlog is skew-driven. It tests
    a CONSEQUENCE (is the backlog a first-order term in peak RSS?), not a
    mechanism, and the structural backlog above is a perfectly good way to get
    there. That is the honest reading of H3 for this pipeline: peak RSS is set
    by resident chunk bytes, and `chunk_bytes` spans ~1.5 MB to ~3 GB across
    the swept `S`, so no single static worker cap bounds memory even where the
    knee itself is flat. Which is why H1's evidence now rides along on an H3
    verdict: `knee_spread` of 0 next to a fired H3 reads "a static cap predicts
    the SPEED knee fine, but memory needs a byte budget", and that is a
    different -- and more useful -- statement than either verdict alone.
    `evidence["max_backlog_rss_share_row"]` names the row that fired it.

    A non-positive fitted `kappa` means the RAM law found no per-chunk term at
    all; the backlog demonstrably is not what sets RSS, so H3(a) correctly
    cannot fire. `ram_law=None` (fewer than two RAM rows) leaves H3(a)
    unevaluable rather than silently false; `evidence["max_backlog_rss_share"]`
    is `None` in that case so a verdict can be audited for it.
    """
    evidence: dict[str, object] = {
        "knees": dict(knees),
        "beta_read": read_law.beta,
        "beta_exec": exec_law.beta,
    }

    # Computed BEFORE any return so that EVERY verdict -- including both H3
    # paths -- ships the full evidence. An H3 verdict whose evidence carried
    # neither the knee spread nor the beta CI left a human unable to tell
    # whether H1 also held, which is the whole point of reporting evidence.
    diff_lo = read_law.beta_ci95[0] - exec_law.beta_ci95[1]
    diff_hi = read_law.beta_ci95[1] - exec_law.beta_ci95[0]
    evidence["beta_diff_ci95"] = (diff_lo, diff_hi)

    values = list(knees.values())
    spread = (max(values) - min(values)) if values else 0
    evidence["knee_spread"] = spread
    # Report the support alongside the spread: `knee_spread=0` from one cohort
    # size and from seven are the same number carrying opposite amounts of
    # evidence, and only this field distinguishes them.
    evidence["knee_points"] = len(values)
    h1_supported = len(values) >= H1_MIN_KNEE_POINTS

    # Reported, never gated on: `pending/w` is the count the spec named and is
    # still worth seeing, and `pending - (w - 1)` is that count against a LOWER
    # BOUND on its structural baseline (at least one buffered chunk per
    # non-head unit; the true baseline is `(w - 1) * chunks_per_unit`, which
    # these rows do not carry). Both are diagnostics for the byte share below.
    evidence["max_pending_fraction"] = max(
        (r.pending / r.workers for r in rows if r.workers > 0), default=0.0
    )
    evidence["max_pending_excess_over_structural"] = max(
        (r.pending - (r.workers - 1) for r in rows if r.workers > 0), default=0
    )

    kappa = max(ram_law.kappa, 0.0) if ram_law is not None else None
    evidence["ram_law_kappa"] = kappa
    max_backlog_share: float | None = None
    if kappa is not None:
        max_backlog_share = 0.0
        worst_row: RamRow | None = None
        for row in rows:
            p, cb, rss = row.pending, row.chunk_bytes, row.peak_rss_mb
            if rss <= 0 or p <= 0:
                continue
            share = kappa * p * cb / 1e6 / rss
            if share > max_backlog_share:
                max_backlog_share, worst_row = share, row
        # Name the row so a verdict can be traced back to one measurement
        # rather than to an aggregate; `RamRow` names its own fields.
        evidence["max_backlog_rss_share_row"] = worst_row
    evidence["max_backlog_rss_share"] = max_backlog_share
    if max_backlog_share is not None and max_backlog_share >= H3_BACKLOG_RSS_FRACTION:
        return Verdict(
            "H3",
            (
                f"the reorder backlog accounts for {max_backlog_share:.0%} of measured "
                f"peak RSS (gate: {H3_BACKLOG_RSS_FRACTION:.0%}), so in-flight bytes "
                "rather than worker count set peak RSS"
            ),
            evidence,
        )

    if contig_counterfactual is not None:
        a, b = contig_counterfactual
        delta = abs(a - b) / max(min(a, b), 1e-9)
        evidence["contig_partition_delta"] = delta
        if delta > 0.15:
            return Verdict(
                "H3",
                (
                    f"same total readers split differently across contigs differ by "
                    f"{delta:.0%} wall time: the multi-contig regression is a "
                    "partitioning artifact"
                ),
                evidence,
            )

    if h1_supported and spread <= H1_KNEE_TOLERANCE:
        return Verdict(
            "H1",
            (
                f"w* varies by {spread} across {len(values)} cohort sizes spanning "
                "the full sample range: a static cap suffices, no autotuner needed"
            ),
            evidence,
        )

    if diff_lo > 0 or diff_hi < 0:
        return Verdict(
            "H2",
            (
                f"95% CI of (beta_read - beta_exec) = ({diff_lo:.3f}, {diff_hi:.3f}) "
                "excludes zero: the cost ratio genuinely trends with cohort size"
            ),
            evidence,
        )

    h3_note = (
        f"the backlog is {max_backlog_share:.0%} of peak RSS, under the "
        f"{H3_BACKLOG_RSS_FRACTION:.0%} gate (so not H3)"
        if max_backlog_share is not None
        else "H3(a) could not be evaluated (no fitted RAM law)"
    )
    # Distinguish "the knee genuinely moves" from "too few cohort sizes
    # survived to tell". Both land on `none`, but only the first is a finding;
    # the second is a broken sweep and the operator needs to know which.
    h1_note = (
        f"w* spread is {spread} (> {H1_KNEE_TOLERANCE}, so not H1)"
        if h1_supported
        else (
            f"only {len(values)} cohort size(s) yielded a knee (need "
            f"{H1_MIN_KNEE_POINTS}), so w*'s flatness across the sample range is "
            f"not evaluable -- the observed spread of {spread} is unsupported"
        )
    )
    return Verdict(
        "none",
        (
            f"{h1_note} but the beta difference CI ({diff_lo:.3f}, {diff_hi:.3f}) "
            f"includes zero (so not H2), and {h3_note}. Collect more points."
        ),
        evidence,
    )


def extrapolate(
    v_law: VLaw,
    read_law: CostLaw,
    exec_law: CostLaw,
    ram_law: RamLaw,
    samples: int,
    variants: int,
    chunk_size: int,
    workers: int,
    format_fields: int,
    *,
    v_law_samples: int,
    cohort_beta: float,
    pending: int = 0,
) -> dict[str, float]:
    """Project wall and peak RSS at a target regime.

    `v_law_r2` and `extrapolation_factor` ride along on every projection: the
    V-law is fitted over an 8x range and stretched to 10^9, which is the
    least-supported step in the chain and must not read as better evidenced
    than it is.

    `v_law_samples` is the cohort size the V-ladder was actually measured at
    (the spec fixes it small, e.g. S=250 or S=1000, so billions of variants
    stay affordable to generate). `slope_s_per_variant` is therefore a
    per-variant parse cost measured AT THAT S, not at `samples` -- and
    per-record parse cost is roughly linear in sample count (2000x more
    genotype text per record at S=500,000 than at S=250). Applying the slope
    unscaled at a different S silently assumes cohort size doesn't matter,
    which is exactly the question this harness exists to answer. The
    per-variant term is therefore scaled by `(samples / v_law_samples) **
    cohort_beta`.

    `cohort_beta` MUST come from an ABSOLUTE cost measured against S -- the
    driver fits `phase1_s / variants ~ a * S**b` over the scale sweep's w=1
    rows (`fit_cost_law("cohort", ...)`). It deliberately is NOT
    `read_law.beta`: the cost laws are fitted on `cpu_shard_pct` /
    `cpu_exec_pct`, CPU UTILIZATION percentages bounded in (0, 100]. At w=1
    the bottleneck pegs near 100% at every S (conversion is reader-bound), so
    `c_read(S) ~ 100` for all S and `beta_read ~ 0` -- which made this
    correction evaluate to `400 ** 0 = 1` at S=500,000, i.e. exactly the
    "silently assumes cohort size doesn't matter" failure it exists to
    prevent. The RATIO `c_read/c_exec` is unaffected by that ceiling, so
    `read_law.beta` stays where it is legitimately used: `predicted_knee`
    below and the H2 test in `decide`.

    The intercept (`fill_drain`) is NOT scaled: it is pipeline start/drain
    overhead, not per-variant work, and nothing in this harness measures how
    it moves with S -- scaling it too would extrapolate past what's fitted.
    """
    grid = (samples * PLOIDY) // 8
    fmt = format_fields * samples * 4
    # Bytes for one FULL chunk (`chunk_size` variants), matching what
    # `fit_ram_law` was fitted against. `CorpusManifest.chunk_bytes` in
    # records.py (frozen) returns the PER-VARIANT figure despite its summary
    # line claiming otherwise -- trust its last line ("callers multiply by
    # chunk_size") and its formula, not its summary. `grid + fmt` here is that
    # same per-variant quantity; multiplying by `chunk_size` is that
    # multiplication, done once, in the one place both the RSS and knee
    # projections need it.
    #
    # No `min(chunk_size, variants)` here, unlike `_ram_rows`: this projection
    # is made at V = 10^9 (and, for the hold-out, at a V the caller has already
    # bounded the chunk against), where a chunk genuinely does fill.
    chunk_bytes = chunk_size * (grid + fmt)

    cohort_scale = (samples / v_law_samples) ** cohort_beta
    predicted_wall = (
        v_law.intercept_s + v_law.slope_s_per_variant * variants * cohort_scale
    )
    # `(workers + pending)`, matching the term `fit_ram_law` was actually
    # fitted against (model.py:fit_ram_law: `(w + p) * chunk_bytes`). Dropping
    # `pending` here would project only the in-flight term and silently
    # discard the reorder-skew term entirely -- exactly the term H3 is about.
    # The cohort term is NOT optional at biobank scale: it is the dominant
    # one. Measured at a pinned 10.9 MB chunk, RSS runs 789 MB (S=4,000) ->
    # 5,061 MB (S=500,000) with the chunk term held constant, so projecting to
    # S=500,000 without `per_sample_mb * samples` under-counts by GBs.
    predicted_rss = (
        ram_law.base_mb
        + ram_law.per_sample_mb * samples
        + ram_law.kappa * (workers + pending) * chunk_bytes / 1e6
    )
    return {
        "chunk_bytes": float(chunk_bytes),
        "cohort_scale": float(cohort_scale),
        # NOT named `predicted_wall_s`. The V-law is fitted `phase1_s ~ a +
        # b*V`, so this quantity is a PHASE-1 prediction and excludes the
        # reader-independent rayon merge tail and process startup that
        # `ProbeRecord.wall_s` includes. Under the old name the hold-out check
        # scored it against `wall_s`, which is strictly larger -- a
        # one-sided bias charged to the model as error, in a comparison gated
        # at 25%. The name is the fix: there is no correct way to compare this
        # against a wall time.
        "predicted_phase1_s": float(predicted_wall),
        "predicted_peak_rss_mb": float(predicted_rss),
        "predicted_knee": float(
            max(
                1,
                np.ceil(
                    (read_law.alpha * samples**read_law.beta)
                    / max(exec_law.alpha * samples**exec_law.beta, 1e-12)
                ),
            )
        ),
        "v_law_r2": v_law.r2,
        "v_law_ok": float(v_law.r2 >= V_LAW_MIN_R2),
        # variants / (the largest V the V-law was actually fitted against).
        # `VLaw` (records.py, frozen) doesn't carry that max V directly, only
        # `max_extrapolation_factor = _V_LAW_TARGET_VARIANTS / max(v)`
        # (fit_v_law above), so it's inverted here rather than dividing by
        # `n_points` -- a point COUNT, not a variant count, which is what the
        # previous version did and which is meaningless as an "extrapolation
        # factor" (a 4-point V-law extrapolating to 1e9 variants reported 2.5e8,
        # not the ~5000x the fitted range actually spans).
        "extrapolation_factor": float(
            variants * v_law.max_extrapolation_factor / _V_LAW_TARGET_VARIANTS
        ),
    }


# --- driver: results.ndjson + manifests + plans -> laws + verdict -----------
#
# Everything below is the entry point the spec's component table promises
# (`model.py`: `results.ndjson -> laws + verdict`) and that nothing else in
# the harness previously called. `fit_*`/`decide`/`extrapolate` above stay
# pure; this section is where NDJSON, manifests and plans get turned into the
# tuples those functions want, and where partial/missing data is reported
# instead of crashing -- the overnight sweep is preemptible, so a partial
# results file is the normal case, not an edge case.

# File-layout contract shared with the plan-generation and sbatch agents:
# `<results-dir>/<name>.ndjson`, `<plans-dir>/<name>.json`, for `name` in
# these four sweeps. `vlinear` is new (the V-linearity ladder); it may not
# exist yet in every job dir, and its absence must degrade gracefully.
_SWEEP_NAMES = ("scale", "contig", "holdout", "vlinear")


class _LoadedSweep:
    """One sweep's records, joined against its plan and resolved manifests.

    `records` holds only the subset that is `ok=True` AND resolvable to both
    a `SweepPoint` (via the plan) and a `CorpusManifest` (via `--manifests`).
    Everything dropped is named in `excluded` rather than silently vanishing.
    """

    def __init__(self) -> None:
        self.records: list[ProbeRecord] = []
        self.point_of: dict[str, SweepPoint] = {}
        self.manifest_of: dict[str, CorpusManifest] = {}
        self.excluded: list[str] = []


def _load_manifests(manifests_dir: Path) -> dict[str, CorpusManifest]:
    """Basename -> CorpusManifest.

    Keyed by filename, not by the path recorded on a `SweepPoint`: that path
    was written by whatever job generated the corpus and may point at a
    `$CLAUDE_JOB_DIR` that no longer exists, so `--manifests` is scanned
    independently and matched by name.
    """
    out: dict[str, CorpusManifest] = {}
    for p in sorted(Path(manifests_dir).glob("*.manifest.json")):
        try:
            out[p.name] = from_json(CorpusManifest, p.read_text())
        except Exception as e:  # noqa: BLE001 - report, don't crash the driver
            print(f"WARN: could not parse manifest {p}: {e}", file=sys.stderr)
    return out


def _load_plan_points(plans_dir: Path, name: str) -> dict[str, SweepPoint] | None:
    path = Path(plans_dir) / f"{name}.json"
    if not path.exists():
        return None
    from scripts.bench_svar2.sweep import (
        load_plan,
    )  # local: avoid import cost when unused

    return {pt.point_id: pt for pt in load_plan(path)}


def load_sweep(
    name: str,
    results_dir: Path,
    plans_dir: Path,
    manifests: dict[str, CorpusManifest],
) -> _LoadedSweep:
    """Read `<results_dir>/<name>.ndjson`, join against `<plans_dir>/<name>.json`
    and `manifests`. Never raises: every failure mode is recorded in
    `.excluded` and the caller decides whether the survivors are enough."""
    out = _LoadedSweep()
    results_path = Path(results_dir) / f"{name}.ndjson"
    if not results_path.exists():
        out.excluded.append(f"{name}: no results file at {results_path}")
        return out

    records = read_ndjson(results_path, ProbeRecord)
    if not records:
        out.excluded.append(f"{name}: {results_path} exists but has no records")
        return out

    points = _load_plan_points(plans_dir, name)
    if points is None:
        out.excluded.append(
            f"{name}: no plan at {Path(plans_dir) / f'{name}.json'}; "
            "cannot resolve corpus/workers/chunk_size for any of "
            f"{len(records)} record(s)"
        )
        return out

    for r in records:
        if not r.ok:
            reason = (
                f"oom at {r.oom_at_rss_mb:.0f}MB"
                if r.oom_at_rss_mb is not None
                else (r.error or "failed")
            )
            out.excluded.append(f"{name}/{r.point_id}: not ok ({reason})")
            continue
        pt = points.get(r.point_id)
        if pt is None:
            out.excluded.append(
                f"{name}/{r.point_id}: point_id not in {name}.json (plan changed?)"
            )
            continue
        m = manifests.get(Path(pt.corpus).name)
        if m is None:
            out.excluded.append(
                f"{name}/{r.point_id}: manifest {Path(pt.corpus).name!r} "
                "not found under --manifests"
            )
            continue
        out.records.append(r)
        out.point_of[r.point_id] = pt
        out.manifest_of[r.point_id] = m
    return out


def _v_law_points(vlinear: _LoadedSweep) -> list[tuple[float, float]]:
    """(V, phase1_s) -- one point per corpus in the V-linearity ladder."""
    return [
        (vlinear.manifest_of[r.point_id].variants, r.phase1_s) for r in vlinear.records
    ]


def _cost_and_knee_rows(
    scale: _LoadedSweep,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    dict[int, int],
    list[str],
]:
    """From the w=1 row at each S: (S, c_read), (S, c_exec), (S, per-variant
    wall), and the predicted knee. One run at w=1 is the spec's whole point --
    it replaces an O(|w|) sweep at every scale point.

    The third series is the ABSOLUTE cohort cost `phase1_s / variants` that
    `extrapolate`'s `cohort_beta` is fitted from. It has to be separate from
    `c_read`: `c_read`/`c_exec` are CPU utilization percentages, capped at
    100% per thread, so they cannot express "S=500,000 costs 400x more per
    record than S=250" -- they saturate instead (see `extrapolate`). Rows
    whose `phase1_s` is 0 (no per-contig span parsed out of the trace) are
    dropped from this series only: `log(0)` is not a data point. Every row
    here shares `concurrent_chroms=None`, so summing per-contig spans into
    `phase1_s` is comparable across them (see README).

    The scale plan can carry MORE THAN ONE w=1 point per S (e.g. one at
    `size_corpus`'s derived chunk size plus one at production's
    `PROD_CHUNK_SIZE`, for the "does the current default OOM" check). This
    keeps the FIRST w=1 row seen per S and reports the rest as excluded
    duplicates rather than letting a dict silently overwrite with whichever
    happened to be read last -- `sweep.py` appends in plan order, and the
    derived-chunk-size point is the one the plan builder puts first, so
    "first" is also "the one the spec's w=1-predicts-the-knee design means".
    """
    read_pts: dict[int, float] = {}
    exec_pts: dict[int, float] = {}
    cohort_pts: dict[int, float] = {}
    knees: dict[int, int] = {}
    excluded: list[str] = []
    for r in scale.records:
        pt = scale.point_of[r.point_id]
        if pt.reader_workers != 1:
            continue
        m = scale.manifest_of[r.point_id]
        if m.samples in read_pts:
            excluded.append(
                f"scale/{r.point_id}: extra w=1 row at S={m.samples} "
                f"(chunk_size={pt.chunk_size}), kept the first one seen for "
                "cost-law/knee fitting"
            )
            continue
        costs = _median_costs(r.cpu_shard_pct, r.cpu_exec_pct)
        if costs is None:
            excluded.append(
                f"scale/{r.point_id}: w=1 row at S={m.samples} has no usable "
                "cpu_shard/cpu_exec ticks, dropped from cost-law and knee fitting"
            )
            continue
        c_read, c_exec = costs
        read_pts[m.samples] = c_read
        exec_pts[m.samples] = c_exec
        knees[m.samples] = knee_from_probe(r.cpu_shard_pct, r.cpu_exec_pct)
        if r.phase1_s > 0 and m.variants > 0:
            cohort_pts[m.samples] = r.phase1_s / m.variants
        else:
            excluded.append(
                f"scale/{r.point_id}: w=1 row at S={m.samples} has phase1_s="
                f"{r.phase1_s:g} over {m.variants} variants, dropped from the "
                "cohort per-variant cost fit"
            )
    return (
        sorted(read_pts.items()),
        sorted(exec_pts.items()),
        sorted(cohort_pts.items()),
        knees,
        excluded,
    )


def _resident_chunk_size(chunk_size: int, variants: int) -> int:
    """Variants of a `chunk_size`-variant chunk that are actually RESIDENT.

    A chunk cannot hold more variants than the corpus has, and the part it
    never fills is never touched, so it never becomes resident:
    `BitGrid3::zeros` is `vec![0u64; n_words]` (src/types.rs) -> `alloc_zeroed`
    -> `calloc`, and calloc's untouched pages are served from the zero page.
    Verified empirically on this node: a 3 GB zeroed allocation adds 0 MB to
    `ru_maxrss`.

    This matters because `maxrss_mb` is the metric the harness records, and
    the sweep holds `cells = S * V` fixed, so its large-S corpora are tiny in
    V: at S=500,000 the corpus is 2,800 variants, and a nominal
    `chunk_size=25_000` chunk of 3,125 MB has ~350 MB of it touched. Feeding
    the NOMINAL 3,125 as the regressor put two enormous-leverage points with a
    local slope of ~0.3 into an OLS whose every other row sits under ~120,
    dragging `kappa` down roughly 10x from its true ~3 -- and with it the
    headline projection, which then reported `from_vcf`'s hardcoded
    `chunk_size=25_000` as SAFE at biobank scale (~1.3 GB instead of ~9.8 GB),
    the exact reverse of the design spec's arithmetic.

    (The address-space story is the opposite and unchanged: RLIMIT_AS counts
    the whole reservation, touched or not. See `probe.py:_preexec`.)
    """
    return min(chunk_size, variants)


def _ram_rows(*sweeps: _LoadedSweep) -> list[RamRow]:
    """Every resolved record across every sweep -- the RAM law is not scoped to
    one sweep the way the V-law and cost laws are.

    `samples` rides along because peak RSS carries a cohort-sized term
    independent of chunk bytes (see `RamLaw`); without it the fit has nowhere
    to put that cost but the intercept.
    """
    rows: list[RamRow] = []
    for sweep in sweeps:
        for r in sweep.records:
            pt = sweep.point_of[r.point_id]
            m = sweep.manifest_of[r.point_id]
            chunk_bytes = m.chunk_bytes * _resident_chunk_size(
                pt.chunk_size, m.variants
            )
            rows.append(
                RamRow(
                    workers=pt.reader_workers,
                    pending=r.pending_highwater,
                    chunk_bytes=chunk_bytes,
                    samples=m.samples,
                    peak_rss_mb=r.maxrss_mb,
                )
            )
    return rows


def _contig_counterfactual(contig: _LoadedSweep) -> tuple[float, float] | None:
    """Group contig-sweep records by (corpus, total reader threads) and
    return the pair of wall times with the largest spread within a group --
    same total readers, different split across contigs, is exactly the H3(b)
    counterfactual `decide` checks. Returns the worst-case pair so a single
    `decide()` call sees the strongest evidence available rather than an
    arbitrarily-chosen one."""
    groups: dict[tuple[str, int], list[float]] = {}
    for r in contig.records:
        pt = contig.point_of[r.point_id]
        if pt.concurrent_chroms is None:
            continue
        total = pt.reader_workers * pt.concurrent_chroms
        groups.setdefault((pt.corpus, total), []).append(r.wall_s)

    best: tuple[float, float] | None = None
    best_delta = -1.0
    for walls in groups.values():
        if len(walls) < 2:
            continue
        lo, hi = min(walls), max(walls)
        delta = (hi - lo) / max(lo, 1e-9)
        if delta > best_delta:
            best_delta, best = delta, (lo, hi)
    return best


def _print_law(label: str, law: CostLaw | VLaw | RamLaw | None) -> None:
    if law is None:
        print(f"{label}: SKIPPED (insufficient data)")
        return
    if isinstance(law, VLaw):
        print(
            f"{label}: phase1_s ~ {law.intercept_s:.4g} + "
            f"{law.slope_s_per_variant:.4g}*V  (R^2={law.r2:.4f}, n={law.n_points})"
        )
    elif isinstance(law, CostLaw):
        lo, hi = law.beta_ci95
        print(
            f"{label}: cost(S) ~ {law.alpha:.4g} * S^{law.beta:.4f}  "
            f"(95% CI beta=[{lo:.4f}, {hi:.4f}], n={law.n_points})"
        )
    else:
        print(
            f"{label}: peak_rss_mb ~ {law.base_mb:.4g} + "
            f"{law.per_sample_mb:.4g}*samples + "
            f"{law.kappa:.4g}*(w+pending)*chunk_bytes  (R^2={law.r2:.4f}, n={law.n_points})"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fit the SVAR2 scale-bench laws and print the H1/H2/H3 verdict."
    )
    p.add_argument("--results", type=Path, required=True, help="dir of <name>.ndjson")
    p.add_argument(
        "--manifests", type=Path, required=True, help="dir of *.manifest.json"
    )
    p.add_argument("--plans", type=Path, required=True, help="dir of <name>.json plans")
    p.add_argument("--target-samples", type=int, default=500_000)
    p.add_argument("--target-variants", type=int, default=1_000_000_000)
    p.add_argument("--target-chunk-size", type=int, default=25_000)
    p.add_argument("--target-workers", type=int, default=1)
    p.add_argument("--target-format-fields", type=int, default=3)
    a = p.parse_args()

    manifests = _load_manifests(a.manifests)
    print(f"loaded {len(manifests)} manifest(s) from {a.manifests}")

    sweeps = {
        name: load_sweep(name, a.results, a.plans, manifests) for name in _SWEEP_NAMES
    }
    excluded = [msg for s in sweeps.values() for msg in s.excluded]
    for name, s in sweeps.items():
        print(f"{name}: {len(s.records)} usable record(s)")
    if excluded:
        print(f"\n{len(excluded)} exclusion(s):")
        for msg in excluded:
            print(f"  - {msg}")

    scale, contig, holdout, vlinear = (sweeps[n] for n in _SWEEP_NAMES)

    print()
    v_law: VLaw | None = None
    v_law_samples: int | None = None
    v_points = _v_law_points(vlinear)
    if not vlinear.records:
        print("V-law: SKIPPED (vlinear.ndjson/plan absent or empty)")
    elif len(v_points) < 2:
        print(f"V-law: SKIPPED (only {len(v_points)} usable point(s), need >= 2)")
    else:
        v_law = fit_v_law(v_points)
        v_law_samples = vlinear.manifest_of[vlinear.records[0].point_id].samples
        _print_law("V-law", v_law)
        if v_law.r2 < V_LAW_MIN_R2:
            print(
                f"  WARNING: R^2={v_law.r2:.4f} < {V_LAW_MIN_R2} -- V-linearity "
                "failed, every downstream extrapolation is INVALID"
            )

    read_points, exec_points, cohort_points, knees, cost_excluded = _cost_and_knee_rows(
        scale
    )
    for msg in cost_excluded:
        print(f"  - {msg}")
    read_law = (
        fit_cost_law("read", *zip(*read_points)) if len(read_points) >= 2 else None
    )
    exec_law = (
        fit_cost_law("exec", *zip(*exec_points)) if len(exec_points) >= 2 else None
    )
    # `phase1_s / variants` against S -- an ABSOLUTE per-variant cost, which is
    # what `extrapolate` scales its per-variant term by. NOT `read_law.beta`;
    # see `extrapolate`'s docstring for why a utilization exponent collapses
    # that correction to a no-op.
    cohort_law = (
        fit_cost_law("cohort", *zip(*cohort_points))
        if len(cohort_points) >= 2
        else None
    )
    _print_law("read cost law", read_law)
    _print_law("exec cost law", exec_law)
    _print_law("cohort per-variant cost law", cohort_law)
    if knees:
        print(f"predicted knees by S: {knees}")

    ram_rows = _ram_rows(scale, contig, holdout, vlinear)
    ram_law = fit_ram_law(ram_rows) if len(ram_rows) >= 2 else None
    _print_law("RAM law", ram_law)

    print()
    if read_law is not None and exec_law is not None and ram_rows:
        verdict = decide(
            knees,
            read_law,
            exec_law,
            ram_rows,
            ram_law,
            contig_counterfactual=_contig_counterfactual(contig),
        )
        print(f"VERDICT: {verdict.hypothesis}")
        print(f"  {verdict.rationale}")
        print(f"  evidence: {verdict.evidence}")
    else:
        print(
            "VERDICT: SKIPPED (need a fitted read law, exec law, and >=1 RAM "
            "row; see exclusions above)"
        )

    print()
    if (
        v_law is None
        or read_law is None
        or exec_law is None
        or ram_law is None
        or cohort_law is None
    ):
        print(
            "EXTRAPOLATION: SKIPPED (missing one of V-law/read/exec/cohort/RAM "
            "law). The cohort law in particular is not optional: without it the "
            "projection would have to assume cohort size does not move "
            "per-variant cost, which is the question the sweep exists to answer."
        )
        return
    assert v_law_samples is not None

    # Assume the worst backlog fraction actually observed anywhere in this
    # sweep persists at the target scale -- there is no fitted law for
    # `pending` itself, only for peak RSS given `pending`, so this is the most
    # defensible number available rather than the previous silent 0.
    frac = max((r.pending / r.workers for r in ram_rows if r.workers > 0), default=0.0)
    target_pending = round(frac * a.target_workers)
    proj = extrapolate(
        v_law,
        read_law,
        exec_law,
        ram_law,
        samples=a.target_samples,
        variants=a.target_variants,
        chunk_size=a.target_chunk_size,
        workers=a.target_workers,
        format_fields=a.target_format_fields,
        v_law_samples=v_law_samples,
        cohort_beta=cohort_law.beta,
        pending=target_pending,
    )
    print(
        f"EXTRAPOLATION to S={a.target_samples:,} V={a.target_variants:,} "
        f"chunk_size={a.target_chunk_size:,} workers={a.target_workers} "
        f"(assumed pending={target_pending}, from worst observed fraction {frac:.2f}):"
    )
    for k, v in proj.items():
        print(f"  {k}: {v:,.4g}")
    if not proj["v_law_ok"]:
        print(
            f"  WARNING: extrapolation_factor={proj['extrapolation_factor']:.3g}x "
            "rides on a V-law that failed its own R^2 gate -- do not trust this number"
        )

    print()
    if not holdout.records:
        print("HOLD-OUT CHECK: SKIPPED (no holdout records)")
        return
    # Which FORMAT-field counts the laws were actually FITTED from. The cost
    # laws have no F term at all -- `extrapolate` threads `format_fields` into
    # chunk_bytes for the RSS side only -- so a hold-out at an F the fit never
    # saw is testing an axis the model does not model. Scoring that as "MODEL
    # FAILURE" blames the extrapolation for a dimension nobody fitted; saying
    # nothing would be worse. Name it.
    fitted_f = {
        len(m.format_fields)
        for sweep in (scale, contig, vlinear)
        for m in sweep.manifest_of.values()
    }
    for r in holdout.records:
        pt = holdout.point_of[r.point_id]
        m = holdout.manifest_of[r.point_id]
        pred = extrapolate(
            v_law,
            read_law,
            exec_law,
            ram_law,
            samples=m.samples,
            variants=m.variants,
            # Same touched-prefix bound the RAM law was FITTED under
            # (`_ram_rows` / `_resident_chunk_size`); a hold-out predicted from
            # a nominal chunk the corpus is too small to fill would be scored
            # against a regressor the fit never saw.
            chunk_size=_resident_chunk_size(pt.chunk_size, m.variants),
            workers=pt.reader_workers,
            format_fields=len(m.format_fields),
            v_law_samples=v_law_samples,
            cohort_beta=cohort_law.beta,
            pending=r.pending_highwater,
        )
        # Score the phase-1 prediction against MEASURED phase-1, not against
        # `wall_s`. `wall_s` also carries the rayon merge tail and process
        # startup, neither of which the V-law models, so scoring against it
        # adds a strictly positive term to every error -- a one-sided bias
        # into a 25% gate that would eventually read as "the model is
        # invalid" when the only thing wrong was the comparison. When the
        # trace yielded no per-contig span there is nothing commensurable to
        # compare against, so the wall half of the gate is skipped and said
        # to be skipped, rather than silently falling back to `wall_s`.
        have_phase1 = r.phase1_s > 0
        phase1_err = (
            abs(pred["predicted_phase1_s"] - r.phase1_s) / r.phase1_s
            if have_phase1
            else None
        )
        rss_err = abs(pred["predicted_peak_rss_mb"] - r.maxrss_mb) / max(
            r.maxrss_mb, 1e-9
        )
        phase1_txt = (
            f"phase1 pred={pred['predicted_phase1_s']:.1f}s "
            f"actual={r.phase1_s:.1f}s err={phase1_err:.0%}"
            if phase1_err is not None
            else (
                f"phase1 pred={pred['predicted_phase1_s']:.1f}s actual=n/a "
                "(no per-contig span in trace; NOT scored against wall_s)"
            )
        )
        print(
            f"HOLD-OUT {r.point_id} (S={m.samples}, V={m.variants}, "
            f"F={len(m.format_fields)}): {phase1_txt} | "
            f"rss pred={pred['predicted_peak_rss_mb']:.0f}MB actual={r.maxrss_mb:.0f}MB "
            f"err={rss_err:.0%}"
        )
        over_gate = (
            phase1_err is not None and phase1_err > HOLDOUT_ERROR_GATE
        ) or rss_err > HOLDOUT_ERROR_GATE
        holdout_f = len(m.format_fields)
        if holdout_f not in fitted_f:
            # Out of the fitted domain: report loudly, but do not call it a
            # failure of the S,V extrapolation the gate exists to validate.
            print(
                f"  OUT-OF-DOMAIN: hold-out has F={holdout_f} but every law was "
                f"fitted on F in {sorted(fitted_f)}. No cost law carries an F "
                "term, so this point cannot validate the S,V extrapolation "
                "either way -- the error above is dominated by an axis the "
                "model does not model. Fit an F law, or hold out at an F the "
                "laws were fitted on."
            )
        elif over_gate:
            print(
                f"  MODEL FAILURE: error exceeds the {HOLDOUT_ERROR_GATE:.0%} gate "
                "(spec: this invalidates the model, not just this point)"
            )


if __name__ == "__main__":
    main()
