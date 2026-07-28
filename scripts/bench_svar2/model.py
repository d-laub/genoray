"""Fit the three scaling laws and return a falsifiable hypothesis verdict.

Pure: every function takes numbers and returns dataclasses. That is what makes
the verdict testable against planted synthetic laws rather than only against
cluster runs.

No scipy. With 5-7 scale points a normal approximation to the 95% CI is
materially wrong (t(5)=2.571 vs z=1.96), so a small Student-t table is
inlined instead of taking a dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from scripts.bench_svar2.records import CostLaw, RamLaw, VLaw, Verdict

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
H3_PENDING_FRACTION = 0.5  # pending_hw >= workers/2 makes bytes the invariant
V_LAW_MIN_R2 = 0.98


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
        stderr = float(np.sqrt(ss_res / (n - 2) / sxx)) if sxx > 0 else 0.0
    else:
        stderr = 0.0
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
        max_extrapolation_factor=1e9 / max(v),
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


def fit_ram_law(rows: Sequence[tuple[int, int, int, float]]) -> RamLaw:
    """rows: (workers, pending_highwater, chunk_bytes, peak_rss_mb).

    peak_rss ~ base + kappa * (workers + pending_hw) * chunk_bytes. kappa is the
    observed overhead multiple over the analytic chunk size: a DenseChunk holds
    more than its packed grid.
    """
    x = [(w + p) * cb / 1e6 for (w, p, cb, _) in rows]
    y = [r[3] for r in rows]
    kappa, base, r2, _ = _linfit(x, y)
    return RamLaw(base_mb=base, kappa=kappa, r2=r2, n_points=len(rows))


def knee_from_probe(
    cpu_shard_pct: Sequence[float], cpu_exec_pct: Sequence[float]
) -> int:
    """Predicted knee from a single w=1 run.

    At w=1 the shard aggregate is one reader's cost and cpu_exec is the
    executor's, so their ratio is how many readers it takes to saturate the
    serial executor. Ticks where either is zero are startup/teardown and are
    dropped.
    """
    pairs = [(s, e) for s, e in zip(cpu_shard_pct, cpu_exec_pct) if s > 0 and e > 0]
    if not pairs:
        return 1
    shard = float(np.median([p[0] for p in pairs]))
    exec_ = float(np.median([p[1] for p in pairs]))
    if exec_ <= 0:
        return 1
    return max(1, int(np.ceil(shard / exec_)))


def decide(
    knees: dict[int, int],
    read_law: CostLaw,
    exec_law: CostLaw,
    rows: Sequence[tuple[int, int, int, float]],
    contig_counterfactual: tuple[float, float] | None = None,
) -> Verdict:
    """Apply the spec's falsifiable criteria, in H3-first order.

    H3 supersedes H1/H2: if bytes rather than worker count set peak RSS, or if
    the multi-contig regression is a partitioning artifact, a byte-bounded
    global pool needs no knee prediction at all.
    """
    evidence: dict[str, object] = {
        "knees": dict(knees),
        "beta_read": read_law.beta,
        "beta_exec": exec_law.beta,
    }

    max_pending_frac = max((p / w for (w, p, _, _) in rows if w > 0), default=0.0)
    evidence["max_pending_fraction"] = max_pending_frac
    if max_pending_frac >= H3_PENDING_FRACTION:
        return Verdict(
            "H3",
            (
                f"reorder backlog reached {max_pending_frac:.2f} x workers, so in-flight "
                "bytes rather than worker count set peak RSS"
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

    diff_lo = read_law.beta_ci95[0] - exec_law.beta_ci95[1]
    diff_hi = read_law.beta_ci95[1] - exec_law.beta_ci95[0]
    evidence["beta_diff_ci95"] = (diff_lo, diff_hi)

    values = list(knees.values())
    spread = (max(values) - min(values)) if values else 0
    evidence["knee_spread"] = spread

    if spread <= H1_KNEE_TOLERANCE:
        return Verdict(
            "H1",
            (
                f"w* varies by {spread} across the full sample range: a static cap "
                "suffices, no autotuner needed"
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

    return Verdict(
        "none",
        (
            f"w* spread is {spread} (> {H1_KNEE_TOLERANCE}, so not H1) but the "
            f"beta difference CI ({diff_lo:.3f}, {diff_hi:.3f}) includes zero (so not "
            "H2), and the backlog is immaterial (so not H3). Collect more points."
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
) -> dict[str, float]:
    """Project wall and peak RSS at a target regime.

    `v_law_r2` and `extrapolation_factor` ride along on every projection: the
    V-law is fitted over an 8x range and stretched to 10^9, which is the
    least-supported step in the chain and must not read as better evidenced
    than it is.
    """
    grid = (samples * PLOIDY) // 8
    fmt = format_fields * samples * 4
    chunk_bytes = chunk_size * (grid + fmt)
    predicted_wall = v_law.intercept_s + v_law.slope_s_per_variant * variants
    predicted_rss = ram_law.base_mb + ram_law.kappa * workers * chunk_bytes / 1e6
    return {
        "chunk_bytes": float(chunk_bytes),
        "predicted_wall_s": float(predicted_wall),
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
        "extrapolation_factor": variants / max(v_law.n_points, 1),
    }
