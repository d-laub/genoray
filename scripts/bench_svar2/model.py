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
H3_PENDING_FRACTION = 0.5  # pending_hw >= workers/2 makes bytes the invariant
V_LAW_MIN_R2 = 0.98
HOLDOUT_ERROR_GATE = 0.25  # spec: >25% predicted-vs-actual error is a model failure
# fit_v_law's max_extrapolation_factor is always stated relative to this many
# variants (the biobank target). Keeping it as one constant lets `extrapolate`
# invert it exactly instead of re-guessing what "1e9" in records.py means.
_V_LAW_TARGET_VARIANTS = 1e9
# Backlog counted by `pending_highwater` that is a structural artifact of the
# instrumentation rather than evidence of reorder skew. `decide`'s H3 gate
# judges backlog BEYOND this floor.
#
# This is 0 because the collector's gauge samples BEFORE inserting the
# just-read chunk (`PendingBacklog::insert_observing`, shard_exec.rs), so a
# chunk released the instant it arrives contributes nothing and
# `pending_highwater == 0` genuinely means "no backlog ever observed".
#
# It is kept as a named constant rather than deleted because the gauge USED to
# sample after the insert, which floored every sharded run at 1 and made the H3
# gate fire on the floor alone -- a reviewer planted synthetic H1 and H2 data
# and got H3 both times. If the instrumentation ever regains a floor, this is
# the one place that has to change, and `decide` already reports the value it
# used in `evidence["pending_structural_floor"]` so a verdict can be audited
# against the instrumentation that produced it.
_PENDING_STRUCTURAL_FLOOR = 0


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

    # Subtract the gauge's structural floor (see `_PENDING_STRUCTURAL_FLOOR`)
    # before judging materiality: a single always-present in-flight chunk is
    # not backlog, and treating it as such made this gate fire on the very
    # first row of every sweep regardless of what the row actually measured.
    max_pending_frac = max(
        (max(0, p - _PENDING_STRUCTURAL_FLOOR) / w for (w, p, _, _) in rows if w > 0),
        default=0.0,
    )
    evidence["max_pending_fraction"] = max_pending_frac
    evidence["pending_structural_floor"] = _PENDING_STRUCTURAL_FLOOR
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
    *,
    v_law_samples: int,
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
    which is exactly the question this harness exists to answer. Instead the
    per-variant term is scaled by the fitted read-cost law's exponent,
    `(samples / v_law_samples) ** read_law.beta` -- the same cost law that
    sets `predicted_knee` below, so the correction is consistent with the
    rest of the projection rather than a separately-invented number. The
    intercept (`fill_drain`) is NOT scaled: it is pipeline start/drain
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
    chunk_bytes = chunk_size * (grid + fmt)

    cohort_scale = (samples / v_law_samples) ** read_law.beta
    predicted_wall = (
        v_law.intercept_s + v_law.slope_s_per_variant * variants * cohort_scale
    )
    # `(workers + pending)`, matching the term `fit_ram_law` was actually
    # fitted against (model.py:fit_ram_law: `(w + p) * chunk_bytes`). Dropping
    # `pending` here would project only the in-flight term and silently
    # discard the reorder-skew term entirely -- exactly the term H3 is about.
    predicted_rss = (
        ram_law.base_mb + ram_law.kappa * (workers + pending) * chunk_bytes / 1e6
    )
    return {
        "chunk_bytes": float(chunk_bytes),
        "cohort_scale": float(cohort_scale),
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
    list[tuple[float, float]], list[tuple[float, float]], dict[int, int], list[str]
]:
    """From the w=1 row at each S: (S, c_read), (S, c_exec), and the
    predicted knee. One run at w=1 is the spec's whole point -- it replaces
    an O(|w|) sweep at every scale point.

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
    return (
        sorted(read_pts.items()),
        sorted(exec_pts.items()),
        knees,
        excluded,
    )


def _ram_rows(*sweeps: _LoadedSweep) -> list[tuple[int, int, int, float]]:
    """(workers, pending_highwater, chunk_bytes, peak_rss_mb) from every
    resolved record across every sweep -- the RAM law is not scoped to one
    sweep the way the V-law and cost laws are."""
    rows: list[tuple[int, int, int, float]] = []
    for sweep in sweeps:
        for r in sweep.records:
            pt = sweep.point_of[r.point_id]
            m = sweep.manifest_of[r.point_id]
            chunk_bytes = m.chunk_bytes * pt.chunk_size
            rows.append(
                (pt.reader_workers, r.pending_highwater, chunk_bytes, r.maxrss_mb)
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

    read_points, exec_points, knees, cost_excluded = _cost_and_knee_rows(scale)
    for msg in cost_excluded:
        print(f"  - {msg}")
    read_law = (
        fit_cost_law("read", *zip(*read_points)) if len(read_points) >= 2 else None
    )
    exec_law = (
        fit_cost_law("exec", *zip(*exec_points)) if len(exec_points) >= 2 else None
    )
    _print_law("read cost law", read_law)
    _print_law("exec cost law", exec_law)
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
    if v_law is None or read_law is None or exec_law is None or ram_law is None:
        print("EXTRAPOLATION: SKIPPED (missing one of V-law/read/exec/RAM law)")
        return
    assert v_law_samples is not None

    # Assume the worst backlog fraction actually observed anywhere in this
    # sweep persists at the target scale -- there is no fitted law for
    # `pending` itself, only for peak RSS given `pending`, so this is the most
    # defensible number available rather than the previous silent 0.
    frac = max(
        (
            max(0, p - _PENDING_STRUCTURAL_FLOOR) / w
            for (w, p, _, _) in ram_rows
            if w > 0
        ),
        default=0.0,
    )
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
            chunk_size=pt.chunk_size,
            workers=pt.reader_workers,
            format_fields=len(m.format_fields),
            v_law_samples=v_law_samples,
            pending=r.pending_highwater,
        )
        wall_err = abs(pred["predicted_wall_s"] - r.wall_s) / max(r.wall_s, 1e-9)
        rss_err = abs(pred["predicted_peak_rss_mb"] - r.maxrss_mb) / max(
            r.maxrss_mb, 1e-9
        )
        print(
            f"HOLD-OUT {r.point_id} (S={m.samples}, V={m.variants}, "
            f"F={len(m.format_fields)}): "
            f"wall pred={pred['predicted_wall_s']:.1f}s actual={r.wall_s:.1f}s "
            f"err={wall_err:.0%} | "
            f"rss pred={pred['predicted_peak_rss_mb']:.0f}MB actual={r.maxrss_mb:.0f}MB "
            f"err={rss_err:.0%}"
        )
        if wall_err > HOLDOUT_ERROR_GATE or rss_err > HOLDOUT_ERROR_GATE:
            print(
                f"  MODEL FAILURE: error exceeds the {HOLDOUT_ERROR_GATE:.0%} gate "
                "(spec: this invalidates the model, not just this point)"
            )


if __name__ == "__main__":
    main()
