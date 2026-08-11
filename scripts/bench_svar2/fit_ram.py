"""Fit and gate one backend's `RamLaw` from a sweep's results.

The PGEN law was fitted ad hoc, which is why its refits are hard to reproduce
and why a contaminated dataset once reached a shipped constant. One entry point
means the fit that is quoted in a doc comment is the fit anyone can re-run.

    python -m scripts.bench_svar2.fit_ram \\
        --results $SCRATCH/out --plans $SCRATCH/plans \\
        --manifests $SCRATCH/corpora --backend vcf
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts.bench_svar2.model import (  # `_ram_rows` is private to the package,
    _load_manifests,  # not to this module -- fit_ram is model.py's CLI face.
    _ram_rows,
    RamRow,  # NB: RamRow lives in model.py; only RamLaw is in records.py.
    fit_ram_law,
    load_sweep,
)
from scripts.bench_svar2.records import RamLaw

# Which sweep families carry each backend's RAM rows. `concurrency` and
# `vcf_ram` are the families that vary `concurrent_chroms` at fixed
# (S, chunk_size) and so are the only ones that identify `per_contig_mb`.
BACKEND_SWEEPS: dict[str, tuple[str, ...]] = {
    "vcf": (
        "scale",
        "contig",
        "holdout",
        "vlinear",
        "vlinear2",
        "concurrency",
        "vcf_ram",
    ),
    "pgen": ("pgen",),
}


def predict_mb(law: RamLaw, row: RamRow) -> float:
    """Exactly what `budget.rs:plan_sharded` computes for this row.

    Mirrored here rather than approximated: a gate that scores the law against
    a DIFFERENT equation than the consumer evaluates is how a cc-blind fit
    passed review in the first place (issue #158).
    """
    bracket = law.per_contig_mb + law.kappa * (
        (row.workers + row.pending) * row.chunk_bytes / 1e6
    )
    return (
        law.base_mb + law.per_sample_mb * row.samples + row.concurrent_chroms * bracket
    )


def gate_report(law: RamLaw, rows: Sequence[RamRow]) -> dict:
    """Over-prediction ratios for every row. `passes` is the shipping gate."""
    ratios = [predict_mb(law, r) / r.peak_rss_mb for r in rows]
    under = [r for r, ratio in zip(rows, ratios) if ratio < 1.0]
    return {
        "n": len(rows),
        "passes": not under,
        "n_under": len(under),
        "worst_ratio": max(ratios),
        "mean_ratio": sum(ratios) / len(ratios),
        "min_ratio": min(ratios),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True, help="dir of <name>.ndjson")
    p.add_argument("--plans", type=Path, required=True, help="dir of <name>.json")
    p.add_argument(
        "--manifests", type=Path, required=True, help="dir of *.manifest.json"
    )
    p.add_argument("--backend", choices=sorted(BACKEND_SWEEPS), required=True)
    p.add_argument(
        "--margin",
        type=float,
        default=1.25,
        help="safety factor: every point must be over-predicted by at least this",
    )
    p.add_argument(
        "--interaction",
        action="store_true",
        help="fit the optional per_contig_per_sample term (off by default)",
    )
    a = p.parse_args()

    manifests = _load_manifests(a.manifests)
    sweeps = [
        load_sweep(name, a.results, a.plans, manifests)
        for name in BACKEND_SWEEPS[a.backend]
    ]
    for name, s in zip(BACKEND_SWEEPS[a.backend], sweeps):
        print(f"{name}: {len(s.records)} usable record(s)")
        for msg in s.excluded:
            print(f"  - {msg}")

    rows = _ram_rows(*sweeps)
    print(f"\nfittable rows (cc observed): {len(rows)}")
    if len(rows) < 2:
        raise SystemExit("ABORT: need >= 2 rows with an observed concurrent_chroms")

    law = fit_ram_law(rows, margin=a.margin, interaction=a.interaction)
    report = gate_report(law, rows)

    print(f"\nRamLaw::{a.backend.upper()} (margin {a.margin}):")
    print(f"    base_mb:       {law.base_mb!r},")
    print(f"    per_sample_mb: {law.per_sample_mb!r},")
    print(f"    per_contig_mb: {law.per_contig_mb!r},")
    print(f"    kappa:         {law.kappa!r},")
    print(
        f"\ngate: n={report['n']} passes={report['passes']} "
        f"under={report['n_under']} worst={report['worst_ratio']:.4f}x "
        f"mean={report['mean_ratio']:.4f}x min={report['min_ratio']:.4f}x"
    )
    print(f"(descriptive only, NOT the criterion) r2={law.r2:.4f}")
    if not report["passes"]:
        raise SystemExit(
            f"ABORT: law under-predicts {report['n_under']} point(s). A law is a "
            "BOUND -- do not ship this."
        )


if __name__ == "__main__":
    main()
