"""Calibration: cross-check genoray.fit_signatures against SigProfilerAssignment.

Skips automatically when SigProfilerAssignment is not installed. Run in the
dedicated env:
    pixi run -e sigprofiler pytest tests/test_signatures_calibration.py -v
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("SigProfilerAssignment")

from genoray import cosmic_signatures, fit_signatures  # noqa: E402

pytestmark = [pytest.mark.sigprofiler, pytest.mark.network]


def test_fit_matches_spa_on_synthetic_sbs96(tmp_path):
    # Build a catalogue from a known mix of two real COSMIC signatures.
    ref = cosmic_signatures("SBS96")
    sig_cols = [c for c in ref.columns if c != "MutationType"]
    # pick two well-known signatures present in COSMIC v3.4
    a, b = "SBS1", "SBS5"
    assert a in sig_cols and b in sig_cols
    W = ref.select([a, b]).to_numpy()
    h_true = np.array([300.0, 700.0])
    counts = np.rint(W @ h_true).astype(np.int64)

    catalogue = pl.DataFrame({"MutationType": ref["MutationType"], "sample1": counts})

    # genoray refit
    act = fit_signatures(catalogue, ref)
    g_a = act.filter(pl.col("Sample") == "sample1")[a].item()
    g_b = act.filter(pl.col("Sample") == "sample1")[b].item()

    # SigProfilerAssignment refit on the same matrix
    from SigProfilerAssignment import Analyzer as spa  # noqa: N813

    matrix_path = tmp_path / "samples.txt"
    catalogue.write_csv(matrix_path, separator="\t")
    out_dir = tmp_path / "spa_out"
    spa.cosmic_fit(
        samples=str(matrix_path),
        output=str(out_dir),
        input_type="matrix",
        cosmic_version=3.4,
        genome_build="GRCh38",
        collapse_to_SBS96=True,
        make_plots=False,
        verbose=False,
    )

    # SPA writes: <out_dir>/SBS96/Assignment_Solution/Activities/Assignment_Solution_Activities.txt
    # Fall back to a broader glob if the directory structure differs.
    candidates = list(out_dir.rglob("Assignment_Solution_Activities.txt"))
    if not candidates:
        candidates = list(out_dir.rglob("*Activities*.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"SPA activities file not found under {out_dir}. "
            f"Contents: {list(out_dir.rglob('*'))[:30]}"
        )
    spa_act = pl.read_csv(candidates[0], separator="\t")

    # SPA activities: first column is sample name, remaining columns are signatures.
    sample_col = spa_act.columns[0]
    spa_row = spa_act.filter(pl.col(sample_col) == "sample1")
    spa_a = float(spa_row[a].item()) if a in spa_act.columns else 0.0
    spa_b = float(spa_row[b].item()) if b in spa_act.columns else 0.0

    print(f"\ngenoray activities: {a}={g_a:.1f}, {b}={g_b:.1f}")
    print(f"SPA activities:     {a}={spa_a:.1f}, {b}={spa_b:.1f}")

    # Activities should agree within a modest tolerance (both fit the same data).
    total_g = g_a + g_b
    total_spa = spa_a + spa_b
    assert total_g > 0, "genoray assigned zero activity to both SBS1 and SBS5"
    assert total_spa > 0, "SPA assigned zero activity to both SBS1 and SBS5"
    assert g_a / total_g == pytest.approx(spa_a / total_spa, abs=0.1)
    assert g_b / total_g == pytest.approx(spa_b / total_spa, abs=0.1)


def _spa_activities(tmp_path, catalogue: pl.DataFrame, tag: str) -> dict[str, float]:
    """Run real SigProfilerAssignment cosmic_fit, return {signature: activity}."""
    from SigProfilerAssignment import Analyzer as spa  # noqa: N813

    matrix_path = tmp_path / f"{tag}.txt"
    catalogue.write_csv(matrix_path, separator="\t")
    out_dir = tmp_path / f"spa_out_{tag}"
    spa.cosmic_fit(
        samples=str(matrix_path),
        output=str(out_dir),
        input_type="matrix",
        cosmic_version=3.4,
        genome_build="GRCh38",
        collapse_to_SBS96=True,
        make_plots=False,
        verbose=False,
    )
    candidates = list(out_dir.rglob("Assignment_Solution_Activities.txt"))
    if not candidates:
        candidates = list(out_dir.rglob("*Activities*.txt"))
    if not candidates:
        raise FileNotFoundError(f"SPA activities not found under {out_dir}")
    act = pl.read_csv(candidates[0], separator="\t")
    sample_col = act.columns[0]
    row = act.filter(pl.col(sample_col) == "sample1")
    return {
        c: float(row[c].item())
        for c in act.columns
        if c != sample_col and float(row[c].item()) != 0.0
    }


@pytest.mark.parametrize(
    ("truth", "burden", "tag", "seed"),
    [
        (["SBS1", "SBS2", "SBS7a", "SBS17b"], 100_000, "distinct_high", 11),
        (["SBS1", "SBS2", "SBS7a", "SBS17b"], 1_000, "distinct_low", 22),
        (["SBS1", "SBS5", "SBS40a", "SBS2"], 100_000, "flat_high", 33),
    ],
)
def test_spa_strategy_reproduces_cosmic_fit(tmp_path, truth, burden, tag, seed):
    """strategy=Spa() must agree with real cosmic_fit on support and activities.

    This is the test that earns the word "faithful". If it fails, the
    reimplementation is wrong -- do not loosen the tolerance to make it pass.
    """
    from genoray import Spa

    ref = cosmic_signatures("SBS96")
    # An explicit seed, not hash(tag): str hashing is PYTHONHASHSEED-randomized,
    # and a test that claims faithfulness has to draw the same data every run.
    rng = np.random.default_rng(seed)
    W = ref.select(truth).to_numpy()
    W = W / W.sum(axis=0)
    h = rng.dirichlet(np.ones(len(truth))) * burden
    counts = rng.poisson(W @ h).astype(np.int64)
    catalogue = pl.DataFrame({"MutationType": ref["MutationType"], "sample1": counts})

    spa_act = _spa_activities(tmp_path, catalogue, tag)

    out = fit_signatures(catalogue, ref, strategy=Spa())
    sig_cols = [c for c in out.columns if c not in ("Sample", "cosine_similarity")]
    gen_act = {c: out[c].item() for c in sig_cols if out[c].item() != 0.0}

    assert set(gen_act) == set(spa_act), (
        f"support mismatch for {tag}\n"
        f"  genoray only: {sorted(set(gen_act) - set(spa_act))}\n"
        f"  SPA only:     {sorted(set(spa_act) - set(gen_act))}"
    )
    for sig in sorted(spa_act):
        assert gen_act[sig] == pytest.approx(spa_act[sig], abs=1.0), (
            f"{tag}: {sig} genoray={gen_act[sig]} spa={spa_act[sig]}"
        )
