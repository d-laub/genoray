from pathlib import Path

import numpy as np
import polars as pl
import pytest

from genoray._signatures import (
    _cosine,
    _fit_one,
    _load_signature_file,
    _nnls,
    cosmic_signatures,
    fit_signatures,
)

DATA = Path(__file__).parent / "data"


def test_cosine_identical_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, b) == 0.0


def test_nnls_recovers_nonnegative_solution():
    # W h = m with a known nonnegative h
    W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    h_true = np.array([3.0, 5.0])
    m = W @ h_true
    h = _nnls(W, m)
    assert np.allclose(h, h_true, atol=1e-6)
    assert (h >= 0).all()


def _toy_reference():
    # 4 mutation types, 3 signatures (columns sum to 1).
    W = np.array(
        [
            [0.7, 0.1, 0.25],
            [0.1, 0.7, 0.25],
            [0.1, 0.1, 0.25],
            [0.1, 0.1, 0.25],
        ]
    )
    return W


def test_fit_one_recovers_sparse_truth():
    W = _toy_reference()
    # Only signatures 0 and 1 are active; signature 2 absent.
    h_true = np.array([30.0, 70.0, 0.0])
    m = W @ h_true
    h, cos = _fit_one(W, m, max_delta=0.01, min_activity=0.005)
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert h[2] == 0.0  # unused signature stays out
    assert h[0] == pytest.approx(30.0, rel=1e-3)
    assert h[1] == pytest.approx(70.0, rel=1e-3)


def test_fit_one_zero_sample():
    W = _toy_reference()
    h, cos = _fit_one(W, np.zeros(4), max_delta=0.01, min_activity=0.005)
    assert (h == 0).all()
    assert cos == 0.0


def test_fit_one_prunes_below_min_activity():
    W = _toy_reference()
    # Mostly signature 0, a tiny sliver of signature 1 (< 0.5% of total).
    h_true = np.array([100.0, 0.2, 0.0])
    m = W @ h_true
    h, cos = _fit_one(W, m, max_delta=0.001, min_activity=0.005)
    # signature 1 sliver is below min_activity -> pruned to 0
    assert h[1] == 0.0


def _catalogue_and_reference():
    types = ["A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T"]
    ref = pl.DataFrame(
        {
            "MutationType": types,
            "SBS_X": [0.7, 0.1, 0.1, 0.1],
            "SBS_Y": [0.1, 0.7, 0.1, 0.1],
            "SBS_Z": [0.25, 0.25, 0.25, 0.25],
        }
    )
    W = ref.select(["SBS_X", "SBS_Y", "SBS_Z"]).to_numpy()
    # sample s0 = 30*X + 70*Y ; sample s1 = 100*Z
    m0 = W @ np.array([30.0, 70.0, 0.0])
    m1 = W @ np.array([0.0, 0.0, 50.0])
    cat = pl.DataFrame(
        {
            "MutationType": types,
            "s0": np.rint(m0).astype(np.int64),
            "s1": np.rint(m1).astype(np.int64),
        }
    )
    return cat, ref


def test_fit_signatures_shape_and_columns():
    cat, ref = _catalogue_and_reference()
    act = fit_signatures(cat, ref)
    assert act.columns == ["Sample", "SBS_X", "SBS_Y", "SBS_Z", "cosine_similarity"]
    assert act["Sample"].to_list() == ["s0", "s1"]
    assert act.height == 2


def test_fit_signatures_recovers_activities():
    cat, ref = _catalogue_and_reference()
    act = fit_signatures(cat, ref)
    row0 = act.filter(pl.col("Sample") == "s0")
    assert row0["SBS_X"].item() == pytest.approx(30.0, rel=0.02)
    assert row0["SBS_Y"].item() == pytest.approx(70.0, rel=0.02)
    assert row0["SBS_Z"].item() == 0.0
    assert row0["cosine_similarity"].item() == pytest.approx(1.0, abs=1e-3)


def test_fit_signatures_aligns_rows_by_join_not_position():
    cat, ref = _catalogue_and_reference()
    ref_shuffled = ref.sort("MutationType", descending=True)  # reorder rows
    act = fit_signatures(cat, ref_shuffled)
    row0 = act.filter(pl.col("Sample") == "s0")
    assert row0["SBS_X"].item() == pytest.approx(30.0, rel=0.02)


def test_fit_signatures_missing_type_raises():
    cat, ref = _catalogue_and_reference()
    ref_missing = ref.head(3)  # drop a row present in the catalogue
    with pytest.raises(ValueError, match="MutationType"):
        fit_signatures(cat, ref_missing)


def test_load_signature_file_renames_type_column():
    df = _load_signature_file(DATA / "cosmic_mini.txt")
    assert df.columns[0] == "MutationType"
    assert "SBS1" in df.columns and "SBS5" in df.columns
    assert df["MutationType"].to_list()[0] == "A[C>A]A"


def test_load_signature_file_path_as_str():
    df = _load_signature_file(str(DATA / "cosmic_mini.txt"))
    assert df.height == 4


@pytest.mark.network
def test_cosmic_signatures_sbs96_row_order():
    from genoray._mutcat import labels

    df = cosmic_signatures("SBS96")
    assert df.columns[0] == "MutationType"
    assert df["MutationType"].to_list() == labels("SBS96")  # canonical 96 rows
    # signature columns are the COSMIC SBS set
    assert any(c.startswith("SBS") for c in df.columns[1:])


def test_cosmic_signatures_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown kind"):
        cosmic_signatures("SBS9999")  # type: ignore[arg-type]


def test_cosmic_signatures_unregistered_combo_raises():
    with pytest.raises(ValueError, match="No COSMIC URL registered"):
        cosmic_signatures("SBS96", version="9.9", genome="GRCh38")


def test_public_exports():
    import genoray

    assert hasattr(genoray, "fit_signatures")
    assert hasattr(genoray, "cosmic_signatures")
    assert "fit_signatures" in genoray.__all__
    assert "cosmic_signatures" in genoray.__all__
