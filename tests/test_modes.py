import numpy as np
from genoray._modes import make_array_mode, make_tuple_mode

Geno = make_array_mode("Geno", np.int8, 3, genos=True)
Dose = make_array_mode("Dose", np.float32, 2)
GenoDose = make_tuple_mode("GenoDose", (Geno, Dose), genos_dtype=np.int8)


def test_array_mode_empty_shapes():
    g = Geno.empty(3, 2, 5)
    assert g.shape == (3, 2, 5) and g.dtype == np.int8
    d = Dose.empty(3, 2, 5)  # ploidy ignored for 2D modes
    assert d.shape == (3, 5) and d.dtype == np.float32


def test_array_mode_predicate():
    assert isinstance(np.empty((3, 2, 5), np.int8), Geno)
    assert not isinstance(np.empty((3, 4, 5), np.int8), Geno)  # bad ploidy axis
    assert not isinstance(np.empty((3, 2, 5), np.int16), Geno)  # bad dtype


def test_nbytes_per_variant():
    # genos: n_samples * ploidy * itemsize; dosages: n_samples * 1 * itemsize
    assert Geno.nbytes_per_variant(3, 2) == 3 * 2 * 1
    assert Dose.nbytes_per_variant(3, 2) == 3 * 1 * 4
    assert GenoDose.nbytes_per_variant(3, 2) == 3 * 2 * 1 + 3 * 1 * 4


def test_tuple_mode_empty():
    g, d = GenoDose.empty(3, 2, 5)
    assert g.shape == (3, 2, 5) and d.shape == (3, 5)
    assert isinstance((g, d), GenoDose)


def test_tuple_mode_dtypes():
    assert GenoDose._dtypes == (np.int8, np.float32)
