import numpy as np

from genoray._mutcat import Sentinel


def test_sentinel_values_unchanged():
    # The int values are a public wire contract (they appear in returned arrays).
    assert int(Sentinel.DBS_PARTNER) == -1
    assert int(Sentinel.UNCLASSIFIED) == -2
    assert int(Sentinel.MISSING) == -3
    assert int(Sentinel.NOT_ANNOTATED) == -4


def test_sentinel_is_int_compatible():
    # IntEnum members must behave as ints for numpy fills / comparisons.
    arr = np.full(3, Sentinel.UNCLASSIFIED, dtype=np.int16)
    assert arr.dtype == np.int16
    assert bool((arr == Sentinel.UNCLASSIFIED).all())
    assert np.int16(Sentinel.UNCLASSIFIED) == np.int16(-2)
