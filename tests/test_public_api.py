import pytest

import genoray
from genoray import exprs


def test_reader_alias_removed():
    with pytest.raises(AttributeError):
        genoray.Reader
    assert "Reader" not in genoray.__all__


def test_exprs_internals_are_private():
    assert not hasattr(exprs, "symbolic_ilen")
    assert not hasattr(exprs, "IndexSchema")
    assert hasattr(exprs, "_symbolic_ilen")
    assert hasattr(exprs, "_IndexSchema")
