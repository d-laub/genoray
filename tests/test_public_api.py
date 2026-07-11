import pytest

import genoray


def test_reader_alias_removed():
    with pytest.raises(AttributeError):
        genoray.Reader
    assert "Reader" not in genoray.__all__
