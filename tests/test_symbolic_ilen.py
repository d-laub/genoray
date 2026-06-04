from __future__ import annotations

import polars as pl

from genoray.exprs import is_imprecise, symbolic_ilen


def _frame():
    # one row per scenario; SVLEN/END/IMPRECISE pre-extracted as scalars
    return pl.DataFrame(
        {
            "REF": ["G", "G", "G", "G", "A", "G"],
            "ALT": [["<DEL>"], ["<INS>"], ["<DUP>"], ["<DEL>"], ["AT"], ["<BND>"]],
            "SVLEN": [100, 50, 30, None, None, None],
            "END": [200, None, 130, None, None, None],
            "IMPRECISE": [False, False, False, True, False, False],
            "POS": [1000, 2000, 3000, 4000, 5000, 6000],
        }
    )


def test_symbolic_ilen_precise_and_unsizable():
    out = _frame().with_columns(ILEN=symbolic_ilen()).get_column("ILEN").to_list()
    assert out[0] == [-100]  # <DEL> SVLEN=100
    assert out[1] == [50]  # <INS> SVLEN=50
    assert out[2] == [30]  # <DUP> SVLEN=30
    assert out[3] == [None]  # IMPRECISE <DEL> -> null
    assert out[4] == [1]  # literal insertion AT vs A -> +1
    assert out[5] == [None]  # <BND> unsupported -> null


def test_is_imprecise_flags_only_unsizable():
    df = _frame().with_columns(ILEN=symbolic_ilen()).with_columns(imp=is_imprecise)
    assert df.get_column("imp").to_list() == [False, False, False, True, False, True]
