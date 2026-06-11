import polars as pl

from genoray._svar import _resolve_kept_rows
from genoray._utils import ContigNormalizer


def _index_df(rows):
    """Build a minimal working-index frame: CHROM, POS, ILEN(list[int]), index(row id)."""
    df = pl.DataFrame(
        {
            "CHROM": [r[0] for r in rows],
            "POS": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "ILEN": [[0] for _ in rows],
        }
    )
    return df.with_row_index("index")


def test_resolve_kept_rows_pos_mode():
    # variants at 1-based POS 10, 20, 30 on chr1
    df = _index_df([("chr1", 10), ("chr1", 20), ("chr1", 30)])
    cnorm = ContigNormalizer(["chr1"])
    # region 0-based [9, 21) covers POS 10 (0-based 9) and POS 20 (0-based 19)
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": pl.Series([9], dtype=pl.Int32),
            "end": pl.Series([21], dtype=pl.Int32),
        }
    )
    kept = _resolve_kept_rows(df, cnorm, regions, "pos", merge_overlapping=False)
    assert kept.tolist() == [0, 1]
