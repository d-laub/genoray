from pathlib import Path

import polars as pl

from genoray import VCF, SparseVar
from genoray._svar import _build_working_index, _resolve_kept_rows
from genoray._utils import ContigNormalizer


def _index_df(rows, index_values=None):
    """Build a minimal working-index frame: CHROM, POS, ILEN(list[int]), index(row id).

    If *index_values* is provided those values are used for the ``index`` column
    instead of the default ``with_row_index`` sequential assignment.  This lets
    tests construct frames where the ``index`` values do NOT equal row positions,
    which is the realistic SVAR scenario (subset of a larger index).
    """
    df = pl.DataFrame(
        {
            "CHROM": [r[0] for r in rows],
            "POS": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "ILEN": [[0] for _ in rows],
        }
    )
    if index_values is None:
        return df.with_row_index("index")
    return df.with_columns(pl.Series("index", index_values, dtype=pl.UInt32))


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


def test_resolve_kept_rows_non_positional_index():
    """index column values that do NOT equal row positions must be returned.

    The frame represents three variants drawn from a larger SVAR whose global ids
    are 10, 11, 12 (not 0, 1, 2).  A query covering POS 10 and 20 (0-based [9, 21))
    must return [10, 11] — the *index column values* — not [0, 1] (row positions).

    If someone reverted _resolve_kept_rows to use positional indexing (e.g.
    ``candidates[...]`` row offsets instead of ``is_in`` + ``sort`` on the index
    column) this test would return [0, 1] instead of [10, 11] and fail.
    """
    # Variants with global SVAR ids 10, 11, 12 (not 0-based row offsets).
    df = _index_df(
        [("chr1", 10), ("chr1", 20), ("chr1", 30)],
        index_values=[10, 11, 12],
    )
    cnorm = ContigNormalizer(["chr1"])
    # region 0-based [9, 21) covers 1-based POS 10 and 20
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": pl.Series([9], dtype=pl.Int32),
            "end": pl.Series([21], dtype=pl.Int32),
        }
    )
    kept = _resolve_kept_rows(df, cnorm, regions, "pos", merge_overlapping=False)
    # Must be global ids, not row positions.
    assert kept.tolist() == [10, 11], f"expected [10, 11], got {kept.tolist()}"


def test_resolve_kept_rows_variant_mode():
    """variant mode returns every candidate that *overlaps* the region (ILEN-aware).

    A deletion at POS 5 with ILEN=-5 spans 0-based [4, 9).  Even though its POS-1=4
    is outside the query region [6, 15), the deletion's end (9) extends into it.
    pos mode would exclude it; variant mode must include it.

    The non-contiguous index values ([20, 21, 22]) confirm that the returned ids are
    index-column values, not row positions, even in variant mode.
    """
    # Deletion at POS 5, ILEN=-5 → 0-based span [4, 9); overlaps query [6, 15).
    # SNP at POS 10, ILEN=0  → 0-based span [9, 10); inside query [6, 15).
    # SNP at POS 20, ILEN=0  → 0-based span [19, 20); outside query [6, 15).
    df = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1"],
            "POS": pl.Series([5, 10, 20], dtype=pl.Int32),
            "ILEN": [[-5], [0], [0]],
            "index": pl.Series([20, 21, 22], dtype=pl.UInt32),
        }
    )
    cnorm = ContigNormalizer(["chr1"])
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": pl.Series([6], dtype=pl.Int32),
            "end": pl.Series([15], dtype=pl.Int32),
        }
    )
    # variant mode: deletion (id=20) overlaps via span; SNP at POS 10 (id=21) overlaps.
    kept_variant = _resolve_kept_rows(
        df, cnorm, regions, "variant", merge_overlapping=False
    )
    assert 20 in kept_variant.tolist(), (
        "deletion spanning into region must be kept in variant mode"
    )
    assert 21 in kept_variant.tolist(), "SNP inside region must be kept in variant mode"
    assert 22 not in kept_variant.tolist(), "SNP outside region must not be kept"

    # pos mode: deletion POS-1=4 is outside [6, 15), so id=20 must be excluded.
    kept_pos = _resolve_kept_rows(df, cnorm, regions, "pos", merge_overlapping=False)
    assert 20 not in kept_pos.tolist(), (
        "deletion POS outside region must be excluded in pos mode"
    )
    assert 21 in kept_pos.tolist(), "SNP inside region must be kept in pos mode"


def _make_svar_from_vcf(tmp_path: Path, vcf_path: str) -> Path:
    out = tmp_path / "full.svar"
    SparseVar.from_vcf(out, VCF(vcf_path), max_mem="1g", overwrite=True)
    return out


def test_build_working_index_has_required_columns(tmp_path):
    sv_path = _make_svar_from_vcf(tmp_path, "tests/data/biallelic.vcf.gz")
    df, alt_is_utf8, ilen_added = _build_working_index(
        SparseVar._index_path(sv_path), None
    )
    assert {"CHROM", "POS", "ILEN", "index"} <= set(df.columns)
    assert df["index"].to_list() == list(range(df.height))
    # ALT present as list[str] for filtering
    assert df.schema["ALT"] == pl.List(pl.Utf8)
