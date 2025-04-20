import polars as pl

SCHEMA = {
    "CHROM": pl.Categorical,
    "POS": pl.Int64,
    "REF": pl.Utf8,
    "ALT": pl.List(pl.Utf8),
}

ILEN = pl.col("ALT").str.len_bytes().cast(pl.Int32) - pl.col(
    "REF"
).str.len_bytes().cast(pl.Int32)
