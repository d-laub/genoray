"""Tests for issue #175: a public accessor for the identity of read variants.

`PGEN.var_idxs` returns positions into the reader's index and documents that
`reader._index[var_idxs]` is valid -- but `_index` is private, so anything that
has to outlive the PGEN it was derived from (a fitted model keyed by variant,
say) had to reach into it and assert the lengths lined up by hand.
`var_records` is that lookup, aligned to `read`'s variant axis by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from genoray import exprs
from genoray._pgen import PGEN, Genos

ddir = Path(__file__).parent / "data"

# tests/data/biallelic.pvar, contig 1 (the file spells contigs without `chr`):
#   81262 GAT>A, 81262 G>A, 81265 T>C
_CHR1 = (81261, 81266)


def _pgen(**kwargs) -> PGEN:
    return PGEN(ddir / "biallelic.pgen", **kwargs)


def test_var_records_is_aligned_with_the_variant_axis_of_read():
    """The whole point: column j of the genotype array is row j here."""
    pgen = _pgen()
    genos = pgen.read("chr1", *_CHR1, mode=Genos)
    records = pgen.var_records("chr1", *_CHR1)
    assert records.height == genos.shape[-1]
    assert records.get_column("POS").to_list() == [81262, 81262, 81265]
    assert records.get_column("REF").to_list() == ["GAT", "G", "T"]
    assert records.get_column("ALT").to_list() == [["A"], ["A"], ["C"]]


def test_var_records_returns_the_contig_name_the_file_uses():
    """`chr1` is accepted as a query, but the identity worth persisting is the
    one the file actually stores -- here the unprefixed `1`."""
    pgen = _pgen()
    records = pgen.var_records("chr1", *_CHR1)
    assert records.get_column("CHROM").to_list() == ["1", "1", "1"]


def test_var_records_alt_is_always_a_list_of_alleles():
    """#175 notes ALT is sometimes `Utf8` and sometimes `List(Utf8)` depending
    on how the index was built. The PGEN `.gvi` stores it comma-joined; the
    accessor is where that is settled for callers."""
    pgen = _pgen()
    on_disk = pl.read_ipc(ddir / "biallelic.pvar.gvi")
    assert on_disk.schema["ALT"] == pl.Utf8, "fixture no longer exercises the split"
    assert pgen.var_records("chr1", *_CHR1).schema["ALT"] == pl.List(pl.Utf8)


def test_var_records_labels_which_range_each_row_came_from():
    """Multi-range reads need the grouping, and returning it as a column keeps
    the accessor a single self-describing frame."""
    pgen = _pgen()
    starts = np.array([81261, 81264])
    ends = np.array([81263, 81266])
    records = pgen.var_records("chr1", starts, ends)
    assert records.get_column("range").to_list() == [0, 0, 1]
    assert records.get_column("POS").to_list() == [81262, 81262, 81265]
    counts = pgen.n_vars_in_ranges("chr1", starts, ends)
    per_range = np.bincount(records.get_column("range").to_numpy(), minlength=2)
    assert per_range.tolist() == counts.tolist()


def test_var_records_omits_what_the_filter_omits():
    """A filtered reader's `read` drops variants; so must their identities, or
    the alignment the method promises is a lie."""
    filtered = _pgen(filter=exprs.is_snp)
    genos = filtered.read("chr1", *_CHR1, mode=Genos)
    records = filtered.var_records("chr1", *_CHR1)
    assert records.height == genos.shape[-1]
    # The 81262 GAT>A deletion is filtered out; the two SNPs remain.
    assert records.get_column("REF").to_list() == ["G", "T"]


def test_var_records_is_empty_for_a_range_with_no_variants():
    pgen = _pgen()
    records = pgen.var_records("chr1", 1, 2)
    assert records.height == 0
    assert records.columns == ["range", "CHROM", "POS", "REF", "ALT"]


def test_var_records_matches_the_private_index_lookup_it_replaces():
    """Equivalence with the workaround #175 was filed against."""
    pgen = _pgen()
    idxs, _ = pgen.var_idxs("chr1", *_CHR1)
    assert pgen._index is not None
    expected = pgen._index[idxs].select("CHROM", "POS", "REF", "ALT")
    assert pgen.var_records("chr1", *_CHR1).drop("range").equals(expected)
