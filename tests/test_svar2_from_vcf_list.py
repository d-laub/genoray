from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from genoray import SparseVar2

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _write_ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _ss(d: Path, name: str, sample: str, rows: str) -> Path:
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
    )
    plain = d / f"{name}.vcf"
    plain.write_text(header + rows)
    gz = d / f"{name}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_list_disjoint_sites_hom_ref_fill(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "store"
    dropped = SparseVar2.from_vcf_list(out, [a, b], ref, threads=1)
    assert dropped == 0
    sv = SparseVar2(out)
    assert sv.available_samples == ["SA", "SB"]
    # SA carries SNP@2 on hap0; SB carries INS@6 on hap1.
    counts = sv.region_counts("chr1", [(0, 40)]).reshape(
        -1
    )  # (R,S,P) -> [SA_h0,SA_h1,SB_h0,SB_h1]
    assert counts.tolist() == [1, 0, 0, 1]


def test_from_vcf_list_shared_site_one_variant(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "s"
    dropped = SparseVar2.from_vcf_list(out, [a, b], ref, threads=1)
    assert dropped == 0
    sv = SparseVar2(out)
    rag = sv.decode("chr1", [(0, 40)])
    lengths = rag["pos"].lengths.reshape(-1)
    assert lengths.tolist() == [1, 0, 0, 1]  # same site, one hap each


def test_from_vcf_list_directory_and_manifest_equivalent(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n")
    SparseVar2.from_vcf_list(tmp_path / "by_seq", [a, b], ref, threads=1)
    # directory: put both gz+csi in a subdir
    d = tmp_path / "vcfs"
    d.mkdir()
    for p in (a, b):
        (d / p.name).write_bytes(p.read_bytes())
        (d / (p.name + ".csi")).write_bytes(Path(str(p) + ".csi").read_bytes())
    SparseVar2.from_vcf_list(tmp_path / "by_dir", d, ref, threads=1)
    manifest = tmp_path / "m.txt"
    manifest.write_text(f"# comment\n{a}\n\n{b}\n")
    SparseVar2.from_vcf_list(tmp_path / "by_manifest", manifest, ref, threads=1)
    for name in ("by_dir", "by_manifest"):
        assert SparseVar2(tmp_path / name).available_samples == ["SA", "SB"]


def test_from_vcf_list_rejects_multisample(tmp_path: Path):
    ref = _write_ref(tmp_path)
    two = _ss(
        tmp_path,
        "two",
        "SA\tSB",  # header hack: two sample cols
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|1\n",
    )
    with pytest.raises(ValueError, match="single-sample"):
        SparseVar2.from_vcf_list(tmp_path / "s", [two], ref, threads=1)


def test_from_vcf_list_rejects_duplicate_samples(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "S", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "S", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    with pytest.raises(ValueError, match="duplicate|collision"):
        SparseVar2.from_vcf_list(tmp_path / "s", [a, b], ref, threads=1)


def test_from_vcf_list_requires_reference(tmp_path: Path):
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    with pytest.raises(ValueError, match="reference"):
        SparseVar2.from_vcf_list(tmp_path / "s", [a], threads=1)


def test_from_vcf_list_matches_bcftools_merge_oracle(tmp_path: Path):
    """Oracle parity: `bcftools merge -0` (missing -> hom-ref, exactly our
    semantics) -> `from_vcf` must equal the native `from_vcf_list` k-way merge.

    Mix: a shared SNP (a+b, same site), a private INS (a only), a
    multiallelic split (b only), an anchored DEL with a missing hap (c only),
    and a cross-file join onto b's NON-FIRST ALT (d only) -- and b's
    multiallelic site and c's DEL share POS 7 but differ in ILEN/ALT, so they
    must NOT be spuriously joined.

    The assertions are split into two kinds:

    - ABSOLUTE: `dropped == 0`, the exact `region_counts` array, and the
      exact total decoded-entry count. These are hand-derived from the
      fixture below and pin *reality*, not merely oracle==native agreement --
      without them, a shared-layer regression that dropped every record from
      BOTH stores (oracle and native alike) would still pass the relative
      checks vacuously (empty == empty).
    - RELATIVE (oracle vs. native): region_counts, decoded pos/ilen/allele.
      These catch anything the hand-derived numbers above wouldn't (e.g. a
      difference in *which* records/positions appear despite the same total
      counts).
    """
    ref = _write_ref(tmp_path)
    a = _ss(
        tmp_path,
        "a",
        "SA",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n"  # shared SNP
        "chr1\t12\t.\tG\tGA\t.\t.\t.\tGT\t0|1\n",  # private INS
    )
    b = _ss(
        tmp_path,
        "b",
        "SB",
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t0|1\n"  # shared SNP
        "chr1\t7\t.\tC\tG,T\t.\t.\t.\tGT\t1|2\n",  # multiallelic
    )
    c = _ss(
        tmp_path,
        "c",
        "SC",
        "chr1\t7\t.\tCAT\tC\t.\t.\t.\tGT\t1|.\n",  # anchored DEL + missing hap
    )
    d = _ss(
        tmp_path,
        "d",
        "SD",
        # Joins b's ALT index 2 (T), NOT index 1 (G) -- this is the only
        # place in the whole test suite (Python or Rust) where a cross-file
        # join lands on a non-first source ALT index, exercising the
        # per-file local-ALT-index -> joined-record-slot remap across files
        # rather than within one file.
        "chr1\t7\t.\tC\tT\t.\t.\t.\tGT\t1|0\n",
    )
    paths = [a, b, c, d]

    # Oracle: bcftools merge -0 (missing genotypes -> 0/0, matching our
    # hom-ref-fill semantics) -> bgzip -> index -> from_vcf.
    merge = subprocess.run(
        ["bcftools", "merge", "-0", *map(str, paths)],
        check=True,
        stdout=subprocess.PIPE,
    )
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bgzip", "-c"], input=merge.stdout, check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(merged)], check=True)

    from_vcf_out = tmp_path / "oracle"
    SparseVar2.from_vcf(from_vcf_out, merged, ref, threads=1)
    list_out = tmp_path / "list"
    dropped = SparseVar2.from_vcf_list(list_out, paths, ref, threads=1)
    assert dropped == 0

    oracle, native = SparseVar2(from_vcf_out), SparseVar2(list_out)
    assert (
        oracle.available_samples == native.available_samples == ["SA", "SB", "SC", "SD"]
    )

    region = [(0, len(_REF))]

    # ABSOLUTE: hand-derived expected (region, sample, ploid) counts, so a
    # shared-layer regression that zeroed BOTH stores can't pass vacuously.
    # Per-hap ALT-carrier tally from the fixture above (5 distinct atoms:
    # SNP@2, INS@11, C>G@6, C>T@6, DEL@6):
    #   SA: hap0 <- SNP@2 (shared w/ SB),        hap1 <- private INS@11
    #   SB: hap0 <- C>G@6 (ALT idx 1),            hap1 <- SNP@2 (shared w/ SA)
    #       AND C>T@6 (ALT idx 2, joins d) -> 2 entries on hap1
    #   SC: hap0 <- anchored DEL@6,               hap1 <- missing (unobservable
    #       via decode/region_counts -- see
    #       test_from_vcf_list_missing_hap_is_unobservable_in_decode)
    #   SD: hap0 <- C>T@6 (ALT idx 1 locally, joins b's ALT idx 2),
    #       hap1 <- hom-ref
    expected_counts = np.array(
        [
            [
                [1, 1],  # SA
                [1, 2],  # SB
                [1, 0],  # SC
                [1, 0],  # SD
            ]
        ]
    )
    oracle_counts = oracle.region_counts("chr1", region)
    native_counts = native.region_counts("chr1", region)
    np.testing.assert_array_equal(oracle_counts, expected_counts)
    # Sum of the per-hap tally above: 2 (SA) + 3 (SB) + 1 (SC) + 1 (SD) = 7
    # total ALT-carrying (sample, ploid) entries across the 5 atoms.
    assert int(oracle_counts.sum()) == 7

    # RELATIVE: oracle vs. native must still agree exactly.
    np.testing.assert_array_equal(oracle_counts, native_counts)

    ro, rl = oracle.decode("chr1", region), native.decode("chr1", region)
    assert len(np.asarray(ro["pos"].data)) == 7
    for field in ("pos", "ilen"):
        np.testing.assert_array_equal(
            np.asarray(ro[field].data), np.asarray(rl[field].data)
        )
    # allele: variable-length ALT bytes per (sample, ploid, variant); pure
    # deletions decode to an empty ALT (anchor base is implicit) on BOTH
    # sides, so a like-for-like comparison is still meaningful.
    assert ro["allele"].to_ak().tolist() == rl["allele"].to_ak().tolist()


def test_from_vcf_list_missing_hap_is_unobservable_in_decode(tmp_path: Path):
    """Documents a gap, doesn't close it: `_svar2.py`'s `from_vcf_list`
    docstring says a within-file `./.` (missing) call decodes to `-1`,
    *distinct* from a hom-ref fill (`0`) for a sample lacking a site
    entirely. That `-1`/`0` distinction is real in the internal per-file
    `ploid_codes`/`RawRecord.gt` representation (see
    `src/vcf_list_reader.rs`), but neither `decode()` nor `region_counts()`
    exposes it: the sparse layout only ever stores ALT-*carrying* entries, so
    a missing hap and a hom-ref hap both decode to zero entries -- there is
    no public API surface to tell them apart.

    This test pins that non-distinction (both wind up indistinguishable, and
    both equal between the bcftools-merge oracle and the native
    `from_vcf_list` merge), rather than claiming to evidence the `-1` vs `0`
    split -- no test in this suite currently observes that split through a
    public method.
    """
    ref = _write_ref(tmp_path)
    # SA: ordinary hom-ref hap (hap1) with no site at all.
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    # SB: explicit missing hap (hap1) at the SAME site as SA's hom-ref hap.
    b = _ss(tmp_path, "b", "SB", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|.\n")
    paths = [a, b]

    merge = subprocess.run(
        ["bcftools", "merge", "-0", *map(str, paths)],
        check=True,
        stdout=subprocess.PIPE,
    )
    merged = tmp_path / "merged.vcf.gz"
    with open(merged, "wb") as fh:
        subprocess.run(["bgzip", "-c"], input=merge.stdout, check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(merged)], check=True)

    from_vcf_out = tmp_path / "oracle"
    SparseVar2.from_vcf(from_vcf_out, merged, ref, threads=1)
    list_out = tmp_path / "list"
    dropped = SparseVar2.from_vcf_list(list_out, paths, ref, threads=1)
    assert dropped == 0

    oracle, native = SparseVar2(from_vcf_out), SparseVar2(list_out)
    region = [(0, len(_REF))]
    # Both SA's hom-ref hap1 and SB's missing hap1 decode to 0 entries --
    # indistinguishable from each other, and identical between oracle/native.
    expected = np.array([[[1, 0], [1, 0]]])
    np.testing.assert_array_equal(oracle.region_counts("chr1", region), expected)
    np.testing.assert_array_equal(native.region_counts("chr1", region), expected)
