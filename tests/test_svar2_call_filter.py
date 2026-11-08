"""Tests for ``SparseVar2.write_view(call_filter=...)``: a per-call FORMAT-field
predicate, optionally scoped to a set of samples.

The oracle is numpy: decode the SOURCE with the filter field, apply the
predicate to each (region, sample, ploid, variant) cell, and compare to a
decode of the filtered VIEW. Fixtures force BOTH var_key- and dense-routed
variants (common sites are carried by most of 40 samples, rare ones by <=2) and
sprinkle missing VAF entries.
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from genoray import CallFilter, SparseVar2
from genoray._svar2_fields import FormatField

N_SAMPLES = 40
CONTIG_LEN = 800
THRESH = 0.2
OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def _write_vcf_and_ref(tmp_path: Path, tag: str = "") -> tuple[Path, Path]:
    """Return (bgzipped vcf, ref fasta)."""
    rng = np.random.default_rng(7)
    ref_seq = "".join(rng.choice(list("ACGT"), CONTIG_LEN))
    ref = tmp_path / f"ref{tag}.fa"
    ref.write_text(f">chr1\n{ref_seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    samples = [f"S{i}" for i in range(N_SAMPLES)]
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={CONTIG_LEN}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=VAF,Number=1,Type=Float,Description="VAF">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="DP">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    pos = 5
    for v in range(60):
        pos += int(rng.integers(4, 12))
        kind = ("snp", "ins", "del")[v % 3]
        r = ref_seq[pos - 1]
        if kind == "snp":
            refa, alt = r, next(b for b in "ACGT" if b != r)
        elif kind == "ins":
            refa, alt = r, r + "AT"
        else:
            refa, alt = ref_seq[pos - 1 : pos + 2], r
        common = v % 2 == 0
        row = []
        for s in range(N_SAMPLES):
            p = 0.85 if common else 0.04
            h = [int(rng.random() < p) for _ in range(2)]
            if not any(h):
                row.append("0|0:.:.")
                continue
            vaf = float(np.round(rng.uniform(0.02, 0.6), 3))
            # exact-threshold values and missing values exercise the edges
            u = rng.random()
            vaf_s = "." if u < 0.12 else ("0.2" if u < 0.2 else f"{vaf}")
            row.append(f"{h[0]}|{h[1]}:{vaf_s}:{int(rng.integers(5, 60))}")
        lines.append(f"chr1\t{pos}\t.\t{refa}\t{alt}\t.\t.\t.\tGT:VAF:DP\t" + "\t".join(row))
    vcf = tmp_path / f"in{tag}.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    gz = tmp_path / f"in{tag}.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz, ref


@pytest.fixture(scope="module")
def src_env(tmp_path_factory):
    d = tmp_path_factory.mktemp("callfilter")
    vcf, ref = _write_vcf_and_ref(d)
    out = d / "src.svar2"
    SparseVar2.from_vcf(
        out,
        vcf,
        str(ref),
        signatures=True,
        format_fields=[FormatField("VAF", dtype="f32"), FormatField("DP", dtype="u16")],
        threads=1,
    )
    src = SparseVar2(out)
    # non-vacuity: both routes are populated
    c = out / "chr1"
    for stream in ("var_key/snp", "var_key/indel", "dense/snp", "dense/indel"):
        assert (c / stream / "positions.bin").stat().st_size > 0, stream
    return src, ref, d


def _cells(rag, key, n_regions, n_samples, ploidy):
    """{(r, s, p): array of values} for one ragged key."""
    lengths = np.asarray(rag[key].lengths).reshape(n_regions, n_samples, ploidy)
    data = np.asarray(rag[key].data)
    offs = np.concatenate([[0], np.cumsum(lengths.reshape(-1))])
    out = {}
    for r in range(n_regions):
        for s in range(n_samples):
            for p in range(ploidy):
                k = (r * n_samples + s) * ploidy + p
                out[(r, s, p)] = data[offs[k] : offs[k + 1]]
    return out


def _oracle(src, view, op, thr, scope, keep_missing, keys=("pos", "ilen", "VAF", "DP")):
    """Assert decode(view) == decode(src) with the predicate applied in numpy."""
    n_s = len(src.available_samples)
    src_f = src.with_fields(["VAF", "DP"])
    view_f = SparseVar2(view.path).with_fields(["VAF", "DP"])
    regs = [(0, CONTIG_LEN)]
    a = src_f.decode("chr1", regs)
    b = view_f.decode("chr1", regs)
    cells = {k: _cells(a, k, 1, n_s, 2) for k in keys}
    got = {k: _cells(b, k, 1, n_s, 2) for k in keys}
    scope_idx = (
        set(range(n_s))
        if scope is None
        else {src.available_samples.index(s) for s in scope}
    )
    n_removed = 0
    for s in range(n_s):
        for p in range(2):
            vaf = cells["VAF"][(0, s, p)].astype(np.float32)
            missing = np.isnan(vaf)
            with np.errstate(invalid="ignore"):
                ok = OPS[op](vaf, np.float32(thr))
            keep = np.where(missing, keep_missing, ok)
            if s not in scope_idx:
                keep = np.ones_like(keep)
            n_removed += int((~keep).sum())
            for k in keys:
                np.testing.assert_array_equal(
                    got[k][(0, s, p)],
                    cells[k][(0, s, p)][keep],
                    err_msg=f"{k} sample {s} ploid {p}",
                )
    return n_removed


GATED = ["S3", "S7", "S8", "S20", "S39"]


@pytest.mark.parametrize("reroute", [False, True, "auto"])
@pytest.mark.parametrize("scope", [None, GATED], ids=["all", "scoped"])
@pytest.mark.parametrize("keep_missing", [True, False])
def test_decode_matches_numpy_oracle(src_env, tmp_path, reroute, scope, keep_missing):
    src, _, _ = src_env
    out = tmp_path / "v.svar2"
    src.write_view(
        ("chr1", 0, CONTIG_LEN),
        src.available_samples,
        out,
        fields=["VAF", "DP"],
        call_filter=CallFilter("VAF", ">=", THRESH, samples=scope, keep_missing=keep_missing),
        reroute=reroute,
        threads=1,
    )
    n_removed = _oracle(src, SparseVar2(out), ">=", THRESH, scope, keep_missing)
    assert n_removed > 0  # non-vacuous


@pytest.mark.parametrize("op", list(OPS))
def test_all_operators(src_env, tmp_path, op):
    src, _, _ = src_env
    out = tmp_path / "v.svar2"
    src.write_view(
        ("chr1", 0, CONTIG_LEN),
        src.available_samples,
        out,
        fields=["VAF", "DP"],
        call_filter=CallFilter("VAF", op, THRESH, samples=GATED),
        reroute=False,
    )
    _oracle(src, SparseVar2(out), op, THRESH, GATED, True)


def test_filter_field_need_not_be_carried(src_env, tmp_path):
    src, _, _ = src_env
    full = tmp_path / "full.svar2"
    nofield = tmp_path / "nofield.svar2"
    cf = CallFilter("VAF", ">=", THRESH, samples=GATED)
    src.write_view(("chr1", 0, CONTIG_LEN), src.available_samples, full, fields=["DP"], call_filter=cf)
    src.write_view(("chr1", 0, CONTIG_LEN), src.available_samples, nofield, call_filter=cf)
    ra = SparseVar2(full).decode("chr1", [(0, CONTIG_LEN)])
    rb = SparseVar2(nofield).decode("chr1", [(0, CONTIG_LEN)])
    np.testing.assert_array_equal(np.asarray(ra["pos"].data), np.asarray(rb["pos"].data))


def test_subset_samples_and_regions(src_env, tmp_path):
    """Filter composes with a sample subset (scope names outside the view are
    harmless) and a sub-region."""
    src, _, _ = src_env
    keep = ["S39", "S3", "S10", "S0"]  # reordered subset
    out = tmp_path / "v.svar2"
    src.write_view(
        ("chr1", 100, 500),
        keep,
        out,
        fields=["VAF"],
        call_filter=CallFilter("VAF", ">=", THRESH, samples=["S3", "S20"]),
    )
    ref = tmp_path / "ref.svar2"
    src.write_view(("chr1", 100, 500), keep, ref, fields=["VAF"])
    a = SparseVar2(ref).with_fields(["VAF"]).decode("chr1", [(100, 500)])
    b = SparseVar2(out).with_fields(["VAF"]).decode("chr1", [(100, 500)])
    va = _cells(a, "VAF", 1, 4, 2)
    vb = _cells(b, "VAF", 1, 4, 2)
    pa = _cells(a, "pos", 1, 4, 2)
    pb = _cells(b, "pos", 1, 4, 2)
    for s in range(4):
        for p in range(2):
            v = va[(0, s, p)]
            keepm = np.ones(len(v), bool)
            if keep[s] == "S3":  # S20 is not in the view
                keepm = np.isnan(v) | (v >= np.float32(THRESH))
            np.testing.assert_array_equal(vb[(0, s, p)], v[keepm])
            np.testing.assert_array_equal(pb[(0, s, p)], pa[(0, s, p)][keepm])


def test_region_counts_round_trip(src_env, tmp_path):
    src, _, _ = src_env
    out = tmp_path / "v.svar2"
    src.write_view(
        ("chr1", 0, CONTIG_LEN),
        src.available_samples,
        out,
        fields=["VAF"],
        call_filter=CallFilter("VAF", ">=", THRESH, samples=GATED),
    )
    view = SparseVar2(out)
    regs = [(0, 300), (300, CONTIG_LEN)]
    got = view.region_counts("chr1", regs)
    src_f = src.with_fields(["VAF"])
    want = np.zeros_like(got)
    for ri, (s0, e0) in enumerate(regs):
        dec = src_f.decode("chr1", [(s0, e0)])
        vaf = _cells(dec, "VAF", 1, N_SAMPLES, 2)
        for s in range(N_SAMPLES):
            for p in range(2):
                v = vaf[(0, s, p)]
                keep = np.ones(len(v), bool)
                if src.available_samples[s] in GATED:
                    keep = np.isnan(v) | (v >= np.float32(THRESH))
                want[ri, s, p] = keep.sum()
    np.testing.assert_array_equal(got, want)
    assert got.sum() < src.region_counts("chr1", regs).sum()


def test_zero_carrier_variant_is_dropped(tmp_path):
    """A variant whose every carrier fails is absent from the view (var_key AND
    dense), not left as an empty row."""
    ref = tmp_path / "ref.fa"
    ref.write_text(">chr1\n" + "ACGT" * 50 + "\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    samples = [f"S{i}" for i in range(10)]
    hdr = (
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=200>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="g">\n'
        '##FORMAT=<ID=VAF,Number=1,Type=Float,Description="v">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n"
    )
    # pos 5 (A->C) rare: S0 only, low VAF -> vanishes. pos 9 (A->C) common, all low
    # VAF in S0..S1 but high elsewhere -> survives as a variant.
    rare = ["1|0:0.05"] + ["0|0:."] * 9
    common = ["1|1:0.05", "1|1:0.05"] + ["1|1:0.5"] * 8
    body = (
        "chr1\t5\t.\tA\tC\t.\t.\t.\tGT:VAF\t" + "\t".join(rare) + "\n"
        "chr1\t9\t.\tA\tC\t.\t.\t.\tGT:VAF\t" + "\t".join(common) + "\n"
    )
    vcf = tmp_path / "z.vcf"
    vcf.write_text(hdr + body)
    gz = tmp_path / "z.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    sp = tmp_path / "z.svar2"
    SparseVar2.from_vcf(sp, gz, str(ref), format_fields=[FormatField("VAF", dtype="f32")], threads=1)
    src = SparseVar2(sp)
    for reroute in (False, True):
        out = tmp_path / f"zv{reroute}.svar2"
        src.write_view(
            ("chr1", 0, 200), src.available_samples, out, fields=["VAF"],
            call_filter=CallFilter("VAF", ">=", 0.2), reroute=reroute,
        )
        dec = SparseVar2(out).decode("chr1", [(0, 200)])
        pos = _cells(dec, "pos", 1, 10, 2)
        assert all(5 - 1 not in v and 4 not in v for v in pos.values())  # 0-based pos 4 gone
        assert 8 in pos[(0, 5, 0)]
        assert 8 not in pos[(0, 0, 0)] and 8 not in pos[(0, 1, 1)]


def test_none_is_byte_identical(src_env, tmp_path):
    """call_filter=None, and a scope that matches nothing in the data, change no byte."""
    src, ref, _ = src_env

    def digest(root: Path):
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    for reroute in (False, True):
        a = tmp_path / f"a{reroute}.svar2"
        b = tmp_path / f"b{reroute}.svar2"
        kw = dict(fields=["VAF", "DP"], reference=str(ref), reroute=reroute, threads=1)
        src.write_view(("chr1", 0, CONTIG_LEN), src.available_samples, a, **kw)
        src.write_view(("chr1", 0, CONTIG_LEN), src.available_samples, b, call_filter=None, **kw)
        assert digest(a) == digest(b)
        # a filter that removes nothing (>= -inf) is also the identity
        c = tmp_path / f"c{reroute}.svar2"
        src.write_view(
            ("chr1", 0, CONTIG_LEN), src.available_samples, c,
            call_filter=CallFilter("VAF", ">", -1.0), **kw,
        )
        assert digest(a) == digest(c)


def test_no_stale_mutcat_without_reference(src_env, tmp_path):
    src, ref, _ = src_env
    assert (Path(src.path) / "chr1" / "mutcat").exists()
    out = tmp_path / "v.svar2"
    src.write_view(
        ("chr1", 0, CONTIG_LEN), src.available_samples, out,
        call_filter=CallFilter("VAF", ">=", THRESH, samples=GATED),
    )
    assert not (out / "chr1" / "mutcat").exists()
    with pytest.raises(Exception):
        SparseVar2(out).mutation_matrix("SBS96")


@pytest.mark.parametrize("reroute", [False, True])
def test_mutation_matrix_after_reference_recompute(src_env, tmp_path, reroute):
    """Mutation counts of the filtered view equal those of a store built from a
    VCF whose failing calls were zeroed -- an independent from_vcf oracle."""
    src, ref, d = src_env
    out = tmp_path / "v.svar2"
    src.write_view(
        ("chr1", 0, CONTIG_LEN), src.available_samples, out, fields=["VAF"],
        reference=str(ref), reroute=reroute,
        call_filter=CallFilter("VAF", ">=", THRESH, samples=GATED),
    )
    view = SparseVar2(out)

    # oracle VCF: rewrite the source VCF with failing GT haplotypes zeroed
    import gzip

    in_gz = d / "in.vcf.gz"
    lines = []
    gated_cols = {9 + int(s[1:]) for s in GATED}
    with gzip.open(in_gz, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"):
                lines.append(ln)
                continue
            f = ln.rstrip("\n").split("\t")
            for c in gated_cols:
                gt, vaf, dp = f[c].split(":")
                if vaf != "." and float(np.float32(vaf)) < float(np.float32(THRESH)):
                    gt = "0|0"
                f[c] = ":".join((gt, vaf, dp))
            lines.append("\t".join(f) + "\n")
    ovcf = d / f"oracle{reroute}.vcf"
    ovcf.write_text("".join(lines))
    ogz = d / f"oracle{reroute}.vcf.gz"
    subprocess.run(f"bgzip -c {ovcf} > {ogz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", "-f", str(ogz)], check=True)
    osv = d / f"oracle{reroute}.svar2"
    SparseVar2.from_vcf(osv, ogz, str(ref), signatures=True, threads=1, overwrite=True)
    oracle = SparseVar2(osv)

    for kind in ("SBS96", "ID83"):
        mv = view.mutation_matrix(kind)
        mo = oracle.mutation_matrix(kind)
        assert mv.equals(mo), kind
        assert mv.select(mv.columns[1:]).to_numpy().sum() > 0


def test_validation_errors(src_env, tmp_path):
    src, _, _ = src_env
    out = tmp_path / "v.svar2"
    with pytest.raises(ValueError, match="not in this store"):
        src.write_view(("chr1", 0, 10), src.available_samples, out, call_filter=CallFilter("NOPE", ">=", 1))
    with pytest.raises(ValueError, match="not in"):
        src.write_view(
            ("chr1", 0, 10), src.available_samples, out,
            call_filter=CallFilter("VAF", ">=", 1, samples=["nope"]),
        )
    with pytest.raises(ValueError, match="op"):
        CallFilter("VAF", "~", 1)  # type: ignore[arg-type]
    assert not out.exists()
