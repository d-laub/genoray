"""Calibration test: cross-check genoray mutation classifiers against SigProfilerMatrixGenerator.

This test builds a tiny custom genome, runs SigProfiler to classify a small variant set,
and asserts that genoray's classifier assigns the same channels.

Skip automatically in environments without SigProfilerMatrixGenerator (e.g. the default
pixi env). Run in the dedicated sigprofiler env:
    pixi run -e sigprofiler pytest tests/test_mutcat_calibration.py -v
"""

from __future__ import annotations

import gzip
import textwrap
from pathlib import Path

import polars as pl
import pytest

# Skip the entire module when SigProfilerMatrixGenerator is not installed.
SigProfilerMatrixGenerator = pytest.importorskip("SigProfilerMatrixGenerator")

from genoray._mutcat import (  # noqa: E402
    ID83,
    ID83_OFFSET,
    SBS96,
    SBS96_OFFSET,
    DBS78_OFFSET,
    classify_variants,
)
from genoray._reference import Reference  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny genome constants
# ---------------------------------------------------------------------------
# 30-base chr1 sequence:
#   pos 0..3 : AACG  (flanking / SNV context)
#   pos 4    : A     (anchor for the C deletion)
#   pos 5..9 : CCCCC (5-C homopolymer)
#   pos 10   : G     (right flank of homopolymer)
#   pos 11.. : T's  (filler)
GENOME_SEQ = "AACGACCCCCG" + "T" * 19  # 30 bases

# Use unprefixed contig name so SigProfiler's VCF converter (which strips "chr"
# from names longer than 2 chars) doesn't mangle the lookup key.
CHROM = "1"

# VCF variants (1-based POS):
#   SNV  : POS=3, REF=C, ALT=A  -> 0-based pos 2, context A[C>A]G
#   Del  : POS=5, REF=AC, ALT=A -> 0-based anchor 4, 4 remaining C's -> 1:Del:C:4
VCF_LINES = [
    (CHROM, 3, "C", "A"),  # SNV  -> A[C>A]G
    (CHROM, 5, "AC", "A"),  # Del  -> 1:Del:C:4
]

GENOME_NAME = "tiny_calibration"
SAMPLE_NAME = "SAMPLE1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_genome(genome_dir: Path) -> None:
    """Write gzipped per-chrom FASTA as SigProfiler custom install expects.

    SigProfiler's VCF converter strips the leading "chr" from chromosome names
    longer than 2 characters (``chrom = chrom[3:]``), so we name the FASTA
    contig without the prefix (">1") to keep the lookup key consistent.
    """
    genome_dir.mkdir(parents=True, exist_ok=True)
    fa_path = genome_dir / f"{CHROM}.fa.gz"
    with gzip.open(fa_path, "wt") as fh:
        fh.write(f">{CHROM}\n{GENOME_SEQ}\n")


def _write_stub_transcript_file(tmp_path: Path) -> Path:
    """Write a minimal stub transcript file (single file) for SigProfiler's install.

    SigProfiler's custom install expects ``transcriptPath`` to be a path to a
    *file*, not a directory — it calls ``shutil.copy(transcriptPath, dest_dir)``.

    The filename matters: ``convertVCF`` reads ``out_chroms`` from the transcripts
    directory as ``[x.replace("_transcripts.txt", "") for x in os.listdir(...)]``,
    which it uses to name per-chromosome SNV output files.  So the stub must be
    named ``<CHROM>_transcripts.txt`` so that ``out_chroms == [CHROM]`` and the
    SNV temp file gets the correct ``<CHROM>_<project>.genome`` name.

    The file also needs at least one valid record so that ``save_tsb_192.py``
    can open its output file handle (it fails with ``UnboundLocalError`` when
    the transcript file is empty and the loop body never executes).  The record
    format is tab-separated: geneID  transcriptID  chrom  strand  start  end.
    We put a dummy gene at position 1-1 (1-based) which covers nothing useful
    but satisfies the parser.
    """
    stub = tmp_path / f"{CHROM}_transcripts.txt"
    # One dummy record: geneID  transcriptID  chrom  strand  start(1-based)  end(1-based)
    # Note: save_tsb_192.py sorts transcripts against a bare-number list (no "chr" prefix).
    stub.write_text(f"DUMMY_GENE\tDUMMY_TX\t{CHROM}\t1\t1\t1\n")
    return stub


def _write_tsb_file(tsb_dir: Path) -> None:
    """Pre-generate the TSB binary file for the tiny genome.

    SigProfilerMatrixGeneratorFunc checks for ``tsb/<genome>/<chrom>.txt``
    before processing a chromosome.  For a custom genome the install calls
    ``save_tsb_192.py``, which fails when the transcript file is empty.

    The TSB binary format is one byte per base position.  For a fully
    non-transcribed genome every byte is the Non-transcribed base code:
      A=0, C=1, G=2, T=3, N=16  (everything else also maps to N=16).
    """
    base_to_byte = {b"A": 0, b"C": 1, b"G": 2, b"T": 3}
    tsb_dir.mkdir(parents=True, exist_ok=True)
    tsb_path = tsb_dir / f"{CHROM}.txt"
    seq_bytes = GENOME_SEQ.encode()
    with open(tsb_path, "wb") as fh:
        for base in seq_bytes:
            code = base_to_byte.get(bytes([base]), 16)
            fh.write(bytes([code]))


def _write_vcf(vcf_path: Path, variants: list[tuple[str, int, str, str]]) -> None:
    """Write a minimal valid VCF with the given variants."""
    header = textwrap.dedent(f"""\
        ##fileformat=VCFv4.2
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##contig=<ID={CHROM},length={len(GENOME_SEQ)}>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{SAMPLE_NAME}
    """)
    rows = []
    for chrom, pos, ref, alt in variants:
        rows.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t0/1")
    vcf_path.write_text(header + "\n".join(rows) + "\n")


def _write_plain_fasta(fa_path: Path) -> None:
    """Write a plain (non-gzipped) FASTA for pysam/genoray Reference."""
    fa_path.write_text(f">{CHROM}\n{GENOME_SEQ}\n")


def _read_sigprofiler_matrix(output_dir: Path, kind: str) -> pl.DataFrame:
    """Read SigProfiler output matrix (tab-separated) into a Polars DataFrame."""
    candidates = list(output_dir.glob(f"output/{kind}/*.{kind}*.all"))
    if not candidates:
        candidates = list(output_dir.rglob(f"*.{kind}*.all"))
    if not candidates:
        raise FileNotFoundError(
            f"SigProfiler {kind} output not found under {output_dir}. "
            f"Contents: {list(output_dir.rglob('*'))[:30]}"
        )
    path = candidates[0]
    return pl.read_csv(path, separator="\t")


def _code_to_label(code: int) -> str | None:
    if SBS96_OFFSET <= code < DBS78_OFFSET:
        return SBS96[code - SBS96_OFFSET]
    if ID83_OFFSET <= code < ID83_OFFSET + len(ID83):
        return ID83[code - ID83_OFFSET]
    return None


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.sigprofiler
def test_sigprofiler_calibration(tmp_path: Path) -> None:
    """Cross-check genoray classifiers against SigProfilerMatrixGenerator.

    Builds a tiny custom genome, classifies two variants (one SNV, one deletion)
    with both SigProfiler and genoray, and asserts they agree.
    """
    from SigProfilerMatrixGenerator import install as spm_install
    from SigProfilerMatrixGenerator.scripts import (
        SigProfilerMatrixGeneratorFunc as spmg,
    )

    # ---- 1. Write genome and transcript stubs ----
    genome_dir = tmp_path / "genome_fastas"
    _write_genome(genome_dir)
    tx_file = _write_stub_transcript_file(tmp_path)

    # ---- 2. Prepare SigProfiler reference dirs ----
    # Get the reference directory layout via SigProfiler's own API.
    from SigProfilerMatrixGenerator.scripts import ref_install as spm_ref_install

    reference_dir = spm_ref_install.reference_dir(secondary_chromosome_install_dir=None)
    tsb_dir = reference_dir.get_tsb_dir() / GENOME_NAME
    ref_root = Path(str(reference_dir.path))
    transcript_dir = (
        ref_root / "references" / "chromosomes" / "transcripts" / GENOME_NAME
    )

    # SigProfiler 1.3 uses reset_directory() which prompts interactively when the
    # dir already exists.  Pre-delete stale dirs so install runs non-interactively.
    import shutil

    for d in [tsb_dir, transcript_dir]:
        if d.exists():
            shutil.rmtree(d)

    # ---- 3. Install the custom genome into SigProfiler's references dir ----
    spm_install.install(
        GENOME_NAME,
        custom=True,
        fastaPath=str(genome_dir),
        transcriptPath=str(tx_file),
    )

    # ---- 4. Write the TSB binary file ----
    # save_tsb_192.py may fail (or write a wrong file) for minimal transcript
    # files.  We write the correct TSB binary directly after install completes.
    _write_tsb_file(tsb_dir)

    # ---- 4b. Restore transcript stub after install ----
    # save_tsb_192.py deletes the transcript file after processing.  convertVCF
    # reads out_chroms from the transcripts dir, so we need the stub present
    # when SPMG runs convertVCF.
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{CHROM}_transcripts.txt").write_text(
        f"DUMMY_GENE\tDUMMY_TX\t{CHROM}\t1\t1\t1\n"
    )

    # ---- 5. Monkey-patch is_genome_installed for custom genomes ----
    # SigProfilerMatrixGenerator >= 1.3 validates genomes against a checksum
    # registry.  Custom genomes are not registered, so is_genome_installed always
    # returns False and SPMG raises an exception.  We patch it for GENOME_NAME.
    from SigProfilerMatrixGenerator.scripts import reference_genome_manager as _rgm

    _orig_is_installed = _rgm.ReferenceGenomeManager.is_genome_installed

    def _patched_is_installed(
        self: _rgm.ReferenceGenomeManager, genome_name: str
    ) -> bool:
        if genome_name == GENOME_NAME:
            return True
        return _orig_is_installed(self, genome_name)

    _rgm.ReferenceGenomeManager.is_genome_installed = _patched_is_installed  # type: ignore[method-assign]

    # ---- 6. Write the VCF into project_dir/ ----
    project_name = "tiny_calib"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    vcf_path = project_dir / f"{project_name}.vcf"
    _write_vcf(vcf_path, VCF_LINES)

    # ---- 7. Run SigProfilerMatrixGeneratorFunc ----
    spmg.SigProfilerMatrixGeneratorFunc(
        project_name,
        GENOME_NAME,
        str(project_dir),
        plot=False,
        seqInfo=False,
    )

    # ---- 5. Read back the SigProfiler output matrices ----
    sbs_df = _read_sigprofiler_matrix(project_dir, "SBS96")
    id83_df = _read_sigprofiler_matrix(project_dir, "ID83")

    # Convert to dicts: label -> count for our sample
    sbs_col = SAMPLE_NAME
    if sbs_col not in sbs_df.columns:
        sbs_col = [c for c in sbs_df.columns if c != "MutationType"][0]
    id83_col = SAMPLE_NAME
    if id83_col not in id83_df.columns:
        id83_col = [c for c in id83_df.columns if c != "MutationType"][0]

    sbs_counts = dict(zip(sbs_df["MutationType"].to_list(), sbs_df[sbs_col].to_list()))
    id83_counts = dict(
        zip(id83_df["MutationType"].to_list(), id83_df[id83_col].to_list())
    )

    # ---- 6. Find which channels SigProfiler assigned (nonzero) ----
    sbs_nonzero = {k: v for k, v in sbs_counts.items() if v != 0}
    id83_nonzero = {k: v for k, v in id83_counts.items() if v != 0}

    # ---- 7. Build genoray classification for the same variants ----
    fa_path = tmp_path / "ref.fa"
    _write_plain_fasta(fa_path)
    ref = Reference.from_path(fa_path)

    # VCF_LINES has 1-based POS; classify_variants converts internally.
    geno_index = pl.DataFrame(
        {
            "CHROM": [v[0] for v in VCF_LINES],
            "POS": [v[1] for v in VCF_LINES],  # 1-based; classify_variants converts
            "REF": [v[2] for v in VCF_LINES],
            "ALT": [[v[3]] for v in VCF_LINES],
        }
    )
    codes = classify_variants(geno_index, ref)
    genoray_labels = [_code_to_label(int(c)) for c in codes]

    print(f"\nSigProfiler nonzero SBS-96: {sbs_nonzero}")
    print(f"SigProfiler nonzero ID-83:  {id83_nonzero}")
    print(f"genoray labels: {genoray_labels}")

    # ---- 8. Homopolymer deletion assertion (variant index 1) ----
    # Rule: repeat count = homopolymer_length - 1 = 5 - 1 = 4 remaining C's -> bucket 4
    expected_id83_channel = "1:Del:C:4"
    assert expected_id83_channel in id83_nonzero, (
        f"SigProfiler did not assign {expected_id83_channel!r}. "
        f"Nonzero ID-83 channels: {id83_nonzero}"
    )
    assert genoray_labels[1] == expected_id83_channel, (
        f"genoray assigned {genoray_labels[1]!r} for the homopolymer deletion; "
        f"expected {expected_id83_channel!r}."
    )

    # ---- 9. SNV assertion (variant index 0) ----
    expected_sbs96_channel = "A[C>A]G"
    assert expected_sbs96_channel in sbs_nonzero, (
        f"SigProfiler did not assign {expected_sbs96_channel!r}. "
        f"Nonzero SBS-96 channels: {sbs_nonzero}"
    )
    assert genoray_labels[0] == expected_sbs96_channel, (
        f"genoray assigned {genoray_labels[0]!r} for the SNV; "
        f"expected {expected_sbs96_channel!r}."
    )
