import hashlib
import io
import subprocess
import threading
from pathlib import Path

import genoray._core as core
import pytest
from genoray import SparseVar2
from genoray._logging import ProgressRenderer, resolve_log_level, write_reporting
from rich.console import Console


def test_event_channel_roundtrip():
    rx = core.PyEventReceiver(flush_every=1)
    # No producer yet: recv_timeout returns None promptly.
    assert rx.recv_timeout(1) is None


def _events():
    return [
        ("contig_start", "chr1", None, None, None),
        ("progress", "chr1", 100, None, None),
        ("log", "info", "chr1", "excluded 12 records (check_ref=x)", "genoray"),
        ("contig_done", "chr1", 230, 20, 1234),
    ]


def test_heartbeat_non_tty_summary_lines():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    r = ProgressRenderer(console, show_bar=True)
    for e in _events():
        r.handle(e)
    r.close()
    out = buf.getvalue()
    assert "excluded 12 records" in out
    assert "chr1 done" in out
    assert "230" in out and "20" in out


def test_progress_false_suppresses_percent_lines():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    r = ProgressRenderer(console, show_bar=False)
    for e in _events():
        r.handle(e)
    r.close()
    out = buf.getvalue()
    # summaries still present, but no "%" throttled progress line
    assert "chr1 done" in out
    assert "%" not in out


def test_resolve_log_level_validates_and_env(monkeypatch):
    assert resolve_log_level("info") == "info"
    monkeypatch.setenv("GENORAY_LOG", "debug")
    assert resolve_log_level("info") == "debug"
    monkeypatch.delenv("GENORAY_LOG", raising=False)
    with pytest.raises(ValueError):
        resolve_log_level("loud")


def test_write_reporting_disabled_yields_none():
    with write_reporting(progress=False, log_level="off") as (rx, level):
        assert rx is None
        assert level == "off"


def test_write_reporting_drains_and_joins():
    n_threads_before = threading.active_count()
    with write_reporting(progress=False, log_level="info") as (rx, level):
        assert rx is not None
        assert level == "info"
        # Simulate a producer finishing immediately by not sending anything;
        # context exit must drop the sender and join the drain thread.
    # After exit, no leaked drain thread.
    assert threading.active_count() == n_threads_before


_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _tiny_vcf(d: Path) -> tuple[Path, Path]:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz, ref


def test_from_vcf_emits_summary(tmp_path, capsys):
    src, ref = _tiny_vcf(tmp_path)
    out = tmp_path / "out.svar"
    SparseVar2.from_vcf(out, src, ref, progress=False, log_level="info")
    captured = capsys.readouterr()
    assert "done" in captured.out.lower()


def test_from_pgen_emits_summary(tmp_path, capsys):
    gz, ref = _tiny_vcf(tmp_path)
    pgen = tmp_path / "in.pgen"
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(gz),
            "--out",
            str(tmp_path / "in"),
        ],
        check=True,
    )

    out_info = tmp_path / "out_info.svar2"
    ret_info = SparseVar2.from_pgen(
        out_info, pgen, ref, progress=False, log_level="info"
    )
    captured = capsys.readouterr()
    assert "done" in captured.out.lower()

    out_off = tmp_path / "out_off.svar2"
    ret_off = SparseVar2.from_pgen(out_off, pgen, ref, progress=False, log_level="off")

    assert ret_info == ret_off
    a = SparseVar2(out_info)
    b = SparseVar2(out_off)
    assert a.available_samples == b.available_samples
    counts_a = a.region_counts("chr1", [(0, 40)])
    counts_b = b.region_counts("chr1", [(0, 40)])
    assert counts_a.tolist() == counts_b.tolist()


def test_from_vcf_list_emits_summary(tmp_path, capsys):
    ref_seq = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{ref_seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    def _ss(name: str, sample: str, rows: str) -> Path:
        header = (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=chr1,length=40>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
        )
        plain = tmp_path / f"{name}.vcf"
        plain.write_text(header + rows)
        gz = tmp_path / f"{name}.vcf.gz"
        with open(gz, "wb") as fh:
            subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
        subprocess.run(["bcftools", "index", str(gz)], check=True)
        return gz

    a = _ss("a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss("b", "SB", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")

    out_info = tmp_path / "out_info"
    dropped_info = SparseVar2.from_vcf_list(
        out_info, [a, b], ref, threads=1, progress=False, log_level="info"
    )
    captured = capsys.readouterr()
    assert "done" in captured.out.lower()

    out_off = tmp_path / "out_off"
    dropped_off = SparseVar2.from_vcf_list(
        out_off, [a, b], ref, threads=1, progress=False, log_level="off"
    )

    assert dropped_info == dropped_off == 0
    sv_info = SparseVar2(out_info)
    sv_off = SparseVar2(out_off)
    assert sv_info.available_samples == sv_off.available_samples
    counts_info = sv_info.region_counts("chr1", [(0, 40)])
    counts_off = sv_off.region_counts("chr1", [(0, 40)])
    assert counts_info.tolist() == counts_off.tolist()


def test_below_pool_logs_surface_at_debug(tmp_path, capsys):
    """Regression guard for the process-global tracing subscriber fix.

    `resolve_fasta_contig`/`load_contig_seq` (contig-name resolution) and the
    check_ref=x exclusion path both run INSIDE `process_chromosome`, below
    the rayon pool `from_vcf` dispatches onto -- exactly the code path that
    was silently dropped under the old thread-local `tracing` subscriber.
    Trigger both: a VCF whose #CHROM is "1" against a FASTA contig "chr1"
    (forces contig-name normalization), plus a record whose REF disagrees
    with the reference under check_ref="x" (forces a below-pool exclusion
    log). If either below-pool message goes missing, the fix regressed.
    """
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        # pos 3 is really 'A' in _REF -- 'T' here is a deliberate REF mismatch.
        "1\t3\t.\tT\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = tmp_path / "in.vcf"
    plain.write_text(body)
    gz = tmp_path / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    out = tmp_path / "out.svar2"
    SparseVar2.from_vcf(out, gz, ref, check_ref="x", progress=False, log_level="debug")
    captured = capsys.readouterr()
    lower = captured.out.lower()
    assert "resolv" in lower or "normaliz" in lower, captured.out
    assert "exclud" in lower, captured.out


def test_genoray_log_env_overrides_rendered_verbosity(tmp_path, capsys, monkeypatch):
    """Regression guard for the bug where `GENORAY_LOG` was resolved on the
    Python side but the RAW `log_level` argument was still forwarded to the
    Rust `_core.*` call, so the channel/renderer never saw the override.

    Reuses the `check_ref="x"` REF-mismatch fixture from
    `test_below_pool_logs_surface_at_debug`. That fixture actually fires two
    "excluded" messages: a per-contig `tracing::info!` summary
    (`report_ref_excluded` in `orchestrator.rs`, always visible at "info")
    and a below-pool `tracing::debug!` first-offender detail in
    `chunk_assembler.rs` ("... further exclusions on this contig are
    counted, not logged individually") that's genuinely gated by the
    resolved level. Assert on the latter's unique text, not the generic
    "exclud" substring, since the info-level summary would make that
    substring present regardless of the bug.

    Calling with `log_level="info"`:
    - without `GENORAY_LOG` set, the effective level is "info" -> the debug
      detail line must NOT reach stdout.
    - with `GENORAY_LOG=debug`, the effective level must be raised to
      "debug" -> the debug detail line MUST reach stdout.
    Fails under the pre-fix code because the raw (unresolved) "info" reaches
    the Rust channel gate regardless of GENORAY_LOG.
    """
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        # pos 3 is really 'A' in _REF -- 'T' here is a deliberate REF mismatch.
        "1\t3\t.\tT\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    plain = tmp_path / "in.vcf"
    plain.write_text(body)
    gz = tmp_path / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    monkeypatch.delenv("GENORAY_LOG", raising=False)
    out_no_env = tmp_path / "out_no_env.svar2"
    SparseVar2.from_vcf(
        out_no_env, gz, ref, check_ref="x", progress=False, log_level="info"
    )
    captured = capsys.readouterr()
    lower = captured.out.lower()
    assert "done" in lower, captured.out
    assert "not logged individually" not in lower, captured.out

    monkeypatch.setenv("GENORAY_LOG", "debug")
    out_with_env = tmp_path / "out_with_env.svar2"
    SparseVar2.from_vcf(
        out_with_env, gz, ref, check_ref="x", progress=False, log_level="info"
    )
    captured = capsys.readouterr()
    lower = captured.out.lower()
    assert "not logged individually" in lower, captured.out


def test_from_svar1_emits_summary(tmp_path, capsys):
    from genoray import SparseVar
    from genoray import VCF as _V1VCF

    src, ref = _tiny_vcf(tmp_path)
    v1_out = tmp_path / "in.svar"
    v1 = _V1VCF(str(src))
    SparseVar.from_vcf(v1_out, v1, max_mem="10m", overwrite=True)

    out_info = tmp_path / "out_info.svar2"
    dropped_info = SparseVar2.from_svar1(
        out_info, v1_out, ref, threads=1, progress=False, log_level="info"
    )
    captured = capsys.readouterr()
    assert "done" in captured.out.lower()

    out_off = tmp_path / "out_off.svar2"
    dropped_off = SparseVar2.from_svar1(
        out_off, v1_out, ref, threads=1, progress=False, log_level="off"
    )

    assert dropped_info == dropped_off == 0
    sv_info = SparseVar2(out_info)
    sv_off = SparseVar2(out_off)
    assert sv_info.available_samples == sv_off.available_samples
    counts_info = sv_info.region_counts("chr1", [(0, 40)])
    counts_off = sv_off.region_counts("chr1", [(0, 40)])
    assert counts_info.tolist() == counts_off.tolist()


def _dir_digest(root: Path) -> dict[str, str]:
    """Content hash of every file under `root` except `meta.json` (see
    `tests/test_svar2_write_view.py::_dir_digest` -- `meta.json` is excluded
    there too, keeping this comparison purely about sidecar bytes)."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "meta.json"
    }


def test_write_view_emits_summary(svar2_store, tmp_path, capsys):
    """`write_view`'s progress/log_level plumbing (unlike the other four
    `from_*` writers, `run_slice_view` was not wired for Task 11 -- this is
    Task 12's write_view leg). Reuses the shared `svar2_store` fixture
    (`tests/conftest.py`) rather than building a fresh store, mirroring how
    `tests/test_svar2_write_view.py` slices that same fixture."""
    sv = SparseVar2(svar2_store)

    out_info = tmp_path / "view_info.svar2"
    sv.write_view(
        (sv.contigs[0], 0, 40),
        sv.available_samples,
        out_info,
        progress=False,
        log_level="info",
    )
    captured = capsys.readouterr()
    assert "done" in captured.out.lower()

    out_off = tmp_path / "view_off.svar2"
    sv.write_view(
        (sv.contigs[0], 0, 40),
        sv.available_samples,
        out_off,
        progress=False,
        log_level="off",
    )
    captured_off = capsys.readouterr()
    assert captured_off.out == ""

    # Logging is a pure side channel: the sliced output is byte-identical
    # regardless of log_level.
    assert _dir_digest(out_info) == _dir_digest(out_off)
    a = SparseVar2(out_info)
    b = SparseVar2(out_off)
    assert a.available_samples == b.available_samples
    assert a.contigs == b.contigs
