from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import genoray._svar2 as svar2_module
from genoray import SparseVar2
from genoray._progress import ProgressEvent
from tests.test_svar2_from_vcf import _write_ref, _write_vcf
from tests.test_svar2_from_vcf_list import _ss


def _write_two_contig_vcf(directory: Path) -> Path:
    plain = directory / "two-contig.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        "##contig=<ID=chr2,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n"
        "chr2\t7\t.\tC\tT\t.\t.\t.\tGT\t0|1\n"
    )
    gz = directory / "two-contig.vcf.gz"
    with open(gz, "wb") as stream:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=stream)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_progress_uses_completed_non_overlapping_contigs(tmp_path: Path):
    source = _write_two_contig_vcf(tmp_path)
    out = tmp_path / "store"
    events: list[ProgressEvent] = []

    SparseVar2.from_vcf(
        out,
        source,
        no_reference=True,
        threads=2,
        progress_callback=events.append,
    )

    converted = [
        event for event in events if event.phase == "converting" and event.completed > 0
    ]
    assert [event.completed for event in converted] == [1, 2]
    assert {event.details["contig"] for event in converted} == {"chr1", "chr2"}
    assert all(event.total == 2 and event.unit == "contig" for event in converted)
    assert all("cursor" not in event.details for event in converted)


def test_from_vcf_progress_snapshot_and_callback_finish_after_publication(
    tmp_path: Path,
):
    ref = _write_ref(tmp_path)
    source = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store"
    snapshot = tmp_path / "snapshot.json"
    events: list[ProgressEvent] = []

    def capture(event: ProgressEvent) -> None:
        events.append(event)
        if event.percent == 100.0:
            assert (out / "meta.json").is_file()

    SparseVar2.from_vcf(
        out,
        source,
        ref,
        threads=1,
        progress=True,
        progress_callback=capture,
        progress_path=snapshot,
        progress_identity={"run_id": "explicit-run"},
    )

    assert [(event.phase, event.state, event.percent) for event in events[:2]] == [
        ("preparing", "running", 0.0),
        ("preparing", "completed", 5.0),
    ]
    assert events[-1].phase == "complete"
    assert events[-1].state == "completed"
    assert events[-1].percent == 100.0
    assert all(event.percent < 100.0 for event in events[:-1])
    payload = json.loads(snapshot.read_text())
    assert payload["percent"] == 100.0
    assert payload["identity"] == {"run_id": "explicit-run"}


def test_from_vcf_failure_preserves_output_and_never_emits_100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ref = _write_ref(tmp_path)
    source = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store"
    out.mkdir()
    marker = out / "original"
    marker.write_text("keep")
    events: list[ProgressEvent] = []

    def fail_after_partial_write(*args, **kwargs):
        staging = Path(args[3])
        (staging / "partial").write_text("not durable")
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(
        svar2_module._core, "run_conversion_pipeline", fail_after_partial_write
    )

    with pytest.raises(RuntimeError, match="pipeline exploded"):
        SparseVar2.from_vcf(
            out,
            source,
            ref,
            threads=1,
            overwrite=True,
            progress_callback=events.append,
        )

    assert marker.read_text() == "keep"
    assert not (out / "partial").exists()
    assert events[-1].state == "failed"
    assert all(event.percent < 100.0 for event in events)


def test_from_vcf_native_finalization_failure_reports_finalizing_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ref = _write_ref(tmp_path)
    source = _write_vcf(tmp_path, symbolic=False, indexed=True)
    events: list[ProgressEvent] = []

    def fail_during_finalization(*args, **kwargs):
        contig_callback = args[-2]
        finalizing_callback = args[-1]
        contig_callback("chr1")
        finalizing_callback()
        raise RuntimeError("finalization exploded")

    monkeypatch.setattr(
        svar2_module._core, "run_conversion_pipeline", fail_during_finalization
    )

    with pytest.raises(RuntimeError, match="finalization exploded"):
        SparseVar2.from_vcf(
            tmp_path / "store",
            source,
            ref,
            threads=1,
            progress_callback=events.append,
        )

    assert events[-1].state == "failed"
    assert events[-1].phase == "finalizing"
    assert events[-1].percent == 95.0


def test_from_vcf_rejects_snapshot_inside_existing_output(tmp_path: Path):
    ref = _write_ref(tmp_path)
    source = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store"
    out.mkdir()
    marker = out / "original"
    marker.write_text("keep")

    with pytest.raises(ValueError, match="progress snapshot.*outside output"):
        SparseVar2.from_vcf(
            out,
            source,
            ref,
            threads=1,
            overwrite=True,
            progress_path=out / "progress.json",
        )

    assert marker.read_text() == "keep"
    assert not (out / "progress.json").exists()


def test_from_vcf_list_wires_structured_progress(tmp_path: Path):
    ref = _write_ref(tmp_path)
    a = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    b = _ss(tmp_path, "b", "SB", "chr1\t7\t.\tC\tT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "store"
    events: list[ProgressEvent] = []

    SparseVar2.from_vcf_list(
        out, [a, b], ref, threads=1, progress_callback=events.append
    )

    assert any(
        event.phase == "converting" and event.details.get("contig") == "chr1"
        for event in events
    )
    assert events[-1].operation == "svar2.from_vcf_list"
    assert events[-1].percent == 100.0
    assert (out / "meta.json").is_file()


def test_from_vcf_list_progress_preserves_check_ref_policy(tmp_path: Path):
    ref = _write_ref(tmp_path)
    bad = _ss(tmp_path, "bad", "SA", "chr1\t3\t.\tG\tT\t.\t.\t.\tGT\t1|0\n")
    good = _ss(tmp_path, "good", "SB", "chr1\t7\t.\tC\tT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "store"
    events: list[ProgressEvent] = []

    SparseVar2.from_vcf_list(
        out,
        [bad, good],
        ref,
        threads=1,
        check_ref="x",
        progress_callback=events.append,
    )

    assert events[-1].percent == 100.0
    assert int(SparseVar2(out).region_counts("chr1", [(0, 40)]).sum()) == 1


def test_from_vcf_list_rejects_env_snapshot_inside_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ref = _write_ref(tmp_path)
    source = _ss(tmp_path, "a", "SA", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    out = tmp_path / "store"
    monkeypatch.setenv("NF_SEQLAB_PROGRESS_PATH", str(out / "progress.json"))

    with pytest.raises(ValueError, match="progress snapshot.*outside output"):
        SparseVar2.from_vcf_list(out, [source], ref, threads=1)

    assert not out.exists()
