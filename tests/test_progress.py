from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import genoray._progress as progress_module
from genoray._progress import ProgressContext, ProgressEvent, SnapshotSink


def _event(**overrides: object) -> ProgressEvent:
    values: dict[str, object] = {
        "operation": "svar2.from_vcf",
        "phase": "converting",
        "state": "running",
        "completed": 1,
        "total": 2,
        "unit": "contig",
        "percent": 50.0,
        "sequence": 2,
        "timestamp": datetime(2026, 7, 14, tzinfo=timezone.utc),
        "identity": {"run_id": "run-7"},
        "message": "converted chr1",
        "details": {"contig": "chr1"},
    }
    values.update(overrides)
    return ProgressEvent(**values)  # type: ignore[arg-type]


def test_progress_event_is_json_serializable():
    payload = _event().to_dict()

    assert payload == {
        "schema_version": 1,
        "source": "genoray",
        "operation": "svar2.from_vcf",
        "phase": "converting",
        "state": "running",
        "completed": 1,
        "total": 2,
        "unit": "contig",
        "percent": 50.0,
        "sequence": 2,
        "timestamp": "2026-07-14T00:00:00+00:00",
        "identity": {"run_id": "run-7"},
        "message": "converted chr1",
        "details": {"contig": "chr1"},
    }
    json.dumps(payload)


def test_progress_event_mappings_are_immutable():
    event = _event()

    with pytest.raises(TypeError):
        event.identity["run_id"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        event.details["contig"] = "chr2"  # type: ignore[index]


def test_snapshot_sink_atomically_replaces_json(tmp_path):
    path = tmp_path / "progress" / "snapshot.json"
    sink = SnapshotSink(path)

    sink(_event(sequence=1, percent=10.0))
    sink(_event(sequence=2, percent=50.0))

    assert json.loads(path.read_text())["sequence"] == 2
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_snapshot_sink_writes_managed_nf_seqlab_protocol(tmp_path):
    path = tmp_path / ".nf-seqlab-progress.json"
    identity = {
        "run_id": "aou-v8",
        "stage_id": "build_svar2",
        "process": "SEQLAB_BUILD_SVAR2",
        "file_id": "chr22",
        "parent_file_id": "chr22",
        "task_id": "99/b6e5e2",
        "attempt": "2",
    }

    SnapshotSink(path)(
        _event(
            phase="complete",
            state="completed",
            completed=2,
            total=2,
            percent=100.0,
            identity=identity,
            message=None,
        )
    )

    assert json.loads(path.read_text()) == {
        "schema": "nf-seqlab.progress/v1",
        "run_id": "aou-v8",
        "stage_id": "build_svar2",
        "process": "SEQLAB_BUILD_SVAR2",
        "file_id": "chr22",
        "parent_file_id": "chr22",
        "task_id": "99/b6e5e2",
        "attempt": 2,
        "state": "completed",
        "phase": "complete",
        "completed": 2,
        "total": 2,
        "unit": "contig",
        "percent": 100.0,
        "message": None,
        "updated_at": "2026-07-14T00:00:00Z",
    }


def test_context_reads_env_identity_and_explicit_values_win(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    snapshot = tmp_path / "snapshot.json"
    monkeypatch.setenv("NF_SEQLAB_PROGRESS_PATH", str(snapshot))
    monkeypatch.setenv("NF_SEQLAB_PROGRESS_RUN_ID", "run-env")
    monkeypatch.setenv("NF_SEQLAB_PROGRESS_TASK_ID", "task-env")
    monkeypatch.setenv("NF_SEQLAB_PROGRESS_ATTEMPT", "3")
    events: list[ProgressEvent] = []
    context = ProgressContext(
        "svar2.from_vcf",
        callback=events.append,
        identity={"task_id": "task-explicit", "cohort": "ADNI"},
    )

    context.emit(
        phase="preparing",
        state="running",
        completed=0,
        total=2,
        unit="contig",
        percent=0.0,
    )

    assert events[0].identity == {
        "attempt": "3",
        "run_id": "run-env",
        "task_id": "task-explicit",
        "cohort": "ADNI",
    }
    assert json.loads(snapshot.read_text())["identity"] == events[0].identity


def test_context_disables_a_failing_progress_callback():
    calls = 0

    def fail(_event: ProgressEvent) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("dashboard unavailable")

    context = ProgressContext("svar2.from_vcf", callback=fail)
    context.emit(
        phase="preparing",
        state="running",
        completed=0,
        total=2,
        unit="contig",
        percent=0.0,
    )
    context.emit(
        phase="preparing",
        state="completed",
        completed=0,
        total=2,
        unit="contig",
        percent=5.0,
    )

    assert calls == 1


def test_context_rejects_regressive_phase_and_percent():
    context = ProgressContext("svar2.from_vcf")
    context.emit(
        phase="converting",
        state="running",
        completed=1,
        total=2,
        unit="contig",
        percent=50.0,
    )

    with pytest.raises(ValueError, match="percent"):
        context.emit(
            phase="converting",
            state="running",
            completed=0,
            total=2,
            unit="contig",
            percent=40.0,
        )
    with pytest.raises(ValueError, match="phase"):
        context.emit(
            phase="preparing",
            state="running",
            completed=1,
            total=2,
            unit="contig",
            percent=50.0,
        )


def test_context_does_not_reopen_a_completed_phase():
    context = ProgressContext("svar2.from_vcf")
    context.emit(
        phase="converting",
        state="completed",
        completed=1,
        total=2,
        unit="contig",
        percent=50.0,
    )

    with pytest.raises(ValueError, match="state"):
        context.emit(
            phase="converting",
            state="running",
            completed=2,
            total=2,
            unit="contig",
            percent=95.0,
        )


def test_context_emits_failed_state_without_claiming_100_percent():
    events: list[ProgressEvent] = []

    with pytest.raises(RuntimeError, match="conversion failed"):
        with ProgressContext("svar2.from_vcf", callback=events.append) as context:
            context.emit(
                phase="converting",
                state="running",
                completed=1,
                total=2,
                unit="contig",
                percent=50.0,
            )
            raise RuntimeError("conversion failed")

    assert events[-1].state == "failed"
    assert events[-1].percent == 50.0
    assert events[-1].message == "conversion failed"


def test_callback_and_native_display_receive_the_same_events(monkeypatch):
    callback_events: list[ProgressEvent] = []
    display_events: list[ProgressEvent] = []
    monkeypatch.setattr(
        progress_module,
        "NativeProgressSink",
        lambda: display_events.append,
    )
    context = ProgressContext(
        "svar2.from_vcf", callback=callback_events.append, display=True
    )

    context.emit(
        phase="preparing",
        state="running",
        completed=0,
        total=2,
        unit="contig",
        percent=0.0,
    )

    assert callback_events == display_events
