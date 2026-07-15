from __future__ import annotations

import os
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from tqdm.auto import tqdm

from genoray._io import atomic_write_path

ProgressState = Literal["running", "completed", "failed"]
ProgressCallback = Callable[["ProgressEvent"], None]

_ENV_PREFIX = "NF_SEQLAB_PROGRESS_"
_SNAPSHOT_ENV_NAMES = (
    f"{_ENV_PREFIX}PATH",
    f"{_ENV_PREFIX}FILE",
    f"{_ENV_PREFIX}SNAPSHOT_PATH",
)
_DEFAULT_PHASE_ORDER = (
    "preparing",
    "converting",
    "finalizing",
    "publishing",
    "complete",
)
_MANAGED_IDENTITY_FIELDS = (
    "run_id",
    "stage_id",
    "process",
    "file_id",
    "parent_file_id",
    "task_id",
    "attempt",
)
_logger = logging.getLogger("genoray.progress")


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One structured, JSON-serializable progress observation."""

    operation: str
    phase: str
    state: ProgressState
    completed: int
    total: int | None
    unit: str | None
    percent: float
    sequence: int
    timestamp: datetime
    identity: Mapping[str, str] = field(default_factory=dict)
    message: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": "genoray",
            "operation": self.operation,
            "phase": self.phase,
            "state": self.state,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
            "percent": self.percent,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "identity": dict(self.identity),
            "message": self.message,
            "details": dict(self.details),
        }


class SnapshotSink:
    """Persist the latest event as an atomically replaced JSON document."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __call__(self, event: ProgressEvent) -> None:
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = event.to_dict()
        if all(event.identity.get(field) for field in _MANAGED_IDENTITY_FIELDS):
            try:
                attempt = int(event.identity["attempt"])
            except (TypeError, ValueError):
                pass
            else:
                payload = {
                    "schema": "nf-seqlab.progress/v1",
                    **{
                        field: event.identity[field]
                        for field in _MANAGED_IDENTITY_FIELDS
                        if field != "attempt"
                    },
                    "attempt": attempt,
                    "state": event.state,
                    "phase": event.phase,
                    "completed": event.completed,
                    "total": event.total,
                    "unit": event.unit,
                    "percent": event.percent,
                    "message": event.message,
                    "updated_at": event.timestamp.isoformat().replace("+00:00", "Z"),
                }
        with atomic_write_path(self.path) as temporary:
            temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )


class NativeProgressSink:
    """Render structured events with tqdm without replacing other sinks."""

    def __init__(self) -> None:
        self._bar: Any | None = None
        self._percent = 0.0

    def __call__(self, event: ProgressEvent) -> None:
        if self._bar is None:
            self._bar = tqdm(total=100.0, unit="%", leave=True)
        self._bar.set_description_str(event.phase)
        delta = max(0.0, event.percent - self._percent)
        if delta:
            self._bar.update(delta)
        self._percent = event.percent
        if event.state == "failed" or (
            event.phase == "complete" and event.state == "completed"
        ):
            self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _identity_from_environment() -> dict[str, str]:
    excluded = {name.removeprefix(_ENV_PREFIX) for name in _SNAPSHOT_ENV_NAMES}
    return {
        name.removeprefix(_ENV_PREFIX).lower(): value
        for name, value in sorted(os.environ.items())
        if name.startswith(_ENV_PREFIX)
        and name.removeprefix(_ENV_PREFIX) not in excluded
        and value
    }


def _snapshot_path_from_environment() -> Path | None:
    for name in _SNAPSHOT_ENV_NAMES:
        if value := os.environ.get(name):
            return Path(value)
    return None


class ProgressContext:
    """Validate progress transitions and fan each event out to all configured sinks."""

    def __init__(
        self,
        operation: str,
        *,
        callback: ProgressCallback | None = None,
        snapshot_path: str | Path | None = None,
        identity: Mapping[str, object] | None = None,
        display: bool = False,
        phase_order: Sequence[str] = _DEFAULT_PHASE_ORDER,
    ) -> None:
        self.operation = operation
        self._phase_order = {phase: index for index, phase in enumerate(phase_order)}
        self._lock = threading.RLock()
        self._sequence = 0
        self._last_event: ProgressEvent | None = None
        self._terminal = False

        resolved_identity = _identity_from_environment()
        if identity is not None:
            resolved_identity.update(
                {str(key): str(value) for key, value in identity.items()}
            )
        self.identity: Mapping[str, str] = resolved_identity

        self._sinks: list[ProgressCallback] = []
        if callback is not None:
            self._sinks.append(callback)
        resolved_snapshot = (
            Path(snapshot_path)
            if snapshot_path is not None
            else _snapshot_path_from_environment()
        )
        self.snapshot_path = resolved_snapshot
        if resolved_snapshot is not None:
            self._sinks.append(SnapshotSink(resolved_snapshot))
        if display:
            self._sinks.append(NativeProgressSink())

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    def validate_snapshot_outside(self, directory: str | Path) -> None:
        """Reject snapshots that would mutate an atomically published directory."""
        if self.snapshot_path is None:
            return
        output = Path(directory).resolve()
        snapshot = self.snapshot_path.resolve()
        if snapshot == output or output in snapshot.parents:
            raise ValueError(
                "progress snapshot path must be outside output directory "
                f"{directory}: {self.snapshot_path}"
            )

    def __enter__(self) -> ProgressContext:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        if exc is not None and not self._terminal:
            last = self._last_event
            self.emit(
                phase=last.phase if last is not None else "preparing",
                state="failed",
                completed=last.completed if last is not None else 0,
                total=last.total if last is not None else None,
                unit=last.unit if last is not None else None,
                percent=last.percent if last is not None else 0.0,
                message=str(exc),
                details=last.details if last is not None else None,
            )
        return False

    def emit(
        self,
        *,
        phase: str,
        state: ProgressState,
        completed: int,
        total: int | None,
        unit: str | None,
        percent: float,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> ProgressEvent:
        with self._lock:
            if self._terminal:
                raise RuntimeError("cannot emit progress after a terminal event")
            if phase not in self._phase_order:
                raise ValueError(f"unknown progress phase: {phase!r}")
            if not 0.0 <= percent <= 100.0:
                raise ValueError("progress percent must be between 0 and 100")
            if completed < 0 or (total is not None and completed > total):
                raise ValueError("progress completed units must be within the total")

            previous = self._last_event
            if previous is not None:
                if percent < previous.percent:
                    raise ValueError("progress percent cannot decrease")
                if self._phase_order[phase] < self._phase_order[previous.phase]:
                    raise ValueError("progress phase cannot move backward")
                if (
                    phase == previous.phase
                    and previous.state == "completed"
                    and state == "running"
                ):
                    raise ValueError("progress state cannot reopen a completed phase")

            self._sequence += 1
            event = ProgressEvent(
                operation=self.operation,
                phase=phase,
                state=state,
                completed=completed,
                total=total,
                unit=unit,
                percent=float(percent),
                sequence=self._sequence,
                timestamp=datetime.now(timezone.utc),
                identity=dict(self.identity),
                message=message,
                details={} if details is None else dict(details),
            )
            self._last_event = event
            self._terminal = state == "failed" or (
                phase == "complete" and state == "completed"
            )
            active_sinks: list[ProgressCallback] = []
            for sink in self._sinks:
                try:
                    sink(event)
                except Exception:
                    _logger.warning(
                        "progress callback failed; disabling it",
                        exc_info=True,
                    )
                else:
                    active_sinks.append(sink)
            self._sinks = active_sinks
            return event


class _ConversionProgress:
    """SVAR2 conversion phase accounting over completed, disjoint contigs."""

    def __init__(self, context: ProgressContext, contigs: Sequence[str]) -> None:
        self.context = context
        self._contigs = frozenset(contigs)
        self._completed: set[str] = set()
        self._lock = threading.Lock()

    @property
    def callback(self) -> Callable[[str], None] | None:
        return self.contig_completed if self.context.enabled else None

    @property
    def finalizing_callback(self) -> Callable[[], None] | None:
        return self.finalizing if self.context.enabled else None

    def start(self) -> None:
        total = len(self._contigs)
        self.context.emit(
            phase="preparing",
            state="running",
            completed=0,
            total=total,
            unit="contig",
            percent=0.0,
        )
        self.context.emit(
            phase="preparing",
            state="completed",
            completed=0,
            total=total,
            unit="contig",
            percent=5.0,
        )
        self.context.emit(
            phase="converting",
            state="running",
            completed=0,
            total=total,
            unit="contig",
            percent=5.0,
        )

    def contig_completed(self, contig: str) -> None:
        with self._lock:
            if contig not in self._contigs:
                raise ValueError(f"progress reported unknown contig: {contig}")
            if contig in self._completed:
                raise ValueError(f"progress reported contig twice: {contig}")
            self._completed.add(contig)
            completed = len(self._completed)
            total = len(self._contigs)
            percent = 5.0 + 90.0 * completed / total
            self.context.emit(
                phase="converting",
                state="completed" if completed == total else "running",
                completed=completed,
                total=total,
                unit="contig",
                percent=percent,
                message=f"converted {contig}",
                details={"contig": contig, "span_kind": "contig"},
            )

    def finalizing(self) -> None:
        total = len(self._contigs)
        self.context.emit(
            phase="finalizing",
            state="running",
            completed=total,
            total=total,
            unit="contig",
            percent=95.0,
        )

    def finalized(self) -> None:
        total = len(self._contigs)
        self.context.emit(
            phase="finalizing",
            state="completed",
            completed=total,
            total=total,
            unit="contig",
            percent=99.0,
        )

    def publishing(self) -> None:
        total = len(self._contigs)
        self.context.emit(
            phase="publishing",
            state="running",
            completed=total,
            total=total,
            unit="contig",
            percent=99.0,
        )

    def complete(self) -> None:
        total = len(self._contigs)
        self.context.emit(
            phase="complete",
            state="completed",
            completed=total,
            total=total,
            unit="contig",
            percent=100.0,
        )
