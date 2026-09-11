from __future__ import annotations

import re
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Literal

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

LOG_LEVELS = ("off", "critical", "error", "warning", "info", "debug")
# What `parse_log_level` returns. Narrower than the full `LOG_LEVELS` input
# set: "critical" is an accepted input spelling but never an output, since it
# canonicalizes to "error".
CanonicalLogLevel = Literal["off", "error", "warning", "info", "debug"]

# What Rust's `level_rank` understands. "critical" is an accepted spelling but
# not a distinct rank: `tracing` has no CRITICAL level, and no genoray call site
# emits `error!` or `warn!` today, so it gates the same (currently empty) set as
# "error".
_CANONICAL: dict[str, CanonicalLogLevel] = {
    "off": "off",
    "critical": "error",
    "error": "error",
    "warning": "warning",
    "info": "info",
    "debug": "debug",
}

# Python `logging` integer thresholds, ascending. An integer that falls between
# two named levels rounds UP to the more severe one, matching Python's own
# gate: a logger set to 25 suppresses INFO (20) and admits WARNING (30).
_INT_LEVELS: tuple[tuple[int, CanonicalLogLevel], ...] = (
    (10, "debug"),
    (20, "info"),
    (30, "warning"),
    (40, "error"),
)

# A `logging` constant written as text, which is all the CLI can deliver.
# Anchored via `fullmatch`, so "10 " is fine (stripped) but "info10" is not.
# `[0-9]` rather than `\d`: `\d` is Unicode-aware, and accepting "１０" as a log
# level would be an accident rather than a feature.
_INT_TEXT_RE = re.compile(r"[+-]?[0-9]+")

_HEARTBEAT_SECS = 5.0  # min seconds between throttled % lines per contig


class ProgressRenderer:
    """Render SVAR2 write events. Single-consumer: only the drain thread calls in."""

    def __init__(self, console: Console, show_bar: bool) -> None:
        self.console = console
        self.show_bar = show_bar
        self._live = bool(show_bar) and (console.is_terminal or console.is_jupyter)
        self._progress: Progress | None = None
        self._tasks: dict[str, TaskID] = {}
        self._done: dict[str, int] = {}
        self._totals: dict[str, int | None] = {}
        self._last_beat: dict[str, float] = {}
        if self._live:
            self._progress = Progress(
                TextColumn("[bold blue]{task.fields[chrom]}"),
                BarColumn(),
                TextColumn("{task.completed:,} var"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
            self._progress.start()

    def handle(self, event: tuple) -> None:
        tag = event[0]
        if tag == "contig_start":
            _, chrom, total, _, _ = event
            self._totals[chrom] = total
            self._done[chrom] = 0
            self._last_beat[chrom] = 0.0
            if self._progress is not None:
                self._tasks[chrom] = self._progress.add_task(
                    "", chrom=chrom, total=total
                )
        elif tag == "progress":
            _, chrom, delta, _, _ = event
            self._done[chrom] = self._done.get(chrom, 0) + int(delta)
            if self._progress is not None:
                self._progress.update(self._tasks[chrom], advance=int(delta))
            elif self.show_bar:
                self._maybe_beat(chrom)
        elif tag == "contig_done":
            _, chrom, kept, excluded, elapsed_ms = event
            secs = int(elapsed_ms) / 1000.0
            if self._progress is not None and chrom in self._tasks:
                self._progress.update(
                    self._tasks[chrom], completed=int(kept), total=int(kept)
                )
            self.console.print(
                f"[green][svar2][/green] {chrom} done: "
                f"{int(kept):,} kept, {int(excluded):,} excluded ({secs:.1f}s)"
            )
        elif tag == "log":
            _, level, chrom, message, _target = event
            style = {"warning": "yellow", "info": "cyan", "debug": "dim"}.get(level, "")
            prefix = f"[svar2] {chrom}: " if chrom else "[svar2] "
            self.console.print(
                f"[{style}]{prefix}{message}[/{style}]"
                if style
                else f"{prefix}{message}"
            )

    def _maybe_beat(self, chrom: str) -> None:
        now = time.monotonic()
        if now - self._last_beat.get(chrom, 0.0) < _HEARTBEAT_SECS:
            return
        self._last_beat[chrom] = now
        done = self._done.get(chrom, 0)
        total = self._totals.get(chrom)
        if total:
            pct = 100.0 * done / total
            self.console.print(f"[svar2] {chrom} {pct:4.0f}% ({done:,}/{total:,}) ...")
        else:
            self.console.print(f"[svar2] {chrom} {done:,} variants ...")

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None


def parse_log_level(log_level: str | int) -> CanonicalLogLevel:
    """Normalize a Python-convention level to the name Rust's gate understands.

    Accepts the names in `LOG_LEVELS` (case-insensitive), `logging` integer
    constants, and the decimal spelling of those integers (`"10"`). Returns one
    of "off", "error", "warning", "info", "debug".

    The digit-string spelling exists for the CLI: `--log-level` can only hand
    this function text, so without it `log_level=10` would work from Python
    while `--log-level 10` was rejected (#179).

    `"warn"` is deliberately NOT accepted: `logging.warn()` was removed in
    Python 3.13, so it is a deprecated spelling rather than a convention.
    """
    if isinstance(log_level, bool):  # bool is an int; nobody means this
        raise ValueError(
            f"log_level must not be a bool; got {log_level!r}. Pass one of "
            f"{LOG_LEVELS} or a logging level int."
        )
    if isinstance(log_level, int):
        if log_level < 0:
            raise ValueError(
                f"log_level as an int must be >= 0 (logging.NOTSET); got {log_level!r}"
            )
        if log_level == 0:
            # logging.NOTSET means "inherit from the parent logger"; there is no
            # parent here, so the only coherent reading is silence.
            return "off"
        for threshold, name in _INT_LEVELS:
            if log_level <= threshold:
                return name
        return "error"  # >= CRITICAL
    if isinstance(log_level, str):
        text = log_level.strip()
        canonical = _CANONICAL.get(text.lower())
        if canonical is not None:
            return canonical
        if _INT_TEXT_RE.fullmatch(text):
            # Delegate rather than re-deriving, so the digit spelling inherits
            # the round-up rule and the negative rejection unchanged.
            return parse_log_level(int(text))
    raise ValueError(
        f"log_level must be one of {LOG_LEVELS} (case-insensitive) or a "
        f"logging level int (or its decimal spelling); got {log_level!r}"
    )


@contextmanager
def write_reporting(
    progress: bool, log_level: str | int
) -> Iterator[tuple[object | None, str]]:
    level = parse_log_level(log_level)
    if not progress and level == "off":
        yield None, level
        return

    from genoray import _core

    console = Console()
    renderer = ProgressRenderer(console, show_bar=progress)
    rx = _core.PyEventReceiver()
    stop = threading.Event()

    def _drain() -> None:
        while not stop.is_set():
            try:
                ev = rx.recv_timeout(100)
            except StopIteration:
                break
            if ev is not None:
                try:
                    renderer.handle(ev)
                except Exception:
                    pass  # never let rendering crash the write
        # drain any straggling events after disconnect
        while True:
            try:
                ev = rx.recv_timeout(0)
            except StopIteration:
                break
            if ev is None:
                break
            try:
                renderer.handle(ev)
            except Exception:
                pass

    t = threading.Thread(target=_drain, name="genoray-log-drain", daemon=True)
    t.start()
    try:
        yield rx, level
    finally:
        stop.set()
        # Setting `stop` makes the drain loop exit within one recv_timeout(100)
        # tick; the bounded join below reaps it. StopIteration is never relied
        # on here: `rx` (PyEventReceiver) holds its own keep-alive `tx`, so the
        # channel never actually reports Disconnected.
        t.join(timeout=5.0)
        renderer.close()
