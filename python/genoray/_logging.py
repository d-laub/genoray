from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Literal

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

LOG_LEVELS = ("off", "warning", "info", "debug")
LogLevel = Literal["off", "warning", "info", "debug"]

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


def resolve_log_level(log_level: str) -> str:
    if log_level not in LOG_LEVELS:
        raise ValueError(f"log_level must be one of {LOG_LEVELS}; got {log_level!r}")
    env = os.environ.get("GENORAY_LOG", "").strip().lower()
    if env in LOG_LEVELS:
        return env
    return log_level


@contextmanager
def write_reporting(
    progress: bool, log_level: str
) -> Iterator[tuple[object | None, str]]:
    level = resolve_log_level(log_level)
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
        # Dropping the Rust-side pipeline senders triggers StopIteration in the
        # drain loop; the receiver's own internal sender is released when `rx`
        # is garbage-collected. Give the drain a bounded join.
        t.join(timeout=5.0)
        renderer.close()
