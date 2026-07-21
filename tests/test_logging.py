import io
import threading

import genoray._core as core
import pytest
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
    with write_reporting(progress=False, log_level="off") as rx:
        assert rx is None


def test_write_reporting_drains_and_joins():
    n_threads_before = threading.active_count()
    with write_reporting(progress=False, log_level="info") as rx:
        assert rx is not None
        # Simulate a producer finishing immediately by not sending anything;
        # context exit must drop the sender and join the drain thread.
    # After exit, no leaked drain thread.
    assert threading.active_count() == n_threads_before
