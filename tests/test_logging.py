import io

import genoray._core as core
from genoray._logging import ProgressRenderer
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
