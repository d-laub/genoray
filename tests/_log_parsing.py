"""Shared parsing for the write-time log lines that tests capture.

Rich injects ANSI escapes *inside* rendered numbers and wraps long lines, so
any test that reads a value off a banner has to de-style and de-wrap before
matching. Three test modules carried near-identical copies of that logic
(#180), which is three places for a parsing assertion to silently start
matching nothing.
"""

from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def plain_text(captured_text: str) -> str:
    """Strip Rich's ANSI styling and collapse its line-wrapping to single spaces."""
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", captured_text))


def log_field_int(captured_text: str, field: str) -> int:
    """Return the value of ``<field>=<int>`` logged on a captured tracing event.

    Asserts rather than returning `None`: every caller treats a missing field
    as a test failure, and the assertion carries the raw capture for triage.
    """
    m = re.search(rf"{re.escape(field)}=(\d+)", plain_text(captured_text))
    assert m is not None, (
        f"no {field} field found in captured output:\n{captured_text!r}"
    )
    return int(m.group(1))
