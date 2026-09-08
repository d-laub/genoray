from __future__ import annotations

import logging

import pytest

from genoray._logging import LOG_LEVELS, parse_log_level


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("off", "off"),
        ("debug", "debug"),
        ("info", "info"),
        ("warning", "warning"),
        ("error", "error"),
        # "critical" is an alias: tracing has no CRITICAL level.
        ("critical", "error"),
        # Case-insensitive, like logging.getLevelName's own inputs.
        ("DEBUG", "debug"),
        ("Warning", "warning"),
        ("CRITICAL", "error"),
    ],
)
def test_named_levels(given, expected):
    assert parse_log_level(given) == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (logging.DEBUG, "debug"),
        (logging.INFO, "info"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
        (logging.CRITICAL, "error"),
        (0, "off"),  # logging.NOTSET: no parent to inherit from -> silence
        # Between named levels, round UP to the more severe one: a logger set
        # to 25 suppresses INFO(20) and admits WARNING(30).
        (25, "warning"),
        (11, "info"),
        (35, "error"),
        (99, "error"),
    ],
)
def test_integer_levels(given, expected):
    assert parse_log_level(given) == expected


def test_warn_is_rejected_as_deprecated():
    with pytest.raises(ValueError) as excinfo:
        parse_log_level("warn")
    # The message must name the accepted set, since "warn" is what tracing
    # calls this level and is a natural thing to type.
    assert "warning" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["", "verbose", "trace", "none", "quiet"])
def test_unknown_names_rejected(bad):
    with pytest.raises(ValueError):
        parse_log_level(bad)


def test_negative_int_rejected():
    with pytest.raises(ValueError):
        parse_log_level(-1)


def test_accepted_spellings_are_exported():
    assert set(LOG_LEVELS) == {
        "off",
        "critical",
        "error",
        "warning",
        "info",
        "debug",
    }
    # Every advertised spelling must actually parse.
    for name in LOG_LEVELS:
        parse_log_level(name)
