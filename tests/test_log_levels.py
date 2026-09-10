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


@pytest.mark.parametrize("bad", [True, False])
def test_bool_is_rejected_despite_being_an_int(bad: bool):
    # `bool` is a subclass of `int`, so without an explicit guard
    # `parse_log_level(True)` would silently mean logging level 1 -> "debug".
    with pytest.raises(ValueError, match="must not be a bool"):
        parse_log_level(bad)


@pytest.mark.parametrize("bad", [None, 3.5, (), ["info"]])
def test_non_str_non_int_is_rejected(bad: object):
    with pytest.raises(ValueError, match="log_level must be one of"):
        parse_log_level(bad)  # pyrefly: ignore[bad-argument-type]


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (" info ", "info"),
        ("\tDEBUG\n", "debug"),
        ("  Critical  ", "error"),
    ],
)
def test_surrounding_whitespace_and_case_are_ignored(given: str, expected: str):
    assert parse_log_level(given) == expected


def test_a_positive_int_below_debug_is_debug():
    # 1..9 sit below logging.DEBUG (10) but above NOTSET (0); the round-UP rule
    # makes them the most verbose real level rather than silence.
    assert parse_log_level(1) == "debug"
    assert parse_log_level(9) == "debug"
