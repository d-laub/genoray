"""The `pipeline config` line must report every knob and where it came from.

On genoray 4.0.1 this line printed the planner's `reader_workers` while an
environment variable had set a different value, and a downstream operator spent
a day chasing the difference. With the environment channel gone the line is
correct by construction -- these tests hold it to reporting provenance too.
"""

from __future__ import annotations

import re

from genoray import SparseVar2, Tuning

# `small_vcf`/`small_pgen` (a 3-record, 2-sample, single-contig cohort) live in
# tests/conftest.py -- these tests only assert on a log line, not on conversion
# output, so that tiny shared cohort is plenty and this module is not the only
# consumer anymore (see tests/test_tuning.py).

# `capfd`, not `capsys`: the `pipeline config` line is written by Rust's
# `tracing` fmt layer straight to fd 2 (`with_writer(std::io::stderr)`),
# below Python, so only an OS-fd-level capture (`capfd`) can see it.


def _banner(caplog_text: str) -> dict[str, str]:
    """Parse the flat `key=value` pairs out of the `pipeline config` line."""
    for line in caplog_text.splitlines():
        if "pipeline config" not in line:
            continue
        return dict(re.findall(r"(\w+)=([^\s]+)", line))
    raise AssertionError(f"no `pipeline config` line found in:\n{caplog_text}")


def test_explicit_values_are_tagged_explicit(tmp_path, small_vcf, capfd):
    out = tmp_path / "explicit.svar"
    SparseVar2.from_vcf(
        out,
        small_vcf,
        no_reference=True,
        tuning=Tuning(reader_workers=2, overshard=3, dense_cap=7),
        log_filter="genoray=info",
    )
    fields = _banner(capfd.readouterr().err)
    assert fields["reader_workers"] == "2"
    assert fields["reader_workers_src"] == "explicit"
    assert fields["overshard"] == "3"
    assert fields["overshard_src"] == "explicit"
    assert fields["dense_cap"] == "7"
    assert fields["dense_cap_src"] == "explicit"


def test_unset_values_are_tagged_planner(tmp_path, small_vcf, capfd):
    out = tmp_path / "planner.svar"
    SparseVar2.from_vcf(out, small_vcf, no_reference=True, log_filter="genoray=info")
    fields = _banner(capfd.readouterr().err)
    for knob in (
        "concurrent_chroms",
        "reader_workers",
        "overshard",
        "dense_cap",
        "merge_threads",
        "sample_interval",
    ):
        assert fields[f"{knob}_src"] == "planner", knob


def test_banner_omits_knobs_the_backend_cannot_use(tmp_path, small_pgen, capfd):
    out = tmp_path / "pgen.svar"
    SparseVar2.from_pgen(out, small_pgen, no_reference=True, log_filter="genoray=info")
    fields = _banner(capfd.readouterr().err)
    # from_pgen pins P=1 and never shards within a contig, so advertising these
    # would be advertising an inert knob.
    assert "overshard" not in fields
    assert "reader_workers_src" not in fields
    assert fields["concurrent_chroms_src"] == "planner"
