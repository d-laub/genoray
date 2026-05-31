"""Render every fixture in fixtures.FIXTURES to tests/data/<name>.vcf."""

from __future__ import annotations

from pathlib import Path

from fixtures import FIXTURES  # run from tests/data/ via gen_from_vcf.sh


def main() -> None:
    ddir = Path(__file__).parent
    for name, build in FIXTURES.items():
        out = ddir / f"{name}.vcf"
        build().write(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
