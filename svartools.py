#! /usr/bin/env python

from importlib.metadata import version
from pathlib import Path

from cyclopts import App

app = App(
    help_on_error=True,
    version=f"[cyan]svartools[/cyan] 0.1.0\n[magenta]genoray[/magenta] {version('genoray')}",
    version_format="rich",
    help="Tools for SVAR files.",
)


@app.command
def write(
    source: Path, out: Path, max_mem: str = "1g", overwrite: bool = False
) -> None:
    """
    Convert a VCF or PGEN file to a SVAR file.

    Parameters
    ----------
    source : Path
        Path to the input VCF or PGEN file.
    out : Path
        Path to the output SVAR file.
    max_mem : str, optional
        Maximum memory to use for conversion e.g. 1g, 250 MB, etc.
    overwrite : bool, optional
        Whether to overwrite the output file if it exists.
    """
    from genoray import PGEN, VCF, SparseVar
    from genoray._utils import variant_file_type

    file_type = variant_file_type(source)
    if file_type == "vcf":
        vcf = VCF(source)
        SparseVar.from_vcf(out, vcf, max_mem, overwrite)
    elif file_type == "pgen":
        pgen = PGEN(source)
        SparseVar.from_pgen(out, pgen, max_mem, overwrite)
    else:
        raise ValueError(f"Unsupported file type: {source}")


if __name__ == "__main__":
    app()
