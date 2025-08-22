#! /usr/bin/env python

from importlib.metadata import version
from pathlib import Path
from typing import Optional, Union

from cyclopts import App

app = App(
    help_on_error=True,
    version=f"[cyan]svartools[/cyan] 0.1.0\n[magenta]genoray[/magenta] {version('genoray')}",
    version_format="rich",
    help="Tools for SVAR files.",
)


@app.command
def index(source: Path):
    """Create a genoray index for a VCF or PGEN file."""
    from genoray import PGEN, VCF
    from genoray._utils import variant_file_type

    file_type = variant_file_type(source)
    if file_type == "vcf":
        vcf = VCF(source)
        vcf._write_gvi_index()
    elif file_type == "pgen":
        _ = PGEN(source)
    else:
        raise ValueError(f"Unsupported file type: {source}")


@app.command
def write(
    source: Path,
    out: Path,
    max_mem: str = "1g",
    overwrite: bool = False,
    dosages: Optional[Union[bool, Path]] = None,
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
    dosages : bool | Path | None, optional
        Whether to write dosages. If :code:`source` is a PGEN, this must be a path to a PGEN of dosages.
        If :code:`source` is a VCF, this should be a boolean.
    """
    from genoray import PGEN, VCF, SparseVar
    from genoray._utils import variant_file_type

    file_type = variant_file_type(source)
    if file_type == "vcf":
        vcf = VCF(source)
        SparseVar.from_vcf(out, vcf, max_mem, overwrite, with_dosages=dosages)
    elif file_type == "pgen":
        if dosages is False:
            dosages = None
            with_dosages = False
        elif dosages is True:
            raise ValueError(
                "Dosages must be provided as a path to a PGEN if source is a PGEN."
            )
        else:
            with_dosages = True

        pgen = PGEN(source, dosage_path=dosages)
        SparseVar.from_pgen(out, pgen, max_mem, overwrite, with_dosages=with_dosages)
    else:
        raise ValueError(f"Unsupported file type: {source}")


if __name__ == "__main__":
    app()
