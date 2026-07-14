from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Literal

Mode = Literal["copy", "hardlink", "symlink", "move"]


def _load_meta(store: Path) -> dict:
    return json.loads((Path(store) / "meta.json").read_text())


def _copy_contig_dir(src_dir: Path, dst_dir: Path, mode: Mode) -> None:
    if mode == "copy":
        shutil.copytree(src_dir, dst_dir)
    elif mode == "hardlink":
        shutil.copytree(src_dir, dst_dir, copy_function=os.link)
    elif mode == "symlink":
        dst_dir.symlink_to(src_dir.resolve(), target_is_directory=True)
    elif mode == "move":
        shutil.move(str(src_dir), str(dst_dir))
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _assert_concat_compatible(metas: list[dict]) -> None:
    ref = metas[0]
    for i, m in enumerate(metas[1:], start=1):
        for key in ("samples", "ploidy", "format_version", "fields"):
            if m.get(key) != ref.get(key):
                raise ValueError(
                    f"concat source #{i} differs from source #0 in {key!r}; "
                    "all stores must share samples, ploidy, format_version, and fields"
                )


def _write_store(
    output: Path,
    contig_sources: dict[str, Path],
    meta: dict,
    mode: Mode,
    overwrite: bool,
) -> None:
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for contig, src_store in contig_sources.items():
        _copy_contig_dir(Path(src_store) / contig, output / contig, mode)
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
