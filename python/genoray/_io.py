from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _unique_sibling(dest: Path, suffix: str) -> Path:
    """Return a not-yet-existing sibling path of *dest* with *suffix* injected.

    Used to move an existing output dir aside before an atomic swap. The name is
    hidden (leading dot) and keyed by PID; a counter disambiguates collisions.
    """
    base = dest.with_name(f".{dest.name}{suffix}.{os.getpid()}")
    candidate = base
    i = 0
    while candidate.exists():
        i += 1
        candidate = dest.with_name(f"{base.name}.{i}")
    return candidate


@contextmanager
def atomic_write_path(dest: Path) -> Iterator[Path]:
    """Write a single file atomically: yield a sibling temp path to write to, then ``os.replace`` it onto *dest* on clean exit.

    On any exception the temp file is removed and *dest* is left untouched.

    The temp file is created in ``dest.parent`` so the final rename stays on one
    filesystem (a true atomic rename, no cross-device copy).
    """
    dest = Path(dest)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    else:
        os.replace(tmp, dest)


@contextmanager
def atomic_write_dir(dest: Path) -> Iterator[Path]:
    """Write a directory atomically: yield a sibling staging dir to populate, then swap it into place on clean exit.

    Swap is backup-then-swap: if *dest* already exists it is first moved aside to a
    fresh sibling name (so the staging dir is renamed onto a *non-existent* path,
    which is portable on POSIX and Windows), then the moved-aside dir is removed.
    If the final rename fails, the moved-aside dir is rolled back. The staging dir
    is always cleaned up; on an exception in the body *dest* is left untouched.

    The staging dir is created in ``dest.parent`` (same filesystem) so the swap is
    a true atomic rename and intermediate writes never cross devices.
    """
    dest = Path(dest)
    staging = Path(
        tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    )
    backup: Path | None = None
    try:
        yield staging
        if dest.exists():
            backup = _unique_sibling(dest, ".old")
            os.replace(dest, backup)
        os.replace(staging, dest)
    except BaseException:
        # If we moved dest aside but failed before/at the swap, roll it back.
        if backup is not None and backup.exists() and not dest.exists():
            os.replace(backup, dest)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
