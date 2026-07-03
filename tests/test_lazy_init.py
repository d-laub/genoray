"""Regression: importing genoray must not pull in heavy optional deps.

Heavy deps are loaded lazily on first attribute access (PEP 562 __getattr__ in
__init__.py). Verified in a subprocess so that test-collection imports don't
pollute the result.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_genoray_does_not_load_heavy_deps():
    code = (
        "import sys\n"
        "import genoray\n"
        "heavy = {'cyvcf2', 'numba', 'polars_bio', 'pgenlib'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, f'genoray import pulled in heavy modules: {loaded}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
