import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "make_matched_pgen",
    Path(__file__).resolve().parents[2] / "scripts" / "make_matched_pgen.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_plink2_cmd_shape():
    cmd = mod.plink2_cmd(Path("/d/chr21.bcf"), Path("/out/chr21"))
    assert cmd[0] == "plink2"
    assert "--bcf" in cmd and "/d/chr21.bcf" in cmd
    assert "--make-pgen" in cmd
    assert cmd[cmd.index("--out") + 1] == "/out/chr21"
