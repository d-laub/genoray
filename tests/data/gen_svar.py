import shutil
from pathlib import Path

import joblib

from genoray import PGEN, VCF, SparseVar


def main():
    ddir = Path(__file__).parent

    vcf = VCF(ddir / "biallelic.vcf.gz", dosage_field="DS")
    vcf._write_gvi_index()

    vcf_path = ddir / "biallelic.vcf.svar"
    if vcf_path.exists():
        shutil.rmtree(vcf_path)

    max_mem = vcf._mem_per_variant(vcf.Genos8Dosages) * min(
        len(vcf.contigs), joblib.cpu_count()
    )
    SparseVar.from_vcf(vcf_path, vcf, max_mem, overwrite=True, with_dosages=True)
    SparseVar(vcf_path).cache_afs()

    pgen = PGEN(ddir / "biallelic.pgen", dosage_path=ddir / "biallelic.pgen")

    pgen_path = ddir / "biallelic.pgen.svar"
    if pgen_path.exists():
        shutil.rmtree(pgen_path)

    assert pgen.contigs is not None
    max_mem = pgen._mem_per_variant(pgen.GenosDosages) * min(
        len(pgen.contigs), joblib.cpu_count()
    )
    SparseVar.from_pgen(pgen_path, pgen, max_mem, overwrite=True, with_dosages=True)
    SparseVar(pgen_path).cache_afs()


if __name__ == "__main__":
    main()
