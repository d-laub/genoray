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

    # indels fixture (with_length edge cases)
    ivcf = VCF(ddir / "indels.vcf.gz", dosage_field="DS")
    ivcf._write_gvi_index()
    ivcf_path = ddir / "indels.vcf.svar"
    if ivcf_path.exists():
        shutil.rmtree(ivcf_path)
    max_mem = ivcf._mem_per_variant(ivcf.Genos8Dosages) * min(
        len(ivcf.contigs), joblib.cpu_count()
    )
    SparseVar.from_vcf(ivcf_path, ivcf, max_mem, overwrite=True, with_dosages=True)
    SparseVar(ivcf_path).cache_afs()

    ipgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    ipgen_path = ddir / "indels.pgen.svar"
    if ipgen_path.exists():
        shutil.rmtree(ipgen_path)
    assert ipgen.contigs is not None
    max_mem = ipgen._mem_per_variant(ipgen.GenosDosages) * min(
        len(ipgen.contigs), joblib.cpu_count()
    )
    SparseVar.from_pgen(ipgen_path, ipgen, max_mem, overwrite=True, with_dosages=True)
    SparseVar(ipgen_path).cache_afs()


if __name__ == "__main__":
    main()
