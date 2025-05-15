import shutil
from pathlib import Path

from genoray import PGEN, VCF, SparseVar


def main():
    ddir = Path(__file__).parent
    vcf = VCF(ddir / "biallelic.vcf.gz", dosage_field="DS")
    vcf._write_gvi_index()
    vcf._load_index()
    pgen = PGEN(ddir / "biallelic.pgen")
    vcf_path = ddir / "biallelic.vcf.svar"
    shutil.rmtree(vcf_path)
    pgen_path = ddir / "biallelic.pgen.svar"
    shutil.rmtree(pgen_path)
    SparseVar.from_vcf(vcf_path, vcf, "1g", overwrite=True, with_dosages=True)
    SparseVar.from_pgen(pgen_path, pgen, "1g", overwrite=True, with_dosages=True)


if __name__ == "__main__":
    main()
