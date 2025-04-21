from pathlib import Path

from genoray import PGEN, VCF, SparseVar


def main():
    ddir = Path(__file__).parent
    vcf = VCF(ddir / "biallelic.vcf.gz")
    pgen = PGEN(ddir / "biallelic.pgen")
    SparseVar.from_vcf(ddir / "biallelic.vcf.svar", vcf, "1g", overwrite=True)
    SparseVar.from_pgen(ddir / "biallelic.pgen.svar", pgen, "1g", overwrite=True)


if __name__ == "__main__":
    main()
