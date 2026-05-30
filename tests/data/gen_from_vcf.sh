#! /bin/bash

set -e

ddir=$(dirname "$0")
ddir=$(realpath "$ddir")

bi=$ddir/biallelic.vcf
multi=$ddir/multiallelic.vcf
unsorted=$ddir/three_samples_unsorted.vcf

echo "Bgzipping and indexing VCF files..."
bgzip -c "$bi" >| "$bi".gz
bcftools index "$bi".gz
rm -f "$bi".gz.gvi

bgzip -c "$multi" >| "$multi".gz
bcftools index "$multi".gz
rm -f "$multi".gz.gvi

bgzip -c "$unsorted" >| "$unsorted".gz
bcftools index "$unsorted".gz
rm -f "$unsorted".gz.gvi

indels=$ddir/indels.vcf
bgzip -c "$indels" >| "$indels".gz
bcftools index "$indels".gz
rm -f "$indels".gz.gvi

echo "Converting VCF to PLINK format..."
prefix="${bi%.vcf}"
plink2 --make-pgen --vcf "$bi".gz 'dosage=DS' --out "$prefix" --vcf-half-call r
rm -f "$prefix".log
rm -f "$prefix".pvar.gvi

prefix="${multi%.vcf}"
plink2 --make-pgen --vcf "$multi".gz --out "$prefix" --vcf-half-call r
rm -f "$prefix".log
rm -f "$prefix".pvar.gvi

prefix="${bi%.vcf}.zst"
plink2 --make-pgen vzs --vcf "$bi".gz 'dosage=DS' --out "$prefix" --vcf-half-call r
rm -f "$prefix".log
rm -f "$prefix".pvar.zst.gvi

prefix="${indels%.vcf}"
plink2 --make-pgen --vcf "$indels".gz 'dosage=DS' --out "$prefix" --vcf-half-call r
rm -f "$prefix".log
rm -f "$prefix".pvar.gvi

echo "Converting VCF and PGEN to SVAR format..."
python "$ddir"/gen_svar.py