#! /bin/bash

set -e

ddir=$(dirname "$0")
ddir=$(realpath "$ddir")

bi=$ddir/biallelic.vcf
multi=$ddir/multiallelic.vcf

echo "Bgzipping and indexing VCF files..."
bgzip -c "$bi" >| "$bi".gz
bcftools index "$bi".gz
rm -f "$bi".gz.gvi

bgzip -c "$multi" >| "$multi".gz
bcftools index "$multi".gz
rm -f "$multi".gz.gvi

echo "Converting VCF to PLINK format..."
prefix="${bi%.vcf}"
plink2 --make-pgen --vcf "$bi".gz 'dosage=DS' --out "$prefix" --vcf-half-call r
rm -f "$prefix".log
rm -f "$prefix".pvar.gvi

prefix="${multi%.vcf}"
plink2 --make-pgen --vcf "$multi".gz --out "$prefix" --vcf-half-call r
rm -f "$prefix".log
rm -f "$prefix".pvar.gvi


echo "Converting VCF and PGEN to SVAR format..."
python "$ddir"/gen_svar.py