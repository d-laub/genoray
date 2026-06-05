#! /bin/bash

set -e

ddir=$(dirname "$0")
ddir=$(realpath "$ddir")

bi=$ddir/biallelic.vcf
multi=$ddir/multiallelic.vcf
unsorted=$ddir/three_samples_unsorted.vcf
indels=$ddir/indels.vcf

echo "Generating VCFs from vcfixture builders..."
python "$ddir"/gen_vcfs.py

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

# Symbolic SVs: plink2 passes <DEL>/<INS>/<DUP> verbatim but may reject
# <CNV>/<INV> in some versions. We pre-filter to only the three precise SV
# types that PGEN ILEN tests require (rows 0-2: POS 1000/2000/3000).
sym=$ddir/symbolic.vcf
bgzip -c "$sym" >| "$sym".gz
bcftools index "$sym".gz
rm -f "$sym".gz.gvi

prefix="${sym%.vcf}"
# plink2 requires a seekable file; filter via a temp file and then discard it.
# bcftools -a drops alleles absent from the filtered rows so ALT counts are
# correct.  <CNV>/<INV> are excluded here — plink2 rejects them as unsupported
# alt allele types; the PGEN test only needs the precise SV rows.
tmp=$(mktemp /tmp/sym_precise.XXXXXX.vcf.gz)
bcftools view -a -i 'ALT="<DEL>" || ALT="<INS>" || ALT="<DUP>"' -Oz -o "$tmp" "$sym".gz
bcftools index "$tmp"
plink2 --make-pgen --vcf "$tmp" --out "$prefix" --vcf-half-call r
rm -f "$tmp" "$tmp".csi
rm -f "$prefix".log
rm -f "$prefix".pvar.gvi

echo "Converting VCF and PGEN to SVAR format..."
python "$ddir"/gen_svar.py