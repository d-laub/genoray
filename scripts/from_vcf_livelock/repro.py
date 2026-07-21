import argparse
import subprocess
import sys
import textwrap


def run(bcf, out, threads, timeout):
    code = textwrap.dedent(f"""
        from genoray import SparseVar2, FormatField
        SparseVar2.from_vcf({out!r}, {bcf!r}, no_reference=True,
            format_fields=[FormatField("VAF")], threads={threads},
            chunk_size=5000, overwrite=True)
    """)
    try:
        subprocess.run([sys.executable, "-c", code], timeout=timeout, check=True)
        return 0
    except subprocess.TimeoutExpired:
        return 124


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bcf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args()
    sys.exit(run(a.bcf, a.out, a.threads, a.timeout))
