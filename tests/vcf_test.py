
import io
import os
from venv import main
import pandas as pd


def read_vcf(path, n):
    with open(path, 'r') as f:
        lines = []
        metadata = []
        for i in range(1, n):
            line = f.readline()
            if line.startswith('##'):
                metadata.append(metadata)
            if line.startswith('#'):
                header = line
            if not line.startswith('##'):
                lines.append(line)
        f.close()
    return metadata, header, lines
    # return pd.read_csv(
    #     io.StringIO(''.join(lines)),
    #     dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
    #            'QUAL': str, 'FILTER': str, 'INFO': str},
    #     sep='    ', engine = 'python'
    # ).rename(columns={'#CHROM': 'CHROM'})


def write_vcf_to_csv(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)


if __name__ == '__main__':
    metadata, lines, headers = read_vcf('chr1.vcf', 3000)
    for line in metadata:
        print(line[3])
