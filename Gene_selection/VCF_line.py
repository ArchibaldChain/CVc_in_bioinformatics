import os
import numpy as np
import time


class VCF_line:
    HEADER = None
    SI = None  # start index for the data
    N = None  # Number of indvidiuals

    def __init__(self, line: str) -> None:
        self.line = line
        self.words = line.split('\t')
        self.info = self.words[:self.SI]

        def split_to_int(x):
            return [int(a) for a in x.split('|')]

        self.data = np.array([split_to_int(x) for x in self.words[self.SI:]])

    def get_frequency(self) -> int:
        return np.sum(self.data) / (2 * self.N)

    def get_info(self, index) -> str:
        if isinstance(index, str):
            try:
                i = self.HEADER.index(index)
            except ValueError:
                print('String index not in the header')
                return None

        elif isinstance(index, int):
            i = index

        try:
            info = self.info[i]
        except IndexError:
            print('Index out of Bounds')
            return None

        return info

    def write_csv_line(self, filename):
        line = ','.join(self.words[self.SI:])
        if os.path.exists(filename):
            with open(filename, 'a') as f:
                f.write(line)
        else:
            with open(filename, 'w') as f:
                if self.HEADER is not None:
                    f.write(','.join(self.HEADER[self.SI:]))
                f.write(line)

    def write_vcf_line(self, filename):
        if os.path.exists(filename):
            with open(filename, 'a') as f:
                f.write(self.line)
        else:
            with open(filename, 'w') as f:
                if self.HEADER is not None:
                    f.write('#' + '\t'.join(self.HEADER))
                f.write(self.line)

    def write_vcf(lines, f, header=False):
        if header:
            f.write('#' + '\t'.join(VCF_line.HEADER))

        line = ''.join(lines)
        f.write(line)

    def set_headers(head_line: str):
        VCF_line.HEADER = head_line[1:].split('\t')
        VCF_line.SI = (VCF_line.HEADER).index('FORMAT') + 1
        VCF_line.N = len(VCF_line.HEADER) - VCF_line.SI


if __name__ == '__main__':
    filename = 'data/chr1First2000.vcf'
    with open(filename, 'r') as f:
        n = 1
        while n < 22:
            line = f.readline()

            if line.startswith('#') and (not line.startswith('##')):
                VCF_line.set_headers(line)
                print(VCF_line.HEADER.index('FILTER'))
                print(VCF_line.HEADER[:VCF_line.SI])

            if not line.startswith('#'):
                vcf = VCF_line(line)
                print(vcf.get_info('INFO'))
                print(vcf.get_frequency())
                print(vcf.data)
                print(vcf.N)
            n += 1
