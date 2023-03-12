from concurrent.futures import process
import os
import numpy as np
import time
from VCF_line import VCF_line


def vcf_filtering(lines, write_file):
    filtered = []
    for line in lines:
        if line.startswith('#'):
            continue
        vcf = VCF_line(line)
        if vcf.get_frequency() > 0.05 and vcf.get_info('FILTER') == 'PASS':
            # vcf.write_vcf_line(write_file)
            filtered.append(line)

    if os.path.exists(write_file):
        header = False
    else:
        header = True

    with open(write_file, 'a') as f:
        VCF_line.write_vcf(filtered, f, header)


def chunk_read(read_file, process_fun, args, chunk_size=1000):

    with open(read_file, 'r') as f:
        # Set Headr
        flag = True
        while flag:
            line = f.readline()
            if line.startswith('##'):
                continue
            elif line.startswith('#'):
                VCF_line.set_headers(line)
            else:
                flag = False

        # Chunk Read
        flag = True
        count = 0
        n = 0
        while flag:
            lines = []

            # One Chunk
            for i in range(chunk_size):
                count = i+1
                lines.append(line)
                line = f.readline()
                if len(line) == 0:
                    flag = False
                    break

            process_fun(lines, *args)
            print('Finished ' + str(n + count) + ' times')
            n += chunk_size


if __name__ == "__main__":

    file = 'Chr1First2000.vcf'
    write_file = 'filtered_test.vcf'

    if os.path.exists(write_file):
        os.remove(write_file)

    st = time.time()
    chunk_read(file, vcf_filtering, [write_file], chunk_size=1000)
    et = time.time()
    print("--- %s seconds ---" % (et - st))
    # test_vcf(file)
