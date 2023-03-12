from calendar import c
import os
from VCF_line import VCF_line
from typing import List
import vcf_preprocessing as vp
import numpy as np
import pandas as pd


def gene_annotation_reader(file):
    with open(file, 'r') as f:
        annotations_text = f.read()
        annotations = annotations_text.split('\n')
        if len(annotations[-1]) == 0:
            annotations = annotations[:-1]
        genes = [anno.split('\t')[0] for anno in annotations]

        def get_position(anno):
            pos = anno.split('\t')[1].split(':')[1:3]
            return [int(x) for x in pos]

        gene_positions = [get_position(anno) for anno in annotations]

    return pd.DataFrame({'GENE': genes, 'POS': gene_positions})


def SNP_selection(lines: List[str], write_file: str, genes, gene_positions) -> None:
    # since gene annotation is in order, we are going to use dynamic programming to save some time.

    def is_gene(pos: int, gene_position: List[int])-> int:
        return (gene_position[0] <= pos and pos <= gene_position[1])

    filtered = []
    for line in lines:
        vcf = VCF_line(line)
        pos = int(vcf.get_info('POS'))
        for i, gene_position in enumerate(gene_positions):
            if is_gene(pos, gene_position):
                data = [str(np.sum(x)) for x in vcf.data]
                temp = [genes[i], str(pos)] + data
                filtered.append(','.join(temp))

    # write to csv
    print('Selected ', len(filtered), ' genes')
    if not os.path.exists(write_file):
        with open(write_file, 'w') as f:
            f.write('GENE,POS,' + ','.join(vcf.HEADER[vcf.SI:]))
    if len(filtered) > 0:
        with open(write_file, 'a') as f:
            f.write('\n'.join(filtered) + '\n')


def testing(read_file, args):
    with open(read_file, 'r') as f:
        # Set Headr

        chunk_size = 1000
        lines = []
        n = 0
        for i in range(chunk_size):
            count = i+1
            line = f.readline()
            if line.startswith('#') and not line.startswith('##'):
                VCF_line.set_headers(line)
                continue

            if len(line) == 0:
                break
            lines.append(line)

        SNP_selection(lines, *args)
        print('Finished ' + str(n + count) + ' times')
        n += chunk_size


if __name__ == '__main__':
    anno_file = 'Gene Annotation/1.txt'
    read_file = 'data/filtered_test.vcf'
    write_file = 'data/test.csv'
    print(write_file)

    gene_df = gene_annotation_reader(anno_file)

    # select some genes
    n = len(gene_df)
    print('Choose ', n, ' Genes:')

    df = gene_df.sample(n, replace=False, random_state=1)
    # df = gene_df
    genes = df['GENE'].to_list()
    gene_positions = df['POS'].to_list()

    print(genes)
    df.to_csv('data/selected_genes.csv')

    # Get SNP from gene annotation
    print('Get Annotation')
    if os.path.exists(write_file):
        pa = (write_file).split('\\')
        pa[-1] = pa[-1] + '.csv'
        write_file = '\\'.join(pa)

    args = [write_file, genes, gene_positions]
    vp.chunk_read(read_file, SNP_selection, args, chunk_size=200)
    # testing(readfile, args)
