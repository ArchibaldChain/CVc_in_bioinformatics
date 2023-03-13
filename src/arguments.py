import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--SNP_file',
                        type=str,
                        default='data/SNP_in_200GENE_chr1.csv',
                        help="SNP file Path")
    # path related arguments
    parser.add_argument('--save_path',
                        type=str,
                        default='simulation_output/frequentist_CVc',
                        help="Save path to simulation error output")
    parser.add_argument(
        '--bimbam_path',
        type=str,
        default='./bimbam_data/bimbam_10000_full_false_major_minor.txt')
    parser.add_argument(
        '--bslmm_save_path',
        type=str,
        default='./simulation_output/CVc_error_simulation_bslmm.csv')

    # phenotype generation related arguments
    parser.add_argument('--num_large_effect', type=int, default=10)
    parser.add_argument('--large_effect', type=float, default=400.0)
    parser.add_argument('--small_effect', type=float, default=2.0)

    # simulation related arguments
    parser.add_argument(
        '--num_fixed_snps',
        type=int,
        default=500,
        help=
        "Number of fixed terms (SNPs). Set -1 means using all SNPs as fixed terms. "
    )
    parser.add_argument('--simulation_times', type=int, default=1000)
    parser.add_argument('--n_folds', type=int, default=10)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    print(args.simulation_times)
