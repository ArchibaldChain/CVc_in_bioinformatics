import csv
import sys
import time
from typing import List
import os

sys.path.append('./src')
from utils import timing
from cross_validation_correction.cross_validation import CrossValidation
from cross_validation_correction.methods import *
from arguments import get_args
from bimbam import Bimbam

class_dict = {
    'ols': OrdinaryLeastRegressor,
    'gls': GeneralizedLeastRegressor,
    'blup': BestLinearUnbiasedPredictor
}


def create_method(class_name, **kwargs):
    if class_name in class_dict:
        return class_dict[class_name](**kwargs)
    else:
        raise ValueError(f"No class found for name '{class_name}'")


def create_filename(save_path, method, num_large_effect, num_fixed_snps,
                    is_correcting, alpha, l1_ratio):
    os.makedirs(save_path, exist_ok=True)
    if is_correcting:
        temp = 'correction'
    else:
        temp = ''
    if alpha == 0:
        # filename example
        # blup_0_reg_l1_cv_correction_100_fixed_100_large.csv
        filename = f'{method}_{alpha}_reg_cv_{temp}_{num_fixed_snps}_fixed_{num_large_effect}_large.csv'

    else:
        # filename example
        # blup_10_reg_0.00_l1_cv__100_fixed_100_large.csv
        # blup_10_reg_0.50_l1_cv_correction_100_fixed_100_large.csv
        filename = f'{method}_{alpha}_reg_{l1_ratio:.2f}_l1_cv_{temp}_{num_fixed_snps}_fixed_{num_large_effect}_large.csv'

    return os.path.join(save_path, filename)


@timing
def cross_validation_simulation(
        bimbam_file='./bimbam_data/bimbam_10000_full_false_major_minor.txt',
        save_path='simulation_output',
        num_fixed_snps=500,
        simulation_times=1000,
        num_large_effect=10,
        large_effect=400,
        small_effect=2,
        n_folds=10,
        is_correcting=True,
        alpha=0,
        l1_ratio=1,
        method='blup'):

    bimbam = Bimbam(bimbam_file)
    bimbam.pheno_simulator(100)

    filename = create_filename(save_path=save_path,
                               method=method,
                               num_fixed_snps=num_fixed_snps,
                               num_large_effect=num_large_effect,
                               is_correcting=is_correcting)

    regressor = create_method(method, alpha=alpha, l1_ratio=l1_ratio)
    for i in range(simulation_times):
        start_time = time.time()
        print(f'--------- simulation: {i}  ---------')
        bimbam.pheno_simulator(num_large_effect=num_large_effect,
                               large_effect=large_effect,
                               small_effect=small_effect)

        # split the train and test
        bimbam_tr, bimbam_te = bimbam.train_test_split()
        cv = CrossValidation(
            regressor,
            n_folds=n_folds,
            is_correcting=is_correcting,
            var_method='gemma_lmm',
        )
        re = cv(bimbam_tr, indices_fixed_effects=slice(0, num_fixed_snps))
        print('Finished cross-validation')
        # testing
        re['mse_te'] = CrossValidation.mean_square_error(
            cv.predict(bimbam_te), bimbam_te.pheno)
        bimbam_fake = bimbam_tr.fake_sample_generate(bimbam_te.n)
        re['mes_fake'] = CrossValidation.mean_square_error(
            cv.predict(bimbam_fake), bimbam_fake.pheno)

        # create the correction
        if is_correcting:
            re['w_fake'] = cv.correct(bimbam_fake)
            re['w_te'] = cv.correct(bimbam_te)
            re['w_resample'] = cv.correct(bimbam_tr.resample(bimbam_te.n))
        print(re)

        with FileSaver(filename) as file_saver:
            file_saver.save_dict(re)

        end_time = time.time() - start_time
        print(f'---- simulation {i} used {end_time:.4f} seconds ----')


class FileSaver:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.header = None

    def __enter__(self):
        # 1. check if filename exists and store it in the object namesapece

        # If filename does not exist, create it
        if not os.path.isfile(self.filename) or\
            os.stat(self.filename).st_size == 0:
            self.file = open(self.filename, 'w', newline='')

        # If filename exists, read the first line and saperate it by comma into a list, store it in a object variable header using self.read_header
        else:
            self.file = open(self.filename, 'a',
                             newline='')  # Open in append mode
            self.header = self.read_header()
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def save_dict(self, data_dict: dict):
        # save the dictionay data
        writer = csv.DictWriter(self.file, fieldnames=data_dict.keys())
        # 1. check if the file is empty
        # If the file is empty, write the dictionary key as header in the first line
        if self.header is None:
            self.header = list(data_dict.keys())
            writer.writeheader()

        # 2. If not, write the dictionary object as its keys in order of header
        writer.writerow(data_dict)
        self.file.flush()  # Makes sure the data is written immediately

    def read_header(self) -> List[int]:
        # read the first line as the header, seperate it by comma
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
        return header

    def write_head(self, data_dict: dict):
        # write the data_dict keys into the first line as the header
        writer = csv.DictWriter(self.file, fieldnames=data_dict.keys())
        writer.writeheader()
        self.header = list(data_dict.keys())


if __name__ == '__main__':
    args = get_args()
    cross_validation_simulation(args.bimbam_path, args.save_path,
                                args.num_fixed_snps, args.simulation_times,
                                args.num_large_effect, args.large_effect,
                                args.small_effect, args.n_folds,
                                args.correcting, args.alpha, args.l1_ratio,
                                args.method)
