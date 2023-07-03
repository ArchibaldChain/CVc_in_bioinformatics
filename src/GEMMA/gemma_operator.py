from typing import List, Union
import numpy as np
import pandas as pd
import os
import shutil

command = "./gemma-0.98.5"
if not os.path.exists(command):
    raise FileNotFoundError('gemma-0.98.5 operation not found')

temp_dir = '.temp_dir'
os.system("chmod +x " + command)
# print(os.getcwd())


def gemma_bslmm_train(train_data,
                      phenotype,
                      prefix: str = "bslmm_tr_output",
                      related_matrix=None):

    # if we get the data, we store it on temporary files.
    save_data = type(train_data) == pd.DataFrame and type(
        phenotype) == np.ndarray
    if save_data:
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        rand_number = str(np.random.randint(100000))
        train_file = os.path.join(temp_dir, 'geno_tr' + rand_number)
        phenotype_file = os.path.join(temp_dir, 'pheno_tr' + rand_number)
        train_data.to_csv(train_file, index=False, header=False)
        np.savetxt(phenotype_file, phenotype, fmt="%s")

    # if we get the files, then train with the files.
    elif type(train_data) == str and type(phenotype) == str:
        if not os.path.exists(phenotype):
            raise FileNotFoundError(f'Phenotype file {phenotype} not found')
        if not os.path.exists(train_data):
            raise FileNotFoundError(f'Genotype file {train_data} not found')
        train_file = train_data
        phenotype_file = phenotype

    else:
        raise Exception('Phenotype or genotype file format not supported')

    if not os.path.exists(phenotype_file):
        raise FileNotFoundError(f"Could not find {phenotype_file}")

    print('\n>>> GEMMA BSLMM Train:')
    if related_matrix is None:
        print(command +
              f" -g {train_file} -p {phenotype_file} -o {prefix} -bslmm 1")
        os.system(command +
                  f" -g {train_file} -p {phenotype_file} -o {prefix} -bslmm 1")
    else:
        if isinstance(related_matrix, np.ndarray):
            related_matrix_dir = os.path.join(temp_dir,
                                              'relatedness_dir' + rand_number)
            np.savetxt(related_matrix_dir, related_matrix)
        elif isinstance(related_matrix, str):
            related_matrix_dir = related_matrix
        else:
            raise Exception("Relateness format not supported.")

        print(
            command +
            f" -g {train_file} -p {phenotype_file} -o {prefix} -bslmm 1 -k {related_matrix_dir}"
        )
        os.system(
            command +
            f" -g {train_file} -p {phenotype_file} -o {prefix} -bslmm 1 -k {related_matrix_dir}"
        )

        os.remove(related_matrix_dir)

    if save_data:
        print(train_file)
        print(os.path.exists(train_file))
        os.remove(train_file)
        os.remove(phenotype_file)


def gemma_bslmm_test(test_data,
                     phenotype,
                     train_prefix: str = "bslmm_tr_output",
                     train_output_path: str = "./output",
                     test_prefix: str = "bslmm_te_output",
                     related_matrix=None):

    save_data = type(test_data) == pd.DataFrame and type(
        phenotype) == np.ndarray
    if save_data:
        rand_number = str(np.random.randint(100000))
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        test_file = os.path.join(temp_dir, 'geno_te' + rand_number)
        phenotype_file = os.path.join(temp_dir, 'pheno_tr' + rand_number)
        test_data.to_csv(test_file, index=False, header=False)
        np.savetxt(phenotype_file, phenotype, fmt="%s")

    # if we get the files, then train with the files.
    elif type(test_data) == str and type(phenotype) == str:
        if not os.path.exists(phenotype):
            raise FileNotFoundError(f'Phenotype file {phenotype} not found')
        if not os.path.exists(test_data):
            raise FileNotFoundError(f'Genotype file {test_data} not found')
        phenotype_file = phenotype
    else:
        raise Exception('Phenotype or genotype file format not supported')

    if not os.path.exists(train_output_path):
        raise FileNotFoundError(
            f"output file {train_output_path} does not exist")

    epm = os.path.join(train_output_path, train_prefix + ".param.txt")
    emu = os.path.join(train_output_path, train_prefix + ".log.txt")
    ebv = os.path.join(train_output_path, train_prefix + ".bv.txt")

    print('\n>>> GAMMA BSLMM Predict:')
    if related_matrix is None:
        print(
            command +
            f' -predict 1 -g {test_file} -epm {epm} -emu {emu} -p {phenotype_file} -o {test_prefix}'
        )
        os.system(
            command +
            f' -predict 1 -g {test_file} -epm {epm} -emu {emu} -p {phenotype_file} -o {test_prefix}'
        )
    else:
        if isinstance(related_matrix, np.ndarray):
            related_matrix_dir = os.path.join(temp_dir,
                                              'relatedness_dir' + rand_number)
            np.savetxt(related_matrix_dir, related_matrix)
        elif isinstance(related_matrix, str):
            related_matrix_dir = related_matrix
        else:
            raise Exception("Relateness format not supported.")

        print(
            command +
            f' -predict 1 -g {test_file} -epm {epm} -emu {emu} -p {phenotype_file} -o {test_prefix}\
             -k {related_matrix_dir} -ebv {ebv}')
        os.system(
            command +
            f' -predict 1 -g {test_file} -epm {epm} -emu {emu} -p {phenotype_file} -o {test_prefix}\
             -k {related_matrix_dir} -ebv {ebv}')
        os.remove(related_matrix_dir)

    if save_data:
        os.remove(test_file)
        os.remove(phenotype_file)


# using gemma to estimate the variance components for each relatedness matrix.
def gemma_multi_var_estimator(pheno, multi_relatedness, var_prefix):
    rand_number = str(np.random.randint(100000))
    if type(pheno) == np.ndarray:
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        pheno_file = os.path.join(temp_dir, 'temp_pheno' + rand_number)
        np.savetxt(pheno_file, pheno)
    elif type(pheno) == str:
        if not os.path.exists(pheno):
            raise FileNotFoundError(f"Phenotype file {pheno} does not exist")
        pheno_file = pheno
    else:
        raise Exception(f"Phenotype file format {type(pheno)} not supported")

    print('\n>>> GEMMA multi-var estimate:')
    # if multi_relatedness is a file that indicates the path of multiple relatedness files.
    if isinstance(multi_relatedness, str):
        if not os.path.exists(multi_relatedness):
            raise FileNotFoundError(multi_relatedness)
        print(
            command +
            f' -p {pheno_file} -mk {multi_relatedness} -o {var_prefix} -vc 1')
        os.system(
            command +
            f' -p {pheno_file} -mk {multi_relatedness} -o {var_prefix} -vc 1')

    # in case multi_relatedness is a list of relatedness files or nparray
    elif isinstance(multi_relatedness, list):
        temp_relatedness_files = [
        ]  # store the file names of relatedness files
        temp_relatedness_dir = os.path.join(
            temp_dir, 'temp_multi_relatness_path_' + rand_number)

        for i, rel in enumerate(multi_relatedness):
            # if rel is a relatedness file
            if isinstance(rel, str):
                assert os.path.exists(rel), f'File {rel} not found.'
                with open(temp_relatedness_dir, 'a') as f:
                    f.write(rel + '\n')

            # if rel is a numpy array then save it as a file
            elif isinstance(rel, np.ndarray):
                assert len(rel.shape) == 2 and rel.shape[0] == rel.shape[1],\
                          f'Unrecognized shape of array {rel.shape} used.'
                temp_relatness = os.path.join(
                    temp_dir, 'temp_relatedness_' + rand_number + '_' + str(i))
                np.savetxt(temp_relatness, rel)
                temp_relatedness_files.append(temp_relatness)
                with open(temp_relatedness_dir, 'a') as f:
                    f.write(temp_relatness + '\n')

        print(
            command +
            f' -p {pheno_file} -mk {temp_relatedness_dir} -o {var_prefix} -vc 1'
        )
        os.system(
            command +
            f' -p {pheno_file} -mk {temp_relatedness_dir} -o {var_prefix} -vc 1'
        )
        for file in temp_relatedness_files:
            os.remove(file)
        os.remove(temp_relatedness_dir)
    else:
        raise Exception(
            f"Relatness Matrix file format {type(multi_relatedness)} not supported"
        )

    if type(pheno) == np.ndarray:
        os.remove(pheno_file)

    return gemma_var_reader(var_prefix)


def gemma_var_estimator(pheno, relatedness, var_prefix):

    rand_number = str(np.random.randint(100000))
    if type(pheno) == np.ndarray:
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        pheno_file = os.path.join(temp_dir, 'temp_pheno' + rand_number)
        np.savetxt(pheno_file, pheno)
    elif type(pheno) == str:
        if not os.path.exists(pheno):
            raise FileNotFoundError(f"Phenotype file {pheno} does not exist")
        pheno_file = pheno
    else:
        raise Exception(f"Phenotype file format {type(pheno)} not supported")

    print('\n>>> GEMMA var estimate:')
    if type(relatedness) == 'string':
        if not os.path.exists(relatedness):
            raise FileNotFoundError(relatedness)
        print(command +
              f' -p {pheno_file} -k {relatedness} -o {var_prefix} -vc 1')
        os.system(command +
                  f' -p {pheno_file} -k {relatedness} -o {var_prefix} -vc 1')
    elif type(relatedness) == np.ndarray:
        temp_relatness_dir = os.path.join(temp_dir,
                                          'temp_relatness_' + rand_number)
        np.savetxt(temp_relatness_dir, relatedness)
        print(
            command +
            f' -p {pheno_file} -k {temp_relatness_dir} -o {var_prefix} -vc 1')
        os.system(
            command +
            f' -p {pheno_file} -k {temp_relatness_dir} -o {var_prefix} -vc 1')
        os.remove(temp_relatness_dir)

    elif isinstance(relatedness, list):
        gemma_multi_var_estimator(pheno, relatedness, var_prefix)
    else:
        raise Exception(
            f"Relatness Matrix file format {type(relatedness)} not supported")

        ## Read the variance components from the input data

    if type(pheno) == np.ndarray:
        os.remove(pheno_file)

    return gemma_var_reader(var_prefix)


def gemma_lmm_train(train_data,
                    phenotype,
                    prefix: str = "lmm_tr_output",
                    related_matrix=None):

    # if we get the data, we store it on temporary files.
    save_data = type(train_data) == pd.DataFrame and type(
        phenotype) == np.ndarray
    if save_data:
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        rand_number = str(np.random.randint(100000))
        train_file = os.path.join(temp_dir, 'geno_tr' + rand_number)
        phenotype_file = os.path.join(temp_dir, 'pheno_tr' + rand_number)
        train_data.to_csv(train_file, index=False, header=False)
        np.savetxt(phenotype_file, phenotype)

    # if we get the files, then train with the files.
    elif type(train_data) == str and type(phenotype) == str:
        if not os.path.exists(phenotype):
            raise FileNotFoundError(f'Phenotype file {phenotype} not found')
        if not os.path.exists(train_data):
            raise FileNotFoundError(f'Genotype file {train_data} not found')
        train_file = train_data
        phenotype_file = phenotype

    else:
        raise Exception('Phenotype or genotype file format not supported')

    if not os.path.exists(phenotype_file):
        raise FileNotFoundError(f"Could not find {phenotype_file}")

    print('\n>>> GEMMA LMM:')
    if related_matrix is None:
        excute_command = command +\
              f" -g {train_file} -p {phenotype_file} -o {prefix} -lmm 1"
        print(excute_command)
        os.system(excute_command)
    else:
        if isinstance(related_matrix, np.ndarray):
            related_matrix_dir = os.path.join(temp_dir,
                                              'relatedness_dir' + rand_number)
            np.savetxt(related_matrix_dir, related_matrix)
        elif isinstance(related_matrix, str):
            related_matrix_dir = related_matrix
        else:
            raise Exception("Relateness format not supported.")

        excute_command =  command + \
            f" -g {train_file} -p {phenotype_file} -o {prefix} -lmm 1 -k {related_matrix_dir}"
        print(excute_command)
        os.system(excute_command)

        os.remove(related_matrix_dir)

    if save_data:
        os.remove(train_file)
        os.remove(phenotype_file)

    def lmm_var_reader():
        file = os.path.join('./output', prefix + '.log.txt')
        with open(file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith('## vg'):
                vg = line.strip('\n').split('=')[-1].strip()
                vg = float(vg)
            if line.startswith('## ve'):
                ve = line.strip('\n').split('=')[-1].strip()
                ve = float(ve)

        return vg, ve

    return lmm_var_reader()


def gemma_var_reader(var_prefix, dir='./output'):
    file = os.path.join(dir, var_prefix + '.log.txt')
    with open(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('## sigma2 estimates'):
            data = line.strip('\n').split('=')[-1].strip()
            vars = [float(var) for var in data.split('  ')]

    return vars


class GemmaOutputReader:
    def __init__(self,
                 bimbam_tr_dir: Union[str, pd.DataFrame],
                 prefix: str,
                 relatedness_dir: Union[str, pd.DataFrame, None] = None,
                 output_path: str = './output',
                 y=None):

        # get information from the training file
        if isinstance(bimbam_tr_dir, str):
            bimbam_tr = pd.read_csv(bimbam_tr_dir, header=None, index_col=None)
        else:
            bimbam_tr = bimbam_tr_dir
        self.bimbam_tr = bimbam_tr
        X = bimbam_tr.iloc[:, 3:].to_numpy().T
        self.n = X.shape[0]
        self.p = X.shape[1]

        if isinstance(relatedness_dir, str):
            K = np.loadtxt(relatedness_dir)
        elif isinstance(relatedness_dir, np.ndarray):
            K = relatedness_dir
        else:
            K = 1 / self.p * X @ X.T

        self.s_a = 1 / (self.n * self.p) * np.sum(X)
        self.s_b = 1 / self.n * np.sum(np.diag(K))
        self.breeding_value, self.gamma, self.hyper, self.para =\
                    GemmaOutputReader.read_output(prefix, output_path)

        self.rho = self.hyper.rho.mean()
        self.h = self.hyper.h.mean()
        self.pi = self.hyper.pi.mean()
        self.pve = self.hyper.pve.mean()
        self.pge = self.hyper.pge.mean()

    def get_hyper(self):
        hyper = {
            'rho': self.rho,
            'h': self.h,
            'pve': self.pve,
            'pge': self.pge,
            'pi': self.pi,
        }
        return hyper

    def get_para(self):
        return self.para

    def get_var_component(self):
        rho = self.rho
        h = self.h
        # rho = self.hyper.pge.mean()
        # h = self.hyper.pve.mean()
        pi = self.pi
        p = self.p
        sigma_a2 = rho * h / ((1 - h) * p * pi * self.s_a)
        sigma_b2 = h * (1 - rho) / ((1 - h) * self.s_b)
        return sigma_a2, sigma_b2

    def pred_train(self):
        self.bimbam_tr.rename(columns={0: 'rs'}, inplace=True)
        analyzed_SNPs = pd.merge(self.bimbam_tr,
                                 self.para[['rs', 'beta', 'alpha']],
                                 how='inner',
                                 on="rs")
        beta = analyzed_SNPs['beta'].to_numpy()
        alpha = analyzed_SNPs['alpha'].to_numpy()

        analyzed_SNPs.drop(['beta', 'alpha'], axis=1, inplace=True)
        analyzed_X = analyzed_SNPs.iloc[:, 3:].to_numpy().T
        # pred = analyzed_X @ beta + self.breeding_value
        # pred = analyzed_X @ beta +  analyzed_X @ alpha
        pred = analyzed_X @ beta
        return pred

    @staticmethod
    def gemma_pred_reader(test_prefix, test_dir='./output'):
        pred = np.loadtxt(os.path.join(test_dir, test_prefix + '.prdt.txt'),
                          dtype='str')
        pred = pred[pred != 'NA']
        # print(sum(pred=='NA'))
        return pred.astype('float')

    @staticmethod
    def read_output(prefix, dir='./output'):
        suffix_list = [
            'bv.txt', 'gamma.txt', 'hyp.txt', 'log.txt', 'param.txt'
        ]
        file_list = [os.path.join(dir, prefix + '.' + s) for s in suffix_list]
        bv = np.loadtxt(file_list[0], dtype=str)

        # filtered the NA
        mask = bv != 'NA'
        bv = bv[mask].astype('float')

        gamma = None
        # try:
        #     gamma = pd.read_csv(file_list[1], sep='\t', index_col=False)
        # except Exception as e:
        #     print(e)
        #     gamma = None
        # else:
        #     try:
        #         gamma = gamma.drop('Unnamed: 300', axis=1)
        #     except KeyError as e:
        #         print(e)
        #     except Exception as e:
        #         raise e

        hyperparameter = pd.read_csv(file_list[2], sep='\t', index_col=False)

        hyperparameter.rename(columns=lambda s: s.strip(), inplace=True)
        parameter = pd.read_csv(file_list[4], sep='\t', index_col=False)

        return bv, gamma, hyperparameter, parameter
