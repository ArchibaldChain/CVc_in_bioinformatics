import numpy as np
import pandas as pd
import sys
import os

command = "./gemma-0.98.5"
print(os.getcwd())


def gamma_train(train_file: str,
                phenotype_file: str,
                related_matrix=None,
                output: str = None):
    if output is None:
        output = "./output"
    if not os.path.exists(output):
        os.mkdir(output)

    if not os.path.exists(phenotype_file):
        raise FileNotFoundError(f"Could not find {phenotype_file}")

    os.system("chmod +x " + command)
    if related_matrix is None:
        print(command +
              f" -g {train_file} -p {phenotype_file} -o {output} -bslmm 1")
        os.system(command +
                  f" -g {train_file} -p {phenotype_file} -o {output} -bslmm 1")
    else:
        print(command +
              f" -g {train_file} -p {phenotype_file} -o {output} -bslmm 1")
        os.system(
            command +
            f" -g {train_file} -p {phenotype_file} -o {output} -bslmm 1 -k {related_matrix}"
        )


def gamma_test(test_file: str,
               phenotype_file: str,
               train_output_path: str,
               test_output=None,
               related_matrix=None):

    if test_output is None:
        test_output = "./test_output"
    if not os.path.exists(test_output):
        os.mkdir(test_output)
    if not os.path.exists(train_output_path):
        raise FileNotFoundError(
            f"output file {train_output_path} does not exist")

    epm = os.path.join(train_output_path, "output.param.txt")
    emu = os.path.join(train_output_path, "output.log.txt")
    ebv = os.path.join(train_output_path, "output.bv.txt")

    if related_matrix is None:
        os.system(
            command +
            f' -predict 1 -g {test_file} -epm {epm} -emu {emu} -p {phenotype_file} -o {test_output}'
        )
    else:
        os.system(
            command +
            f' -predict 1 -g {test_file} -epm {epm} -emu {emu} -p {phenotype_file} -o {test_output}\
             -k {related_matrix} -ebv {ebv}')


class GammaOutputReader:
    def __init__(self, output_path, X, K, y=None):
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.s_a = 1 / (self.n * self.p) * np.sum(X)
        self.s_b = 1 / self.n * np.sum(np.diag(K))
        self.breeding_value, self.gamma, self.hyper, self.para =\
                    GammaOutputReader.read_output(output_path)

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

    @staticmethod
    def read_output(path):
        file_list = [
            'output.bv.txt', 'output.gamma.txt', 'output.hyp.txt',
            'output.log.txt', 'output.param.txt'
        ]
        bv = np.loadtxt(path + file_list[0])
        gamma = pd.read_csv(path + file_list[1], sep='\t', index_col=False)
        try:
            gamma = gamma.drop('Unnamed: 300', axis=1)
        except KeyError as e:
            print(e)
        except Exception as e:
            raise e

        hyperparameter = pd.read_csv(path + file_list[2],
                                     sep='\t',
                                     index_col=False)

        hyperparameter.rename(columns=lambda s: s.strip(), inplace=True)
        parameter = pd.read_csv(path + file_list[4], sep='\t', index_col=False)

        return bv, gamma, hyperparameter, parameter


if __name__ == '__main__':
    # bv, gamma, hyper, para = GammaOutputReader.read_output('./gamma_output/')
    train_file = "./bimbam_data/bimbam_train"
    phenotype_file = "./bimbam_data/phenotype_train"
    output = "./gamma_output/training_output"
    gamma_train(train_file, phenotype_file)
