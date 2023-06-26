from typing import Union, Tuple
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# from bslmm_simulation.bimbam import data_spliter


class Bimbam(object):
    def __init__(self,
                 bimbam: Union[str, pd.DataFrame, None] = None,
                 pheno: Union[str, np.ndarray, None] = None):
        if isinstance(bimbam, str):
            self.bimbam_data = pd.read_csv(bimbam, sep='\t', index_col=None)
        elif isinstance(bimbam, pd.DataFrame):
            self.bimbam_data = bimbam.copy()
        else:
            self.bimbam_data = None

        self.info = self.bimbam_data[['POS', 'major', 'minor']]
        self._n = self.bimbam_data.shape[1] - 3
        self._p = self.bimbam_data.shape[0]
        self._shape = (self.p, self.n)

        if isinstance(pheno, str):
            self.pheno = np.loadtxt(pheno)
        elif isinstance(pheno, np.ndarray):
            self.pheno = pheno.copy()
        else:
            self.pheno = None

        self._beta = None
        self.mean_Y = None
        self.sd_Y = None

    def pheno_simulator(self,
                        num_large_effect=10,
                        large_effect=400,
                        small_effect=1):

        if self.pheno is not None:
            Warning.warn("You have esixted Phenotypes")

        if self._beta is None:
            # Get Beta
            d = num_large_effect
            beta_large = np.random.normal(0, np.sqrt(large_effect), d)
            n = len(self.bimbam_data)
            beta_small = np.random.normal(0, np.sqrt(small_effect), n - d)
            beta = np.concatenate([beta_large, beta_small])
        else:
            beta = self._beta

        # Shufftle the data
        self.bimbam_data = self.bimbam_data.sample(len(self.bimbam_data),
                                                   replace=False)

        # calculate the sigma_e2
        SNP = self.bimbam_data.drop(['major', 'minor', 'POS'],
                                    axis=1,
                                    inplace=False).T
        SNP.head()
        temp = SNP.values @ beta
        sigma_g2 = np.var(temp, ddof=1)  # get the overall variance
        h2 = 0.5  # heritability
        sigma_e2 = sigma_g2 * (1 - h2) / h2
        print("Phenotype generated with sigma_g2: {:.5}, and sigma_e2: {:.5}".
              format(sigma_g2, sigma_e2))

        # simulate beta_0 and calculate y
        residual = np.random.normal(0, np.sqrt(sigma_e2), len(SNP))
        Y = SNP.values @ beta + residual

        if self.mean_Y is None and self.sd_Y is None:
            self.mean_Y = np.mean(Y)
            self.sd_Y = np.std(Y)

        pheno = (Y - self.mean_Y) / self.sd_Y
        self._beta = beta
        self.pheno = pheno
        return pheno

    def to_dataframe(self) -> pd.DataFrame:
        return self.bimbam_data

    def train_test_split(self, test_size=0.2) -> Tuple['Bimbam', 'Bimbam']:
        # Train test split
        sample_size = self.n
        train_idx, test_idx = train_test_split(np.arange(sample_size),
                                               test_size=test_size)
        self.split_index = (train_idx, test_idx)
        return self.iloc_Samples[train_idx], self.iloc_Samples[test_idx]

    class _IlocSamples:
        def __init__(self, bimbam: 'Bimbam'):
            self.bimbam = bimbam

        def __getitem__(self, indices):
            data = self.bimbam.bimbam_data.iloc[:, 3:]
            bimbam_data = pd.concat([self.bimbam.info, data.iloc[:, indices]],
                                    axis=1)
            new_bimbam = Bimbam(bimbam_data, self.bimbam.pheno)
            new_bimbam.beta = self.bimbam.beta
            new_bimbam.mean_Y = self.bimbam.mean_Y
            new_bimbam.sd_Y = self.bimbam.sd_Y
            return new_bimbam

    @property
    def shape(self):
        return self._shape

    @property
    def n(self):
        # number of samples
        return self._n

    @property
    def p(self):
        # number of samples
        return self._p

    @property
    def beta_adjusted(self):
        return (self._beta - self.mean_Y) / self.sd_Y

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta: Union[None, np.ndarray]):
        assert (beta is None) or (len(beta) == self.p)
        print(type(beta))
        self._beta = beta

    @property
    def iloc_Samples(self):
        return self._IlocSamples(self)

    def __getitem__(self, SNP_index):
        new_bimbam = Bimbam(self.bimbam_data.iloc[SNP_index, :], self.pheno)
        new_bimbam.beta = self.beta[SNP_index]
        new_bimbam.sd_Y = self.sd_Y
        new_bimbam.sd_Y = self.sd_Y
        return new_bimbam

    @property
    def SNPs(self):
        return self.bimbam_data.iloc[:, 3:].to_numpy().T

    @property
    def Relatedness(self):
        return 1 / self.p * self.SNPs @ self.SNPs.T

    @staticmethod
    def data_spliter(data: pd.DataFrame, pheno: np.ndarray, train_idx,
                     test_idx):
        if isinstance(train_idx, np.ndarray):
            train_idx = train_idx.tolist()
        if isinstance(test_idx, np.ndarray):
            test_idx = test_idx.tolist()

        if len(train_idx) < 0 or len(test_idx) < 0:
            raise IndexError('Specify index for training and testing set.')

        if not isinstance(test_idx[0], int):
            raise IndexError('Index has to be integer.')

        temp = data.iloc[:, 3:]
        train_data = temp.iloc[:, train_idx]
        test_data = temp.iloc[:, test_idx]

        info = data.iloc[:, :3]
        train_bim = pd.concat([info, train_data], axis=1)
        test_bim = pd.concat([info, test_data], axis=1)

        pheno_tr = pheno[train_idx]
        pheno_te = pheno[test_idx]

        print(f"Training set has {train_data.shape[1] - 3} samples")
        print(f"Test set has {test_data.shape[1] - 3} samples")

        return (train_bim, pheno_tr), (test_bim, pheno_te)

    @staticmethod
    def test_data_parperation(geno_tr: pd.DataFrame, geno_te: pd.DataFrame,
                              pheno_tr: np.ndarray, pheno_te: np.ndarray):
        info = geno_tr.iloc[:, :3]
        full_bim = pd.concat([info, geno_tr.iloc[:, 3:], geno_te.iloc[:, 3:]],
                             axis=1)
        pheno_full = np.concatenate(
            [pheno_tr, np.array(['NA'] * len(pheno_te))])
        return full_bim, pheno_full

    @staticmethod
    def data_concatenate(bim1, bim2, phyno1, phyno2):
        temp = bim2.loc[:, 3:]
        bim_full = pd.concat([bim1, temp], axis=1)
        phyno = np.concatenate(phyno1, phyno2)
        return bim_full, phyno


if __name__ == "__main__":
    bimbam_path = './bimbam_data/'
    bimbam_dir = os.path.join(bimbam_path,
                              'bimbam_10000_full_false_major_minor.txt')
    bimbam = Bimbam(bimbam_dir)
    bimbam.pheno_simulator()
    print(bimbam.n, bimbam.p, bimbam.shape)
    W = bimbam.SNPs
    print(W.shape)
    K = bimbam.Relatedness
    print(type(K), K.shape)
    new_bimbam = bimbam[:10]
    print(new_bimbam.shape)
    print(new_bimbam.to_dataframe().index)
    print(type(new_bimbam))
    tr, te = bimbam.train_test_split(0.2)
    print(tr.to_dataframe().index)
    print(type(tr), type(te))
    print(tr.shape, te.shape)
    print(tr.p, tr.n, te.p, te.n, bimbam.p, bimbam.n)