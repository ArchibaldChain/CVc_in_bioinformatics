import time
from typing import Union, Tuple
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import warnings


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

    # simulat SNP effects to generate phenotype
    def pheno_simulator(self,
                        num_large_effect=10,
                        large_effect=400,
                        small_effect=1,
                        heritability=0.5):

        if self.pheno is not None:
            warnings.warn("You esixted Phenotypes will be removed.")

        # get beta
        if self._beta is not None:
            warnings.warn("Origin beta will be removed after new beta")

        d = num_large_effect
        beta_large = np.random.normal(0, np.sqrt(large_effect), d)
        n = len(self.bimbam_data)
        beta_small = np.random.normal(0, np.sqrt(small_effect), n - d)
        self._beta = np.concatenate([beta_large, beta_small])

        # Shufftle the data
        self.bimbam_data = self.bimbam_data.sample(len(self.bimbam_data),
                                                   replace=False)
        return self.pheno_generator(heritability)

    # using the current SNPs effects to regenerate the phenotype
    def pheno_generator(self, heritability=0.5):
        # calculate the sigma_e2
        SNP = self.bimbam_data.drop(['major', 'minor', 'POS'],
                                    axis=1,
                                    inplace=False).T
        SNP.head()
        temp = SNP.values @ self._beta
        sigma_g2 = np.var(temp, ddof=1)  # get the overall variance
        h2 = heritability  # heritability
        sigma_e2 = sigma_g2 * (1 - h2) / h2
        print("Phenotype generated with s_g2: {:.2}, and s_e2: {:.2}".format(
            sigma_g2, sigma_e2))

        # simulate beta_0 and calculate y
        residual = np.random.normal(0, np.sqrt(sigma_e2), len(SNP))
        Y = SNP.values @ self._beta + residual

        if self.mean_Y is None and self.sd_Y is None:
            self.mean_Y = np.mean(Y)
            self.sd_Y = np.std(Y)

        pheno = (Y - self.mean_Y) / self.sd_Y
        self.pheno = pheno
        return pheno

    @property
    def MAF(self) -> np.ndarray:
        return np.sum(self.SNPs, axis=0) / (2 * self.n)

    # generate random samples
    def fake_sample_generate(self,
                             n_samples: int,
                             heritability=0.5,
                             seed: Union[int, None] = None,
                             use_MAF=True):
        if seed is not None:
            np.random.seed(seed)
        if use_MAF:
            temp_p = np.tile(self.MAF, [n_samples, 1]).T
            random_SNP = np.random.binomial(2, temp_p)
        else:
            random_SNP = np.random.choice([0, 1, 2], [self._p, n_samples],
                                          [0.25, 0.5, 0.25])

        random_SNP_df = pd.DataFrame(random_SNP)

        temp_info = self.info.reset_index(inplace=False, drop=True)
        random_bimbam_df = pd.concat([temp_info, random_SNP_df], axis=1)

        new_bimbam = Bimbam(random_bimbam_df)
        new_bimbam.beta = self.beta
        new_bimbam.mean_Y = self.mean_Y
        new_bimbam.sd_Y = self.sd_Y
        new_bimbam.pheno_generator(heritability)
        return new_bimbam

    # resample from the samples
    def resample(self, n_samples, replace=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(np.arange(0, self.n),
                                   n_samples,
                                   replace=replace)

        return self.iloc_Samples[indices]

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
            new_bimbam = Bimbam(bimbam_data, self.bimbam.pheno[indices])
            new_bimbam.beta = self.bimbam.beta
            new_bimbam.mean_Y = self.bimbam.mean_Y
            new_bimbam.sd_Y = self.bimbam.sd_Y
            return new_bimbam

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
        assert (beta is None) or (len(beta) == self.p), f'Incorrect beta'
        if isinstance(beta, np.ndarray):
            self._beta = beta.copy()
        else:
            self._beta = beta

    @property
    def SNPs(self):
        _SNPs = self.bimbam_data.iloc[:, 3:].to_numpy().T
        if not (str(_SNPs.dtype).startswith('int')
                or str(_SNPs.dtype).startswith('float')):
            raise AttributeError(
                f'The attribute type is not supported {_SNPs.dtype}')
        return _SNPs

    @property
    def Relatedness(self, scale_type='centered'):
        types = ('centered', 'standardized')
        assert scale_type in types, f'scale_typle is given as{types} and it has to be in {scale_type}.'
        if scale_type == 'centered':
            mean = self.SNPs.mean(axis=0)
            temp = (self.SNPs - mean) @ (self.SNPs - mean).T
        elif scale_type == 'standardized':
            mean = self.SNPs.mean(axis=0)
            sd = self.SNPs.std(axis=0, ddof=1)
            temp0 = ((self.SNPs - mean) / sd)
            temp = temp0 @ temp0.T
        return 1 / self.p * temp

    def create_relatedness_with(self,
                                other_bimbam: 'Bimbam',
                                scale_type='centered'):
        types = ('centered', 'standardized')
        assert scale_type in types, f'scale_typle is given as{types} and it has to be in {scale_type}.'

        if not isinstance(other_bimbam, Bimbam):
            raise TypeError('other_bimbam must be a Bimbam instance')
        if self.p != other_bimbam.p:
            raise ValueError(
                f'The bimbam matrix must have the same number of SNPs, i.e. self.p.shape: {self.shape}, other_bimbam.shape: {other_bimbam.shape}')
        start_time = time.time()
        if scale_type == 'centered':
            mean1 = self.SNPs.mean(axis=0)
            mean2 = other_bimbam.SNPs.mean(axis=0)
            temp = (self.SNPs - mean1) @ (other_bimbam.SNPs - mean2).T
        elif scale_type == 'standardized':
            mean1 = self.SNPs.mean(axis=0)
            sd1 = self.SNPs.std(axis=0, ddof=1)
            mean2 = other_bimbam.SNPs.mean(axis=0)
            sd2 = other_bimbam.SNPs.std(axis=0, ddof=1)
            temp = ((self.SNPs - mean1) / sd1) @ (
                (other_bimbam.SNPs - mean2) / sd2).T
        print(f'Calulation for K_te_tr using time: {time.time() - start_time}')
        return (1 / self.p) * temp

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
    def test_data_preparation(geno_tr: pd.DataFrame, geno_te: pd.DataFrame,
                              pheno_tr: np.ndarray, len_pheno_te: int):
        info = geno_tr.iloc[:, :3]
        full_bim = pd.concat([info, geno_tr.iloc[:, 3:], geno_te.iloc[:, 3:]],
                             axis=1)
        pheno_full = np.concatenate(
            [pheno_tr, np.array(['NA'] * len_pheno_te)])
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
    s = slice(0,0)
    new_bimbam = bimbam[s]
    print(type(new_bimbam.SNPs))
    print(new_bimbam.SNPs.shape)

    print(new_bimbam.SNPs.shape == (0,))




    exit(0)
    print(new_bimbam.shape)
    print(new_bimbam.to_dataframe().index)
    print(type(new_bimbam))
    tr, te = bimbam.train_test_split(0.2)
    print(tr.to_dataframe().index)
    print(type(tr), type(te))
    print(tr.shape, te.shape)
    print(tr.p, tr.n, te.p, te.n, bimbam.p, bimbam.n)
    print(bimbam.iloc_Samples[3:20].shape)
    print(bimbam.fake_sample_generate(20).to_dataframe)
