import sklearn
import os
import numpy as np
from typing import Union
from abc import ABC, abstractmethod
import sys

sys.path.append('./src')
from utils import inv


class BaseRegressor(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0
        self.l1_ratio = kwargs['l1_ratio'] if 'l1_ratio' in kwargs else 0

        assert self.alpha >= 0, f'{self.alpha=} must be greater than 0'
        self.alpha = self.alpha
        assert 0 <= self.l1_ratio <= 1, f'{self.l1_ratio=} should betwen 0 and 1'
        self.using_sklearn = (self.l1_ratio != 0 and self.alpha != 0)

    @abstractmethod
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            K: Union[np.ndarray, None] = None,
            **kwargs):
        assert isinstance(X, np.ndarray), f'X must be an np.ndarray'
        assert isinstance(y, np.ndarray), f'y must be np.ndarray'
        assert len(X.shape) == 2, f'X must be a 2d array'
        assert X.shape[0] == y.shape[
            0], f'Inconsistent for the number of samples in X {X.shape} and y {y.shape}'
        if K is None:
            K = 1 / X.shape[1] * X @ X.T
            self.using_default_K = True
        else:
            self.using_default_K = False

        assert K.shape[0] == y.shape[
            0], f'K must has the same number of rows {K.shape[0]} as length of phenotype {y.shape}'

        self.X = X
        self.y = y
        self.K = K

    @abstractmethod
    def predict(self,
                X_te: np.ndarray,
                K_te_tr: Union[np.ndarray, None] = None):
        assert isinstance(X_te, np.ndarray), f'X must be an np.ndarray'
        self._create_k_te_tr(X_te, K_te_tr)

    @abstractmethod
    def generate_h_te(self,
                      X_te,
                      K_te_tr: Union[np.ndarray, None] = None) -> np.ndarray:
        assert isinstance(X_te, np.ndarray), f'X must be an np.ndarray'
        self._create_k_te_tr(X_te, K_te_tr)

    def _predicion_check(self, X_te, K_te_tr):
        assert len(X_te.shape) == 2, f'X must be a 2d array'
        assert X_te.shape[1] == self.X.shape[1],\
              f'Inconsistent for the number of features in X_te shape{X_te.shape}, X shape {self.X.shape}'
        if K_te_tr is not None:
            assert X_te.shape[0] == K_te_tr.shape[
                0], f'Relatedness Matrix should has the same number of column as testing samples'

            assert K_te_tr.shape[1] == self.X.shape[0], \
            f'The K_te_tr matrix should has the number of column as number of training samples.'

    def _create_k_te_tr(self, X_te, K_te_tr):
        if K_te_tr is None:
            if not self.using_default_K:
                raise ValueError(
                    'K_te_tr must be specified since K_tr is specified')
            K_te_tr = 1 / self.X.shape[1] * X_te @ self.X.T

        self.K_te_tr = K_te_tr
        self._predicion_check(X_te, K_te_tr)

    # calculating the components of the variance
    @staticmethod
    def var_components_estimate(X: np.ndarray, y: np.ndarray,
                                K: Union[np.ndarray, None], method):
        if K is None:
            print('Use default setting for K')
            K = 1 / (X.shape[1]) * X @ X.T
            W = 2 / np.sqrt(X.shape[1]) * X
        else:
            W = np.sqrt(K)

        if method == 'gemma_var':
            try:
                from GEMMA import gemma_operator as gemma
            except FileNotFoundError as f:
                method = 'fast_lmm'
            except Exception:
                print(Exception)
                raise Exception

            else:
                sigmas = gemma.gemma_var_estimator(y, K, 'var_components')
                return sigmas

        from FAST_LMM import FASTLMM
        fast = FASTLMM(lowRank=True, REML=False)
        fast.fit(X, y, W)
        sigmas = (fast.sigma_g2, fast.sigma_e2)
        return sigmas


# Ordinary linear regression (OLS)
class OrdinaryLeastRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, K=None, **kwargs):
        super().fit(X, y, K, **kwargs)
        if self.using_sklearn:
            from sklearn.linear_model import ElasticNet as ENET
            self.enet = ENET(alpha=self.alpha, l1_ratio=self.l1_ratio)
            self.enet.fit(X, y)

        else:
            self.mid_matrix = inv(X.T @ X + self.l1_ratio *
                                  np.identity(X.shape[1])) @ X.T
            self.beta = self.mid_matrix @ y

    def predict(self, X_te: np.ndarray, K_te_tr=None):
        if self.using_sklearn:
            return self.enet.predict(X_te)

        super().predict(X_te, K_te_tr)
        return X_te @ self.beta

    def generate_h_te(self, X_te, K_te_tr=None):
        if self.using_sklearn:
            raise NotImplementedError(
                "Hat vector generation is not supported for LASSO and Elastic Net."
            )
        super().generate_h_te(X_te, K_te_tr)
        return X_te @ self.mid_matrix

    def __str__(self):
        return f'Using Ordinary Least Square (OLS) with alpha={alpha}, l1_ratio={self.l1_ratio}'


# generalized least squares (GLS)
class GeneralizedLeastRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.using_sklearn:
            raise NotImplementedError(
                'GLS is not supported for l1 regularization.')

    def fit(self, X: np.ndarray, y: np.ndarray, K: Union[np.ndarray, None],
            **kwargs):
        super().fit(X, y, K)
        if 'sigmas' in kwargs and kwargs['sigmas'] is not None:
            self.sigmas = kwargs['sigmas']
        else:
            self.sigmas = self.var_components_estimate(X,
                                                       y,
                                                       self.K,
                                                       method='gemma_var')

        self.V = self.sigmas[0] * self.K + self.sigmas[1] * np.identity(
            X.shape[0])
        self.V_inv = inv(self.V)
        self.mid_matrix = inv(X.T @ self.V_inv @ X + self.alpha *
                              np.identity(self.X.shape[1])) @ X.T @ self.V_inv
        self.beta = self.mid_matrix @ y

    def predict(self, X_te: np.ndarray, K_te_tr=None):
        super().predict(X_te, K_te_tr)
        return X_te @ self.beta

    def generate_h_te(self, X_te, K_te_tr=None):
        super().generate_h_te(X_te, K_te_tr)
        return X_te @ self.mid_matrix


# best linear unbiased prediction (BLUP)
class BestLinearUnbiasedPredictor(GeneralizedLeastRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            K: Union[np.ndarray, None] = None,
            **kwargs):
        super().fit(X, y, K, **kwargs)

    def predict(self,
                X_te: np.ndarray,
                K_te_tr: Union[np.ndarray, None] = None):
        self._create_k_te_tr(X_te, K_te_tr)

        u_hat = self.sigmas[0] * self.K_te_tr @ self.V_inv @ \
            (self.y - self.X @self.beta)
        return super().predict(X_te, K_te_tr) + u_hat

    def generate_h_te(self, X_te, K_te_tr=None):
        self._create_k_te_tr(X_te, K_te_tr)

        return (super().generate_h_te(X_te, K_te_tr) +
                self.sigmas[0] * self.K_te_tr @ self.V_inv
                @ (np.identity(self.X.shape[0]) - self.X @ self.mid_matrix))


if __name__ == '__main__':
    from bimbam import Bimbam
    bimbam = Bimbam('./bimbam_data/bimbam_10000_full_false_major_minor.txt')
    bimbam.pheno_simulator(100)
    bimbam_tr, bimbam_te = bimbam.train_test_split()
    bimbam_tr_x = bimbam_tr[:100]
    bimbam_te_x = bimbam_te[:100]
    mse = lambda y1, y2: np.sum((y1 - y2)**2) / y1.shape[0]

    alpha = 100
    # ols = OrdinaryLeastRegressor(alpha=alpha)
    # ols.fit(bimbam_tr_x.SNPs, bimbam_tr_x.pheno, sigmas=[0.5, 0.5])
    # y_ols_pred = ols.predict(bimbam_te_x.SNPs)
    # print(mse(y_ols_pred, bimbam_te_x.pheno))
    # print(ols.generate_h_te(bimbam_te_x.SNPs).shape)

    # gls = GeneralizedLeastRegressor(alpha=alpha)
    # gls.fit(bimbam_tr_x.SNPs,
    #         bimbam_tr_x.pheno,
    #         bimbam_tr.Relatedness,
    #         sigmas=[0.5, 0.5])
    # y_gls_pred = gls.predict(bimbam_te_x.SNPs,
    #                          bimbam_te.create_relatedness_with(bimbam_tr))
    # print(mse(y_gls_pred, bimbam_te_x.pheno))
    # print(
    #     gls.generate_h_te(bimbam_te_x.SNPs,
    #                       bimbam_te.create_relatedness_with(bimbam_tr)).shape)

    # ols = OrdinaryLeastRegressor(alpha=10, l1_ratio=0.5)
    # ols.fit(bimbam_tr_x.SNPs, bimbam_tr_x.pheno)
    # y_ols_pred = ols.predict(bimbam_te_x.SNPs)
    # print(mse(y_ols_pred, bimbam_te_x.pheno))

    blup = BestLinearUnbiasedPredictor(alpha=alpha)
    blup.fit(bimbam_tr_x.SNPs,
             bimbam_tr_x.pheno,
             bimbam_tr.Relatedness,
             sigmas=[0.5, 0.5])
    y_blup_pred = blup.predict(bimbam_te_x.SNPs,
                               bimbam_te.create_relatedness_with(bimbam_tr))
    print(mse(y_blup_pred, bimbam_te_x.pheno))
    print(
        blup.generate_h_te(bimbam_te_x.SNPs,
                           bimbam_te.create_relatedness_with(bimbam_tr)).shape)

    blup = BestLinearUnbiasedPredictor(alpha=alpha)
    blup.fit(
        bimbam_tr_x.SNPs,
        bimbam_tr_x.pheno,
        #  bimbam_tr.Relatedness,
        sigmas=[0.5, 0.5])
    y_blup_pred = blup.predict(bimbam_te_x.SNPs)
    print(mse(y_blup_pred, bimbam_te_x.pheno))
    print(blup.generate_h_te(bimbam_te_x.SNPs).shape)