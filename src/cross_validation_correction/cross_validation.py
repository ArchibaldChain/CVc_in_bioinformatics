import copy
import time
import numpy as np
import sys
from typing import List, Union
import warnings

sys.path.append('./src')
from bimbam import Bimbam
from cross_validation_correction.methods import *
from utils import timing


class FoldsMoreThanSampleError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CrossValidation:
    def __init__(self,
                 model: 'BaseRegressor',
                 n_folds=10,
                 is_correcting: bool = True,
                 **kwargs):
        # Extract parameters
        self.model = model
        self.n_folds = n_folds
        self.is_correcting = is_correcting
        self.is_fitted = False
        if 'var_method' in kwargs:
            self.var_method = kwargs['var_method']
            method_l = ['gemma_var', 'fast_lmm', 'gemma_lmm']
            assert self.var_method in method_l,\
                f'The Variance components methods are {method_l}.'+\
                f'Method {self.var_method} is not supported.'
        else:
            self.var_method = None

    @timing
    def __call__(self,
                 training_data_bimbam: 'Bimbam',
                 indices_fixed_effects: Union[slice, List[int], int,
                                              None] = None,
                 **kwargs):

        self.data_bimbam = copy.deepcopy(training_data_bimbam)
        self.indices_fixed_effects = indices_fixed_effects
        start_time = time.time()

        # 0. Preprocessing the training data
        if training_data_bimbam.pheno is None:
            raise ValueError('Phenotype data not found')

        # 1. Shuffle the data
        shuffle_indices = np.arange(0, training_data_bimbam.n)
        np.random.shuffle(shuffle_indices)
        bimbam_shuffled = training_data_bimbam.iloc_Samples[shuffle_indices]
        self.shuffle_indices = shuffle_indices

        # 2. Estimate the variance of components if using LMM or if correcting

        #checking if using the indices to get the Fixed Effects
        if 'sigmas' in kwargs:
            sigmas = kwargs['sigmas']

        # calculating the variance when correcting or using GLS BLUP
        elif (self.is_correcting) or \
            (self.model.if_needs_sigmas):
            if indices_fixed_effects is not None:
                bimbam_shuffled_fixed = bimbam_shuffled[indices_fixed_effects]
                K_all = bimbam_shuffled.Relatedness
            else:
                bimbam_shuffled_fixed = bimbam_shuffled
                K_all = None

            if self.var_method == 'gemma_lmm':
                from GEMMA import gemma_operator as gemma
                sigmas = gemma.gemma_lmm_train(bimbam_shuffled.to_dataframe(),
                                               bimbam_shuffled.pheno,
                                               related_matrix=K_all)
            else:
                sigmas = BaseRegressor.var_components_estimate(
                    bimbam_shuffled_fixed.SNPs, bimbam_shuffled_fixed.pheno,
                    K_all, self.var_method)

        else:
            sigmas = None
        print('sigmas: ', sigmas)
        self.sigmas = sigmas

        # 3. Create a H_cv matrix if correcting
        if self.is_correcting:
            n = bimbam_shuffled.n
            H_cv = np.zeros([n, n])

        # 4. Use for loop over k folds calculating the y_pred_cv
        mses = []

        for k, (k_fold, k_minus) in \
            enumerate(KFolds(self.n_folds, bimbam_shuffled.n)):

            bimbam_shuffled_k = bimbam_shuffled.iloc_Samples[k_fold]
            bimbam_shuffled_minus_k = bimbam_shuffled.iloc_Samples[k_minus]

            if indices_fixed_effects is not None:
                snp_k_fixed = bimbam_shuffled_k[indices_fixed_effects]
                snp_k_minus_fixed = bimbam_shuffled_minus_k[
                    indices_fixed_effects]
                # creating relatedness matrix for minus k flods
                K_minus_k = bimbam_shuffled_minus_k.Relatedness
                K_k_minusk = bimbam_shuffled_k.create_relatedness_with(
                    bimbam_shuffled_minus_k)
            else:
                snp_k_fixed = bimbam_shuffled_k
                snp_k_minus_fixed = bimbam_shuffled_minus_k
                K_minus_k = None
                K_k_minusk = None

            self.model.fit(snp_k_minus_fixed.SNPs,
                           snp_k_minus_fixed.pheno,
                           K_minus_k,
                           sigmas=self.sigmas)
            pheno_pred = self.model.predict(snp_k_fixed.SNPs, K_k_minusk)
            mse = self.mean_square_error(pheno_pred, snp_k_fixed.pheno)

            if self.is_correcting:
                H_cv[k_fold, k_minus] = self.model.generate_h_te(
                    snp_k_fixed.SNPs, K_k_minusk)

            if mse > 100:
                warnings.warn(
                    f"Large MSE Detected in {self.model} {k}-th fold cv")
            mses.append(mse)

        # 5. Calculate the prediction using y_pred_cv
        if self.is_correcting:
            self.H_cv = H_cv
            y_cv = H_cv @ bimbam_shuffled.pheno
        mse = np.mean(mses)
        mse_h_cv = self.mean_square_error(y_cv, bimbam_shuffled.pheno)
        using_time = time.time() - start_time

        return {'cv': mse, 'cv_hcv': mse_h_cv, 'using_time': using_time}

    # fitting a model with current training data or new data
    @timing
    def fit(self):
        self.is_fitted = True

        bimbam_fixed, K = self._fixed_snp_extraction(self.data_bimbam, 'train')

        self.model.fit(bimbam_fixed.SNPs,
                       bimbam_fixed.pheno,
                       K,
                       sigmas=self.sigmas)

    # training with current model and giving a prediction
    @timing
    def predict(self, bimbam_te: 'Bimbam') -> np.ndarray:
        # using model training on the whole training set
        if not self.is_fitted:
            self.fit()

        bimbam_fixed, K_te_tr = self._fixed_snp_extraction(bimbam_te, 'test')

        return self.model.predict(bimbam_fixed.SNPs, K_te_tr)

    @timing
    def correct(self, bimbam_correct: Union['Bimbam', None] = None) -> int:
        if not self.is_correcting:
            raise NotImplementedError('During cv, is_correcting is set False')

        if not self.is_fitted:
            self.fit()
        _, K = self._fixed_snp_extraction(
            self.data_bimbam.iloc_Samples[self.shuffle_indices], 'train')
        V = self.sigmas[0] * K + self.sigmas[1] * np.identity(K.shape[0])

        # calculate the first part
        w = 2 / self.H_cv.shape[0] * np.trace(self.H_cv @ V)
        if bimbam_correct is None:
            return w

        bimbam_fixed, K_te_tr = self._fixed_snp_extraction(
            bimbam_correct, 'test')
        h_te = self.model.generate_h_te(bimbam_fixed.SNPs, K_te_tr)
        v_te_tr = self.sigmas[0] * K_te_tr

        # fist part minus second part
        w = w - 2 * 1 / (bimbam_correct.n) * np.trace(h_te @ v_te_tr.T)

        return w

    def _fixed_snp_extraction(self, bimbam: 'Bimbam', K_mode='train'):
        assert K_mode in ['train', 'test'], f'Invalid K mode {K_mode}.'+\
            f' It can only be train or test'

        if self.indices_fixed_effects is not None:
            bimbam_fixed = bimbam[self.indices_fixed_effects]
            if K_mode == 'train':
                K = bimbam.Relatedness
            else:
                K = bimbam.create_relatedness_with(self.data_bimbam)
        else:
            bimbam_fixed = bimbam
            K = None

        return bimbam_fixed, K

    @staticmethod
    def mean_square_error(y1, y2):
        return np.mean(np.square(y1 - y2))


class FoldsMoreThanSampleError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# K folds indices helper
# >>> kfolds = _K_folds(n_folds=5, n=100)
# >>> for k_fold_indices, k_minus_indices in kfolds:
# >>>   # Perform operations on the k-th fold and the remaining samples
class KFolds:
    def __init__(self, n_folds, n):

        if n_folds > n:
            raise FoldsMoreThanSampleError(
                f"Number of folds {n_folds} is larger than numer of samples {n}"
            )

        fold_size = int(np.floor(n / n_folds))
        resi = n % n_folds
        fold_sizes = [0] + \
            resi * [fold_size + 1] + (n_folds - resi) * [fold_size]
        indices = np.cumsum(fold_sizes)
        self.folds_indices = [(indices[i], indices[i + 1])
                              for i in range(n_folds)]
        self.fold = 0
        self.n_folds = n_folds
        self.n = n

    def __iter__(self):
        # Returns the iterator object itself
        return self

    def __next__(self):
        if self.fold >= self.n_folds:
            # This stops the iteration
            raise StopIteration
        a, b = self.folds_indices[self.fold]
        k_fold_indices = slice(a, b)
        k_minus_indices = list(range(0, a)) + list(range(b, self.n))
        self.fold += 1
        return k_fold_indices, k_minus_indices


def _test_correcting_gls(bimbam: Bimbam):
    bimbam_tr, bimbam_te = bimbam.train_test_split()
    gls = GeneralizedLeastRegressor()
    cv_gls = CrossValidation(gls, 10, True)
    re = cv_gls(bimbam_tr,
                slice(0, 100),
                var_methdod='fast_lmm',
                sigmas=[0.5, 0.5])
    re['w_te'] = cv_gls.correct(bimbam_te)
    print(re)
    re['mse_te'] = CrossValidation.mean_square_error(cv_gls.predict(bimbam_te),
                                                     bimbam_te.pheno)
    bimbam_fake = bimbam_tr.fake_sample_generate(bimbam_te.n)
    print(bimbam_fake.SNPs.dtype)
    re['w_fake'] = cv_gls.correct(bimbam_fake)
    re['mse_fake'] = CrossValidation.mean_square_error(
        cv_gls.predict(bimbam_fake), bimbam_fake.pheno)
    re['w_resample'] = cv_gls.correct(bimbam_tr.resample(bimbam_te.n))
    print(re)


def _test_correcting_blup(bimbam: Bimbam):
    bimbam_tr, bimbam_te = bimbam.train_test_split()
    gls = BestLinearUnbiasedPredictor(alpha=10)
    cv_blup = CrossValidation(gls, 10, True)
    re = cv_blup(bimbam_tr,
                 slice(0, 100),
                 var_methdod='fast_lmm',
                 sigmas=[0.5, 0.5])
    re['w_te'] = cv_blup.correct(bimbam_te)
    print(re)
    re['mse_te'] = CrossValidation.mean_square_error(
        cv_blup.predict(bimbam_te), bimbam_te.pheno)
    bimbam_fake = bimbam_tr.fake_sample_generate(bimbam_te.n)
    re['w_fake'] = cv_blup.correct(bimbam_fake)
    re['mes_fake'] = CrossValidation.mean_square_error(
        cv_blup.predict(bimbam_fake), bimbam_fake.pheno)
    re['w_resample'] = cv_blup.correct(bimbam_tr.resample(bimbam_te.n))
    print(re)


if __name__ == '__main__':

    bimbam = Bimbam('./bimbam_data/bimbam_10000_full_false_major_minor.txt')
    bimbam.pheno_simulator(100)
    _test_correcting_blup(bimbam)
