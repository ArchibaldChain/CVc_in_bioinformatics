import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from numpy.linalg import det
from numpy.linalg import LinAlgError
import scipy.optimize as opt
import warnings
import utils as u


class FASTLMM:
    beta = None  # coefficients

    sigma_g2 = None  # gene variance
    sigma_e2 = None  # phenotype variance
    delta = None  # ratio of sigma_e2 to sigma_g2
    U = None  # eigenvector matrix of K
    S = None  # eigenvalues array of K
    delta = None  # temporary delta for efficiency

    def __init__(self, lowRank=False, REML=False):
        print('FAST-LMM')
        print('------------------------------------')
        if REML:
            print('LowRank is set as {}, using REML'.format(lowRank))
        else:
            print('LowRank is set as {}, not using REML'.format(lowRank))

        self.lowRank = lowRank
        self.REML = REML

    def fit(self, X, y, W=None):
        self.X = np.array(X).astype('float64')
        self.y = np.array(y).reshape([-1, 1])

        d = X.shape[1]
        # K = W.T @ W
        # or K = 1/d * X.T @ X
        # so we can set W = 1/ sqrt(d) * X
        if W is None:
            W = 1 / np.sqrt(d) * X.copy()

        n, sc = W.shape
        print(y.shape)
        if (n != X.shape[0]) or (n != self.y.shape[0]) or (self.y.shape[1]) > 1:
            raise ValueError(
                'Incompatible shape of X(shape of {}), y(shape of {}) and w(shape of {}).'
                .format(X.shape, self.y.shape, W.shape)
            )

        K = W @ W.T
        self.K = K.copy()
        self.rank = matrix_rank(K)
        print('Rank of W is {}, shape of W is {}.'.format(self.rank, W.shape))

        if self.lowRank:
            if self.rank == max(n, sc):
                warnings.warn("W is set as lowRank, but actually not lowRank.")
                self.lowRank = False

            if self.rank < min(n, sc):
                # in case the rank smaller than min(n, sc)
                U, S, _ = svds(K, self.rank)
            else:
                # in case the rank larger than n or sc
                U, S, _ = svd(K, overwrite_a=True)
                U = U[:, :self.rank]
                S = S[: self.rank]
        else:
            if self.rank < max(n, sc):
                warnings.warn('W is set lowRank False, but actually lowRank.')

            # incase that set lowRank is False
            U, S, _ = svd(K, overwrite_a=True)
            # setting the last n - sc eigenvalues as 0
            S[self.rank:] = 0

        # check if S is a matrix
        if S.ndim > 1:
            print('Get 2d S')
            print(S)
            S = np.diag(S)

        # in case that length of S is smaller than the columns of U
        # This only will happen when W is lowRank but set as not LowRank
        if len(S) < U.shape[1]:
            S = np.concatenate([S, np.zeros(U.shape[1] - len(S))])

        self.U = U
        self.S = S ** 2
        self._buffer_preCalculation()

    def _buffer_preCalculation(self):
        n, _ = self.X.shape
        self.UTX = self.U.T @ self.X
        self.UTy = self.U.T @ self.y

        if self.lowRank:
            self.I_minus_UUT = np.identity(n) - self.U @ self.U.T
            self.I_minus_UUT_X = self.I_minus_UUT @ self.X
            self.I_minus_UUT_y = self.I_minus_UUT @ self.y
            self.I_UUTX_sq = self.I_minus_UUT_X.T @ self.I_minus_UUT_X
            self.I_UUTX_I_UUTy = self.I_minus_UUT_X.T @ self.I_minus_UUT_y

        if self.REML:
            self.log_XTX = np.log(det(self.X.T @ self.X))

    def testing_sigmag2(self, delta):
        U = self.U
        U1 = self.U[:, :self.rank]
        U2 = self.U[:, :self.rank]
        S1 = self.S[:self.rank]
        # Update buffer
        if delta != self.delta:
            self._buffer_preCalculation_with_delta(delta)

        a = self.y - self.X @ self.beta_delta
        test1 = (U.T @ a).T @ u.inv(np.diag(self.S + delta)) @  (U.T @ a)
        test2 = (U1.T @ a).T @ u.inv(np.diag(S1 + delta)) @ (U1.T @ a) +\
            (U2.T @ a) ** 2 / delta
        print('if difference???')
        print(np.sum((test1 - test2)**2))

        n, d = self.X.shape

        temp = U.T@(self.y - self.X @ self.beta_delta).squeeze()
        print(self.y.shape)
        print(temp.shape)
        sigma_g2 = 1/n * \
            np.sum(temp
                   ** 2/(self.S + delta))

        sigma_g22 = 1/n * \
            np.sum(temp.T @ u.inv(np.diag(self.S + delta)) @ temp)

        print('sigma_g2:', sigma_g2, 'sigma_g22:', sigma_g22)

        sigma_temp = 1/n * \
            np.sum((U1.T@(self.y - self.X @ self.beta_delta))
                   ** 2/(self.S[:self.rank] + delta))
        I_minus_U1U1t = np.identity(n) - U1 @ U1.T
        # I_minus_U1U1t = U2 @ U2.T

        I_U1U1Ty_minus_I_U1U1TXbeta = I_minus_U1U1t@self.y - \
            I_minus_U1U1t@self.X @ self.beta_delta

        sigma_g2_lowrank = sigma_temp + 1/n * 1/delta * \
            np.sum(I_U1U1Ty_minus_I_U1U1TXbeta ** 2)

        print('see what happened')
        print(sigma_g2, sigma_g2_lowrank)

    def _buffer_preCalculation_with_delta(self, delta):
        '''
        It is a pre-calculation of some matrix calculatons.
        When delta is given, some matrix calculations take place several time.
        This function is meant to calculate these pieces in advance to save some time.
        '''
        self.delta = delta
        self.beta_delta = self._beta(delta)

        self.UTy_minus_UTXbeta = self.UTy - self.UTX @ self.beta_delta

        # already calculated in _beta
        # self.UTXT_inv_S_delta_UTX = \
        #     (self.UTX).T / (self.S + delta) @ (self.UTX)
        # self.UTXT_inv_S_delta_UTy = \
        #     (self.UTX).T / (self.S + delta) @ (self.UTy)

        if self.lowRank:
            self.I_UUTy_minus_I_UUTXbeta = self.I_minus_UUT_y - \
                self.I_minus_UUT_X @ self.beta_delta

    def _beta(self, delta):
        '''
        beta_function of delta
        '''
        self.UTXT_inv_S_delta_UTX = \
            (self.UTX).T / (self.S + delta) @ (self.UTX)
        self.UTXT_inv_S_delta_UTy = \
            (self.UTX).T / (self.S + delta) @ (self.UTy)

        if self.lowRank:
            inversepart = self.UTXT_inv_S_delta_UTX +\
                1/delta * self.I_minus_UUT_X.T @ self.I_minus_UUT_X
            beta = u.inv(inversepart) @\
                (self.UTXT_inv_S_delta_UTy + 1/delta * self.I_UUTX_I_UUTy)

        else:
            inversepart = self.UTXT_inv_S_delta_UTX
            beta = u.inv(inversepart) @ \
                self.UTXT_inv_S_delta_UTy

        return beta

    def _sigma_g2(self, delta):
        '''
        Sigma_g2 function of delta 
        '''
        # Update buffer
        if delta != self.delta:
            self._buffer_preCalculation_with_delta(delta)

        n, d = self.X.shape

        # the squeeze is making shape from (n,1) to (n,)
        sigma_g2 = 1/n * \
            np.sum(self.UTy_minus_UTXbeta.squeeze() ** 2/(self.S + delta))

        if self.lowRank:

            sigma_g2 += 1/n * 1/delta * \
                np.sum(self.I_UUTy_minus_I_UUTXbeta ** 2)

        # from formula in page 10, the sigma_g2 of REML is given by
        if self.REML:
            sigma_g2 = sigma_g2 * n / (n-d)

        return sigma_g2.squeeze()

    def _log_likelhood_delta(self, delta):
        '''
        log likehood function of delta
        '''

        # Update buffer
        if delta != self.delta:
            self._buffer_preCalculation_with_delta(delta)

        n = self.X.shape[0]

        if self.lowRank:
            k = self.rank
            LL = -1/2 * (
                n*np.log(2*np.pi) + np.sum(np.log(self.S + delta)) +
                (n - k) * np.log(delta) + n +
                n * np.log(1/n * (
                    np.sum(self.UTy_minus_UTXbeta.squeeze()**2/(self.S + delta)) +
                    np.sum((self.I_UUTy_minus_I_UUTXbeta)**2) / delta
                ))
            )
        else:
            LL = -1/2 * (
                n*np.log(2*np.pi) + np.sum(np.log(self.S + delta)) + n +
                n * np.log(
                    1/n * np.sum((self.UTy_minus_UTXbeta.squeeze()
                                 ** 2)/(self.S + delta))
                )
            )
        return LL.squeeze()

    def _restricted_log_likelihood(self, delta):
        '''
        restricted log likelihood function
        '''
        # Update buffer
        if delta != self.delta:
            self._buffer_preCalculation_with_delta(delta)

        n, d = self.X.shape

        if self.lowRank:
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta)) + self.log_XTX -
                np.log(
                    det(self.UTXT_inv_S_delta_UTX + self.I_UUTX_sq/delta)
                )
            )
        else:
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta)) + self.log_XTX -
                np.log(
                    det(self.UTXT_inv_S_delta_UTX)
                )
            )

        if REMLL.shape == (1, 1):
            REMLL = REMLL.reshape((1,))

        return REMLL

    def _neg_cover(self):
        if self.REML:
            def neg_LL(d):
                self._buffer_preCalculation_with_delta(d)
                return -self._restricted_log_likelihood(d)
        else:
            def neg_LL(d):
                self._buffer_preCalculation_with_delta(d)
                return -self._log_likelhood_delta(d)

        return neg_LL

    def _optimization(self, fun):
        # Using - 'brent' method for optimization

        deltas = np.logspace(-10, 10, 21)

        local_minimums = []
        minimum_values = []
        for i in range(len(deltas) - 1):
            # bracket = opt.bracket(fun, xa = deltas[i], xb = deltas[i+1])
            bounds = (deltas[i], deltas[i+1])
            minimize_result = opt.minimize_scalar(
                fun, bounds=bounds, method='bounded')
            x = minimize_result.x
            funs = minimize_result.fun

            if (type(x) != np.ndarray):
                local_minimums.append(x)
            else:
                local_minimums += x.tolist()
            if (type(fun) != np.ndarray):
                minimum_values.append(funs)
            else:
                minimum_values += funs.tolist()

        min_value = min(minimum_values)
        # minmums = [local_minimums[i] for i, v in enumerate(minimum_values) if v == min_value]
        minmum = local_minimums[minimum_values.index(min_value)]
        return minmum, min_value

    def _optimization2(self, fun):
        minimize_result = opt.minimize_scalar(fun, method='brent')
        x = minimize_result.x
        minimize_value = minimize_result.fun
        return x, minimize_value

    def test(self, d):
        print('testing using delta: ', d)
        print('beta is {}'.format(self._beta(d)))
        print('sigma g2 is {}'.format(self._sigma_g2(d)))
        print('liklihood is {}'.format(self._log_likelhood_delta(d)))
        if self.REML:
            print('restricted liklihood is {}'.format(
                self._restricted_log_likelihood(d)))
        print('end of testing')
