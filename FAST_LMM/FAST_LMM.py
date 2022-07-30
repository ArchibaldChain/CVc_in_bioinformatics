import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse.csgraph import structural_rank
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

    def __init__(self, sparse=False, REML=False):
        self.sparse = sparse
        self.REML = REML

    def fit(self, X, y, W=None):
        self.X = np.array(X).astype('float64')
        self.y = np.array(y).reshape([-1, 1])
        n, d = self.X.shape
        self.K = 1/d * self.X @ self.X.T
        Xcopy = self.X.copy()
        if self.sparse:
            if not u.issparse(X):
                warnings.warn('X is set as sparse, but actually not sparse.')
            self.rank = matrix_rank(Xcopy)

            print('rank of X is {}'.format(self.rank))
            if self.rank >= min(X.shape):
                warnings.warn(
                    'The rank of X is equal to the min of shape of X, so set sparse as False.')
                self.sparse = False
                U, S, _ = svd(Xcopy, overwrite_a=True)
            else:
                U, S, _ = svds(Xcopy, self.rank)
        else:
            if u.issparse(X):
                warnings.warn(
                    "X is not set as sparse, but actually sparse.")
            U, S, _ = svd(Xcopy, overwrite_a=True)

        self.U = U

        # check if S is a matrix
        if S.ndim > 1:
            S = np.diag(S)

        # in case that U.shape[1] > S.shape[0]
        k = U.shape[1]
        if len(S) < k:
            S = np.concatenate([S, np.zeros(k - len(S))])

        self.S = 1/d * S ** 2
        self._buffer_preCalculation()

    def _buffer_preCalculation(self):
        n, _ = self.X.shape
        self.UTX = self.U.T @ self.X
        self.UTy = self.U.T @ self.y
        self.I_minus_UUT = np.identity(n) - self.U @ self.U.T
        self.I_minus_UUT_X = self.I_minus_UUT @ self.X
        self.I_minus_UUT_y = self.I_minus_UUT @ self.y
        self.I_UUTX_sq = self.I_minus_UUT_X.T @ self.I_minus_UUT_X
        self.I_UUTX_I_UUTy = self.I_minus_UUT_X.T @ self.I_minus_UUT_y

    def _buffer_preCalculation_with_delta(self, delta):
        '''
        It is a pre-caldulation of some matrix calculatons.
        When delta is given, some matrix calculations take place several time.
        This function is meant to calculate these pieces in advance to save some time.
        '''
        self.delta = delta
        self.UTXT_inv_S_delta_UTX = \
            (self.UTX).T / (self.S + delta) @ (self.UTX)
        self.UTXT_inv_S_delta_UTy = \
            (self.UTX).T / (self.S + delta) @ (self.UTy)
        self.beta_delta = self._beta(delta)
        self.UTy_minus_UTXbeta = self.UTy - self.UTX @ self.beta_delta
        self.I_UUTy_minus_I_UUTXbeta = self.I_minus_UUT_y - \
            self.I_minus_UUT_X @ self.beta_delta

    def _beta(self, delta):
        '''
        beta_function of delta
        '''

        # Update buffer
        if delta != self.delta:
            self._buffer_preCalculation_with_delta(delta)

        if self.sparse:
            inversepart = self.UTXT_inv_S_delta_UTX +\
                1/delta * self.I_minus_UUT_X.T @ self.I_minus_UUT_X
            beta = u.inv(inversepart) @\
                (self.UTXT_inv_S_delta_UTy + 1/delta * self.I_UUTX_I_UUTy)

        else:
            inversepart = self.UTX.T / (self.S + delta) @ self.UTX

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

        sigma_g2 = 1/n * np.sum(self.UTy_minus_UTXbeta ** 2/(self.S + delta))

        if self.sparse:

            sigma_g2 += 1/n * 1/delta * \
                np.sum(self.I_UUTy_minus_I_UUTXbeta ** 2)
            if self.REML:
                pass  # waiting to implement

        elif self.REML:
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

        if self.sparse:
            k = self.rank
            LL = -1/2 * (
                n*np.log(2*np.pi) + np.sum(np.log(self.S + delta)) +
                (n - k) * np.log(delta) + n +
                n * np.log(1/n * (
                    np.sum(self.UTy_minus_UTXbeta**2/(self.S + delta)) +
                    np.sum(np.square(self.I_UUTy_minus_I_UUTXbeta)) / delta
                ))
            )
        else:
            LL = -1/2 * (
                n*np.log(2*np.pi) + np.sum(np.log(self.S + delta)) + n +
                n * np.log(
                    1/n * np.sum((self.UTy_minus_UTXbeta**2)/(self.S + delta))
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

        if self.sparse:
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta)) -
                np.log(
                    det(self.UTXT_inv_S_delta_UTX + self.I_UUTX_sq/delta)
                )
            )
        else:
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta)) -
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
            print(x, bounds)
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
        print('testing')
        print('beta is {}'.format(self._beta(d)))
        print('sigma g2 is {}'.format(self._sigma_g2(d)))
        print('liklihood is {}'.format(self._log_likelhood_delta(d)))
        print('restricted liklihood is {}'.format(
            self._restricted_log_likelihood(d)))
        print('end of testing')
