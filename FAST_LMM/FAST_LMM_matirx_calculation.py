import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import issparse
from scipy.sparse.csgraph import structural_rank
from numpy.linalg import matrix_rank
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from numpy.linalg import LinAlgError
import scipy.optimize as opt
import warnings


class FASTLMM:
    beta = None  # coefficients

    sigma_g2 = None  # gene variance
    sigma_e2 = None  # phenotype variance
    delta = None  # ratio of sigma_e2 to sigma_g2
    U = None  # eigenvector matrix of K
    S = None  # eigenvalues array of K

    def __init__(self, sparse=False, REML=False):
        self.sparse = sparse
        self.REML = REML

    def fit(self, X, y, w = None):
        self.X = np.array(X).astype('float64')
        self.y = np.array(y).reshape([-1, 1])
        n, d = self.X.shape
        self.K = 1/d * self.X @ self.X.T
        if self.sparse:
            if issparse(X):
                if not self.sparse:
                    raise warnings.warn('X is sparse.')
                self.rank = structural_rank(X)
            else:
                if self.sparse:
                    raise warnings.warn('X is not sparse.')
                self.rank = matrix_rank(X.copy())
            print('rank of X is {}'.format(self.rank))

            if self.rank == min(X.shape):
                self.sparse = False
            U, S, _ = svds(self.X, self.rank)
        else:
            Xcopy = self.X.copy()
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

    def _beta(self, delta):
        '''
        beta_function of delta
        '''

        # some assoication of matrix multiplication, for computaion simplicity
        UTX = self.U.T @ self.X
        S_plus_delta_inv = np.diag(1 / (self.S + delta))

        if self.sparse:
            n = self.X.shape[0]
            temp1 = UTX.T @ S_plus_delta_inv @ UTX
            temp2 = np.identity(n) - self.U @ self.U.T
            temp3 = (temp2 @ self.X).T 
            inversepart = temp1 + 1/delta * temp3 @ temp3.T
            beta = self._inv(inversepart) @\
                (UTX.T @ S_plus_delta_inv@ (self.U.T @self.y) 
                + 1/delta * temp3 @ temp2 @ self.y)

        else:
            inversepart = UTX.T @ S_plus_delta_inv @ UTX
            beta = self._inv(inversepart) @ \
                (UTX.T @ S_plus_delta_inv @ self.U.T @ self.y)

        return beta

    def _sigma_g2(self, delta):
        '''
        Sigma_g2 function of delta
        '''
        S_plus_delta_inv = np.diag(1 / (self.S + delta))
        n,d = self.X.shape

        temp1 = self.U.T @ self.y - \
            self.U.T @ self.X @ self._beta(delta)
        sigma_g2 = 1/n * (temp1.T @ S_plus_delta_inv @ temp1)

        if self.sparse:
            temp2 = (np.identity(n) - self.U @ self.U.T)@self.y - \
                (np.identity(n) - self.U @ self.U.T)@self.X @ self._beta(delta)
            
            sigma_g2 += 1/n * 1/delta * (temp2.T @ temp2)
            if self.REML:
                pass # waiting to implement
        
        elif self.REML:
            sigma_g2 = sigma_g2 * n / (n-d)

        return sigma_g2.squeeze()

    def _log_likelhood_delta(self, delta):
        '''
        log likehood function of delta
        '''
        n = self.X.shape[0]
        S_plus_delta_inv = np.diag(1 / (self.S + delta))

        UTy_minus_UTXbeta = self.U.T @ self.y - \
            self.U.T @ self.X @ self._beta(delta)
        y_UUTY_X_UUTXbeta = self.y - self.U @ (self.U.T @self.y) - \
            (self.X - self.U @ (self.U.T @ self.X)) @ self._beta(delta)

        if self.sparse:
            k = self.rank
            LL = -1/2 * (
                n*np.log(2*np.pi) + np.sum(np.log(self.S + delta)) +
                (n - k) * np.log(delta) + n +
                n * np.log(1/n * (
                    UTy_minus_UTXbeta.T @ S_plus_delta_inv @ UTy_minus_UTXbeta +
                    np.sum(np.square(y_UUTY_X_UUTXbeta))/delta
                ))
            )
        else:
            LL = -1/2 * (
                n*np.log(2*np.pi) + np.sum(np.log(self.S + delta)) + n +
                n * np.log(1/n * UTy_minus_UTXbeta.T @
                           S_plus_delta_inv @ UTy_minus_UTXbeta)
            )
        return LL.squeeze()

    def _restricted_log_likelihood(self, delta):
        S_plus_delta_inv = np.diag(1 / (self.S + delta))
        n, d = self.X.shape

        if self.sparse:
            I_UUTX = (np.identity(n) - self.U @ self.U.T)@ self.X
            temp = I_UUTX.T @ I_UUTX
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta)) -
                np.log(
                    det((self.U.T@self.X).T @
                           S_plus_delta_inv @ (self.U.T@self.X) + temp/delta
                           ))
                )
        else:
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta)) -
                np.log(det((self.U.T@self.X).T @ 
                              S_plus_delta_inv @ (self.U.T@self.X)))
            )
            
            # beta = self._beta(delta)

            # V_inv = self.U.T @ S_plus_delta_inv @ self.U
            # temp = (self.y - self.X) @ beta
            # REMLL = self._log_likelhood_delta(delta) - \
            #     1/2 * ( np.log(det(V_inv)) + temp.T @ V_inv @ temp)

        if REMLL.shape == (1, 1): 
            REMLL = REMLL.reshape((1,))
        
        return REMLL

    def _neg_cover(self):
        if self.REML:
            neg_LL = lambda d: -self._restricted_log_likelihood(d)
        else:
            neg_LL = lambda d: -self._log_likelhood_delta(d)

        return neg_LL

    def _optimization(self, fun):
        # Using - 'brent' method for optimization
        
        deltas = np.logspace(-10, 10, 21)

        local_minimums = []
        minimum_values = []
        for i in range(len(deltas) - 1):
            # bracket = opt.bracket(fun, xa = deltas[i], xb = deltas[i+1])
            bounds = (deltas[i], deltas[i+1])
            minimize_result = opt.minimize_scalar(fun, bounds=bounds, method='bounded')
            x = minimize_result.x
            funs = minimize_result.fun
            print(x, bounds)
            if (type(x) != np.ndarray ):
                local_minimums.append(x)
            else:
                local_minimums += x.tolist()
            if (type(fun) != np.ndarray ):
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



    def _inv(self, Matrix):
        print('call inv')
        try:
            inv_mat = inv(Matrix)
        except LinAlgError as lae:
            if str(lae) != "Singular matrix":
                print('Determint is {}'.format(det(Matrix)))
                print('shape is {}'.format(Matrix.shape))
                raise lae

            print('Singluar Matrix')
            inv_mat = pinv(Matrix)
        finally:
            return inv_mat
            

    def test(self,d):
        # print('testing')
        # print('beta is {}'.format(self._beta(d)))
        # print('sigma g2 is {}'.format(self._sigma_g2(d)))
        # print('liklihood is {}'.format(self._log_likelhood_delta(d)))
        print('restricted liklihood is {}'.format(self._restricted_log_likelihood(d)))
        print('end of testing')


