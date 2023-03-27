import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from FAST_LMM import FASTLMM
from k_folds_cv_helper import *


## load the data
def load(file='data/SNP_in_200GENE_chr1.csv'):
    print(os.getcwd())
    print(os.listdir('./data'))
    data = pd.read_csv(file, error_bad_lines=False)
    print(data.head())

    data_selected = data.iloc[:10000, :]
    SNP = data_selected.drop(['GENE', 'POS'], axis=1).T
    return SNP


def one_time_simulation(SNP):
    ## get the Beta

    large_effect = 400
    d = 10
    beta_large = np.random.normal(0, np.sqrt(large_effect), d)

    n = SNP.shape[1]
    small_effect = 2
    beta_small = np.random.normal(0, np.sqrt(small_effect), n - d)

    beta = np.concatenate([beta_large, beta_small])

    ## calculate residual and y
    temp = SNP.values @ beta
    sigma_g2 = np.var(temp, ddof=1)  # get the overall variance

    h2 = 0.5  # heritability

    sigma_e2 = sigma_g2 * (1 - h2) / h2
    beta0 = np.random.normal(0, np.sqrt(large_effect))
    beta_ = np.insert(beta, 0, beta0)

    residual = np.random.normal(0, np.sqrt(sigma_e2), len(SNP))
    y = SNP.values @ beta + residual + beta0

    ## standardnize y
    mu_y = np.mean(y)
    sd_y = np.std(y, ddof=1)
    y = (y - mu_y) / sd_y

    ## train test split
    # add 1 to the first row
    data = np.concatenate([np.ones([SNP.shape[0], 1]), SNP.values], axis=1)
    G_tr, G_te, y_tr, y_te = train_test_split(data,
                                              y,
                                              test_size=0.2,
                                              random_state=123)
    d = 500
    X_tr, X_te = G_tr[:, :d + 1], G_te[:, :d + 1]
    W_tr, W_te = G_tr[:, 1:], G_te[:, 1:]

    ## using FaST-LMM
    sc = W_tr.shape[1]
    fast = FASTLMM(lowRank=True, REML=False)
    fast.fit(X_tr, y_tr, 1 / np.sqrt(sc) * W_tr)

    ## calculating H for 10-folds CV
    def H_function_ols(X_minus_k, X_k, y_minus_k, y_k):
        return X_k @ u.inv(X_minus_k.T @ X_minus_k) @ X_minus_k.T

    H_cv_ols_k = getHcv_for_Kfolds(X_tr, y_tr, H_function_ols, nfolds=10)

    def H_function_wls(X_minus_k, X_k, y_minus_k, y_k, V):
        V_inv = u.inv(V)
        inverse = u.inv(X_minus_k.T @ V_inv @ X_minus_k)
        return X_k @ inverse @ X_minus_k.T @ V_inv

    H_cv_wls_k = getHcv_for_Kfolds(X_tr,
                                   y_tr,
                                   H_function_wls,
                                   fast.V(),
                                   nfolds=10)

    def H_function_lmm():
        n_tr = X_tr.shape[0]
        H_cv_lmm_k = np.zeros([n_tr, n_tr])
        V_inv = fast.V_inv()
        V = fast.V()
        inverse = u.inv(X_tr.T @ V_inv @ X_tr)
        H_cv_temp = X_tr @ inverse @ X_tr.T @ V_inv

        folds_indices = get_folds_indices(10, n_tr)
        # get H_temp
        for Kindices in folds_indices:
            ia, ib = Kindices
            indices_minus_K = list(range(0, ia)) + list(range(ib, n_tr))

            V_minus_k = V[indices_minus_K, :][:, indices_minus_K]
            X_k = X_tr[ia:ib, :]
            X_minus_k = X_tr[indices_minus_K, ]

            V_inv = u.inv(V_minus_k)

            H_cv_temp = X_minus_k @ inverse @ X_minus_k.T @ V_inv

            temp_u = V[ia:ib, indices_minus_K] @ u.inv(V_minus_k) @ (
                np.identity(n_tr - (ib - ia)) -
                H_cv_temp  #[indices_minus_K,:][:, indices_minus_K]
            )

            H_cv_lmm_k[ia:ib,
                       indices_minus_K] = H_cv_wls_k[ia:ib,
                                                     indices_minus_K] + temp_u
        return H_cv_lmm_k

    H_cv_lmm_k = H_function_lmm()

    lamb = 100

    def H_function_ridge(X_minus_k, X_k, y_minus_k, y_k):
        return X_k @ u.inv(X_minus_k.T @ X_minus_k + lamb *
                           np.identity(X_minus_k.shape[1])) @ X_minus_k.T

    H_cv_ridge_k = getHcv_for_Kfolds(X_tr, y_tr, H_function_ridge, nfolds=10)

    ## get random generated and sampled X and W
    n_tr = X_tr.shape[0]
    n_te = X_te.shape[0]
    W_te_random = np.random.choice([0, 1, 2], [n_te, sc])
    X_te_random = np.concatenate([np.ones([n_te, 1]), W_te_random[:, :d]],
                                 axis=1)
    samples = np.random.choice(np.arange(n_tr), n_te)
    W_te_sample = W_tr[samples, :]
    X_te_sample = X_tr[samples, :]

    ## calculating CV CVc and test error
    # Using Test Error to estimate the Generalization error
    Error_cv_lmm = 1 / n_tr * (np.sum(np.square(y_tr - H_cv_lmm_k @ y_tr)))
    Error_cv_wls = 1 / n_tr * (np.sum(np.square(y_tr - H_cv_wls_k @ y_tr)))
    Error_cv_ols = 1 / n_tr * (np.sum(np.square(y_tr - H_cv_ols_k @ y_tr)))
    Error_cv_ridge = 1 / n_tr * (np.sum(np.square(y_tr - H_cv_ridge_k @ y_tr)))

    V_inv = fast.V_inv()
    # using estimated sigma_g2 to get the Covariance(y_tr, y_te)
    V_tr_te = 1 / sc * fast.sigma_g2 * W_tr @ W_te.T

    H_te_wls = X_te @ u.inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv
    H_te_lmm = H_te_wls + V_tr_te.T @ V_inv @ (np.identity(
        n_tr) - X_tr @ u.inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv)
    H_te_ols = X_te @ u.inv(X_tr.T @ X_tr) @ X_tr.T
    H_te_ridge = X_te @ u.inv(X_tr.T @ X_tr +
                              lamb * np.identity(X_tr.shape[1])) @ X_tr.T

    Error_te_lmm = 1 / n_te * (np.sum(np.square(y_te - H_te_lmm @ y_tr)))
    Error_te_wls = 1 / n_te * (np.sum(np.square(y_te - H_te_wls @ y_tr)))
    Error_te_ols = 1 / n_te * (np.sum(np.square(y_te - H_te_ols @ y_tr)))
    Error_te_ridge = 1 / n_te * (np.sum(np.square(y_te - H_te_ridge @ y_tr)))

    V = fast.V()
    Correction_lmm = 2 * (1 / n_tr * np.trace(H_cv_lmm_k @ V) -
                          1 / n_te * np.trace(H_te_lmm @ V_tr_te))
    Correction_wls = 2 * (1 / n_tr * np.trace(H_cv_wls_k @ V) -
                          1 / n_te * np.trace(H_te_wls @ V_tr_te))
    Correction_ols = 2 * (1 / n_tr * np.trace(H_cv_ols_k @ V) -
                          1 / n_te * np.trace(H_te_ols @ V_tr_te))
    Correction_ridge = 2 * (1 / n_tr * np.trace(H_cv_ols_k @ V) -
                            1 / n_te * np.trace(H_te_ols @ V_tr_te))

    Error_cv_lmm_c = Error_cv_lmm + Correction_lmm
    Error_cv_wls_c = Error_cv_wls + Correction_wls
    Error_cv_ols_c = Error_cv_ols + Correction_ols
    Error_cv_ridge_c = Error_cv_ridge + Correction_ols

    # using  te_generated to correct
    V_tr_te_random = 1 / sc * fast.sigma_g2 * W_tr @ W_te_random.T

    H_te_wls_r = X_te_random @ u.inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv
    H_te_lmm_r = H_te_wls_r + V_tr_te_random.T @ V_inv @ (np.identity(
        n_tr) - X_tr @ u.inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv)
    H_te_ols_r = X_te_random @ u.inv(X_tr.T @ X_tr) @ X_tr.T
    H_te_ridge_r = X_te_random @ u.inv(X_tr.T @ X_tr + lamb *
                                       np.identity(X_tr.shape[1])) @ X_tr.T

    Correction_lmm_r = 2 * (1 / n_tr * np.trace(H_cv_lmm_k @ V) -
                            1 / n_te * np.trace(H_te_lmm_r @ V_tr_te_random))
    Correction_wls_r = 2 * (1 / n_tr * np.trace(H_cv_wls_k @ V) -
                            1 / n_te * np.trace(H_te_wls_r @ V_tr_te_random))
    Correction_ols_r = 2 * (1 / n_tr * np.trace(H_cv_ols_k @ V) -
                            1 / n_te * np.trace(H_te_ols_r @ V_tr_te_random))
    Correction_ridge_r = 2 * (1 / n_tr * np.trace(H_cv_ols_k @ V) - 1 / n_te *
                              np.trace(H_te_ridge_r @ V_tr_te_random))

    Error_cv_lmm_c_r = Error_cv_lmm + Correction_lmm_r
    Error_cv_wls_c_r = Error_cv_wls + Correction_wls_r
    Error_cv_ols_c_r = Error_cv_ols + Correction_ols_r
    Error_cv_ridge_c_r = Error_cv_ridge + Correction_ols_r

    # using  sampled from trainint data to correct
    V_tr_te_sample = 1 / sc * fast.sigma_g2 * W_tr @ W_te_sample.T

    H_te_wls_s = X_te_sample @ u.inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv
    H_te_lmm_s = H_te_wls_s + V_tr_te_sample.T @ V_inv @ (np.identity(
        n_tr) - X_tr @ u.inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv)
    H_te_ols_s = X_te_sample @ u.inv(X_tr.T @ X_tr) @ X_tr.T
    H_te_ridge_s = X_te_sample @ u.inv(X_tr.T @ X_tr + lamb *
                                       np.identity(X_tr.shape[1])) @ X_tr.T

    Correction_lmm_s = 2 * (1 / n_tr * np.trace(H_cv_lmm_k @ V) -
                            1 / n_te * np.trace(H_te_lmm_s @ V_tr_te_sample))
    Correction_wls_s = 2 * (1 / n_tr * np.trace(H_cv_wls_k @ V) -
                            1 / n_te * np.trace(H_te_wls_s @ V_tr_te_sample))
    Correction_ols_s = 2 * (1 / n_tr * np.trace(H_cv_ols_k @ V) -
                            1 / n_te * np.trace(H_te_ols_s @ V_tr_te_sample))
    Correction_ridge_s = 2 * (1 / n_tr * np.trace(H_cv_ols_k @ V) - 1 / n_te *
                              np.trace(H_te_ridge_s @ V_tr_te_sample))

    Error_cv_lmm_c_s = Error_cv_lmm + Correction_lmm_s
    Error_cv_wls_c_s = Error_cv_wls + Correction_wls_s
    Error_cv_ols_c_s = Error_cv_ols + Correction_ols_s
    Error_cv_ridge_c_s = Error_cv_ridge + Correction_ols_s

    lmm_error = [
        Error_cv_lmm, Error_cv_lmm_c, Error_cv_lmm_c_r, Error_cv_lmm_c_s,
        Error_te_lmm
    ]
    print(
        'The CV error of   lmm is {:.4f}, wls is {:.4f}, and ols is {:.4f}, and ridge is {:.4f}'
        .format(Error_cv_lmm, Error_cv_wls, Error_cv_ols, Error_cv_ridge))
    print(
        'The CVc error of  lmm is {:.4f}, wls is {:.4f}, and ols is {:.4f}, and ridge is {:.4f}'
        .format(Error_cv_lmm_c, Error_cv_wls_c, Error_cv_ols_c,
                Error_cv_ridge_c))
    print(
        'The CVc random of  lmm is {:.4f}, wls is {:.4f}, and ols is {:.4f}, and ridge is {:.4f}'
        .format(Error_cv_lmm_c_r, Error_cv_wls_c_r, Error_cv_ols_c_r,
                Error_cv_ridge_c_r))
    print(
        'The CVc sample of  lmm is {:.4f}, wls is {:.4f}, and ols is {:.4f}, and ridge is {:.4f}'
        .format(Error_cv_lmm_c_s, Error_cv_wls_c_s, Error_cv_ols_c_s,
                Error_cv_ridge_c_s))
    print(
        'The test error of lmm is {:.4f}, wls is {:.4f}, and ols is {:.4f}, and ridge is {:.4f}'
        .format(Error_te_lmm, Error_te_wls, Error_te_ols, Error_te_ridge))

    lmm_error = [
        Error_cv_lmm, Error_cv_lmm_c, Error_cv_lmm_c_r, Error_cv_lmm_c_s,
        Error_te_lmm
    ]
    wls_error = [
        Error_cv_wls, Error_cv_wls_c, Error_cv_wls_c_r, Error_cv_wls_c_s,
        Error_te_wls
    ]
    ols_error = [
        Error_cv_ols, Error_cv_ols_c, Error_cv_ols_c_r, Error_cv_ols_c_s,
        Error_te_ols
    ]
    ridge_error = [
        Error_cv_ridge, Error_cv_ridge_c, Error_cv_ridge_c_r,
        Error_cv_ridge_c_s, Error_te_ridge
    ]
    return lmm_error, wls_error, ols_error, ridge_error


if __name__ == '__main__':
    start_time = time.time()

    file_lmm = './data/CV_errors_lmm.csv'
    file_wls = './data/CV_errors_wls.csv'
    file_ols = './data/CV_errors_ols.csv'
    file_ridge = './data/CV_errors_ridge.csv'

    head_lmm = [
        'Error_cv_lmm', 'Error_cv_lmm_c', 'Error_cv_lmm_c_r',
        'Error_cv_lmm_c_s', 'Error_te_lmm'
    ]
    head_wls = [
        'Error_cv_wls', 'Error_cv_wls_c', 'Error_cv_wls_c_r',
        'Error_cv_wls_c_s', 'Error_te_wls'
    ]
    head_ols = [
        'Error_cv_ols', 'Error_cv_ols_c', 'Error_cv_ols_c_r',
        'Error_cv_ols_c_s', 'Error_te_ols'
    ]
    head_ridge = [
        'Error_cv_ridge', 'Error_cv_ridge_c', 'Error_cv_ridge_c_r',
        'Error_cv_ridge_c_s', 'Error_te_ridge'
    ]
    if not os.path.exists(file_lmm):
        with open(file_lmm, 'w') as f:
            head = ','.join(head_lmm)
            f.write(head)
            f.write('\n')

    if not os.path.exists(file_wls):
        with open(file_wls, 'w') as f:
            head = ','.join(head_wls)
            f.write(head)
            f.write('\n')

    if not os.path.exists(file_ols):
        with open(file_ols, 'w') as f:
            head = ','.join(head_ols)
            f.write(head)
            f.write('\n')

    if not os.path.exists(file_ridge):
        with open(file_ridge, 'w') as f:
            head = ','.join(head_ridge)
            f.write(head)
            f.write('\n')

    SNP = load()

    f_lmm = open(file_lmm, 'a')
    f_wls = open(file_wls, 'a')
    f_ols = open(file_ols, 'a')
    f_ridge = open(file_ridge, 'a')

    for i in range(158):
        start_time_i = time.time()
        lmm, wls, ols, ridge = one_time_simulation(SNP)
        temp = ','.join([str(s) for s in lmm])
        print(temp)
        f_lmm.write(temp)
        f_lmm.write('\n')

        f_wls.write(','.join([str(s) for s in wls]))
        f_wls.write('\n')

        f_ols.write(','.join([str(s) for s in ols]))
        f_ols.write('\n')

        f_ridge.write(','.join([str(s) for s in ridge]))
        f_ridge.write('\n')
        print('------ simulation: {}, {:.4f} seconds -----'.format(
            i,
            time.time() - start_time_i))

    f_lmm.close()
    f_wls.close()
    f_ols.close()
    f_ridge.close()

    print('------ {:.4f} seconds -----'.format(time.time() - start_time))