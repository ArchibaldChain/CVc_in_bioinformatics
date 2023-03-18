import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, sys

sys.path.append('./src')
from utils import *
from bslmm_simulation import gemma_operator as gemma
from arguments import get_args
from frequentist_simulation.H_functions import *


## load the data
def load(file) -> np.ndarray:
    data = pd.read_csv(file)

    data_drop_gene = data.drop('GENE', axis=1)
    data_drop_gene.drop_duplicates(inplace=True)
    data_drop = data_drop_gene.drop('POS', axis=1)
    data_drop_gene_selected = data_drop.sample(10_000, axis=0, random_state=0)
    SNP = data_drop_gene_selected.values.T
    return SNP


def phenotype_simulator(SNPs,
                        num_large_effect=10,
                        large_effect=400,
                        small_effect=1):
    ## get the Beta
    d = num_large_effect
    beta_large = np.random.normal(0, np.sqrt(large_effect), d)

    n, p = SNPs.shape
    small_effect = 2
    beta_small = np.random.normal(0, np.sqrt(small_effect), p - d)

    beta = np.concatenate([beta_large, beta_small])

    ## calculate residual and y
    temp = SNPs @ beta
    sigma_g2 = np.var(temp, ddof=1)  # get the overall variance

    h2 = 0.5  # heritability

    sigma_e2 = sigma_g2 * (1 - h2) / h2
    beta0 = np.random.normal(0, np.sqrt(large_effect))
    beta_ = np.insert(beta, 0, beta0)

    residual = np.random.normal(0, np.sqrt(sigma_e2), n)
    y = SNPs @ beta + residual + beta0

    ## standardnize y
    mu_y = np.mean(y)
    sd_y = np.std(y, ddof=1)
    y = (y - mu_y) / sd_y
    return y


def cross_validation_correction(SNPs: np.ndarray,
                                y: np.ndarray,
                                num_fixed_snps=500,
                                n_folds=10):

    ## train test split
    # add 1 to the first row
    n = SNPs.shape[0]
    data = np.concatenate([np.ones([n, 1]), SNPs], axis=1)
    G_tr, G_te, y_tr, y_te = train_test_split(data, y, test_size=0.2)
    d = num_fixed_snps
    X_tr, X_te = G_tr[:, :d + 1], G_te[:, :d + 1]
    W_tr, W_te = G_tr[:, 1:], G_te[:, 1:]
    n_tr, sc = W_tr.shape

    ## using FaST-LMM
    # fast = FASTLMM(lowRank=True, REML=False)
    # fast.fit(X_tr, y_tr, 1 / np.sqrt(sc) * W_tr)

    ##  Using GEMMA to estimate the variance component
    K_relatedness = 1 / sc * W_tr @ W_tr.T
    sigmas = gemma.gemma_var_estimator(y_tr, K_relatedness, 'var_component')
    print(sigmas)
    V = sigmas[0] * K_relatedness + sigmas[1] * np.identity(n_tr)
    V_inv = inv(V)

    ## calculating H for 10-folds CV
    ### OLS
    H_cv_ols_k = getHcv_for_Kfolds(X_tr, y_tr, H_function_ols, nfolds=n_folds)

    ### WLS
    H_cv_wls_k = getHcv_for_Kfolds(X_tr,
                                   y_tr,
                                   H_function_wls,
                                   nfolds=n_folds,
                                   V=V)

    ### LMM
    H_cv_lmm_k = H_function_lmm(X_tr, H_cv_wls_k, V=V)

    #### Ridge
    lamb = 100
    H_cv_ridge_k = getHcv_for_Kfolds(X_tr,
                                     y_tr,
                                     H_function_ridge,
                                     nfolds=n_folds,
                                     lamb=lamb)

    ## get random generated and sampled from X and W
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

    # using estimated sigma_g2 to get the Covariance(y_tr, y_te)
    V_tr_te = 1 / sc * sigmas[0] * W_tr @ W_te.T

    H_te_wls = X_te @ inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv
    H_te_lmm = H_te_wls + V_tr_te.T @ V_inv @ (
        np.identity(n_tr) - X_tr @ inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv)
    H_te_ols = X_te @ inv(X_tr.T @ X_tr) @ X_tr.T
    H_te_ridge = X_te @ inv(X_tr.T @ X_tr +
                            lamb * np.identity(X_tr.shape[1])) @ X_tr.T

    Error_te_lmm = 1 / n_te * (np.sum(np.square(y_te - H_te_lmm @ y_tr)))
    Error_te_wls = 1 / n_te * (np.sum(np.square(y_te - H_te_wls @ y_tr)))
    Error_te_ols = 1 / n_te * (np.sum(np.square(y_te - H_te_ols @ y_tr)))
    Error_te_ridge = 1 / n_te * (np.sum(np.square(y_te - H_te_ridge @ y_tr)))

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
    Error_cv_ridge_c = Error_cv_ridge + Correction_ridge

    # using  generated Test set to correct
    V_tr_te_random = 1 / sc * sigmas[0] * W_tr @ W_te_random.T

    H_te_wls_r = X_te_random @ inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv
    H_te_lmm_r = H_te_wls_r + V_tr_te_random.T @ V_inv @ (
        np.identity(n_tr) - X_tr @ inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv)
    H_te_ols_r = X_te_random @ inv(X_tr.T @ X_tr) @ X_tr.T
    H_te_ridge_r = X_te_random @ inv(X_tr.T @ X_tr + lamb *
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
    Error_cv_ridge_c_r = Error_cv_ridge + Correction_ridge_r

    # using sampled dataset from training data to correct
    V_tr_te_sample = 1 / sc * sigmas[0] * W_tr @ W_te_sample.T

    H_te_wls_s = X_te_sample @ inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv
    H_te_lmm_s = H_te_wls_s + V_tr_te_sample.T @ V_inv @ (
        np.identity(n_tr) - X_tr @ inv(X_tr.T @ V_inv @ X_tr) @ X_tr.T @ V_inv)
    H_te_ols_s = X_te_sample @ inv(X_tr.T @ X_tr) @ X_tr.T
    H_te_ridge_s = X_te_sample @ inv(X_tr.T @ X_tr + lamb *
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
    Error_cv_ridge_c_s = Error_cv_ridge + Correction_ridge_s

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


def create_files(save_path, num_fixed_snps):
    save = os.path.join(save_path, f"{num_fixed_snps}_fixed_snps")
    if not os.path.exists(save):
        os.makedirs(save, exist_ok=True)

    file_lmm = os.path.join(save_path, f"{num_fixed_snps}_fixed_snps",
                            f'CV_errors_lmm.csv')
    file_wls = os.path.join(save_path, f"{num_fixed_snps}_fixed_snps",
                            f'CV_errors_wls.csv')
    file_ols = os.path.join(save_path, f"{num_fixed_snps}_fixed_snps",
                            f'CV_errors_ols.csv')
    file_ridge = os.path.join(save_path, f"{num_fixed_snps}_fixed_snps",
                              f'CV_errors_ridge.csv')

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
    return file_lmm, file_wls, file_ols, file_ridge


def cross_validation_simulation(SNP_file='data/SNP_in_200GENE_chr1.csv',
                                save_path='simulation_output',
                                num_fixed_snps=500,
                                simulation_times=1000,
                                num_large_effect=10,
                                large_effect=400,
                                small_effect=2,
                                n_folds=10):
    start_time = time.time()

    SNPs = load(SNP_file)
    file_lmm, file_wls, file_ols, file_ridge =\
          create_files(save_path, num_fixed_snps)

    # open the file in advance to save some time
    f_lmm = open(file_lmm, 'a')
    f_wls = open(file_wls, 'a')
    f_ols = open(file_ols, 'a')
    f_ridge = open(file_ridge, 'a')

    for i in range(simulation_times):
        print(f'------ simulation: {i} step -----')
        start_time_i = time.time()
        # simluate y
        y = phenotype_simulator(SNPs, num_large_effect, large_effect,
                                small_effect)
        # get CV error and correction
        lmm, wls, ols, ridge = cross_validation_correction(SNPs,
                                                           y,
                                                           num_fixed_snps,
                                                           n_folds=n_folds)
        temp = ','.join([str(s) for s in lmm])
        f_lmm.write(temp)
        f_lmm.write('\n')

        temp = ','.join([str(s) for s in wls])
        f_wls.write(temp)
        f_wls.write('\n')

        temp = ','.join([str(s) for s in ols])
        f_ols.write(temp)
        f_ols.write('\n')

        temp = ','.join([str(s) for s in ridge])
        f_ridge.write(temp)
        f_ridge.write('\n')
        time_period = round(time.time() - start_time_i, 4)
        print(f'------ simulation: {i}, using {time_period} seconds -----')

    f_lmm.close()
    f_wls.close()
    f_ols.close()
    f_ridge.close()

    print('------ {:.4f} seconds -----'.format(time.time() - start_time))


if __name__ == "__main__":
    args = get_args()

    cross_validation_simulation(args.SNP_file, args.save_path,
                                args.num_fixed_snps, args.simulation_times,
                                args.num_large_effect, args.large_effect,
                                args.small_effect, args.n_folds)
