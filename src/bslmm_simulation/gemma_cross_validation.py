from gemma_operator import *
import bimbam
import pandas as pd
import numpy as np
import os, sys

sys.path.append('./src')
import utils as u
from utils import get_folds_indices, timing
import re

train_output_prefix = "train_ouput"
test_output_prefix = "test_output"


@timing
def bslmm_train_test(geno_tr: pd.DataFrame,
                     geno_te: pd.DataFrame,
                     pheno_tr: np.ndarray,
                     pheno_te: np.ndarray,
                     train_output_prefix="train_output",
                     test_output_prefix="test_output",
                     relatedness_tr=None,
                     relatedness_full=None):

    train_output = './output'
    var_prefix = 'var_est'

    # extract the data
    X = geno_tr.iloc[:, 3:].T.to_numpy()
    n_tr, p = X.shape
    # X_te = geno_te.iloc[n_tr:, 3:].T.to_numpy()

    # training and testing using GEMMA
    if relatedness_tr is None:
        # train and test
        gemma_train(geno_tr, pheno_tr, prefix=train_output_prefix)
        # creating the full geno file for testing
        geno_full, pheno_full = bimbam.test_data_parperation(
            geno_tr, geno_te, pheno_tr, pheno_te)
        gemma_test(geno_full,
                   pheno_full,
                   train_prefix=train_output_prefix,
                   test_prefix=test_output_prefix,
                   train_output_path=train_output)

        # calculate the variance compoents from gemma output
        K = 1 / p * X @ X.T  # relatedness matrix for training set
        sigmas = gemma_var_estimator(pheno_tr, K, var_prefix)
        assert len(
            sigmas) == 2, f'Incorrect sigma number detected. sigmas = {sigmas}'

    elif relatedness_full is not None:
        # train and test
        gemma_train(geno_tr,
                    pheno_tr,
                    prefix=train_output_prefix,
                    related_matrix=relatedness_tr)
        # creating the full geno file for testing
        geno_full, pheno_full = bimbam.test_data_parperation(
            geno_tr, geno_te, pheno_tr, pheno_te)
        gemma_test(geno_full,
                   pheno_full,
                   train_prefix=train_output_prefix,
                   test_prefix=test_output_prefix,
                   train_output_path=train_output,
                   related_matrix=relatedness_full)

        # calculate the variance compoents from gemma output
        K = 1 / p * X @ X.T  # relatedness matrix for training set
        multi_relatedness = [K, relatedness_tr]
        sigmas = gemma_var_estimator(pheno_tr, multi_relatedness, var_prefix)
        assert len(
            sigmas) == 3, f'Incorrect sigma number detected. sigmas = {sigmas}'

    # get the test prediction
    pheno_te_pred = GemmaOutputReader.gemma_pred_reader(test_output_prefix)
    error_te = np.mean(np.square(pheno_te_pred - pheno_te))
    return sigmas, error_te


def cv_correction(geno_tr: pd.DataFrame,
                  geno_te: pd.DataFrame,
                  H_cv: np.ndarray,
                  sigmas,
                  K_relatedness_tr: np.ndarray = None,
                  K_relatedness_tr_te: np.ndarray = None):

    # extract geno data from bimbam format
    X = geno_tr.iloc[:, 3:].to_numpy().T
    n_tr, p = X.shape
    X_te = geno_te.iloc[:, 3:].to_numpy().T
    n_te = X_te.shape[0]

    # get data from bimbam format creating variance and covariance
    if (K_relatedness_tr == None) or (K_relatedness_tr_te == None):
        K_tr = 1 / p * X @ X.T
        K_te_tr = 1 / p * X_te @ X.T
    else:
        assert (K_relatedness_tr.shape[0] == K_relatedness_tr.shape[1]) and (K_relatedness_tr.shape[0] == n_tr),\
        f'Unsupported K_tr_relatedness shape {K_relatedness_tr.shape}'
        assert (K_relatedness_tr_te.shape[0] == n_tr, K_relatedness_tr_te.shape[1] == n_te),\
        f'Unsupported K_tr_te_relatedness shape {K_relatedness_tr_te.shape}'

        K_tr = K_relatedness_tr
        K_te_tr = K_relatedness_tr_te.T

    V_tr = sigmas[0] * K_tr + sigmas[1] * np.identity(n_tr)
    V_te_tr = sigmas[0] * K_te_tr

    # calculating the h_te
    V_tr_inv = u.inv(V_tr)
    tmp = u.inv(X.T @ V_tr_inv @ X) @ X.T @ V_tr_inv
    h_te = X_te @ tmp\
            + (sigmas[0]) * K_te_tr @ V_tr_inv @  (np.identity(n_tr) - X @ tmp)

    w = 1 / n_tr * (np.trace(H_cv @ V_tr)) - 1 / n_te * (np.trace(
        h_te @ V_te_tr.T))
    w = 2 * w

    return w, h_te


def gemma_cross_validation(geno_tr: pd.DataFrame,
                           pheno_tr: np.ndarray,
                           K_relatedness: np.ndarray = None,
                           nfolds=5):

    # get cross-validation folds
    n_tr = len(pheno_tr)
    indices = get_folds_indices(nfolds, n_tr)

    # cross-validation
    CV_error = 0
    H_cv = np.zeros([n_tr, n_tr])

    for k, ind in enumerate(indices):

        print(f'##### Calculating the {k}-th folds #####')
        rand_num = str(np.random.randint(10000))

        # creating the k-th folds
        ia, ib = ind
        ind_k_fold = list(range(ia, ib))
        ind_minus_k_fold = list(range(ia)) + list(range(ib, n_tr))

        # split the data by k-th folds and minus k-th folds
        output_files = bimbam.data_spliter(geno_tr, pheno_tr, ind_minus_k_fold,
                                           ind_k_fold)
        (geno_minus_k, pheno_minus_k), (geno_k, pheno_k) = output_files

        ## extract data from the bimbam
        X_minus_k = geno_minus_k.iloc[:, 3:].to_numpy().T.astype(np.float64)
        X_k = geno_k.iloc[:, 3:].to_numpy().T.astype(np.float64)
        n_minus_k, p = X_minus_k.shape

        if K_relatedness is None:
            K_kk = 1 / p * X_minus_k @ X_minus_k.T
            K_k_minusk = 1 / p * X_k @ X_minus_k.T

            # getting error for k fold and sigma estimates
            sigmas, error_k = bslmm_train_test(geno_minus_k, geno_k,
                                               pheno_minus_k, pheno_k,
                                               f"cv_{rand_num}_{k}th_fold_tr",
                                               f"cv_{rand_num}_{k}th_fold_te")
        else:
            K_kk = K_relatedness
            K_k_minusk = K_relatedness[ind_minus_k_fold, :][:,
                                                            ind_minus_k_fold]

            # getting error for k fold and sigma estimates
            sigmas, error_k = bslmm_train_test(geno_minus_k, geno_k,
                                               pheno_minus_k, pheno_k,
                                               f"cv_{rand_num}_{k}th_fold_tr",
                                               f"cv_{rand_num}_{k}th_fold_te",
                                               K_k_minusk, K_relatedness)

        # creating the Variance for minus-k folds and for k minus-k folds
        V_minus_k = sigmas[0] * K_kk + sigmas[1] * np.identity(n_minus_k)
        V_minus_k_inv = u.inv(V_minus_k)
        inverse_part = u.inv(X_minus_k.T @ V_minus_k_inv @ X_minus_k)
        tmp = inverse_part @ X_minus_k.T @ V_minus_k_inv

        # Calculating Hcv
        H_cv_block = X_k @ tmp+ \
            (sigmas[0]) * K_k_minusk @ V_minus_k_inv @ (np.identity(n_minus_k) - X_minus_k @ tmp )

        H_cv[ia:ib, ind_minus_k_fold] = H_cv_block

        # calculating the cv error
        CV_error += error_k

        def remove_output_files():
            # Delete the temporary output files
            try:
                output = './output'
                dirs = os.listdir(output)
                rex = re.compile(f'(cv_{rand_num})')
                rm_dirs = [d for d in dirs if rex.match(d)]
                print('removing files: ', rm_dirs)
                for dir in rm_dirs:
                    os.remove(os.path.join(output, dir))
            except FileNotFoundError as e:
                print(e)

        remove_output_files()

    CV_error *= 1 / nfolds

    return CV_error, H_cv


def simulation_with_num_fixed_snps(num_fixed_snps: int,
                                   geno_tr: pd.DataFrame,
                                   geno_te: pd.DataFrame,
                                   pheno_tr: np.ndarray,
                                   pheno_te: np.ndarray,
                                   nfolds=10):

    # only using first num_fixed_snps for the fixed terms.
    geno_x_tr = geno_tr.iloc[:num_fixed_snps, :]
    geno_x_te = geno_te.iloc[:num_fixed_snps, :]

    W_tr = geno_tr.iloc[:, 3:].values.T
    W_te = geno_te.iloc[:, 3:].values.T
    print(W_tr.shape, W_te.shape)

    # using 10_000 SNPs to consturct a relatedness maatrix
    W_full = np.concatenate([W_tr, W_te], axis=0)
    n, sc = W_full.shape
    K_related_full = 1 / sc * W_full @ W_full.T
    K_tr = 1 / sc * W_tr @ W_tr.T
    K_tr_te = 1 / sc * W_tr @ W_tr.T
    sigmas, error_te = bslmm_train_test(geno_x_tr, geno_x_te, pheno_tr,
                                        pheno_te, train_output_prefix,
                                        test_output_prefix, K_tr,
                                        K_related_full)

    print('\n####Finished Training####\n', 'sigmas: ', sigmas,
          '  test_error: ', error_te)

    CV_error, H_cv = gemma_cross_validation(geno_tr,
                                            pheno_tr,
                                            K_relatedness=K_tr,
                                            nfolds=nfolds)
    print('\n####Finished Cross-validation####\n', 'CV error: ', CV_error)

    CV_error_H_cv = np.mean(np.square(pheno_tr - H_cv @ pheno_tr))
    print('CV error using H_cv: ', CV_error_H_cv)

    ## save H_cv
    H_cv_dir = './H_dir'
    if not os.path.exists(H_cv_dir):
        os.mkdir(H_cv_dir)
    np.save(os.path.join(H_cv_dir, f'H_cv_{nfolds}_folds'), H_cv)

    ## using h_te to predict
    w, h_te = cv_correction(geno_tr,
                            geno_te,
                            H_cv,
                            sigmas,
                            K_relatedness_tr=K_tr,
                            K_relatedness_tr_te=K_tr_te)
    te_error_h_te = np.mean(np.square(pheno_te - h_te @ pheno_tr))
    print('test error h_te: ', te_error_h_te)

    # save h_te
    print('w: ', w)
    np.save(os.path.join(H_cv_dir, f'H_te_{nfolds}'), h_te)

    print(f"CV error, w, Test error are {CV_error}, {w}, {error_te}.")
    return CV_error, CV_error_H_cv, error_te, te_error_h_te, w


def simulation_with_all_snps(geno_tr: pd.DataFrame,
                             geno_te: pd.DataFrame,
                             pheno_tr: np.ndarray,
                             pheno_te: np.ndarray,
                             nfolds=10):

    sigmas, error_te = bslmm_train_test(geno_tr, geno_te, pheno_tr, pheno_te,
                                        train_output_prefix,
                                        test_output_prefix)

    print('\n####Finished Training####\n', 'sigmas: ', sigmas,
          '  test_error: ', error_te)

    CV_error, H_cv = gemma_cross_validation(geno_tr, pheno_tr, nfolds=nfolds)
    print('\n####Finished Cross-validation####\n', 'CV error: ', CV_error)

    CV_error_H_cv = np.mean(np.square(pheno_tr - H_cv @ pheno_tr))
    print('CV error using H_cv: ', CV_error_H_cv)

    ## save H_cv
    H_cv_dir = './H_dir'
    if not os.path.exists(H_cv_dir):
        os.mkdir(H_cv_dir)
    np.save(os.path.join(H_cv_dir, f'H_cv_{nfolds}_folds'), H_cv)

    ### temp
    # H_cv_dir = './H_dir'
    # H_cv = np.load('./H_dir/H_cv_10.npy')
    # sigmas, error_te = [1.9384, 0.362577]   ,  0.6800525767858235
    # CV_error =  0.624261999338271
    # CV_error_H_cv = np.mean(np.square(pheno_tr - H_cv @ pheno_tr))
    # print('CV error using H_cv: ', CV_error_H_cv)

    ## using h_te to predict
    w, h_te = cv_correction(geno_tr, geno_te, H_cv, sigmas)
    te_error_h_te = np.mean(np.square(pheno_te - h_te @ pheno_tr))
    print('test error h_te: ', te_error_h_te)

    # save h_te
    print('w: ', w)
    np.save(os.path.join(H_cv_dir, f'H_te_{nfolds}'), h_te)

    print(f"CV error, w, Test error are {CV_error}, {w}, {error_te}.")
    return CV_error, CV_error_H_cv, error_te, te_error_h_te, w


if __name__ == '__main__':
    geno_tr_dir = './bimbam_data/bimbam_train'
    geno_te_dir = './bimbam_data/bimbam_test'
    pheno_tr_dir = './bimbam_data/pheno_tr'
    pheno_te_dir = './bimbam_data/pheno_te'
    geno_tr = pd.read_csv(geno_tr_dir, header=None, index_col=None)
    geno_te = pd.read_csv(geno_te_dir, header=None, index_col=None)
    pheno_tr = np.loadtxt(pheno_tr_dir)
    pheno_te = np.loadtxt(pheno_te_dir)
    # simulation_with_all_snps(geno_tr, geno_te, pheno_tr, pheno_te, 10)
    simulation_with_num_fixed_snps(100, geno_tr, geno_te, pheno_tr, pheno_te,
                                   10)
