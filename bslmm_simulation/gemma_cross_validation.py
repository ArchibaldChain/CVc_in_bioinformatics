from gemma_operator import *
import bimbam
import pandas as pd
import numpy as np
import os
from k_folds_cv_helper import get_folds_indices
import utils as u
import re


def bslmm_train_test(geno_tr: pd.DataFrame,
                     geno_te: pd.DataFrame,
                     pheno_tr: np.ndarray,
                     pheno_te: np.ndarray,
                     train_output_prefix="train_output",
                     test_output_prefix="test_output",
                     relateness_tr=None,
                     relateness_full=None):

    train_output = './output'
    var_prefix = 'var_est'

    # extract the data
    X = geno_tr.iloc[:, 3:].T.to_numpy()
    n_tr, p = X.shape
    # X_te = geno_te.iloc[n_tr:, 3:].T.to_numpy()

    # training and testing using GEMMA
    if relateness_tr is None:
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
        K = 1 / p * X @ X.T  # relateness matrix for training set
        gemma_var_estimator(pheno_tr, K, var_prefix)


    # read the variance components
    sigmas = GemmaOutputReader.gemma_var_reader(var_prefix)

    # get the test prediction
    pheno_te_pred = GemmaOutputReader.gemma_pred_reader(test_output_prefix)
    error_te = np.mean(np.square(pheno_te_pred - pheno_te))
    return sigmas, error_te

def cv_correction(
        geno_tr:pd.DataFrame,
        geno_te:pd.DataFrame,
        pheno_tr:np.ndarray,
        pheno_te:np.ndarray,
        H_cv:np.ndarray,
        sigmas):
    
    # extract geno data from bimbam format
    X = geno_tr.iloc[:, 3:].to_numpy().T
    n_tr, p = X.shape
    X_te = geno_te.iloc[:, 3:].to_numpy().T
    n_te = X_te.shape[0]

    # get data from bimbam format creating variance and covariance
    K_tr = 1/p * X @ X.T
    K_te_tr = 1/p * X_te @ X.T
    V_tr = sigmas[0] *  K_tr + sigmas[1] * np.identity(n_tr)
    V_te_tr = sigmas[0] * K_te_tr

    # calculating the h_te
    V_tr_inv = u.inv(V_tr)
    tmp = u.inv(X.T @ V_tr_inv @ X) @ X.T @ V_tr_inv
    h_te = X_te @ tmp\
            + (sigmas[0] + sigmas[1]) * K_te_tr @ V_tr @  (np.identity(n_tr) - X @ tmp)
    

    w = 1/n_tr* (np.trace(H_cv @ V_tr)) - 1/n_te * (np.trace(h_te @ V_te_tr.T))
    w = 2 * w

    return w, h_te


def gemma_cross_validation(
        geno_tr,
        pheno_tr,
        nfolds=5):
    
    # get cross-validation folds
    n_tr = len(pheno_tr)
    indices = get_folds_indices(nfolds, n_tr)

    # cross-validation
    CV_error = 0
    H_cv = np.zeros([n_tr, n_tr])

    for k, ind in enumerate(indices):

        print(f'##### Calculating the {k}-th folds #####')

        # creating the k-th folds
        ia, ib = ind
        ind_k_fold = list(range(ia, ib))
        ind_minus_k_fold = list(range(ia)) + list(range(ib, n_tr))

        # split the data by k-th folds and minus k-th folds
        output_files = bimbam.data_spliter(geno_tr, pheno_tr,
                                                 ind_minus_k_fold, ind_k_fold)
        (geno_minus_k, pheno_minus_k), (geno_k, pheno_k ) = output_files

        ## extract data from the bimbam
        X_minus_k = geno_minus_k.iloc[:, 3:].to_numpy().T.astype(np.float64)
        X_k = geno_k.iloc[:, 3:].to_numpy().T.astype(np.float64)
        n_minus_k, p = X_minus_k.shape
        
        print(X_minus_k.dtype)
        K_kk = 1/p * X_minus_k @ X_minus_k.T
        K_k_minusk= 1/p * X_k @ X_minus_k.T

        # getting error for k fold and sigma estimates
        sigmas, error_k = bslmm_train_test(geno_minus_k, geno_k,
                                                pheno_minus_k, pheno_k,
                                                 f"cv_{k}th_fold_tr",
                                                 f"cv_{k}th_fold_te")

        # creating the Variance for minus-k folds and for k minus-k folds
        V_minus_k = sigmas[0] *  K_kk + sigmas[1] * np.identity(n_minus_k)
        V_minus_k_inv = u.inv(V_minus_k)
        inverse_part = u.inv(X_minus_k.T @ V_minus_k_inv @ X_minus_k)
        tmp = inverse_part @ X_minus_k.T @ V_minus_k_inv

        # Calculating Hcv
        H_cv_block = X_k @ tmp+ \
            (sigmas[0] + sigmas[1]) * K_k_minusk @ V_minus_k_inv @ (np.identity(n_minus_k) - X_minus_k @ tmp )

        H_cv[ia:ib, ind_minus_k_fold] = H_cv_block

        # calculating the cv error
        CV_error += error_k

    CV_error *= 1/n_tr

    # Delete the temporary output files
    try:
        output = './output'
        dirs = os.listdir(output)
        rex = re.compile('(cv_)')
        rm_dirs = [d for d in dirs if rex.match(d)]
        for dir in rm_dirs:
            os.remove(os.path.join(output,dir))
    except FileNotFoundError as e:
        print(e)

    return CV_error, H_cv  

def main():

    geno_tr_dir = './bimbam_data/bimbam_train'
    geno_te_dir = './bimbam_data/bimbam_test'
    pheno_tr_dir = './bimbam_data/pheno_tr'
    pheno_te_dir = './bimbam_data/pheno_te'
    geno_tr = pd.read_csv(geno_tr_dir, header=None, index_col=None)
    geno_te = pd.read_csv(geno_te_dir, header=None, index_col=None)
    pheno_tr = np.loadtxt(pheno_tr_dir)
    pheno_te = np.loadtxt(pheno_te_dir)

    train_output="train_ouput",
    test_output="test_output"
    
    sigmas, error_te = bslmm_train_test(geno_tr, geno_te, 
                                        pheno_tr,pheno_te, 
                                        train_output, test_output)

    CV_error, H_cv = gemma_cross_validation(geno_tr, geno_te,  nfolds=2)
    w = cv_correction(geno_tr,geno_te, pheno_tr, pheno_te, H_cv, sigmas )

    print(CV_error, w, error_te)
    print(f"Test error is {error_te}.")

if __name__ == '__main__':

    # x = geno_tr.iloc[:, 3:].to_numpy().T.astype(np.float64)
    # print(x.dtype)

    # sigma, error_tr, error_te, tr_pred,tr = bslmm_train_test(geno_tr, geno_te, pheno_tr,
    # print(sigma, error_tr, error_te)



