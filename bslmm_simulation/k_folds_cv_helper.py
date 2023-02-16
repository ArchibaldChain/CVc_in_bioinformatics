import numpy as np
import os
import time
import functools


def get_folds_indices(nfolds, n_tr):
    if nfolds <= n_tr:
        fold_size = int(np.floor(n_tr / nfolds))
        resi = n_tr % nfolds
        fold_sizes = [
            0
        ] + resi * [fold_size + 1] + (nfolds - resi) * [fold_size]
        indices = np.cumsum(fold_sizes)
        folds_indices = [(indices[i], indices[i + 1]) for i in range(nfolds)]

    else:
        raise Exception("Number of folds is larger than numer of samples")
    print('First 5 fold indices : {}'.format(folds_indices[:5]))
    return folds_indices




def get_fold_H(X_k, y_k, X_minus_K, y_minus_K, H_function, V_minus_k,
               return_dict, i):
    print('Calculating fold # ', i)
    if V_minus_k is not None:
        temp = H_function(X_minus_K, X_k, y_minus_K, y_k, V_minus_k)
    else:
        temp = H_function(X_minus_K, X_k, y_minus_K, y_k)

    return_dict[i] = temp


def getHcv_for_Kfolds(X_tr, y_tr, H_function, V=None, nfolds=10):
    n_tr, p = X_tr.shape
    Hcv_k = np.zeros([n_tr, n_tr])

    folds_indices = get_folds_indices(nfolds=nfolds, n_tr=n_tr)
    for Kindices in folds_indices:
        ia, ib = Kindices
        indices_minus_K = list(range(0, ia)) + list(range(ib, n_tr))

        X_minus_K = X_tr[indices_minus_K, :]
        y_minus_K = y_tr[indices_minus_K]
        X_k = X_tr[ia:ib, :]
        y_K = y_tr[ia:ib]

        if V is not None:
            V_minus_k = V[indices_minus_K, :][:, indices_minus_K]
            temp = H_function(X_minus_K, X_k, y_minus_K, y_K, V_minus_k)
        else:
            temp = H_function(X_minus_K, X_k, y_minus_K, y_K)

        Hcv_k[ia:ib, indices_minus_K] = temp

    return Hcv_k



def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print('--- start function ' + func.__name__, '---')
        result = func(*args, **kwargs)
        print('------ {:.4f} seconds -----'.format(time.time() - start_time))
        return result

    return wrapper