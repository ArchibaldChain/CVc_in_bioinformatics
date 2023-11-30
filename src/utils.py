import warnings
from numpy.linalg import inv as inv_
from numpy.linalg import pinv
from numpy.linalg import LinAlgError
import numpy as np
import time
import functools

import re
from typing import List

def create_new_file_with_prefix(prefix:str, files:str) -> str:
    """
    Create a new file with the given prefix, with a number in sequence appended to it.  The number is one higher than the highest number used in the existing files.

    Parameters
    -------
    prefix : str
        The prefix for the new file
    files : str
        The list of existing files

    Returns
    -------
    str
        The new prefix name
    """

    # Regular expression to match files with the given prefix and a number, ignoring the suffix
    pattern = re.compile(f'^{prefix}(\d+)(\..+)?$')

    # Find the highest number used in the existing files
    max_number = 0
    for file in files:
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number

    # Create a new file with the next number in sequence, with a generic '.txt' suffix
    new_prefix = f'{prefix}{max_number + 1}'
    return new_prefix

# write a function that takes a list of files and a prefix, and returns all the files with the given prefix
def get_files_with_prefix(prefix:str, files:str) -> List[str]:
    """
    Get all the files with the given prefix

    Parameters
    -------
    prefix : str
        The prefix for the new file
    files : str
        The list of existing files

    Returns
    -------
    str
        The new prefix name
    """

    # Regular expression to match files with the given prefix and a number, ignoring the suffix
    pattern = re.compile(f'^{prefix}(\..+)?$')

    final_list = []
    for file in files:
        match = pattern.match(file)
        if match:
            final_list.append(file)

    return final_list

def issparse(m):
    return np.sum(m == 0) > (m.shape[0] * m.shape[1] / 2)


def inv(matrix):
    try:
        inv_mat = inv_(matrix)
    except LinAlgError as lae:
        warnings.warn(f"Singluar matrix with shape {matrix.shape}")
        if str(lae) != "Singular matrix":
            print('shape is {}'.format(matrix.shape))
            raise lae

        inv_mat = pinv(matrix)
    finally:
        return inv_mat


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
    return folds_indices


def getHcv_for_Kfolds(X_tr, y_tr, H_function, nfolds=10, **kwargs):
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

        if 'V' in kwargs.keys():
            V = kwargs['V']
            V_minus_k = V[indices_minus_K, :][:, indices_minus_K]
            kwargs_copy = kwargs.copy()
            kwargs_copy['V'] = V_minus_k
        else:
            kwargs_copy = kwargs.copy()

        temp = H_function(X_minus_K, X_k, **kwargs_copy)

        Hcv_k[ia:ib, indices_minus_K] = temp

    return Hcv_k


def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print('\n----- start ' + func.__name__, '-----')
        result = func(*args, **kwargs)
        print('--- Finish {} in {:.4f} seconds ---\n'.format(
            func.__name__,
            time.time() - start_time))
        return result

    return wrapper