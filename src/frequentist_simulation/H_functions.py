from utils import *


def H_function_lmm(X_tr, H_cv_wls_k, V):
    n_tr = X_tr.shape[0]
    H_cv_lmm_k = np.zeros([n_tr, n_tr])
    V_inv = inv(V)
    inverse = inv(X_tr.T @ V_inv @ X_tr)
    H_cv_temp = X_tr @ inverse @ X_tr.T @ V_inv

    folds_indices = get_folds_indices(10, n_tr)
    # get H_temp
    for Kindices in folds_indices:
        ia, ib = Kindices
        indices_minus_K = list(range(0, ia)) + list(range(ib, n_tr))

        V_minus_k = V[indices_minus_K, :][:, indices_minus_K]
        X_k = X_tr[ia:ib, :]
        X_minus_k = X_tr[indices_minus_K, ]

        V_inv = inv(V_minus_k)

        H_cv_temp = X_minus_k @ inverse @ X_minus_k.T @ V_inv

        temp_u = V[ia:ib, indices_minus_K] @ inv(V_minus_k) @ (
            np.identity(n_tr - (ib - ia)) -
            H_cv_temp  #[indices_minus_K,:][:, indices_minus_K]
        )

        H_cv_lmm_k[ia:ib,
                   indices_minus_K] = H_cv_wls_k[ia:ib,
                                                 indices_minus_K] + temp_u
    return H_cv_lmm_k


def H_function_wls(X_minus_k, X_k, **kwargs):
    V = kwargs['V']
    V_inv = inv(V)
    inverse = inv(X_minus_k.T @ V_inv @ X_minus_k)
    return X_k @ inverse @ X_minus_k.T @ V_inv


def H_function_ols(X_minus_k, X_k, **kwargs):
    return X_k @ inv(X_minus_k.T @ X_minus_k) @ X_minus_k.T


def H_function_ridge(X_minus_k, X_k, **kwargs):
    try:
        lamb = kwargs['lamb']
    except KeyError as e:
        print(e, 'key lamb not exist and will be set as 100')
        lamb = 100

    return X_k @ inv(X_minus_k.T @ X_minus_k +
                     lamb * np.identity(X_minus_k.shape[1])) @ X_minus_k.T
