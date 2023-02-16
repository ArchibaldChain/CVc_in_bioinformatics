import numpy as np
# from multiprocessing import Process, Manager
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


proceses = []
num_processors = os.cpu_count()
print("Num processors: ", num_processors)


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


# def getHcv_for_Kfolds(X_tr, y_tr, H_function, V=None, nfolds=10):
#     # def helper function for parallel computing

#     n_tr, p = X_tr.shape
#     Hcv_k = np.zeros([n_tr, n_tr])

#     folds_indices = get_folds_indices(nfolds=nfolds, n_tr=n_tr)
#     # manager = Manager()
#     # return_dict = manager.dict()
#     return_dict = None
#     jobs = []
#     num_useable_processor = os.cpu_count() - 1

#     # initialize jobs
#     for i, Kindices in enumerate(folds_indices):
#         ia, ib = Kindices
#         indices_minus_K = list(range(0, ia)) + list(range(ib, n_tr))

#         X_minus_K = X_tr[indices_minus_K, :]
#         y_minus_K = y_tr[indices_minus_K]
#         X_k = X_tr[ia:ib, :]
#         y_k = y_tr[ia:ib]
#         V_minus_k = None
#         if V is not None:
#             V_minus_k = V[indices_minus_K, :][:, indices_minus_K]

#         get_fold_H(X_k, y_k, X_minus_K, y_minus_K, H_function, V_minus_k,
#                    return_dict, i)
#         # p = Process(target=get_fold_H,
#         #             args=(X_k, y_k, X_minus_K, y_minus_K, H_function,
#         #                   V_minus_k, return_dict, i))
#         # p.start()
#         # jobs.append(p)

#     # # start jobs. And dealing with the # of folds more than usable processors.
#     # alive_jobs = []
#     # print(len(jobs))
#     # while jobs:
#     #     # processors are all working
#     #     if len(alive_jobs) >= num_useable_processor:
#     #         print('# folds is more than # of processors ',
#     #               num_useable_processor)
#     #         for aj in alive_jobs:
#     #             if aj.is_alive():
#     #                 continue
#     #             else:
#     #                 print("job {} removed".format(aj))
#     #                 alive_jobs.remove(aj)

#     #         continue

#     #     # if some processors are not working
#     #     for job in jobs:
#     #         print(job)
#     #         if job.is_alive():
#     #             continue
#     #         job.start()
#     #         alive_jobs.append(job)
#     #         jobs.remove(job)
#     #         if len(alive_jobs) >= num_useable_processor:
#     #             break

#     # finish all jobs to continue
#     # for job in jobs:
#     #     if job.is_alive():
#     #         job.join()

#     for i, Kindices in enumerate(folds_indices):
#         ia, ib = Kindices
#         indices_minus_K = list(range(0, ia)) + list(range(ib, n_tr))
#         Hcv_k[ia:ib, indices_minus_K] = return_dict[i]

#     return Hcv_k


def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print('--- start function ' + func.__name__, '---')
        result = func(*args, **kwargs)
        print('------ {:.4f} seconds -----'.format(time.time() - start_time))
        return result

    return wrapper