import pandas as pd
import numpy as np
import os


def data_spliter(data, pheno, train_idx, test_idx):
    if isinstance(train_idx, np.ndarray):
        train_idx = train_idx.tolist()
    if isinstance(test_idx, np.ndarray):
        test_idx = test_idx.tolist()

    if len(train_idx) < 0 or len(test_idx) < 0:
        raise IndexError('Specify index for training and testing set.')

    if not isinstance(test_idx[0], int):
        raise IndexError('Index has to be integer.')

    temp = data.iloc[:, 3:]
    train_data = temp.iloc[:, train_idx]
    test_data = temp.iloc[:, test_idx]

    info = data.iloc[:, :3]
    train_bim = pd.concat([info, train_data], axis=1)
    test_bim = pd.concat([info, test_data], axis=1)

    pheno_tr = pheno[train_idx]
    pheno_te = pheno[test_idx]

    print(f"Training set has {train_data.shape[1] - 3} samples")
    print(f"Test set has {test_data.shape[1] - 3} samples")

    return (train_bim, pheno_tr), (test_bim, pheno_te)


def test_data_preparation(geno_tr: pd.DataFrame, geno_te: pd.DataFrame,
                          pheno_tr: np.ndarray, len_pheno_te: int):
    info = geno_tr.iloc[:, :3]
    full_bim = pd.concat([info, geno_tr.iloc[:, 3:], geno_te.iloc[:, 3:]],
                         axis=1)
    pheno_full = np.concatenate([pheno_tr, np.array(['NA'] * len_pheno_te)])
    return full_bim, pheno_full


def data_concatenate(bim1, bim2, phyno1, phyno2):
    temp = bim2.loc[:, 3:]
    bim_full = pd.concat([bim1, temp], axis=1)
    phyno = np.concatenate(phyno1, phyno2)
    return bim_full, phyno
