import sys

sys.path.append('./src')
from GEMMA import gemma_operator as gemma
from bimbam import Bimbam
import os
import numpy as np

if __name__ == "__main__":
    bimbam_path = './bimbam_data/'
    bimbam_dir = os.path.join(bimbam_path,
                              'bimbam_10000_full_false_major_minor.txt')
    n_large_effects = 100
    bimbam = Bimbam(bimbam_dir)
    bimbam.pheno_simulator(num_large_effect=n_large_effects)
    bimbam_tr, bimbam_te = bimbam.train_test_split()
    bimbam_tr_x = bimbam_tr[:n_large_effects]
    bimbam_te_x = bimbam_te[:n_large_effects]

    K = bimbam.get_K()

    bimbam_full, pheno_full_tmp = \
        Bimbam.test_data_parperation(bimbam_tr_x.to_dataframe(),
                                     bimbam_te_x.to_dataframe(),
                                     bimbam_tr_x.pheno,
                                     bimbam_te_x.pheno )
    print(K.shape)
    print(bimbam_full.shape)

    # gemma.gemma_train(bimbam_full, pheno_full_tmp, related_matrix=K)

    # gemma.gemma_test(bimbam_full, pheno_full_tmp, related_matrix=K)

    pheno_te_pred = gemma.GemmaOutputReader.gemma_pred_reader('te_output')

    mse = lambda x, y: np.sum((x - y)**2) / len(x)
    print(mse(pheno_te_pred, bimbam_te.pheno))
