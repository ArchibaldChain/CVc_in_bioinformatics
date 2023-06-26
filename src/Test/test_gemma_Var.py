import numpy as np
import sys

sys.path.append('./src')
from GEMMA import gemma_operator as gemma
from FAST_LMM.FaST_LMM import FASTLMM
from frequentist_simulation.simulation import load
from sklearn.model_selection import train_test_split
from bimbam import Bimbam
import os


def test_var_estimator(num_fixed_snps=0):
    bimbam_path = './bimbam_data/'
    bimbam_dir = os.path.join(bimbam_path,
                              'bimbam_10000_full_false_major_minor.txt')
    bimbam = Bimbam(bimbam_dir)
    bimbam.pheno_simulator(num_fixed_snps)

    bimbam_new = bimbam[:num_fixed_snps]
    ## using FaST-LMM
    fast = FASTLMM(lowRank=True, REML=True)
    fast.fit(bimbam_new.get_X(), bimbam.pheno,
             1 / np.sqrt(bimbam.p) * bimbam.get_X())
    sigmas0 = (fast.sigma_g2, fast.sigma_e2)

    ## Using GEMMA LMM
    K_relatedness = bimbam.get_K()
    gemma.gemma_lmm_train(bimbam_new.to_dataframe(), bimbam.pheno,
                          "lmm_tr_output", K_relatedness)

    ##  Using GEMMA to estimate the variance component
    sigmas1 = gemma.gemma_var_estimator(bimbam.pheno, K_relatedness,
                                        'var_component')

    # 2 variance components
    k0 = bimbam_new.get_K()
    sigmas2 = gemma.gemma_multi_var_estimator(k0, [k0, K_relatedness],
                                              '2_var_component')

    sigmas3 = gemma.gemma_lmm_train(bimbam_new.to_dataframe(),
                                    bimbam_new.pheno,
                                    related_matrix=K_relatedness,
                                    prefix="train_lmm_output")
    # GEMMA LMM
    print(sigmas0, sigmas1, sigmas2, sigmas3)

    # return sigmas, sigmas2, (fast.sigma_g2, fast.sigma_e2)


if __name__ == '__main__':
    test_var_estimator(200)
