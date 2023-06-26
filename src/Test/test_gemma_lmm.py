import sys

sys.path.append('./src')
from GEMMA import gemma_operator as gemma
from bimbam import Bimbam
import os

if __name__ == "__main__":
    bimbam_path = './bimbam_data/'
    bimbam_dir = os.path.join(bimbam_path,
                              'bimbam_10000_full_false_major_minor.txt')
    bimbam = Bimbam(bimbam_dir)
    bimbam.pheno_simulator(100)
    bimbam_x = bimbam[:100]
    K = bimbam.get_K()

    gemma.gemma_lmm_train(bimbam_x.to_dataframe(),
                          bimbam_x.pheno,
                          related_matrix=K,
                          prefix="train_lmm_output")
