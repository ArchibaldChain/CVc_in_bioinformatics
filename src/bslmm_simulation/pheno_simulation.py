import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split as split
from bimbam import data_spliter


def phenotype_generator(bimbam_dir):
    bimbam_data = pd.read_csv(bimbam_dir, sep = '\t',index_col=None)

    # Shufftle the data
    bimbam_data = bimbam_data.sample(len(bimbam_data), replace=False)

    # Get Beta
    large_effect = 400
    d = 10
    beta_large = np.random.normal(0, np.sqrt(large_effect), d)
    n = len(bimbam_data)
    small_effect = 1
    beta_small = np.random.normal(0, np.sqrt(small_effect), n-d)
    beta = np.concatenate([beta_large, beta_small])


    # calculate the sigma_e2
    SNP = bimbam_data.drop(['major', 'minor', 'POS'], axis=1, inplace=False).T
    SNP.head()
    temp = SNP.values @ beta
    sigma_g2 = np.var(temp, ddof = 1) # get the overall variance
    h2 = 0.5 # heritability
    sigma_e2 = sigma_g2 * (1 - h2)/ h2
    print("The sigma_g^2 is {:.5}, and sigma_e^2 is {:.5}".format(sigma_g2, sigma_e2))

    # simulate beta_0 and calculate y
    beta0 = np.random.normal(0, np.sqrt(large_effect))
    residual = np.random.normal(0, np.sqrt(sigma_e2), len(SNP))
    Y = SNP.values @ beta + residual + beta0
    mean_Y = np.mean(Y)
    sd_Y = np.std(Y)
    pheno = (Y-mean_Y)/sd_Y
    return bimbam_data, pheno

def train_test_split(bimbam_data, pheno):
    # Train test split
    sample_size = bimbam_data.shape[1] - 3
    train_idx, test_idx = split(np.arange(sample_size), test_size=0.2)
    return data_spliter(bimbam_data ,pheno,train_idx.tolist(),test_idx.tolist())

if __name__ == "__main__":
    bimbam_path = './bimbam_data/'
    bimbam_dir = os.path.join(bimbam_path,'bimbam_10000_full_false_major_minor.txt')
    bimbam_data, pheno = phenotype_generator(bimbam_dir)
    (bimbam_train, pheno_tr), (bimbam_test, pheno_te) = train_test_split(bimbam_data, pheno)
    print(bimbam_train.shape, bimbam_test.shape, pheno_tr.shape, pheno_te.shape)