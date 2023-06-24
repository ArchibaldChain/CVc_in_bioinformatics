import numpy as np
import sys

sys.path.append('./src')
from bslmm_simulation import gemma_operator as gemma
from FAST_LMM.FaST_LMM import FASTLMM
from frequentist_simulation.simulation import load
from sklearn.model_selection import train_test_split


def phenotype_simulator(SNPs,
                        num_large_effect=10,
                        large_effect=400,
                        small_effect=1):
    ## get the Beta
    np.random.seed(1)
    d = num_large_effect
    beta_large = np.random.normal(0, np.sqrt(large_effect), d)

    n, p = SNPs.shape
    small_effect = 2
    beta_small = np.random.normal(0, np.sqrt(small_effect), p - d)

    beta = np.concatenate([beta_large, beta_small])

    ## calculate residual and y
    temp = SNPs @ beta
    sigma_g2 = np.var(temp, ddof=1)  # get the overall variance

    h2 = 0.5  # heritability

    sigma_e2 = sigma_g2 * (1 - h2) / h2
    beta0 = np.random.normal(0, np.sqrt(large_effect))
    beta_ = np.insert(beta, 0, beta0)

    residual = np.random.normal(0, np.sqrt(sigma_e2), n)
    y = SNPs @ beta + residual + beta0

    ## standardnize y
    mu_y = np.mean(y)
    sd_y = np.std(y, ddof=1)
    y = (y - mu_y) / sd_y
    return y


def test_var_estimator(num_fixed_snps=0):
    ## train test split
    # add 1 to the first row
    SNPs = load("data/SNP_in_200GENE_chr1.csv")
    y = phenotype_simulator(SNPs)
    n = SNPs.shape[0]
    data = np.concatenate([np.ones([n, 1]), SNPs], axis=1)
    G_tr, G_te, y_tr, y_te = train_test_split(data, y, test_size=0.2)
    d = num_fixed_snps
    X_tr, X_te = G_tr[:, :d + 1], G_te[:, :d + 1]
    W_tr, W_te = G_tr[:, 1:], G_te[:, 1:]
    n_tr, sc = W_tr.shape

    ## using FaST-LMM
    fast = FASTLMM(lowRank=True, REML=True)
    fast.fit(X_tr, y_tr, 1 / np.sqrt(sc) * W_tr)

    ##  Using GEMMA to estimate the variance component
    K_relatedness = 1 / (sc) * W_tr @ W_tr.T
    sigmas = gemma.gemma_var_estimator(y_tr, K_relatedness, 'var_component')
    print(sigmas)
    print(fast.sigma_g2, fast.sigma_e2)


if __name__ == '__main__':
    test_var_estimator()