import numpy as np
import pandas as pd
from scipy.sparse.csgraph import structural_rank

from FAST_LMM import FASTLMM 
from scipy.sparse import issparse
from numpy.linalg import matrix_rank
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

if not 'SNP' in locals():
    SNP = pd.read_csv('Simulated_SNPs.csv', index_col=0)
y = SNP['Y']
SNP.drop('Y', inplace=True, axis=1)
X = SNP.values


f = FASTLMM(sparse= False, REML = True)
f.fit(X, y)
# f.test(10000000)


neg_LL = f._neg_cover()
deltas = np.logspace(-10, 10, 21)
negative_LL_values = [neg_LL(d) for d in deltas]
x_ = np.log10(deltas)
plt.plot(x_, negative_LL_values)
plt.show()

# x, funs = f._optimization2(neg_LL)
# print(x)
# print(funs)

# neg_LL = f._neg_cover()
