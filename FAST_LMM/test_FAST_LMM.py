import numpy as np
import pandas as pd
from scipy.sparse.csgraph import structural_rank
import matplotlib.pyplot as plt

from FAST_LMM import FASTLMM
# import FAST.FAST_LMM

n, p = 100, 5
np.random.seed(5)
X = np.random.normal(0, 1, size=[n, p])
n_clusters = 10
W = np.zeros([n, n_clusters])
for i in range(n_clusters):
    cluster_size = n // n_clusters

    W[(i * cluster_size):((i+1) * cluster_size-1), i] = 1

beta = np.random.normal(0, 20, p)
sigma_g2 = 20
delta = 0.5
sigma_e2 = delta * sigma_g2
print('sigma_g2: ', sigma_g2)
print('sigma_e2: ', sigma_e2)
print('beta: ', beta)

epsilon = np.random.normal(0, np.sqrt(sigma_e2), n)
u = np.random.normal(0, np.sqrt(sigma_g2), n_clusters)
z = W @ u

y = X @ beta + z + epsilon

data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
np.savetxt("testData.csv", data, delimiter=",")
#####################################

f = FASTLMM(False, REML=False)


f.fit(X, y, W)
# f.testing_sigmag2(delta)


neg_LL = f._neg_cover()


deltas = np.logspace(-10, 10, 21)
negative_LL_values = [neg_LL(d) for d in deltas]


x_ = np.log10(deltas)
plt.plot(x_, negative_LL_values)
plt.show()


x, funs = f._optimization(neg_LL)
print(x)
print(funs)

f.test(x)
