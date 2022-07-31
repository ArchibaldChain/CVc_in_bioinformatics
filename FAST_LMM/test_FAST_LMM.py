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
delta = 5
sigma_e2 = delta * sigma_g2
print('sigma_g2: ', sigma_g2)
print('sigma_e2: ', sigma_e2)
print('beta: ', beta)
epsilon = np.random.normal(0, sigma_e2, n)
u = np.random.normal(0, sigma_g2, n_clusters)
z = W @ u

y = X @ beta + z + epsilon

######################

f = FASTLMM(True, REML=False)


f.fit(X, y, W)
f.test(delta)
neg_LL = f._neg_cover()


# print('logL(0)')
# print(f._log_likelhood_delta(0))

deltas = np.logspace(-10, 10, 21)
negative_LL_values = [neg_LL(d) for d in deltas]
# negative_LL_values = [f._log_likelhood_delta(d) for d in deltas]

print(negative_LL_values)

x_ = np.log10(deltas)
plt.plot(x_, negative_LL_values)
plt.show()

# mini = min(negative_LL_values)
# mini_index = negative_LL_values.index(mini)

x, funs = f._optimization(neg_LL)
print(x)
print(funs)
