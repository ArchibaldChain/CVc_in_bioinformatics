import numpy as np
import pandas as pd
from scipy.sparse.csgraph import structural_rank
import matplotlib.pyplot as plt

from FAST_LMM import FASTLMM
# import FAST.FAST_LMM

n, p = 100, 5
np.random.seed(5)
X = np.random.normal(0, 1, size=[n, p])

y = np.random.normal(0, 5, n)
f = FASTLMM(sparse=False, REML=False)


f.fit(X, y)
f.test(0.5)
neg_LL = f._neg_cover()


deltas = np.logspace(-10, 10, 21)
negative_LL_values = [neg_LL(d) for d in deltas]
print(negative_LL_values)

# mini = min(negative_LL_values)
# mini_index = negative_LL_values.index(mini)

x_ = np.log10(deltas)
plt.plot(x_, negative_LL_values)
plt.show()

# x, funs = f._optimization(neg_LL)
# print(x)
# print(funs)
