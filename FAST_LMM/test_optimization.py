import numpy as np
from FAST_LMM import FASTLMM
import matplotlib.pyplot as plt

f = lambda x: 3* x**2 - 50 * x + 10
x = np.arange(-10,20,0.01)
plt.plot(x,f(x))
plt.show()

flmm = FASTLMM()


x, f = flmm._optimization(f)
print(x)
print(f)