from re import X
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import pandas as pd

print(os.listdir())

data = np.loadtxt('testData.csv', delimiter=',')
print(data.shape)
y = data[:, -1]
X = data[:, :-1]
data = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
# data = pd.DataFrame(data, columns=['x1',  'y'])
print(data.head())

groupData = np.array(range(1, 11))
groupData = np.repeat(groupData, 100)

# md = smf.mixedlm("y ~ x1  -1", data, groups=groupData)
md = smf.mixedlm("y ~ x1 + x2 + x3 + x4 + x5 -1", data, groups=groupData)
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())
