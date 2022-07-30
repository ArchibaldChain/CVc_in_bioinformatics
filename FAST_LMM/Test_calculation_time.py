import time
import numpy as np


def fun0(A, U):
    print('fun0')
    return A.T / (U) @ A


def fun1(A, U):
    print('fun0')
    U_inv = np.diag(1/U)
    return A.T @ U_inv @ A


def fun2(A, U):
    n, p = A.shape
    sum = np.zeros([n, n])
    U_inv = 1/U
    for i in range(n):
        sum += (U_inv[i] * (A[i:i+1, :].T @ A[i:i+1, :]))

    return sum


def fun3(y, U):
    print('fun3')
    U_inv = np.diag(1/U)
    return y.T @ U_inv @ y


def fun4(y, U):
    print('fun4')
    return np.sum(y**2 / U)


def fun5(X, U, y):
    print('fun5')
    U_inv = np.diag(1/U)
    return X.T @ U_inv @ y.reshape([-1, 1])


def fun6(X, U, y):
    print('fun6')
    # shape(n, n)/ shape(n,) is calculated for each columns
    return X.T / U @ y.reshape([-1, 1])


np.random.seed(0)
n = 10000
dim = [n, n]
A = np.random.normal(0, 1, dim)
y = np.random.normal(0, 1, n)
U = np.random.normal(0, 1, n)
print('Start the time')

# out = fun2(A,  U)
start_time = time.time()
out = fun0(A, U)
print(out.shape)
print(out[:5, :5])
print("--- %s seconds ---" % (time.time() - start_time))
