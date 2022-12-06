import numpy as np
from numba import njit, cuda
import timeit


def matmul_transpose_trivial(X):
    y = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            temp = 0
            for t in range(X.shape[1]):
                temp += X[j][t] * X[i][t]
            y[i][j] = temp
    return y





@njit
def matmul_transpose_numba(X):
    y = np.zeros(X.shape)
    for i in prange(X.shape[0]):
        for j in prange(X.shape[0]):
            temp = 0
            for t in prange(X.shape[1]):
                temp += X[j][t] * X[i][t]
            y[i][j] = temp
    return y


def matmul_transpose_gpu(X):
    raise NotImplementedError("To be implemented")

@cuda.jit
def matmul_kernel(A, C):
    raise NotImplementedError("To be implemented")

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))

if __name__ == '__main__':
    matmul_comparison()
