import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    y = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for t in range(X.shape[1]):
                y[j][i] += X[j][t] * X[i][t]
    return y


@njit
def matmul_transpose_numba(X):
    y = np.zeros((X.shape[0], X.shape[0]))
    for i in prange(X.shape[0]):
        for j in prange(X.shape[0]):
            for t in prange(X.shape[1]):
                y[j][i] += X[j][t] * X[i][t]
    return y


def matmul_transpose_gpu(X):
    result = np.zeros((X.shape[0], X.shape[0]))
    result_gpu = cuda.to_device(result)
    matmul_kernel[1, 1024](cuda.to_device(X), result_gpu)
    final_result = result_gpu.copy_to_host()
    return final_result


@cuda.jit
def matmul_kernel(A, C):
    t_id = cuda.threadIdx.x
    b_id = cuda.blockIdx.x
    for x in range(t_id * (C.shape[0] ^ 2) // 1024, (t_id + 1) * (C.shape[0] ^ 2) // 1024):
        for t in range(A.shape[1]):
            C[i][j] += A[j][t] * A[i][t]


# this is the comparison function - keep it as it is, don't change X or Y.
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
