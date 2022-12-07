import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def max_cpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    c = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            c[i, j] = max(A[i, j], B[i, j])
    return c


@njit(parallel=True)
def max_numba(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    c = np.zeros(A.shape)
    for i in prange(A.shape[0]):
        for j in prange(A.shape[1]):
            c[i, j] = max(A[i, j], B[i, j])
    return c


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    result_gpu = cuda.to_device(A)
    max_kernel[1000, 1000](result_gpu, cuda.to_device(B))
    final_result = result_gpu.copy_to_host()
    return final_result


@cuda.jit
def max_kernel(A, B):
    t_id = cuda.threadIdx.x
    b_id = cuda.blockIdx.x
    if A.shape[0] > b_id and A.shape[1] > t_id:
        cuda.atomic.max(A[b_id], t_id, B[b_id, t_id])


# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    print('     [*] CPU:', timer(max_cpu))
    print('     [*] Numba:', timer(max_numba))
    print('     [*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    max_comparison()
