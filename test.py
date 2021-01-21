from LUfactorization import naive_lu, par_lu
import numpy as np
from scipy.linalg import lu

B = np.array([
    [2, 0, 4, 0, -2],
    [3, 1, 0, 1, 0],
    [-1, 0, -1, 0, -2],
    [0, -1, 0, 0, -6],
    [0, 0, 1, 0, 4]
])
L = np.zeros(B.shape)
U = np.zeros(B.shape)
U = B.copy()
naive_lu(U, L)
print(U)
U = B.copy()
par_lu(U, L)
sciP, sciL, sciU = lu(B)


def bench_loop(file):
    obj = open(file)
    count = 0
    for i in obj:
        count += 1
    return count


def bench_next(file):
    obj = open(file)
    count = 0
    while obj:
        next(obj)
        count += 1
    return count


test_file = "D:/benchmark/neos-3402454-bohle.mps"
# %timeit bench_loop(test_file)
bench_next(test_file)

