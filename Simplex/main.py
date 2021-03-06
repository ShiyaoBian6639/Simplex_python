import numpy as np
from Simplex.utils import primal_simplex, random_test, load_data
from scipy.optimize import linprog

"""
linprog format: linprog(f, A, b, Aeq, beq)
"""
A = np.array([
    [8, 6, 1, 1, 0, 0],
    [4, 2, 1.5, 0, 1, 0],
    [2, 1.5, 0.5, 0, 0, 1],
    [60, 30, 20, 0, 0, 0]
])
obj_row_ind = 3  # index of objective function in A matrix
n = len(A) - 1  # number of elements in each column
inv_b = np.eye(n)  # initial inv b is identity matrix
rhs = np.array([48.0, 20.0, 8.0])  # right hand side
bv = np.array([3, 4, 5], dtype=int)  # index of basic variable
nbv = np.array([0, 1, 2], dtype=int)  # index of non basic variable

# refactor matrix A:
obj = A[obj_row_ind]  # coefficients of the objective row
A = np.delete(A, obj_row_ind, axis=0)  # pure constraint coefficients
variable_value, basis, obj_val, simplex_iter = primal_simplex(obj, A, rhs)

# obj, A, rhs = random_test(100)
# n = 100
obj, A, rhs = load_data()
# obj = np.hstack((obj, np.zeros(n)))
# A_left = A[:, :n]
# A_mid = np.zeros((n, n))
# np.fill_diagonal(A_mid, np.random.random(n) / 2)
# for i in range(n):
#     A_mid[i, i] /= 2 ** i
# A_right = A[:, 100:299]
# temp = np.hstack((A_left, A_mid))
# A = np.hstack((temp, A_right))
# %timeit
#
variable_value, basis, obj_val, simplex_iter = primal_simplex(obj, A, rhs)
# %timeit
# linprog(-obj, A_ub=A, b_ub=rhs, method="revised simplex")
