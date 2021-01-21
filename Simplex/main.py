import numpy as np
from Simplex.utils import ratio_test, prod_inv, update_inv_b, update_basis, update_columns

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

# beginning of primal simplex
simplex_iter = 0
# refactor matrix A:
obj = A[obj_row_ind]  # coefficients of the objective row
A = np.delete(A, obj_row_ind, axis=0)  # pure constraint coefficients
A_bv = A[:, bv]
A_nbv = A[:, nbv]

# get basic variable coefficients
c_bv = obj[bv]
c_nbv = obj[nbv]

# price out non basic variables
reduced_cost = np.dot(c_bv.dot(inv_b), A_nbv) - c_nbv  # future remark: consider partial pricing
enter_index = np.argmin(reduced_cost)  # function: get entering column (may not be the most negative one)
while reduced_cost[enter_index] < 0:
    entering_column = A_nbv[:, enter_index]
    leaving_index = ratio_test(entering_column, rhs, inv_b)
    E = prod_inv(entering_column, inv_b, leaving_index)
    inv_b = update_inv_b(E, inv_b)
    # update A_bv and A_nbv
    A_bv, A_nbv = update_columns(A_bv, A_nbv, leaving_index, enter_index)
    # update basic variable and non basic variable
    update_basis(bv, nbv, enter_index, leaving_index)
    update_basis(c_bv, c_nbv, enter_index, leaving_index)
    reduced_cost = np.dot(c_bv.dot(inv_b), A_nbv) - c_nbv
    enter_index = np.argmin(reduced_cost)
    simplex_iter += 1

# obtaining results
variable_value = np.dot(inv_b, rhs)
obj_val = np.dot(c_bv.dot(inv_b), rhs)  # use intermediate results in reduced cost
