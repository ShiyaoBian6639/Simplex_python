"""
variable name convention:
bv: basic variable
nvb: non basic variable
A: column matrix
prev_column: columns that are in A (without being pre multiplied by inv_b)
inv_b = inverse of A_bv (this should always hold)
degeneracy happens when rhs[j] == 0
"""

import numpy as np
from numba import njit
from simplex_param import TOLERANCE


@njit()
def update_columns(A_bv, A_nbv, leaving_index, enter_index):
    """
    swap columns in A_bv and A_nbv: A_nbv[:, enter_index] enters basis and A_bv[:, leaving_index] leaves
    :param A_bv:
    :param A_nbv:
    :param leaving_index:
    :param enter_index:
    :return:
    """
    for i in range(len(A_nbv)):
        temp = A_nbv[i, enter_index]
        A_nbv[i, enter_index] = A_bv[i, leaving_index]
        A_bv[i, leaving_index] = temp
    return A_bv, A_nbv


def ratio_test(prev_column, prev_rhs, inv_b):
    """
    1. rhs should be pre-multiplied with inv_b
    2. both rhs and new_column are required to be positive
    :param prev_column: no ratio is computed on negative values!
    :param prev_rhs: in primal simplex rhs is not allowed to be negative, otherwise we loose basic feasible solution
    :param inv_b:
    :return:
    """
    n = len(prev_column)
    new_rhs = np.dot(inv_b, prev_rhs)
    if np.any(new_rhs < - TOLERANCE):
        print("negative rhs")
    leaving = -1
    leaving_backup = -1
    col_max = 0
    min_value = np.inf
    for i in range(n):
        if prev_column[i] > TOLERANCE:
            if prev_column[i] > col_max:
                col_max = prev_column[i]
                leaving_backup = i
            if new_rhs[i] > TOLERANCE:
                value = abs(new_rhs[i] / prev_column[i])
                if min_value > value:
                    min_value = value
                    leaving = i

    print(f"min value: {min_value}")
    if leaving == -1:
        return leaving_backup
    else:
        return leaving


def prod_inv(prev_column, inv_b, r):
    """
    :param prev_column: column in matrix A
    :param inv_b: invert of B
    :param r: pivot
    :return: E (to be multiplied with B^{-1})
    """
    column = np.dot(inv_b, prev_column)
    n = len(column)
    res = np.eye(n)
    temp = column[r]
    ero = - column / temp
    print(f"product of inverse divides {temp}")
    ero[r] = 1.0 / temp
    res[:, r] = ero
    return res


@njit()
def update_inv_b(mat_e, inv_b):
    return mat_e.dot(inv_b)


@njit()
def update_basis(bv, nbv, enter_index, leaving_index):
    temp = nbv[enter_index]
    nbv[enter_index] = bv[leaving_index]
    bv[leaving_index] = temp


@njit()
def get_initial_basis(A):
    n, m = A.shape
    nbv = np.arange(m - n)
    bv = np.arange(m - n, m)
    inv_b = np.eye(n)
    return bv, nbv, inv_b


def primal_simplex(obj, A, rhs):
    # beginning of primal simplex
    simplex_iter = 0
    bv, nbv, inv_b = get_initial_basis(A)
    A_bv = A[:, bv]
    A_nbv = A[:, nbv]

    # get basic variable coefficients
    c_bv = obj[bv]
    c_nbv = obj[nbv]
    # price out non basic variables
    reduced_cost = np.dot(c_bv.dot(inv_b), A_nbv) - c_nbv  # future remark: consider partial pricing
    enter_index = np.argmin(reduced_cost)  # function: get entering column (may not be the most negative one)
    while reduced_cost[enter_index] < -TOLERANCE:
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
        print(f"iteration: {simplex_iter}")
        print(f"reduced cost: {reduced_cost[enter_index]} ")

    # obtaining results
    variable_value = np.dot(inv_b, rhs)
    obj_val = np.dot(c_bv.dot(inv_b), rhs)  # use intermediate results in reduced cost
    return variable_value, bv, obj_val, simplex_iter


def random_test(n):
    A_left = np.random.random((n, n))
    A_eye = np.eye(n)
    A = np.hstack((A_left, A_eye))
    obj_left = np.random.random(n)
    obj_right = np.zeros(n)
    obj = np.hstack((obj_left, obj_right))
    rhs = np.random.random(n)
    # store generated values
    np.savetxt("./obj.txt", obj)
    np.savetxt("./A.txt", A)
    np.savetxt("./rhs.txt", rhs)
    return obj, A, rhs


def load_data():
    obj = np.loadtxt("../obj.txt")
    A = np.loadtxt("../A.txt")
    rhs = np.loadtxt("../rhs.txt")
    return obj, A, rhs
