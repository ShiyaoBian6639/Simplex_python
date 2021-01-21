"""
variable name convention:
bv: basic variable
nvb: non basic variable
A: column matrix
prev_column: columns that are in A (without being pre multiplied by inv_b)
inv_b = inverse of A_bv (this should always hold)
"""

import numpy as np
from numba import njit


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


@njit()
def ratio_test(prev_column, prev_rhs, inv_b):
    """
    1. rhs should be pre-multiplied with inv_b
    2. both rhs and new_column are required to be positive
    :param prev_column:
    :param prev_rhs:
    :param inv_b:
    :return:
    """
    new_rhs = np.dot(inv_b, prev_rhs)
    leaving = -1
    min_value = np.inf
    for i in range(len(prev_column)):
        if prev_column[i] > 0 and new_rhs[i] > 0:
            value = new_rhs[i] / prev_column[i]
            if min_value > value:
                min_value = value
                leaving = i
    return leaving


@njit()
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
    ero[r] = 1 / temp
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
