import numpy as np
from numba import njit


@njit()
def ratio_test(inv_b, column, rhs):
    new_column = np.dot(inv_b, column)
    new_rhs = np.dot(inv_b, rhs)
    leaving = -1
    min_value = np.inf
    for i in range(len(new_column)):
        if new_column[i] > 0 and new_rhs[i] > 0:
            value = new_rhs[i] / new_column[i]
            if min_value > value:
                min_value = value
                leaving = i
    return leaving, new_column


@njit()
def prod_inv(num, n, r):
    """
    :param num: new column
    :param n: number of elements in num
    :param r: pivot
    :return: E (to be multiplied with B^{-1})
    """
    res = np.eye(n)
    temp = num[r]
    ero = - num / temp
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

