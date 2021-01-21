import numpy as np
from numba import njit

"""
Dakota problem
max z = 60 x1 + 30 x2 + 20 x3
st       8 x1 + 6 x2 +     x3 + s1           = 48
         4 x1 + 2 x2 +  1.5x3      + s2      = 20
         2 x1 + 1.5 x2 + 0.5 x3         + s3 = 8
"""


@njit()
def primal_simplex(A, N, basic_variable, non_basic_variable, coef_basic_variable, coef_non_basic_variable, b):
    B = np.linalg.inv(N)
    while 1:
        # pricing
        # tmp = coef_basic_variable.dot(B)
        tmp = np.dot(coef_basic_variable, B)
        price = tmp.dot(A) - coef_non_basic_variable
        # index of leaving basis
        leaving_basis, leaving_value = get_leaving_basis(price.flatten())
        if leaving_value < 0:
            # compute right hand side B^-1b
            rhs = B.dot(b)
            # compute B^-1 A
            denom = B.dot(A[:, leaving_basis])
            # perform ratio test
            ratio = rhs / denom
            # index of entering basis
            entering_basis = ratio_test(ratio)

            # updating matrix B
            tmp_col = N[:, entering_basis].copy()
            N[:, entering_basis] = A[:, leaving_basis]
            A[:, leaving_basis] = tmp_col
            B = np.linalg.inv(N)

            # updating row 0
            tmp_value = coef_non_basic_variable[leaving_basis]
            coef_non_basic_variable[leaving_basis] = coef_basic_variable[entering_basis]
            coef_basic_variable[entering_basis] = tmp_value

            # updating basis
            tmp = basic_variable[entering_basis]
            basic_variable[entering_basis] = non_basic_variable[leaving_basis]
            non_basic_variable[leaving_basis] = tmp
        else:
            break


@njit()
def get_leaving_basis(price):
    index = np.argmin(price)
    return index, price[index]


@njit()
def ratio_test(ratio):
    value = np.inf
    res = -1
    for ind in range(len(ratio)):
        if 0 <= ratio[ind] < value:
            value = ratio[ind]
            res = ind
    return res


def prod_form_of_inverse(r, k, A):
    E = np.eye()
    E[:, k] = A[:, k] / A[r, k]
    E[r, k] = 1 / A[r, k]
    return E


def cyclic_detection():
    pass
