from numba import njit, prange


@njit()
def naive_lu(B, L):
    nrow, ncol = B.shape
    for k in range(nrow):
        up_left = B[k, k]
        for i in range(k + 1, nrow):
            value = B[i, k]
            if value:
                L[i, k] = B[i, k]
                ratio = value / up_left
                for j in range(k, ncol):
                    B[i, j] = B[i, j] - ratio * B[k, j]
    construct_lower(L, B, nrow, ncol)


@njit(parallel=True)
def construct_lower(L, U, nrow, ncol):
    for i in prange(ncol):
        for j in range(i, nrow):
            L[j, i] /= U[i, i]
            L[i, i] = 1


@njit()
def par_lu(B, L):
    nrow, ncol = B.shape
    for k in range(nrow):
        up_left = B[k, k]
        row_operation(B, L, k, up_left, nrow, ncol)


@njit(parallel=True)
def row_operation(B, L, k, up_left, nrow, ncol):
    for i in prange(k + 1, nrow):
        value = B[i, k]
        if value:
            L[i, k] = B[i, k]
            ratio = value / up_left
            for j in range(k, ncol):
                B[i, j] = B[i, j] - ratio * B[k, j]
