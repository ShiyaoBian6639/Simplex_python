import numpy as np
from PrimalSimplex import primal_simplex
from readMPS import read_mps

"""
max z = 60 x1 +  30 x2 +  20 x3
s.t.     8 x1 +   6 x2 +     x3 + s1        = 48
         4 x1 +   2 x2 + 1.5 x3     + s2    = 20
         2 x1 + 1.5 x2 + 0.5 x3       + s3  = 8   

Cbv * inv_v * aj - cj: coefficient of xj in row 0

"""
problem_name, row_name_idx_dict, row_name_sense_dict, objective_name, num_row, A, col_name_idx_dict, variable_type, variable_bound = read_mps(
    "D:/benchmark/markshare_4_0.mps")

A = np.array([[8, 6, 1],
              [4, 2, 1.5],
              [2, 1.5, 0.5],
              [60, 30, 20]
              ], dtype=float)
B = np.eye(3)

basic_variable = np.array([3.0, 4, 5], dtype=float)
non_basic_variable = np.array([0, 1, 2], dtype=float)

coef_non_basic_variable = np.array([60.0, 30, 20], dtype=float)
coef_basic_variable = np.array([0, 0, 0.0], dtype=float)

b = np.array([48, 20, 8.0], dtype=float).transpose()
primal_simplex(A, B, basic_variable, non_basic_variable, coef_basic_variable, coef_non_basic_variable, b)
