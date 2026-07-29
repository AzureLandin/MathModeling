"""
TOPSIS排序模块
"""
import numpy as np


def topsis(Y, weights):
    Z = Y * weights

    Z_plus = Z.max(axis=0)
    Z_minus = Z.min(axis=0)

    D_plus = np.sqrt(((Z - Z_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z - Z_minus) ** 2).sum(axis=1))

    C = D_minus / (D_plus + D_minus)

    return C, D_plus, D_minus
