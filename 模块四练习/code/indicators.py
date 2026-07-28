"""
指标计算模块
"""
import numpy as np


def compute_indicators(order_data, supply_data):
    n = order_data.shape[0]
    F1 = np.zeros(n)
    F2 = np.zeros(n)
    F3 = np.zeros(n)
    F4 = np.zeros(n)
    N  = np.zeros(n)

    for i in range(n):
        valid = order_data[i, :] > 0
        N[i] = valid.sum()

        if N[i] == 0:
            continue

        O = order_data[i, valid]
        S = supply_data[i, valid]

        F1[i] = S.mean()
        F2[i] = np.maximum(0, O - S).mean()
        F3[i] = (S >= O).mean()
        F4[i] = ((O - S) ** 2).mean()

    return F1, F2, F3, F4, N


def build_indicator_matrix(F1, F2, F3, F4):
    matrix = np.column_stack([F1, F2, F3, F4])
    is_benefit = [True, False, True, False]
    return matrix, is_benefit
