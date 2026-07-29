"""
数据标准化模块
"""
import numpy as np


def minmax_normalize(matrix, is_benefit, epsilon=0.0001):
    n, m = matrix.shape
    normalized = np.zeros_like(matrix, dtype=float)

    for k in range(m):
        col = matrix[:, k]
        mn, mx = col.min(), col.max()

        if mx == mn:
            normalized[:, k] = 0.5
            continue

        if is_benefit[k]:
            normalized[:, k] = (col - mn) / (mx - mn)
        else:
            normalized[:, k] = (mx - col) / (mx - mn)

    normalized = normalized + epsilon

    return normalized
