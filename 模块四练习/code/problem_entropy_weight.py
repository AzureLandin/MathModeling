"""
熵权法模块
"""
import numpy as np


def entropy_weight(Y):
    n, m = Y.shape

    col_sums = Y.sum(axis=0)
    P = Y / col_sums

    entropy = np.zeros(m)
    for k in range(m):
        mask = P[:, k] > 0
        entropy[k] = -np.sum(P[mask, k] * np.log(P[mask, k])) / np.log(n)

    diff = 1 - entropy
    weights = diff / diff.sum()

    return weights, entropy
