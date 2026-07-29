"""
CRITIC法模块
"""
import numpy as np


def critic_weight(Y):
    sigma = np.std(Y, axis=0, ddof=1)
    R = np.corrcoef(Y, rowvar=False)
    gamma = np.sum(1 - R, axis=0)
    info = sigma * gamma
    weights = info / info.sum()

    return weights, sigma, R, gamma
