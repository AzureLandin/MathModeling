"""
熵权法模块：基于信息熵的客观赋权
"""
import numpy as np


def entropy_weight(Y):
    """
    熵权法计算指标权重
    
    参数:
        Y: n×m 标准化矩阵（已含epsilon平移）
    
    返回:
        weights: 长度m的权重向量
        entropy: 长度m的信息熵向量
    """
    n, m = Y.shape

    # 计算特征比重
    col_sums = Y.sum(axis=0)
    P = Y / col_sums

    # 计算信息熵
    entropy = np.zeros(m)
    for k in range(m):
        mask = P[:, k] > 0
        entropy[k] = -np.sum(P[mask, k] * np.log(P[mask, k])) / np.log(n)

    # 计算权重
    diff = 1 - entropy  # 差异系数
    weights = diff / diff.sum()

    return weights, entropy
