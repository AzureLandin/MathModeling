"""
TOPSIS排序模块：逼近理想解排序法
"""
import numpy as np


def topsis(Y, weights):
    """
    TOPSIS综合排序
    
    参数:
        Y:       n×m 标准化矩阵
        weights: 长度m的权重向量
    
    返回:
        C:       贴近度向量
        D_plus:  到正理想解的距离
        D_minus: 到负理想解的距离
    """
    # 加权标准化矩阵
    Z = Y * weights

    # 正理想解和负理想解
    Z_plus = Z.max(axis=0)
    Z_minus = Z.min(axis=0)

    # 欧氏距离
    D_plus = np.sqrt(((Z - Z_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z - Z_minus) ** 2).sum(axis=1))

    # 相对贴近度
    C = D_minus / (D_plus + D_minus)

    return C, D_plus, D_minus
