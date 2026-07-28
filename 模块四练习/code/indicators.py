"""
指标计算模块：计算供应商四维评价指标
"""
import numpy as np


def compute_indicators(order_data, supply_data):
    """
    计算四维评价指标
    
    参数:
        order_data: 订货量矩阵 (n×240)
        supply_data: 供货量矩阵 (n×240)
    
    返回:
        F1: 周均供货量（极大型）
        F2: 周均供货缺失量（极小型）
        F3: 供货达标率（极大型）
        F4: 供货偏差均方（极小型）
        N:  有效周数
    """
    n = order_data.shape[0]
    F1 = np.zeros(n)
    F2 = np.zeros(n)
    F3 = np.zeros(n)
    F4 = np.zeros(n)
    N  = np.zeros(n)

    for i in range(n):
        # 有效订货周集合：O_ij > 0
        valid = order_data[i, :] > 0
        N[i] = valid.sum()

        if N[i] == 0:
            continue

        O = order_data[i, valid]
        S = supply_data[i, valid]

        # F1: 周均供货量
        F1[i] = S.mean()

        # F2: 周均供货缺失量（仅统计缺货方向）
        F2[i] = np.maximum(0, O - S).mean()

        # F3: 供货达标率（S >= O 的比例）
        F3[i] = (S >= O).mean()

        # F4: 供货偏差均方
        F4[i] = ((O - S) ** 2).mean()

    return F1, F2, F3, F4, N


def build_indicator_matrix(F1, F2, F3, F4):
    """
    构建指标矩阵并返回效益类型标记
    
    返回:
        matrix: n×4 指标矩阵
        is_benefit: [True, False, True, False] 各指标效益类型
    """
    matrix = np.column_stack([F1, F2, F3, F4])
    is_benefit = [True, False, True, False]
    return matrix, is_benefit
