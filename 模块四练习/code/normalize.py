"""
数据标准化模块：Min-Max正向化处理
"""
import numpy as np


def minmax_normalize(matrix, is_benefit, epsilon=0.0001):
    """
    Min-Max标准化（含正向化），并平移epsilon防止log(0)
    
    参数:
        matrix:      n×m 原始指标矩阵
        is_benefit:  长度m的布尔数组，True=极大型，False=极小型
        epsilon:     平移量，防止后续对数运算出现0
    
    返回:
        normalized:  n×m 标准化后的矩阵
    """
    n, m = matrix.shape
    normalized = np.zeros_like(matrix, dtype=float)

    for k in range(m):
        col = matrix[:, k]
        mn, mx = col.min(), col.max()

        if mx == mn:
            normalized[:, k] = 0.5
            continue

        if is_benefit[k]:
            # 极大型正向化
            normalized[:, k] = (col - mn) / (mx - mn)
        else:
            # 极小型正向化
            normalized[:, k] = (mx - col) / (mx - mn)

    # 平移epsilon防止log(0)
    normalized = normalized + epsilon

    return normalized
