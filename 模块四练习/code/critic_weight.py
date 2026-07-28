"""
CRITIC法模块：基于对比强度和冲突性的客观赋权
"""
import numpy as np


def critic_weight(Y):
    """
    CRITIC法计算指标权重
    
    参数:
        Y: n×m 标准化矩阵（Min-Max标准化后，含epsilon平移）
    
    返回:
        weights:  长度m的权重向量
        sigma:    对比强度（标准差）
        R:        相关系数矩阵
        gamma:    冲突性向量
    """
    # 对比强度：标准差
    sigma = np.std(Y, axis=0, ddof=1)

    # 指标冲突性：Pearson相关系数矩阵
    R = np.corrcoef(Y, rowvar=False)
    gamma = np.sum(1 - R, axis=0)

    # 信息承载量
    info = sigma * gamma

    # 归一化得权重
    weights = info / info.sum()

    return weights, sigma, R, gamma
