# -*- coding: utf-8 -*-
"""
问题二熵权-TOPSIS 及稳健性检查。
"""

from __future__ import annotations

import numpy as np


def _normalize_matrix(X: np.ndarray, cost_cols: list[int], benefit_cols: list[int]) -> np.ndarray:
    """
    无量纲化：成本型用 min/x，效益型用 x/max。
    返回归一化矩阵（非负，每列至少有一个正元素）。
    """
    n, m = X.shape
    Xn = np.zeros_like(X, dtype=float)
    eps = 1e-12

    for j in range(m):
        col = X[:, j]
        if j in cost_cols:
            cmin = np.min(col)
            if cmin < eps:
                cmin = eps
            Xn[:, j] = cmin / np.maximum(col, eps)
        elif j in benefit_cols:
            cmax = np.max(col)
            if cmax < eps:
                cmax = eps
            Xn[:, j] = col / cmax
        else:
            # 默认按效益型处理
            cmax = np.max(col)
            if cmax < eps:
                cmax = eps
            Xn[:, j] = col / cmax

    return Xn


def entropy_weight(Xn: np.ndarray) -> np.ndarray:
    """
    熵权法计算权重。
    Xn: 归一化后的非负矩阵 (K, m)，K 为方案数，m 为指标数。
    """
    K, m = Xn.shape
    eps = 1e-12

    # 列求和
    col_sum = np.sum(Xn, axis=0)
    col_sum = np.maximum(col_sum, eps)

    # 概率矩阵
    P = Xn / col_sum  # (K, m)

    # 信息熵
    lnP = np.log(np.maximum(P, eps))
    e = -np.sum(P * lnP, axis=0) / np.log(K)  # (m,)

    # 差异系数
    d = 1.0 - e
    d = np.maximum(d, eps)

    # 归一化权重
    w = d / np.sum(d)
    return w


def topsis(
    X: np.ndarray,
    cost_cols: list[int],
    benefit_cols: list[int],
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    熵权-TOPSIS。
    X: 原始指标矩阵 (K, m)，每行一个方案。
    cost_cols: 成本型指标列索引（越小越好）。
    benefit_cols: 效益型指标列索引（越大越好）。
    weights: 外部权重（None 则用熵权）。
    返回：(贴近度 Gamma, 权重 w)
    """
    Xn = _normalize_matrix(X, cost_cols, benefit_cols)

    if weights is None:
        w = entropy_weight(Xn)
    else:
        w = weights.copy()
        w = w / np.sum(w)

    # 加权规范化
    V = Xn * w  # (K, m)

    # 正负理想解
    v_plus = np.max(V, axis=0)
    v_minus = np.min(V, axis=0)

    # 距离
    S_plus = np.sqrt(np.sum((V - v_plus) ** 2, axis=1))
    S_minus = np.sqrt(np.sum((V - v_minus) ** 2, axis=1))

    # 贴近度
    Gamma = S_minus / np.maximum(S_plus + S_minus, 1e-12)

    return Gamma, w


def select_best_solution(
    solutions: list[dict],
    model: str,
    weights: np.ndarray | None = None,
) -> tuple[int, np.ndarray, float]:
    """
    从 Pareto 候选集中选择最优方案。
    返回：(最优方案索引, 权重, 贴近度)
    """
    if not solutions:
        return -1, np.array([]), 0.0

    K = len(solutions)
    # 构造指标矩阵: [f1, C_city, f3]
    X = np.zeros((K, 3))
    for k, s in enumerate(solutions):
        X[k, 0] = s["f1"]
        X[k, 1] = s["C_city"]
        X[k, 2] = s["f3"]

    # 指标类型：f1=成本型, C_city=效益型, f3=成本型
    cost_cols = [0, 2]
    benefit_cols = [1]

    Gamma, w = topsis(X, cost_cols, benefit_cols, weights)
    best_idx = int(np.argmax(Gamma))
    return best_idx, w, float(Gamma[best_idx])


def robustness_check(solutions: list[dict], model: str) -> dict:
    """
    稳健性检查：三种权重方案下推荐解是否一致。
    返回字典包含 entropy, equal, cost_bias 的推荐索引和贴近度。
    """
    if len(solutions) <= 1:
        return {
            "entropy_idx": 0, "entropy_gamma": 1.0,
            "equal_idx": 0, "equal_gamma": 1.0,
            "cost_bias_idx": 0, "cost_bias_gamma": 1.0,
            "consistent": True,
            "weights_entropy": np.array([1/3, 1/3, 1/3]),
        }

    K = len(solutions)
    X = np.zeros((K, 3))
    for k, s in enumerate(solutions):
        X[k, 0] = s["f1"]
        X[k, 1] = s["C_city"]
        X[k, 2] = s["f3"]

    cost_cols = [0, 2]
    benefit_cols = [1]

    # 熵权
    Gamma_e, w_e = topsis(X, cost_cols, benefit_cols, None)
    idx_e = int(np.argmax(Gamma_e))

    # 等权
    w_eq = np.array([1/3, 1/3, 1/3])
    Gamma_eq, _ = topsis(X, cost_cols, benefit_cols, w_eq)
    idx_eq = int(np.argmax(Gamma_eq))

    # 偏成本
    w_cb = np.array([0.50, 0.25, 0.25])
    Gamma_cb, _ = topsis(X, cost_cols, benefit_cols, w_cb)
    idx_cb = int(np.argmax(Gamma_cb))

    consistent = (idx_e == idx_eq == idx_cb) or (
        abs(solutions[idx_e]["f1"] - solutions[idx_eq]["f1"]) / max(solutions[idx_e]["f1"], 1) < 0.05
        and abs(solutions[idx_e]["f1"] - solutions[idx_cb]["f1"]) / max(solutions[idx_e]["f1"], 1) < 0.05
        and abs(solutions[idx_e]["C_city"] - solutions[idx_eq]["C_city"]) < 0.01
        and abs(solutions[idx_e]["C_city"] - solutions[idx_cb]["C_city"]) < 0.01
    )

    return {
        "entropy_idx": idx_e,
        "entropy_gamma": float(Gamma_e[idx_e]),
        "equal_idx": idx_eq,
        "equal_gamma": float(Gamma_eq[idx_eq]),
        "cost_bias_idx": idx_cb,
        "cost_bias_gamma": float(Gamma_cb[idx_cb]),
        "consistent": consistent,
        "weights_entropy": w_e,
    }
