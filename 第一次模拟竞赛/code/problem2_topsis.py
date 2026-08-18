# -*- coding: utf-8 -*-
"""
熵权 TOPSIS：在去重后的 Pareto 解集上选取折中方案。

F1、F3 为成本型，F2 为效益型。极差为 0 的指标熵权置零后对其余权重再归一化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def minmax_benefit(col: np.ndarray, cost_type: bool) -> np.ndarray:
    lo, hi = float(np.min(col)), float(np.max(col))
    if hi - lo <= EPS:
        return np.full_like(col, 0.0, dtype=float)
    if cost_type:
        return (hi - col) / (hi - lo)
    return (col - lo) / (hi - lo)


def entropy_weights(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入已同向化的效益型矩阵 Z (m, n)。
    返回 (w, e, d)。
    """
    m, n = Z.shape
    if m <= 1:
        w = np.ones(n) / n
        return w, np.zeros(n), np.zeros(n)
    P = (Z + EPS)
    P = P / P.sum(axis=0, keepdims=True)
    e = -(P * np.log(P)).sum(axis=0) / np.log(m)
    d = 1.0 - e
    if float(d.sum()) <= EPS:
        w = np.ones(n) / n
    else:
        w = d / d.sum()
    return w, e, d


def topsis_rank(F1: np.ndarray, F2: np.ndarray, F3: np.ndarray) -> pd.DataFrame:
    """
    对 Pareto 目标做熵权 TOPSIS。
    返回含标准化值、权重、距离和贴近度的表，按 C_k 降序。
    """
    F1 = np.asarray(F1, dtype=float)
    F2 = np.asarray(F2, dtype=float)
    F3 = np.asarray(F3, dtype=float)
    m = len(F1)
    if m == 0:
        raise ValueError("Pareto 解集为空，不能进行熵权 TOPSIS")

    A = np.column_stack([F1, F2, F3])
    cost_flags = [True, False, True]
    Z = np.column_stack([minmax_benefit(A[:, j], cost_flags[j]) for j in range(3)])

    # 无区分度的列置零后再归一化权重
    ranges = A.max(axis=0) - A.min(axis=0)
    w, e, d = entropy_weights(Z)
    w = np.where(ranges <= EPS, 0.0, w)
    if float(w.sum()) <= EPS:
        w = np.ones(3) / 3.0
    else:
        w = w / w.sum()

    V = Z * w
    v_pos = V.max(axis=0)
    v_neg = V.min(axis=0)
    d_pos = np.sqrt(((V - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - v_neg) ** 2).sum(axis=1))
    C = d_neg / (d_pos + d_neg + EPS)

    df = pd.DataFrame({
        "方案编号": np.arange(1, m + 1),
        "F1_投资成本": F1,
        "F2_需求加权覆盖率": F2,
        "F3_电网目标": F3,
        "标准化_F1": Z[:, 0],
        "标准化_F2": Z[:, 1],
        "标准化_F3": Z[:, 2],
        "加权_F1": V[:, 0],
        "加权_F2": V[:, 1],
        "加权_F3": V[:, 2],
        "D_plus": d_pos,
        "D_minus": d_neg,
        "贴近度_C": C,
    })
    df["熵权_w1"] = w[0]
    df["熵权_w2"] = w[1]
    df["熵权_w3"] = w[2]
    df["信息熵_e1"] = e[0]
    df["信息熵_e2"] = e[1]
    df["信息熵_e3"] = e[2]
    df = df.sort_values("贴近度_C", ascending=False).reset_index(drop=True)
    df["TOPSIS排名"] = np.arange(1, m + 1)
    df.attrs["weights"] = w
    df.attrs["entropy"] = e
    return df


def equal_weight_topsis(F1: np.ndarray, F2: np.ndarray, F3: np.ndarray) -> pd.DataFrame:
    """等权 TOPSIS 对照，用于熵权极端化检查。"""
    F1 = np.asarray(F1, dtype=float)
    F2 = np.asarray(F2, dtype=float)
    F3 = np.asarray(F3, dtype=float)
    A = np.column_stack([F1, F2, F3])
    cost_flags = [True, False, True]
    Z = np.column_stack([minmax_benefit(A[:, j], cost_flags[j]) for j in range(3)])
    w = np.ones(3) / 3.0
    V = Z * w
    v_pos, v_neg = V.max(axis=0), V.min(axis=0)
    d_pos = np.sqrt(((V - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - v_neg) ** 2).sum(axis=1))
    C = d_neg / (d_pos + d_neg + EPS)
    df = pd.DataFrame({
        "方案编号": np.arange(1, len(F1) + 1),
        "等权贴近度": C,
    })
    df = df.sort_values("等权贴近度", ascending=False).reset_index(drop=True)
    df["等权排名"] = np.arange(1, len(F1) + 1)
    return df


def pick_representatives(front_df: pd.DataFrame, topsis_df: pd.DataFrame) -> dict[str, int]:
    """
    返回 front_df 中的行号（0-based）：
      TOPSIS 推荐、最低成本、最高覆盖、最低电网压力。
    """
    rec_id = int(topsis_df.iloc[0]["方案编号"]) - 1
    cost_id = int(front_df["F1_投资成本"].idxmin())
    cover_id = int(front_df["F2_需求加权覆盖率"].idxmax())
    grid_id = int(front_df["F3_电网目标"].idxmin())
    return {
        "topsis": rec_id,
        "min_cost": cost_id,
        "max_cover": cover_id,
        "min_grid": grid_id,
    }
