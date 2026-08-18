# -*- coding: utf-8 -*-
"""
问题二数据加载与预处理：读取附件 1-5，构造约束参数与 2026 需求。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from problem2_utils import (
    ATTACH_DIR, N_REGIONS, WD_WEIGHT, WE_WEIGHT, WEEK_LEN,
    C_F, C_S, P_F, P_S, S_F, S_S, COVERAGE_MIN, GROWTH_RATE,
    find_attachment, HOUR_LABELS,
)


def load_attachment1() -> pd.DataFrame:
    """附件1：区域基础数据。"""
    path = find_attachment("附件 1")
    df = pd.read_excel(path, header=0)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    # 取前10行
    df = df.head(N_REGIONS).copy()
    return df


def _read_hourly_matrix(path: pd.ExcelFile | str, sheet: str) -> np.ndarray:
    """读取 10×24 分时矩阵。"""
    raw = pd.read_excel(path, sheet_name=sheet, header=0)
    raw = raw.dropna(subset=[raw.columns[0]]).copy()
    raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()
    raw.iloc[:, 0] = raw.iloc[:, 0].astype(int)
    raw = raw.sort_values(raw.columns[0]).reset_index(drop=True)
    raw = raw.head(N_REGIONS).copy()
    block = raw.iloc[:, 1:25]
    return block.to_numpy(dtype=float)


def load_attachment2() -> tuple[np.ndarray, np.ndarray]:
    """附件2：工作日/周末分时充电车次。"""
    path = find_attachment("附件2")
    q_wd = _read_hourly_matrix(path, "工作日分时段充电车次数据")
    q_we = _read_hourly_matrix(path, "周末充电车次数据")
    return q_wd, q_we


def load_attachment3() -> tuple[np.ndarray, np.ndarray]:
    """附件3：工作日/周末分时充电量 (kWh)。"""
    path = find_attachment("附件3")
    e_wd = _read_hourly_matrix(path, "工作日分时段充电负荷数据")
    e_we = _read_hourly_matrix(path, "周末充电负荷数据（修改后）")
    return e_wd, e_we


def load_attachment4() -> np.ndarray:
    """附件4：分时电网最大允许负荷 (kW)。"""
    path = find_attachment("附件4")
    raw = pd.read_excel(path, sheet_name="Sheet1", header=0)
    raw = raw.dropna(subset=[raw.columns[0]]).copy()
    raw = raw.head(N_REGIONS).copy()
    block = raw.iloc[:, 1:25]
    return block.to_numpy(dtype=float)


def build_problem_data() -> dict:
    """
    构造问题二所需的全部参数。
    返回字典包含：
      - A_i: 区域总面积
      - A0_i: 既有覆盖面积
      - F0_i: 现有快充桩数
      - S0_i: 现有慢充桩数
      - N0_i: 现有总桩数
      - G_i: 区域电网总容量 (kW)
      - theta_i: 现有快充占比
      - a_i: 单桩有效覆盖面积
      - n100_i: 100%覆盖所需新增总桩数上界
      - n_lb_i: 90%覆盖所需新增总桩数下界
      - Q_plan_i: 2026年规划充电车次需求 (车次/日)
      - E_star_i_t: 综合典型日分时充电量 (kWh)
      - L_plan_i_t: 2026年分时充电功率 (kW)
      - L_max_i_t: 分时电网最大允许负荷 (kW)
    """
    # 附件1
    df1 = load_attachment1()
    A_i = df1["区域总面积(km²)"].to_numpy(dtype=float)
    A0_i = df1["充电覆盖面积面积(km²)"].to_numpy(dtype=float)
    F0_i = df1["其中快充桩（个）"].to_numpy(dtype=float)
    S0_i = df1["其中慢充桩（个）"].to_numpy(dtype=float)
    N0_i = F0_i + S0_i
    g_i = df1["区域电网总容量（万千瓦）"].to_numpy(dtype=float)
    G_i = g_i * 1e4  # 转为 kW

    # 附件2：分时充电车次
    q_wd, q_we = load_attachment2()
    Q_wd = q_wd.sum(axis=1)  # 工作日日总车次
    Q_we = q_we.sum(axis=1)  # 周末日总车次
    Q_2025 = (WD_WEIGHT * Q_wd + WE_WEIGHT * Q_we) / WEEK_LEN
    Q_plan = Q_2025 * (1 + GROWTH_RATE)

    # 附件3：分时充电量
    e_wd, e_we = load_attachment3()
    E_star = (WD_WEIGHT * e_wd + WE_WEIGHT * e_we) / WEEK_LEN  # 综合典型日
    L_plan = E_star * (1 + GROWTH_RATE)  # 2026年分时功率 (kW, Δt=1h)

    # 附件4：电网最大允许负荷
    L_max = load_attachment4()

    # 覆盖效率校准
    a_i = A0_i / N0_i  # 单桩有效覆盖面积

    # 编码上界
    n100_i = np.ceil((A_i - A0_i) / a_i).astype(int)

    # 90%覆盖下界
    n_lb_i = np.maximum(0, np.ceil((COVERAGE_MIN * A_i - A0_i) / a_i)).astype(int)

    # 快充占比
    theta_i = F0_i / N0_i

    return {
        "region": np.arange(1, N_REGIONS + 1),
        "A_i": A_i,
        "A0_i": A0_i,
        "F0_i": F0_i,
        "S0_i": S0_i,
        "N0_i": N0_i,
        "G_i": G_i,
        "g_i": g_i,
        "theta_i": theta_i,
        "a_i": a_i,
        "n100_i": n100_i,
        "n_lb_i": n_lb_i,
        "Q_plan": Q_plan,
        "Q_2025": Q_2025,
        "Q_wd": Q_wd,
        "Q_we": Q_we,
        "E_star": E_star,
        "e_wd": e_wd,
        "e_we": e_we,
        "L_plan": L_plan,
        "L_max": L_max,
    }


def grid_safety_check(data: dict) -> pd.DataFrame:
    """
    电网逐时安全预检查（表2）。
    检查 L_plan_i_t <= L_max_i_t 对所有 i,t。
    """
    L_plan = data["L_plan"]  # (10, 24)
    L_max = data["L_max"]    # (10, 24)
    ratio = L_plan / L_max
    rows = []
    for i in range(N_REGIONS):
        max_ratio = float(np.max(ratio[i]))
        min_margin = float(np.min(L_max[i] - L_plan[i]))
        violated = max_ratio > 1.0
        rows.append({
            "区域": i + 1,
            "最大负荷率": max_ratio,
            "最小安全裕度_kW": min_margin,
            "是否越限": "是" if violated else "否",
        })
    return pd.DataFrame(rows)


def build_input_table(data: dict) -> pd.DataFrame:
    """表1：输入与约束预处理表。"""
    rows = []
    for i in range(N_REGIONS):
        rows.append({
            "区域": i + 1,
            "总面积_km2": data["A_i"][i],
            "既有覆盖面积_km2": data["A0_i"][i],
            "单桩覆盖效率_a_i": data["a_i"][i],
            "90pct覆盖所需新增下界": int(data["n_lb_i"][i]),
            "100pct覆盖上界_n100": int(data["n100_i"][i]),
            "2026预测车次_Q_plan": data["Q_plan"][i],
            "既有服务能力_车次日": S_F * data["F0_i"][i] + S_S * data["S0_i"][i],
            "快充占比_theta": data["theta_i"][i],
        })
    return pd.DataFrame(rows)
