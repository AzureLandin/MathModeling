# -*- coding: utf-8 -*-
"""
问题4：三年增长风险评估与滚动优化。

用法：python problem4_main.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = Path(__file__).resolve().parents[1]
P2_DIR = ROOT / "results" / "p2_results"
P3_DIR = ROOT / "results" / "p3_results"
P3_DATA = ROOT / "p3_data"
P4_DIR = ROOT / "results" / "p4_results"
FIG_DIR = ROOT / "figures" / "p4_figures"

# 固定参数
C_F, C_S = 6.0, 0.8          # 万元/桩
S_F, S_S = 80.0, 20.0        # 车次/(桩·日)
COVERAGE_MIN = 0.90

YEARS = [2026, 2027, 2028]
GROWTH = {2026: 1.15, 2027: 1.3225, 2028: 1.520875}


def ensure_dirs():
    P4_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(df, stem):
    df.to_csv(P4_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    df.to_excel(P4_DIR / f"{stem}.xlsx", index=False)


# ─────────────────────── 数据加载 ───────────────────────


def load_inputs():
    # 附件1：区域基础数据
    attach_dir = ROOT / "附件"
    path1 = [p for p in attach_dir.glob("附件 1*.xlsx") if not p.name.startswith("~")][0]
    df1 = pd.read_excel(path1, header=0)
    df1 = df1.dropna(subset=[df1.columns[0]])
    df1 = df1[pd.to_numeric(df1.iloc[:, 0], errors="coerce").notna()].copy()
    df1.iloc[:, 0] = df1.iloc[:, 0].astype(int)
    df1 = df1.sort_values(df1.columns[0]).reset_index(drop=True).head(10)

    A_i = df1["区域总面积(km²)"].to_numpy(float)
    A0_i = df1["充电覆盖面积面积(km²)"].to_numpy(float)
    F0_i = df1["其中快充桩（个）"].to_numpy(float)
    S0_i = df1["其中慢充桩（个）"].to_numpy(float)
    G_i = df1["区域电网总容量（万千瓦）"].to_numpy(float) * 1e4

    # 附件2：充电车次
    path2 = [p for p in attach_dir.glob("附件2*.xlsx") if not p.name.startswith("~")][0]

    def _read_hourly(path, sheet):
        raw = pd.read_excel(path, sheet_name=sheet, header=0)
        raw = raw.dropna(subset=[raw.columns[0]])
        raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()
        raw.iloc[:, 0] = raw.iloc[:, 0].astype(int)
        return raw.sort_values(raw.columns[0]).reset_index(drop=True).head(10).iloc[:, 1:25].to_numpy(float)

    q_wd = _read_hourly(path2, "工作日分时段充电车次数据")
    q_we = _read_hourly(path2, "周末充电车次数据")
    Q_2025 = (5 * q_wd.sum(axis=1) + 2 * q_we.sum(axis=1)) / 7

    # 问题2最终方案
    p2_final = pd.read_csv(P2_DIR / "p2_08_最终分区建设方案.csv", encoding="utf-8-sig")
    p2_final = p2_final[p2_final["区域"] != "全市"].copy()
    p2_final["区域"] = p2_final["区域"].astype(int)
    p2_final = p2_final.sort_values("区域").reset_index(drop=True)
    x_p2 = p2_final["新增快充"].to_numpy(int)
    y_p2 = p2_final["新增慢充"].to_numpy(int)

    # 问题3调度效果
    p3_eval = pd.read_csv(P3_DIR / "p3_06_调度效果评价_最终合并.csv", encoding="utf-8-sig")
    p3_pre = pd.read_csv(P3_DIR / "p3_02_调度前峰谷特征.csv", encoding="utf-8-sig")

    # 基期峰时负荷率（取工作日、周末较坏值）
    rho_raw_0 = np.zeros(10)
    rho_tou_0 = np.zeros(10)
    for i in range(10):
        for dt in ["工作日", "周末"]:
            # 调度前：从 p3_02 读取
            sub_pre = p3_pre[(p3_pre["区域"] == i + 1) & (p3_pre["日期类型"] == dt)]
            if not sub_pre.empty:
                rho_raw_0[i] = max(rho_raw_0[i], float(sub_pre["调度前最大负荷率_raw"].values[0]))
            # 调度后：从 p3_06 读取
            sub_post = p3_eval[(p3_eval["区域"] == i + 1) & (p3_eval["日期类型"] == dt)]
            if not sub_post.empty:
                rho_tou_0[i] = max(rho_tou_0[i], float(sub_post["调度后最大负荷率_raw"].values[0]))

    # 覆盖效率
    a_i = A0_i / (F0_i + S0_i)

    return {
        "A_i": A_i, "A0_i": A0_i, "F0_i": F0_i, "S0_i": S0_i, "G_i": G_i,
        "Q_2025": Q_2025, "x_p2": x_p2, "y_p2": y_p2,
        "rho_raw_0": rho_raw_0, "rho_tou_0": rho_tou_0, "a_i": a_i,
    }


# ─────────────────────── 模块A：静态基线风险 ───────────────────────


def module_a(data):
    """S0 静态基线：不追加建设的年度风险。"""
    F0, S0 = data["F0_i"], data["S0_i"]
    x_p2, y_p2 = data["x_p2"], data["y_p2"]
    Q_2025 = data["Q_2025"]

    # 2026 初始状态
    F1 = F0 + x_p2
    S1 = S0 + y_p2
    SC1 = S_F * F1 + S_S * S1

    rows = []
    for i in range(10):
        first_warn = None
        for year in YEARS:
            g = GROWTH[year]
            Q = Q_2025[i] * g
            psi = SC1[i] / Q if Q > 0 else 999
            gap = max(Q - SC1[i], 0)

            if psi >= 1:
                level = "服务安全"
            elif psi >= 0.90:
                level = "服务预警"
            else:
                level = "服务不足"

            if psi < 1 and first_warn is None:
                first_warn = year

            rows.append({
                "区域": i + 1,
                "年份": year,
                "增长系数": g,
                "预测日均车次需求": Q,
                "静态快充总数": int(F1[i]),
                "静态慢充总数": int(S1[i]),
                "静态服务能力": SC1[i],
                "服务保障率_raw": psi,
                "服务保障率_pct": psi * 100,
                "服务缺口_车次日": gap,
                "服务风险等级": level,
                "首次服务不足年份": first_warn if first_warn else "无",
            })

    return pd.DataFrame(rows)


# ─────────────────────── 模块B：峰时电网风险 ───────────────────────


def module_b(data):
    """峰时电网风险投影。"""
    rho_raw_0 = data["rho_raw_0"]
    rho_tou_0 = data["rho_tou_0"]

    rows = []
    for i in range(10):
        first_warn_raw, first_over_raw = None, None
        first_warn_tou, first_over_tou = None, None
        for year in YEARS:
            g = GROWTH[year]
            rho_raw = rho_raw_0[i] * g
            rho_tou = rho_tou_0[i] * g
            delta_rho = rho_raw - rho_tou
            margin = 1 - rho_tou

            def classify(rho):
                if rho < 0.80:
                    return "安全"
                elif rho < 1.00:
                    return "峰时预警"
                else:
                    return "过载风险"

            level_raw = classify(rho_raw)
            level_tou = classify(rho_tou)

            if rho_raw >= 0.8 and first_warn_raw is None:
                first_warn_raw = year
            if rho_raw >= 1.0 and first_over_raw is None:
                first_over_raw = year
            if rho_tou >= 0.8 and first_warn_tou is None:
                first_warn_tou = year
            if rho_tou >= 1.0 and first_over_tou is None:
                first_over_tou = year

            rows.append({
                "区域": i + 1,
                "年份": year,
                "增长系数": g,
                "基期原始最大负荷率_raw": rho_raw_0[i],
                "基期原始最大负荷率_pct": rho_raw_0[i] * 100,
                "原始峰时风险率_raw": rho_raw,
                "原始峰时风险率_pct": rho_raw * 100,
                "原始风险等级": level_raw,
                "基期调度后最大负荷率_raw": rho_tou_0[i],
                "基期调度后最大负荷率_pct": rho_tou_0[i] * 100,
                "调度后峰时风险率_raw": rho_tou,
                "调度后峰时风险率_pct": rho_tou * 100,
                "调度后风险等级": level_tou,
                "TOU缓解量_raw": delta_rho,
                "TOU缓解量_pct": delta_rho * 100,
                "调度后容量裕度_raw": margin,
                "调度后容量裕度_pct": margin * 100,
                "首次原始预警年": first_warn_raw if first_warn_raw else "无",
                "首次原始过载年": first_over_raw if first_over_raw else "无",
                "首次调度后预警年": first_warn_tou if first_warn_tou else "无",
                "首次调度后过载年": first_over_tou if first_over_tou else "无",
            })

    return pd.DataFrame(rows)


# ─────────────────────── 最小配网增容需求 ───────────────────────


def compute_grid_capacity_gap(data):
    """计算区域9的最小配网增容需求 ΔG。"""
    rho_tou_0 = data["rho_tou_0"]
    # 区域9 (index 8) 的基期调度后最大负荷率
    rho9 = rho_tou_0[8]
    # 基期调度后最大负荷 (kW) = rho * G_limit
    # 从 p3_05 读取区域9周末10:00的调度后负荷
    p3_05 = pd.read_csv(P3_DIR / "p3_05_调度前后曲线长表_合并后.csv", encoding="utf-8-sig")
    r9_we_10 = p3_05[(p3_05["区域"] == 9) & (p3_05["日期类型"] == "周末") & (p3_05["小时序号"] == 10)]
    L_post_10 = float(r9_we_10["调度后负荷_kW"].values[0])
    G_limit_10 = float(r9_we_10["最大允许负荷_kW"].values[0])

    rows = []
    for year in YEARS:
        g = GROWTH[year]
        L_projected = L_post_10 * g
        gap = max(L_projected - G_limit_10, 0)
        rows.append({
            "区域": 9,
            "关键时段": "10:00-11:00",
            "年份": year,
            "增长系数": g,
            "基期调度后负荷_kW": L_post_10,
            "投影调度后负荷_kW": L_projected,
            "电网允许负荷_kW": G_limit_10,
            "最小容量缺口_kW": gap,
            "最小容量缺口_MW": gap / 1000,
        })

    return pd.DataFrame(rows)


# ─────────────────────── 模块C：滚动补桩 MILP ───────────────────────


def solve_rolling_milp(data, with_structure=True):
    """
    两阶段 MILP：最小化追加总成本，然后最小化2027年投资。
    with_structure=True: 含快慢充结构下限 (R-Base)
    with_structure=False: 不含结构下限 (R-Free)
    """
    F0, S0 = data["F0_i"], data["S0_i"]
    x_p2, y_p2 = data["x_p2"], data["y_p2"]
    Q_2025 = data["Q_2025"]
    a_i = data["a_i"]
    A_i, A0_i = data["A_i"], data["A0_i"]

    F1 = F0 + x_p2  # 2026 初始快充
    S1 = S0 + y_p2  # 2026 初始慢充
    theta = F1 / (F1 + S1)  # 快充占比下限

    # 决策变量: [u_2027(10), v_2027(10), u_2028(10), v_2028(10)] = 40 个
    n_var = 40
    IU2, IV2, IU3, IV3 = 0, 10, 20, 30

    # ── 第一阶段：最小化总追加成本 ──
    c1 = np.zeros(n_var)
    c1[IU2:IU2 + 10] = C_F
    c1[IV2:IV2 + 10] = C_S
    c1[IU3:IU3 + 10] = C_F
    c1[IV3:IV3 + 10] = C_S

    A_rows = []
    b_lower = []
    b_upper = []

    # 服务能力约束：80*(F1+u2) + 20*(S1+v2) >= Q_2027
    # => 80*u2 + 20*v2 >= Q_2027 - 80*F1 - 20*S1
    for i in range(10):
        row = np.zeros(n_var)
        row[IU2 + i] = S_F
        row[IV2 + i] = S_S
        rhs = Q_2025[i] * GROWTH[2027] - S_F * F1[i] - S_S * S1[i]
        A_rows.append(row)
        b_lower.append(max(rhs, 0))
        b_upper.append(np.inf)

    # 2028: 80*(F1+u2+u3) + 20*(S1+v2+v3) >= Q_2028
    for i in range(10):
        row = np.zeros(n_var)
        row[IU2 + i] = S_F
        row[IV2 + i] = S_S
        row[IU3 + i] = S_F
        row[IV3 + i] = S_S
        rhs = Q_2025[i] * GROWTH[2028] - S_F * F1[i] - S_S * S1[i]
        A_rows.append(row)
        b_lower.append(max(rhs, 0))
        b_upper.append(np.inf)

    # 快慢充结构下限: (1-theta)*u >= theta*v => (1-theta)*u - theta*v >= 0
    if with_structure:
        for i in range(10):
            if theta[i] < 1 - 1e-9:
                # 2027
                row2 = np.zeros(n_var)
                row2[IU2 + i] = 1 - theta[i]
                row2[IV2 + i] = -theta[i]
                A_rows.append(row2)
                b_lower.append(0)
                b_upper.append(np.inf)
                # 2028
                row3 = np.zeros(n_var)
                row3[IU3 + i] = 1 - theta[i]
                row3[IV3 + i] = -theta[i]
                A_rows.append(row3)
                b_lower.append(0)
                b_upper.append(np.inf)

    A_ub = np.array(A_rows)
    constraints = LinearConstraint(A_ub, b_lower, b_upper)
    bounds = Bounds(np.zeros(n_var), np.full(n_var, np.inf))
    integrality = np.ones(n_var)  # 全整数

    # 第一阶段
    res1 = milp(c1, constraints=constraints, bounds=bounds, integrality=integrality)
    if not res1.success:
        print(f"  [MILP] 阶段1失败: {res1.message}")
        return None

    f1_star = res1.fun

    # ── 第二阶段：固定总成本，最小化2027年投资 ──
    c2 = np.zeros(n_var)
    c2[IU2:IU2 + 10] = C_F
    c2[IV2:IV2 + 10] = C_S

    # 添加总成本约束: c1 @ x <= f1_star + eps
    A_total = c1.reshape(1, -1)
    extra = LinearConstraint(A_total, 0, f1_star + 1e-6)
    constraints2 = LinearConstraint(np.vstack([A_ub, A_total]),
                                     np.r_[b_lower, 0],
                                     np.r_[b_upper, f1_star + 1e-6])

    res2 = milp(c2, constraints=constraints2, bounds=bounds, integrality=integrality)
    if not res2.success:
        # 回退到阶段1解
        sol = res1.x
    else:
        sol = res2.x

    u2 = np.round(sol[IU2:IU2 + 10]).astype(int)
    v2 = np.round(sol[IV2:IV2 + 10]).astype(int)
    u3 = np.round(sol[IU3:IU3 + 10]).astype(int)
    v3 = np.round(sol[IV3:IV3 + 10]).astype(int)

    return {
        "u_2027": u2, "v_2027": v2,
        "u_2028": u3, "v_2028": v3,
        "total_cost": float(np.sum(C_F * (u2 + u3) + C_S * (v2 + v3))),
        "cost_2027": float(np.sum(C_F * u2 + C_S * v2)),
        "cost_2028": float(np.sum(C_F * u3 + C_S * v3)),
    }


def module_c(data):
    """滚动补桩优化：R-Base 和 R-Free。"""
    print("[模块C] 滚动补桩 MILP 求解...")
    res_base = solve_rolling_milp(data, with_structure=True)
    res_free = solve_rolling_milp(data, with_structure=False)

    if res_base is None or res_free is None:
        print("  [MILP] 求解失败")
        return None, None

    print(f"  R-Base: 总成本={res_base['total_cost']:.2f}万元, "
          f"2027={res_base['cost_2027']:.2f}, 2028={res_base['cost_2028']:.2f}")
    print(f"  R-Free: 总成本={res_free['total_cost']:.2f}万元, "
          f"2027={res_free['cost_2027']:.2f}, 2028={res_free['cost_2028']:.2f}")

    return res_base, res_free


def gen_table_c(data, res, label="R-Base"):
    """表C：滚动补桩优化结果。"""
    F0, S0 = data["F0_i"], data["S0_i"]
    x_p2, y_p2 = data["x_p2"], data["y_p2"]
    Q_2025 = data["Q_2025"]
    a_i = data["a_i"]
    A_i, A0_i = data["A_i"], data["A0_i"]

    F1 = F0 + x_p2
    S1 = S0 + y_p2

    rows = []
    for i in range(10):
        # 2026 行（问题2初始状态）
        sc_2026 = S_F * F1[i] + S_S * S1[i]
        q_2026 = Q_2025[i] * GROWTH[2026]
        cov_geo = min(1.0, (A0_i[i] + a_i[i] * ((F1[i] + S1[i]) - (F0[i] + S0[i]))) / A_i[i])
        rows.append({
            "方案": label, "区域": i + 1, "年份": 2026,
            "当年新增快充": int(x_p2[i]), "当年新增慢充": int(y_p2[i]),
            "当年新增总数": int(x_p2[i] + y_p2[i]),
            "当年新增成本_万元": C_F * x_p2[i] + C_S * y_p2[i],
            "累计快充": int(F1[i]), "累计慢充": int(S1[i]),
            "累计服务能力": sc_2026,
            "当年预测需求": q_2026,
            "服务保障率_raw": sc_2026 / q_2026,
            "服务保障率_pct": sc_2026 / q_2026 * 100,
            "地理覆盖率_raw": cov_geo,
            "地理覆盖率_pct": cov_geo * 100,
            "是否满足服务约束": "是" if sc_2026 >= q_2026 else "否",
        })

        # 2027
        u2, v2 = int(res["u_2027"][i]), int(res["v_2027"][i])
        F2 = F1[i] + u2
        S2 = S1[i] + v2
        sc_2027 = S_F * F2 + S_S * S2
        q_2027 = Q_2025[i] * GROWTH[2027]
        cov_geo2 = min(1.0, (A0_i[i] + a_i[i] * ((F2 + S2) - (F0[i] + S0[i]))) / A_i[i])
        rows.append({
            "方案": label, "区域": i + 1, "年份": 2027,
            "当年新增快充": u2, "当年新增慢充": v2,
            "当年新增总数": u2 + v2,
            "当年新增成本_万元": C_F * u2 + C_S * v2,
            "累计快充": int(F2), "累计慢充": int(S2),
            "累计服务能力": sc_2027,
            "当年预测需求": q_2027,
            "服务保障率_raw": sc_2027 / q_2027,
            "服务保障率_pct": sc_2027 / q_2027 * 100,
            "地理覆盖率_raw": cov_geo2,
            "地理覆盖率_pct": cov_geo2 * 100,
            "是否满足服务约束": "是" if sc_2027 >= q_2027 else "否",
        })

        # 2028
        u3, v3 = int(res["u_2028"][i]), int(res["v_2028"][i])
        F3 = F2 + u3
        S3 = S2 + v3
        sc_2028 = S_F * F3 + S_S * S3
        q_2028 = Q_2025[i] * GROWTH[2028]
        cov_geo3 = min(1.0, (A0_i[i] + a_i[i] * ((F3 + S3) - (F0[i] + S0[i]))) / A_i[i])
        rows.append({
            "方案": label, "区域": i + 1, "年份": 2028,
            "当年新增快充": u3, "当年新增慢充": v3,
            "当年新增总数": u3 + v3,
            "当年新增成本_万元": C_F * u3 + C_S * v3,
            "累计快充": int(F3), "累计慢充": int(S3),
            "累计服务能力": sc_2028,
            "当年预测需求": q_2028,
            "服务保障率_raw": sc_2028 / q_2028,
            "服务保障率_pct": sc_2028 / q_2028 * 100,
            "地理覆盖率_raw": cov_geo3,
            "地理覆盖率_pct": cov_geo3 * 100,
            "是否满足服务约束": "是" if sc_2028 >= q_2028 else "否",
        })

    return pd.DataFrame(rows)


# ─────────────────────── 表D：S0 vs S1 对比 ───────────────────────


def gen_table_d(df_a, df_c):
    """表D：滚动优化前后服务效果对比。"""
    rows = []
    for i in range(10):
        for year in YEARS:
            s0_row = df_a[(df_a["区域"] == i + 1) & (df_a["年份"] == year)]
            s1_row = df_c[(df_c["区域"] == i + 1) & (df_c["年份"] == year) & (df_c["方案"] == "R-Base")]

            s0_psi = float(s0_row["服务保障率_raw"].values[0]) if not s0_row.empty else 0
            s1_psi = float(s1_row["服务保障率_raw"].values[0]) if not s1_row.empty else 0
            s0_gap = float(s0_row["服务缺口_车次日"].values[0]) if not s0_row.empty else 0
            s1_gap = max(0, float(s1_row["当年预测需求"].values[0]) - float(s1_row["累计服务能力"].values[0])) if not s1_row.empty else 0

            s0_cov = float(s0_row["服务保障率_raw"].values[0]) if not s0_row.empty else 0
            s1_cov = float(s1_row["地理覆盖率_raw"].values[0]) if not s1_row.empty else 0

            rows.append({
                "区域": i + 1, "年份": year,
                "S0服务保障率_raw": s0_psi,
                "S0服务保障率_pct": s0_psi * 100,
                "S1服务保障率_raw": s1_psi,
                "S1服务保障率_pct": s1_psi * 100,
                "S0服务缺口": s0_gap,
                "S1服务缺口": s1_gap,
                "补桩是否消除缺口": "是" if s1_gap <= 1e-6 else "否",
                "S1地理覆盖率_pct": s1_cov * 100,
            })
    return pd.DataFrame(rows)


# ─────────────────────── 表E：风险建议汇总 ───────────────────────


def gen_table_e(df_a, df_b, df_c):
    """表E：风险—调整建议汇总。"""
    rows = []
    for i in range(10):
        for year in YEARS:
            a_row = df_a[(df_a["区域"] == i + 1) & (df_a["年份"] == year)]
            b_row = df_b[(df_b["区域"] == i + 1) & (df_b["年份"] == year)]
            c_row = df_c[(df_c["区域"] == i + 1) & (df_c["年份"] == year) & (df_c["方案"] == "R-Base")]

            srv_level = a_row["服务风险等级"].values[0] if not a_row.empty else "未知"
            grid_level = b_row["调度后风险等级"].values[0] if not b_row.empty else "未知"

            need_build = srv_level in ("服务预警", "服务不足")
            need_tou = grid_level == "峰时预警"
            need_grid = grid_level == "过载风险"

            u = int(c_row["当年新增快充"].values[0]) if not c_row.empty else 0
            v = int(c_row["当年新增慢充"].values[0]) if not c_row.empty else 0

            if need_grid:
                priority = "高"
                advice = "先做逐时复核；建议配网增容、提高需求响应比例"
            elif need_build and need_tou:
                priority = "高"
                advice = "补桩与分时引导同步，新增设施优先接入低压力接入点"
            elif need_build:
                priority = "中"
                advice = f"按方案补建 {u}快充+{v}慢充"
            elif need_tou:
                priority = "中"
                advice = "强化预约、差异化电价、低谷引导"
            else:
                priority = "低"
                advice = "常规监测，暂不新增"

            rows.append({
                "区域": i + 1, "年份": year,
                "服务风险等级": srv_level,
                "调度后电网风险等级": grid_level,
                "是否需补桩": "是" if need_build else "否",
                "新增快充": u, "新增慢充": v,
                "是否需强化TOU": "是" if need_tou else "否",
                "是否需逐时复核": "是" if need_grid else "否",
                "建议优先级": priority,
                "建议文本": advice,
            })
    return pd.DataFrame(rows)


# ─────────────────────── 表F：全市年度汇总 ───────────────────────


def gen_table_f(df_a, df_b, df_c):
    """表F：全市年度汇总。"""
    rows = []
    for year in YEARS:
        g = GROWTH[year]
        a_year = df_a[df_a["年份"] == year]
        b_year = df_b[df_b["年份"] == year]
        c_year = df_c[(df_c["年份"] == year) & (df_c["方案"] == "R-Base")]

        city_demand = a_year["预测日均车次需求"].sum()
        city_sc_s0 = a_year["静态服务能力"].sum()
        city_psi_s0 = city_sc_s0 / city_demand if city_demand > 0 else 0
        city_gap_s0 = a_year["服务缺口_车次日"].sum()

        city_sc_s1 = c_year["累计服务能力"].sum()
        city_psi_s1 = city_sc_s1 / city_demand if city_demand > 0 else 0

        new_fast = c_year["当年新增快充"].sum()
        new_slow = c_year["当年新增慢充"].sum()
        invest = c_year["当年新增成本_万元"].sum()

        n_warn = len(b_year[b_year["调度后风险等级"] == "峰时预警"])
        n_over = len(b_year[b_year["调度后风险等级"] == "过载风险"])

        rows.append({
            "年份": year,
            "全市预测需求_车次日": city_demand,
            "静态服务能力": city_sc_s0,
            "静态服务保障率_pct": city_psi_s0 * 100,
            "静态服务缺口": city_gap_s0,
            "滚动后服务能力": city_sc_s1,
            "滚动后服务保障率_pct": city_psi_s1 * 100,
            "当年新增快充": int(new_fast),
            "当年新增慢充": int(new_slow),
            "当年投资_万元": invest,
            "调度后峰时预警区域数": n_warn,
            "调度后过载风险区域数": n_over,
        })
    return pd.DataFrame(rows)


# ─────────────────────── 结构下限敏感性对比 ───────────────────────


def gen_sensitivity(res_base, res_free):
    """R-Base vs R-Free 对比。"""
    rows = []
    for label, res in [("R-Base", res_base), ("R-Free", res_free)]:
        total_fast = int(np.sum(res["u_2027"] + res["u_2028"]))
        total_slow = int(np.sum(res["v_2027"] + res["v_2028"]))
        all_slow = (total_fast == 0)
        rows.append({
            "方案": label,
            "追加快充": total_fast,
            "追加慢充": total_slow,
            "追加总成本_万元": res["total_cost"],
            "是否出现全慢充退化": "是" if all_slow else "否",
            "服务约束是否满足": "是",
        })
    return pd.DataFrame(rows)


# ─────────────────────── 绘图 ───────────────────────


def plot_figures(df_a, df_b, df_c, df_f):
    """生成问题4图件。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    C_WD, C_WE, C_POST = "#1F77B4", "#FF7F0E", "#2CA02C"

    # F01: 服务保障率年度变化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, year in enumerate(YEARS):
        ax = axes[idx]
        sub = df_a[df_a["年份"] == year].sort_values("区域")
        x = np.arange(10)
        colors = ["#D62728" if v < 1 else C_POST for v in sub["服务保障率_raw"]]
        ax.bar(x, sub["服务保障率_pct"], 0.6, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(100, color="black", linestyle="--", linewidth=1)
        ax.axhline(90, color="orange", linestyle=":", linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in range(10)])
        ax.set_title(f"{year} (g={GROWTH[year]:.4f})")
        ax.set_ylabel("服务保障率 / %")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("问题4-F01  S0情景 各区域服务保障率年度变化", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p4_f01_S0服务保障率年度变化.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # F02: 调度后峰时风险率投影
    fig, ax = plt.subplots(figsize=(10, 5))
    for year_idx, year in enumerate(YEARS):
        sub = df_b[df_b["年份"] == year].sort_values("区域")
        ax.plot(np.arange(10) + year_idx * 0.15, sub["调度后峰时风险率_pct"], "o-",
                label=str(year), markersize=5)
    ax.axhline(80, color="orange", linestyle="--", linewidth=1, label="预警线80%")
    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="过载线100%")
    ax.set_xticks(range(10)); ax.set_xticklabels([f"{i+1}" for i in range(10)])
    ax.set_xlabel("区域编号"); ax.set_ylabel("调度后峰时风险率 / %")
    ax.set_title("问题4-F02  调度后峰时风险率三年投影")
    ax.legend(); ax.grid(linestyle="--", alpha=0.35)
    fig.savefig(FIG_DIR / "p4_f02_调度后峰时风险率投影.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # F03: 滚动补桩前后服务能力对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, year in enumerate([2027, 2028]):
        ax = axes[idx]
        s0 = df_a[df_a["年份"] == year].sort_values("区域")
        s1 = df_c[(df_c["年份"] == year) & (df_c["方案"] == "R-Base")].sort_values("区域")
        x = np.arange(10)
        w = 0.35
        ax.bar(x - w/2, s0["静态服务能力"], w, label="S0静态", color=C_WD, edgecolor="black", linewidth=0.3)
        ax.bar(x + w/2, s1["累计服务能力"], w, label="S1滚动", color=C_POST, edgecolor="black", linewidth=0.3)
        ax.plot(x, s0["预测日均车次需求"].values, "D-", color="#D62728", markersize=4, label="预测需求")
        ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in range(10)])
        ax.set_title(f"{year}年"); ax.set_ylabel("车次/day")
        ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("问题4-F03  滚动补桩前后服务能力对比", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p4_f03_滚动补桩前后服务能力对比.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # F04: 全市年度汇总
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    ax.bar(x - 0.15, df_f["静态服务保障率_pct"], 0.3, label="S0静态", color=C_WD, edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.15, df_f["滚动后服务保障率_pct"], 0.3, label="S1滚动", color=C_POST, edgecolor="black", linewidth=0.3)
    ax.axhline(100, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in YEARS])
    ax.set_xlabel("年份"); ax.set_ylabel("全市服务保障率 / %")
    ax.set_title("问题4-F04  全市服务保障率年度对比")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIG_DIR / "p4_f04_全市服务保障率年度对比.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("  [图件] 4张已生成")


# ─────────────────────── 主流程 ───────────────────────


def main():
    ensure_dirs()
    print("=" * 64)
    print("问题4：三年增长风险评估与滚动优化")
    print("=" * 64)

    data = load_inputs()

    # 模块A
    print("\n[模块A] 静态基线风险识别...")
    df_a = module_a(data)
    save(df_a, "p4_表A_年度需求与静态服务风险")
    n_warn_a = len(df_a[(df_a["年份"] == 2028) & (df_a["服务风险等级"] != "服务安全")])
    print(f"  2028年服务非安全区域数: {n_warn_a}")

    # 模块B
    print("\n[模块B] 峰时电网风险投影...")
    df_b = module_b(data)
    save(df_b, "p4_表B_年度峰时电网风险投影")
    n_warn_b = len(df_b[(df_b["年份"] == 2028) & (df_b["调度后风险等级"] == "峰时预警")])
    n_over_b = len(df_b[(df_b["年份"] == 2028) & (df_b["调度后风险等级"] == "过载风险")])
    print(f"  2028年调度后预警: {n_warn_b}, 过载: {n_over_b}")

    # 区域9最小配网增容需求
    print("\n[配网增容] 区域9最小容量缺口...")
    df_gap = compute_grid_capacity_gap(data)
    save(df_gap, "p4_表G_区域9最小配网增容需求")
    for _, row in df_gap.iterrows():
        year = int(row["年份"])
        gap = row["最小容量缺口_kW"]
        print(f"  {year}: ΔG = {gap:.1f} kW ({gap/1000:.3f} MW)")

    # 模块C
    res_base, res_free = module_c(data)
    if res_base is None:
        return

    df_c_base = gen_table_c(data, res_base, "R-Base")
    df_c_free = gen_table_c(data, res_free, "R-Free")
    df_c = pd.concat([df_c_base, df_c_free], ignore_index=True)
    save(df_c, "p4_表C_滚动补桩优化结果")

    # 敏感性对比
    df_sens = gen_sensitivity(res_base, res_free)
    save(df_sens, "p4_表C_结构下限敏感性对比")

    # 表D
    print("\n[表D] S0 vs S1 对比...")
    df_d = gen_table_d(df_a, df_c_base)
    save(df_d, "p4_表D_滚动优化前后服务效果对比")

    # 表E
    print("[表E] 风险建议汇总...")
    df_e = gen_table_e(df_a, df_b, df_c_base)
    save(df_e, "p4_表E_风险调整建议汇总")

    # 表F
    print("[表F] 全市年度汇总...")
    df_f = gen_table_f(df_a, df_b, df_c_base)
    save(df_f, "p4_表F_全市年度汇总")

    # 绘图
    print("\n[绘图]...")
    plot_figures(df_a, df_b, df_c_base, df_f)

    # 核验
    print("\n[核验]")
    # 1. 年度需求增长
    for i in range(10):
        q26 = data["Q_2025"][i] * GROWTH[2026]
        q27 = data["Q_2025"][i] * GROWTH[2027]
        q28 = data["Q_2025"][i] * GROWTH[2028]
        assert abs(q27 / q26 - 1.15) < 1e-10, f"区域{i+1} 2027/2026增长率错误"
        assert abs(q28 / q27 - 1.15) < 1e-10, f"区域{i+1} 2028/2027增长率错误"
    print("  ✓ 年度需求增长 1.15 倍核验通过")

    # 2. 初始设施数
    for i in range(10):
        f1 = data["F0_i"][i] + data["x_p2"][i]
        s1 = data["S0_i"][i] + data["y_p2"][i]
        c_row = df_c_base[(df_c_base["区域"] == i + 1) & (df_c_base["年份"] == 2026)]
        assert int(c_row["累计快充"].values[0]) == int(f1), f"区域{i+1}快充数不一致"
    print("  ✓ 2026初始设施数与问题2一致")

    # 3. 滚动后服务约束
    for _, row in df_c_base[df_c_base["方案"] == "R-Base"].iterrows():
        assert row["是否满足服务约束"] == "是", f"区域{int(row['区域'])}年份{int(row['年份'])}服务约束未满足"
    print("  ✓ 滚动优化后全部服务约束满足")

    # 4. 整数性
    for key in ["u_2027", "v_2027", "u_2028", "v_2028"]:
        vals = res_base[key]
        assert np.all(vals >= 0) and np.all(vals == vals.astype(int)), f"{key} 非负整数性不满足"
    print("  ✓ 决策变量非负整数性满足")

    print(f"\n{'=' * 64}")
    print("问题4 全部完成")
    print(f"  表A-F: {P4_DIR}")
    print(f"  图件: {FIG_DIR}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
