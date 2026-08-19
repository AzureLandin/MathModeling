# -*- coding: utf-8 -*-
"""
问题3 · Step 3：调度效能指标计算、正确性核验与三张结果表输出（任务单 §9 / §10）
===============================================================================

输入：
    results/p3_data/  L_pre_wd.csv, L_pre_we.csv, G_limit.csv, time_sets.json
    results/p3_solve/subproblem_results.json   （Step2 的 20 个子问题求解记录）

对每个 (i,d) 输出三类表：
    表1（§9.1）调度前峰谷特征：峰值/谷值/峰谷差/方差/原始最大负荷率 + 并列极值时段
    表2（§9.2）低谷分配与可行性：M、B、是否可严格转移20%、7 个低谷逐时分配量、最小安全裕度
    表3（§9.3）调度效果评价：峰谷差改善率 / 最大负荷削减率 / 方差改善率 / 调度后最大负荷率
                                  / 最小安全裕度 / 2100kW 风险小时数

核验断言（§10.1，全部子问题必须通过）：
    |ΣL̃ − ΣL| < 1e-6；max(L̃−G) ≤ 1e-6；|Σ_V z − M_used| < 1e-6；z ≥ −1e-6

输出（results/p3_results/）：
    p3_表1_调度前峰谷特征.csv / .xlsx
    p3_表2_低谷分配与可行性.csv / .xlsx
    p3_表3_调度效果评价.csv   / .xlsx
    p3_curves_all.csv         —— 20 条调度后曲线 + 电网上限（长格式，供绘图/论文引用）
    p3_checks.json            —— 核验明细与求解器状态汇总（回传建模组用）

用法：python problem3_metrics_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "p3_data"
SOLVE_DIR = ROOT / "results" / "p3_solve"
OUT_DIR = ROOT / "results" / "p3_results"

TOL_EQ = 1e-6        # §10.1 等式容差
EPS_KAPPA = 1e-6     # §4 峰谷比 ε
RISK_KW = 2100.0     # §7.4 补充过载审查阈值（kW）

HOUR_LABELS = [f"{h:02d}-{(h+1) % 24:02d}" for h in range(24)]


def load_base() -> tuple[pd.DataFrame, dict]:
    with open(DATA_DIR / "time_sets.json", encoding="utf-8") as f:
        TS = json.load(f)
    L_wd = pd.read_csv(DATA_DIR / "L_pre_wd.csv")
    L_we = pd.read_csv(DATA_DIR / "L_pre_we.csv")
    Gm = pd.read_csv(DATA_DIR / "G_limit.csv")
    sub_results = json.loads((SOLVE_DIR / "subproblem_results.json").read_text(encoding="utf-8"))
    return (pd.concat([L_wd.assign(日期类型="工作日"), L_we.assign(日期类型="周末")], ignore_index=True), TS, Gm, sub_results)


def curves_of(L: np.ndarray, z: list[float], peak_hours: list[int]) -> np.ndarray:
    """按调度规则构造 24h 曲线：峰段 ×0.8、谷段 +z_k、其余保持。"""
    Lp = L.copy().astype(float)
    peak = [TS_GLOB["peak_hours"][j] for j in range(len(TS_GLOB["peak_hours"]))]
    Lp[np.array(peak)] *= (1 - TS_GLOB["transfer_ratio_peak"])
    for k, t in enumerate(TS_GLOB["valley_hours"]):
        Lp[int(t)] += float(z[k])
    return Lp


TS_GLOB: dict = {}   # 由 main() 填充，避免参数穿透


def main() -> None:
    global TS_GLOB
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all, TS_, Gm, sub_results = load_base()
    TS_GLOB = TS_
    G = {int(row["区域"]): row[[f"{h:02d}-{(h+1)%24:02d}" for h in range(24)]].to_numpy(float)
         for _, row in Gm.iterrows()}

    t1_rows, t2_rows, t3_rows = [], [], []
    curve_long: list[dict] = []
    checks: list[dict] = []
    n_valley_new_max = 0   # §10.2-6：谷段是否成为调度后新峰值（全天）

    for rec in sub_results:
        i, d_name = int(rec["region"]), rec["date_type"]
        is_wd = (d_name == "wd")
        L = df_all[(df_all["区域"] == i) & (df_all["日期类型"] == ("工作日" if is_wd else "周末"))]
        assert len(L) == 1, f"缺数：区域{i}/{d_name}"
        Lv = L.iloc[0][HOUR_LABELS].to_numpy(float)
        Gi = G[i]

        z = np.array(rec["z_valley"], dtype=float)          # 7维分配向量
        post = curves_of(Lv, rec["z_valley"], TS_["peak_hours"])
        pre_max_t = list(np.flatnonzero(Lv == Lv.max()))    # §4：完整记录并列极值时段
        pre_min_t = list(np.flatnonzero(Lv == Lv.min()))

        # ---------- 表1：调度前特征（§9.1） ----------
        Lp_mean_pre_cap = float((Lv / Gi).max())            # 原始最大负荷率 φ
        t1_rows.append({
            "区域": i, "日期类型": "工作日" if is_wd else "周末",
            "调度前峰值kW": round(float(Lv.max()), 2),
            "峰值时段(并列)": ",".join(HOUR_LABELS[t] for t in pre_max_t[:6]),
            "调度前谷值kW": round(float(Lv.min()), 2),
            "谷值时段(并列)": ",".join(HOUR_LABELS[t] for t in pre_min_t[:6]),
            "调度前峰谷差kW": round(float(Lv.max() - Lv.min()), 2),
            "调度前方差_kW2_": round(float(np.mean((Lv - Lv.mean()) ** 2)), 2),
            "原始最大负荷率%": round(Lp_mean_pre_cap * 100, 2)})

        # ---------- 表2：低谷分配与可行性（§9.2） ----------
        pre_safe = float(Gi.min() - post.min())              # 调度后最小安全裕度 S_min
        reason_key = "infeas_reason"
        feas_txt = "是" if rec["feasible_20pct"] else f"否（{rec[reason_key]}）"
        t2_rows.append({
            "区域": i, "日期类型": "工作日" if is_wd else "周末",
            "高峰负荷总量kW": round(rec["peak_total_kW"], 2),
            "规定转移量M_kW_": round(rec["M_kW"], 2),
            "低谷可接纳量B_kW_": round(rec["B_kW"], 2),
            "是否可严格转移20%": feas_txt,
            **{f"谷分配_{HOUR_LABELS[t]}kW": round(float(z[k]), 2) for k, t in enumerate(TS_["valley_hours"])},
            "分配总量kW核对": round(float(z.sum()), 2),
            "调度后最小安全裕度kW": round(pre_safe, 2)})

        # ---------- 表3：调度效果评价（§9.3） ----------
        gap_pre = float(Lv.max() - Lv.min())
        gap_post = float(post.max() - post.min())
        max_pre, max_post = float(Lv.max()), float(post.max())
        var_pre = float(np.mean((Lv - Lv.mean()) ** 2))      # §9.3 σ² pre（/24口径）
        bar = float(Lv.mean())                                # 电量守恒 ⇒ post 均值 = pre 均值
        var_post = float(np.mean((post - bar) ** 2))          # §9.3 σ² post
        risk_hours_2100 = int((post > RISK_KW).sum())         # §7.4 补充风险标记（不替代逐时约束）

        t3_rows.append({
            "区域": i, "日期类型": "工作日" if is_wd else "周末",
            "调度后峰谷差kW": round(gap_post, 2),
            "峰谷差改善率%": round((gap_pre - gap_post) / gap_pre * 100.0, 2),
            "调度后最大负荷kW": round(max_post, 2),
            "最大负荷削减率%": round((max_pre - max_post) / max_pre * 100.0, 2),
            "调度后方差_kW2_": round(var_post, 2),
            "方差改善率%": round((var_pre - var_post) / var_pre * 100.0, 2),
            "调度后最大负荷率%": round(float((post / Gi).max()) * 100, 2),
            "调度后最小安全裕度kW": round(pre_safe, 2),
            "2100kW风险小时数": risk_hours_2100})

        # ---------- 长格式曲线（绘图用）+ §10.2-6 新峰值标记 ----------
        for h in range(24):
            curve_long.append({"区域": i, "日期类型": "工作日" if is_wd else "周末",
                               "小时段": HOUR_LABELS[h], "hour": int(h),
                               "调度前kW": round(float(Lv[h]), 3),
                               "调度后kW": round(float(post[h]), 3),
                               "电网上限kW": float(Gi[h])})
        if (int(np.argmax(post)) in TS_["valley_hours"]) and (gap_post > gap_pre - 1e-9):
            n_valley_new_max += 1     # 谷底转移后成为全天新峰值

        # ---------- §10.1 正确性核验 ----------
        e_mass = abs(float(post.sum()) - float(Lv.sum()))     # 电量守恒残差（§10.1-1）
        e_cap = float((post - Gi).max())
        e_zmass = abs(float(z.sum()) - rec["M_used_kW"])
        ok_zneg = bool((z >= -TOL_EQ).all())
        checks.append({
            "区域": i, "日期类型": d_name,
            "电量守恒残差": round(e_mass, 9), "安全越限最大量kW": round(max(e_cap, 0.0), 9),
            "转移量闭合残差kW": round(e_zmass, 9), "z非负": ok_zneg,
            "通过": bool(abs(e_mass) < TOL_EQ and e_cap <= TOL_EQ and abs(e_zmass) < TOL_EQ and ok_zneg),
            "求解器状态LP": rec["status_lp"], "求解器状态QP": rec["status_qp"] or "-",
            "次目标启用": rec["stage2_enabled"],
            "LP最优峰谷差kW": rec["delta_star_kW"], "耗时s": rec["solve_time_s"]})

    t1, t2, t3 = pd.DataFrame(t1_rows), pd.DataFrame(t2_rows), pd.DataFrame(t3_rows)
    for name, dfx in [("p3_表1_调度前峰谷特征", t1), ("p3_表2_低谷分配与可行性", t2), ("p3_表3_调度效果评价", t3)]:
        dfx.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(OUT_DIR / f"{name}.xlsx", engine="openpyxl") as w:
            dfx.to_excel(w, index=False, sheet_name=name[:20])

    pd.DataFrame(curve_long).to_csv(OUT_DIR / "p3_curves_all.csv", index=False, encoding="utf-8-sig")

    n_fail = sum(1 for c in checks if not c["通过"])
    payload = {
        "全部子问题可行": bool(all(rec["feasible_20pct"] for rec in sub_results)),
        "核验不通过数(应为0)": int(n_fail),
        "谷段成为调度后全天峰值的子问题数": n_valley_new_max,
        "求解器状态与统计": [
            {"区域": c["区域"], "日期类型": c["日期类型"], **{k: c[k] for k in
             ["通过", "电量守恒残差", "安全越限最大量kW", "转移量闭合残差kW", "z非负",
              "求解器状态LP", "求解器状态QP", "次目标启用", "LP最优峰谷差kW", "耗时s"]}} for c in checks],
    }
    (OUT_DIR / "p3_checks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print("[OK] 三张表 + 曲线长表 + 核验明细已输出 →", OUT_DIR)
    print(f"     20/20 可行 : {payload['全部子问题可行']}   核验失败: {n_fail}   "
          f"谷段成为新全天峰值的子问题数: {n_valley_new_max}")
    show = t3[["区域", "日期类型", "调度后峰谷差kW", "峰谷差改善率%", "最大负荷削减率%", "方差改善率%",
               "调度后最小安全裕度kW"]].copy()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
