# -*- coding: utf-8 -*-
"""
problem2_main.py
问题二主入口：新增快慢充桩多目标规划（NSGA-II + 熵权-TOPSIS）。

用法：
    python problem2_main.py

依赖见同目录 requirements.txt（pymoo >= 0.6）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from problem2_utils import (
    C_F, C_S, P_F, P_S, S_F, S_S, COVERAGE_MIN, GROWTH_RATE,
    N_REGIONS, SEEDS, COLORS,
    ensure_dirs, setup_plot_style, save_table, save_json, df_to_markdown,
    RESULTS_DIR, FIGURES_DIR, REPORTS_DIR,
)
from problem2_data import (
    build_problem_data, grid_safety_check, build_input_table,
)
from problem2_nsga2 import (
    run_model_seeds, merge_and_filter_solutions, compute_objectives,
)
from problem2_topsis import (
    select_best_solution, robustness_check,
)
from problem2_figures import (
    plot_pareto_3d, plot_stacked_bar, plot_pressure_bar,
    plot_pressure_comparison, plot_topsis_sensitivity,
)


# ─────────────────────────── 稳定性分析 ───────────────────────────


def stability_analysis(
    seed_results: list[dict],
    pareto_merged: list[dict],
    model: str,
) -> pd.DataFrame:
    """表3：算法稳定性（5次独立运行的 Pareto 解数、推荐解指标均值±标准差）。"""
    rows = []
    for r in seed_results:
        sols = r["solutions"]
        if sols:
            idx, _, gamma = select_best_solution(sols, model)
            s = sols[idx]
            rows.append({
                "模型": model,
                "种子": r["seed"],
                "Pareto解数": len(sols),
                "推荐成本": s["f1"],
                "推荐覆盖率": s["C_city"],
                "推荐nubar": s["nubar"],
                "推荐nu_max": s["nu_max"],
                "推荐nu_std": s["nu_std"],
                "TOPSIS贴近度": gamma,
            })
        else:
            rows.append({
                "模型": model,
                "种子": r["seed"],
                "Pareto解数": 0,
                "推荐成本": np.nan,
                "推荐覆盖率": np.nan,
                "推荐nubar": np.nan,
                "推荐nu_max": np.nan,
                "推荐nu_std": np.nan,
                "TOPSIS贴近度": np.nan,
            })

    df = pd.DataFrame(rows)

    # 添加均值±标准差行
    numeric_cols = ["Pareto解数", "推荐成本", "推荐覆盖率", "推荐nubar", "推荐nu_max", "推荐nu_std"]
    stats = {}
    for c in numeric_cols:
        vals = df[c].dropna()
        if len(vals) > 0:
            stats[f"{c}_均值"] = vals.mean()
            stats[f"{c}_标准差"] = vals.std()
    stats["模型"] = model
    stats["种子"] = "汇总"
    stats["Pareto解数_合并"] = len(pareto_merged)

    return df


def topsis_report_table(
    seed_results: list[dict],
    pareto_merged: list[dict],
    model: str,
) -> pd.DataFrame:
    """表4：模型内部 TOPSIS 推荐方案。"""
    rows = []
    for r in seed_results:
        sols = r["solutions"]
        if not sols:
            continue
        # 熵权
        idx_e, w_e, gamma_e = select_best_solution(sols, model, None)
        # 稳健性
        rob = robustness_check(sols, model)

        rows.append({
            "模型": model,
            "随机种子": r["seed"],
            "Pareto候选数": len(sols),
            "熵权": np.array2string(w_e, precision=4),
            "熵权TOPSIS贴近度": gamma_e,
            "等权方案一致": "是" if rob["equal_idx"] == rob["entropy_idx"] else "否",
            "偏成本方案一致": "是" if rob["cost_bias_idx"] == rob["entropy_idx"] else "否",
        })

    # 合并后的推荐
    if pareto_merged:
        rob_merged = robustness_check(pareto_merged, model)
        idx_e, w_e, gamma_e = select_best_solution(pareto_merged, model, None)
        rows.append({
            "模型": model,
            "随机种子": "合并",
            "Pareto候选数": len(pareto_merged),
            "熵权": np.array2string(w_e, precision=4),
            "熵权TOPSIS贴近度": gamma_e,
            "等权方案一致": "是" if rob_merged["equal_idx"] == rob_merged["entropy_idx"] else "否",
            "偏成本方案一致": "是" if rob_merged["cost_bias_idx"] == rob_merged["entropy_idx"] else "否",
        })

    return pd.DataFrame(rows)


# ─────────────────────────── 最终方案输出 ───────────────────────────


def final_region_table(solution: dict, data: dict, model: str) -> pd.DataFrame:
    """表6：最终区域新增桩配置方案。"""
    x = solution["x"]
    y = solution["y"]
    A_i = data["A_i"]
    A0 = data["A0_i"]
    a = data["a_i"]
    G = data["G_i"]
    F0 = data["F0_i"]
    S0 = data["S0_i"]
    Q_plan = data["Q_plan"]

    rows = []
    for i in range(N_REGIONS):
        C_new = min(1.0, (A0[i] + a[i] * (x[i] + y[i])) / A_i[i])
        P_add = P_F * x[i] + P_S * y[i]
        nu_i = P_add / G[i]
        cap = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i])
        margin = cap - Q_plan[i]
        rows.append({
            "区域": i + 1,
            "新增快充_xi": int(x[i]),
            "新增慢充_yi": int(y[i]),
            "新增总桩数": int(x[i] + y[i]),
            "新增成本_万元": C_F * x[i] + C_S * y[i],
            "新覆盖率": C_new,
            "新增额定接入功率_kW": P_add,
            "新增接入压力_nu": nu_i,
            "服务能力裕度_车次日": margin,
        })

    # 全市汇总行
    total_x = int(np.sum(x))
    total_y = int(np.sum(y))
    total_cost = float(np.sum(C_F * x + C_S * y))
    total_p = float(np.sum(P_F * x + P_S * y))
    C_city = solution["C_city"]
    nubar = solution["nubar"]

    rows.append({
        "区域": "全市",
        "新增快充_xi": total_x,
        "新增慢充_yi": total_y,
        "新增总桩数": total_x + total_y,
        "新增成本_万元": total_cost,
        "新覆盖率": C_city,
        "新增额定接入功率_kW": total_p,
        "新增接入压力_nu": nubar,
        "服务能力裕度_车次日": "",
    })

    return pd.DataFrame(rows)


def model_comparison_table(sol_a: dict, sol_b: dict) -> pd.DataFrame:
    """表5：M-A 与 M-B 统一对比。"""
    rows = []
    for label, sol in [("M-A", sol_a), ("M-B", sol_b)]:
        rows.append({
            "模型": label,
            "总成本_万元": sol["f1"],
            "覆盖率": sol["C_city"],
            "平均压力_nubar": sol["nubar"],
            "最大压力_numax": sol["nu_max"],
            "压力标准差": sol["nu_std"],
            "压力平方和": sol["nu_sqsum"],
            "压力方差": sol["nu_var"],
            "新增快充": int(np.sum(sol["x"])),
            "新增慢充": int(np.sum(sol["y"])),
            "是否可行": "是",
        })
    return pd.DataFrame(rows)


# ─────────────────────────── 模型选择 ───────────────────────────


def select_final_model(sol_a: dict, sol_b: dict) -> str:
    """
    按任务单 §9 规则选择最终模型。
    返回 "M-A" 或 "M-B"。
    """
    # 规则1：M-A 成本与覆盖率不劣于 M-B，且 nubar/numax 更低或近似
    cost_ratio = (sol_b["f1"] - sol_a["f1"]) / max(sol_a["f1"], 1)
    cov_diff = sol_a["C_city"] - sol_b["C_city"]
    nubar_diff = sol_a["nubar"] - sol_b["nubar"]
    numax_diff = sol_a["nu_max"] - sol_b["nu_max"]

    if cost_ratio >= -0.05 and cov_diff >= -0.01 and nubar_diff <= 0.05 and numax_diff <= 0.05:
        # M-A 成本和覆盖率不劣，压力不显著更高
        return "M-A"

    # 规则2：M-B 显著降低方差，且恶化不超过5%
    if sol_b["nu_std"] < sol_a["nu_std"] * 0.8:
        if cost_ratio < 0.05 and nubar_diff < 0.05 and numax_diff < 0.05:
            return "M-B"

    # 规则3：M-B 仅降低方差但显著提高 nubar/numax
    if sol_b["nu_std"] < sol_a["nu_std"] and (nubar_diff > 0.05 or numax_diff > 0.05):
        return "M-A"  # 高压力伪均衡

    # 规则4：差异很小，优先 M-A
    return "M-A"


# ─────────────────────────── 报告生成 ───────────────────────────


def write_report(
    data: dict,
    recommended_a: dict,
    recommended_b: dict,
    final_model: str,
    final_sol: dict,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3a: pd.DataFrame,
    table3b: pd.DataFrame,
    table4a: pd.DataFrame,
    table4b: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
) -> None:
    """生成结果分析报告。"""
    sol_a = recommended_a["solution"]
    sol_b = recommended_b["solution"]

    grid_ok = table2["是否越限"].eq("否").all()
    grid_conclusion = (
        "全部区域均未越限，2026年15%情景下仅增加桩数量即可满足电网安全。"
        if grid_ok
        else "存在越限区域，需转交问题3调度或问题4电网扩容处理。"
    )

    report = f"""# 问题 2：新增快慢充桩多目标规划 — 计算结果报告

> 由 `code/problem2_main.py` 自动生成。

---

## 1. 输入与预处理

### 1.1 区域参数

{df_to_markdown(table1, floatfmt="{:.4f}")}

### 1.2 电网逐时安全预检查

{df_to_markdown(table2, floatfmt="{:.4f}")}

**结论**：{grid_conclusion}

---

## 2. NSGA-II 算法稳定性

### 2.1 M-A 模型

{df_to_markdown(table3a, floatfmt="{:.4f}")}

### 2.2 M-B 模型

{df_to_markdown(table3b, floatfmt="{:.4f}")}

---

## 3. TOPSIS 推荐方案

### 3.1 M-A 模型

{df_to_markdown(table4a, floatfmt="{:.4f}")}

### 3.2 M-B 模型

{df_to_markdown(table4b, floatfmt="{:.4f}")}

---

## 4. M-A 与 M-B 统一对比

{df_to_markdown(table5, floatfmt="{:.4f}")}

### 模型选择

最终选择 **{final_model}**。

选择依据：
- M-A 压力平方和 = {sol_a["nu_sqsum"]:.6f}，M-B 压力平方和 = {sol_b["nu_sqsum"]:.6f}
- M-A 平均压力 ν̄ = {sol_a["nubar"]:.6f}，M-B 平均压力 ν̄ = {sol_b["nubar"]:.6f}
- M-A 最大压力 ν_max = {sol_a["nu_max"]:.6f}，M-B 最大压力 ν_max = {sol_b["nu_max"]:.6f}
- M-A 压力标准差 σ = {sol_a["nu_std"]:.6f}，M-B 压力标准差 σ = {sol_b["nu_std"]:.6f}

---

## 5. 最终区域新增桩配置方案

{df_to_markdown(table6, floatfmt="{:.4f}")}

---

## 6. 约束验证

"""
    # 约束验证
    sol = final_sol
    x, y = sol["x"], sol["y"]
    F0, S0 = data["F0_i"], data["S0_i"]
    A_i, A0, a = data["A_i"], data["A0_i"], data["a_i"]
    Q_plan = data["Q_plan"]

    all_ok = True
    checks = []
    for i in range(N_REGIONS):
        c1 = (A0[i] + a[i] * (x[i] + y[i])) / A_i[i] >= COVERAGE_MIN - 1e-9
        c2 = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i]) >= Q_plan[i] - 1e-9
        c3 = S0[i] * x[i] >= F0[i] * y[i] - 1e-9
        ok = c1 and c2 and c3
        if not ok:
            all_ok = False
        checks.append(f"| {i+1} | {'是' if c1 else '否'} | {'是' if c2 else '否'} | {'是' if c3 else '否'} | {'是' if ok else '否'} |")

    report += "| 区域 | 覆盖率≥90% | 服务能力 | 快充结构 | 全部通过 |\n"
    report += "|---:|---|---|---|---|\n"
    report += "\n".join(checks)
    report += f"\n\n**总体结论**：{'全部约束均满足。' if all_ok else '存在约束违反，需进一步检查。'}\n"

    report += """
---

## 7. 输出清单

| 类型 | 路径 |
| --- | --- |
| 输入预处理表 | `results/问题2_表1_输入与约束预处理.*` |
| 电网安全预检查 | `results/问题2_表2_电网逐时安全预检查.*` |
| M-A 稳定性 | `results/问题2_表3A_M-A算法稳定性.*` |
| M-B 稳定性 | `results/问题2_表3B_M-B算法稳定性.*` |
| M-A TOPSIS | `results/问题2_表4A_M-A_TOPSIS推荐.*` |
| M-B TOPSIS | `results/问题2_表4B_M-B_TOPSIS推荐.*` |
| 统一对比 | `results/问题2_表5_M-A_M-B统一对比.*` |
| 最终配置 | `results/问题2_表6_最终区域新增桩配置.*` |
| Pareto 前沿 3D | `figures/图1_Pareto前沿_3D.png` |
| 新增桩堆叠图 | `figures/图2_新增快慢充桩堆叠柱状图.png` |
| 接入压力柱状图 | `figures/图3_新增接入压力率柱状图.png` |
| 压力对比折线图 | `figures/图4_压力分布对比折线图.png` |
| TOPSIS 敏感性 | `figures/图5_TOPSIS敏感性_*.png` |

---

*生成完毕。*
"""

    sol_a = recommended_a["solution"]
    sol_b = recommended_b["solution"]
    out = REPORTS_DIR / "问题2_新增快慢充桩多目标规划_结果报告.md"
    out.write_text(report, encoding="utf-8")
    print(f"[OK] 报告 -> {out}")


# ─────────────────────────── 主流程 ───────────────────────────


def main() -> None:
    t0 = time.time()
    ensure_dirs()
    setup_plot_style()

    print("=" * 64)
    print("问题二：新增快慢充桩多目标规划")
    print("=" * 64)

    # ── 1. 数据加载 ──
    print("\n[1/7] 加载数据与预处理...")
    data = build_problem_data()
    table1 = build_input_table(data)
    save_table(table1, "问题2_表1_输入与约束预处理")
    print(f"  -> 表1: {N_REGIONS} 个区域参数加载完成")

    # ── 2. 电网安全预检查 ──
    print("\n[2/7] 电网逐时安全预检查...")
    table2 = grid_safety_check(data)
    save_table(table2, "问题2_表2_电网逐时安全预检查")
    n_violate = int(table2["是否越限"].eq("是").sum())
    print(f"  -> 越限区域数: {n_violate}/{N_REGIONS}")

    # ── 3. NSGA-II 运行 M-A ──
    print("\n[3/7] 运行 M-A（压力平方和）NSGA-II...")
    results_ma = run_model_seeds(data, "MA")
    pareto_ma = merge_and_filter_solutions(results_ma, "MA")
    print(f"  -> M-A 合并 Pareto 候选: {len(pareto_ma)} 个")

    # ── 4. NSGA-II 运行 M-B ──
    print("\n[4/7] 运行 M-B（压力方差）NSGA-II...")
    results_mb = run_model_seeds(data, "MB")
    pareto_mb = merge_and_filter_solutions(results_mb, "MB")
    print(f"  -> M-B 合并 Pareto 候选: {len(pareto_mb)} 个")

    # ── 5. TOPSIS 择优 ──
    print("\n[5/7] 熵权-TOPSIS 择优...")

    # 稳定性分析
    table3a = stability_analysis(results_ma, pareto_ma, "MA")
    table3b = stability_analysis(results_mb, pareto_mb, "MB")
    save_table(table3a, "问题2_表3A_M-A算法稳定性")
    save_table(table3b, "问题2_表3B_M-B算法稳定性")

    # TOPSIS 推荐
    table4a = topsis_report_table(results_ma, pareto_ma, "MA")
    table4b = topsis_report_table(results_mb, pareto_mb, "MB")
    save_table(table4a, "问题2_表4A_M-A_TOPSIS推荐")
    save_table(table4b, "问题2_表4B_M-B_TOPSIS推荐")

    # 合并 Pareto 集的推荐解
    idx_ma, w_ma, gamma_ma = select_best_solution(pareto_ma, "MA")
    idx_mb, w_mb, gamma_mb = select_best_solution(pareto_mb, "MB")

    recommended_a = {
        "model": "M-A",
        "solution": pareto_ma[idx_ma],
        "weights": w_ma,
        "gamma": gamma_ma,
        "robustness": robustness_check(pareto_ma, "MA"),
    }
    recommended_b = {
        "model": "M-B",
        "solution": pareto_mb[idx_mb],
        "weights": w_mb,
        "gamma": gamma_mb,
        "robustness": robustness_check(pareto_mb, "MB"),
    }

    print(f"  -> M-A 推荐: 成本={recommended_a['solution']['f1']:.2f}, "
          f"覆盖={recommended_a['solution']['C_city']:.4f}, "
          f"ν̄={recommended_a['solution']['nubar']:.6f}")
    print(f"  -> M-B 推荐: 成本={recommended_b['solution']['f1']:.2f}, "
          f"覆盖={recommended_b['solution']['C_city']:.4f}, "
          f"ν̄={recommended_b['solution']['nubar']:.6f}")

    # ── 6. 模型对比与选择 ──
    print("\n[6/7] M-A 与 M-B 统一对比...")
    table5 = model_comparison_table(recommended_a["solution"], recommended_b["solution"])
    save_table(table5, "问题2_表5_M-A_M-B统一对比")

    final_model = select_final_model(recommended_a["solution"], recommended_b["solution"])
    if final_model == "M-A":
        final_sol = recommended_a["solution"]
    else:
        final_sol = recommended_b["solution"]
    print(f"  -> 最终选择: {final_model}")

    # 最终区域配置表
    table6 = final_region_table(final_sol, data, final_model)
    save_table(table6, "问题2_表6_最终区域新增桩配置")

    # ── 7. 生成图表与报告 ──
    print("\n[7/7] 生成图表与报告...")
    plot_pareto_3d(pareto_ma, pareto_mb)
    plot_stacked_bar({"solution": final_sol, "model": final_model})
    plot_pressure_bar({"solution": final_sol, "model": final_model})
    plot_pressure_comparison(recommended_a, recommended_b)
    plot_topsis_sensitivity(pareto_ma, "MA",
                            recommended_a["robustness"]["entropy_idx"],
                            recommended_a["robustness"]["equal_idx"],
                            recommended_a["robustness"]["cost_bias_idx"])
    plot_topsis_sensitivity(pareto_mb, "MB",
                            recommended_b["robustness"]["entropy_idx"],
                            recommended_b["robustness"]["equal_idx"],
                            recommended_b["robustness"]["cost_bias_idx"])

    write_report(data, recommended_a, recommended_b, final_model, final_sol,
                 table1, table2, table3a, table3b, table4a, table4b, table5, table6)

    # 保存汇总 JSON
    save_json({
        "final_model": final_model,
        "total_cost": float(final_sol["f1"]),
        "coverage_city": float(final_sol["C_city"]),
        "nubar": float(final_sol["nubar"]),
        "nu_max": float(final_sol["nu_max"]),
        "nu_std": float(final_sol["nu_std"]),
        "total_fast": int(np.sum(final_sol["x"])),
        "total_slow": int(np.sum(final_sol["y"])),
        "pareto_ma_count": len(pareto_ma),
        "pareto_mb_count": len(pareto_mb),
    }, "问题2_汇总")

    elapsed = time.time() - t0
    print(f"\n{'=' * 64}")
    print(f"问题二全部完成 (耗时 {elapsed:.1f}s)")
    print(f"最终模型: {final_model}")
    print(f"总成本: {final_sol['f1']:.2f} 万元")
    print(f"全市覆盖率: {final_sol['C_city']:.4f}")
    print(f"平均接入压力: {final_sol['nubar']:.6f}")
    print(f"最大接入压力: {final_sol['nu_max']:.6f}")
    print(f"新增快充: {int(np.sum(final_sol['x']))} 台")
    print(f"新增慢充: {int(np.sum(final_sol['y']))} 台")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
