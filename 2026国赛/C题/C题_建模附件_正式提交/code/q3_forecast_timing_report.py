#!/usr/bin/env python
"""问题三 新增预报时刻必要性分析 —— 报告层（只读渲染）。

只读 ``results/q3_forecast_timing_diagnostic/``，渲染结果报告与最多两张图。
**不重算分组、不重建预报、不调用求解器**；改文案或图表不触发
``code/q3_forecast_timing_diagnostic.py``。

本版已按 ``reports/问题三/问题三_新增预报时刻分析接收核查.md`` 修正五处：
候选筛选不排除其他时刻、储能/当期误差不作因果排除、晚间实测光伏非严格为零与计数、
SOC 起止标签、预报年龄不可辨识性的适用范围。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_forecast_timing_report.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/q3_forecast_timing_diagnostic"
FIG = ROOT / "figures/q3_forecast_timing_diagnostic"
REPORT_MD = ROOT / "reports/问题三/问题三_新增预报时刻必要性分析结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_新增预报时刻必要性轻量分析方案.md"
REVIEW_MD = ROOT / "reports/问题三/问题三_新增预报时刻分析接收核查.md"
COST_REVIEW = ROOT / "reports/问题三/问题三_分位费用实验接收核查.md"
COST_REPORT = ROOT / "reports/问题三/问题三_偏差校正与分位水平费用实验结果.md"

assert Path(sys.prefix).name == "math_modeling", sys.prefix

COLOUR = {"L75": "#4C72B0", "L80": "#DD8452"}
NODE_COLOUR = {0: "#4C72B0", 6: "#55A868", 12: "#DD8452", 18: "#8172B3"}
EVENING, MORNING = (19, 20, 21, 22, 23), (6, 7, 8, 9, 10)


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def num(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def pct(value, digits=1):
    """Percentage that stays readable when the subset has no paired rows (midnight hour)."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def make_hourly_figure(hourly):
    block = hourly.sort_values("clock_hour")
    x = np.arange(24)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))
    ax = axes[0]
    width = 0.4
    ax.bar(x - width / 2, block.emergency_kWh_L75.to_numpy(), width, label="L75（Linear+q75）",
           color=COLOUR["L75"])
    ax.bar(x + width / 2, block.emergency_kWh_L80.to_numpy(), width, label="L80（Linear+q80）",
           color=COLOUR["L80"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in block.clock_hour], fontsize=8)
    ax.set_xlabel("时钟小时（区间起点）")
    ax.set_ylabel("实际应急电量（kWh）")
    ax.set_title("每时钟小时实际应急电量（全部 48096 段）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - width / 2, block.emergency_cost_yuan_L75.to_numpy(), width, label="L75",
           color=COLOUR["L75"])
    ax.bar(x + width / 2, block.emergency_cost_yuan_L80.to_numpy(), width, label="L80",
           color=COLOUR["L80"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in block.clock_hour], fontsize=8)
    ax.set_xlabel("时钟小时（区间起点）")
    ax.set_ylabel("实际应急费（元，5 倍电价）")
    ax.set_title("每时钟小时实际应急费；20 时占 L75 全期 40.8%")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("应急集中在傍晚（19—23 时）与上午（6—10 时）；其余 13 个时钟小时为 0")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG / "hourly_emergency.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_age_figure(hourly):
    """Publication node x forecast age: emergency rate and PV error, one panel each.

    (node, age) is in one-to-one correspondence with the clock hour, so each cell is exactly one
    hour; the figure is therefore a relabelling of the hourly profile, which the caption states.
    """
    block = hourly.sort_values("clock_hour").reset_index(drop=True)
    block = block.assign(version=block.clock_hour // 6, age_bin=block.clock_hour % 6)
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.4), sharex=True)
    x = np.arange(6)
    width = 0.2
    ax = axes[0]
    for offset, node in enumerate((0, 6, 12, 18)):
        part = block[block.version == node // 6].sort_values("age_bin")
        counts = part.n_intervals.to_numpy()
        ax.bar(x + (offset - 1.5) * width, part.emergency_rate_L75.to_numpy(), width,
               label=f"{node:02d}:00 发布（每格 {counts.min()}–{counts.max()} 段）",
               color=NODE_COLOUR[node])
    ax.set_ylabel("L75 应急区间占比")
    ax.set_title("发布节点 × 预报年龄：应急区间占比（同一实际口径下的 L75）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for offset, node in enumerate((0, 6, 12, 18)):
        part = block[block.version == node // 6].sort_values("age_bin")
        ax.plot(x + (offset - 1.5) * width, part.pv_mae_kW_all.to_numpy(), marker="o",
                markersize=4, linewidth=1.2, label=f"{node:02d}:00 发布", color=NODE_COLOUR[node])
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{i},{i + 1})" for i in range(6)])
    ax.set_xlabel("预报年龄 τ（距最近一次发布的时长，小时）")
    ax.set_ylabel("光伏 MAE（kW）")
    ax.set_title("同一单元格上的光伏预报误差：年龄越大并不单调更大，取决于所在时钟小时")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.suptitle("6 小时发布网格下 (发布节点, 预报年龄) 与时钟小时一一对应，年龄与日内效应不可分离")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIG / "node_age_grid.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_integrity(path):
    """Programmatic only: this agent cannot view images, so aesthetics stay unverified."""
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ink = float((rgb < 250).any(axis=2).mean())
    colours = int(len(np.unique(rgb.reshape(-1, 3), axis=0)))
    assert ink > 0.005, f"{path.name} looks blank (ink={ink:.4f})"
    assert colours > 20, f"{path.name} has too few colours ({colours})"
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                width=int(rgb.shape[1]), height=int(rgb.shape[0]), ink_coverage=round(ink, 4),
                distinct_colours=colours, note="programmatic only; the agent cannot view images")


def manifest_integrity(manifest):
    mismatched, missing = [], []
    for relative, expected in manifest["outputs"].items():
        path = OUT / relative
        if not path.exists():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatched.append(relative)
    return dict(checked=int(len(manifest["outputs"])), mismatched=mismatched, missing=missing)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_csv(OUT / "hourly_summary.csv")
    monthly_hourly = pd.read_csv(OUT / "monthly_hourly.csv")
    age = pd.read_csv(OUT / "age_summary.csv")
    risk = pd.read_csv(OUT / "risk_window_evidence.csv")
    candidates = pd.read_csv(OUT / "candidate_windows.csv")
    validation = load_json("validation.json")
    registration = load_json("registration.json")
    manifest = load_json("run_manifest.json")
    checks = validation["aggregates"]
    samples = validation["samples"]
    thresholds = registration["parameters"]["descriptive_thresholds"]

    paths = [make_hourly_figure(hourly), make_age_figure(hourly)]
    (OUT / "figure_integrity.json").write_text(
        json.dumps({p.name: figure_integrity(p) for p in paths}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    integrity = manifest_integrity(manifest)
    assert not integrity["mismatched"] and not integrity["missing"], integrity

    totals = validation["totals"]
    by_hour = hourly.set_index("clock_hour")
    total_cost = totals["L75"]["emergency_cost_yuan"]
    evening_cost = float(by_hour.loc[list(EVENING), "emergency_cost_yuan_L75"].sum())
    morning_cost = float(by_hour.loc[list(MORNING), "emergency_cost_yuan_L75"].sum())
    evening_energy = float(by_hour.loc[list(EVENING), "emergency_kWh_L75"].sum())
    morning_energy = float(by_hour.loc[list(MORNING), "emergency_kWh_L75"].sum())
    zero_hours = [h for h in range(24) if float(by_hour.loc[h, "emergency_kWh_L75"]) <= 1e-9]
    evening_pv = float(by_hour.loc[list(EVENING), "actual_pv_energy_kWh"].sum())
    midnight = age[age.group_type == "midnight_carry"].set_index("strategy_id")
    node_age = age[(age.group_type == "node_age") & (age.strategy_id == "L75")]
    merged = age[(age.group_type == "merged_age") & (age.strategy_id == "L75")]
    gap_total = float((hourly.emergency_cost_yuan_L75 - hourly.emergency_cost_yuan_L80).sum())
    top_gap = hourly.assign(gap=hourly.emergency_cost_yuan_L75 - hourly.emergency_cost_yuan_L80) \
        .nlargest(3, "gap")
    extra_hours = [int(h) for h in validation["rule_satisfying_hours"]]
    hour9 = risk[risk.clock_hour == 9].iloc[0]
    hour10 = risk[risk.clock_hour == 10].iloc[0]

    def hourly_table():
        head = ("| 时钟小时 | 实测光伏为正的区间 | 最大实测光伏（kW） | 实测光伏电量（kWh） | "
                "L75 应急电量（kWh） | L75 应急费（元） | 占 L75 全期 | L75 应急区间 | "
                "L80 应急费（元） | L75−L80 应急费（元） |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for hour in range(24):
            row = by_hour.loc[hour]
            body += (f"| {hour:02d} | {int(row.pv_positive_intervals)} | "
                     f"{num(row.max_actual_pv_kW, 4)} | {num(row.actual_pv_energy_kWh, 3)} | "
                     f"{num(row.emergency_kWh_L75, 2)} | {num(row.emergency_cost_yuan_L75, 2)} | "
                     f"{100.0 * row.emergency_cost_yuan_L75 / total_cost:.2f}% | "
                     f"{int(row.n_emergency_L75)} | {num(row.emergency_cost_yuan_L80, 2)} | "
                     f"{row.emergency_cost_yuan_L75 - row.emergency_cost_yuan_L80:+,.2f} |\n")
        return head + body

    def error_table():
        head = ("| 时钟小时 | 光伏 MAE（kW） | 平均 $a^V$（kWh） | 光伏高估占比 | 平均 $a^L$（kWh） | "
                "负载低估占比 | 平均 $g$（kWh） | $g>0$ 占比 | 光伏主导占比 |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for hour in range(24):
            row = by_hour.loc[hour]
            body += (f"| {hour:02d} | {num(row.pv_mae_kW_all, 2)} | {num(row.mean_a_pv_kWh_all, 3)} | "
                     f"{pct(row.share_pv_over_all)} | {num(row.mean_a_load_kWh_all, 3)} | "
                     f"{pct(row.share_load_under_all)} | {num(row.mean_g_kWh_all, 2)} | "
                     f"{pct(row.share_g_positive_all)} | "
                     f"{pct(row.pv_prominent_share_all)} |\n")
        return head + body

    def emergency_subset_table():
        head = ("| 时钟小时 | 应急区间 | 光伏高估占比 | 光伏主导占比 | 平均 $g$（kWh） | $g>0$ 占比 | "
                "平均 $a^L$（kWh） | 起点库存中位数（kWh） | 终点库存中位数（kWh） | "
                "起点高于下限的段数 | 库存受限占比 | 功率受限占比 |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        ordered = list(risk.clock_hour) + [h for h in range(24) if h not in set(risk.clock_hour)]
        for hour in ordered:
            row = by_hour.loc[hour]
            if int(row.n_emergency_L75) == 0:
                continue
            body += (f"| {hour:02d} | {int(row.n_emergency_L75)} | {pct(row.share_pv_over_em)} | "
                     f"{pct(row.pv_prominent_share_em)} | {num(row.mean_g_kWh_em, 2)} | "
                     f"{pct(row.share_g_positive_em)} | {num(row.mean_a_load_kWh_em, 2)} | "
                     f"{num(row.median_state_start_kWh_emsoc, 1)} | "
                     f"{num(row.median_state_end_kWh_emsoc, 1)} | "
                     f"{int(row.intervals_start_above_floor_emsoc)} | "
                     f"{pct(row.inventory_limited_share_emsoc)} | "
                     f"{pct(row.power_limited_share_emsoc)} |\n")
        return head + body

    def node_age_table():
        head = ("| 发布节点 | 预报年龄 τ | 对应时钟小时 | 样本段数 | L75 应急电量（kWh） | "
                "L75 应急费（元） | 光伏 MAE（kW） |\n|---|---|---|---:|---:|---:|---:|\n")
        body = ""
        for row in node_age.sort_values(["publication_hour", "age_lower_h"]).itertuples():
            body += (f"| {row.publication_hour:02d}:00 | [{row.age_lower_h},{row.age_upper_h}) | "
                     f"{row.clock_hours} | {int(row.n_intervals_self)} | "
                     f"{num(row.emergency_kWh_self, 2)} | {num(row.emergency_cost_yuan_self, 2)} | "
                     f"{num(row.pv_mae_kW_all, 2)} |\n")
        return head + body

    def merged_age_table():
        head = ("| 预报年龄 τ | 覆盖的时钟小时 | 样本段数 | L75 应急电量（kWh） | L75 应急费（元） | "
                "光伏 MAE（kW） | 负载低估占比 |\n|---|---|---:|---:|---:|---:|---:|\n")
        body = ""
        for row in merged.sort_values("age_lower_h").itertuples():
            body += (f"| [{row.age_lower_h},{row.age_upper_h}) | {row.clock_hours} | "
                     f"{int(row.n_intervals_self)} | {num(row.emergency_kWh_self, 2)} | "
                     f"{num(row.emergency_cost_yuan_self, 2)} | {num(row.pv_mae_kW_all, 2)} | "
                     f"{pct(row.share_load_under_all)} |\n")
        return head + body

    def risk_table():
        head = ("| 选取 | 费用排名 | 风险小时 | L75 应急费（元） | 占全期 | L75 应急电量（kWh） | "
                "应急区间 | 发生天数 | 出现月份数 | 应急子集光伏高估占比 | 平均 $g$（kWh） | "
                "贡献最大 3 日（元） | 三者占该小时 |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n")
        body = ""
        for row in risk.itertuples():
            label = "费用前 3" if row.selection == "top3_by_cost" else "规则补充"
            body += (f"| {label} | {int(row.cost_rank)} | {int(row.clock_hour):02d} | "
                     f"{num(row.emergency_cost_yuan_L75, 2)} | "
                     f"{100.0 * row.emergency_cost_yuan_L75 / total_cost:.2f}% | "
                     f"{num(row.emergency_kWh_L75, 2)} | {int(row.n_emergency_L75)} | "
                     f"{int(row.days_with_emergency_L75)} | {int(row.months_with_emergency_L75)} | "
                     f"{pct(row.share_pv_over_em)} | {num(row.mean_g_kWh_em, 2)} | "
                     f"{row.top_dates} | {num(row.top_dates_share_pct, 1)}% |\n")
        return head + body

    def candidate_table():
        head = ("| 选取 | 风险小时 | 候选发布时刻 | 是否现有节点 | 出现月份 | 光伏高估占比 | "
                "平均 $g$（kWh） | 判定 | 说明 |\n|---|---|---|---:|---:|---:|---:|---|---|\n")
        body = ""
        for row in candidates.itertuples():
            label = "费用前 3" if row.selection == "top3_by_cost" else "规则补充"
            body += (f"| {label} | {int(row.risk_clock_hour):02d} | "
                     f"{int(row.candidate_publication_clock):02d}:00 | "
                     f"{'是' if row.candidate_is_existing_node else '否'} | "
                     f"{int(row.months_with_emergency)}/11 | "
                     f"{pct(row.emergency_subset_pv_over_share)} | "
                     f"{num(row.emergency_subset_mean_g_kWh, 2)} | {row.verdict} | {row.reason} |\n")
        return head + body

    def month_hour_matrix():
        pivot = monthly_hourly.pivot(index="clock_hour", columns="month",
                                     values="emergency_cost_yuan_L75")
        months = list(pivot.columns)
        head = ("| 时钟小时 | " + " | ".join(m[2:] for m in months) + " |\n"
                "|---|" + "---:|" * len(months) + "\n")
        body = ""
        for hour in risk.clock_hour:
            row = pivot.loc[hour]
            body += (f"| {hour:02d} | " + " | ".join(num(row[m], 0) for m in months) + " |\n")
        return head + body

    month_matrix = month_hour_matrix()

    text = f"""# 问题三新增预报时刻必要性分析结果

日期：2026-09-12。状态：只读描述性诊断已完成并通过三项轻量检查；本报告只读渲染，未新增任何求解。
本版已按 [新增预报时刻分析接收核查]({REVIEW_MD.as_posix()}) 修正五处（见第 8 节更正记录）。
依据方案：[问题三_新增预报时刻必要性轻量分析方案]({PLAN_MD.as_posix()})；
输入身份见[分位费用实验接收核查]({COST_REVIEW.as_posix()})与[四组费用实验结果]({COST_REPORT.as_posix()})。
诊断层：`code/q3_forecast_timing_diagnostic.py`（登记签名 `{registration['signature'][:16]}`，0 训练 / 0 求解）；
报告层：`code/q3_forecast_timing_report.py`（改文案与图表不触发诊断层）。

## 1. 结论

1. **应急分布（本轮可接收的主要统计）**：傍晚 19—23 时占 L75 全期应急费
   {100.0 * evening_cost / total_cost:.1f}%（{num(evening_energy, 1)} kWh），上午 6—10 时占
   {100.0 * morning_cost / total_cost:.1f}%（{num(morning_energy, 1)} kWh）；评价期
   （2025-02-01—2025-12-31）内其余 **{len(zero_hours)} 个时钟小时**（1—5 时与 11—18 时）应急为 0。
   单小时最高是 **20 时，占 {100.0 * by_hour.loc[20, 'emergency_cost_yuan_L75'] / total_cost:.1f}%**。
2. **本轮不排除任何新增预报时刻，也无法确认新增的必要性或经济收益。** 按费用取前 3 热点只是为了限制报告
   规模；9 时（费用第 4）同样满足“应急子集光伏高估占比 ≥{thresholds['pv_over_share_min']:.0%}、
   平均 $g>0$、出现月份 ≥{thresholds['months_with_emergency_min']}”，已补入第 5.4 节。
   可以提出的方向是 **8:00—9:00 面向 9—10 时风险的数据补充**（沿用“风险小时提前 1 小时”的未验证约定），
   但**没有该时刻的真实预报与匹配调度结果，其必要性与节费效果尚未验证**。
   这两个阈值是本轮新增、**不是事先登记**的，只用于组织表格，不构成对研究方向的否决。
3. **傍晚段的伴随现象与因果边界**：该时段当期光伏误差很小（19—23 时实测光伏电量合计
   {num(evening_pv, 3)} kWh、其中 21—23 时严格为零），负载低估、保护后净需求低估与储能受限是**明显的
   伴随现象**。但本轮**没有隔离各因素的因果贡献**：储能具有跨时段耦合，较早的光伏预测可能通过购电与充放电
   改变傍晚库存；库存或功率受限也不排除改善预测后经普通购电减少应急的可能。因此“优先改负载、保护或储能”
   只能作为**待研究方向**，尚未证明优于新增预报；中午没有应急同样不能推出中午预报对后续库存没有价值。
4. **上午段是值得补充数据检验的方向，但覆盖与集中度都有限**：10 时应急子集光伏高估占比
   {pct(hour10.share_pv_over_em)}、平均 $g$ {num(hour10.mean_g_kWh_em, 2)} kWh，但只出现在
   {int(hour10.months_with_emergency_L75)}/11 个月、且 {num(hour10.top_dates_share_pct, 1)}% 的费用来自
   2025-12-01/02/03 三天——应表述为“**存在跨月重复，但月份覆盖有限且费用集中于少数日期**”；
   9 时重复性更好（{int(hour9.months_with_emergency_L75)}/11 个月、光伏高估占比 {pct(hour9.share_pv_over_em)}、
   平均 $g$ {num(hour9.mean_g_kWh_em, 2)} kWh），但前 3 日仍占该小时 {num(hour9.top_dates_share_pct, 1)}%。
   两处都不能据现有诊断断定新增时刻的经济收益。
5. **“预报年龄”的不可辨识性只限本轮分组**：本轮每个实际区间只取最新发布版本，因此
   （发布节点，年龄箱）与 24 个时钟小时**一一对应**，**仅凭本轮分组**无法隔离预报年龄对实际应急的因果影响。
   原始档案保留了同一目标的多个发布版本，前轮 D2 已做过相同未来目标的新旧预测比较；但多版本误差比较
   **不等于**新增时刻的信息价值，也不能替代“同刻只用旧预报”与“同刻用新预报”的受控调度对照。
6. **口径提醒**：晚间实测光伏并非严格为零（19 时 {int(by_hour.loc[19, 'pv_positive_intervals'])} 段为正、
   {num(by_hour.loc[19, 'actual_pv_energy_kWh'], 3)} kWh；20 时 {int(by_hour.loc[20, 'pv_positive_intervals'])} 段、
   {num(by_hour.loc[20, 'actual_pv_energy_kWh'], 3)} kWh；21—23 时为 0）；午夜继承段占 L75 应急费
   {midnight.loc['L75', 'cost_share_pct']:.2f}%，不主导全期。

## 2. 问题分析

题目只提供 0/6/12/18 点四版光伏预报，没有其他时刻的真实预报，也没有新增预报服务的成本。
因此本轮只能回答“是否需要引入其他时刻预报”的**方向性**问题，最多形成“建议优先补充哪个时窗的预报及其依据”，
**不能计算新增时刻的实际节费、投资回报或最优更新时间**。应急发生本身不等于新增光伏信息有用；
反过来，应急与光伏误差同时出现也不能证明新增预报一定改善调度。

观察对象为当前费用首选候选 L75（Linear 预测 + q75 保护 + 四节点滚动 MILP 与实际 SOC 反馈），
以 L80（同预测 + q80 保护）为对照。两者共用同一套点预测，只有保护量与真实轨迹不同。

## 3. 数据预处理

| 项目 | 口径 |
|---|---|
| 实际现金区间 | 2025-02-01 00:00 — 2025-12-31 23:50，共 48096 段 |
| 末模板尾段 | 2026-01-01 00:00—00:10，本轮排除 |
| 非午夜预测配对 | {samples['L75']['paired_non_midnight']} 段（每个实际区间每策略一行） |
| 午夜继承段 | {samples['L75']['midnight_carry']} 段，现金与应急保留在小时/月份合计，但不进入最新预报配对与 τ 诊断 |
| 时钟映射 | 非午夜：`v=floor(小时/6)`、`h=6*小时+分钟/10`、`τ=(h−36v)/6` 小时 |
| 所用预报 | `pv_linear[k, v, h]`、`issued_load[k, h]`（两个策略相同）与该组自己的 `rho`（L75 用 q75、L80 用 q80） |

**最新预报与实际有效购电计划是两个不同的东西**：即使新预报没有让购电计划被接受，它仍属于当时可得的信息，
本轮一律用“当时最新已发布”的预报评价误差；不因为有效计划来自更早版本就把旧预报冒充为最新预报，
也不声称最新预测直接决定了所有有效购电量。不会用 2025 年真值回填、也不用后续版本冒充提前发布的信息。

数据规则：两账本时间戳一对一、每 10 分钟连续、实际供需与价格相同（net/price/load/pv 四项最大差
{num(samples['ledger_delta_net_kWh_max_abs'], 1)}，容差 {samples['ledger_alignment_tolerance']['atol']:.0e}，rtol=0）；
缺失不补零；不删除异常日、不平滑应急峰值；不重新插值。

## 4. 诊断模型

对区间 $t$（$\\Delta t=1/6$ 小时）：实际应急 $e_t$、应急费 $f_t=5p_te_t$；
$E_A=\\sum_{{t\\in A}}e_t$、$F_A=\\sum_{{t\\in A}}f_t$、
$R_A=\\#\\{{t\\in A:e_t>10^{{-6}}\\}}/\\#A$。分组为 24 个时钟小时、11 个月 × 24 小时、
四个发布节点 × 6 个年龄箱 $[0,1),\\ldots,[5,6)$，午夜继承单列。

误差分解（$\\widehat L,\\widehat V$ 为当时最新预测，kW）：

$$a^L_t=(L_t-\\widehat L_t)\\Delta t,\\quad a^V_t=(\\widehat V_t-V_t)\\Delta t,\\quad
\\varepsilon_t=a^L_t+a^V_t=n_t-\\widehat n_t,\\quad g_t=\\varepsilon_t-\\rho_t .$$

$a^L>0$ 为负载低估、$a^V>0$ 为光伏高估，二者都会抬高未预料净需求；$g>0$ 表示保护曲线仍低估实际净需求。
实际应急还取决于普通购电与电池，因此**不能令 $e=g^+$**。以“$a^V>0$ 且 $a^V>\\max(a^L,0)$”作为
“光伏误差较突出”的描述性指标（下称光伏主导占比），不称其为“光伏造成的应急占比”。

储能伴随状态：$D^{{\\max}}_t=\\min\\{{M,0.9(E_t-1200)\\}}$（$M=5000/6$ kWh，$E_t$ 为**区间起点**内部储电量），
$A_t=(n_t-q^{{\\rm eff}}_t)^+$，现有贪心反馈下 $e_t=(A_t-D^{{\\max}}_t)^+$；
$0.9(E_t-1200)<M-10^{{-6}}$ 记“库存受限”、$>M+10^{{-6}}$ 记“功率受限”、差值在容差内记“接近”，三类互斥完整。
这些是当期状态的限制类型，不是已隔离的应急原因；库存受限**不等于**每段开始时已经放空。

风险小时与候选方向：保留方案规定的“按 L75 小时应急费取前 3”，并**另外列出所有满足描述性规则的其余小时**
（不因费用排名而弃用）。候选发布时刻取 $c=(H-1)\\bmod 24$，是**未验证的**提前 1 小时工程约定；
若 $c$ 已是现有节点则只讨论该节点。规则阈值（光伏高估占比 ≥{thresholds['pv_over_share_min']:.0%}、
平均 $g>0$、出现月份 ≥{thresholds['months_with_emergency_min']}）为本轮新增、非事先登记，
只用于组织表格，不是统计显著性检验，也不是最优选址算法。

## 5. 结果分析

### 5.1 应急的时钟分布

![每时钟小时应急电量与应急费]({(FIG / 'hourly_emergency.png').as_posix()})

{hourly_table()}
L75 与 L80 的应急差合计 {num(gap_total, 2)} 元，其中 20 时贡献
{100.0 * float(top_gap.gap.iloc[0]) / gap_total:.1f}%、9 时 {100.0 * float(top_gap.gap.iloc[1]) / gap_total:.1f}%、
10 时 {100.0 * float(top_gap.gap.iloc[2]) / gap_total:.1f}%；即降低保护水平带来的额外应急也集中在同样的时段。

### 5.2 预报年龄与时钟小时的对应关系（本轮分组）

![发布节点 × 预报年龄]({(FIG / 'node_age_grid.png').as_posix()})

{node_age_table()}
24 个（发布节点，年龄）单元格与 24 个时钟小时一一对应：除午夜单列的那一格（{int(node_age.n_intervals_self.min())} 段 =
2004 − 334）外，每格样本段数恒为 2004，且每格只对应一个时钟小时。“合并年龄”表（跨四个发布节点合并）
看似呈现“年龄越大应急越多”的驼峰，但它只是把 0/6/12/18、1/7/13/19 等四个特定小时平均在一起——每一档
恰好包含一个傍晚小时：

{merged_age_table()}
**因此仅凭本轮“每区间只取最新版本”的分组，无法隔离预报年龄对实际应急的因果影响。**
这不等于“本年数据完全无法研究年龄”：原始档案保留了同一目标的多个发布版本，前轮 D2 做过同目标新旧预测比较，
但多版本误差比较不等于新增时刻的信息价值，也不能替代同刻旧/新预报的受控调度对照。

### 5.3 误差方向与保护后缺口

（下表为全部非午夜配对的逐小时口径；$a^L,a^V,g$ 单位 kWh，正值含义见第 4 节。）

{error_table()}
光伏误差的时钟形态很清晰：21—23 时与 1—2 时的光伏预报误差为 0，且这些时钟小时的实测光伏也为 0
（3—4 时实测仅 0.03—5.16 kW 的零星正值）；上午 6—7 时“光伏高估占比”最高
（{pct(by_hour.loc[6, 'share_pv_over_all'])}—{pct(by_hour.loc[7, 'share_pv_over_all'])}，8 时为
{pct(by_hour.loc[8, 'share_pv_over_all'])}，不在该范围内）；正午 11—12 时光伏 MAE 最大
（{num(by_hour.loc[11, 'pv_mae_kW_all'], 1)} kW）却转为低估（平均 $a^V$ = {num(by_hour.loc[11, 'mean_a_pv_kWh_all'], 2)} kWh）。

应急子集（只含 L75 实际发生应急的区间；前 3 行为费用前 3、第 4 行为规则补充，其余按小时升序；
00 时的应急区间全部来自午夜继承段、不进入预报配对故比例为“—”）：

{emergency_subset_table()}
两段的伴随证据不同：**傍晚 19—23 时当期光伏误差很小**（光伏高估占比 0—{pct(by_hour.loc[19, 'share_pv_over_em'])}、
光伏主导占比 0），而**上午 6—10 时既含明显光伏误差（{pct(by_hour.loc[7, 'share_pv_over_em'])}—
{pct(by_hour.loc[6, 'share_pv_over_em'])} 高估）又含库存/功率受限**。表中风险小时的 $g>0$ 占比都在 85% 以上，
说明保护后的净需求曲线在这些时段仍在低估实际需求——这是**伴随现象**，不是已隔离的原因。
“起点高于下限的段数”说明库存受限不等于起点已放空：10 时 36 段中仍有
{int(by_hour.loc[10, 'intervals_start_above_floor_emsoc'])} 段起点高于 1200 kWh，9 时 40 段中有
{int(by_hour.loc[9, 'intervals_start_above_floor_emsoc'])} 段，21 时 113 段中有
{int(by_hour.loc[21, 'intervals_start_above_floor_emsoc'])} 段。

### 5.4 风险时窗与候选方向

对 L75 应急费最高的 3 个小时，**加上所有满足描述性规则的其余小时**（本轮为
{", ".join(f"{h:02d} 时" for h in extra_hours)}）：

{risk_table()}
风险小时的逐月应急费（L75，元）：

{month_matrix}
候选方向判定：

{candidate_table()}
判定汇总：费用前 3 中现有节点复核 {int((candidates[candidates.selection == 'top3_by_cost'].verdict == 'existing_node_review').sum())} 个、
月份覆盖有限 {int((candidates.verdict == 'direction_with_limited_month_coverage').sum())} 个、
无光伏特征 {int((candidates.verdict == 'companion_evidence_not_pv').sum())} 个；
规则补充方向 {int((candidates.verdict == 'candidate_direction_for_verification').sum())} 个（8:00—9:00）。
**没有任何小时被排除**：所有行的“是否排除该新增时刻”均为否，费用排名第 4 的 9 时已按规则补入。
本轮能给出的只是“8:00—9:00 值得补充数据检验”这一待验证方向，而不是“新增时刻必要”或“新增时刻能节费”的结论。

### 5.5 午夜继承与口径提醒

午夜继承段（334 段）应急电量 {num(midnight.loc['L75', 'emergency_kWh_self'], 2)} kWh、
应急费 {num(midnight.loc['L75', 'emergency_cost_yuan_self'], 2)} 元（分别占 L75 全期
{midnight.loc['L75', 'energy_share_pct']:.2f}% 与 {midnight.loc['L75', 'cost_share_pct']:.2f}%），
全部为库存受限，起点中位数 1200 kWh。它不主导全期，也不计入“当日最新预报—当期计划”的配对。

## 6. 可用于论文的审慎结论

> 在本评价期（2025 年 2—12 月）的实际运行中，应急并不均匀分布：它集中在傍晚 19—23 时与上午 6—10 时，
> 其余 13 个时钟小时没有应急。傍晚段当期光伏误差很小（实测光伏电量合计约 71 kWh，且 21—23 时为零），
> 负载低估、保护后净需求低估与储能受限是明显的伴随现象，但储能跨时段耦合，本轮并未隔离各因素的因果贡献；
> 上午段与光伏高估同时出现，其中 9 时覆盖 6 个月、10 时覆盖 4 个月，两处的费用都相对集中于少数日期。
> 由于发布网格为 6 小时且本轮每区间只取最新版本，（发布节点，预报年龄）与时钟小时一一对应，
> 无法据此隔离预报年龄的因果影响。**综合来看，尚无新增时刻的实际预报与匹配调度结果，
> 无法确认新增预报时刻的必要性或经济收益；可以提出的只是待验证的数据补充方向（8:00—9:00，面向 9—10 时风险），
> 其价值需要用新增数据与受控调度对照来检验。** 本轮不给出任何节费估计。

## 7. 验证范围与限制

三项轻量检查（详见 `results/q3_forecast_timing_diagnostic/validation.json`）：

- **样本与时间映射**：两账本自然区间 {samples['L75']['natural_intervals']}、
  非午夜配对 {samples['L75']['paired_non_midnight']}、午夜继承 {samples['L75']['midnight_carry']}，
  日期时间一对一且 10 分钟连续；配对行预报字段有限单元 {samples['L75']['paired_forecast_finite_cells']} = 4 × 47762，
  午夜行的预报字段全部为空（{samples['L75']['midnight_forecast_null_cells']} = 3 × 334）；末模板尾段排除。
- **单日映射**：2025-06-21 的 05:50 / 06:00 / 11:50 / 12:00 / 17:50 / 18:00 六个边界，
  版本切换为 0/1/1/2/2/3、发布时间不晚于区间起点、$\\tau$ 为 5.833/0/5.833/0/5.833/0 小时，
  预报值与冻结档案逐位一致（容差 {validation['mapping']['tolerance']['atol']:.0e}，rtol=0）；午夜单列规则同表核对。
- **汇总与恒等式**：24 小时应急电量与费用之和与两组已核总量差
  {checks['L75_hourly_energy_delta_vs_total_kWh']:.1e}（容差 {checks['tolerances']['cash_atol']:.0e} 元）；
  节点年龄组加午夜、合并年龄组加午夜均恰好覆盖全期（残差 0）；
  $\\varepsilon=a^L+a^V$、$g=\\varepsilon-\\rho$、$n=(L-V)\\Delta t$、
  $\\widehat n=(\\widehat L-\\widehat V)\\Delta t$、$e=(A-D^{{\\max}})^+$ 的最大残差分别为
  {checks['L75_error_identity_max_abs_kWh']:.1e}、{checks['L75_g_identity_max_abs_kWh']:.1e}、
  {checks['L75_truth_identity_max_abs_kWh']:.1e}、{checks['L75_net_forecast_identity_max_abs_kWh']:.1e}、
  {checks['L75_emergency_identity_max_abs_kWh']:.1e} kWh；
  储能三类互斥完整：L75 库存受限 {checks['L75_emergency_class_counts']['inventory_limited']} 段、
  功率受限 {checks['L75_emergency_class_counts']['power_limited']} 段、接近
  {checks['L75_emergency_class_counts'].get('close', 0)} 段。

零重跑声明：训练 0、MILP/LP/DP 求解 0、储能回放 0、公共 1 月重跑 0、预测/偏差/q75/q80 重建 0、
更新时刻消融 0、参数扫描 0；读取产物哈希一致（核对 {integrity['checked']} 项，0 项不匹配、0 项缺失）。
L75/L80 账本与冻结输入均未修改。

未做到、也不宣称：

- **描述性且非排除性**：本轮没有合成新时刻预报、没有为新时刻重跑调度，因此**没有任何节费数字**，
  也**不排除任何新增时刻**；某个风险时窗的应急费不是新增预报的信息价值上界——提前调整会同时改变普通费、
  调整费、库存与其他时段费用。
- 应急子集内的费用只是这些记录的费用总额，不是该因素的因果成本贡献，也不是可回收节费；
  储能跨时段耦合意味着“当期受限”不能推出“改善预测无用”。
- 候选发布时刻由固定“提前 1 小时”的工程约定导出，**不是估计出的最优提前量**；
  不能按全年数据为每一天挑选不同更新时间再宣称在线策略。
- 规则阈值是本轮新增、非事先登记；它们只组织表格，不用于否决研究方向。
- 2025 年已用于方法设计，这是对单一已开发年度的描述性分析，不是独立盲测或跨年结论。
- 图件只做程序化完整性检查（尺寸、非空、颜色数），**未做目视验收**；本 Agent 无法查看图像。
- 本轮未做更新时刻消融、未重新训练负载或光伏模型、未改动主策略。

## 8. 更正记录（本轮接收核查后）

按 [新增预报时刻分析接收核查]({REVIEW_MD.as_posix()}) 的五条意见修正，均只改诊断层输出字段与报告文案，
未重跑调度、未改 L75/L80 数值：

1. **候选筛选遗漏**：风险小时表不再只列费用前 3——凡满足描述性规则的小时都列入，并新增
   `selection` / `cost_rank` 列区分；9 时（费用第 4，{int(hour9.months_with_emergency_L75)} 个月、光伏高估占比
   {pct(hour9.share_pv_over_em)}、平均 $g$ {num(hour9.mean_g_kWh_em, 2)} kWh、前三日占
   {num(hour9.top_dates_share_pct, 1)}%）已补入；结论改为“无法确认新增的必要性或经济收益，
   8:00—9:00 为待验证的数据补充方向”，并明确两个阈值非事先登记、不用于排除方向；
   10 时由“跨月重复不成立”改为“存在跨月重复，但月份覆盖有限且费用集中于少数日期”。
2. **储能与当期误差不作因果排除**：删除“傍晚风险与光伏信息无关”“能量不足而非信息不足”的判定式表述，
   改为伴随现象 + 因果未隔离 + 储能跨时段耦合 + “待研究方向”，并补上“中午没有应急不能推出中午预报
   对后续库存没有价值”。
3. **晚间实测光伏与计数**：19—23 时不再写成严格为零，改为列出实测光伏为正的区间数与电量
   （19 时 {int(by_hour.loc[19, 'pv_positive_intervals'])} 段 / {num(by_hour.loc[19, 'actual_pv_energy_kWh'], 3)} kWh，
   20 时 {int(by_hour.loc[20, 'pv_positive_intervals'])} 段 / {num(by_hour.loc[20, 'actual_pv_energy_kWh'], 3)} kWh，
   21—23 时为 0）；零应急小时数由“12 个”更正为 **{len(zero_hours)} 个**；“全年”改为“评价期
   （2025-02-01—12-31）”；第 5.3 节“6—8 时占比最高 91.8%—92.2%”更正为“6—7 时”，并注明 8 时为
   {pct(by_hour.loc[8, 'share_pv_over_all'])}。
4. **SOC 标签**：表头由“期末库存中位数”更正为“**起点库存中位数**”（限制类型按区间起点定义），
   并新增“终点库存中位数”与“起点高于下限的段数”两列，说明库存受限不等于起点已放空。
5. **年龄不可辨识性的范围**：由“不能用本年数据回答”收窄为“仅凭本轮最新版本分组无法隔离年龄的因果影响”，
   并指出原始档案保留同一目标的多个发布版本（前轮 D2 已比较过同目标新旧预测），但多版本误差比较不等于
   新增时刻的信息价值，也不能替代同刻旧/新预报的受控调度对照。

## 9. 产物

- 诊断层：`code/q3_forecast_timing_diagnostic.py`；报告层：`code/q3_forecast_timing_report.py`。
- 结果：`results/q3_forecast_timing_diagnostic/`（`interval_diagnostic.csv`、`hourly_summary.csv`、
  `monthly_hourly.csv`、`age_summary.csv`、`risk_window_evidence.csv`、`candidate_windows.csv`、
  `validation.json`、`registration.json`、`run_manifest.json`）。
- 图件：`figures/q3_forecast_timing_diagnostic/`（2 张）。
- 未修改 L75/L80 账本、冻结输入、问题二文件与正式 `result3.xlsx`；未自动启动后续实验。
"""
    REPORT_MD.write_text(text, encoding="utf-8")
    print(f"report written: {REPORT_MD.relative_to(ROOT)} ({len(text.splitlines())} lines)")
    print(f"figures: {[p.name for p in paths]}")
    print(json.dumps({"integrity": integrity, "reported_hours": [int(h) for h in risk.clock_hour],
                      "verdicts": list(candidates.verdict)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
