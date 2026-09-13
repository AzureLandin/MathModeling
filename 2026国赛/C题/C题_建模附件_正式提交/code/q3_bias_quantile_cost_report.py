#!/usr/bin/env python
"""问题三 偏差校正 × 分位水平 四组费用对照实验 —— 报告层（只读渲染）。

只读 ``results/q3_bias_quantile_cost/`` 与 ``reports/问题三/问题三_偏差校正与分位水平四组费用实验方案.md``，
渲染结果报告与最多两张图。**不生成保护档案、不调用求解器、不重跑任何轨迹**；改文案或图表
不触发 ``code/q3_bias_quantile_cost_experiment.py``。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_bias_quantile_cost_report.py
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
OUT = ROOT / "results/q3_bias_quantile_cost"
FIG = ROOT / "figures/q3_bias_quantile_cost"
REPORT_MD = ROOT / "reports/问题三/问题三_偏差校正与分位水平费用实验结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_偏差校正与分位水平四组费用实验方案.md"

assert Path(sys.prefix).name == "math_modeling", sys.prefix

STRATEGIES = ("L80", "L75", "C80", "C75")
LABEL = {"L80": "L80（Linear+q80，复用）", "L75": "L75（Linear+q75）",
         "C80": "C80（校正+q80）", "C75": "C75（校正+q75）"}
COLOUR = {"L80": "#4C72B0", "L75": "#8FB3D9", "C80": "#DD8452", "C75": "#F0B27A"}
CASH_TOL = 1e-4
ENERGY_TOL = 1e-6
# revision window lengths by update hour: 6/12/18 revise 109/73/37 remaining slots
REVISION_WINDOW_LENGTH = {6: 109, 12: 73, 18: 37}


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def num(value, digits=2):
    return f"{float(value):,.{digits}f}"


def make_cost_figure(summary):
    order = list(STRATEGIES)
    rows = summary.set_index("strategy_id").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    ax = axes[0]
    x = np.arange(len(order))
    ordinary = rows.ordinary_cost_yuan.to_numpy() / 1e4
    adjustment = rows.adjustment_cost_yuan.to_numpy() / 1e4
    emergency = rows.emergency_cost_yuan.to_numpy() / 1e4
    ax.bar(x, ordinary, 0.55, label="普通购电费", color="#4C72B0")
    ax.bar(x, adjustment, 0.55, bottom=ordinary, label="调整费", color="#55A868")
    ax.bar(x, emergency, 0.55, bottom=ordinary + adjustment, label="应急费", color="#C44E52")
    for i, total in enumerate(rows.natural_total_yuan.to_numpy()):
        share = 100.0 * rows.emergency_cost_yuan.to_numpy()[i] / total
        # two lines above the bar: an in-bar label would land in the 1%-thin emergency band
        ax.text(i, total / 1e4 + 30, f"{total / 1e4:,.1f}", ha="center", fontsize=9)
        ax.text(i, total / 1e4 + 8, f"应急 {emergency[i]:,.2f} 万元（{share:.1f}%）", ha="center",
                fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[s] for s in order], fontsize=8)
    ax.set_ylabel("自然日费用（万元）")
    ax.set_ylim(0, float((rows.natural_total_yuan / 1e4).max()) * 1.14)
    ax.set_title("四组现金总费与三项分项")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    width = 0.38
    ax.bar(x - width / 2, rows.emergency_kWh.to_numpy(), width, label="应急电量（kWh）",
           color="#C44E52")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, rows.emergency_events.to_numpy(), width, label="应急事件数（次）",
            color="#8172B3")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[s] for s in order], fontsize=8)
    ax.set_ylabel("应急电量（kWh）", color="#C44E52")
    ax2.set_ylabel("应急事件数（次）", color="#8172B3")
    ax.set_title("并列风险指标：应急电量与事件数（L80 为读复用）")
    ax.grid(axis="y", alpha=0.3)
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle("L75 最省但应急最高；C80 既不节费也不降应急")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIG / "four_group_costs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_monthly_figure(monthly_contrasts):
    block = monthly_contrasts.sort_values("month")
    x = np.arange(len(block))
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), sharex=True)
    ax = axes[0]
    ax.bar(x - 0.2, block.d_quantile_L / 1e4, 0.4, label="L75−L80（分位）", color="#4C72B0")
    ax.bar(x + 0.2, block.d_bias_80 / 1e4, 0.4, label="C80−L80（校正）", color="#DD8452")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("月度费用差额（万元）")
    ax.set_title("分月费用差额：分位下降在 11/11 月为负；偏差校正在 10/11 月为正")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(x - 0.2, block.em_d_quantile_L, 0.4, label="Δ应急电量 L75−L80", color="#4C72B0")
    ax.bar(x + 0.2, block.em_d_bias_80, 0.4, label="Δ应急电量 C80−L80", color="#DD8452")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[2:] for m in block.month], rotation=45, ha="right")
    ax.set_ylabel("月度应急电量差额（kWh）")
    ax.set_title("同月应急电量变化：降分位抬升应急，偏差校正方向不一")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIG / "monthly_contrasts.png"
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
    """Read-only: the saved artifacts must still match the hashes the run recorded."""
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
    summary = pd.read_csv(OUT / "summary.csv")
    contrasts = pd.read_csv(OUT / "contrasts.csv")
    monthly_long = pd.read_csv(OUT / "monthly.csv")
    monthly_contrasts = pd.read_csv(OUT / "monthly_contrasts.csv")
    daily_long = pd.read_csv(OUT / "daily.csv")
    daily_contrasts = pd.read_csv(OUT / "daily_contrasts.csv")
    selected = pd.read_csv(OUT / "selected_dates.csv")
    selected_contrasts = pd.read_csv(OUT / "selected_dates_contrasts.csv")
    validation = load_json("validation.json")
    registration = load_json("registration.json")
    manifest = load_json("run_manifest.json")
    checks = load_json("checks.json")
    with np.load(OUT / "protection_q75.npz") as archive:
        protection = {key: archive[key] for key in archive.files}

    paths = [make_cost_figure(summary), make_monthly_figure(monthly_contrasts)]
    (OUT / "figure_integrity.json").write_text(
        json.dumps({p.name: figure_integrity(p) for p in paths}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    integrity = manifest_integrity(manifest)
    assert not integrity["mismatched"] and not integrity["missing"], integrity

    totals = summary.set_index("strategy_id").natural_total_yuan
    order = list(totals.sort_values().index)
    best, worst = order[0], order[-1]
    c = contrasts.set_index("contrast")
    bias80 = c.loc["bias_80"]
    bias75 = c.loc["bias_75"]
    ql = c.loc["quantile_L"]
    qc = c.loc["quantile_C"]
    comb = c.loc["combined_75_vs_80"]
    inter = float(c.loc["interaction_bias_x_quantile", "delta_yuan"])
    base = float(totals["L80"])
    nu = float(summary.inventory_value_coefficient_yuan_per_kWh.iloc[0])
    rows = summary.set_index("strategy_id")

    monthly_better = {}
    daily_better = {}
    for name in ("d_quantile_L", "d_quantile_C", "d_bias_80", "d_bias_75",
                 "d_combined_75_vs_80"):
        monthly_better[name] = int((monthly_contrasts[name] < -CASH_TOL).sum())
        daily_better[name] = (int((daily_contrasts[name] < -CASH_TOL).sum()),
                              int((daily_contrasts[name] > CASH_TOL).sum()))
    top_days = {}
    for name in ("d_quantile_L", "d_bias_80", "d_combined_75_vs_80"):
        series = daily_contrasts[name]
        picked = daily_contrasts.reindex(series.abs().sort_values(ascending=False).index).head(5)
        top_days[name] = dict(
            days=[(str(d), float(v)) for d, v in zip(picked.date, picked[name])],
            sum_of_five=float(picked[name].sum()),
            remainder=float(series.sum() - picked[name].sum()),
            costliest=(str(daily_contrasts.loc[series.idxmax(), "date"]), float(series.max())),
            cheapest=(str(daily_contrasts.loc[series.idxmin(), "date"]), float(series.min())))
    rejected = {}
    for sid, path in (("L80", ROOT / "results/q3_rolling_baseline/B2_revision_decisions.csv"),
                      ("L75", OUT / "L75/revision_decisions.csv"),
                      ("C80", OUT / "C80/revision_decisions.csv"),
                      ("C75", OUT / "C75/revision_decisions.csv")):
        decisions = pd.read_csv(path)
        accepted = decisions[decisions.accepted]
        denied = decisions[~decisions.accepted]
        hours = sorted(int(h) for h in denied.update_hour.unique())
        segments = int(sum(REVISION_WINDOW_LENGTH[h] for h in hours)) if hours else 0
        rejected[sid] = dict(
            decisions=int(len(decisions)), accepted=int(len(accepted)), rejected=int(len(denied)),
            rejected_hours=hours, rejected_window_segments=segments,
            max_rejected_revision_kWh=float(denied.revision_kWh.max()),
            max_rejected_max_per_segment=(float(denied.revision_kWh.max()) / segments
                                          if segments else float("nan")),
            max_rejected_date=str(denied.loc[denied.revision_kWh.idxmax(), "date"]),
            total_rejected_revision_kWh=float(denied.revision_kWh.sum()),
            min_accepted_revision_kWh=float(accepted.revision_kWh.min()))
    min_accepted = min(v["min_accepted_revision_kWh"] for v in rejected.values())
    max_rejected = max(v["max_rejected_revision_kWh"] for v in rejected.values())
    rejected_segments = max(v["rejected_window_segments"] for v in rejected.values())
    saved = validation["saved_results"]
    quant = validation["quantile_inputs"]
    adapt = validation["adaptation"]

    def cost_table():
        head = ("| 组别 | 光伏预测 | 分位 | 普通费 | 调整费 | 应急费 | 现金总费 | 与 L80 差 |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for sid in STRATEGIES:
            row = rows.loc[sid]
            delta = float(row.natural_total_yuan) - base
            body += (f"| {sid} | {row.predictor} | {row.alpha:.2f} | "
                     f"{num(row.ordinary_cost_yuan)} | {num(row.adjustment_cost_yuan)} | "
                     f"{num(row.emergency_cost_yuan)} | {num(row.natural_total_yuan)} | "
                     f"{'+' if delta >= 0 else ''}{num(delta, 2)} |\n")
        return head + body

    def contrast_table():
        head = ("| 差额 | 定义 | 费用差（元） | 相对基准 | 普通费差 | 调整费差 | 应急费差 | "
                "应急电量差（kWh） |\n|---|---|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for name, label in (("bias_80", "C80−L80（偏差校正，q80）"),
                            ("bias_75", "C75−L75（偏差校正，q75）"),
                            ("quantile_L", "L75−L80（降低分位，Linear）"),
                            ("quantile_C", "C75−C80（降低分位，校正预测）"),
                            ("combined_75_vs_80", "C75−L80（组合）")):
            row = c.loc[name]
            body += (f"| {name} | {label} | {'+' if row.delta_yuan >= 0 else ''}"
                     f"{num(row.delta_yuan)} | {row.delta_pct:+.4f}% | "
                     f"{'+' if row.delta_ordinary_yuan >= 0 else ''}{num(row.delta_ordinary_yuan)} | "
                     f"{'+' if row.delta_adjustment_yuan >= 0 else ''}{num(row.delta_adjustment_yuan)} | "
                     f"{'+' if row.delta_emergency_yuan >= 0 else ''}{num(row.delta_emergency_yuan)} | "
                     f"{'+' if row.delta_emergency_kWh >= 0 else ''}{num(row.delta_emergency_kWh, 1)} |\n")
        body += (f"| interaction | (C75−C80)−(L75−L80) | {'+' if inter >= 0 else ''}{num(inter)} | "
                 f"{100.0 * inter / base:+.4f}% | — | — | — | — |\n")
        return head + body

    def risk_table():
        head = ("| 组别 | 应急电量（kWh） | 应急区间数 | 应急天数 | 应急事件数 | 未使用电量（kWh） | "
                "充电量（kWh） | 放电量（kWh） | 损耗（kWh） | 自然年末库存（kWh） | "
                "末模板库存（kWh） |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for sid in STRATEGIES:
            row = rows.loc[sid]
            body += (f"| {sid} | {num(row.emergency_kWh, 1)} | {int(row.emergency_intervals)} | "
                     f"{int(row.emergency_days)} | {int(row.emergency_events)} | "
                     f"{num(row.unused_kWh, 1)} | {num(row.charge_kWh, 1)} | "
                     f"{num(row.discharge_kWh, 1)} | {num(row.loss_kWh, 1)} | "
                     f"{num(row.final_natural_state_kWh, 1)} | "
                     f"{num(row.final_template_state_kWh, 1)} |\n")
        return head + body

    def revision_table():
        head = ("| 组别 | 正式求解 | 修订决定 | 接受 | 拒绝 | 06/12/18 点接受次数 | 候选修订量（kWh） | "
                "实际接受修订量（kWh） |\n|---|---:|---:|---:|---:|---|---:|---:|\n")
        body = ""
        for sid in STRATEGIES:
            row = rows.loc[sid]
            body += (f"| {sid} | {int(row.solves_total)} | {int(row.revision_decisions)} | "
                     f"{int(row.revisions_accepted)} | {int(row.revisions_rejected)} | "
                     f"{int(row.accepted_06)}/{int(row.accepted_12)}/{int(row.accepted_18)} | "
                     f"{num(row.candidate_revision_kWh, 1)} | "
                     f"{num(row.accepted_revision_kWh, 1)} |\n")
        return head + body

    def monthly_table():
        head = ("| 月份 | L80 | L75 | C80 | C75 | L75−L80 | C80−L80 | C75−C80 | "
                "Δ应急 L75−L80（kWh） | Δ应急 C80−L80（kWh） |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for row in monthly_contrasts.sort_values("month").itertuples():
            body += (f"| {row.month} | {num(row.L80, 0)} | {num(row.L75, 0)} | {num(row.C80, 0)} | "
                     f"{num(row.C75, 0)} | {row.d_quantile_L:+,.0f} | {row.d_bias_80:+,.0f} | "
                     f"{row.d_quantile_C:+,.0f} | {row.em_d_quantile_L:+,.1f} | "
                     f"{row.em_d_bias_80:+,.1f} |\n")
        return head + body

    def selected_table():
        head = ("| 日期 | L80 | L75 | C80 | C75 | L75−L80 | C80−L80 |\n"
                "|---|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for row in selected_contrasts.sort_values("date").itertuples():
            body += (f"| {row.date} | {num(row.L80, 0)} | {num(row.L75, 0)} | {num(row.C80, 0)} | "
                     f"{num(row.C75, 0)} | {row.d_quantile_L:+,.0f} | {row.d_bias_80:+,.0f} |\n")
        return head + body

    def spot_table():
        head = ("| 发布版本 | 目标 h | 有效历史数 m | q75 升序位置 | q80 升序位置 | "
                "q80 档案值（kWh） | 重算 q80（kWh） |\n|---|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for row in quant["spot_check"]:
            body += (f"| {row['publication_hour']:02d}:00 | {row['target_h']} | {row['m']} | "
                     f"{row['q75_linear_position']} | {row['q80_position']} | "
                     f"{row['q80_archive_value_kWh']:.6f} | {row['q80_recomputed_value_kWh']:.6f} |\n")
        return head + body

    def top_days_table():
        head = ("| 差额 | 绝对贡献最大的 5 日 | 5 日合计（元） | 其余日期合计（元） | 单日最贵 | "
                "单日最省 |\n|---|---|---:|---:|---|---|\n")
        labels = {"d_quantile_L": "L75−L80", "d_bias_80": "C80−L80",
                  "d_combined_75_vs_80": "C75−L80"}
        body = ""
        for name, label in labels.items():
            block = top_days[name]
            days = "；".join(f"{d}({v:+,.0f})" for d, v in block["days"])
            body += (f"| {label} | {days} | {block['sum_of_five']:+,.0f} | "
                     f"{block['remainder']:+,.0f} | {block['costliest'][0]} "
                     f"({block['costliest'][1]:+,.0f}) | {block['cheapest'][0]} "
                     f"({block['cheapest'][1]:+,.0f}) |\n")
        return head + body

    def rejected_table():
        head = ("| 组别 | 拒绝次数 | 拒绝发生时刻 | 该窗口段数 | 总修订量最大日 | 该日总修订量（kWh） | "
                "全部被拒总量（kWh） |\n|---|---:|---|---:|---|---:|---:|\n")
        body = ""
        for sid in STRATEGIES:
            block = rejected[sid]
            hours = "/".join(f"{h:02d}:00" for h in block["rejected_hours"])
            body += (f"| {sid} | {block['rejected']} | {hours} | "
                     f"{block['rejected_window_segments']} | {block['max_rejected_date']} | "
                     f"{block['max_rejected_revision_kWh']:.3e} | "
                     f"{block['total_rejected_revision_kWh']:.3e} |\n")
        return head + body

    def saved_table():
        head = ("| 组别 | 求解日志 | 修订决定 | 接受不等式反例 | 修订后前 36 段有效计划比对段数 | "
                "比对不一致 | 现金重算最大偏差（元） | 供需平衡最大残差（kWh） |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|\n")
        body = ""
        for sid in STRATEGIES:
            block = saved[sid]
            body += (f"| {sid} | {block['solves_total']} | {block['revision_decisions']} | "
                     f"{block['accept_inequality_mismatches']} | "
                     f"{block['accepted_candidate_slots_compared']} | "
                     f"{block['accepted_candidate_execution_mismatches']} | "
                     f"{block['adjustment_recompute_max_abs_yuan']:.2e} | "
                     f"{block['bus_balance_max_abs_kWh']:.2e} |\n")
        return head + body

    text = f"""# 问题三偏差校正与分位水平费用实验结果

日期：2026-09-12。状态：四组固定对照已跑完并通过三类精简检查；本报告只读渲染，未重跑任何轨迹。
依据方案：[问题三_偏差校正与分位水平四组费用实验方案]({PLAN_MD.as_posix()})。
计算层：`code/q3_bias_quantile_cost_experiment.py`（登记签名 `{registration['signature'][:16]}`）；
报告层：`code/q3_bias_quantile_cost_report.py`（改文案与图表不触发计算层）。

## 1. 结论

四个候选的自然日现金总费排名（低→高）为
**{' < '.join(f'{s} {num(totals[s])}' for s in order)}** 元。

1. **本轮最低的是 {best}，比控制组 L80 低 {num(-(totals[best] - base))} 元（{100.0 * (totals[best] - base) / base:+.4f}%）。**
2. **偏差校正在两个分位水平上都不节费**：C80 比 L80 贵 {num(bias80.delta_yuan)} 元
   （{bias80.delta_pct:+.4f}%），C75 比 L75 贵 {num(bias75.delta_yuan)} 元
   （{bias75.delta_pct:+.4f}%）。因此 C75 虽然低于 L80，但**高于 L75**，本轮不能宣称
   偏差校正带来了额外节费；C80 未节费也不支持“校正与降分位配合才节费”的说法。
3. **交互项 (C75−C80)−(L75−L80) = {num(inter)} 元**，约为 Linear 降分位节费的
   {100.0 * inter / abs(float(ql.delta_yuan)):.2f}%（{100.0 * inter / base:+.4f}% 的基准费用），
   相对降分位主效应较小。本轮**没有预设经济实质阈值或统计显著性检验**，因此只描述相对量级，
   不宣称已证明“交互无实质影响”；同一 0.75 分位在校正预测上少省
   {num(qc.delta_yuan - ql.delta_yuan)} 元。
4. **节费的来源是分位水平，代价是风险**：L75 的普通购电费下降 {num(-ql.delta_ordinary_yuan)} 元、
   调整费下降 {num(-ql.delta_adjustment_yuan)} 元，但应急费上升 {num(ql.delta_emergency_yuan)} 元，
   应急电量由 {num(rows.loc['L80', 'emergency_kWh'], 1)} 升到 {num(rows.loc['L75', 'emergency_kWh'], 1)} kWh
   （{100.0 * ql.delta_emergency_kWh / rows.loc['L80', 'emergency_kWh']:+.2f}%），
   应急事件由 {int(rows.loc['L80', 'emergency_events'])} 次增到 {int(rows.loc['L75', 'emergency_events'])} 次，
   应急天数由 {int(rows.loc['L80', 'emergency_days'])} 天增到 {int(rows.loc['L75', 'emergency_days'])} 天。
   这是“费用更低但应急更多”的结果，不隐去风险，也不事后添加惩罚权重重新宣布胜出。
5. **建议：把 L75 作为候选提交给用户与主Agent接收，本轮不自动替换 B2、不修改正式 `result3.xlsx`。**
   q75 是固定的单个较低保护候选，不是已知最优参数；{100.0 * (totals[best] - base) / base:+.2f}%
   的差额来自已用于方法设计的 2025 年数据，不是独立盲测下的稳健收益。
6. L75 的优势不是由少数异常日造成——月度 {monthly_better['d_quantile_L']}/11 个月为负、
   逐日 {daily_better['d_quantile_L'][0]}/{len(daily_contrasts)} 天更省，且绝对贡献最大的 5 日全部反向；
   但应急抬升是明确的反向指标，故本轮只提候选、不自动替换。

## 2. 实验设置

| 组别 | 光伏预测 | 分位水平 | 调度 |
|---|---|---:|---|
| L80 | 冻结 Linear | 0.80 | 只读复用既有 B2 完整账本，不重跑全年 |
| L75 | 与 L80 相同 | 0.75 | 新增 334 日四节点滚动轨迹 |
| C80 | 冻结 Bias28 校正预测 | 0.80 | 复用已重算 C1 q80，首次运行对应 334 日调度 |
| C75 | 与 C80 相同 | 0.75 | 新增 334 日四节点滚动轨迹 |

固定口径：`start_time_v1` 区间起点口径；15 列负载 LightGBM 发布档案（日内不更新）；
同发布小时-同目标日期偏移-同目标时钟的 W28、最少 7 样本次序统计（`q75` 位置：m=27→21、
m=28→21；`q80` 位置：m=27→22、m=28→23，不足 7 样本零修正）；自由末态；
公共 1 月预运行产生的 2025-02-01 00:00 实际初态 6075.795025925926 kWh 与 0 kWh 午夜原始/有效承诺（本月不重跑 1 月，公共 1 月重跑 {int(validation['budget']['public_january_reruns'])} 次；不是从 2025-01-01 的 6000 kWh 重新开始）；
最终一次交付价结算；SciPy/HiGHS、相对 gap 1e-9、120 秒上限。

q75 与 q80 都由**各自预测器**的历史残差重新取舍：C75 不使用 Linear 残差，
也不是 q80 乘系数。检查显示两个预测器的 q75 在 {quant['predictor_isolation']['cells_rho75_corrected_differs_from_linear']:,} 个单元上互不相同，
最大差 {num(quant['predictor_isolation']['max_abs_rho75_corrected_minus_linear_kWh'], 4)} kWh。

## 3. 四组费用、风险与修订行为

{cost_table()}
另有并列风险指标（未使用量不能全部称为弃光，见第 6 节）：

{risk_table()}
修订行为（候选修订量与实际接受修订量分开报告）：

{revision_table()}

被拒绝的候选与旧计划差异很小：

{rejected_table()}
拒绝全部发生在 18:00 修订，窗口为 37 段（不是 109 段）。L80 在 {rejected['L80']['max_rejected_date']}
的最大总绝对修订量为 {rejected['L80']['max_rejected_revision_kWh']:.3e} kWh，除以 37 段约为
{rejected['L80']['max_rejected_max_per_segment']:.2e} kWh/段——这只是**平均**，不是单段最大差，也不构成逐段相等证据。
按规则只能表述为“**拒绝候选与旧计划的总差异很小，接受阈值未通过**”。现有结果**不证明**所有拒绝都来自同一顶点
或数学等价解，因此不能用它论证择优步骤可删除。被接受候选的最小总修订量为 {min_accepted:.2e} kWh。

## 4. 配对差额

{contrast_table()}
差额为负表示前项费用更低。组间变化包含保护、购电与 SOC 反馈的系统效果，不是单时段误差对费用的直接因果分解。

## 5. 分月与逐日稳定性

{monthly_table()}
分月计数（费用差 < −1e-4 元记“更省”）：
L75−L80 在 {monthly_better['d_quantile_L']}/11 个月更省、C75−C80 在 {monthly_better['d_quantile_C']}/11 个月更省；
C80−L80 有 {11 - monthly_better['d_bias_80']}/11 个月为正（仅 2025-06 为负）。
逐日计数：L75−L80 更省 {daily_better['d_quantile_L'][0]} 天、更贵 {daily_better['d_quantile_L'][1]} 天；
C80−L80 更省 {daily_better['d_bias_80'][0]} 天、更贵 {daily_better['d_bias_80'][1]} 天。
月度改善次数不能单独排除少数异常日，故同时给出绝对贡献最大的 5 日与其余日期的合计：

{top_days_table()}
对 L75−L80，绝对贡献最大的 5 日**全部为正**（即反向），5 日合计 {top_days['d_quantile_L']['sum_of_five']:+,.0f} 元，
其余日期合计 {top_days['d_quantile_L']['remainder']:+,.0f} 元，说明整体节费来自大量小额改善日而非少数异常日，
本轮不删除任何日期重跑或修改规则。对 C80−L80，5 日合计 {top_days['d_bias_80']['sum_of_five']:+,.0f} 元，
其中 4 日集中在 12 月初，其余日期合计 {top_days['d_bias_80']['remainder']:+,.0f} 元，方向一致但幅度分散。

给定 4 个指定日期（3 月 20 日、6 月 21 日、9 月 23 日、12 月 21 日）：

{selected_table()}
指定日期上四组互有胜负（例如 3 月 20 日 L75 反而更贵），单日结果不能替代全年比较。

## 6. 库存估值与口径说明

统一辅助估值 ν = median(交付电价)/0.9 = {num(nu, 6)} 元/kWh（仅辅助，非真实现金，也不搜索估值系数）：

| 组别 | 现金总费（元） | 自然年末库存（kWh） | C − ν(E_end − E_start)（元） |
|---|---:|---:|---:|
""" + "".join(
        f"| {sid} | {num(rows.loc[sid, 'natural_total_yuan'])} | "
        f"{num(rows.loc[sid, 'final_natural_state_kWh'], 1)} | "
        f"{num(rows.loc[sid, 'cost_net_of_inventory_yuan'])} |\n" for sid in STRATEGIES
    ) + f"""
L75 比 L80 少留 {num(-ql.delta_final_natural_state_kWh, 1)} kWh 库存，扣除库存估值后差额由
{num(ql.delta_yuan)} 变为 {num(ql.delta_inventory_adjusted_yuan)} 元，方向不变；
C80−L80 的库存差仅 {num(bias80.delta_final_natural_state_kWh, 4)} kWh，估价调整不足 1 元。

口径提醒：应急电量是实际执行后的缺口电量，不是预测短缺；未使用电量在模板口径下包含
“计划购电未被使用”的部分，不能全部称为弃光；保护后预测与保护量不进入本报告的费用口径。

## 7. 验证范围与限制

三类检查的实际范围（详见 `results/q3_bias_quantile_cost/validation.json` 与 `checks.json`）：

- **类一 分位输入检查，0 次 MILP。** 两个 q80 用同一规则重算后与冻结档案**逐段完全一致**
  （最大绝对差 {num(quant['q80_alignment']['c0_rho_max_abs_kWh'], 1)} 与 {num(quant['q80_alignment']['c1_rho_max_abs_kWh'], 1)} kWh，样本数差 0）；
  小数组覆盖 m=6/7/21/27/28 的回退与位置；q75 ≤ q80 在全部
  {quant['monotonicity']['c0_cells_m_ge_7']:,} 个有效样本单元上成立。2025-06-21 的三组抽检目标如下
  （有效历史数、两处分位位置，以及 q80 档案值与重算值的逐位对齐）：

{spot_table()}

- **类二 适配控制检查，2 次 MILP。** 用保存的 L80 输入在 2025-06-21 重解 0:00 计划与 06:00 候选：
  0:00 目标差 {adapt['midnight']['objective_delta_yuan']:.2e} 元、计划最大差
  {adapt['midnight']['plan_max_delta_kWh']:.2e} kWh；06:00 目标差
  {adapt['six']['revision_objective_delta_yuan']:.2e} 元、候选最大差
  {adapt['six']['candidate_max_delta_kWh']:.2e} kWh、接受标记与保存一致
  （{adapt['six']['saved_accepted']}）。
- **类三 保存结果核对，0 次求解。** 四组均从磁盘产物重读，逐项复核物理与现金：

{saved_table()}
该表“修订后前 36 段有效计划比对”一列对每个修订决定比较其窗口前 36 段：接受时比 `candidate`、拒绝时比 `previous`，
因此 {saved['L75']['accepted_candidate_slots_compared']:,} 段是**全部修订决定**（含接受与拒绝两个分支）的比对总数，
不是仅接受候选。18:00 窗口共 37 段，最后一段（次日 00:00—00:10）未进入该项计划来源核对；该尾段已被物理、
收费与首尾桥接检查覆盖，但不等同于此项来源检查。读取清单与运行清单哈希一致（核对 {integrity['checked']} 项，0 项不匹配、0 项缺失）。

求解预算：本脚本实跑 {validation['budget']['formal_milp_solves']} 次正式 MILP（L75/C80/C75 各 1336 次）
加 {validation['budget']['extra_validation_milp']} 次验证 MILP；新训练 0 次、公共 1 月重跑 0 次、
L80 全年重跑 0 次（L80 为只读复用，其 1336 次求解是 B2 既有记录）。

未做到、也不宣称的部分：

- 本轮不是独立盲测：2025 年已用于方法设计，四组是开发数据上的固定滚动因果比较。
- 不是完整独立审计：类三只从磁盘产物复算，未换一套实现路径重算 MILP；类二只覆盖一天一次修订。
- α 全程固定，回测期间不按时段、发布版本或日期切换组别；“事后看全年最低组”不等于部署时已知最优。
- 图件只做程序化完整性检查（尺寸、非空、颜色数），**未做目视验收**；agent 无法查看图像。
- C80 复用上一轮已接受的 C1 保护曲线，本脚本未再次解析该曲线；q75 的位置与样本数由本脚本重新生成。
- 时间与价格口径仍受已约定的区间起点假设 `start_time_v1` 约束（`00:10` 表示 00:10—00:20，前日末段属于次日 00:00—00:10）；题目转写历史上有过漏字，但本轮数值输入链是 `read_attachments` 直接以 openpyxl 读取附件 1/2，MinerU 转写不是本轮数值依赖，两者不可混为一谈。本轮未复核上游转换。

## 8. 建议与后续边界

1. 建议由用户与主Agent在接收本结果后决定是否采纳 L75；本轮不替换正式 B2、不填 `result3.xlsx`。
   应急抬升（+{100.0 * ql.delta_emergency_kWh / rows.loc['L80', 'emergency_kWh']:.1f}% 电量、
   +{int(rows.loc['L75', 'emergency_events'] - rows.loc['L80', 'emergency_events'])} 次事件）是必须并列披露的风险指标：应急已按 5 倍电价计入总费，模型未另设应急额度或次数约束，因此它不自动构成违约、失供或新的不可行条件。按本轮既定费用目标，L75 是首选候选；既不为了挑选 L75 而忽略风险，也不为了保留 L80 而临时增设未定义的风险约束。
2. 不开展 q70/q85、分时分位、偏差校正变种、更新时刻消融或问题四实验；本轮只回答方案中的三个问题。
3. 若要继续提升稳健性，需要的不是扩大分位网格，而是对 {best} 与 L80 的差异做独立复核
   （换路径重算关键日、检查应急事件的时段分布），本轮未做。

## 9. 产物

- 计算层：`code/q3_bias_quantile_cost_experiment.py`；报告层：`code/q3_bias_quantile_cost_report.py`。
- 结果：`results/q3_bias_quantile_cost/`（`summary.csv`、`contrasts.csv`、`monthly*.csv`、`daily*.csv`、
  `selected_dates*.csv`、`protection_q75.npz`、`validation.json`、`checks.json`、`registration.json`、
  `run_manifest.json`，以及 `L75/`、`C80/`、`C75/` 各组 dispatch、plan_versions、
  revision_decisions、solver_log、nominal_trajectory）。
- 图件：`figures/q3_bias_quantile_cost/`（2 张）。
- L80 未复制：直接引用 `results/q3_rolling_baseline/B2_*.csv`，本报告已核对其汇总与保存行一致。
"""
    REPORT_MD.write_text(text, encoding="utf-8")
    print(f"report written: {REPORT_MD.relative_to(ROOT)} ({len(text.splitlines())} lines)")
    print(f"figures: {[p.name for p in paths]}")
    print(json.dumps({"integrity": integrity, "ranking": [str(s) for s in order]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
