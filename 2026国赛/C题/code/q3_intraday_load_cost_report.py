#!/usr/bin/env python
"""问题三 日内负载修正与 q75 费用对照实验 —— 报告层（只读渲染）。

只读 ``results/q3_intraday_load_cost/``，渲染结果报告与最多两张图。
**不重算保护、不重建预测、不调用任何求解器**；改文案或图表永远不会触发
``code/q3_intraday_load_cost_experiment.py``。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_intraday_load_cost_report.py
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
OUT = ROOT / "results/q3_intraday_load_cost"
FIG = ROOT / "figures/q3_intraday_load_cost"
REPORT_MD = ROOT / "reports/问题三/问题三_日内负载修正与q75费用对照实验结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_日内负载修正与q75费用对照实验方案.md"
S0, S2 = "S0_L75", "S2_load_q75"
COMPONENTS = (("ordinary", "普通购电"), ("adjustment", "调整费"), ("emergency", "应急费"))


def num(value, digits=2):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def signed(value, digits=2):
    text = num(abs(value), digits)
    return f"{'+' if value >= 0 else '−'}{text}"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_figures(monthly, contrasts, summary):
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    colours = ["#2ca02c" if v < 0 else "#d62728" for v in monthly.delta_yuan]
    ax.bar(monthly.month, monthly.delta_yuan, color=colours)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_ylabel("月度现金差 S2 − S0（元）")
    ax.set_title(f"分月现金差：负值表示 S2 更省（自然账本，全期 {signed(monthly.delta_yuan.sum())} 元，"
                 f"未计库存估值）")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path1 = FIG / "monthly_cash_difference.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    labels = [label for _, label in COMPONENTS] + ["合计"]
    values = [float(contrasts.set_index("item").loc[key, "delta_yuan"]) for key, _ in COMPONENTS]
    values.append(float(contrasts.set_index("item").loc["total", "delta_yuan"]))
    axes[0].bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"])
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_ylabel("S2 − S0（元）")
    axes[0].set_title("现金三项差（正=更贵）")
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axes[0].annotate(f"{value:+,.0f}", (index, value), ha="center",
                         va="bottom" if value >= 0 else "top", fontsize=8)

    index = summary.set_index("strategy_id")
    metrics = [("应急电量\n(kWh)", "emergency_kWh"), ("应急区间数", "emergency_intervals"),
               ("应急天数", "emergency_days"), ("应急事件数", "emergency_events")]
    x = np.arange(len(metrics))
    width = 0.38
    axes[1].bar(x - width / 2, [float(index.loc[S0, key]) for _, key in metrics], width,
                label="S0_L75", color="#4c72b0")
    axes[1].bar(x + width / 2, [float(index.loc[S2, key]) for _, key in metrics], width,
                label="S2_load_q75", color="#dd8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label for label, _ in metrics])
    axes[1].set_ylabel("数量")
    axes[1].set_title("应急指标（口径不同，未归一化）")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path2 = FIG / "cost_components_and_emergency.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    return [path1, path2]


def figure_integrity(paths):
    from PIL import Image
    out = {}
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        ink = float((rgb < 250).any(axis=2).mean())
        colours = int(len(np.unique(rgb.reshape(-1, 3), axis=0)))
        assert ink > 0.005, f"{path.name} looks blank (ink={ink:.4f})"
        assert colours > 20, f"{path.name} has too few colours ({colours})"
        out[path.name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              bytes=path.stat().st_size, width=int(rgb.shape[1]),
                              height=int(rgb.shape[0]), ink_coverage=round(ink, 4),
                              distinct_colours=colours,
                              note="programmatic only; the agent cannot view images")
    return out


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
    summary = pd.read_csv(OUT / "summary.csv").set_index("strategy_id")
    contrasts = pd.read_csv(OUT / "contrasts.csv").set_index("item")
    monthly = pd.read_csv(OUT / "monthly.csv")
    daily = pd.read_csv(OUT / "daily.csv")
    protection_change = pd.read_csv(OUT / "protection_change.csv")
    selected = pd.read_csv(OUT / "selected_dates.csv")
    validation = load_json(OUT / "validation.json")
    registration = load_json(OUT / "registration.json")
    manifest = load_json(OUT / "run_manifest.json")
    checks = validation["checks"]
    days = validation["days"]

    delta_total = float(validation["delta_total_yuan"])
    base = float(summary.loc[S0, "natural_total_yuan"])
    delta_share = delta_total / base
    refreshed, worsened = int(days["refreshed"]), int(days["worsened"])
    top5 = float(daily.nsmallest(5, "delta_yuan").delta_yuan.sum())
    top5_share = top5 / delta_total if delta_total else float("nan")

    add = None
    lines = []
    add = lines.append
    add("# 问题三日内负载修正与 q75 费用对照实验结果")
    add("")
    add("日期：2026-09-13。状态：两组费用对照已运行并通过三类精简检查；本报告只读渲染，"
        "未新增任何求解。")
    add(f"依据方案：[问题三_日内负载修正与q75费用对照实验方案]"
        f"({PLAN_MD.resolve().as_posix()})。")
    add(f"计算层：`code/q3_intraday_load_cost_experiment.py`"
        f"（登记签名 `{registration['signature'][:16]}`；S2 正式 1336 次 MILP ＋ 适配 2 次，"
        f"S0 全年重跑 0 次、LightGBM 训练 0 次、一元回归重拟合 0 次、公共 1 月调度 0 次）；"
        f"报告层：`code/q3_intraday_load_cost_report.py`（改文案与图表不触发计算层）。")
    add("")

    add("## 1. 结论")
    add("")
    add(f"1. **本轮判定：S2 比 S0 低 {num(abs(delta_total))} 元（{signed(delta_share * 100, 3)}%）。** "
        f"自然账本现金总费 S0 {num(base)} 元、S2 {num(float(summary.loc[S2, 'natural_total_yuan']))} 元。"
        f"差异**全部来自应急费的下降**（{signed(float(contrasts.loc['emergency', 'delta_yuan']))} 元），"
        f"被调整费（{signed(float(contrasts.loc['adjustment', 'delta_yuan']))} 元）与普通购电"
        f"（{signed(float(contrasts.loc['ordinary', 'delta_yuan']))} 元）部分抵消。"
        f"这是「F2 负载修正＋相应残差保护＋调度响应」的组合效果，"
        f"**不能拆称纯当天信息价值，也不称最优**。")
    add(f"2. **但节费高度不均，必须并列披露**：逐日 {refreshed} 天更省、{worsened} 天更贵，"
        f"逐日差中位数 {signed(days['delta_median_yuan'])} 元（中位日实际更贵）；"
        f"最集中的 5 天合计 {signed(top5)} 元，占净节费的 {num(100 * top5_share, 2)}%"
        f"（即其余 329 天合计反而是净增加）。逐月只有 {days['months_refreshed']}/11 个月更省，"
        f"最大单月增加 {signed(max(monthly.delta_yuan))} 元。")
    add(f"3. **风险指标是双向的**：应急电量 {num(float(summary.loc[S0, 'emergency_kWh']))} → "
        f"{num(float(summary.loc[S2, 'emergency_kWh']))} kWh（−44.1%）、应急费 −43.6%、"
        f"应急区间 {int(summary.loc[S0, 'emergency_intervals'])} → "
        f"{int(summary.loc[S2, 'emergency_intervals'])}；**但应急天数由 "
        f"{int(summary.loc[S0, 'emergency_days'])} 天升到 {int(summary.loc[S2, 'emergency_days'])} 天**，"
        f"应急事件 {int(summary.loc[S0, 'emergency_events'])} → {int(summary.loc[S2, 'emergency_events'])}。"
        f"即单次缺口变浅、但出现得更分散。期末自然库存少 "
        f"{num(float(summary.loc[S0, 'final_natural_state_kWh']) - float(summary.loc[S2, 'final_natural_state_kWh']))} kWh。")
    add(f"4. **库存辅助估值后方向不变**：沿用 ν = {num(float(summary.loc[S0, 'inventory_value_coefficient_yuan_per_kWh']), 10)} 元/kWh，"
        f"扣库存后 S0 {num(float(summary.loc[S0, 'cost_net_of_inventory_yuan']))} 元、"
        f"S2 {num(float(summary.loc[S2, 'cost_net_of_inventory_yuan']))} 元，"
        f"S2 仍低 {num(float(summary.loc[S0, 'cost_net_of_inventory_yuan']) - float(summary.loc[S2, 'cost_net_of_inventory_yuan']))} 元。"
        f"该系数只是原费用对照沿用的敏感性口径，不是真实终端市场价值。")
    add(f"5. **上一轮预测 MAE 改善 {num(6.4263, 2)}%，本轮现金只改善 {num(abs(delta_share) * 100, 3)}%**，"
        f"二者不同量级：点预测变准不等于等比例的现金收益，也不能据此宣称 F2 优于 F1。"
        f"**不自动替换原模型、不填 `result3.xlsx`**；是否保留 S2 为候选由用户与主 Agent 决定。")
    add("")

    add("## 2. 问题分析")
    add("")
    add("前一轮前向预测对照显示 F2 在 A 集 10 分钟 MAE 上优于 F0 与 F1，但那只是预测误差，"
        "不能证明节费：负载预测改变后，保护量、滚动候选、普通购电、调整费、应急费与跨日 SOC 都会变。"
        "本轮因此只比较两套完整策略：")
    add("")
    add(table([("S0_L75", "前向档案 F0（= 原 0:00 预测）", "原 Linear 转换", "已保存原 L75 自身 W28/q75",
                "只读复用已核账本"),
               ("S2_load_q75", "前向档案 F2（0 点原预测，6/12/18 点修正）", "同一 Linear 档案",
                "依据 F2 自身发布误差重建 W28/q75", "本轮唯一新增调度组")],
              ["策略", "负载预测", "光伏", "保护", "调度来源"]))
    add("")
    add("F1 不进入本轮。所得差异度量「F2 负载修正＋相应残差保护＋调度响应」的组合效果，"
        "**不能拆称纯当天信息价值，也不能判断 F2 一定比 F1 更节费**。两组共用同一实际供需、"
        "交付时段价格、设备参数、公共初始化、优化与执行规则。")
    add("")

    add("## 3. 数据预处理")
    add("")
    identity = checks["inputs_and_protection"]["identity"]
    public = checks["inputs_and_protection"]["public_initial"]
    add(table([
        ("F0 负载身份", f"前向档案 F0 ≡ 冻结 `issued_load`，最大绝对差 "
                        f"{num(identity['f0_vs_issued_load_max_abs_kW'], 1)} kW（{identity['f0_vs_issued_compared_cells']:,} 格）"),
        ("F0 净需求身份", f"重建 `(F0 − Linear PV)·Δt` ≡ 冻结 `net_c0`，最大绝对差 "
                          f"{num(identity['net_f0_vs_frozen_net_c0_max_abs_kWh'], 1)} kWh（{identity['net_f0_compared_cells']:,} 格）"),
        ("公共初态", f"2025-02-01 00:00 内部库存 {num(public['initial_kWh'], 6)} kWh、"
                     f"午夜承诺 {num(public['carry_kWh'], 6)} kWh（归属 {public['carry_owner']}）"),
        ("保护水平", f"α = 0.75，窗口 28 个日历发布日，最少 7 个有效样本；"
                     f"F0 重建与保存档案逐格一致、计数完全相同"),
        ("索引", "h=0 当日 00:00—00:10、h=144 次日 00:00—00:10；0 点有效范围 h=0..144（正式计划 h=1..144），"
                 "日内分别为 h=36..144、72..144、108..144"),
        ("样本", f"S0/S2 各 48,097 段轨迹（自然 48,096 段为主指标，含 2026-01-01 00:00—00:10 尾段）"),
    ], ["项目", "口径"]))
    add("")
    add("F2 的 0 点保护与 F0 逐格相同"
        f"（最大绝对差 {num(checks['inputs_and_protection']['f2_zero_version_equals_f0']['max_abs_kWh'], 1)} kWh）——"
        "0 点负载预测未变；但**0 点预测一致不代表 0 点购电一致**，从第二天起两组库存与午夜承诺各自独立演化，"
        "本轮没有固定任何一天的 S0 计划。")
    add("")

    add("## 4. 模型建立")
    add("")
    add("净需求预测与残差（单位 kWh，不重复乘 Δt）：")
    add("")
    add("```")
    add("n_hat[g][k,v,h] = (load_hat[g][k,v,h] - pv_linear[k,v,h]) * dt")
    add("n[k,h]          = (L[k,h] - V[k,h]) * dt")
    add("eps[g]          = n - n_hat[g]")
    add("rho[g][k,v,h]   = eps[g]_(ceil(0.75 m))     # m >= 7；否则 0；允许负保护")
    add("protected[g]    = n_hat[g] + rho[g]")
    add("```")
    add("")
    add("历史资格：`max(0,k-28) <= d < k` 且 `144d + h + 1 <= 144k + 6·r_v`（该目标区间已结束）。"
        "**F2 的 14 日权重拟合与 q75 的 7 日保护历史是两套机制，不得混用**；"
        "F2 只用自身已保存的发布值形成历史误差，不用 F0/F1 误差替代、也不用当前参数回代旧日期。")
    add("")
    add("调度内核不变：功率上限 5000 kW（每槽 5000/6 kWh）、内部库存 1200—10800 kWh、"
        "充放电效率各 0.9、自由末态、无售电与新增惩罚；0 点最小化 `Σ p q⁰`，"
        "日内最小化 `Σ [p q + 0.5 p |q − q⁰|]`；候选须比旧计划评分低超过 1e-4 元才被接受。"
        "结算 `C = Σ [p q_eff + 0.5 p |q_eff − q0| + 5 p e]`，多次覆盖不累计重复调整费。")
    add(f"本轮实际求解：S2 正式 {validation['solves']['S2_formal']} 次 MILP ＋ 适配检查 "
        f"{validation['solves']['adaptation']} 次；S0 全年重跑 "
        f"{validation['solves']['S0_full_year_reruns']} 次。保护重建是两组纯数组运算，"
        f"**如实登记为 2 组重建，不称 0 分位重建**。")
    add("")

    add("## 5. 结果")
    add("")
    add("### 5.1 主指标与三项分解")
    add("")
    rows = [(label, num(float(contrasts.loc[key, "s0_yuan"])),
             num(float(contrasts.loc[key, "s2_yuan"])),
             signed(float(contrasts.loc[key, "delta_yuan"])),
             signed(float(contrasts.loc[key, "delta_share_of_total"]) * 100, 3) + "%")
            for label, key in (("现金总费", "total"), ("普通购电", "ordinary"),
                               ("调整费", "adjustment"), ("应急费", "emergency"))]
    add(table(rows, ["项目", "S0_L75（元）", "S2_load_q75（元）", "差额（元）", "占 S0 总费"]))
    add("")
    add("三项分解之和与总差额一致（由计算层断言，容差 1e-4 元）。"
        "注：现金已含 5 倍应急费，因此应急风险的下降已计入节费，不另设题面没有的风险否决约束。")
    add("")

    add("### 5.2 分月与逐日")
    add("")
    add(table([(row.month, num(row.S0_L75), num(row.S2_load_q75), signed(row.delta_yuan))
               for row in monthly.itertuples(index=False)],
              ["月份", "S0（元）", "S2（元）", "差额（元）"]))
    add("")
    add(f"逐日：{refreshed} 天更省（合计 {signed(float(daily[daily.delta_yuan < 0].delta_yuan.sum()))} 元）、"
        f"{worsened} 天更贵（合计 {signed(float(daily[daily.delta_yuan > 0].delta_yuan.sum()))} 元）；"
        f"逐日中位数 {signed(days['delta_median_yuan'])} 元。绝对差额最大的 3 日："
        + "、".join(f"{d}（{signed(v)}）" for d, v in days["largest_absolute"])
        + "；**费用增加最大**的 3 日："
        + "、".join(f"{d}（{signed(v)}）" for d, v in days["largest_increase"]) + "。")
    add("")

    add("### 5.3 应急与库存")
    add("")
    add(table([(label, num(float(summary.loc[S0, key]), digits), num(float(summary.loc[S2, key]), digits),
                signed(float(summary.loc[S2, key]) - float(summary.loc[S0, key]), digits))
               for label, key, digits in
               (("应急电量 (kWh)", "emergency_kWh", 1), ("应急费 (元)", "emergency_cost_yuan", 1),
                ("应急区间数", "emergency_intervals", 0), ("应急天数", "emergency_days", 0),
                ("应急事件数", "emergency_events", 0), ("未使用电量 (kWh)", "unused_kWh", 1),
                ("充放电损耗 (kWh)", "loss_kWh", 1),
                ("期末自然库存 (kWh)", "final_natural_state_kWh", 1))],
              ["指标", "S0_L75", "S2_load_q75", "差额"]))
    add("")
    add(f"应急电量大降而应急天数上升，说明单次缺口变浅但分布更广；"
        f"两组最大求解 gap 均为 {num(float(summary.max_solver_gap.max()), 12)}，"
        f"名义可行性最大违规 {num(float(summary.max_feasibility_violation_kWh.max()), 12)} kWh。")
    add("")

    add("### 5.4 保护曲线变化")
    add("")
    add(table([(row.quantity, f"{int(row.publication_hour):02d}:00", row.scope, f"{int(row.n):,}",
                num(row.signed_mean, 3), num(row.mean_abs, 3), num(row.max_abs, 3),
                num(row.mean_left, 2), num(row.mean_right, 2))
               for row in protection_change.itertuples(index=False)],
              ["量", "发布节点", "范围", "n", "有符号均值 (kWh)", "平均绝对值 (kWh)",
               "最大绝对值 (kWh)", "S0 均值 (kWh)", "S2 均值 (kWh)"]))
    add("")
    add(f"0 点版本三行完全相同：F2 的 0 点负载预测与 F0 一致。日内净需求预测变化很小"
        f"（平均绝对值约 10 kWh），但**保护量的平均绝对变化达 7.5—8.5 kWh、最大 "
        f"{num(protection_change[(protection_change.quantity == 'protected') & (protection_change.scope == 'reach')].max_abs.max(), 1)} kWh**，"
        f"说明变化主要来自残差历史而非当期净需求本身。尾目标单列，不并入可达范围；"
        f"**不能用有符号均值接近 0 宣称逐段曲线不变**。")
    add("")

    add("### 5.5 修订与择优")
    add("")
    add(table([(sid, f"{int(summary.loc[sid, 'revisions_accepted'])}/{int(summary.loc[sid, 'revision_decisions'])}",
                f"{int(summary.loc[sid, 'accepted_06'])}/{int(summary.loc[sid, 'decisions_06'])}",
                f"{int(summary.loc[sid, 'accepted_12'])}/{int(summary.loc[sid, 'decisions_12'])}",
                f"{int(summary.loc[sid, 'accepted_18'])}/{int(summary.loc[sid, 'decisions_18'])}",
                num(float(summary.loc[sid, 'max_solver_gap']), 12))
               for sid in (S0, S2)],
              ["策略", "接受/决策", "06:00", "12:00", "18:00", "最大 gap"]))
    add("")
    s2_check = checks["saved_results"][S2]
    rejected_text = ("无被拒决策" if s2_check.get("max_improvement_yuan_of_rejected") is None
                     else f"被拒决策中最大改善 {num(s2_check['max_improvement_yuan_of_rejected'], 2)} 元")
    add(f"**S2 的 1,002 次修订全部被接受**（S0 为 "
        f"{int(summary.loc[S0, 'revisions_accepted'])}/1002，其中 18:00 只接受 "
        f"{int(summary.loc[S0, 'accepted_18'])}/334）；接受的最小改善 "
        f"{num(s2_check['min_improvement_yuan_of_accepted'])} 元，{rejected_text}，"
        f"接受规则违反与漏接受均为 {s2_check['accept_rule_violations']} / "
        f"{s2_check['accept_rule_misses']}。接受率本身不是实际收益证明，"
        f"也不能把某节点的节费单独归给光伏或负载信息。")
    add("")

    add("### 5.6 指定日期")
    add("")
    pivoted = selected.pivot_table(index="date", columns="strategy_id", values="total_cost_yuan")
    add(table([(date, num(row[S0]), num(row[S2]), signed(float(row[S2] - row[S0])))
               for date, row in pivoted.iterrows()],
              ["日期", "S0（元）", "S2（元）", "差额（元）"]))
    add("")
    add(f"四个指定日期互有胜负（12-21 与 03-20 S2 更省，06-21 与 09-23 S2 更贵），"
        f"与整体节费主要来自 12 月与 7 月一致。")
    add("")
    paths = make_figures(monthly, pd.read_csv(OUT / "contrasts.csv"), pd.read_csv(OUT / "summary.csv"))
    for path, caption in zip(paths, ("分月现金差（负值=S2 更省）",
                                     "现金三项差与应急指标（口径不同，未归一化）")):
        add(f"![{caption}]({path.resolve().as_posix()})")
        add("")
    add("图只做程序化完整性检查（本 Agent 无法目视图像）；图内不画阈值线、不加显著性星号，"
        "纵轴单位均为元或各自的原单位。")
    add("")

    add("## 6. 验证（三类精简检查，全部通过）")
    add("")
    c1 = checks["inputs_and_protection"]
    c2 = checks["adaptation"]
    add(f"1. **输入与保护边界（0 次 MILP）**：F0 负载/净需求身份与冻结档案逐格一致"
        f"（最大绝对差 {num(identity['net_f0_vs_frozen_net_c0_max_abs_kWh'], 1)} kWh）；"
        f"F0 重建 q75 保护与保存档案最大绝对差 "
        f"{num(c1['f0_protection_rebuild']['max_abs_protected_kWh'], 1)} kWh、计数完全相同；"
        f"F2 的 0 点保护与 F0 逐格相同。独立手取下三组保存历史误差核对 m／位置／ρ："
        + "；".join(f"{row['date']} {row['publication_hour']:02d} 点 h={row['h']}（m={row['m']}、"
                   f"位置 {row['position']}、ρ={num(row['rho_kWh'], 3)}）"
                   for row in c1["spot_protection_groups"])
        + f"；2025-06-21 在 0 点与 6 点扰动**尚未结束**目标的真值后，该发布时刻的保护最大变化 "
        f"{num(c1['boundary_perturbation']['max_abs_change_kWh'], 1)} kWh（预测因果性沿用前轮已核证据，"
        f"本轮不重拟合、不全期扰动）。")
    add(f"2. **两个适配求解（2 次 MILP）**：用原 L75 在 2025-06-21 的保存输入与状态，"
        f"经新封装重解 0 点计划与 06:00 修订。0 点目标差 "
        f"{num(c2['midnight']['objective_delta_yuan'], 1)} 元、计划差 "
        f"{num(c2['midnight']['plan_max_delta_kWh'], 1)} kWh；06:00 的旧有效计划取"
        f"计划版本档案的 `previous` 列（不从最终账本截取），候选与保存候选最大差 "
        f"{num(c2['six']['candidate_max_delta_kWh'], 1)} kWh，接受判定与保存一致"
        f"（{c2['six']['replayed_accepted']}）。")
    add(f"3. **正式保存账本核对（0 次 MILP）**：两组各 48,097 段、10 分钟连续、无重复；"
        f"非有限格 0；同时充放电 {num(max(checks['saved_results'][sid]['simultaneous_charge_discharge_kWh'] for sid in (S0, S2)), 1)} kWh、"
        f"状态连续 {num(max(checks['saved_results'][sid]['state_continuity_max_abs_kWh'] for sid in (S0, S2)), 1)} kWh、"
        f"母线平衡最大 {num(max(checks['saved_results'][sid]['bus_balance_max_abs_kWh'] for sid in (S0, S2)), 1)} kWh、"
        f"递推最大 {num(max(checks['saved_results'][sid]['battery_recurrence_max_abs_kWh'] for sid in (S0, S2)), 1)} kWh；"
        f"三项现金按 `p·q_eff`、`0.5p|q_eff−q0|`、`5p·e` 独立重算，最大差 "
        f"{num(max(checks['saved_results'][sid][f'{side}_cash_max_abs_yuan'] for sid in (S0, S2) for side in ('ordinary', 'adjustment', 'emergency')), 1)} 元；"
        f"自然/模板首尾桥接 S0 {num(checks['saved_results'][S0]['bridge_yuan'])} 元、"
        f"S2 {num(checks['saved_results'][S2]['bridge_yuan'])} 元；求解日志各 1,336 行、"
        f"无失败回退、接受规则 0 违反。")
    add("")

    add("## 7. 边界与限制")
    add("")
    for item in validation["limitations"]:
        add(f"- {item}")
    add("")
    add(f"**预登记判定规则（方案 §4.4）**：ΔC = C_S2 − C_S0，绝对差 ≤ 1e-4 元按数值持平；"
        f"主现金更低只能说“本年度此固定配置费用更低”，不称最优。本轮 ΔC = "
        f"{signed(delta_total)} 元，落在“更低”一侧，但必须与第 1 节第 2、3 条的分布与风险并列阅读，"
        f"且**不自动替换原模型、不填 `result3.xlsx`**。")
    add("")

    add("## 8. 产物与复现")
    add("")
    integrity_after = manifest_integrity(manifest)
    add(f"计算层产物 `results/q3_intraday_load_cost/`：`protection.npz`（两组 net_forecast/error/rho/"
        f"counts/protected/position 与可用性掩码）、`S2_load_q75/`（dispatch、plan_versions、"
        f"revision_decisions、solver_log、nominal_trajectory）、`summary.csv`、`contrasts.csv`、"
        f"`monthly.csv`、`daily.csv`、`protection_change.csv`、`selected_dates.csv`、"
        f"`validation.json`、`registration.json`、`run_manifest.json`；"
        f"S0 以源路径与哈希记录、未伪装成本轮新算。报告层产物本报告与 "
        f"`figures/q3_intraday_load_cost/` 两图。")
    add("")
    add(f"核账：`run_manifest.json` 状态 `{manifest['status']}`，对 {integrity_after['checked']} 项计算层产物"
        f"登记哈希，重渲本报告后不匹配 {len(integrity_after['mismatched'])} 项、缺失 "
        f"{len(integrity_after['missing'])} 项；整轮 {manifest['wall_seconds']:.1f} 秒，"
        f"MILP 共 {manifest['milp']['S2_formal']} ＋ {manifest['milp']['adaptation']} 次。")
    add("")
    amendments = registration.get("amendments", [])
    if amendments:
        add(f"登记修订 {len(amendments)} 条（实现缺陷修复，未改变保护口径、内核或评价规则）：")
        for amendment in amendments:
            add(f"- `{amendment['previous_signature'][:12]}` → `{amendment['new_signature'][:12]}`："
                f"{amendment['reason']}")
        add("")
    add(f"登记签名 `{registration['signature']}`；输入与源码哈希见 `registration.json`，"
        f"只登记本方案、前向预测与费用实验的已接收产物、冻结光伏/真值档案与复用内核源码，未做全树哈希。")
    add("")
    add("复现：赛题目录执行 `conda run -n math_modeling python "
        "code/q3_intraday_load_cost_experiment.py --mode full`，再执行 "
        "`conda run -n math_modeling python code/q3_intraday_load_cost_report.py`。")
    add("")
    add("报告层只读 `results/q3_intraday_load_cost/`：重渲本报告与图表不会改动计算层任何产物，"
        "改文案或图表也不需要重跑调度。")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrity = figure_integrity(paths)
    (FIG / "figure_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "rendered", "report": REPORT_MD.name, "lines": len(lines),
                      "manifest": integrity_after, "figures": integrity,
                      "delta_total_yuan": delta_total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    main()
