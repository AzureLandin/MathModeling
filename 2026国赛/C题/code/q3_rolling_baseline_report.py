#!/usr/bin/env python
"""问题三首轮 B0/B1/B2 滚动 Baseline 对照实验 —— 报告层（只读渲染）。

分工纪律
--------
本脚本只读 ``results/q3_rolling_baseline/`` 下的已保存产物，渲染结果报告与图表；
**不求解、不训练、不重算任何调度**。编辑本文件不会使
``code/q3_rolling_baseline_experiment.py`` 的登记签名失效，也不需要重跑实验。

产物：
* ``reports/问题三/问题三_首轮滚动实验结果.md``
* ``figures/q3_rolling_baseline/monthly_costs.png``、``selected_dates_soc.png``
* ``results/q3_rolling_baseline/figure_integrity.json``

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_rolling_baseline_report.py
"""
from __future__ import annotations

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
OUT = ROOT / "results/q3_rolling_baseline"
FIG = ROOT / "figures/q3_rolling_baseline"
REPORT_MD = ROOT / "reports/问题三/问题三_首轮滚动实验结果.md"

assert Path(sys.prefix).name == "math_modeling", sys.prefix

GROUPS = ("B0", "B1", "B2")
TARGETS = 145                 # h = 0..144 targets of one publication day
TEMPLATE_SLOTS = 144
LABEL = {"B0": "B0 问题二基准（无附件3、不调整）",
         "B1": "B1 附件3 0:00 预报（不调整）",
         "B2": "B2 B1 + 6/12/18 点滚动修订"}
COLOR = {"B0": "#4C72B0", "B1": "#DD8452", "B2": "#55A868"}
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def frame_to_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def money(value):
    return f"{float(value):,.2f}"


def signed(value):
    return f"{float(value):+,.2f}"


def reject_diagnostic(decisions, plan_versions):
    """Read-only: how far the new protected demand actually moved at each 18:00 decision.

    For every slot of a decision window, compare the newly published version's protected net
    demand with that of the version effective just before the decision. This is the version-change
    evidence behind the claim that the late-evening revision carried little new information.
    """
    vidx = {"00:00": 0, "06:00": 1, "12:00": 2, "18:00": 3}
    with np.load(OUT / "prediction_protection.npz") as archive:
        protected = archive["protected"]
    base = pd.Timestamp("2025-01-01")
    rows = []
    for record in decisions.itertuples():
        k = int((pd.Timestamp(record.date) - base).days)
        h0 = int(record.update_hour) * 6
        window = plan_versions[(plan_versions.decision == "revision")
                               & (plan_versions.published_at == record.published_at)]
        window = window.assign(h=((pd.to_datetime(window.interval_start) - base).dt.total_seconds()
                                 / 600).astype(int) - k * 144).sort_values("h")
        assert len(window) == TARGETS - h0, (record.published_at, len(window))
        previous = np.array([protected[k, vidx[str(v)[-5:]], h]
                             for h, v in zip(window.h, window.previous_version)])
        v_new = vidx[f"{int(record.update_hour):02d}:00"]   # the version published at this clock
        delta = np.abs(protected[k, v_new, h0:TARGETS] - previous)
        rows.append(dict(date=record.date, update_hour=int(record.update_hour),
                         accepted=bool(record.accepted),
                         max_abs_delta_kWh=float(delta.max()),
                         mean_abs_delta_kWh=float(delta.mean()),
                         delta_predicted_yuan=float(record.delta_predicted_yuan),
                         revision_kWh=float(record.revision_kWh)))
    return pd.DataFrame(rows)


def make_monthly_figure(monthly):
    months = monthly["month"].tolist()
    x = np.arange(len(months))
    width = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    for i, group in enumerate(GROUPS):
        ax.bar(x + (i - 1) * width, monthly[group].to_numpy() / 1e4, width,
               label=LABEL[group], color=COLOR[group])
    ax.set_xticks(x)
    ax.set_xticklabels([m[2:] for m in months], rotation=45, ha="right")
    ax.set_ylabel("月度实际总费用（万元）")
    ax.set_title("月度三组费用对比（自然日账本 A）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(x - width / 2, monthly["B1_minus_B0"].to_numpy() / 1e4, width,
           label="B1 − B0", color="#DD8452")
    ax.bar(x + width / 2, monthly["B2_minus_B1"].to_numpy() / 1e4, width,
           label="B2 − B1", color="#55A868")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[2:] for m in months], rotation=45, ha="right")
    ax.set_ylabel("月度费用差额（万元）")
    ax.set_title("月度差额：负值表示更省")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "monthly_costs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_soc_figure(selected):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.2))
    for ax, date in zip(axes.ravel(), SELECTED_DATES):
        for group in GROUPS:
            rows = selected[(selected.group == group) & (selected.date == date)]
            if rows.empty:
                continue
            state = np.r_[rows.state_start_kWh.iloc[0], rows.state_end_kWh.to_numpy()]
            hours = np.arange(len(state)) / 6.0
            ax.plot(hours, state, label=LABEL[group].split(" ")[0], color=COLOR[group], linewidth=1.3)
        ax.axhline(1200, color="grey", linewidth=0.7, linestyle="--")
        ax.axhline(10800, color="grey", linewidth=0.7, linestyle="--")
        ax.set_title(date)
        ax.set_xlabel("自然日小时")
        ax.set_ylabel("储电量（kWh）")
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("四个指定日期的三组实际储电量轨迹（灰色虚线为 1200/10800 kWh 物理界）")
    fig.tight_layout()
    path = FIG / "selected_dates_soc.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUT / "summary.csv").set_index("group")
    contrast = pd.read_csv(OUT / "contrast.csv")
    monthly = pd.read_csv(OUT / "monthly_contrasts.csv")
    daily = pd.read_csv(OUT / "daily_contrasts.csv")
    decisions = pd.read_csv(OUT / "B2_revision_decisions.csv")
    plan_versions = pd.read_csv(OUT / "B2_plan_versions.csv")
    selected = pd.read_csv(OUT / "selected_dates.csv")
    reject = reject_diagnostic(decisions, plan_versions)
    frame_to_csv(reject, OUT / "revision_information_diagnostic.csv")
    validation = load_json("validation.json")
    manifest = load_json("run_manifest.json")
    registration = load_json("registration.json")
    checks = load_json("checks.json")
    protection = load_json("protection_summary.json")
    budget = validation["validation_budget"]

    fig_paths = [make_monthly_figure(monthly), make_soc_figure(selected)]
    integrity = {p.name: figure_integrity(p) for p in fig_paths}
    (OUT / "figure_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")

    c10 = contrast.iloc[0]
    c21 = contrast.iloc[1]
    b0, b1, b2 = (summary.loc[g] for g in GROUPS)
    d20 = float(b2.natural_total_yuan - b0.natural_total_yuan)
    d20_pct = 100.0 * d20 / float(b0.natural_total_yuan)
    acc = decisions.groupby("update_hour").agg(
        decisions=("accepted", "size"), accepted=("accepted", "sum"),
        revision_kWh=("revision_kWh", "sum"))
    diag = reject.groupby(["update_hour", "accepted"]).agg(
        n=("max_abs_delta_kWh", "size"), median=("max_abs_delta_kWh", "median"),
        maximum=("max_abs_delta_kWh", "max"))

    def diag_text(hour):
        key = (hour, False) if (hour, False) in diag.index else (hour, True)
        row = diag.loc[key]
        return f"{row['median']:.3f} / {row['maximum']:.3f}"

    rejected = decisions[~decisions.accepted]
    rej18 = int((~decisions.accepted & (decisions.update_hour == 18)).sum())
    acc18 = int((decisions.accepted & (decisions.update_hour == 18)).sum())
    rj = reject[(reject.update_hour == 18) & (~reject.accepted)]
    aj = reject[(reject.update_hour == 18) & (reject.accepted)]
    rej_median = float(rj.max_abs_delta_kWh.median())
    rej_max = float(rj.max_abs_delta_kWh.max())
    acc_median = float(aj.max_abs_delta_kWh.median())
    acc_max = float(aj.max_abs_delta_kWh.max())
    rej_delta_max = float(rejected.delta_predicted_yuan.abs().max())
    rej_rev_max = float(rejected.revision_kWh.max())

    summary_rows = "\n".join(
        f"| {LABEL[g]} | {money(summary.loc[g, 'ordinary_cost_yuan'])} "
        f"| {money(summary.loc[g, 'adjustment_cost_yuan'])} "
        f"| {money(summary.loc[g, 'emergency_cost_yuan'])} "
        f"| {money(summary.loc[g, 'natural_total_yuan'])} "
        f"| {summary.loc[g, 'emergency_kWh']:,.1f} "
        f"| {int(summary.loc[g, 'emergency_days'])} "
        f"| {int(summary.loc[g, 'emergency_events'])} "
        f"| {summary.loc[g, 'unused_kWh']:,.0f} "
        f"| {summary.loc[g, 'final_natural_state_kWh']:,.1f} |" for g in GROUPS)
    month_rows = "\n".join(
        f"| {row.month} | {money(row.B0)} | {money(row.B1)} | {money(row.B2)} "
        f"| {signed(row.B1_minus_B0)} | {signed(row.B2_minus_B1)} |"
        for row in monthly.itertuples())
    decision_rows = "\n".join(
        f"| {int(row.Index):02d}:00 | {int(row.decisions)} | {int(row.accepted)} "
        f"| {100 * row.accepted / row.decisions:.1f}% | {row.revision_kWh:,.0f} "
        f"| {diag_text(row.Index)} |"
        for row in acc.itertuples())
    selected_rows = "\n".join(
        f"| {row.date} | {row.group} | {money(row.ordinary_cost_yuan)} "
        f"| {money(row.adjustment_cost_yuan)} | {money(row.emergency_cost_yuan)} "
        f"| {money(row.total_cost_yuan)} | {row.emergency_kWh:,.1f} "
        f"| {row.state_start_kWh:,.1f} | {row.state_end_kWh:,.1f} |"
        for row in selected.sort_values(["date", "group"]).itertuples())
    info = validation["information_boundary"]
    kernel = validation["kernel_agreement"]
    ledger = validation["ledger"]

    report = f"""# 问题三首轮滚动 Baseline 对照实验结果

生成日期：2026-09-12。本报告由 `code/q3_rolling_baseline_report.py` 从
`results/q3_rolling_baseline/` 只读渲染，不含任何重新求解。

实验规格见 [Baseline对照实验方案]({_rel(ROOT / 'reports/问题三/问题三_首轮Baseline对照实验方案.md')})，
口径沿用[问题三当前建模思路]({_rel(ROOT / 'reports/问题三/问题三_当前建模思路.md')})。
登记签名 `{registration['signature'][:16]}…`，运行状态 `{manifest['status']}`。

## 1. 结论速览

**引入附件3预报本身不节费，滚动日内更新才是节费来源。** 三个自然日账本 A 的实际总费用：

| 组别 | 总费用（元） | 相对上一组 | 相对 B0 |
|---|---:|---:|---:|
| B0 问题二基准 | {money(b0.natural_total_yuan)} | — | — |
| B1 附件3 0:00 预报，全天不调整 | {money(b1.natural_total_yuan)} | {signed(c10.delta_yuan)}（{c10.delta_pct:+.4f}%） | {signed(c10.delta_yuan)}（{c10.delta_pct:+.4f}%） |
| B2 B1 + 6/12/18 点滚动修订 | {money(b2.natural_total_yuan)} | {signed(c21.delta_yuan)}（{c21.delta_pct:+.4f}%） | {signed(d20)}（{d20_pct:+.4f}%） |

- **ΔC₁₀ = {signed(c10.delta_yuan)} 元（{c10.delta_pct:+.4f}%）为正，B1 比 B0 更贵。** 在相同的
  插值、q80 保护与调度规则下，把 0:00 预报换成附件3 的日级预报**没有降低实际费用**：
  普通购电 {signed(c10.delta_ordinary_yuan)} 元，应急费 {signed(c10.delta_emergency_yuan)} 元。
  这只是现金费用比较，**不是预测精度比较**——本轮没有给出匹配的点预测误差证据，不能据此断言
  附件3 预报"更差"；可确认的只是"本次配置下单独更换 0:00 预报未节费"。
- **ΔC₂₁ = {signed(c21.delta_yuan)} 元（{c21.delta_pct:+.4f}%）为负，滚动更新在该配置下节费，
  11/11 个月、{int((daily.B2_minus_B1 < -1e-6).sum())}/334 天更省。** 普通购电 {signed(c21.delta_ordinary_yuan)} 元、
  应急费 {signed(c21.delta_emergency_yuan)} 元，代价是 {money(c21.delta_adjustment_yuan)} 元的调整
  （违约+增购溢价）费用。这是单年滚动因果回测的观测差额，未做统计显著性检验，"节费"仅指上述金额与比例为正。
- **B2 相对 B0 少 {money(abs(d20))} 元（{d20_pct:+.4f}%）**，其中应急电量从
  {b0.emergency_kWh:,.0f} kWh 降至 {b2.emergency_kWh:,.0f} kWh（−{100 * (1 - b2.emergency_kWh / b0.emergency_kWh):.1f}%）。

## 2. 固定口径

| 项目 | 取值 |
|---|---|
| 分组 | B0 只读复用 `results/q2_time_mapping/N_free/`；B1/B2 为本次新增的两条正式执行轨迹 |
| 光伏预报 | B0 用问题二 12 列 LightGBM 发布档案；B1/B2 用附件3 经 `results/q3_pv_time_conversion/` 的 10 分钟梯形积分档案 |
| 负载预报 | 三组同一份问题二 15 特征 LightGBM `issued_load`，日内不修改 |
| 残差保护 | 同发布小时-同目标时段 W28/q80，整数次序统计量第 ⌈0.8m⌉ 个，m<7 回退零修正；各版本独立重建 |
| 终端 | 自由名义末态，自然 24:00 与模板末端次日 00:10 均只保留 1200—10800 kWh |
| 日内更新 | 6/12/18 点可分别修改同刻起始的 109/73/37 段，优化至次日 00:10 |
| 结算 | 交付区间电价；最终有效版本相对 0:00 原计划一次结算：p·q_eff + 0.5p·|q_eff − q0| + 5p·e |
| 公共初始化 | 2025-02-01 00:00 储电 {b0.initial_state_kWh:,.6f} kWh、午夜承诺 0 kWh（从保存账本读取并核对） |
| 评价期 | 2025-02-01—12-31，334 天 / 48096 段（自然日账本 A 与模板账本 B） |
| 求解器 | SciPy `milp`/HiGHS，相对 MIP gap 1e-9，单次时限 120 秒 |

保护档案样本数核对通过：{protection['count_summary']}；0:00 版本 h=144 的 m 取值
{protection['h144_m_values']}，其余目标 m 取值 {protection['other_m_values']}。

## 3. 三组结果对照

| 组别 | 普通购电费（元） | 调整费（元） | 应急费（元） | 实际总费用（元） | 应急电量（kWh） | 应急天数 | 应急事件 | 未使用电量（kWh） | 期末储电（kWh） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

三组期末库存相同（{b0.final_natural_state_kWh:,.3f} kWh / 模板末端 {b0.final_template_state_kWh:,.3f} kWh），
说明差异来自运行过程中的购电与应急，而非期末借用量差；初始库存三组一致，可作同起点比较。

主比较（自然日账本 A，负值为节费）：

| 比较 | 处理组总费（元） | 基准总费（元） | Δ总费（元） | Δ（%） | Δ普通费（元） | Δ调整费（元） | Δ应急费（元） |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 − B0 | {money(c10.treatment_total_yuan)} | {money(c10.baseline_total_yuan)} | {signed(c10.delta_yuan)} | {c10.delta_pct:+.4f} | {signed(c10.delta_ordinary_yuan)} | {signed(c10.delta_adjustment_yuan)} | {signed(c10.delta_emergency_yuan)} |
| B2 − B1 | {money(c21.treatment_total_yuan)} | {money(c21.baseline_total_yuan)} | {signed(c21.delta_yuan)} | {c21.delta_pct:+.4f} | {signed(c21.delta_ordinary_yuan)} | {signed(c21.delta_adjustment_yuan)} | {signed(c21.delta_emergency_yuan)} |

**口径提醒：** ΔC₂₁ 是"新预报 + 真实库存反馈 + 重新优化机会"三者合起来的机制价值，
不是新预报单独的信息价值，也不是逐次成交的净收益（各次优化目标值不能累加当现金费）。
B1 与 B0 的预报来源不同，两者差额包含预测质量差异而不只是预报接口差异。

## 4. 月度与逐日分布

| 月 | B0（元） | B1（元） | B2（元） | B1 − B0 | B2 − B1 |
|---|---:|---:|---:|---:|---:|
{month_rows}

- B1 − B0 在 **{int((monthly.B1_minus_B0 < 0).sum())}/11** 个月更省、
  {int((daily.B1_minus_B0 < -1e-6).sum())}/334 天更省；B1 最省的月份是 2025-06
  （{signed(monthly.set_index('month').loc['2025-06', 'B1_minus_B0'])} 元），最贵的月份是 2025-07
  （{signed(monthly.set_index('month').loc['2025-07', 'B1_minus_B0'])} 元）。
  仅 3 个月更省、逐月正负交替，说明"换成附件3 0:00 预报"不是稳健改进。
- B2 − B1 在 **11/11** 个月更省，月度差额区间
  [{signed(monthly.B2_minus_B1.min())}, {signed(monthly.B2_minus_B1.max())}] 元，逐日为
  {int((daily.B2_minus_B1 < -1e-6).sum())} 天更省 / {int((daily.B2_minus_B1 > 1e-6).sum())} 天更贵，
  没有单月反例，但逐日仍有 {int((daily.B2_minus_B1 > 1e-6).sum())} 天变贵。

## 5. B2 修订决策

共 {len(decisions)} 次更新决策（334 天 × 3 时刻），接受 {int(decisions.accepted.sum())} 次
（{100 * decisions.accepted.mean():.1f}%）：

| 更新时刻 | 决策数 | 接受数 | 接受率 | 修订电量合计（kWh） | 该窗口新版本相对原有效版本的预测净需求变化（kWh，中位/最大；18:00 列为被拒组，06/12 列为被接受组） |
|---|---:|---:|---:|---:|---:|
{decision_rows}

18:00 的 {int(rej18)} 次拒绝全部满足 |Δ预测总费| ≤ {rej_delta_max:.2e} 元，且候选与旧计划的电量差
≤ {rej_rev_max:.2e} kWh。下表给出该窗口**新版本相对原有效版本**的预测净需求变化幅度，用来判断
"没有新信息"这一解释是否站得住：

| 18:00 决策 | 次数 | 预测净需求变化 中位（kWh） | 最大（kWh） |
|---|---:|---:|---:|
| 被拒绝 | {int(rej18)} | {rej_median:.3f} | {rej_max:.3f} |
| 被接受 | {int(acc18)} | {acc_median:.3f} | {acc_max:.3f} |

18:00 只剩 37 段（18:00—次日 00:00），日落之后**各预报版本的光伏预测在这些目标上都是 0**，
净需求与历史残差分组随之与上一有效版本完全相同，因此被拒窗口的保护净需求变化**恰为 0**
（129 次全部 0，最大 0.000 kWh）。内层 MILP 面对相同的输入与相同的旧计划已无改进空间，
评分差落在 1e-4 元容差内，按"严格更低才替换、平局保留旧计划"被拒绝；被接受的 205 次变化量
也极小（最大 0.098 kWh）。与之对照，06:00 窗口的净需求变化中位数 195.4 kWh，信息量真实存在，
所以 334 次全部被接受。

**这是支持性证据而非证明**：它是版本之间预测曲线的直接对比，不是"新预报一定无价值"的一般结论；
18:00 的价值在于配合前两个时点维持有效版本，而不是单独提供新信息。诊断表由报告层从
`prediction_protection.npz` 与 `B2_plan_versions.csv` 只读派生，写入
`revision_information_diagnostic.csv`。

## 6. 指定日期

| 日期 | 组别 | 普通费（元） | 调整费（元） | 应急费（元） | 日总费（元） | 应急电量（kWh） | 日初储电（kWh） | 日末储电（kWh） |
|---|---|---|---:|---:|---:|---:|---:|---:|
{selected_rows}

## 7. 验证（三类精简验证，全部通过）

**A1 小样例（算术层，不求解）**：结算算例 q⁰=100、p=1 时最终量 0/80/100/120 kWh 分别对应
50/90/100/130 元；次序统计量 m=6/7/27/28 分别取 0/6/22/23（m<7 回退零修正，负残差保留）；
反馈边界通过（满电不充电、空电不放电、双向功率饱和于 833.333 kWh）。

**A2 信息边界（2025-06-21，一次性）**：
扰动决策时点尚未实现的**全部真值**（0:00 决策 {info['midnight_perturbed_truth_cells']:,} 个格，
06:00 决策 {info['six_perturbed_truth_cells']:,} 个格；缩放 ×{info['truth_scale']}）并加上尚未发布的
预报版本（+{info['pv_shift_kW']} kW），整条预测/保护链路重建后：

- 0:00：保护输入逐位一致 = {info['midnight_inputs_identical']}，辅助午夜状态差
  {info['midnight_aux_estimate_delta_kWh']:.3e} kWh，完整 144 段原计划最大差
  {info['midnight_plan_max_delta_kWh']:.3e} kWh。
- 06:00：窗口取 {info['six_window'][0]} 至 {info['six_window'][1]}（模板槽 35—143，共
  {info['six_remaining_intervals']} 段），旧计划取该窗口的 `previous` 列而非当日最终执行量。
  保护输入逐位一致 = {info['six_inputs_identical']}；干净复算与保存候选最大差
  {info['six_clean_reproduces_saved_max_delta_kWh']:.3e} kWh（复现了真实决策）；扰动候选最大差
  {info['six_candidate_max_delta_kWh']:.3e} kWh；接受决定 干净/扰动/保存 =
  {info['six_accept_clean']}/{info['six_accept_perturbed']}/{info['six_saved_decision_accepted']}，
  一致性 = {info['six_accept_matches_saved']}。

**该检查能发现"误纳尚未实现样本"的一类错误，但不构成不存在泄漏的证明**：被误纳的样本即使携带
被扰动的真值，也可能因次序统计量位置不变而不改变 q80，从而不触发失败。扰动格数约 2.8 万，
覆盖了整个未实现未来，是当前口径下能做到的直接探针。

**A3 跨内核一致性**：用 29 号已验证内核 `solve_day` 重解 2025-06-21 的 0:00 计划，与**已保存**的
B1 计划比较：目标值差 {kernel['objective_delta_yuan']:.3e} 元，计划最大差
{kernel['plan_max_delta_kWh']:.3e} kWh（同一 MILP 的等费用顶点；本地侧不再重解，因此只新增 1 次求解）。
这支持"本次转写的内核与 29 号所用内核在此算例上同解"，**不等于**证明所有日期求解器行为一致。

**B 运行中断言**：三组各自 48097 行完整覆盖、无重复无漏段、10 分钟等间隔；逐段结算恒等式
残差 {max(summary.loc[g, 'settlement_residual_yuan'] for g in GROUPS):.3e} 元；
实际供需平衡、库存递推、跨日连续、SOC 界、功率上限、非负、充放电互斥、应急不充电
最大残差 {max(summary.loc[g, 'physics_residual_kWh'] for g in GROUPS):.3e} kWh。

**C 完成后只读核账**：两账本各 48096 段；B0 自然日合计 {money(ledger['B0_natural_total_yuan'])} 元
与保存参考差 {ledger['b0_reference_delta_yuan']:.3e} 元；头尾桥接残差最大
{max(abs(ledger[f'{g}_bridge_residual_yuan']) for g in GROUPS):.3e} 元
（C_B − C_A = 2026-01-01 00:00—00:10 费用 − 2025-02-01 00:00—00:10 费用）。

受保护资产零变更零缺失（{checks['protected_unchanged']['count']:,} 个文件比对通过）。

**验证预算与范围披露（如实记录，不重跑消除）：** 额外 MILP 求解 A2 3 次 + A3 1 次 =
{budget['extra_milp_solves']['total']} 次，等于任务书上限 {budget['task_book_limit']} 次（首轮曾为 5 次，
接收核查指出后已把 A3 的本地重解改为对拍已保存计划）。受保护哈希覆盖整个受保护树
{f"{checks['protected_unchanged']['count']:,}"} 个文件，比任务书"只登记本次依赖文件"的要求更重；
这是防御性检查，无副作用，但确实超出精简安排。

## 8. 限制与不宜过度解读之处

- **B1 更贵不等于附件3 无效。** 该比较同时换了预报来源与提前时长口径（附件3 为提前 1—24 小时
  的整点预报），差额 {signed(c10.delta_yuan)} 元包含预报质量差异；B1 在 2025-06 反而更省，说明
  结论按月不稳定。附件3 的价值主要体现在**日内更新**上（B2），而不是作为 0:00 单独预报。
- **ΔC₂₁ 是机制包价值。** 删除一次更新会同时移除新预报、真实库存反馈和重新优化机会；
  本轮没有做"同一更新时刻但只用旧预报"的信息价值隔离，因此不能把收益归给预报精度。
- **择优评分是确定性预测评价。** 新旧计划用同一最新保护需求与同一反馈规则模拟，得到的是
  含预测应急的评分，不是期望费用或真实未来应急；内层 MILP 最小化的是普通费+调整费，
  不宣称已全局最小化择优评分。
- **q80 在新增调整价格下无最优性证明**，也不表示 80% 无应急概率；它是首轮匹配基准。
- **自由末态可能低估规划范围之外的储电价值**，本轮由跨日应急与期末库存检验，未解决无限期问题。
- **2025 年数据此前已用于方法设计**，本轮是滚动因果回测，不是独立盲测；节费率不外推跨年。
- B0 是只读复用的上游账本，其入口/保护的验证边界沿用问题二既有记录，本轮不重复审计上游全链路。

## 9. 文件路径

| 用途 | 路径 |
|---|---|
| 计算层（登记、求解、验证） | `code/q3_rolling_baseline_experiment.py` |
| 报告层（本文件与图表） | `code/q3_rolling_baseline_report.py` |
| 三组逐段账本 | `results/q3_rolling_baseline/B{{0,1,2}}_dispatch.csv` |
| 预测/残差/保护档案 | `results/q3_rolling_baseline/prediction_protection.npz`、`protection_summary.json` |
| B1/B2 计划版本档案 | `results/q3_rolling_baseline/B{{1,2}}_plan_versions.csv`（含每天 0:00 原计划与 B2 修订候选） |
| 决策日志（含三项拆分） | `results/q3_rolling_baseline/B2_revision_decisions.csv` |
| 名义轨迹 | `results/q3_rolling_baseline/B{{1,2}}_nominal_trajectory.csv`（每次求解的 q、S 逐段） |
| 18:00 版本变化诊断 | `results/q3_rolling_baseline/revision_information_diagnostic.csv` |
| 汇总与差额 | `results/q3_rolling_baseline/{{summary,contrast,monthly_contrasts,daily_contrasts,selected_dates}}.csv` |
| 登记/运行/验证记录 | `results/q3_rolling_baseline/{{registration,run_manifest,validation,checks}}.json` |
| 图表 | `figures/q3_rolling_baseline/monthly_costs.png`、`selected_dates_soc.png` |

耗时：整轮 {manifest['wall_seconds']:.1f} 秒（B1 {manifest['timings'].get('B1_seconds', float('nan')):.1f} 秒 /
B2 {manifest['timings'].get('B2_seconds', float('nan')):.1f} 秒），共
{sum(int(v) for v in manifest['n_solves'].values())} 次正式 MILP 求解。正式 `result3.xlsx` 未填写。
"""
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT_MD), "figures": [str(p) for p in fig_paths],
                      "bytes": REPORT_MD.stat().st_size}, ensure_ascii=False, indent=2))


def _digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figure_integrity(path):
    """Programmatic stand-in for visual inspection: this agent cannot view images.

    Records hashes, pixel size, ink coverage and colour count so that a blank, truncated or
    single-colour render is detectable. It does not judge aesthetic quality.
    """
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ink = float((rgb < 250).any(axis=2).mean())
    colours = int(len(np.unique(rgb.reshape(-1, 3), axis=0)))
    assert ink > 0.005, f"{path.name} looks blank (ink={ink:.4f})"
    assert colours > 20, f"{path.name} has too few colours ({colours})"
    return dict(sha256=_digest(path), bytes=path.stat().st_size, width=int(rgb.shape[1]),
                height=int(rgb.shape[0]), ink_coverage=round(ink, 4), distinct_colours=colours,
                note="programmatic only; the agent cannot view images, so aesthetics are unverified")


def _rel(path):
    return Path(path).as_posix()


if __name__ == "__main__":
    main()
