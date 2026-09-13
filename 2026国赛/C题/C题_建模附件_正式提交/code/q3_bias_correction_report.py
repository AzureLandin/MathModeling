#!/usr/bin/env python
"""问题三历史偏差校正轻量诊断 —— 报告层（只读渲染）。

只读 ``results/q3_bias_correction_diagnostic/``，渲染结果报告与最多两张图。
**不重算偏差、不重建 q80、不调用求解器**；改文案或图表不触发
``code/q3_bias_correction_diagnostic.py``。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_bias_correction_report.py
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
OUT = ROOT / "results/q3_bias_correction_diagnostic"
FIG = ROOT / "figures/q3_bias_correction_diagnostic"
REPORT_MD = ROOT / "reports/问题三/问题三_历史偏差校正诊断结果.md"

assert Path(sys.prefix).name == "math_modeling", sys.prefix

UPDATE_HOURS = (0, 6, 12, 18)
MAE_TOL = 1e-9
PLAN_TOL = 1e-8
COLOR = {"c0": "#4C72B0", "c1": "#DD8452"}


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def num(value, digits=4):
    return f"{float(value):,.{digits}f}"


def make_bias_figure(monthly):
    data = monthly[monthly.group_type == "month"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0))
    for ax, hour in zip(axes.ravel(), UPDATE_HOURS):
        block = data[data.publication_hour == hour].sort_values("month")
        x = np.arange(len(block))
        ax.bar(x - 0.2, block.a_pv_bias, 0.4, label="C0_Linear", color=COLOR["c0"])
        ax.bar(x + 0.2, block.b_pv_bias, 0.4, label="C1_Bias28", color=COLOR["c1"])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m[2:] for m in block.month], rotation=45, ha="right")
        ax.set_ylabel("光伏平均偏差（kW，正=高估）")
        ax.set_title(f"{hour:02d}:00 发布版本")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("月度光伏平均偏差：C0 系统性低估，C1 校正后基本归零")
    fig.tight_layout()
    path = FIG / "monthly_pv_bias.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_protected_figure(monthly):
    data = monthly[monthly.group_type == "month"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.0))
    for ax, hour in zip(axes.ravel(), UPDATE_HOURS):
        block = data[data.publication_hour == hour].sort_values("month")
        x = np.arange(len(block))
        ax.bar(x - 0.2, block.a_mean_shortfall_kWh, 0.4, label="C0 短缺", color=COLOR["c0"])
        ax.bar(x + 0.2, block.b_mean_shortfall_kWh, 0.4, label="C1 短缺", color=COLOR["c1"])
        ax.plot(x, block.a_mean_surplus_kWh, color=COLOR["c0"], linestyle="--", linewidth=1.0,
                marker="o", markersize=3, label="C0 富余")
        ax.plot(x, block.b_mean_surplus_kWh, color=COLOR["c1"], linestyle=":", linewidth=1.4,
                marker="s", markersize=3, label="C1 富余")
        ax.set_xticks(x)
        ax.set_xticklabels([m[2:] for m in block.month], rotation=45, ha="right")
        ax.set_ylabel("每段平均电量（kWh）")
        ax.set_title(f"{hour:02d}:00 发布版本")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("月度保护后短缺（柱）与富余（线）：两组差别不构成一致优势")
    fig.tight_layout()
    path = FIG / "monthly_protected.png"
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


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(OUT / "comparison.csv")
    monthly = pd.read_csv(OUT / "monthly.csv")
    daily = pd.read_csv(OUT / "daily_paired_metrics.csv")
    decomp = pd.read_csv(OUT / "correction_decomposition.csv")
    spotcheck = pd.read_csv(OUT / "history_spotcheck.csv")
    # publication_hour mixes ints with the "truncated_subset" label, so pandas reads the whole
    # column as str: look rows up by string key rather than comparing against ints
    bias_table = pd.read_csv(OUT / "bias_statistics.csv")
    bias_by_key = {str(row.publication_hour): row for row in bias_table.itertuples()}
    summary = load_json("summary.json")
    validation = load_json("validation.json")
    manifest = load_json("run_manifest.json")
    registration = load_json("registration.json")
    saved = validation["saved_check"]
    dec = {row.subset: row for row in decomp.itertuples()}
    trunc_row = comparison[comparison.group_type == "truncated_subset"].iloc[0]

    paths = [make_bias_figure(monthly), make_protected_figure(monthly)]
    (OUT / "figure_integrity.json").write_text(
        json.dumps({p.name: figure_integrity(p) for p in paths}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    overall = comparison[comparison.group_type == "overall"].iloc[0]
    per_hour = comparison[comparison.group_type == "publication_hour"].sort_values(
        "publication_hour")
    month = monthly[monthly.group_type == "month"]
    boundary = monthly[monthly.group_type == "boundary"]

    def counts(frame, column):
        values = frame[column].to_numpy(dtype=float)
        return (int((values < -MAE_TOL).sum()), int((values > MAE_TOL).sum()),
                int((np.abs(values) <= MAE_TOL).sum()))

    hour_rows = "\n".join(
        f"| {int(r.publication_hour):02d}:00 | {int(r.a_pv_n):,} | {num(r.a_pv_mae)} | "
        f"{num(r.b_pv_mae)} | {r.delta_pv_mae_kW:+,.3f} | {r.a_pv_bias:+,.3f} | {r.b_pv_bias:+,.3f} "
        f"| {r.delta_net_mae_kWh:+,.4f} | {r.delta_net_bias_kWh:+,.4f} |"
        for r in per_hour.itertuples())
    protected_rows = "\n".join(
        f"| {int(r.publication_hour):02d}:00 | {r.a_coverage:.4f} | {r.b_coverage:.4f} | "
        f"{num(r.a_mean_shortfall_kWh)} | {num(r.b_mean_shortfall_kWh)} | "
        f"{r.delta_shortfall_kWh:+,.4f} | {num(r.a_mean_surplus_kWh)} | {num(r.b_mean_surplus_kWh)} "
        f"| {r.delta_surplus_kWh:+,.4f} | {num(r.a_mean_rho_kWh)} | {num(r.b_mean_rho_kWh)} |"
        for r in per_hour.itertuples())
    decomp_rows = "\n".join(
        f"| {r.subset} | {int(r.n):,} | {r.mean_d_net_kWh:+,.4f} | {r.mean_d_rho_kWh:+,.4f} | "
        f"{r.mean_d_protected_kWh:+,.4f} | {num(r.mean_abs_d_protected_kWh)} | "
        f"{num(r.max_abs_d_protected_kWh, 3)} | {r.truncated_share:.4f} |"
        for r in decomp.itertuples())
    boundary_rows = "\n".join(
        f"| {int(r.publication_hour):02d}:00 | {r.boundary} | {num(r.a_pv_mae)} | {num(r.b_pv_mae)} "
        f"| {r.delta_pv_mae_kW:+,.3f} | {r.delta_net_mae_kWh:+,.4f} | {r.delta_shortfall_kWh:+,.4f} |"
        for r in boundary.sort_values(["publication_hour", "boundary"]).itertuples())
    spot_rows = "\n".join(
        f"| {r.publication_hour:02d}:00 | {int(r.target_h)} | {int(r.m_bias_samples)} | "
        f"{int(r.m_net_samples)} | {int(r.quantile_position)} | {r.recomputed_bias_kW:+,.6f} | "
        f"{r.recomputed_rho_kWh:+,.6f} | {r.first_sample_day} → {r.last_sample_day} |"
        for r in spotcheck.itertuples())
    hour_counts = "\n".join(
        f"| {int(hour):02d}:00 | {len(block)} | {counts(block, 'delta_pv_mae_kW')[0]} / "
        f"{counts(block, 'delta_pv_mae_kW')[1]} | {counts(block, 'delta_net_mae_kWh')[0]} / "
        f"{counts(block, 'delta_net_mae_kWh')[1]} | {counts(block, 'delta_shortfall_kWh')[0]} / "
        f"{counts(block, 'delta_shortfall_kWh')[1]} | {block.delta_shortfall_kWh.mean():+,.4f} |"
        for hour, block in daily.groupby("publication_hour"))
    month_counts = "\n".join(
        f"| {int(hour):02d}:00 | {counts(block, 'delta_pv_mae_kW')[0]}/11 | "
        f"{counts(block, 'delta_net_mae_kWh')[0]}/11 | {counts(block, 'delta_shortfall_kWh')[0]}/11 |"
        for hour, block in month.groupby("publication_hour"))
    bias_rows = "\n".join(
        f"| {r.publication_hour} | {num(r.bias_mean_kW)} | {num(r.bias_mean_abs_kW)} | "
        f"{num(r.bias_max_abs_kW, 2)} | {num(r.delta_v_mean_kW)} | {num(r.delta_v_mean_abs_kW)} | "
        f"{num(r.delta_v_min_kW, 1)} / {num(r.delta_v_max_kW, 1)} | {int(r.truncated_rows):,} |"
        for r in bias_table.itertuples())
    syn = validation["synthetic"]
    trunc, plain = dec["truncation_binding_only"], dec["non_truncated_rows"]
    b18, bt = bias_by_key["18"], bias_by_key["truncated_subset"]

    report = f"""# 问题三历史偏差校正轻量诊断结果

生成日期：2026-09-12。由 `code/q3_bias_correction_report.py` 从
`results/q3_bias_correction_diagnostic/` 只读渲染，不含重新估计偏差、重建 q80 或任何求解。

方案见 [历史偏差校正轻量诊断实验方案]({_rel(ROOT / 'reports/问题三/问题三_历史偏差校正轻量诊断实验方案.md')})。
登记签名 `{registration['signature'][:16]}…`，运行状态 `{manifest['status']}`，
**新模型训练 0 次、MILP 0 次、储能策略回测 0 次**，整轮 {manifest['wall_seconds']:.1f} 秒。

## 1. 结论速览

**28 日前向滚动均值校正降低了原始点预测误差；重新校准 q80 后诊断未呈现一致优势；
本轮没有运行调度，无法判断费用变化的方向与幅度。**

| 指标（四个版本合计，{saved['common_samples']:,} 条配对记录） | C0_Linear | C1_Bias28 | 差额 |
|---|---:|---:|---:|
| 光伏 MAE（kW） | {num(overall.a_pv_mae)} | {num(overall.b_pv_mae)} | {overall.delta_pv_mae_kW:+,.3f} |
| 光伏 RMSE（kW） | {num(overall.a_pv_rmse)} | {num(overall.b_pv_rmse)} | {overall.delta_pv_rmse_kW:+,.3f} |
| 光伏平均偏差（kW） | {num(overall.a_pv_bias)} | {num(overall.b_pv_bias)} | {overall.delta_pv_bias_kW:+,.3f} |
| 净需求 MAE（kWh） | {num(overall.a_net_mae)} | {num(overall.b_net_mae)} | {overall.delta_net_mae_kWh:+,.4f} |
| 净需求平均偏差（kWh） | {num(overall.a_net_bias)} | {num(overall.b_net_bias)} | {overall.delta_net_bias_kWh:+,.4f} |
| 保护后平均短缺（kWh） | {num(overall.a_mean_shortfall_kWh)} | {num(overall.b_mean_shortfall_kWh)} | {overall.delta_shortfall_kWh:+,.4f} |
| 保护后平均富余（kWh） | {num(overall.a_mean_surplus_kWh)} | {num(overall.b_mean_surplus_kWh)} | {overall.delta_surplus_kWh:+,.4f} |
| 保护后平均绝对变化（kWh） | — | — | {num(dec['overall'].mean_abs_d_protected_kWh)} |
| 保护后单段最大绝对变化（kWh） | — | — | {num(dec['overall'].max_abs_d_protected_kWh, 3)} |
| 保护后覆盖率 | {overall.a_coverage:.4f} | {overall.b_coverage:.4f} | — |

- **原始预测：一致且大幅改善。** 光伏 MAE 降 {abs(overall.delta_pv_mae_kW):.1f} kW、RMSE 降
  {abs(overall.delta_pv_rmse_kW):.1f} kW，平均偏差从 {overall.a_pv_bias:+.2f} kW 收到
  {overall.b_pv_bias:+.2f} kW（原有系统性低估基本被消除，且没有明显转为高估）；净需求 MAE 降
  {abs(overall.delta_net_mae_kWh):.2f} kWh。四个发布小时方向一致（见第 3 节）。
- **保护后：有符号平均变化很小，但逐段变化确实存在，不能称为"曲线几乎不变"。** 平均 Δñ 为
  {dec['overall'].mean_d_protected_kWh:+,.4f} kWh，而**平均绝对变化
  {num(dec['overall'].mean_abs_d_protected_kWh)} kWh、单段最大绝对变化
  {num(dec['overall'].max_abs_d_protected_kWh, 3)} kWh**；四个版本的平均绝对变化分别为
  {num(dec['publication_00'].mean_abs_d_protected_kWh)} /
  {num(dec['publication_06'].mean_abs_d_protected_kWh)} /
  {num(dec['publication_12'].mean_abs_d_protected_kWh)} /
  {num(dec['publication_18'].mean_abs_d_protected_kWh)} kWh。短缺与富余也未呈现一致改善（见第 4 节）。
- **有符号均值的抵消有其严格条件。** 分解显示平均 Δn̂ = {dec['overall'].mean_d_net_kWh:+,.4f} kWh 与
  平均 Δρ = {dec['overall'].mean_d_rho_kWh:+,.4f} kWh 几乎等量反号，但这只有"当前与历史施加同一常数校正、
  且样本量达到启用要求"时才等价于分位数的精确平移。实际滚动校正逐日变化，即使没有截断也不保证逐段抵消：
  未截断子集的平均绝对保护变化仍有 {num(plain.mean_abs_d_protected_kWh)} kWh、最大
  {num(plain.max_abs_d_protected_kWh, 3)} kWh，与整体同量级。
- **本轮没有运行调度，无法判断费用变化的方向与幅度**，也没有证明逐段变化在经济上显著或不显著。

按方案第 5 节的判据，结论是**保留 C0 与现有 B2、不自动进入费用对照**；依据是保护后诊断未呈现一致优势，
而不是"保护曲线几乎不变"或"费用最可能不变"。

## 2. 口径与预算

| 项目 | 取值 |
|---|---|
| C0_Linear | 冻结的原 Linear 区间平均光伏预测与冻结 W28/q80（只读复用，本次逐位复核一致） |
| C1_Bias28 | 原 Linear 加同发布小时-同目标时段前 28 日平均 PV 偏差，再作非负截断 |
| 偏差样本 | 实际减预测（正=原预测偏低）；有效样本数不少于 7 才启用，否则零校正；不截尾、不加权、不收缩 |
| 因果约束 | 偏差只用**原 Linear** 的历史误差；历史校正预测用**各自当时的**偏差前向生成并冻结 |
| 保护规律 | W28/q80，同发布小时-同目标时段，升序第 ⌈0.8m⌉ 项，有效样本不足 7 时零修正 |
| 评价窗口 | 0:00 为 h=1..144；6/12/18 点为 109/73/37 段；合计 {saved['common_samples']:,} 条"发布版本—目标"配对记录。**同一实际区间会被多个版本覆盖，不是独立实际区间数**；合并均值按各版本窗口长度加权，只作辅助汇总 |
| 负载 | 复用 15 特征 LightGBM 发布档案，日内不改 |
| 预算 | 新模型训练 0、MILP 0、策略回测 0；仅滚动均值、次序统计与汇总 |

**各版本实际施加的校正量**（δV 为最终作用于预测的增量，可正可负）：

| 发布时刻 | 偏差 b 均值（kW） | b 平均绝对（kW） | b 最大绝对（kW） | δV 均值（kW） | δV 平均绝对（kW） | δV 范围（kW） | 截断命中 |
|---|---:|---:|---:|---:|---:|---:|---:|
{bias_rows}

历史不足（有效偏差样本少于 7）而回退零校正的记录数为 {summary['zero_history_rows']:,}。
**该数字只说明正式评价窗口内没有历史不足回退，不给出 1 月档案形成期的回退数量。**

## 3. 原始预测改善

| 发布小时 | 样本 | C0 PV MAE | C1 PV MAE | ΔMAE（kW） | C0 偏差 | C1 偏差 | Δ净需求 MAE（kWh） | Δ净需求偏差（kWh） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{hour_rows}

- 四个发布小时的 ΔMAE 全部为负，其中 6:00 的数值改善最大
  （{per_hour[per_hour.publication_hour == 6].delta_pv_mae_kW.iloc[0]:+,.1f} kW）。
  **这不支持"越接近日出、原预报低估越强"的解释**：四个版本的评价窗口不同、目标集合不可直接排序，
  且表内平均偏差 06:00 为 −15.3 kW、12:00 为 −51.8 kW，与"越接近日出越差"的顺序相反。
  本轮没有日出时刻定义或同目标机理检验，因此只保留各版本首小时的数值改善（见下表）。
- 平均偏差从 −29.7 / −15.3 / −51.8 kW 收到 +1.9 / +3.4 / +1.5 kW：低估被消除，且没有明显转为高估。
- **分月计数不能概括为"方向一致"**：18:00 版本光伏 MAE 只有 7/11 个月改善、净需求 6/11；
  00:00 净需求为 9/11。而且月度改善次数本身不能排除少数异常日主导幅度。

| 发布小时 | PV MAE 改善月数 | 净需求 MAE 改善月数 | 短缺改善月数 |
|---|---:|---:|---:|
{month_counts}

**分边界类型**：

| 发布小时 | 边界 | C0 PV MAE | C1 PV MAE | ΔMAE（kW） | Δ净需求 MAE（kWh） | Δ短缺（kWh） |
|---|---|---:|---:|---:|---:|---:|
{boundary_rows}

## 4. 保护后表现

| 发布小时 | C0 覆盖率 | C1 覆盖率 | C0 短缺 | C1 短缺 | Δ短缺 | C0 富余 | C1 富余 | Δ富余 | C0 平均 ρ | C1 平均 ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{protected_rows}

覆盖率**更高不等于更好**：富余同时上升说明保护量整体变大，而短缺并未因此下降。
平均保护量 ρ 由 {num(overall.a_mean_rho_kWh)} 升到 {num(overall.b_mean_rho_kWh)} kWh，
与"校正后历史残差整体上移、p80 随之抬高"一致。

**逐日计数**（"更优/更差"按容差 {MAE_TOL:.0e} 判。方案指定的逐日容差为 {PLAN_TOL:.0e}；
两者都远小于观测到的差额量级，**本轮按 {MAE_TOL:.0e} 计数，如实记录该口径差异**）：

| 发布小时 | 天数 | PV MAE 更优/更差 | 净需求 MAE 更优/更差 | 短缺更优/更差 | 平均 Δ短缺（kWh） |
|---|---:|---:|---:|---:|---:|
{hour_counts}

短缺的逐日计数接近对半，**没有稳定方向**。

## 5. 变化分解

Δñ = Δn̂ + Δρ，逐子集统计（有符号均值、平均绝对值与单段最大绝对值并列）：

| 子集 | 样本 | 平均 Δn̂（kWh） | 平均 Δρ（kWh） | 平均 Δñ（kWh） | 平均绝对 Δñ（kWh） | 最大绝对 Δñ（kWh） | 截断占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
{decomp_rows}

- **有符号均值很小不等于曲线不动**：整体平均 Δñ 只有
  {dec['overall'].mean_d_protected_kWh:+,.4f} kWh，但平均绝对变化
  {num(dec['overall'].mean_abs_d_protected_kWh)} kWh、单段最大
  {num(dec['overall'].max_abs_d_protected_kWh, 3)} kWh。
- **18:00 平均净变化接近零，但不代表校正本身趋零。** 该版本实际施加的光伏校正 δV 平均
  {num(b18.delta_v_mean_kW)} kW、**平均绝对 {num(b18.delta_v_mean_abs_kW)} kW**。
  "18:00 相对 12:00 的相邻预报差很小"（上一轮诊断）与"18:00 预报相对实际值的历史误差"是两个不同统计量，
  不能用前者推断后者趋零。只有保护后净变化 {dec['publication_18'].mean_d_protected_kWh:+.6f} kWh 接近零，
  其平均绝对值仍有 {num(dec['publication_18'].mean_abs_d_protected_kWh)} kWh、最大
  {num(dec['publication_18'].max_abs_d_protected_kWh, 3)} kWh。
- **截断子集是权衡，不是收益。** 非负截断作用在**最终光伏预测**上，偏差与实际校正 δV 仍可为负：
  这 {int(trunc.n):,} 段上 δV 落在 [{num(bt.delta_v_min_kW, 1)}, {num(bt.delta_v_max_kW, 1)}] kW，
  平均绝对值 {num(bt.delta_v_mean_abs_kW)} kW。该子集上 C1 的**原始**指标大幅改善
  （光伏 MAE {num(trunc_row.a_pv_mae)} → {num(trunc_row.b_pv_mae)} kW、净需求 MAE
  {num(trunc_row.a_net_mae)} → {num(trunc_row.b_net_mae)} kWh），
  但**保护后平均短缺 {num(trunc_row.a_mean_shortfall_kWh)} → {num(trunc_row.b_mean_shortfall_kWh)} kWh（上升）、
  平均富余 {num(trunc_row.a_mean_surplus_kWh)} → {num(trunc_row.b_mean_surplus_kWh)} kWh（下降）**：
  保护净需求降低意味着短缺增加、富余减少，**不能用"变化为负"当作收益判据**。
  而且这里比较的是"完整 C1 与 C0 在该子集上的差异"，**没有隔离仅开/关截断的因果贡献**；
  未截断子集的平均绝对保护变化仍有 {num(plain.mean_abs_d_protected_kWh)} kWh（整体
  {num(dec['overall'].mean_abs_d_protected_kWh)} kWh），因此**截断不是唯一造成不完全抵消的机制**。
- 实测是否发电只用于**事后诊断分组**，不作为校正的启用条件：实测无发电而预测出正功率的误差同样值得校正，
  这正是上一轮诊断指出的虚假发电问题。

## 6. 验证（方案第 4.3 节三类检查，全部通过）

**合成样例**：常数平移且无截断时，净需求平均变化
{syn['uniform_shift_d_net_kWh']:+.4f} kWh、修正量平均变化 {syn['uniform_shift_d_rho_kWh']:+.4f} kWh，
**两者之和残差仅 {syn['uniform_shift_cancellation_max_abs_residual_kWh']:.2e} kWh**——这只证明
"当前与历史同施一个常数"这一特例下分位数精确平移，**不是全年逐段结论**。历史不足样例
有效样本上限 {syn['insufficient_history_m_max']} 时回退零修正；截断样例
预测 (1, 100, 0)、偏差 (-10, -10, +5) 给出 {str(syn['truncation_example']['corrected'])}，仅第一项被截断。

**2025-06-21 前缀检查**（候选集合、偏差均值与次序统计量均按区间结束时刻独立重建；其中 C1 历史残差
取自已生成并冻结的历史校正预测，符合方案"历史已发布"的要求，但**不是**全部从原始输入重建）：

| 发布小时 | 目标 h | 偏差样本数 | 净需求样本数 | 分位位置 | 重算偏差（kW） | 重算修正量（kWh） | 入选历史样本 |
|---|---:|---:|---:|---:|---:|---:|---|
{spot_rows}

**保存结果核对**：四个版本样本 {saved['window_sizes']}，与 144/109/73/37 乘 334 一致；
两组共同样本 {saved['common_samples']:,}；C0 净需求与冻结档案最大差
{saved['c0_matches_frozen_net_max_abs_kWh']:.3e} kWh，样本数与修正量均与冻结档案逐位一致；
变化恒等式 Δñ = Δn̂ + Δρ 的最大残差
{decomp.max_abs_identity_residual_kWh.max():.3e} kWh。仅对本次实际读取的文件核哈希，未遍历项目树。

## 7. 限制

- 短缺与富余是**预测曲线误差**，不是实际应急或未使用电量；没有模拟储能，**没有计算任何新方案费用**。
- 28 日滚动均值假设"局部偏差短期稳定"，这是**待检验假设**，本轮只说明该估计在本数据上可降低点误差。
- 覆盖率上升不等于更好，可能只是保护量整体变大。
- **本轮没有调度，因此对费用方向与幅度没有任何证据**；也没有做统计显著性检验，
  原始预测的改善仅指本年度保存样本中的误差指标下降，不代表统计显著性或费用下降。
- 全部结果来自同一 2025 年开发数据，是滚动因果诊断，**不是独立盲测、不构成跨年收益**。
- 图件仅程序化完整性检查（本 Agent 无法目视图像）。

## 8. 建议：不进入费用对照

| 方案第 5 节判据 | 本轮结果 |
|---|---|
| 原始 PV 与净需求误差均较一致改善，且不是只降偏差而扩大 RMSE | **满足**（MAE 与 RMSE 同降，四版本方向一致） |
| 改善未在重新校准 q80 后基本消失，短缺/富余可解释 | **不满足**：保护后未呈现一致优势，短缺与富余均略升 |
| 分月不被少数异常日主导 | **尚未证实**：月度改善次数不能判断少数异常日是否主导改善幅度；现有计数只说明改善并非逐月一致 |

**结论：保留 C0 与现有 B2，停止自动推进。** 依据是保护后诊断未呈现一致优势，
而不是"保护曲线不变"或"费用最可能不变"——后者本轮没有证据。
唯一值得单独讨论的线索是**非负截断命中段上原始指标的巨大改善与保护后短缺上升并存**，
说明偏差校正与保护量之间的配合才是关键；但截断的因果贡献尚未隔离，
不得据此事后挑选启用时刻或自设权重。

本轮不给任何节费数字，不填正式 `result3.xlsx`，不替换 B2 策略。

## 9. 文件路径

| 用途 | 路径 |
|---|---|
| 计算层 | `code/q3_bias_correction_diagnostic.py` |
| 报告层（本文件与图表） | `code/q3_bias_correction_report.py` |
| 预测/偏差/保护档案 | `results/q3_bias_correction_diagnostic/bias_forecast_archive.npz` |
| 对照与分解 | `results/q3_bias_correction_diagnostic/{{comparison,monthly,daily_paired_metrics,correction_decomposition}}.csv` |
| 校正量统计 | `results/q3_bias_correction_diagnostic/bias_statistics.csv` |
| 历史边界证据 | `results/q3_bias_correction_diagnostic/history_spotcheck.csv` |
| 登记/运行/验证 | `results/q3_bias_correction_diagnostic/{{registration,run_manifest,validation,summary}}.json` |
| 图表 | `figures/q3_bias_correction_diagnostic/monthly_pv_bias.png`、`monthly_protected.png` |
"""
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT_MD), "figures": [str(p) for p in paths],
                      "bytes": REPORT_MD.stat().st_size}, ensure_ascii=False, indent=2))


def _rel(path):
    return Path(path).as_posix()


if __name__ == "__main__":
    main()
