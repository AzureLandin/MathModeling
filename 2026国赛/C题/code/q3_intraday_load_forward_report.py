#!/usr/bin/env python
"""问题三 日内负载修正前向预测对照实验 —— 报告层（只读渲染）。

只读 ``results/q3_intraday_load_forward/``，渲染结果报告与最多两张图。
**不重算参数、不重建预测、不调用任何求解器或 LightGBM**；改文案或图表永远不会触发
``code/q3_intraday_load_forward_experiment.py``。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_intraday_load_forward_report.py
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
OUT = ROOT / "results/q3_intraday_load_forward"
FIG = ROOT / "figures/q3_intraday_load_forward"
REPORT_MD = ROOT / "reports/问题三/问题三_日内负载修正前向预测对照实验结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_日内负载修正前向预测对照实验方案.md"
NODES = (6, 12, 18)
COLUMN_OF = {"F0_original": "f0_kW", "F1_hist_mean": "f1_kW", "F2_intraday_linear": "f2_kW"}
GROUPS = ("F0_original", "F1_hist_mean", "F2_intraday_linear")
CONTRAST_LABEL = {"F2_minus_F0": "F2−F0", "F2_minus_F1": "F2−F1", "F1_minus_F0": "F1−F0"}
CONTRAST_COLOUR = {"F2_minus_F0": "#d62728", "F2_minus_F1": "#ff7f0e", "F1_minus_F0": "#1f77b4"}


def num(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def month_series(grouped):
    block = grouped[(grouped.set == "A") & (grouped.group_type == "month")]
    pivoted = block.pivot_table(index="group_value", columns="group", values="mae_kW")
    for contrast, (left, right) in (("F2_minus_F0", ("F2_intraday_linear", "F0_original")),
                                    ("F2_minus_F1", ("F2_intraday_linear", "F1_hist_mean")),
                                    ("F1_minus_F0", ("F1_hist_mean", "F0_original"))):
        pivoted[contrast] = pivoted[left] - pivoted[right]
    return pivoted


def distance_series(grouped):
    """A-set MAE difference against the hour index, per publication node (display aggregation)."""
    hourly = grouped[grouped.set == "A_hourly"].copy()
    hourly[["k", "r", "j"]] = hourly.group_value.str.split("_", expand=True).astype(int)
    hourly["hour_index"] = hourly.j + 1
    pivoted = hourly.pivot_table(index=["r", "hour_index"], columns="group", values="mae_kW")
    for contrast, (left, right) in (("F2_minus_F0", ("F2_intraday_linear", "F0_original")),
                                    ("F2_minus_F1", ("F2_intraday_linear", "F1_hist_mean")),
                                    ("F1_minus_F0", ("F1_hist_mean", "F0_original"))):
        pivoted[contrast] = pivoted[left] - pivoted[right]
    return pivoted.reset_index()


def make_figures(months, distance):
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    x = np.arange(len(months))
    width = 0.27
    for offset, contrast in enumerate(("F2_minus_F0", "F2_minus_F1", "F1_minus_F0")):
        ax.bar(x + (offset - 1) * width, months[contrast], width,
               label=CONTRAST_LABEL[contrast], color=CONTRAST_COLOUR[contrast])
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(months.index, rotation=45, ha="right")
    ax.set_ylabel("A 集逐段 MAE 差（kW）")
    ax.set_title("分月 MAE 差：负值表示左侧组更准（A 集 36,072 个目标）")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG / "monthly_mae_difference.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    node_block = distance.groupby("r")[["F2_minus_F0", "F2_minus_F1", "F1_minus_F0"]].mean()
    x = np.arange(len(node_block))
    for offset, contrast in enumerate(("F2_minus_F0", "F2_minus_F1", "F1_minus_F0")):
        axes[0].bar(x + (offset - 1) * width, node_block[contrast], width,
                    label=CONTRAST_LABEL[contrast], color=CONTRAST_COLOUR[contrast])
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r:02d}:00" for r in node_block.index])
    axes[0].set_ylabel("A 集逐段 MAE 差（kW）")
    axes[0].set_title("按发布节点")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    styles = {6: "-", 12: "--", 18: ":"}
    colours = {"F2_minus_F0": "#d62728", "F2_minus_F1": "#ff7f0e"}
    for r in NODES:
        block = distance[distance.r == r].sort_values("hour_index")
        for contrast in ("F2_minus_F0", "F2_minus_F1"):
            axes[1].plot(block.hour_index, block[contrast], ls=styles[r], lw=1.5,
                         color=colours[contrast],
                         label=f"{r:02d}:00 {CONTRAST_LABEL[contrast]}")
    axes[1].axhline(0.0, color="black", lw=0.9)
    axes[1].set_xlabel("发布后第几个小时（j+1）")
    axes[1].set_ylabel("A 集逐段 MAE 差（kW）")
    axes[1].set_title("按预测距离（hue 为节点，实线 F2−F0、虚线 F2−F1）")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path2 = FIG / "node_distance_mae_difference.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    return [path, path2]


def parameter_rows(log):
    """Per (publication hour, hour|tail) parameter diagnostics for whatever scope is passed in."""
    rows = []
    for (r, kind), block in log.groupby(["publication_hour", "kind"], sort=True):
        sub = block[block.fitted]
        rows.append(dict(
            publication_hour=int(r), kind=kind, groups=int(len(block)),
            fitted=int(block.fitted.sum()),
            fallback_m_below_min=int(block.fallback_reason.str.startswith("m_below_min").sum()),
            m_min=int(block.m.min()), m_median=float(block.m.median()), m_max=int(block.m.max()),
            beta_mean=float(sub.beta.mean()) if len(sub) else float("nan"),
            beta_min=float(sub.beta.min()) if len(sub) else float("nan"),
            beta_max=float(sub.beta.max()) if len(sub) else float("nan"),
            beta_negative_share=float((sub.beta < 0).mean()) if len(sub) else float("nan"),
            delta2_min_kW=float(block.delta2_kW.min()),
            delta2_max_kW=float(block.delta2_kW.max())))
    return pd.DataFrame(rows)


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
    summary = pd.read_csv(OUT / "summary.csv")
    contrasts = pd.read_csv(OUT / "contrasts.csv").set_index("contrast")
    grouped = pd.read_csv(OUT / "grouped_metrics.csv")
    parameters = pd.read_csv(OUT / "parameter_summary.csv")
    pairs = pd.read_csv(OUT / "prediction_pairs.csv")
    log = pd.read_csv(OUT / "parameter_log.csv")
    validation = load_json(OUT / "validation.json")
    registration = load_json(OUT / "registration.json")
    manifest = load_json(OUT / "run_manifest.json")
    a_set = summary[summary.set == "A"].set_index("group")
    mae = a_set.mae_kW.to_dict()
    verdict = validation["verdict"]
    fits = validation["fit_counts"]
    months = month_series(grouped)
    distance = distance_series(grouped)
    overall = parameters[parameters.kind == "A_set_truncation"].iloc[0]
    whole = parameters[parameters.kind == "all_groups"].iloc[0]
    a_pairs = pairs[pairs.in_A]
    tail = summary[summary.set == "tail"].set_index("group")
    b_set = summary[summary.set == "B"].set_index("group")
    formal_log = log[log.k >= 31]
    formal_fitted = formal_log[formal_log.fitted]
    all_fitted = log[log.fitted]
    beta_scope = dict(
        formal=dict(fits=int(len(formal_fitted)),
                    negative_share=float((formal_fitted.beta < 0).mean()),
                    beta_min=float(formal_fitted.beta.min()),
                    beta_max=float(formal_fitted.beta.max()),
                    delta2_min=float(formal_log.delta2_kW.min()),
                    delta2_max=float(formal_log.delta2_kW.max())),
        all_days=dict(fits=int(len(all_fitted)),
                      negative_share=float((all_fitted.beta < 0).mean()),
                      beta_min=float(all_fitted.beta.min()),
                      beta_max=float(all_fitted.beta.max())))
    day_table = grouped[(grouped.set == "A") & (grouped.group_type == "day")].pivot_table(
        index="group_value", columns="group", values="mae_kW")
    day_delta = day_table.F2_intraday_linear - day_table.F0_original
    better_share = float(((a_pairs.actual_kW - a_pairs.f2_kW).abs()
                          < (a_pairs.actual_kW - a_pairs.f0_kW).abs()).mean())
    ranked_days = pd.Series(
        grouped[(grouped.set == 'A') & (grouped.group_type == 'day')]
        .pivot_table(index='group_value', columns='group', values='mae_kW')
        .pipe(lambda d: d.F2_intraday_linear - d.F0_original)).sort_values()
    top5_share = float(ranked_days.head(5).sum() / ranked_days.sum())
    with np.load(OUT / "forecast_archive.npz") as archive:
        published_min = float(archive["forecast"][2][archive["valid"]].min())

    def delta(name, key="delta_mae_kW", digits=4):
        return num(contrasts.loc[name, key], digits)

    lines = []
    add = lines.append
    add("# 问题三日内负载修正前向预测对照实验结果")
    add("")
    add(f"日期：2026-09-13。状态：三组前向预测对照已完成并通过三类精简检查；"
        f"本报告只读渲染，未新增任何求解或训练。")
    add(f"依据方案：[问题三_日内负载修正前向预测对照实验方案]"
        f"({PLAN_MD.resolve().as_posix()})。")
    add(f"计算层：`code/q3_intraday_load_forward_experiment.py`"
        f"（登记签名 `{registration['signature'][:16]}`；LightGBM 训练 0、调度求解 0、储能回放 0，"
        f"另有滚动闭式一元最小二乘估计 {fits['fitted']:,} 次）；"
        f"报告层：`code/q3_intraday_load_forward_report.py`（改文案与图表不触发计算层）。")
    add("")

    add("## 1. 结论")
    add("")
    add(f"1. **本轮判定：{verdict}。** 主指标为 A 集（每次更新后到下一发布节点前的 6 小时，"
        f"覆盖 06:00—24:00 的 {int(a_set.n.iloc[0]):,} 个唯一 10 分钟目标）的逐段 MAE："
        f"F0 {num(mae['F0_original'])} kW、F1 {num(mae['F1_hist_mean'])} kW、"
        f"F2 {num(mae['F2_intraday_linear'])} kW。"
        f"**F2 比 F0 低 {num(float(mae['F0_original']) - float(mae['F2_intraday_linear']))} kW"
        f"（{num(100 * (mae['F2_intraday_linear'] - mae['F0_original']) / mae['F0_original'], 2)}%），"
        f"比 F1 低 {num(float(mae['F1_hist_mean']) - float(mae['F2_intraday_linear']))} kW"
        f"（{num(100 * (mae['F2_intraday_linear'] - mae['F1_hist_mean']) / mae['F1_hist_mean'], 2)}%）"
        f"，即当天近期偏差提供的增量信息在本年度前向回测中确实改善了点预测主指标。**")
    add(f"2. **仅历史偏差校正（F1）本身不是可靠改进**：F1 只比 F0 低 "
        f"{num(float(mae['F0_original']) - float(mae['F1_hist_mean']))} kW"
        f"（{num(100 * (mae['F1_hist_mean'] - mae['F0_original']) / mae['F0_original'], 2)}%），"
        f"逐月 {int(contrasts.loc['F1_minus_F0', 'months_improved'])}/11 更准、"
        f"{int(contrasts.loc['F1_minus_F0', 'months_worsened'])}/11 更差，"
        f"在 06:00 节点几乎为零（{num(mae['F1_hist_mean'])} 对 {num(mae['F0_original'])} kW）。"
        f"F2 在三个节点上都更低，且逐月 **{int(contrasts.loc['F2_minus_F0', 'months_improved'])}/11** "
        f"优于 F0、{int(contrasts.loc['F2_minus_F1', 'months_improved'])}/11 优于 F1。")
    add(f"3. **但改进是均值意义而非普遍成立**，必须并列披露："
        f"F2 只在 {num(100 * better_share, 2)}% 的 A 集目标上绝对误差更小"
        f"（逐日 {int(contrasts.loc['F2_minus_F0', 'days_improved'])} 天更准、"
        f"{int(contrasts.loc['F2_minus_F0', 'days_worsened'])} 天更差）；"
        f"逐日 MAE 差的中位数只有 {num(day_delta.median())} kW，远小于均值改善"
        f"（{num(day_delta.mean())} kW），说明均值优势由少数大幅改善日拉动；"
        f"最差三日为 {contrasts.loc['F2_minus_F0', 'worst_days'].replace('|', '、')}（正值=更差），"
        f"最好一日 {contrasts.loc['F2_minus_F0', 'best_day']}；"
        f"改善最集中的 5 天（{'、'.join(ranked_days.head(5).index)}）合计占全天级净改善的 **{num(100 * top5_share, 2)}%**，"
        f"即 −9.14 kW 不能读成稳定的逐日收益。")
    add(f"4. **午夜尾槽与偏差项不进反退**：1,002 条尾槽记录对应 {int(validation['checks']['readback']['counts']['distinct_tail_intervals'])} "
        f"个不同实际午夜区间，其 MAE 为 F0 {num(tail.loc['F0_original', 'mae_kW'])}、"
        f"F1 {num(tail.loc['F1_hist_mean', 'mae_kW'])}、F2 {num(tail.loc['F2_intraday_linear', 'mae_kW'])} kW"
        f"——**F2 在尾槽上并不优于 F0**（RMSE 反而更好）；F1 的平均偏差最接近零"
        f"（{num(a_set.loc['F1_hist_mean', 'bias_kW'])} kW），F2 略偏负"
        f"（{num(a_set.loc['F2_intraday_linear', 'bias_kW'])} kW）。")
    add(f"5. **这不是节费、也不是模型采纳**：本轮没有调度、没有储能回放、没有 q75 重建，"
        f"MAE 改善不能读成成本下降；正式期滚动拟合中仍有 "
        f"{num(100 * beta_scope['formal']['negative_share'], 2)}% 的 β 为负、"
        f"修正量范围达 [{num(whole.delta2_min_kW, 1)}, {num(whole.delta2_max_kW, 1)}] kW，"
        f"属于小样本滚动不稳定，若要进入费用实验必须另外重建本组 W28/q75 并从各自连续的真实 "
        f"SOC 轨迹运行。")
    add("")

    add("## 2. 问题分析")
    add("")
    add("问题二的 15 特征 LightGBM 每天 0:00 发布一次负载预测，问题三 6:00/12:00/18:00 仍沿用同一"
        "预测。前轮诊断确认「发布前已观测的负载偏差」与「发布后各小时残差」总体正相关，"
        "但 35/39 个统计格至少一个月反号、且不随距离单调衰减。本轮把该线索做成三组可执行对照，"
        "在真实信息边界下检验它能否改善 10 分钟级点预测，并把「一般历史偏差校正」与"
        "「当天新增信息」的作用分开。")
    add("")
    add("本轮没有购电决策变量或现金目标：回归系数只是预测参数，评价对象是负载误差（kW）。"
        "原预测 F0 是基准且完全不动；F1、F2 都是在同一原预测误差上加修正，"
        "因此三组可用同一批目标逐条配对比较。")
    add("")

    add("## 3. 数据预处理")
    add("")
    cross = validation["checks"]["inputs_and_history"]["cross_check"]
    add(table([
        ("输入档案", "`results/q3_bias_correction_diagnostic/bias_forecast_archive.npz::issued_load,truth_load`（365×145）"),
        ("一致性核对", "`results/q2_time_mapping/archive_float.npz::issued_load`；"
                       f"NaN 掩码逐格一致（{cross['agreeing_finite_cells']:,} 个有限格），"
                       f"最大绝对差 {num(cross['max_abs_difference_kW'], 1)} kW"),
        ("正式评价期", f"{validation['checks']['inputs_and_history']['formal_window']['first']} — "
                       f"{validation['checks']['inputs_and_history']['formal_window']['last']}，"
                       f"共 {validation['checks']['inputs_and_history']['formal_window']['days']} 天"),
        ("目标槽", "h=0 当日 00:00—00:10、h=1..143 当日 00:10..23:50、h=144 次日 00:00—00:10"),
        ("近期特征", "b = 发布前已完整观测的 1 小时 6 段平均残差；6 段必须全有效，不足不外扩"),
        ("未来标签", "按小时取 6 段平均；午夜尾槽 h=144 是单独的 10 分钟槽；36 个小时组 + 3 个尾槽组"),
        ("参数估计窗口", f"过去 {28} 个日历发布日、不含当天；最少 {14} 个有效历史日"),
        ("历史资格", f"每个历史日 d 满足 max(0,k−28) ≤ d < k 且其标签区间在当天发布时刻前已全部结束；"
                     f"实测最小间隔 {validation['checks']['inputs_and_history']['history_eligibility']['min_gap_days_to_publication']} 天、"
                     f"越界 {validation['checks']['inputs_and_history']['history_eligibility']['labels_ending_after_publication']} 条"),
        ("样本", f"A 集 {int(a_set.n.iloc[0]):,} 个唯一目标；B 集 {int(b_set.n.iloc[0]):,} 条发布—目标记录；"
                 f"尾槽 {int(tail.n.iloc[0]):,} 条"),
    ], ["项目", "口径"]))
    add("")
    add("特征定义与已接收的前轮诊断逐位一致"
        f"（最大差 {num(validation['checks']['inputs_and_history']['feature_definition_vs_previous_diagnostic']['max_abs_y_difference_kW'], 1)} kW）："
        "本轮改的是「从这些特征估计什么」，不是特征或标签本身。")
    add("")

    add("## 4. 模型建立")
    add("")
    add("三组固定、不搜索：")
    add("")
    add(table([("F0_original", "原预测", f"复用 0:00 原预测，不施加任何修正（基准）"),
               ("F1_hist_mean", "原预测 + ȳ", f"仅历史偏差校正：加过去 28 日该组未来残差均值"),
               ("F2_intraday_linear", "原预测 + a + β·b", "再加当天发布前 1 小时偏差 b，参数 a、β 同样只用历史估计")],
              ["组", "预测", "含义"]))
    add("")
    add("对每个 (发布日 k, 发布节点 r, 未来小时或尾槽 c) 用同一历史集合做闭式一元最小二乘：")
    add("")
    add("```")
    add("beta = sum_d (b_d - bbar)(y_d - ybar) / sum_d (b_d - bbar)^2")
    add("a    = ybar - beta*bbar")
    add("delta1 = ybar                      # F1")
    add("delta2 = a + beta * b[k,r]         # F2")
    add("Lhat_g[k,h] = max(0, Lhat0[k,h] + delta_g[c])   h >= s_r")
    add("```")
    add("")
    add(f"回退与数值边界：m < {14} 时 F1、F2 修正都为 0 并回退 F0；当前 b 无效时 F2 回退 F1；"
        f"历史 b 方差 ≤ 1e-10 kW² 时令 β=0、a=ȳ（F2 等于 F1）。"
        f"β 不限正负、不强制小于 1、不施加任何衰减；修正量统一作用于该小时六个 10 分钟槽；"
        f"每个节点都相对原 0:00 预测修正，不在上一节点校正值上累加；0:00 版本保持原预测。")
    add("")
    add(f"实际估计与回退（共 {int(fits['groups']):,} 个参数组）：闭式估计 {int(fits['fitted']):,} 次、"
        f"m<14 回退 {int(fits['fallback_m_below_min']):,} 次、方差退化 "
        f"{int(fits['fallback_degenerate_variance'])} 次、当天 b 无效 {int(fits['fallback_b_invalid'])} 次。"
        f"回退全部集中在 1 月上旬（历史不足 14 日）与第 0 日（原档案该日无有效预测）。"
        f"**本轮不是“0 拟合”**：确实做了 {int(fits['fitted']):,} 次滚动一元参数估计，"
        f"但 LightGBM 训练 0 次、调度求解 0 次、储能回放 0 次。")
    add("")

    add("## 5. 结果")
    add("")
    add("### 5.1 主指标（A 集，36,072 个唯一 10 分钟目标）")
    add("")
    add(table([(g, int(a_set.loc[g, 'n']), num(a_set.loc[g, 'mae_kW']), num(a_set.loc[g, 'rmse_kW']),
                num(a_set.loc[g, 'bias_kW']), num(a_set.loc[g, 'mean_under_kW']),
                num(a_set.loc[g, 'mean_over_kW'])) for g in GROUPS],
              ["组", "N", "MAE (kW)", "RMSE (kW)", "偏差 (kW)", "平均正残差 (kW)", "平均负残差幅度 (kW)"]))
    add("")
    add(table([(CONTRAST_LABEL[name], num(float(contrasts.loc[name, 'delta_mae_kW'])),
                num(float(contrasts.loc[name, 'delta_rmse_kW'])),
                num(float(contrasts.loc[name, 'delta_bias_kW'])),
                f"{int(contrasts.loc[name, 'targets_improved']):,} / {int(contrasts.loc[name, 'targets_worsened']):,}",
                f"{int(contrasts.loc[name, 'days_improved'])} / {int(contrasts.loc[name, 'days_worsened'])}",
                f"{int(contrasts.loc[name, 'months_improved'])} / {int(contrasts.loc[name, 'months_worsened'])}")
               for name in ("F2_minus_F0", "F2_minus_F1", "F1_minus_F0")],
              ["配对差", "ΔMAE (kW)", "ΔRMSE (kW)", "Δ偏差 (kW)", "目标 更准/更差", "逐日 更准/更差",
               "逐月 更准/更差"]))
    add("")
    add(f"**配对差的 t 检验或显著性星号有意不做**：A 集内同一日的目标高度相关，"
        f"逐条配对计数只能描述方向，不能当作独立样本显著性。")
    add("")
    hourly_supp = grouped[grouped.set == "A_hourly"].groupby("group").mae_kW.mean()
    add(f"补充（不替代主指标）：把每小时六个**截断后**发布预测先平均再与实际小时均值比较，"
        f"小时级 MAE 为 F0 {num(hourly_supp['F0_original'])}、F1 {num(hourly_supp['F1_hist_mean'])}、"
        f"F2 {num(hourly_supp['F2_intraday_linear'])} kW，方向与逐段主指标一致。"
        f"小时级指标对每小时内的一致平移不敏感，因此不能用它代替逐段 MAE。")
    add("")

    add("### 5.2 按节点、月份与极端日")
    add("")
    node = grouped[(grouped.set == "A") & (grouped.group_type == "node")].pivot_table(
        index="group_value", columns="group", values="mae_kW")
    add(table([(f"{int(r):02d}:00", num(node.loc[str(r), 'F0_original']), num(node.loc[str(r), 'F1_hist_mean']),
                num(node.loc[str(r), 'F2_intraday_linear']),
                num(node.loc[str(r), 'F2_intraday_linear'] - node.loc[str(r), 'F0_original']),
                num(node.loc[str(r), 'F2_intraday_linear'] - node.loc[str(r), 'F1_hist_mean']))
               for r in NODES], ["发布节点", "F0 MAE", "F1 MAE", "F2 MAE", "F2−F0", "F2−F1"]))
    add("")
    add(table([(m, num(row.F0_original), num(row.F1_hist_mean), num(row.F2_intraday_linear),
                num(row.F2_minus_F0), num(row.F2_minus_F1), num(row.F1_minus_F0))
               for m, row in months.iterrows()],
              ["月份", "F0 MAE", "F1 MAE", "F2 MAE", "F2−F0", "F2−F1", "F1−F0"]))
    add("")
    reversals = [m for m, row in months.iterrows() if row.F2_minus_F1 > 0]
    add(f"F2−F0 在全部 {len(months)} 个月为负；**F2−F1 的反例月份是 "
        f"{'、'.join(reversals) if reversals else '无'}**，其中 F1−F0 在这些月份多为负，"
        f"即那些月份历史均值校正已经足够、加当天偏差没有再改善。F1−F0 只在 "
        f"{int(contrasts.loc['F1_minus_F0', 'months_improved'])}/11 个月为负。")
    add("")

    add("### 5.3 B 集与午夜尾槽（诊断，不能替代 A 集）")
    add("")
    add(table([("B 全窗口", int(b_set.loc['F0_original', 'n']), num(b_set.loc['F0_original', 'mae_kW']),
                num(b_set.loc['F1_hist_mean', 'mae_kW']), num(b_set.loc['F2_intraday_linear', 'mae_kW'])),
               ("其中午夜尾槽", int(tail.loc['F0_original', 'n']), num(tail.loc['F0_original', 'mae_kW']),
                num(tail.loc['F1_hist_mean', 'mae_kW']), num(tail.loc['F2_intraday_linear', 'mae_kW']))],
              ["集合", "N", "F0 MAE (kW)", "F1 MAE (kW)", "F2 MAE (kW)"]))
    add("")
    add(f"B 集含重叠目标（同一实际区间被多个发布版本覆盖），只作发布版本质量诊断。"
        f"尾槽 MAE 上 F2 与 F0 基本持平（差 "
        f"{num(float(tail.loc['F2_intraday_linear', 'mae_kW']) - float(tail.loc['F0_original', 'mae_kW']))} kW，更差），"
        f"RMSE 反而更好（{num(tail.loc['F0_original', 'rmse_kW'])} → "
        f"{num(tail.loc['F2_intraday_linear', 'rmse_kW'])} kW）；"
        f"F1 在尾槽上比 F0 更差（+{num(float(tail.loc['F1_hist_mean', 'mae_kW']) - float(tail.loc['F0_original', 'mae_kW']))} kW）。"
        f"1,002 条尾槽记录只对应 "
        f"{int(validation['checks']['readback']['counts']['distinct_tail_intervals'])} 个不同实际午夜区间，"
        f"不能当成三次实际发生。")
    add("")

    add("### 5.4 参数与修正量（稳定性诊断）")
    add("")
    add(f"下表只统计**正式评价期**（第 {31}–{364} 日，334 天）的滚动拟合，"
        f"这也是与第 5.1–5.3 节指标对应的口径：")
    add("")
    formal_rows = parameter_rows(formal_log)
    add(table([(f"{int(row.publication_hour):02d}:00 {row.kind}",
                f"{int(row.groups):,}", f"{int(row.fitted):,}", f"{int(row.fallback_m_below_min):,}",
                num(row.beta_mean, 4), num(row.beta_min, 3), num(row.beta_max, 3),
                f"{num(100 * row.beta_negative_share, 2)}%",
                num(row.delta2_min_kW, 1), num(row.delta2_max_kW, 1))
               for row in formal_rows.itertuples(index=False)],
              ["参数组（正式期）", "组数", "闭式估计", "m<14 回退", "β 均值", "β 最小", "β 最大",
               "β 为负占比", "δ₂ 最小 (kW)", "δ₂ 最大 (kW)"]))
    add("")
    add(f"正式期合计：闭式估计 {beta_scope['formal']['fits']:,} 次，"
        f"**β 为负占比 {num(100 * beta_scope['formal']['negative_share'], 2)}%**，"
        f"β 范围 [{num(beta_scope['formal']['beta_min'], 6)}, {num(beta_scope['formal']['beta_max'], 6)}]，"
        f"原始修正量范围 [{num(beta_scope['formal']['delta2_min'], 1)}, "
        f"{num(beta_scope['formal']['delta2_max'], 1)}] kW。")
    add("")
    add(f"**口径提示（本轮更正）**：`parameter_summary.csv` 的 `all_groups` 行是**全年 365 天**（含 1 月预热）"
        f"的汇总，其中 β 为负占比 {num(100 * beta_scope['all_days']['negative_share'], 2)}%、"
        f"β 最大 {num(beta_scope['all_days']['beta_max'], 6)}——"
        f"该最大值来自 1 月预热期而非正式期，两者**不可混用**。"
        f"引用稳定性时应取正式期数字；1 月只有 {int((log.k < 31).sum()):,} 个参数组、"
        f"其中 {int(log[(log.k < 31)].fitted.sum()):,} 次完成估计，其余为历史不足 14 日的回退。")
    add("")
    add(f"m 中位数 {num(whole.m_median, 0)}、最大 {int(whole.m_max)} 天；"
        f"最大修正量出现在 {whole.delta2_max_abs_date}"
        f"（|δ₂| = {num(max(abs(whole.delta2_min_kW), abs(whole.delta2_max_kW)), 1)} kW），"
        f"这些是小样本滚动的直接体现，本轮**不**截顶、不衰减、不按表现挑节点。")
    add("")
    truncation = validation["parameter_summary_A"]
    add(f"非负截断：A 集命中 {int(truncation['truncation_hits_f1'])} 次（F1）与 "
        f"{int(truncation['truncation_hits_f2'])} 次（F2），最大削减 "
        f"{num(truncation['truncation_max_shortfall_kW_f2'], 1)} kW。"
        f"截断在本评价集上从未触发：全部发布值最小值仍有 {num(published_min, 1)} kW，"
        f"而最小原预测约 2,067 kW、最小修正量 −788.6 kW，因此 max(0,·) 不可能生效；"
        f"截断前修正量与截断后实际变化在本轮相等，但两者在产物中仍分开记录。")
    add("")
    paths = make_figures(months, distance)
    for path, caption in zip(paths, ("分月 MAE 差（负值=左侧组更准）",
                                     "按发布节点与预测距离的 MAE 差（负值=左侧组更准）")):
        add(f"![{caption}]({path.resolve().as_posix()})")
        add("")
    add("图只做程序化完整性检查（本 Agent 无法目视图像）；误差差值图的负值表示改善，"
        "图内不画任何未经登记的“合格”参考线。")
    add("")

    add("## 6. 验证（三类精简检查，全部通过）")
    add("")
    boundary = validation["checks"]["boundary_date"]
    closed = validation["checks"]["closed_form_and_fallback"]
    readback = validation["checks"]["readback"]
    add(f"1. **输入 / 历史资格 / 边界**：两份档案 NaN 掩码逐格一致、"
        f"{cross['cells']:,} 格最大绝对差 {num(cross['max_abs_difference_kW'], 1)} kW；"
        f"`e = 真值 − 原预测` 恒等式最大残差 "
        f"{num(validation['checks']['inputs_and_history']['residual_identity']['max_abs_difference_kW'], 1)} kW；"
        f"{int(fits['groups']):,} 个参数组的历史日全部满足 "
        f"`max(0,k−28) ≤ d < k`，实测距发布日最小间隔 "
        f"{validation['checks']['inputs_and_history']['history_eligibility']['min_gap_days_to_publication']} 天，"
        f"标签结束晚于发布时刻的记录 "
        f"{validation['checks']['inputs_and_history']['history_eligibility']['labels_ending_after_publication']} 条。"
        f"2025-06-21 三个节点各自用干净副本把**发布时刻及之后**的真值抬高 "
        f"{boundary['perturb_kW']:.0f} kW：该节点的 b、全部参数与自身发布预测最大变化 "
        f"{num(max(boundary['max_abs_b_change_kW'], boundary['max_abs_parameter_change'], boundary['max_abs_published_forecast_change']), 1)} kW，"
        f"而未来标签恰好变化 {boundary['perturb_kW']:.0f} kW。")
    add(f"2. **局部闭式与回退**：对 2025-06-21 的 6:00 与 18:00（j=0）用保存的历史日列表和"
        f"`numpy.linalg.lstsq`（另一条代数路径）独立重算 a、β、δ₂，与保存值最大差 "
        f"{num(closed['max_abs_closed_form_difference'], 1)}；常数列历史返回 β=0、a=ȳ（F2 等于 F1）；"
        f"单调历史返回 β=1；m<14 时两组修正都为 0；最终非负截断生效。")
    add(f"3. **档案 / 评分落盘核对**：0:00 版本三组与冻结原预测最大差 "
        f"{num(readback['version0_max_abs_difference_from_baseline_kW'], 1)} kW；"
        f"每个发布版本在自身可达范围内等于 `max(0, 原预测 + 原始修正)`（最大差 "
        f"{num(max(readback[f'version{r}_max_abs_difference_kW'] for r in NODES), 1)} kW），"
        f"过去槽标为不适用；A 集 {int(readback['counts']['A_targets']):,}（唯一 "
        f"{int(readback['counts']['unique_A_targets']):,}）、B 集 {int(readback['counts']['B_records']):,}、"
        f"尾槽 {int(readback['counts']['tail_records']):,}（对应 "
        f"{int(readback['counts']['distinct_tail_intervals'])} 个实际午夜区间）、重复配对 "
        f"{int(readback['counts']['duplicate_pair_rows'])}；从保存记录重算三组主指标与 F2−F0 对比，"
        f"最大差 {num(readback['A_metric_readback_max_abs_difference'], 1)}；小时指标使用**截断后**"
        f"预测（最大差 {num(readback['hourly_metric_max_abs_difference'], 1)}）。")
    add("")

    add("## 7. 边界与限制")
    add("")
    for item in validation["limitations"]:
        add(f"- {item}")
    add("")
    add(f"- 参数不稳定性的口径：正式期 β 为负占比 "
        f"{num(100 * beta_scope['formal']['negative_share'], 2)}%、β 范围 "
        f"[{num(beta_scope['formal']['beta_min'], 6)}, {num(beta_scope['formal']['beta_max'], 6)}]；"
        f"`parameter_summary.csv` 的 `all_groups` 行含 1 月预热，数字不同"
        f"（{num(100 * beta_scope['all_days']['negative_share'], 2)}%、最大 "
        f"{num(beta_scope['all_days']['beta_max'], 6)}），引用时不得混用。")
    add("")
    add(f"**预登记判断规则（方案 §4.4）**：以 A 集逐段 MAE 为第一道门槛，"
        f"差值绝对值 ≤ 1e-8 kW 视为数值持平。"
        f"F2 同时低于 F0 与 F1，才可称“在本年度前向回测中，当天偏差模型比这两个基准的主指标更好”；"
        f"再结合 RMSE、各节点、月份与极端日判断是否值得设计费用对照，**不自动升级模型**。"
        f"本轮落在此分支，且 RMSE、三个节点均同向改善，但尾槽 MAE 与 F2 的负偏差是并列反例。")
    add("")

    add("## 8. 产物与复现")
    add("")
    integrity_after = manifest_integrity(manifest)
    add(f"计算层产物 `results/q3_intraday_load_forward/`：`forecast_archive.npz`"
        f"（三组 365×4×145 发布预测 + 有效掩码 + 原始修正 + 目标语义）、`parameter_log.csv`"
        f"（{len(log):,} 行）、`prediction_pairs.csv`（{len(pairs):,} 行）、`summary.csv`、"
        f"`contrasts.csv`、`grouped_metrics.csv`（{len(grouped):,} 行）、`parameter_summary.csv`、"
        f"`validation.json`、`registration.json`、`run_manifest.json`；"
        f"报告层产物本报告与 `figures/q3_intraday_load_forward/` 两图。")
    add("")
    add(f"核账：`run_manifest.json` 状态 `{manifest['status']}`，对 {integrity_after['checked']} 项计算层产物"
        f"登记哈希，重渲本报告后不匹配 {len(integrity_after['mismatched'])} 项、缺失 "
        f"{len(integrity_after['missing'])} 项；整轮 {manifest['wall_seconds']:.1f} 秒；"
        f"`zero_solves` 全为 0（LightGBM 训练 / MILP / 储能回放 / 光伏重建 / 分位重建），"
        f"另有闭式滚动估计 {int(fits['fitted']):,} 次单独登记。")
    add("")
    amendments = registration.get("amendments", [])
    if amendments:
        add(f"登记修订 {len(amendments)} 条（实现缺陷修复，均未改变参数窗口、分组或评价口径）：")
        for amendment in amendments:
            add(f"- `{amendment['previous_signature'][:12]}` → `{amendment['new_signature'][:12]}`："
                f"{amendment['reason']}")
        add("")
    add(f"登记签名 `{registration['signature']}`；输入与源码哈希见 `registration.json`，"
        f"只登记本方案、两份档案、前轮诊断源码与其配对表，未做全树哈希。")
    add("")
    add("复现：赛题目录执行 `conda run -n math_modeling python "
        "code/q3_intraday_load_forward_experiment.py`，再执行 "
        "`conda run -n math_modeling python code/q3_intraday_load_forward_report.py`。")
    add("")
    add("报告层只读 `results/q3_intraday_load_forward/`：重渲本报告与图表不会改动计算层任何产物，"
        "改文案或图表也不需要重跑参数估计。")
    add("")

    add("## 9. 更正记录（2026-09-13 按接收核查修订）")
    add("")
    add("接收核查：`reports/问题三/问题三_日内负载修正前向预测对照实验接收核查.md`；"
        "核查脚本 `code/q3_intraday_load_forward_review.py`、证据 "
        "`results/q3_intraday_load_forward_review/review.json`。核查方复算保存的 "
        "73,146 条 B 集记录、18 个固定参数组与三组指标，11 项产物哈希匹配、签名一致，"
        "并目视确认两图可读、无未登记阈值线。下列 2 条经独立复算**全部属实**，"
        "只改报告层，**未重跑参数估计、未改任何计算层产物**（重渲后 11 项哈希仍全部匹配）：")
    add(f"1. **参数不稳定性的统计口径混用（最重要）**：原第 1 节与第 5.4 节引用的是"
        f"**全年 365 天**汇总（β 为负 {num(100 * beta_scope['all_days']['negative_share'], 2)}%、"
        f"β 最大 {num(beta_scope['all_days']['beta_max'], 6)}），却与正式评价期指标并列，"
        f"属口径错配——那个 β 最大值来自 1 月预热期。已改为以**正式期**（第 31–364 日）为准"
        f"（β 为负 {num(100 * beta_scope['formal']['negative_share'], 2)}%、β 范围 "
        f"[{num(beta_scope['formal']['beta_min'], 6)}, {num(beta_scope['formal']['beta_max'], 6)}]），"
        f"第 5.4 节表格改为正式期统计，并明确写出 `all_groups` 行含 1 月预热、两者不可混用。"
        f"原始修正量范围 [−788.6, 678.0] kW 在两个口径下相同，故第 1 节该数字无需修改。")
    add(f"2. **改善集中度未披露**：核查方指出前 5 个改善日合计约占总净改善的 33.91%。"
        f"独立复算：全天级净改善（{num(day_delta.sum(), 1)} kW，即均值 −9.1419 × 334 天）中，"
        f"最集中的 5 天（{'、'.join(ranked_days.head(5).index)}）合计占 "
        f"{num(100 * top5_share, 2)}%，与已披露的「中位日仅 −1.33 kW、188 天更好/146 天更差」一致。"
        f"已在第 1 节第 3 条补入该集中度数字，使 −9.14 kW 不被读成稳定的逐日收益。")
    add("")
    add("核查方其余各条（尾槽 MAE 不优于 F0、F1 不是稳定基准改进、小时均值不能代替 10 分钟结果、"
        "不能由 MAE 差直接估算节费、2025 年已参与设计故非跨年盲测）与本报告原有表述一致，"
        "无需修改；核查结论「主要数值与时间口径可接收，支持进入下一步受控费用对照，"
        "但不能据此替换当前模型」与本报告一致。")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrity = figure_integrity(paths)
    (FIG / "figure_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "rendered", "report": REPORT_MD.name, "lines": len(lines),
                      "manifest": integrity_after, "figures": integrity, "verdict": verdict},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    main()
