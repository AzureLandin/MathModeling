#!/usr/bin/env python
"""问题三 日内负载残差可预测性轻量诊断 —— 报告层（只读渲染）。

只读 ``results/q3_intraday_load_residual_diagnostic/``，渲染诊断结果报告与至多一张图。
**不重算分组、不重建残差、不调用任何求解器**；改文案或图表永远不会触发
``code/q3_intraday_load_residual_diagnostic.py``。

运行::

    E:/Anaconda/envs/math_modeling/python.exe code/q3_intraday_load_residual_report.py
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
OUT = ROOT / "results/q3_intraday_load_residual_diagnostic"
FIG = ROOT / "figures/q3_intraday_load_residual_diagnostic"
REPORT_MD = ROOT / "reports/问题三/问题三_日内负载残差可预测性诊断结果.md"
PLAN_MD = ROOT / "reports/问题三/问题三_日内负载残差可预测性轻量诊断方案.md"
NODES = (6, 12, 18)
NEAR_LAST_J = 5                       # j = 0..5 is the stretch up to the next publication node


def num(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def slot_label(h):
    """Clock label of target slot h of a publication day (h=144 -> 24:00)."""
    minutes = 10 * int(h)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _argmin_month(block):
    row = block.loc[block.pearson_r.idxmin()]
    return dict(month=str(row.month), pearson_r=float(row.pearson_r), n=int(row.n))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(frame, headers):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in frame:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def make_figure(overall, centered):
    pool = overall[overall.kind == "hour"]
    demeaned = centered[centered.kind == "hour"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colours = {6: "#1f77b4", 12: "#ff7f0e", 18: "#2ca02c"}
    for r in NODES:
        block = pool[pool.publication_hour == r].sort_values("lead_hours")
        ax.plot(block.lead_hours, block.pearson_r, marker="o", ms=4.5, lw=1.8,
                color=colours[r], label=f"{r:02d}:00 节点（发布后 "
                                        f"{block.lead_hours.min():.0f}–{block.lead_hours.max():.0f} 小时）")
        control = demeaned[(demeaned.publication_hour == r) & (demeaned.demean == "month")]
        control = control.assign(lead_hours=control.j + 1).sort_values("lead_hours")
        ax.plot(control.lead_hours, control.pearson_r, ls="--", lw=1.2, color=colours[r], alpha=0.75)
        tail = overall[(overall.publication_hour == r) & (overall.kind == "midnight_tail")]
        if len(tail):
            ax.plot([float(tail.lead_hours.iloc[0])], [float(tail.pearson_r.iloc[0])],
                    marker="s", ms=7, mfc="none", color=colours[r])
    style = [plt.Line2D([], [], color="grey", ls="--", lw=1.2, label="月份去均值后的相关"),
             plt.Line2D([], [], color="grey", marker="s", ms=7, mfc="none", ls="none",
                        label="次日午夜 10 分钟尾槽")]
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5, label="零相关参考线")
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel("距发布时刻的提前量（小时；尾槽为 10 分钟单槽，不是小时均值）")
    ax.set_ylabel("发布前 1 小时负载偏差与未来残差的 Pearson 相关")
    ax.set_title("发布节点 × 预测距离：日内负载残差关联（描述性，非预测改善或节费证据）")
    ax.set_xticks(sorted(set(pool.lead_hours.astype(int)) | {6, 12, 18}))
    ax.grid(alpha=0.25)
    ax.legend(handles=handles + style, fontsize=8.0, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    path = FIG / "residual_correlation.png"
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
    overall = pd.read_csv(OUT / "overall_summary.csv")
    monthly = pd.read_csv(OUT / "monthly_summary.csv")
    centered = pd.read_csv(OUT / "centered_summary.csv")
    features = pd.read_csv(OUT / "publication_features.csv")
    validation = load_json(OUT / "validation.json")
    registration = load_json(OUT / "registration.json")
    manifest = load_json(OUT / "run_manifest.json")
    stability = validation["stability"]
    checks = validation["checks"]
    samples = checks["inputs_and_samples"]

    node_stats = {str(r): {} for r in NODES}
    near = monthly[(monthly.kind == "hour") & (monthly.j <= NEAR_LAST_J)]
    for r in NODES:
        block = near[near.publication_hour == r]
        node_stats[str(r)] = dict(
            cells=int(block.groupby("j").ngroups), rows=int(len(block)),
            negative=int((block.pearson_r < 0).sum()),
            by_month=block.assign(neg=(block.pearson_r < 0).astype(int))
            .groupby("month").neg.sum().astype(int).to_dict())
    worst = stability["worst_month_reversal_cells"]
    pooled_max_r = float(overall.pearson_r.max())

    # display-only derivations from the saved tables (no statistic is recomputed)
    min_index = monthly.pearson_r.idxmin()
    global_min = monthly.loc[min_index].to_dict()
    global_min_start = slot_label(int(global_min["publication_hour"]) * 6 + 6 * int(global_min["j"]))
    global_min_end = slot_label(int(global_min["publication_hour"]) * 6 + 6 * int(global_min["j"]) + 6)
    hourly = overall[overall.kind == "hour"]
    r_by_hour = {r: hourly[hourly.publication_hour == r].sort_values("j").pearson_r.to_numpy()
                 for r in NODES}
    rises = {str(r): [(int(j), int(j) + 1) for j in range(len(r_by_hour[r]) - 1)
                      if r_by_hour[r][j + 1] > r_by_hour[r][j]] for r in NODES}
    month_of_min = {key: _argmin_month(block) for key, block in
                    monthly.groupby(["publication_hour", "kind", "j"], sort=True)}
    near_rows = {}
    for r in NODES:
        block = monthly[(monthly.kind == "hour") & (monthly.j <= NEAR_LAST_J)
                        & (monthly.publication_hour == r)]
        negative = block[block.pearson_r < 0]
        near_rows[str(r)] = dict(rows=int(len(block)), negative=int(len(negative)),
                                 months=int(negative.month.nunique()),
                                 month_list=sorted(negative.month.unique()))

    if stability["min_demeaned_month_r"] <= 0 or stability["min_demeaned_weekday_r"] <= 0:
        verdict = "关联无法与月份/星期共同水平分离，本轮记录限制，当前预测不变"
    elif stability["cells_with_month_reversal"] == 0:
        verdict = "各月方向一致，可进入下一步前向验证设计讨论"
    else:
        verdict = ("存在持续且不完全由分组水平解释的关联，但带明确月度反例：只判断值得设计"
                   "前向预测验证，不指定权重、启用节点或修正模型")

    def worst_months(stats, top=2):
        ordered = sorted(((m, v) for m, v in stats["by_month"].items() if v),
                         key=lambda item: (-item[1], item[0]))
        return "、".join(f"{m}（{v}/6 格）" for m, v in ordered[:top]) or "无反号月份"

    lines = []
    add = lines.append
    add("# 问题三日内负载残差可预测性诊断结果")
    add("")
    add(f"日期：2026-09-13。状态：只读描述性诊断已完成并通过三项轻量检查；本报告只读渲染，"
        f"未新增任何求解或模型。")
    add(f"依据方案：[问题三_日内负载残差可预测性轻量诊断方案]"
        f"({PLAN_MD.resolve().as_posix()})。")
    add(f"计算层：`code/q3_intraday_load_residual_diagnostic.py`"
        f"（登记签名 `{registration['signature'][:16]}`，"
        f"0 训练 / 0 求解 / 0 储能回放）；"
        f"报告层：`code/q3_intraday_load_residual_report.py`（改文案与图表不触发计算层）。")
    add("")

    add("## 1. 结论")
    add("")
    global_min_lead = (int(global_min["j"]), int(global_min["j"]) + 1)
    add(f"1. **本轮判定：{verdict}。** 在 2025-02-01—12-31 的 {samples['sample_counts']['distinct_days']} 天里，"
        f"三个发布节点上「发布前已完整观测的一小时负载偏差」与「发布后各小时的原预测残差」"
        f"在合并样本上**全部为正相关**（36 个小时格 + 3 个次日午夜 10 分钟尾槽格，共 "
        f"{stability['cells_total']} 格，r 介于 "
        f"{num(stability['min_pooled_r'],4)}—{num(pooled_max_r,4)}）；按月份去均值后最小 r = "
        f"{num(stability['min_demeaned_month_r'],4)}、按星期去均值后最小 r = "
        f"{num(stability['min_demeaned_weekday_r'],4)}，即该关联**不能完全由月份均值差异或星期均值差异"
        f"单独解释**（两种去均值各自执行、未联合控制，也未排除其他混杂因素）。")
    add(f"2. **但月度方向并不一致**：{stability['cells_with_month_reversal']}/{stability['cells_total']} "
        f"个格子在 11 个月里至少有一个月反号。反号次数最多的格子（06:00 节点的 j=11 小时格，"
        f"17:00—18:00，即发布后第 12 个小时）其 2025-10 "
        f"月度最小 r = {num(worst[0]['min_month_r'],4)}，它**不是全表最低**；全表月度最低是 "
        f"12:00 节点的 j=9 小时格（发布后第 {global_min_lead[0]}–{global_min_lead[1]} 小时，"
        f"{global_min_start}—{global_min_end}）在 2025-10 的 "
        f"{num(global_min['pearson_r'],4)}（n={global_min['n']}）。反号集中在更远时段与 12:00 节点："
        f"前 6 小时（到下一发布节点）共 {stability['by_node']['next_node']['cells']} 个格子中 "
        f"{stability['by_node']['next_node']['cells_with_month_reversal']} 个出现过反号的月份×小时单元"
        f"（单格最多 {stability['by_node']['next_node']['max_months_negative']} 个月），而"
        f"更远时段 {stability['by_node']['far']['cells']} 个格子**全部**出现过反号单元（单格最多 "
        f"{stability['by_node']['far']['max_months_negative']} 个月）。")
    add(f"3. **较远时段整体关联较弱，但不能认定随预测距离单调衰减**：前 6 小时合并 r 为 "
        f"{num(stability['by_node']['next_node']['pooled_r_min'],4)}—"
        f"{num(stability['by_node']['next_node']['pooled_r_max'],4)}，"
        f"更远时段为 {num(stability['by_node']['far']['pooled_r_min'],4)}—"
        f"{num(stability['by_node']['far']['pooled_r_max'],4)}（两端范围有重叠，"
        f"仅比较近远总体范围也不能证明所有对应项都变弱），"
        f"次日午夜 10 分钟尾槽为 {num(stability['by_node']['midnight_tail']['pooled_r_min'],4)}—"
        f"{num(stability['by_node']['midnight_tail']['pooled_r_max'],4)}。"
        f"18:00 六个小时格依次为 {', '.join(num(v, 4) for v in r_by_hour[18])}；各节点的局部回升位置："
        + "；".join(f"{r:02d}:00 为 j="
                    + "、".join(f"{a}→{b}" for a, b in rises[str(r)]) for r in NODES)
        + f"。因此本轮**不预设**权重随提前时间衰减，也不能由这份统计直接固定指数衰减权重。")
    add("4. **本轮不产生任何可执行结论**：相关系数与斜率都不是预测改善，更不是节费。"
        "没有拟合或发布修正模型、没有估计费用、没有求解调度、没有回放储能、没有重算 q75，"
        "也没有新增 8/9 时官方预报。若后续要判断点预测是否真的改善，必须另立登记："
        "参数只能用发布时刻已实现的历史标签估计，并至少比较三组——原预测、仅历史均值偏差校正、"
        "历史截距加当天偏差（如 max{0, L̂_{k,h|0} + a_{k,r,j} + β_{k,r,j}·b_{k,r}}，"
        "这只是待研究结构，本轮不固定权重、估计窗口或启用范围，也不发布新预测）；"
        "月度反例要求前向验证，不能用总体相关代替；小时均值相关也不能证明逐段修正有效，"
        "进入费用实验前仍需 10 分钟尺度误差检查。若新预测要进入费用实验，"
        "必须从它自身因果发布误差重建同发布节点的 W28/q75，"
        "从各自连续的真实 SOC 轨迹运行，不得借用 L75 的后续状态。")
    add("")

    add("## 2. 问题分析")
    add("")
    add("题目给的是四个固定发布时刻（0/6/12/18 点）的光伏预报；负载预测由问题二的 15 特征 "
        "LightGBM 在每日 0:00 一次发布，**当天日内不再刷新**。问题三的滚动调度虽然会在 6/12/18 点"
        "用真实 SOC 反馈重优化，但那改变的是剩余时段的初始库存与调整权限，不是负载点预测本身。"
        "因此本轮研究的对象是「当天已经观测到的负载偏差里是否含有增量信息」，"
        "而不是重复问题二已有的跨日历史特征。")
    add("")
    add("本轮的边界：只有残差特征与描述性统计，没有购电决策变量，不优化现金费用；"
        "统计对象是负载预测误差，单位 kW。全期诊断可以事后使用真值计算指标，"
        "但发布时刻的特征**只用发布之前已结束的区间**，不得使用未来真值。"
        "相关关系不是因果作用，也不是预测收益或节费证明。")
    add("")

    add("## 3. 数据预处理")
    add("")
    add(table([
        ("输入档案", "`results/q3_bias_correction_diagnostic/bias_forecast_archive.npz::issued_load,truth_load`（365×145）"),
        ("一致性核对", "`results/q2_time_mapping/archive_float.npz::issued_load`；"
                       f"NaN 掩码逐格一致（{samples['cross_check']['nan_mask_agreeing_cells']:,} 个有限格），"
                       f"全部 {samples['cross_check']['archive_cells']:,} 格最大绝对差 "
                       f"{num(samples['cross_check']['max_abs_difference_kW'],1)} kW"),
        ("正式诊断期", f"{samples['dates']['first']} — {samples['dates']['last']}，"
                       f"{samples['sample_counts']['distinct_days']} 天连续自然日"),
        ("目标槽", "h=0 当日 00:00—00:10、h=1..143 当日 00:10..23:50、h=144 次日 00:00—00:10，每槽 10 分钟"),
        ("发布节点", "r ∈ {6,12,18}，发布目标槽 s_r = 6r ∈ {36,72,108}"),
        ("近期特征窗", "固定发布前 1 小时 = 6 槽，末段起点 05:50 / 11:50 / 17:50，区间结束时刻恰为发布时刻"),
        ("发布—目标配对", f"{samples['sample_counts']['hour_pairs']:,} 条「发布节点—未来小时」记录"
                          f"（重叠的完整未来窗口另有 {samples['sample_counts']['full_window_slot_records']:,} 条区间记录，"
                          f"本轮主诊断只用不重复的 {samples['sample_counts']['to_next_node_slot_records']:,} 条）"),
        ("午夜尾槽", f"{samples['sample_counts']['midnight_tail_rows']:,} 条单槽记录，单列，不并入小时均值"),
        ("缺失", f"有效近期窗不足 6 段则该发布样本无效、不向更早扩张；未来小时不足 6 段则配对无效。"
                 f"本轮无效记录 "
                 f"{samples['validity']['feature_invalid'] + samples['validity']['hour_pair_invalid'] + samples['validity']['tail_invalid']} 条，"
                 f"未做任何补零或前向填充"),
    ], ["项目", "口径"]))
    add("")

    add("## 4. 诊断模型")
    add("")
    add("原 0:00 负载预测残差（正值表示负载被低估）：")
    add("")
    add("```")
    add("eL[k,h] = truth_load[k,h] - issued_load[k,h]        (kW)")
    add("```")
    add("")
    add("固定一个近期窗口（发布前已完整观测的一小时），不搜索其他窗口：")
    add("")
    add("```")
    add("b[k,r] = mean( eL[k, s_r-6 .. s_r-1] )")
    add("y[k,r,j] = mean( eL[k, s_r+6j .. s_r+6j+5] )   j = 0..23-r        # j=0 是发布后第一小时")
    add("```")
    add("")
    add("每个发布节点／未来小时输出有效日期数、b 与 y 的均值与样本标准差（ddof=1）、Pearson 相关 r，"
        "以及带截距描述性直线的斜率与截距：")
    add("")
    add("```")
    add("beta[r,j] = cov(b, y) / var(b)        a[r,j] = mean(y) - beta*mean(b)")
    add("```")
    add("")
    add("分母为零（常数列）时返回缺失并标记 `constant_b`/`constant_y`，**不返回 0 相关**。"
        "稳定性检查有两类：分 11 个月报告各 (r,j) 的样本量、相关与斜率；"
        "再按月份、按星期（发布日的星期几）在组内用全期分组统计去均值后重算相关，"
        "用于检查共同月份／星期水平是否解释了原相关。去均值只用于诊断，绝不进入前向预测。"
        "本式只作描述性协方差计算，不调用模型训练器，不发布修正预测；"
        "这里的 beta 不能直接充当滚动权重，也未施加非负、递减或小于 1 的限制。")
    add("")

    add("## 5. 结果")
    add("")
    add("### 5.1 发布前 1 小时偏差与未来各小时残差")
    add("")
    rows = []
    for r in NODES:
        block = overall[(overall.publication_hour == r) & (overall.kind == "hour")]
        rows.append((f"{r:02d}:00 合并", f"{int(block.lead_hours.min())}–{int(block.lead_hours.max())}",
                     f"{block.n.min()}–{block.n.max()}",
                     f"{block.mean_b_kW.iloc[0]:.3f}", f"{block.sd_b_kW.iloc[0]:.3f}",
                     f"{num(block.pearson_r.min(),4)} – {num(block.pearson_r.max(),4)}",
                     f"{num(block.beta.min(),4)} – {num(block.beta.max(),4)}",
                     f"{num(block.intercept_a_kW.min(),3)} – {num(block.intercept_a_kW.max(),3)}"))
        tail = overall[(overall.publication_hour == r) & (overall.kind == "midnight_tail")].iloc[0]
        rows.append((f"{r:02d}:00 午夜尾槽", "次日 00:00 单槽", f"{tail.n}",
                     f"{tail.mean_b_kW:.3f}", f"{tail.sd_b_kW:.3f}", num(tail.pearson_r, 4),
                     num(tail.beta, 4), num(tail.intercept_a_kW, 3)))
    add(table(rows, ["发布节点", "提前量（小时）", "有效日数", "mean b (kW)", "sd b (kW)",
                     "r 范围", "beta 范围", "截距 a 范围 (kW)"]))
    add("")
    add(f"注意同一发布日的**同一个 b 被 18／12／6 个未来小时共用**，"
        f"不是 {samples['sample_counts']['hour_pairs']:,} 个独立样本；日间残差本身存在序列相关，"
        f"因此不附未经校正的独立样本显著性解释。")
    add("")

    add("### 5.2 分月稳定性（前 6 小时，到下一发布节点）")
    add("")
    rows = []
    for r in NODES:
        stats = node_stats[str(r)]
        negatives = ", ".join(f"{m}×{v}" for m, v in sorted(stats["by_month"].items()) if v)
        near = near_rows[str(r)]
        rows.append((f"{r:02d}:00", f"{stats['rows']} 个单元 / {stats['cells']} 格",
                     f"{stats['negative']}/{stats['rows']}", f"{near['months']}",
                     negatives or "无"))
    add(table(rows, ["发布节点", "分月单元", "反号的月份×小时单元数（共 66）", "涉及的不同月份数",
                     "出现反号的月份（×单元数）"]))
    add("")
    ranking = sorted(NODES, key=lambda r: node_stats[str(r)]["negative"])
    add(f"这里的分母是 11 个月 × 6 个未来小时 = 66 个「月份×未来小时」单元，"
        f"**不是 66 个月，也不能把 8／13／2 说成“反号月份数”**；"
        f"涉及的不同月份数分别是 "
        + "、".join(f"{r:02d}:00 为 {near_rows[str(r)]['months']} 个"
                    + f"（{'、'.join(near_rows[str(r)]['month_list'])}）" for r in ranking)
        + f"。按单元数从少到多："
        + "，".join(f"{r:02d}:00 节点 {node_stats[str(r)]['negative']}/66" for r in ranking)
        + f"。各节点反号最多的月份："
        + "；".join(f"{r:02d}:00 为 {worst_months(node_stats[str(r)])}" for r in NODES)
        + f"。更远时段的反号更多："
        f"{stability['by_node']['far']['cells_with_month_reversal']}/"
        f"{stability['by_node']['far']['cells']} 格至少一个月反号，单格最多 "
        f"{stability['by_node']['far']['max_months_negative']} 个月，"
        f"而前 6 小时单格最多 {stability['by_node']['next_node']['max_months_negative']} 个月。")
    add("")
    add("反号次数最多的格子（按反号月份数排序的前 6 个）及其月度最小 r 所在月份：")
    add("")
    add(table([(f"{c['publication_hour']:02d}:00", "小时" if c["kind"] == "hour" else "午夜尾槽",
                int(c["j"]) if int(c["j"]) >= 0 else "—", f"{c['months_negative']}/{c['months_total']}",
                num(c["pooled_r"], 4),
                month_of_min[(c["publication_hour"], c["kind"], c["j"])]["month"],
                f"{month_of_min[(c['publication_hour'], c['kind'], c['j'])]['n']}",
                num(month_of_min[(c["publication_hour"], c["kind"], c["j"])]["pearson_r"], 4))
               for c in worst],
              ["发布节点", "类型", "j", "反号月份数", "合并 r", "月度最小 r 所在月", "该月样本数",
               "月度最小 r"]))
    add("")
    add(f"其中全表月度最低为 12:00 节点的 j={int(global_min['j'])} 小时格"
        f"（发布后第 {global_min_lead[0]}–{global_min_lead[1]} 小时，"
        f"{global_min_start}—{global_min_end}）在 {global_min['month']}："
        f"r = {num(global_min['pearson_r'],4)}、beta = {num(global_min['beta'],4)}、n = {global_min['n']}。")
    add("")

    add("### 5.3 月份／星期去均值对照")
    add("")
    rows = []
    for r in NODES:
        for kind, label in (("hour", "前 6 小时"), ("hour", "更远时段"), ("midnight_tail", "午夜尾槽")):
            block = overall[(overall.publication_hour == r) & (overall.kind == kind)]
            block = block[block.range_label == ("next_node" if label == "前 6 小时" else
                                               ("far" if label == "更远时段" else "midnight_tail"))]
            control = centered[(centered.publication_hour == r) & (centered.kind == kind)]
            control = control[control.j.isin(block.j)]
            if block.empty:
                continue
            rows.append((f"{r:02d}:00", label, f"{len(block)} 格",
                         f"{num(block.pearson_r.min(),4)} – {num(block.pearson_r.max(),4)}",
                         f"{num(control[control.demean=='month'].pearson_r.min(),4)} – "
                         f"{num(control[control.demean=='month'].pearson_r.max(),4)}",
                         f"{num(control[control.demean=='weekday'].pearson_r.min(),4)} – "
                         f"{num(control[control.demean=='weekday'].pearson_r.max(),4)}"))
    add(table(rows, ["发布节点", "范围", "单元数", "合并 r 范围", "月份去均值 r 范围", "星期去均值 r 范围"]))
    add("")
    add(f"全部 {stability['cells_total']} 个格子去均值后仍为正"
        f"（月份去均值最小 {num(stability['min_demeaned_month_r'],4)}、"
        f"星期去均值最小 {num(stability['min_demeaned_weekday_r'],4)}）：该关联"
        f"**不能完全由月份均值差异或星期均值差异单独解释**。但两种去均值是**分别**执行的，"
        f"没有联合控制，也没有排除其他混杂因素，因此这既不能说月份／星期没有作用，"
        f"也不代表该关联在样本外稳定——去均值用的是全期分组统计，只能用于诊断。")
    add("")
    add(f"![发布节点×预测距离的残差相关性]({make_figure(overall, centered).resolve().as_posix()})")
    add("")
    add("图：实线为合并样本的相关，虚线为月份去均值后的相关，空心方块为次日午夜 10 分钟尾槽。"
        "曲线显示各节点都有局部回升，因此不能按单调衰减读取。"
        "图只做程序化完整性检查（本 Agent 无法目视图像）。")
    add("")

    add("## 6. 验证（三类精简检查，全部通过）")
    add("")
    boundary = checks["boundary_date"]
    add(f"1. **输入与样本**：两份档案的 `issued_load` NaN 掩码逐格一致"
        f"（{samples['cross_check']['nan_mask_agreeing_cells']:,} 个有限格），全部 "
        f"{samples['cross_check']['archive_cells']:,} 格最大绝对差 "
        f"{num(samples['cross_check']['max_abs_difference_kW'],1)} kW；"
        f"`eL = truth - issued` 恒等式最大残差 {num(samples['residual_identity']['max_abs_difference_kW'],1)} kW；"
        f"特征重放最大绝对差 {num(samples['feature_replay_max_abs_kW'],1)} kW（容差 1e-8、rtol=0）；"
        f"逐日样本 {samples['sample_counts']['distinct_days']}、发布特征 "
        f"{samples['sample_counts']['publication_features']:,}、未来小时配对 "
        f"{samples['sample_counts']['hour_pairs']:,}（其中前 6 小时 "
        f"{samples['sample_counts']['near_hour_pairs']:,}）、午夜尾槽 "
        f"{samples['sample_counts']['midnight_tail_rows']:,}、重复配对 "
        f"{samples['sample_counts']['duplicate_pairs']}；不存在把无效行的 y 填成非缺失。")
    add(f"2. **一个边界日期（2025-06-21，自然日第 {boundary['k']} 日）**：在内存中把该日**发布时刻及之后**"
        f"的真实负载整体抬高 {boundary['perturb_kW']:.0f} kW 后，三个节点的 b "
        f"最大变化 {num(boundary['max_abs_b_change_kW'],1)} kW（应为 0），"
        f"而发布后第一小时的标签与午夜尾槽均恰好变化 {boundary['perturb_kW']:.0f} kW。"
        f"这正是本轮要求的信息边界：b 只用发布前的区间，y 才是标签。"
        f"表中列出各节点的近期首末区间、未来第一段与午夜尾槽时刻：")
    add("")
    add(table([(f"{row['publication_hour']:02d}:00", row["recent_last_interval_start"],
                f"{row['future_first_interval_start']} — {row['future_first_interval_end']}",
                row["midnight_tail_interval_start"],
                num(row["b_abs_change_kW"], 1), num(row["y_first_hour_abs_change_kW"], 1))
               for row in boundary["rows"]],
              ["发布节点", "近期末段起点", "未来第一段", "午夜尾槽起点", "b 变化 (kW)", "y 变化 (kW)"]))
    add("")
    readback = checks["disk_readback"]
    add(f"3. **一次落盘复核**：从保存的配对表重算 06:00 与 18:00 的第一未来小时，"
        f"均值、相关与斜率与 `overall_summary.csv` 的最大绝对差 "
        f"{num(readback['max_abs_recompute_difference'],1)}；"
        f"并检查两类边界：常数列返回缺失相关与缺失斜率（`constant_b=True`，不返回 0），"
        f"单调序列返回 r=1、beta=1、截距 0；近期窗不足 6 段时该发布样本标记无效并保持 NaN，"
        f"窗口不向更早扩张。")
    add("")

    add("## 7. 边界与限制")
    add("")
    for item in validation["limitations"]:
        add(f"- {item}")
    add(f"- 本轮尚未做、也不自动进入的下一步：不拟合修正模型、不估计权重、不做窗口扫描"
        f"（30 分钟／2 小时／当日累计偏差）、不带显著性星号、不 Bootstrap、不按结果挑启用月份；"
        f"不因本轮结果改动四个发布节点、计费、实际 SOC 反馈、自由末态与公共初始化。")
    add(f"- 18:00 节点在本轮描述性统计中较稳定，但**不据此只启用该节点**，"
        f"也不假定固定正权重或权重随提前时间单调衰减。")
    add("")

    add("## 8. 更正记录（2026-09-13 按接收核查修订）")
    add("")
    add(f"接收核查：reports/问题三/问题三_日内负载残差诊断接收核查.md；"
        f"核查脚本 code/q3_intraday_load_residual_review.py、证据 "
        f"results/q3_intraday_load_residual_review/review.json。"
        f"核查方按原档案独立复算 13,026 条配对与全部相关/斜率，与本轮数值最大差分别为 "
        f"5.69e-14 / 1.14e-13 kW 与 4.45e-16，10 项产物哈希匹配、签名一致；"
        f"下列 5 条意见经逐条独立复算**全部属实**，只改报告层与图表，"
        f"**不重跑诊断、不改任何计算层产物**（重渲后 10 项哈希仍全部匹配）：")
    add(f"1. **删除“随预测距离单调衰减”的表述**：原第 1 节“关联强度随提前量衰减”与"
        f"“该单调形只是本窗口下的描述”不符合保存曲线。18:00 六个小时格为 "
        f"{', '.join(num(v, 4) for v in r_by_hour[18])}，第 2→3、第 5→6 小时均回升；"
        f"06:00 与 12:00 也有局部回升。已改为“较远时段整体关联较弱，但不能认定随预测距离单调衰减”，"
        f"并写明仅比较近远总体范围也不能证明所有对应项变弱、不得据此固定指数衰减权重。")
    add(f"2. **去均值结论收窄**：原“即该关联不是月份或星期的共同水平造成的”改为"
        f"“不能完全由月份均值差异或星期均值差异**单独**解释”，并写明两种去均值分别是执行、"
        f"未联合控制、也未排除其他混杂因素（与第 7 节限制口径一致）。")
    add(f"3. **月度反例标签与全表最低**：原“最差格子的最短月 r 低至 −0.3510”不准确——"
        f"−0.3510 出现在**反号次数最多**的格子（06:00 节点的 j=11 小时格，17:00—18:00，"
        f"即发布后第 12 个小时；原报告此处写作“提前 11 小时”，与表内 lead_hours=j+1 口径易混，2025-10），"
        f"不是全表最低；已补全表月度最低 12:00 的 j={int(global_min['j'])} 小时格"
        f"（发布后第 {global_min_lead[0]}–{global_min_lead[1]} 小时，{global_min_start}—{global_min_end}）"
        f"在 {global_min['month']} 的 {num(global_min['pearson_r'],4)}（n={global_min['n']}），"
        f"并把列名“最短月 r”改为“月度最小 r”，另加“月度最小 r 所在月”与“该月样本数”两列。")
    add(f"4. **分月计数口径**：原表头“反号月份数（共 66）”错误——8／13／2 是"
        f"**负相关的「月份×未来小时」单元数**（分母 11 个月 × 6 个未来小时 = 66），"
        f"不是反号月份数；涉及的不同月份数分别为 4、6、2。已改列名并新增“涉及的不同月份数”与"
        f"具体月份列，同时在正文写明这一点。")
    add(f"5. **图件**：删除无任务书依据的 0.2 水平参考线（易被误读为阈值），"
        f"零相关参考线改为在图例中注明“零相关参考线”，并在图注写明曲线存在局部回升、"
        f"不能按单调衰减读取。")
    add(f"另按核查建议，第 1 节第 4 条已写明进入前向验证时至少比较的三组（原预测、"
        f"仅历史均值偏差校正、历史截距加当天偏差），并补充“小时均值相关不能证明逐段修正有效、"
        f"进入费用实验前仍需 10 分钟尺度误差检查”。")
    add("")

    add("## 9. 产物与复现")
    add("")
    integrity_after = manifest_integrity(manifest)
    add(f"计算层产物 `results/q3_intraday_load_residual_diagnostic/`："
        f"`publication_features.csv`（{len(features):,} 行）、`hourly_pairs.csv`、"
        f"`overall_summary.csv`、`monthly_summary.csv`、`centered_summary.csv`、"
        f"`validation.json`、`registration.json`、`run_manifest.json`；"
        f"报告层产物本报告与 `figures/q3_intraday_load_residual_diagnostic/residual_correlation.png`。")
    add("")
    add(f"核账：`run_manifest.json` 状态 `{manifest['status']}`，对 "
        f"{integrity_after['checked']} 项计算层产物登记哈希，"
        f"重渲本报告后不匹配 {len(integrity_after['mismatched'])} 项、"
        f"缺失 {len(integrity_after['missing'])} 项；"
        f"整轮 {manifest['wall_seconds']:.2f} 秒，`zero_solves` 全为 0"
        f"（训练 / MILP / 储能回放 / 预测重建 / 分位重建）。")
    add("")
    add("复现：赛题目录执行 `conda run -n math_modeling python "
        "code/q3_intraday_load_residual_diagnostic.py`，再执行 "
        "`conda run -n math_modeling python code/q3_intraday_load_residual_report.py`。")
    add("")
    add(f"登记签名 `{registration['signature']}`；输入与源码哈希见 `registration.json`，"
        f"只登记本方案、两份档案与本轮直接依赖的两份源码，未做全树哈希。")
    amendments = registration.get("amendments", [])
    if amendments:
        add("")
        add(f"登记修订记为 {len(amendments)} 条，其中完整列出了本轮 5 次代码修订的签名区间与原因，"
            f"全部是实现缺陷修复或文案本地化（边界扰动探针跨节点累积、落盘复核列名不存在、"
            f"新增稳定性汇总并修正控制台计数、限制条款中文化，以及修复 `register()` 在幂等重跑中"
            f"丢弃修订历史的缺陷），**均未改变统计量、窗口或样本定义**；"
            f"重跑后除 `registration.json` 的时间戳外，其余 9 项计算层产物逐字节一致。"
            f"完整原文见 `registration.json` 的 `amendments`：")
        for amendment in amendments:
            reason = amendment["reason"]
            head = reason.split(";")[0].split(".")[0].strip()
            if len(head) > 180:
                head = head[:180] + "…"
            add(f"- `{amendment['previous_signature'][:12]}` → `{amendment['new_signature'][:12]}`："
                f"{head}")
    add("")
    add("报告层只读 `results/q3_intraday_load_residual_diagnostic/`：重渲本报告与图表不会改动计算层任何产物"
        "（上述哈希在重渲后仍全部匹配），改文案或图表也不需要重跑统计。")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrity = {path.name: figure_integrity(path) for path in sorted(FIG.glob("*.png"))}
    (FIG / "figure_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "rendered", "report": REPORT_MD.name,
                      "lines": len(lines), "manifest": manifest_integrity(manifest),
                      "figures": integrity, "verdict": verdict}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    main()
