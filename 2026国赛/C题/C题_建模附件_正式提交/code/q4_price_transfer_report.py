#!/usr/bin/env python
"""问题四 已知电价迁移实验 —— 报告层（只读渲染）。

只读 ``results/q4_price_transfer/``，渲染结果报告与最多两张图。
**不重算保护、不重建预测、不调用任何求解器**；改文案或图表永远不会触发
``code/q4_price_transfer_experiment.py``。

运行::

    conda run -n math_modeling python code/q4_price_transfer_report.py
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
OUT = ROOT / "results/q4_price_transfer"
FIG = ROOT / "figures/q4_price_transfer"
REPORT_MD = ROOT / "reports/问题四/问题四_已知电价迁移实验结果.md"
PLAN_MD = ROOT / "reports/问题四/问题四_已知电价迁移实验方案_实验Agent任务书.md"

GROUPS = ("F42", "Q42", "F43_S0", "Q43_S0", "F43_S2", "Q43_S2")
ROLE = {"F42": "4-2 固定计划对照", "Q42": "4-2 主结果", "F43_S0": "4-3·S0 固定计划对照",
        "Q43_S0": "4-3·S0 主结果", "F43_S2": "4-3·S2 固定计划对照", "Q43_S2": "4-3·S2 主结果"}
PLAN_SOURCE = {"F42": "问题二 N_free 已保存计划量", "Q42": "问题二冻结预测＋q80",
               "F43_S0": "问题三 S0/L75 原始与最终有效量", "Q43_S0": "S0 原 0 点负载＋Linear＋q75",
               "F43_S2": "问题三 S2 原始与最终有效量", "Q43_S2": "S2 的 F2 修正＋Linear＋自身 q75"}
SPECIFIED_HOURS = (10, 12, 14, 16, 18, 20)
BLOCKS = ((0, "00—04 时"), (4, "04—08 时"), (8, "08—12 时"), (12, "12—16 时"),
          (16, "16—20 时"), (20, "20—24 时"))
SELECTED_DAYS = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
NU_GUARD = 0.9


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
    return f"{'+' if value >= 0 else '−'}{num(abs(value), digits)}"


def pct(value, digits=4):
    return signed(100 * float(value), digits) + "%"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def make_figures(summary, contrasts, frames):
    index = summary.set_index("group")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    x = np.arange(len(GROUPS))
    bottom = np.zeros(len(GROUPS))
    for key, label, colour in (("ordinary_cost_yuan", "普通购电", "#4c72b0"),
                               ("adjustment_cost_yuan", "调整费", "#dd8452"),
                               ("emergency_cost_yuan", "应急费", "#55a868")):
        values = np.array([float(index.loc[g, key]) for g in GROUPS])
        axes[0].bar(x, values, bottom=bottom, label=label, color=colour)
        bottom += values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([g.replace("_", "\n") for g in GROUPS], fontsize=8)
    axes[0].set_ylabel("自然账本现金费用（元）")
    axes[0].set_title("六组费用三项分解（附件4 交付电价）")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    totals = contrasts[contrasts.item == "total"].set_index("contrast")
    names = ["S42", "S43_S0", "S43_S2"]
    values = [float(totals.loc[n, "delta_yuan"]) for n in names]
    bars = axes[1].bar(names, values, color=["#4c72b0", "#dd8452", "#55a868"])
    axes[1].axhline(0.0, color="black", lw=0.9)
    axes[1].set_ylabel("节费量（元；正=重优化更省）")
    axes[1].set_title("相对固定计划回放的节费量")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axes[1].annotate(f"{value:+,.0f}", (bar.get_x() + bar.get_width() / 2, value),
                         ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.tight_layout()
    path1 = FIG / "cost_components_and_savings.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)

    day = "2025-06-21"
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.6), sharex=True)
    block = frames["Q43_S2"]
    block = block[block.interval_start.dt.strftime("%Y-%m-%d") == day].sort_values("interval_start")
    hours = block.interval_start.dt.hour + block.interval_start.dt.minute / 60.0
    axes[0].plot(hours, block.price_yuan_kWh, color="#333333", lw=1.4)
    axes[0].set_ylabel("附件4 电价\n(元/kWh)")
    axes[0].set_title(f"{day}：价格、普通购电量与实际内部库存的对应（自然日 00:00—24:00）")
    axes[0].grid(alpha=0.25)
    for group, style in (("Q42", "-"), ("Q43_S0", "--"), ("Q43_S2", ":")):
        part = frames[group]
        part = part[part.interval_start.dt.strftime("%Y-%m-%d") == day].sort_values("interval_start")
        part_hours = part.interval_start.dt.hour + part.interval_start.dt.minute / 60.0
        axes[1].plot(part_hours, part.q_eff_kWh * 6.0, style, lw=1.3, label=group)
    axes[1].set_ylabel("普通购电\n(kW)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for group in GROUPS:
        part = frames[group]
        part = part[part.interval_start.dt.strftime("%Y-%m-%d") == day].sort_values("interval_start")
        part_hours = part.interval_start.dt.hour + part.interval_start.dt.minute / 60.0
        axes[2].plot(part_hours, part.state_start_kWh, lw=1.1, alpha=0.85, label=group)
    axes[2].axhline(1200.0, color="grey", ls=":", lw=0.9)
    axes[2].axhline(10800.0, color="grey", ls=":", lw=0.9)
    axes[2].set_ylabel("内部库存\n(kWh)")
    axes[2].set_xlabel("小时（自然日）")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    path2 = FIG / "selected_day_dispatch.png"
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
    summary = pd.read_csv(OUT / "summary.csv")
    contrasts = pd.read_csv(OUT / "contrasts.csv")
    monthly = pd.read_csv(OUT / "monthly.csv")
    daily = pd.read_csv(OUT / "daily.csv")
    chosen = pd.read_csv(OUT / "selected_dates.csv")
    validation = load_json(OUT / "validation.json")
    checks = load_json(OUT / "checks.json")
    reuse = load_json(OUT / "reuse_review.json")
    initialization = load_json(OUT / "public_initialization.json")
    registration = load_json(OUT / "registration.json")
    manifest = load_json(OUT / "run_manifest.json")
    frames = {g: pd.read_csv(OUT / g / "dispatch.csv", parse_dates=["interval_start"])
              for g in GROUPS}
    index = summary.set_index("group")
    totals = contrasts[contrasts.item == "total"].set_index("contrast")
    components = contrasts.set_index(["contrast", "item"])

    price = float(frames["Q43_S2"].price_yuan_kWh.median())
    nu = price / NU_GUARD
    aux = {g: float(index.loc[g, "natural_total_yuan"]) - nu * (
        float(index.loc[g, "final_natural_state_kWh"]) - float(index.loc[g, "initial_state_kWh"]))
        for g in GROUPS}
    # the task book defines S = C_fixed - C_candidate, so a positive S means the re-optimised
    # branch is cheaper. The compute layer stores delta = candidate - fixed, hence the negation.
    saving = {name: -float(totals.loc[name, "delta_yuan"]) for name in ("S42", "S43_S0", "S43_S2")}
    saving_share = {name: saving[name] / float(components.loc[(name, "total"), "fixed_yuan"])
                    for name in saving}
    component_saving = {(name, key): -float(components.loc[(name, key), "delta_yuan"])
                        for name in saving for key in ("ordinary", "adjustment", "emergency")}
    q42 = float(index.loc["Q42", "natural_total_yuan"])
    overall = {g: (q42 - float(index.loc[g, "natural_total_yuan"]), float(index.loc[g, "natural_total_yuan"]))
               for g in GROUPS if g.startswith("Q43")}

    lines = []
    add = lines.append
    add("# 问题四已知电价迁移实验结果")
    add("")
    add("日期：2026-09-13。状态：六组已运行并通过三类必要验证；本报告只读渲染，未新增任何求解。")
    add(f"依据方案：[问题四_已知电价迁移实验方案_实验Agent任务书]"
        f"({PLAN_MD.resolve().as_posix()})。")
    add(f"计算层：`code/q4_price_transfer_experiment.py`（登记签名 "
        f"`{registration['signature'][:16]}`；主求解 {validation['budget']['main_solves']} 次 "
        f"（一月 {validation['budget']['january']} ＋ Q42 {validation['budget']['Q42']} ＋ "
        f"Q43_S0 与 Q43_S2 各 {validation['budget']['Q43_S0']}）＋ 额外验证 "
        f"{validation['budget']['extra_validation_milp']} 次；训练 0、全年重建保护 0、参数扫描 0、"
        f"旧 q80 分支重调度 0）；报告层：`code/q4_price_transfer_report.py`"
        f"（改文案与图表不触发计算层）。")
    add("")

    add("## 1. 结论")
    add("")
    add("节费量按任务书定义 `S = C_固定计划 − C_重优化`，**正值表示重优化更便宜**。"
        f"1. **三条重优化主分支都比各自的固定计划回放更省**：S42 = "
        f"{signed(saving['S42'])} 元（{pct(saving_share['S42'])}）、S43_S0 = "
        f"{signed(saving['S43_S0'])} 元（{pct(saving_share['S43_S0'])}）、S43_S2 = "
        f"{signed(saving['S43_S2'])} 元（{pct(saving_share['S43_S2'])}）。")
    add(f"2. **六组自然账本总费（附件4 交付电价，2025-02-01—12-31，48096 段）：** "
        + "；".join(f"{g} {num(float(index.loc[g, 'natural_total_yuan']))} 元" for g in GROUPS)
        + "。两个 4-3 候选都低于 4-2 主结果 Q42：Q42 − Q43_S0 = "
        f"{signed(overall['Q43_S0'][0])} 元、Q42 − Q43_S2 = {signed(overall['Q43_S2'][0])} 元"
        f"（占 Q42 的 {pct(overall['Q43_S2'][0] / q42)}）。"
        f"该整体差异同时包含光伏预报来源、更新权限与保护水平，"
        f"**不能称为纯光伏信息价值，也不保证 4-3 必然更省**。")
    add(f"3. **节费来自普通购电，代价是调整费与应急费**。以 S43_S2 为例：普通购电节省 "
        f"{num(component_saving[('S43_S2', 'ordinary')])} 元、"
        f"调整费增加 {num(-component_saving[('S43_S2', 'adjustment')])} 元、"
        f"应急费增加 {num(-component_saving[('S43_S2', 'emergency')])} 元。"
        f"**费用下降与应急上升必须并列披露**；现金已含 5 倍应急费，本轮不额外发明题面没有的"
        f"应急否决门槛。")
    add(f"4. **4-3 内部 S2 仍略优于 S0**：Q43_S0 − Q43_S2 = "
        f"{signed(float(index.loc['Q43_S0', 'natural_total_yuan']) - float(index.loc['Q43_S2', 'natural_total_yuan']))} 元，"
        f"方向与问题三固定电价下一致，幅度不同；两个候选都提交接收，本轮不提前取舍。")
    add(f"5. **库存辅助估值不改变排序**：ν = median(交付电价)/0.9 = {num(nu, 6)} 元/内部 kWh 下，"
        f"C_aux 为 " + "、".join(f"{g} {num(aux[g])}" for g in GROUPS)
        + " 元，Q43_S2 仍最低。该系数只是事后敏感性口径，不进入优化，也不是真实终端市场价值。")
    add(f"6. **本轮不包含任何改进型实验**：没有 DP、情景优化、额外分位、参数扫描，"
        f"也没有重调度旧 q80 分支。正式 `result4-2.xlsx` 与 `result4-3.xlsx` **尚未填写**，"
        f"本轮只备齐填表数据。")
    add("")

    add("## 2. 问题分析")
    add("")
    add("问题四替换前两问的电价条件，不改变各自的信息权限：4-2 每天 0:00 制定普通购电计划、日内固定；"
        "4-3 在 0:00 制定原始计划并在 6:00/12:00/18:00 更新剩余计划。两者都通过实际储能反馈与"
        "应急购电满足负载。**电价不预测**：本轮把「每日 0:00 已知本次 144 段计划范围的附件4 交付价格」"
        "作为明确的建模解释，题面并未证明该公布机制。")
    add("")
    add(table([(g, ROLE[g], PLAN_SOURCE[g]) for g in GROUPS], ["组", "角色", "普通购电来源"]))
    add("")
    add("固定计划对照（F42/F43_S0/F43_S2）**只冻结普通购电量**；充放电、应急与未使用量都从新的公共初态"
        "重新反馈得到。因此它们既不是「按附件4 重计费的原动作」，也不是「从同一新初态重新运行固定价政策」。"
        "三者的首个正式午夜段统一改取新公共承诺，该边界替换已单独登记。")
    add("")

    add("## 3. 数据预处理")
    add("")
    price_all = pd.concat([frames[g].price_yuan_kWh for g in GROUPS])
    add(table([
        ("附件4 电价", f"365×144，逐段范围 [{num(float(price_all.min()), 4)}, "
                       f"{num(float(price_all.max()), 4)}] 元/kWh、均值 {num(float(price_all.mean()), 6)}；"
                       f"原始尖峰与低谷保留，不平滑、不标准化、不截尾"),
        ("价格映射", f"模板第 k 行第 j 列起点 = k 日 00:10＋10j 分钟；自然日午夜价取前一源行最后一列。"
                     f"逐格对照原始工作表 {len(checks['prices_and_mapping']['exact_rows'])} 个命名区间全部相等，"
                     f"午夜映射最大差 {num(checks['prices_and_mapping']['natural_midnight_max_abs'], 1)}"),
        ("公共一月", f"单条共享轨迹、{initialization['solves']} 次 MILP：1 月 1 日 00:00—00:10 缺测设电池静置、"
                     f"供需与费用留作未定义而不伪造零值；1 月 2 日零计划但恢复实际反馈；"
                     f"1 月 3—31 日朴素预测、保护为 0、名义自然 24:00 库存 6000 kWh，下一 00:10 只保留物理界"),
        ("公共初态", f"2025-02-01 00:00 内部库存 {num(initialization['feb1_state_kWh'], 6)} kWh，"
                     f"午夜原始/有效承诺 {num(initialization['feb1_carry_kWh'], 6)} / "
                     f"{num(initialization['feb1_carry_original_kWh'], 6)} kWh；六组严格共用，"
                     f"之后各自连续递推，不取其他组后续库存"),
        ("预测与保护", "Q42 用问题二 0:00 发布负载与光伏＋W28/q80；Q43_S0 用原 0:00 负载＋Linear 光伏＋q75；"
                       "Q43_S2 用 F2 日内负载修正＋同一 Linear 光伏＋自身 q75。"
                       "全部直接读取已审计档案：不重训、不重算分位、不用 q80 冒充 q75"),
        ("账本", "统一 48097 段（2025-02-01 00:00—2026-01-01 00:10）；A 自然日 48096 段为主指标，"
                 "B 模板 48096 段单列并给首尾桥接"),
    ], ["项目", "口径"]))
    add("")

    add("## 4. 模型建立")
    add("")
    add("保护净需求 `ñ = (L̂ − V̂)·Δt + ρ`（ρ 为历史净需求残差分位数，kWh）；名义优化在自由末态、"
        "库存 1200—10800 kWh、单段充放电上限 5000/6 kWh 与互斥二进制下：0 点最小化 `Σ p q⁰`，"
        "日内最小化 `Σ p [q + 0.5 a]` 且 `a ≥ |q − q⁰|`。实际反馈沿用贪心充放电与 5 倍应急价；"
        "Q43 每次把候选与更新前有效计划放在**同一最新保护、真实库存、剩余范围与价格**下评分，"
        "新评分低 1e-4 元以上才接受。结算：F42 用 `Σ p (q⁰ + 5e)`，F43 用 "
        "`Σ p (q_eff + 0.5|q_eff − q⁰| + 5e)`，价格一律取交付区间价。")
    add("")
    add(f"本轮实际调用：主求解 {validation['budget']['main_solves']} 次 MILP ＋ 额外验证 "
        f"{validation['budget']['extra_validation_milp']} 次；固定计划回放不调用 MILP。"
        f"整轮耗时 {manifest['wall_seconds']:.1f} 秒。")
    add("")

    add("## 5. 结果")
    add("")
    add("### 5.1 六组总费与三项分解")
    add("")
    add(table([(g, num(float(index.loc[g, 'natural_total_yuan'])),
                num(float(index.loc[g, 'ordinary_cost_yuan'])),
                num(float(index.loc[g, 'adjustment_cost_yuan'])),
                num(float(index.loc[g, 'emergency_cost_yuan'])),
                num(float(index.loc[g, 'template_total_yuan'])),
                num(float(index.loc[g, 'bridge_yuan']))) for g in GROUPS],
              ["组", "自然总费（元）", "普通购电（元）", "调整费（元）", "应急费（元）",
               "模板总费（元）", "首尾桥接（元）"]))
    add("")
    add(table([(name, num(float(components.loc[(name, 'total'), 'fixed_yuan'])),
                num(float(components.loc[(name, 'total'), 'candidate_yuan'])),
                signed(saving[name]), pct(saving_share[name]),
                signed(component_saving[(name, 'ordinary')]),
                signed(component_saving[(name, 'adjustment')]),
                signed(component_saving[(name, 'emergency')]))
               for name in ("S42", "S43_S0", "S43_S2")],
              ["节费量", "固定计划对照（元）", "重优化（元）", "节费（元，正=更省）", "占对照",
               "普通购电节省", "调整费节省", "应急费节省"]))
    add("")
    add("三项分项节省之和等于总节省（由计算层断言，容差 1e-4 元；"
        "调整费与应急费为负值即表示比固定计划更贵）。整体差异：Q42 − Q43_S0 = "
        f"{signed(overall['Q43_S0'][0])} 元、Q42 − Q43_S2 = {signed(overall['Q43_S2'][0])} 元、"
        f"Q43_S0 − Q43_S2 = "
        f"{signed(float(index.loc['Q43_S0', 'natural_total_yuan']) - float(index.loc['Q43_S2', 'natural_total_yuan']))} 元。")
    add("")

    add("### 5.2 应急风险")
    add("")
    add(table([(g, num(float(index.loc[g, 'emergency_kWh']), 1),
                int(index.loc[g, 'emergency_intervals']), int(index.loc[g, 'emergency_days']),
                int(index.loc[g, 'emergency_events']),
                num(float(index.loc[g, 'emergency_cost_yuan']), 1)) for g in GROUPS],
              ["组", "应急电量（kWh）", "正应急区间数", "应急自然日数", "连续事件数", "应急费（元）"]))
    add("")
    add("事件按自然日内连续正应急合并、跨午夜拆分，正应急阈值 1e-6 kWh。"
        "**重优化后应急电量与应急天数同时上升**：S43_S0 的 S0 侧 "
        f"{num(float(index.loc['F43_S0', 'emergency_kWh']), 1)} → "
        f"{num(float(index.loc['Q43_S0', 'emergency_kWh']), 1)} kWh、天数 "
        f"{int(index.loc['F43_S0', 'emergency_days'])} → {int(index.loc['Q43_S0', 'emergency_days'])}；"
        f"S43_S2 的 S2 侧 {num(float(index.loc['F43_S2', 'emergency_kWh']), 1)} → "
        f"{num(float(index.loc['Q43_S2', 'emergency_kWh']), 1)} kWh、天数 "
        f"{int(index.loc['F43_S2', 'emergency_days'])} → {int(index.loc['Q43_S2', 'emergency_days'])}。")
    add("")

    add("### 5.3 分月、逐日与极端日")
    add("")
    pivoted = monthly.pivot_table(index="month", columns="group", values="total_cost_yuan")
    daily_pivot = daily.pivot_table(index="date", columns="group", values="total_cost_yuan")
    rows = []
    for name, left, right in (("S42", "F42", "Q42"), ("S43_S0", "F43_S0", "Q43_S0"),
                              ("S43_S2", "F43_S2", "Q43_S2")):
        series = (daily_pivot[left] - daily_pivot[right]).sort_values(ascending=False)
        rows.append((name, int((series > 0).sum()), int((series < 0).sum()),
                     int((pivoted[left] - pivoted[right] > 0).sum()),
                     num(series.head(5).sum()), num(series.sum()),
                     num(100 * series.head(5).sum() / series.sum(), 1) + "%",
                     "、".join(f"{d}({signed(v, 0)})" for d, v in series.head(3).items()),
                     "、".join(f"{d}({signed(v, 0)})" for d, v in series.tail(3).items())))
    add(table(rows, ["节费量", "更省天数", "更贵天数", "更省月数", "前 5 日节省合计（元）",
                     "净节省（元）", "前 5 日占比", "最省 3 日（+ = 更省）",
                     "最贵 3 日（− = 更贵）"]))
    add("")
    add("逐日费用差按 `固定计划 − 重优化` 定义，正值表示当天重优化更省。月/日比较直接从连续轨迹聚合，"
        "未逐月重置库存。前 5 日占比明显低于问题三固定电价下的对应数字，"
        "说明本轮节省不是由少数日期拉动，但仍存在若干更贵的日子，不能读成普遍收益。")
    add("")

    add("### 5.4 指定日期")
    add("")
    for day in SELECTED_DAYS:
        block = chosen[chosen.date == day]
        add(f"**{day}** 总费（元）：" + "；".join(
            f"{g} {num(float(block[block.group == g].total_cost_yuan.iloc[0]))}" for g in GROUPS))
        add("")
        specs = []
        for group in GROUPS:
            part = frames[group]
            part = part[part.interval_start.dt.strftime("%Y-%m-%d") == day]
            values = []
            for hour in SPECIFIED_HOURS:
                hit = part[(part.interval_start.dt.hour == hour)
                           & (part.interval_start.dt.minute == 0)]
                values.append(num(float(hit.q_eff_kWh.iloc[0])) if len(hit) else "—")
            specs.append((group, *values))
        add(table(specs, ["实际有效购电 (kWh)"] + [f"{h}:00—{h}:10" for h in SPECIFIED_HOURS]))
        add("")
        blocks = []
        for group in GROUPS:
            part = frames[group]
            part = part[part.interval_start.dt.strftime("%Y-%m-%d") == day]
            cells = []
            for start_hour, _ in BLOCKS:
                sub = part[(part.interval_start.dt.hour >= start_hour)
                           & (part.interval_start.dt.hour < start_hour + 4)]
                cells.append(f"{num(float(sub.charge_kWh.sum()), 0)} / "
                             f"{num(float(sub.discharge_kWh.sum()), 0)}")
            blocks.append((group, *cells))
        add(table(blocks, ["充 / 放 (kWh)"] + [label for _, label in BLOCKS]))
        add("")
    add("上表给出题面指定的六个购电时段对应的实际有效购电量（kWh），以及自然日六个 4 小时块的"
        "充/放电量汇总（kWh）。逐段原始与有效计划、充放电、应急与库存都保存在各组 `dispatch.csv`。")
    add("")

    add("### 5.5 购电、储能与损耗")
    add("")
    add(table([(g, num(float(index.loc[g, 'q_eff_kWh']), 0),
                num(float(index.loc[g, 'charge_kWh']), 0),
                num(float(index.loc[g, 'discharge_kWh']), 0),
                num(float(index.loc[g, 'loss_kWh']), 0),
                num(float(index.loc[g, 'unused_kWh']), 0),
                num(float(index.loc[g, 'final_natural_state_kWh']), 1)) for g in GROUPS],
              ["组", "有效购电（kWh）", "充电（kWh）", "放电（kWh）", "损耗（kWh）",
               "未使用电量（kWh）", "期末自然库存（kWh）"]))
    add("")
    add("损耗按 `Σ[(1−0.9)c + (1/0.9−1)d]` 计算。**未使用电量不能全部称为弃光或弃购电**："
        "它同时包含光伏富余与已付费但未使用的购电量。")
    add("")

    add("### 5.6 库存辅助估值")
    add("")
    add(table([(g, num(float(index.loc[g, 'natural_total_yuan'])),
                num(nu * (float(index.loc[g, 'final_natural_state_kWh'])
                          - float(index.loc[g, 'initial_state_kWh'])), 1),
                num(aux[g])) for g in GROUPS],
              ["组", "现金总费（元）", "库存变动估值（元）", "C_aux（元）"]))
    add("")
    add(f"ν = median(交付电价)/0.9 = {num(nu, 6)} 元/内部 kWh。该口径只作事后敏感性，"
        f"不进入优化或真实现金；校正后排序不变，因此本轮不出现优势依赖库存边界的情形。")
    add("")

    add("### 5.7 价格与调度行为")
    add("")
    add("以 2025-06-21 为例（下图）：Q43 分支在日内 6/12/18 点重优化后，会在低价时段提高普通购电、"
        "在高价时段压低购电并更多依赖储能放电，曲线在日内节点处出现台阶；F42 与 Q42 的 4-2 分支"
        "只有一次 0 点计划，没有日内台阶。价格与净需求同时变化，因此本轮**不作单因素因果解释**。"
        "贪心反馈不会主动比较未来库存价值，这是当前 Baseline 的限制，不能据结果反向声称已实现经济 DP。")
    add("")
    paths = make_figures(summary, contrasts, frames)
    for path, caption in zip(paths, ("六组费用三项分解与三个节费量",
                                     "2025-06-21 的价格、普通购电与实际库存对应")):
        add(f"![{caption}]({path.resolve().as_posix()})")
        add("")
    add("图只做程序化完整性检查（本 Agent 无法目视图像）；图中不画未经登记的阈值线，不做显著性标注。")
    add("")

    add("## 6. 验证（三类必要验证）")
    add("")
    a1, a2, a3, led = (checks["prices_and_mapping"], checks["archive_identity"],
                       checks["price_adaptation"], checks["ledgers"])
    add(f"**A. 输入、时间与价格适配（额外 3 次 MILP）。** 附件4 逐格核对原始工作表 "
        f"{len(a1['exact_rows'])} 个命名区间（含 1 月 31 日尾段、2 月 1 日午夜与 00:10、"
        f"6 月 21 日 05:50/06:00/11:50/12:00/17:50/18:00、12 月 31 日尾段）全部相等，"
        f"自然日午夜映射最大差 {num(a1['natural_midnight_max_abs'], 1)}；同一目标在不同计划版本共用"
        f"交付区间价。预测/保护身份：Q42 负载与冻结档案最大差 "
        f"{num(a2['Q42']['load_max_abs_difference_kW'], 1)} kW；两个 Q43 分支的光伏同为 Linear 且只在"
        f"其发布可达范围内有限；S0/S2 保护与各自档案最大差 "
        f"{num(max(a2['protection_identity'].values()), 1)} kWh，0 点完全一致、日内最大差 "
        f"{num(a2['candidate_separation']['max_abs_inside_day_kWh'], 1)} kWh，说明两个候选确实不同。"
        f"对 2025-06-21 保存的 Q42 0:00 输入用同一内核与同一附件4 价格重解，目标差 "
        f"{num(a3['zero_plan']['objective_delta_yuan'], 1)} 元、名义可行性违规 "
        f"{num(a3['zero_plan']['max_feasibility_violation_kWh'], 1)} kWh；0 点与 6 点两个决策探针"
        f"在合法输入不变时复现保存决策（6 点接受判定 {a3['decision_probes'][1]['probed_accepted']}），"
        f"旧计划取计划档案的 `previous` 列。")
    add("")
    add("**B. 求解与反馈期间的廉价断言。** 逐段检查有限数、非负、库存界、功率界、能量守恒、"
        "充放电互斥与状态连续；逐次保存求解状态、gap、目标值与耗时；Q43_S0 与 Q43_S2 各有 334 次 "
        "0 点计划与 1002 次修订决策，新旧评分只比较同一未来窗口；求解失败会直接抛错，"
        "不静默退回旧计划。")
    add("")
    add(f"**C. 落盘后一次独立公式核账。** 六组各 48097 段、A/B 各 48096 段；母线平衡最大 "
        f"{num(max(led[g]['bus_balance_max_abs_kWh'] for g in GROUPS), 1)} kWh、库存递推最大 "
        f"{num(max(led[g]['recurrence_max_abs_kWh'] for g in GROUPS), 1)} kWh、状态连续最大 "
        f"{num(max(led[g]['continuity_max_abs_kWh'] for g in GROUPS), 1)} kWh、同时充放电最大 "
        f"{num(max(led[g]['simultaneous_charge_discharge_kWh'] for g in GROUPS), 1)} kWh；"
        f"两种费用公式独立重算最大差 "
        f"{num(max(max(led[g][k] for k in ('ordinary_cash_max_abs_yuan', 'adjustment_cash_max_abs_yuan', 'emergency_cash_max_abs_yuan')) for g in GROUPS), 1)} 元；"
        f"分项合计与总费最大差 {num(max(led[g]['parts_vs_total_yuan'] for g in GROUPS), 1)} 元；"
        f"六组起点库存一致。")
    add("")

    add("## 7. 复用核查与修复记录")
    add("")
    jan = reuse["january"]
    add(f"**旧草稿核查。** `results/q4_known_price/` 声明的规则（附件4 价格、同一一月配方、"
        f"Q42 = 问题二冻结预测＋q80）与本任务书一致；但其 Q43/F43 使用 Linear＋q80，"
        f"**不能改名为 S0 或 S2**，其名义表也没有保存二进制充电模式。本轮选择**不复用**，"
        f"在任务书允许的不复用上限内（{validation['budget']['main_solves']} 次）重算一月与 Q42，"
        f"使每个上报数字都有单一可复现来源，把草稿留作交叉核对：一月末态草稿 "
        f"{num(jan['draft_feb1_state_kWh'], 6)} 对本次 {num(jan['this_run_feb1_state_kWh'], 6)} kWh"
        f"（差 {num(jan['difference_kWh'], 1)}）；Q42 草稿 "
        f"{num(reuse['q42']['draft_natural_total_yuan'])} 对本次 "
        f"{num(reuse['q42']['this_run_natural_total_yuan'])} 元（差 "
        f"{num(reuse['q42']['difference_yuan'], 1)}）。两套独立运行在共同口径上互相印证。")
    add("")
    add("**未复用清单：** 旧 Q43 与 F43（Linear＋q80，不能改名替代 S0/S2）、旧草稿名义表中缺失的"
        "二进制充电模式。旧草稿目录只读，未改动。")
    add("")
    amendments = registration.get("amendments", [])
    if amendments:
        add(f"**修复记录（{len(amendments)} 条，均为实现缺陷，未改变保护口径、内核或评价规则）：**")
        for amendment in amendments:
            add(f"- `{amendment['previous_signature'][:12]}` → "
                f"`{amendment['new_signature'][:12]}`：{amendment['reason']}")
        add("")

    add("## 8. 边界与限制")
    add("")
    for item in validation["limitations"]:
        add(f"- {item}")
    add("- 固定计划回放不重新执行日内择优，其冻结的有效量只是历史计划回放基准；")
    add("- 价格适配同时覆盖优化目标、择优评分与结算三处，不是只改最后计费；")
    add("- 已知规划范围内的未来价格是合法输入，因此不能要求「改未来价格而当前计划不变」。")
    add("")

    add("## 9. 产物与复现")
    add("")
    integrity_after = manifest_integrity(manifest)
    add("计算层产物 `results/q4_price_transfer/`：`registration.json`、`run_manifest.json`、"
        "`checks.json`、`reuse_review.json`、`public_initialization.json`、公共一月账本与计划、"
        "六组 `dispatch.csv`、三条主策略的 `plan_versions.csv` / `nominal_trajectory.csv` / "
        "`solver_log.csv`、Q43_S0 与 Q43_S2 的 `revision_decisions.csv`，以及 `summary.csv`、"
        "`daily.csv`、`monthly.csv`、`contrasts.csv`、`selected_dates.csv`、`emergency_events.csv`；"
        "报告层产物本报告与 `figures/q4_price_transfer/` 两图。")
    add("")
    add(f"核账：`run_manifest.json` 状态 `{manifest['status']}`，对 {integrity_after['checked']} 项"
        f"计算层产物登记哈希，重渲本报告后不匹配 {len(integrity_after['mismatched'])} 项、缺失 "
        f"{len(integrity_after['missing'])} 项；整轮 {manifest['wall_seconds']:.1f} 秒，"
        f"MILP 共 {manifest['milp']['total_main']} ＋ {manifest['milp']['extra_validation']} 次，"
        f"`zero_solves` 全为 0。")
    add("")
    add(f"登记签名 `{registration['signature']}`；输入与源码哈希见 `registration.json`，"
        f"只登记本方案、附件2 与附件4、两份冻结档案与复用内核源码，未做全树哈希。")
    add("")
    add("复现：赛题目录执行 `conda run -n math_modeling python "
        "code/q4_price_transfer_experiment.py --mode full`，再执行 "
        "`conda run -n math_modeling python code/q4_price_transfer_report.py`。")
    add("")
    add("报告层只读 `results/q4_price_transfer/`：重渲本报告与图表不会改动计算层任何产物。"
        "**正式 `result4-2.xlsx` 与 `result4-3.xlsx` 尚未填写**，本轮只备齐填表数据；"
        "原附件、问题二正式表、问题三签名结果与旧第四问草稿均未改动。")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrity = figure_integrity(paths)
    (FIG / "figure_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "rendered", "report": REPORT_MD.name, "lines": len(lines),
                      "manifest": integrity_after, "figures": integrity}, ensure_ascii=False,
                     indent=2))


if __name__ == "__main__":
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    main()
