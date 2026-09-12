"""Generate the 5 clean publication-ready figures for Question 2.

Figures produced:
1. Fig 1: 全年费用构成与应急购电占比 (q2_fig1_annual_cost_and_emergency_share)
2. Fig 2: 四个指定日期的策略响应 (q2_fig2_representative_days_dispatch)
3. Fig 3: 名义末态规则各项费用差额 (q2_fig3_annual_cost_differences)
4. Fig 4: 2—12月逐月总费用差额 (q2_fig4_monthly_cost_differences)
5. Fig 5: 关键运行与风险指标对比 (q2_fig5_operational_metrics_comparison)

Characteristics:
- Legends placed strictly outside plot areas (above axes, no data blocking)
- Zero explanatory narrative / commentary inside plots
- Clean concise titles and axis headers
- Precise necessary node annotations only (e.g. 409.681 kWh emergency on 09-23)
- Pure white background, high contrast, academic color scheme
- Output formats: PNG (600 dpi), PDF (vector), SVG (vector)
- Targets: figures/q2_paper/ and figures/q2_time_mapping/
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "q2_time_mapping"
OUT_DIR = ROOT / "figures" / "q2_paper"
MIRROR_DIR = ROOT / "figures" / "q2_time_mapping"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib configuration for publication quality
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        "axes.edgecolor": "#78909C",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Standardized Academic Palette
C_PLAN = "#20639B"        # 计划购电费 / 计划购电量 (深蓝)
C_EMERGENCY = "#D9534F"   # 应急购电费 / 应急购电量 (珊瑚红)
C_NET = "#546E7A"         # 实际净需求 (深灰蓝虚线)
C_CHARGE = "#2E7D32"      # 储能充电 (青绿)
C_DISCHARGE = "#E65100"   # 储能放电 (橙色)
C_SOC = "#0D47A1"         # 电池内部储电量 SOC (深蓝实线)
C_ZERO = "#90A4AE"        # 零轴基准线 (浅灰)
C_BOUND = "#78909C"       # 容量边界线 (虚线灰)


def save_figure(fig: plt.Figure, base_name: str) -> None:
    """Save figure in PNG (600 dpi), PDF, and SVG formats in both directories."""
    for d in (OUT_DIR, MIRROR_DIR):
        p_png = d / f"{base_name}.png"
        p_pdf = d / f"{base_name}.pdf"
        p_svg = d / f"{base_name}.svg"
        fig.savefig(p_png, dpi=600, facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_pdf, facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_svg, facecolor="white", edgecolor="white", bbox_inches="tight")
    print(f"Saved: {base_name}.[png(600dpi), pdf, svg]")


# ==============================================================================
# Figure 1: 全年费用构成与应急购电占比
# ==============================================================================
def plot_figure_1() -> None:
    csv_path = RESULTS_DIR / "N_free" / "natural_dispatch.csv"
    df = pd.read_csv(csv_path, parse_dates=["interval_start"])
    df["month"] = df["interval_start"].dt.to_period("M").astype(str)
    
    # Formal evaluation period (2025-02 to 2025-12)
    df_eval = df[df["month"] >= "2025-02"]
    monthly = df_eval.groupby("month", as_index=False).agg(
        plan=("planned_cost_yuan", "sum"),
        emergency=("emergency_cost_yuan", "sum"),
    )
    monthly["total"] = monthly["plan"] + monthly["emergency"]
    monthly["share_pct"] = 100.0 * monthly["emergency"] / monthly["total"]
    
    months_cn = [f"{int(m.split('-')[1])}月" for m in monthly["month"]]
    x = np.arange(len(months_cn))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True, layout="constrained")
    
    # (a) Stacked bar chart
    bar_width = 0.52
    plan_w = monthly["plan"] / 1e4
    em_w = monthly["emergency"] / 1e4
    total_w = monthly["total"] / 1e4
    
    ax1.bar(x, plan_w, bar_width, label="计划购电费", color=C_PLAN, edgecolor="#1B4E79", lw=0.7)
    ax1.bar(x, em_w, bar_width, bottom=plan_w, label="应急购电费", color=C_EMERGENCY, edgecolor="#B52B27", lw=0.7)
    
    for i, tot in enumerate(total_w):
        ax1.text(i, tot + 2.0, f"{tot:.1f}", ha="center", va="bottom", fontsize=7.8)
    
    ax1.set_ylabel("费用 / 万元", fontsize=9.0, fontweight="bold")
    ax1.set_title("(a) 月度购电费用构成", fontsize=9.5, fontweight="bold", loc="left")
    ax1.set_ylim(0, 195)
    ax1.grid(axis="y", linestyle=":", alpha=0.35, color="#90A4AE")
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=8.5)
    
    # (b) Emergency percentage line plot
    ax2.plot(x, monthly["share_pct"], color=C_EMERGENCY, lw=1.8, marker="o", markersize=5.5,
             markerfacecolor="white", markeredgecolor=C_EMERGENCY, markeredgewidth=1.8, label="月度占比")
    for i, val in enumerate(monthly["share_pct"]):
        ax2.text(i, val + 0.35, f"{val:.2f}%", ha="center", va="bottom", fontsize=7.5, color="#A91E1C")
    
    tot_em_yuan = monthly["emergency"].sum()
    tot_cost_yuan = monthly["total"].sum()
    tot_share_pct = 100.0 * tot_em_yuan / tot_cost_yuan
    
    ax2.axhline(tot_share_pct, color="#546E7A", linestyle="--", lw=1.2,
                label=f"平均占比 ({tot_share_pct:.3f}%)")
    
    ax2.set_ylabel("占比 / %", fontsize=9.0, fontweight="bold")
    ax2.set_title("(b) 应急购电费用占比", fontsize=9.5, fontweight="bold", loc="left")
    ax2.set_xticks(x)
    ax2.set_xticklabels(months_cn, fontsize=9.0)
    ax2.set_xlabel("月份", fontsize=9.0, fontweight="bold")
    ax2.set_ylim(0, 9.2)
    ax2.grid(axis="y", linestyle=":", alpha=0.35, color="#90A4AE")
    ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=8.5)
    
    save_figure(fig, "q2_fig1_annual_cost_and_emergency_share")
    plt.close(fig)


# ==============================================================================
# Figure 2: 四个指定日期的策略响应
# ==============================================================================
def plot_figure_2() -> None:
    csv_path = RESULTS_DIR / "N_free" / "natural_dispatch.csv"
    df = pd.read_csv(csv_path)
    
    four_dates = [
        ("2025-03-20", "春分 (03-20)"),
        ("2025-06-21", "夏至 (06-21)"),
        ("2025-09-23", "秋分 (09-23)"),
        ("2025-12-21", "冬至 (12-21)"),
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 8.8), sharex=True, layout="constrained")
    
    time_hours = np.arange(144) / 6.0
    bar_width = 1.0 / 6.0
    
    for row_idx, (date_str, date_label) in enumerate(four_dates):
        sub = df[df["date"] == date_str].sort_values("natural_slot")
        ax_l = axes[row_idx, 0]
        ax_r = axes[row_idx, 1]
        
        # Left: Supply & Demand
        l_plan = ax_l.plot(time_hours, sub["plan_kWh"], color=C_PLAN, lw=1.5, label="计划购电量")[0]
        l_net = ax_l.plot(time_hours, sub["net_kWh"], color=C_NET, lw=1.1, ls="--", label="实际净需求")[0]
        ax_l.axhline(0, color=C_ZERO, lw=0.6, ls=":")
        
        # Emergency peak on 2025-09-23
        if sub["emergency_kWh"].sum() > 1e-4:
            em_sub = sub[sub["emergency_kWh"] > 1e-4]
            ax_l.bar(em_sub["natural_slot"] / 6.0, em_sub["emergency_kWh"],
                     width=bar_width, align="edge", color=C_EMERGENCY, edgecolor="#B52B27", lw=0.7, zorder=5)
            # Only label the necessary node value
            ax_l.annotate("409.681 kWh", xy=(20.75, 362), xytext=(15.8, 850),
                          arrowprops=dict(facecolor=C_EMERGENCY, edgecolor="#B52B27", arrowstyle="->", lw=1.0),
                          fontsize=8.0, color="#B52B27", fontweight="bold")
        
        ax_l.set_ylabel(f"{date_label}\n电量 / (kWh/10min)", fontsize=8.2, fontweight="bold")
        ax_l.set_ylim(-900, 1750)
        ax_l.grid(True, linestyle=":", alpha=0.3, color="#90A4AE")
        
        if row_idx == 0:
            ax_l.set_title("(a) 供需平衡与计划购电响应", fontsize=9.2, fontweight="bold", loc="left")
            d_bar = mpatches.Patch(facecolor=C_EMERGENCY, edgecolor="#B52B27", label="应急购电量")
            ax_l.legend(handles=[l_plan, l_net, d_bar], labels=["计划购电量", "实际净需求", "应急购电量"],
                        loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False, fontsize=8.0)
        
        # Right: Battery Flow & Stock
        ax_r.bar(time_hours, sub["charge_kWh"], width=bar_width, align="edge",
                 color=C_CHARGE, edgecolor="#1E5E24", lw=0.3, zorder=2)
        ax_r.bar(time_hours, -sub["discharge_kWh"], width=bar_width, align="edge",
                 color=C_DISCHARGE, edgecolor="#BF360C", lw=0.3, zorder=2)
        ax_r.axhline(0, color=C_ZERO, lw=0.6, ls=":")
        ax_r.set_ylabel("充放电 / (kWh/10min)", fontsize=8.2, fontweight="bold")
        ax_r.set_ylim(-900, 900)
        ax_r.grid(True, linestyle=":", alpha=0.3, color="#90A4AE")
        
        # Twinx for SOC
        ax_soc = ax_r.twinx()
        l_soc = ax_soc.plot(time_hours, sub["state_end_kWh"], color=C_SOC, lw=1.6, zorder=4)[0]
        ax_soc.axhline(10800, color=C_BOUND, ls=":", lw=0.8)
        ax_soc.axhline(1200, color=C_BOUND, ls=":", lw=0.8)
        ax_soc.set_ylabel("SOC / kWh", fontsize=8.2, fontweight="bold", color=C_SOC)
        ax_soc.set_ylim(0, 12500)
        ax_soc.tick_params(axis="y", labelcolor=C_SOC)
        
        if row_idx == 0:
            ax_soc.text(0.3, 11000, "10800 kWh", fontsize=7.0, color="#546E7A")
            ax_soc.text(0.3, 1400, "1200 kWh", fontsize=7.0, color="#546E7A")
            ax_r.set_title("(b) 储能充放电与荷电状态 (SOC)", fontsize=9.2, fontweight="bold", loc="left")
            p_c = mpatches.Patch(facecolor=C_CHARGE, edgecolor="#1E5E24", label="充电量")
            p_d = mpatches.Patch(facecolor=C_DISCHARGE, edgecolor="#BF360C", label="放电量")
            p_s = mlines.Line2D([], [], color=C_SOC, lw=1.6, label="SOC")
            ax_r.legend(handles=[p_c, p_d, p_s], labels=["充电量", "放电量", "SOC"],
                        loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False, fontsize=8.0)
    
    # Common X-axis on bottom row
    for c in range(2):
        axes[3, c].set_xlabel("时刻", fontsize=9.0, fontweight="bold")
        axes[3, c].set_xticks(np.arange(0, 25, 4))
        axes[3, c].set_xticklabels([f"{h:02d}:00" for h in np.arange(0, 25, 4)], fontsize=8.2)
    
    save_figure(fig, "q2_fig2_representative_days_dispatch")
    plt.close(fig)


# ==============================================================================
# Figure 3: 名义末态规则各项费用差额
# ==============================================================================
def plot_figure_3() -> None:
    cont = pd.read_csv(RESULTS_DIR / "contrast.csv").iloc[0]
    
    fig, ax = plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    cats = ["计划购电费差额", "应急购电费差额", "现金总费用差额"]
    vals = [
        cont["A_delta_planned_cost_yuan"],
        cont["A_delta_emergency_cost_yuan"],
        cont["A_delta_total_cost_yuan"],
    ]
    cols = ["#2E7D32", C_EMERGENCY, C_PLAN]
    x = np.arange(len(cats))
    bars = ax.bar(x, vals, width=0.45, color=cols, edgecolor="#263238", lw=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("差额 / 元", fontsize=9.0, fontweight="bold")
    ax.set_title("名义末态规则各项费用差额（自由末态 − 固定6000）", fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8.8)
    ax.set_ylim(-48000, 18000)
    ax.grid(axis="y", linestyle=":", alpha=0.35, color="#90A4AE")
    
    for bar, val in zip(bars, vals):
        y_pos = val - 3200 if val < 0 else val + 1200
        va = "top" if val < 0 else "bottom"
        sign = "+" if val > 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{sign}{val:,.2f}",
                ha="center", va=va, fontsize=8.2, fontweight="bold")
    
    save_figure(fig, "q2_fig3_annual_cost_differences")
    plt.close(fig)


# ==============================================================================
# Figure 4: 2—12月逐月总费用差额
# ==============================================================================
def plot_figure_4() -> None:
    monthly = pd.read_csv(RESULTS_DIR / "monthly_contrasts.csv")
    
    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    m_labels = [f"{int(m.split('-')[1])}月" for m in monthly["month"]]
    x_b = np.arange(len(m_labels))
    m_diffs = monthly["free_minus_fixed_yuan"].to_numpy()
    bars_b = ax.bar(x_b, m_diffs, width=0.55, color="#34495E", edgecolor="#1A252C", lw=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("总费用差额 / 元", fontsize=9.0, fontweight="bold")
    ax.set_title("2—12月逐月总费用差额（自由末态 − 固定6000）", fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xticks(x_b)
    ax.set_xticklabels(m_labels, fontsize=8.5)
    ax.set_xlabel("月份", fontsize=9.0, fontweight="bold")
    ax.set_ylim(-5400, 500)
    ax.grid(axis="y", linestyle=":", alpha=0.35, color="#90A4AE")
    
    for bar, val in zip(bars_b, m_diffs):
        ax.text(bar.get_x() + bar.get_width()/2, val - 240, f"{val:.0f}",
                ha="center", va="top", fontsize=7.5)
    
    save_figure(fig, "q2_fig4_monthly_cost_differences")
    plt.close(fig)


# ==============================================================================
# Figure 5: 关键运行与风险指标对比
# ==============================================================================
def plot_figure_5() -> None:
    summ = pd.read_csv(RESULTS_DIR / "summary.csv")
    
    fig, ax = plt.subplots(figsize=(6.8, 4.0), layout="constrained")
    m_labs = ["应急购电量\n(万 kWh)", "储能循环损耗\n(万 kWh)", "实际年末库存\n(千 kWh)", "应急天数\n(天)"]
    f_row = summ[summ["strategy_id"] == "N_fixed6000"].iloc[0]
    r_row = summ[summ["strategy_id"] == "N_free"].iloc[0]
    
    v_fixed = [
        f_row["emergency_kWh"] / 1e4,
        f_row["loss_kWh"] / 1e4,
        f_row["final_kWh"] / 1e3,
        f_row["emergency_days"],
    ]
    v_free = [
        r_row["emergency_kWh"] / 1e4,
        r_row["loss_kWh"] / 1e4,
        r_row["final_kWh"] / 1e3,
        r_row["emergency_days"],
    ]
    
    x_c = np.arange(len(m_labs))
    w_c = 0.35
    b1 = ax.bar(x_c - w_c/2, v_fixed, w_c, label="固定6000", color="#78909C", edgecolor="#37474F", lw=0.7)
    b2 = ax.bar(x_c + w_c/2, v_free, w_c, label="自由末态", color=C_EMERGENCY, edgecolor="#922B21", lw=0.7)
    
    ax.set_title("关键运行与风险指标对比（固定6000 vs 自由末态）", fontsize=9.5, fontweight="bold", loc="left")
    ax.set_xticks(x_c)
    ax.set_xticklabels(m_labs, fontsize=8.8)
    ax.set_ylabel("数值（按对应指标单位）", fontsize=9.0, fontweight="bold")
    ax.set_ylim(0, 145)
    ax.grid(axis="y", linestyle=":", alpha=0.35, color="#90A4AE")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=8.5)
    
    for bar1, bar2, v1, v2 in zip(b1, b2, v_fixed, v_free):
        fmt1 = f"{v1:.1f}" if v1 > 10 else f"{v1:.2f}"
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 2.0, fmt1, ha="center", va="bottom", fontsize=7.5, color="#37474F")
        fmt2 = f"{v2:.1f}" if v2 > 10 else f"{v2:.2f}"
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 2.0, fmt2, ha="center", va="bottom", fontsize=7.5, color="#922B21", fontweight="bold")
    
    save_figure(fig, "q2_fig5_operational_metrics_comparison")
    plt.close(fig)


def main() -> None:
    print("=== Generating 5 Publication Figures for Question 2 ===")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    print("=== All Figures Successfully Generated! ===")


if __name__ == "__main__":
    main()
