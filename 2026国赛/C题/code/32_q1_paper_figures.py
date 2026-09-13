"""Generate publication-ready figures for Question 1.

1. Fig 1: 储能状态全天变化与运行边界 (q1_fig1_soc_trajectory)
   - Data source: results/q1_milp/dispatch_10min.csv
   - Visuals: 145-node state trajectory, upper/lower bounds (1200 / 10800 kWh),
     initial/terminal state (6000 kWh), shaded charging/discharging intervals,
     key extremes labeled clearly without blocking lines.
   - Legend strictly outside the plot area.

2. Fig 2: 储能容量与充放电功率参数灵敏度分析 (q1_fig2_storage_sensitivity)
   - Data source: results/q1_sensitivity/capacity_power_sensitivity.csv
   - Visuals: 1x2 subplots (left: Capacity sensitivity, right: Power sensitivity),
     baseline indicators (12000 kWh, 5000 kW, baseline cost 35126.95 yuan),
     concise node values on baseline & extremes, zero text boxes.
   - Legend placed outside plot areas.

Outputs:
- PNG (600 dpi), PDF (vector), SVG (vector)
- Placed into: figures/q1_paper/ and figures/q1_sensitivity/
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MILP_CSV = ROOT / "results" / "q1_milp" / "dispatch_10min.csv"
SENS_CSV = ROOT / "results" / "q1_sensitivity" / "capacity_power_sensitivity.csv"
OUT_DIR = ROOT / "figures" / "q1_paper"
MIRROR_DIR = ROOT / "figures" / "q1_sensitivity"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"],
        "font.size": 9.0,
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

C_PRIMARY = "#1A5276"    # 深蓝
C_ACCENT = "#C0392B"     # 砖红 (下限/警示)
C_WARN = "#D35400"       # 橙红 (上限)
C_CHARGE = "#27AE60"     # 翡翠绿 (充电)
C_DISCHARGE = "#E67E22"  # 暖橙 (放电)
C_BASE = "#7F8C8D"       # 灰 (基准参考线)


def save_fig(fig: plt.Figure, base_name: str) -> None:
    for d in (OUT_DIR, MIRROR_DIR):
        p_png = d / f"{base_name}.png"
        p_pdf = d / f"{base_name}.pdf"
        p_svg = d / f"{base_name}.svg"
        fig.savefig(p_png, dpi=600, facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_pdf, facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_svg, facecolor="white", edgecolor="white", bbox_inches="tight")


def plot_fig1_soc() -> None:
    df = pd.read_csv(MILP_CSV)
    # Build 145 continuous state points: minutes 0, 10, ..., 1440
    minutes = np.append(df["start_minute"].to_numpy(), df["end_minute"].iloc[-1])
    hours = minutes / 60.0
    soc = np.append(df["state_start_kWh"].to_numpy(), df["state_end_kWh"].iloc[-1])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    # Shading for charge and discharge intervals
    for _, r in df.iterrows():
        t0 = r["start_minute"] / 60.0
        t1 = r["end_minute"] / 60.0
        if r["charge_bus_kWh"] > 1e-4:
            ax.axvspan(t0, t1, color=C_CHARGE, alpha=0.10, lw=0)
        elif r["discharge_bus_kWh"] > 1e-4:
            ax.axvspan(t0, t1, color=C_DISCHARGE, alpha=0.10, lw=0)

    # Physical boundary lines
    ax.axhline(10800, color=C_WARN, lw=1.2, ls="--", zorder=2)
    ax.axhline(1200, color=C_ACCENT, lw=1.2, ls="--", zorder=2)
    ax.axhline(6000, color=C_BASE, lw=0.9, ls=":", zorder=2)

    # Main SOC curve
    ax.plot(hours, soc, color=C_PRIMARY, lw=2.0, label="储电量 (kWh)", zorder=4)

    # Key points: Start, End, Max, Min
    # Initial & terminal
    ax.scatter([0.0, 24.0], [6000.0, 6000.0], color=C_PRIMARY, s=32, zorder=5)
    # Min SOC: index 124 in df (20:40-20:50, end at 20:50 -> 20.833h)
    min_idx = np.argmin(soc)
    max_idx = np.argmax(soc)
    ax.scatter([hours[min_idx], hours[max_idx]], [soc[min_idx], soc[max_idx]],
               color=[C_ACCENT, C_WARN], s=42, zorder=5)

    # Precise non-overlapping annotations
    ax.annotate(f"下限触达: {soc[min_idx]:.0f} kWh\n(20:50)",
                xy=(hours[min_idx], soc[min_idx]),
                xytext=(hours[min_idx] - 3.2, soc[min_idx] + 1300),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.9),
                fontsize=8.2, color=C_ACCENT, ha="center")

    ax.annotate(f"上限触达: {soc[max_idx]:.0f} kWh\n(05:50)",
                xy=(hours[max_idx], soc[max_idx]),
                xytext=(hours[max_idx] + 2.4, soc[max_idx] - 1600),
                arrowprops=dict(arrowstyle="->", color=C_WARN, lw=0.9),
                fontsize=8.2, color=C_WARN, ha="center")

    ax.annotate("初态: 6000 kWh", xy=(0.0, 6000.0), xytext=(1.8, 6300.0),
                fontsize=8.0, color="#424242")
    ax.annotate("末态: 6000 kWh", xy=(24.0, 6000.0), xytext=(21.5, 6300.0),
                fontsize=8.0, color="#424242")

    # Limit labels directly on axis
    ax.text(24.1, 10800, "上限 10800", color=C_WARN, va="center", fontsize=8.0)
    ax.text(24.1, 1200, "下限 1200", color=C_ACCENT, va="center", fontsize=8.0)
    ax.text(24.1, 6000, "基准 6000", color=C_BASE, va="center", fontsize=8.0)

    # Formatting axes
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12000)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.set_xlabel("时间 (时:分)", labelpad=6)
    ax.set_ylabel("储电量 / kWh", labelpad=6)
    ax.grid(axis="y", color="#ECEFF1", ls="-", lw=0.7, zorder=1)

    # External legend positioned at the top-right above the axes
    leg_handles = [
        mlines.Line2D([], [], color=C_PRIMARY, lw=2.0, label="储电量状态"),
        mlines.Line2D([], [], color=C_WARN, lw=1.2, ls="--", label="容量上限 (10800 kWh)"),
        mlines.Line2D([], [], color=C_ACCENT, lw=1.2, ls="--", label="容量下限 (1200 kWh)"),
        mpatches.Patch(facecolor=C_CHARGE, alpha=0.3, label="充电时段"),
        mpatches.Patch(facecolor=C_DISCHARGE, alpha=0.3, label="放电时段"),
    ]
    ax.legend(handles=leg_handles, loc="lower right", bbox_to_anchor=(1.0, 1.02),
              ncol=5, frameon=False, fontsize=8.0, handlelength=1.6, columnspacing=1.0)

    ax.set_title("问题一全天储能SOC状态变化与运行边界", loc="left", fontsize=10.5, fontweight="bold", pad=28)
    save_fig(fig, "q1_fig1_soc_trajectory")
    plt.close(fig)


def plot_fig2_sensitivity() -> None:
    df = pd.read_csv(SENS_CSV)
    df_cap = df[df["scan_axis"] == "capacity"].sort_values("capacity_kWh").copy()
    df_pow = df[df["scan_axis"] == "power"].sort_values("power_max_kW").copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2))

    # --- Left: Capacity Sensitivity ---
    ax1.plot(df_cap["capacity_kWh"] / 1000.0, df_cap["cost_yuan"] / 10000.0,
             color=C_PRIMARY, lw=1.8, marker="o", markersize=5.0, zorder=4)
    # Baseline line (12000 kWh, 3.5127 万元)
    ax1.axvline(12.0, color=C_BASE, lw=0.9, ls="--", zorder=2)
    ax1.axhline(3.512695, color=C_BASE, lw=0.9, ls=":", zorder=2)

    # Label baseline and extremes
    base_r = df_cap[df_cap["capacity_kWh"] == 12000.0].iloc[0]
    min_r = df_cap.iloc[0]
    max_r = df_cap.iloc[-1]

    ax1.scatter([12.0], [base_r["cost_yuan"] / 10000.0], color=C_ACCENT, s=36, zorder=5)
    ax1.annotate(f"基准: 3.513 万元\n(12000 kWh)", xy=(12.0, base_r["cost_yuan"] / 10000.0),
                 xytext=(12.0, base_r["cost_yuan"] / 10000.0 + 0.28),
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.8),
                 fontsize=8.0, color=C_ACCENT, ha="center")

    ax1.annotate(f"{min_r['cost_yuan']/10000.0:.3f} 万元",
                 xy=(min_r["capacity_kWh"] / 1000.0, min_r["cost_yuan"] / 10000.0),
                 xytext=(min_r["capacity_kWh"] / 1000.0 + 1.2, min_r["cost_yuan"] / 10000.0 - 0.08),
                 fontsize=8.0, color="#424242")
    ax1.annotate(f"{max_r['cost_yuan']/10000.0:.3f} 万元",
                 xy=(max_r["capacity_kWh"] / 1000.0, max_r["cost_yuan"] / 10000.0),
                 xytext=(max_r["capacity_kWh"] / 1000.0 - 1.8, max_r["cost_yuan"] / 10000.0 + 0.12),
                 fontsize=8.0, color="#424242")

    ax1.set_xlabel("储能容量 / (×10³ kWh)", labelpad=6)
    ax1.set_ylabel("最优购电费用 / 万元", labelpad=6)
    ax1.set_xlim(5.0, 19.0)
    ax1.set_ylim(3.0, 4.3)
    ax1.set_xticks(df_cap["capacity_kWh"] / 1000.0)
    ax1.grid(axis="y", color="#ECEFF1", ls="-", lw=0.7, zorder=1)
    ax1.set_title("(a) 储能最大容量灵敏度 (功率固定 5000 kW)", loc="left", fontsize=9.2, fontweight="bold", pad=10)

    # --- Right: Power Sensitivity ---
    ax2.plot(df_pow["power_max_kW"] / 1000.0, df_pow["cost_yuan"] / 10000.0,
             color="#2E7D32", lw=1.8, marker="s", markersize=5.0, zorder=4)
    # Baseline line (5000 kW, 3.5127 万元)
    ax2.axvline(5.0, color=C_BASE, lw=0.9, ls="--", zorder=2)
    ax2.axhline(3.512695, color=C_BASE, lw=0.9, ls=":", zorder=2)

    base_p = df_pow[df_pow["power_max_kW"] == 5000.0].iloc[0]
    min_p = df_pow.iloc[0]
    max_p = df_pow.iloc[-1]

    ax2.scatter([5.0], [base_p["cost_yuan"] / 10000.0], color=C_ACCENT, s=36, zorder=5)
    ax2.annotate(f"基准: 3.513 万元\n(5000 kW)", xy=(5.0, base_p["cost_yuan"] / 10000.0),
                 xytext=(5.0, base_p["cost_yuan"] / 10000.0 + 0.22),
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.8),
                 fontsize=8.0, color=C_ACCENT, ha="center")

    ax2.annotate(f"{min_p['cost_yuan']/10000.0:.3f} 万元",
                 xy=(min_p["power_max_kW"] / 1000.0, min_p["cost_yuan"] / 10000.0),
                 xytext=(min_p["power_max_kW"] / 1000.0 + 0.4, min_p["cost_yuan"] / 10000.0 - 0.08),
                 fontsize=8.0, color="#424242")
    ax2.annotate(f"{max_p['cost_yuan']/10000.0:.3f} 万元",
                 xy=(max_p["power_max_kW"] / 1000.0, max_p["cost_yuan"] / 10000.0),
                 xytext=(max_p["power_max_kW"] / 1000.0 - 1.2, max_p["cost_yuan"] / 10000.0 + 0.10),
                 fontsize=8.0, color="#424242")

    ax2.set_xlabel("最大充放电功率 / (×10³ kW)", labelpad=6)
    ax2.set_ylabel("最优购电费用 / 万元", labelpad=6)
    ax2.set_xlim(1.5, 7.5)
    ax2.set_ylim(3.45, 3.85)
    ax2.set_xticks([2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0])
    ax2.grid(axis="y", color="#ECEFF1", ls="-", lw=0.7, zorder=1)
    ax2.set_title("(b) 最大充放电功率灵敏度 (容量固定 12000 kWh)", loc="left", fontsize=9.2, fontweight="bold", pad=10)

    # Global legend outside plot
    leg_handles = [
        mlines.Line2D([], [], color=C_PRIMARY, lw=1.8, marker="o", markersize=4.5, label="容量变化响应"),
        mlines.Line2D([], [], color="#2E7D32", lw=1.8, marker="s", markersize=4.5, label="功率变化响应"),
        mlines.Line2D([], [], color=C_BASE, lw=1.0, ls="--", label="基准参数线"),
        mlines.Line2D([], [], color=C_BASE, lw=1.0, ls=":", label="基准费用线 (3.513 万元)"),
        mlines.Line2D([], [], color="none", marker="o", markerfacecolor=C_ACCENT, markeredgecolor=C_ACCENT,
                      markersize=6, label="基准工作点"),
    ]
    fig.legend(handles=leg_handles, loc="lower right", bbox_to_anchor=(0.98, 0.94),
               ncol=5, frameon=False, fontsize=8.0, handlelength=1.6, columnspacing=1.0)

    fig.suptitle("问题一储能容量与最大充放电功率对购电费用的影响", x=0.08, y=1.03,
                 fontsize=10.5, fontweight="bold", ha="left")

    plt.tight_layout()
    save_fig(fig, "q1_fig2_storage_sensitivity")
    plt.close(fig)


def main() -> None:
    plot_fig1_soc()
    plot_fig2_sensitivity()
    print("Question 1 publication figures successfully generated.")


if __name__ == "__main__":
    main()
