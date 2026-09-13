"""Generate publication-ready SOC trajectory figure for Question 3.

Fixes previous plotting bugs:
1. Previously only took 1 daily summary row (0-10 min) instead of 144-interval continuous trajectory.
2. Now reads full 144 intervals (145 boundary state points) from B0_dispatch.csv, B1_dispatch.csv, B2_dispatch.csv.
3. Places a unified global legend at the top outside the axes to avoid blocking curves.
4. Removes redundant legends and annotations inside individual subplots.
5. Uses formal academic styling and Chinese labels with English sub-labels if appropriate.
6. Outputs 600 DPI PNG, vector PDF, and vector SVG to figures/q3_paper/ and figures/q3_rolling_baseline/.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results" / "q3_rolling_baseline"
OUT_PAPER_DIR = ROOT / "figures" / "q3_paper"
OUT_REPORT_DIR = ROOT / "figures" / "q3_rolling_baseline"

OUT_PAPER_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"],
        "font.size": 9.5,
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

# Colors
C_B0 = "#1F77B4"    # 经典深蓝 (B0 基准)
C_B1 = "#D35400"    # 暖橙 (B1 日前计划)
C_B2 = "#27AE60"    # 翡翠绿 (B2 滚动优化)
C_BOUND = "#7F8C8D" # 物理边界虚线灰

SELECTED_DATES = [
    ("2025-03-20", "(a) 春季分日 (2025-03-20)"),
    ("2025-06-21", "(b) 夏季分日 (2025-06-21)"),
    ("2025-09-23", "(c) 秋季分日 (2025-09-23)"),
    ("2025-12-21", "(d) 冬季分日 (2025-12-21)"),
]

GROUPS = [
    ("B0", "B0: 日前优化基准 (问题二)", C_B0, "-"),
    ("B1", "B1: 日前单次计划 (附件3)", C_B1, "--"),
    ("B2", "B2: 日内滚动多阶段优化", C_B2, "-"),
]


def load_trajectories():
    """Load 145-point continuous state trajectory for each date and group."""
    data = {}
    for g, _, _, _ in GROUPS:
        csv_path = RES_DIR / f"{g}_dispatch.csv"
        df = pd.read_csv(csv_path)
        df["dt"] = pd.to_datetime(df["interval_start"])
        data[g] = df
    return data


def plot_soc_trajectories():
    dfs = load_trajectories()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (date_str, title_text) in enumerate(SELECTED_DATES):
        ax = axes_flat[idx]

        # Plot physical bounds
        ax.axhline(10800, color=C_BOUND, lw=1.0, ls="--", zorder=2)
        ax.axhline(1200, color=C_BOUND, lw=1.0, ls="--", zorder=2)

        # Plot trajectory for each group
        for g, label, col, ls in GROUPS:
            day_df = dfs[g][dfs[g]["dt"].dt.strftime("%Y-%m-%d") == date_str].sort_values("dt")
            if day_df.empty:
                continue

            # 145 points: minute 0, 10, ..., 1440
            state = np.append(day_df["state_start_kWh"].to_numpy(), day_df["state_end_kWh"].iloc[-1])
            hours = np.linspace(0, 24, len(state))

            lw = 1.8 if g == "B2" else 1.3
            alpha = 0.95 if g == "B2" else 0.85
            ax.plot(hours, state, color=col, lw=lw, ls=ls, alpha=alpha, zorder=4)

        # Boundary labels on subplots (right side of plot area)
        if idx in (1, 3):
            ax.text(24.1, 10800, "上限 10800 kWh", color="#566573", fontsize=7.5, va="center")
            ax.text(24.1, 1200, "下限 1200 kWh", color="#566573", fontsize=7.5, va="center")

        ax.set_title(title_text, fontsize=10.0, fontweight="bold", loc="left", pad=8)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 12000)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], fontsize=8.5)
        ax.grid(axis="both", ls=":", lw=0.5, alpha=0.45, zorder=1)

        if idx in (0, 2):
            ax.set_ylabel("储能荷电量 SOC (kWh)", fontsize=9.2)
        if idx in (2, 3):
            ax.set_xlabel("自然日运行时间 (小时)", fontsize=9.2)

    # Unified global legend placed strictly outside the axes at the top
    handles = [
        mlines.Line2D([], [], color=C_B0, lw=1.4, ls="-", label="B0: 日前优化基准 (问题二)"),
        mlines.Line2D([], [], color=C_B1, lw=1.4, ls="--", label="B1: 日前单次计划 (附件3)"),
        mlines.Line2D([], [], color=C_B2, lw=1.8, ls="-", label="B2: 日内滚动多阶段优化"),
        mlines.Line2D([], [], color=C_BOUND, lw=1.0, ls="--", label="物理边界 (1200 / 10800 kWh)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
        fontsize=9.0,
    )

    # Overall formal title
    fig.suptitle(
        "问题三典型代表日储能荷电状态 (SOC) 连续运行轨迹对比",
        fontsize=12.0,
        fontweight="bold",
        y=1.02,
    )

    # Adjust subplot layout
    fig.subplots_adjust(top=0.91, bottom=0.08, left=0.08, right=0.93, hspace=0.25, wspace=0.18)

    # Save to both paper and report directories
    for p_dir, fname in [
        (OUT_PAPER_DIR, "q3_fig1_selected_dates_soc"),
        (OUT_REPORT_DIR, "selected_dates_soc"),
    ]:
        fig.savefig(p_dir / f"{fname}.png", dpi=600, facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_dir / f"{fname}.pdf", facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_dir / f"{fname}.svg", facecolor="white", edgecolor="white", bbox_inches="tight")

    plt.close(fig)
    print("Question 3 SOC trajectory publication figure successfully generated.")


if __name__ == "__main__":
    plot_soc_trajectories()
