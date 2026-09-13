"""Generate publication-ready figures for Question 4.

Fig 1: Q43_S2 相对 Q42 逐月节费趋势与分项解构 (q4_fig1_monthly_savings_and_decomposition)
- Data source: results/q4_price_transfer/monthly.csv
- Panel (a): Q43_S2 相对 Q42 逐月费用差额与累计节费 (2025年2—12月)
  * Bar chart: Monthly net savings (万元, positive = Q43_S2 saves money), 11/11 months positive.
  * Right axis: Cumulative net savings curve (climbing up to 37.51 万元).
  * Seasonal division: Winter (2, 12), Spring (3-5), Summer (6-8), Autumn (9-11).
  * Peak annotations: June summer peak (11.89 万元), December winter peak (6.59 万元).
- Panel (b): 逐月节费分项构成拆解 (普通购电节费 / 应急费节费 / 滚动调整费代价)
  * Diverging stacked bars:
    - Positive stack: Ordinary power purchase savings + Emergency cost savings.
    - Negative stack: Intraday adjustment penalty cost.
  * Overlay line: Monthly net savings curve traversing above zero line across all 11 months.
- Legend strictly outside the plot area on top.
- Outputs: PNG (600 dpi), vector PDF, vector SVG in figures/q4_paper/ and figures/q4_price_transfer/.
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
MONTHLY_CSV = ROOT / "results" / "q4_price_transfer" / "monthly.csv"
OUT_DIR = ROOT / "figures" / "q4_paper"
MIRROR_DIR = ROOT / "figures" / "q4_price_transfer"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MIRROR_DIR.mkdir(parents=True, exist_ok=True)

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
C_NAVY = "#1B4F72"       # 主深蓝 (普通购电)
C_EMERALD = "#229954"    # 翠绿 (应急节费)
C_CORAL = "#CB4335"      # 珊瑚红 (调整惩罚 / 负贡献)
C_PRIMARY = "#2E86C1"    # 亮青蓝 (净节费柱)
C_LINE = "#17202A"       # 净节费折线 (深黑炭色)
C_ACCUM = "#7D3C98"      # 紫色 (累计节费)
C_ZERO = "#566573"       # 零基准线灰


def save_fig(fig: plt.Figure, base_name: str) -> None:
    for d in (OUT_DIR, MIRROR_DIR):
        p_png = d / f"{base_name}.png"
        p_pdf = d / f"{base_name}.pdf"
        p_svg = d / f"{base_name}.svg"
        fig.savefig(p_png, dpi=600, facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_pdf, facecolor="white", edgecolor="white", bbox_inches="tight")
        fig.savefig(p_svg, facecolor="white", edgecolor="white", bbox_inches="tight")


def plot_fig_monthly_savings() -> None:
    df = pd.read_csv(MONTHLY_CSV)
    q42 = df[df["group"] == "Q42"].set_index("month")
    s2 = df[df["group"] == "Q43_S2"].set_index("month")

    months = q42.index.tolist()
    month_labels = [f"{int(m.split('-')[1])}月" for m in months]
    n_months = len(months)
    x = np.arange(n_months)

    # Values in 万元
    net_save = (q42["total_cost_yuan"] - s2["total_cost_yuan"]).to_numpy() / 10000.0
    ord_save = (q42["ordinary_cost_yuan"] - s2["ordinary_cost_yuan"]).to_numpy() / 10000.0
    emg_save = (q42["emergency_cost_yuan"] - s2["emergency_cost_yuan"]).to_numpy() / 10000.0
    adj_cost = s2["adjustment_cost_yuan"].to_numpy() / 10000.0  # Q43_S2 cost, so drag is -adj_cost
    accum_save = np.cumsum(net_save)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"wspace": 0.28})

    # =========================================================================
    # Panel (a): 逐月净节费与累计节省
    # =========================================================================
    # Background seasonal shading
    # 2月 (idx 0): 冬; 3-5月 (idx 1-3): 春; 6-8月 (idx 4-6): 夏; 9-11月 (idx 7-9): 秋; 12月 (idx 10): 冬
    seasons = [
        (-0.5, 0.5, "#EBF5FB", "冬季"),
        (0.5, 3.5, "#EAFAF1", "春季"),
        (3.5, 6.5, "#FEF9E7", "夏季"),
        (6.5, 9.5, "#F8F9F9", "秋季"),
        (9.5, 10.5, "#EBF5FB", "冬季"),
    ]
    for x0, x1, col, sname in seasons:
        ax1.axvspan(x0, x1, color=col, alpha=0.65, lw=0, zorder=0)
        ax1.text((x0 + x1) / 2.0, 13.8, sname, ha="center", va="top", fontsize=8.2, color="#5D6D7E", fontweight="bold")

    # Bar chart for monthly net savings
    bars1 = ax1.bar(x, net_save, width=0.55, color=C_PRIMARY, edgecolor="#1B4F72", lw=0.8, zorder=3, label="当月净节费 (万元)")

    # Bar labels
    for i, bar in enumerate(bars1):
        v = net_save[i]
        va = "bottom"
        y_pos = v + 0.25
        ax1.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"+{v:.2f}", ha="center", va=va, fontsize=8.0, color="#1B4F72", fontweight="bold", zorder=4)

    # Zero baseline
    ax1.axhline(0, color=C_ZERO, lw=1.0, zorder=2)

    # Secondary axis for cumulative savings
    ax1_twin = ax1.twinx()
    ax1_twin.spines["top"].set_visible(False)
    ax1_twin.spines["right"].set_color("#78909C")
    ax1_twin.spines["left"].set_visible(False)
    line_acc = ax1_twin.plot(x, accum_save, color=C_ACCUM, marker="o", markersize=4.5, lw=1.8, ls="--", zorder=5, label="累计净节费 (万元)")[0]
    ax1_twin.set_ylabel("累计净节费 (万元)", color=C_ACCUM, fontsize=9.5)
    ax1_twin.tick_params(axis="y", labelcolor=C_ACCUM)
    ax1_twin.set_ylim(0, 45)

    # Annotate total cumulative savings at last point
    ax1_twin.scatter([x[-1]], [accum_save[-1]], color=C_ACCUM, s=40, zorder=6)
    ax1_twin.text(x[-1] - 0.1, accum_save[-1] + 1.2, f"全年累计\n37.51万元", ha="right", va="bottom", fontsize=8.2, color=C_ACCUM, fontweight="bold", zorder=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels, fontsize=9.0)
    ax1.set_xlabel("时间 (2025年月份)", fontsize=9.5)
    ax1.set_ylabel("净费用差额 $C_{\\mathrm{Q42}} - C_{\\mathrm{Q43\\_S2}}$ (万元)", fontsize=9.5)
    ax1.set_ylim(-0.8, 14.8)
    ax1.grid(axis="y", ls=":", lw=0.6, alpha=0.55, zorder=1)
    ax1.set_title("(a) Q43_S2 相对 Q42 逐月净节费与累计节省趋势", fontsize=10.5, pad=18, fontweight="bold", loc="left")

    # Legend outside on top
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=8.5)

    # =========================================================================
    # Panel (b): 逐月节费分项构成拆解 (普通购电 / 应急费 / 调整费)
    # =========================================================================
    width2 = 0.55
    # Positive stack: Ordinary savings (bottom) + Emergency savings (top)
    bar_ord = ax2.bar(x, ord_save, width=width2, color=C_NAVY, edgecolor="none", lw=0.6, zorder=3, label="普通购电节省 (正贡献)")
    bar_emg = ax2.bar(x, emg_save, bottom=ord_save, width=width2, color=C_EMERALD, edgecolor="none", lw=0.6, zorder=3, label="应急补电节省 (正贡献)")

    # Negative stack: Adjustment cost penalty
    bar_adj = ax2.bar(x, -adj_cost, width=width2, color=C_CORAL, edgecolor="none", lw=0.6, zorder=3, label="滚动调整代价 (负拖累)")

    # Zero baseline
    ax2.axhline(0, color=C_ZERO, lw=1.0, zorder=2)

    # Net savings curve overlay
    line_net = ax2.plot(x, net_save, color=C_LINE, marker="D", markersize=4.0, lw=1.6, ls="-", zorder=5, label="当月净节费")[0]

    # Annotate key emergency spikes in June and December
    ax2.annotate(
        "夏季光伏波动大\n应急防范省9.20万",
        xy=(x[4], ord_save[4] + emg_save[4]),
        xytext=(x[4] + 0.1, ord_save[4] + emg_save[4] + 1.2),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=C_EMERALD,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_EMERALD, lw=0.9),
        zorder=6,
    )
    ax2.annotate(
        "冬季高负荷保供\n应急节约4.45万",
        xy=(x[10], ord_save[10] + emg_save[10]),
        xytext=(x[10] - 0.3, ord_save[10] + emg_save[10] + 1.2),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=C_EMERALD,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_EMERALD, lw=0.9),
        zorder=6,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels, fontsize=9.0)
    ax2.set_xlabel("时间 (2025年月份)", fontsize=9.5)
    ax2.set_ylabel("分项差额贡献 (万元)", fontsize=9.5)
    ax2.set_ylim(-8.5, 20.0)
    ax2.grid(axis="y", ls=":", lw=0.6, alpha=0.55, zorder=1)
    ax2.set_title("(b) 逐月节费来源三项拆解 (普通购电/应急/调整费)", fontsize=10.5, pad=18, fontweight="bold", loc="left")

    # Legend outside on top
    h_b, l_b = ax2.get_legend_handles_labels()
    ax2.legend(h_b, l_b, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=False, fontsize=8.2)

    # Position fine-tuning without tight_layout collision
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.93, wspace=0.28)
    save_fig(fig, "q4_fig1_monthly_savings_and_decomposition")
    plt.close(fig)
    print("Question 4 monthly savings figure successfully generated.")


if __name__ == "__main__":
    plot_fig_monthly_savings()
