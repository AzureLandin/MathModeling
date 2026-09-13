"""Generate publication-ready figures for Question 4.

Fig 1: Q43_S2 相对 Q42 逐月节费与成本解构 (q4_fig1_monthly_savings_and_decomposition)
- Data source: results/q4_price_transfer/monthly.csv
- Panel (a): 逐月净节费与全年累计净节费
  * 柱状图: 当月净节费 (万元, 正数表示 Q43_S2 相比 Q42 更省), 11/11 个月为正.
  * 右轴折线: 累计净节费 (万元, 攀升至全年 37.51 万元).
  * 关键节点标注: 6月峰值 (+11.89 万元), 12月次峰 (+6.59 万元), 11月谷值 (+0.37 万元), 全年累计 (37.51 万元).
- Panel (b): 节费来源三项拆解 (普通购电节省 / 应急补电节省 / 滚动调整代价)
  * 发散堆叠柱: 正向为购电节省与应急节省, 负向为滚动调整代价.
  * 叠加坡度折线: 当月净节费 (全程处于 0 轴上方).
  * 关键节点标注: 6月最大净省 (+11.89 万元).
- 严格遵循学术规范:
  * 正式标题与表头体系 (顶层大标题 + 子图表头 + 物理量及单位).
  * 图例完全外置于绘图区上方独立空间, 层次分明, 杜绝任何重叠与遮挡.
  * 仅标注关键拐点/极值与总量节点, 去除冗余杂乱文本.
  * 输出: 600 DPI 印刷级 PNG, 矢量 PDF, 矢量 SVG.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
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

# Formal academic color palette
C_NAVY = "#1B4F72"       # 普通购电节省 (深蓝)
C_EMERALD = "#229954"    # 应急补电节省 (翠绿)
C_CORAL = "#CB4335"      # 滚动调整代价 (砖红/负贡献)
C_PRIMARY = "#2E86C1"    # 月度净节费柱 (标准蓝)
C_LINE = "#17202A"       # 月度净节费折线 (玄黑)
C_ACCUM = "#7D3C98"      # 累计净节费 (优雅紫)
C_ZERO = "#566573"       # 零基准线深灰


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
    adj_cost = s2["adjustment_cost_yuan"].to_numpy() / 10000.0  # Drag is -adj_cost
    accum_save = np.cumsum(net_save)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8))
    # Ample top clearance (top=0.78) for suptitle, panel titles, and out-of-plot legends
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.07, right=0.93, wspace=0.28)

    st = fig.suptitle(
        "电价迁移场景下日内滚动多阶段优化 (Q43_S2) 相对固定日前计划 (Q42) 逐月节费与成本解构",
        fontsize=12.5,
        y=0.96,
        fontweight="bold",
    )

    # =========================================================================
    # Panel (a): 逐月净节费与累计节省趋势
    # =========================================================================
    ax1.set_title("(a) 逐月净节费与全年累计净节费", fontsize=10.5, pad=34, fontweight="bold")

    # Bar chart for monthly net savings
    bars1 = ax1.bar(
        x,
        net_save,
        width=0.55,
        color=C_PRIMARY,
        edgecolor="#1B4F72",
        lw=0.8,
        zorder=3,
        label="月度净节费 (万元)",
    )

    # Zero baseline
    ax1.axhline(0, color=C_ZERO, lw=0.9, zorder=2)

    # Secondary axis for cumulative savings
    ax1_twin = ax1.twinx()
    ax1_twin.spines["top"].set_visible(False)
    ax1_twin.spines["right"].set_color("#78909C")
    ax1_twin.spines["left"].set_visible(False)
    line_acc = ax1_twin.plot(
        x,
        accum_save,
        color=C_ACCUM,
        marker="o",
        markersize=4.5,
        lw=1.8,
        ls="--",
        zorder=5,
        label="累计净节费 (右轴)",
    )[0]
    ax1_twin.set_ylabel("累计净节费 (万元)", color=C_ACCUM, fontsize=9.5)
    ax1_twin.tick_params(axis="y", labelcolor=C_ACCUM)
    ax1_twin.set_ylim(0, 45)

    # Annotate ONLY key nodes (peaks, valley, and final cumulative total)
    # 1. June summer peak
    ax1.annotate(
        "峰值: +11.89 万元",
        xy=(4, net_save[4]),
        xytext=(4, net_save[4] + 0.9),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#1B4F72",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=0.8),
        zorder=6,
    )
    # 2. December winter peak
    ax1.annotate(
        "次峰: +6.59 万元",
        xy=(10, net_save[10]),
        xytext=(10 - 0.2, net_save[10] + 1.0),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#1B4F72",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=0.8),
        zorder=6,
    )
    # 3. November valley
    ax1.annotate(
        "谷值: +0.37 万元",
        xy=(9, net_save[9]),
        xytext=(9 - 0.3, net_save[9] + 1.5),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#566573",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#566573", lw=0.8),
        zorder=6,
    )
    # 4. Total cumulative milestone
    ax1_twin.scatter([x[-1]], [accum_save[-1]], color=C_ACCUM, s=45, zorder=6)
    ax1_twin.text(
        x[-1] - 0.15,
        accum_save[-1] + 1.3,
        "全年累计: 37.51 万元",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=C_ACCUM,
        fontweight="bold",
        zorder=6,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels, fontsize=9.0)
    ax1.set_xlabel("时间 (2025年月份)", fontsize=9.5)
    ax1.set_ylabel("月度净节费 (万元)", fontsize=9.5)
    ax1.set_ylim(-0.5, 14.5)
    ax1.grid(axis="y", ls=":", lw=0.6, alpha=0.55, zorder=1)

    # Legend outside axes at top (under subplot title, completely above plot area)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    leg1 = ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )

    # =========================================================================
    # Panel (b): 逐月节费来源三项拆解 (普通购电 / 应急费 / 调整费)
    # =========================================================================
    ax2.set_title(
        "(b) 节费来源三项拆解 (购电节省 / 应急节省 / 调整代价)",
        fontsize=10.5,
        pad=34,
        fontweight="bold",
    )
    width2 = 0.55

    # Stacked bars
    b_ord = ax2.bar(
        x,
        ord_save,
        width=width2,
        color=C_NAVY,
        edgecolor="none",
        zorder=3,
        label="普通购电节省",
    )
    b_emg = ax2.bar(
        x,
        emg_save,
        bottom=ord_save,
        width=width2,
        color=C_EMERALD,
        edgecolor="none",
        zorder=3,
        label="应急补电节省",
    )
    b_adj = ax2.bar(
        x,
        -adj_cost,
        width=width2,
        color=C_CORAL,
        edgecolor="none",
        zorder=3,
        label="滚动调整代价",
    )

    # Zero baseline separating positive contributions from negative drag
    ax2.axhline(0, color=C_ZERO, lw=0.9, zorder=2)

    # Net savings curve overlay
    line_net = ax2.plot(
        x,
        net_save,
        color=C_LINE,
        marker="D",
        markersize=4.0,
        lw=1.6,
        ls="-",
        zorder=5,
        label="月度净节费",
    )[0]

    # Key node annotation on Panel (b): Peak net savings
    ax2.annotate(
        "最大净省 (+11.89 万元)",
        xy=(4, net_save[4]),
        xytext=(4, net_save[4] + 1.4),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=C_LINE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LINE, lw=0.8),
        zorder=6,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(month_labels, fontsize=9.0)
    ax2.set_xlabel("时间 (2025年月份)", fontsize=9.5)
    ax2.set_ylabel("分项费用差额贡献 (万元)", fontsize=9.5)
    ax2.set_ylim(-8.5, 19.5)
    ax2.grid(axis="y", ls=":", lw=0.6, alpha=0.55, zorder=1)

    # Legend outside axes at top (under subplot title, completely above plot area)
    leg2 = ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=8.3,
    )

    save_fig(fig, "q4_fig1_monthly_savings_and_decomposition")
    plt.close(fig)
    print("Question 4 monthly savings figure successfully regenerated.")


if __name__ == "__main__":
    plot_fig_monthly_savings()
