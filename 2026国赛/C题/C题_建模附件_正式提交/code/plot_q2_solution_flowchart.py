"""Draw a publication-style workflow schematic for Question 2.

The Mermaid document remains the editable conceptual source; this script creates
the static paper figure with the Python/matplotlib backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "q2_time_mapping" / "q2_solution_flowchart"
OUT.parent.mkdir(parents=True, exist_ok=True)
FIG_WIDTH_IN = 7.2047  # 183 mm, common double-column width.
FIG_HEIGHT_IN = 5.0394  # 128 mm.

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["LXGW Neo XiHei Screen", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


BLUE = "#DCEAF7"
BLUE_EDGE = "#2B6CA3"
PURPLE = "#EEE6F7"
PURPLE_EDGE = "#76529A"
AMBER = "#FFF2CC"
AMBER_EDGE = "#B07A00"
RED = "#FBE4E6"
RED_EDGE = "#B8323B"
GREEN = "#E2F0D9"
GREEN_EDGE = "#4E7D3A"
INK = "#25313B"
MUTED = "#5A6872"


def rounded_box(ax, xy, width, height, text, face, edge, *, fontsize=8.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        facecolor=face,
        edgecolor=edge,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        linespacing=1.25,
        transform=ax.transAxes,
    )
    return (x, y, width, height)


def arrow(ax, start, end, *, color=INK, lw=1.0, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=6,
            shrinkB=6,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def decision(ax, center, width, height, text):
    x, y = center
    verts = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=AMBER,
            edgecolor=AMBER_EDGE,
            linewidth=1.15,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=8.0, color=INK, transform=ax.transAxes)


def panel_a(ax):
    ax.set_axis_off()
    ax.text(-0.025, 1.04, "a", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)
    ax.text(0.0, 1.04, "全年滚动调度主流程", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK)

    nodes = {
        "data": rounded_box(ax, (0.02, 0.69), 0.15, 0.16, "历史负载、光伏\n与已知电价", PURPLE, PURPLE_EDGE),
        "forecast": rounded_box(ax, (0.22, 0.69), 0.15, 0.16, "0:00 可用信息集\n形成日前预测", BLUE, BLUE_EDGE),
        "protect": rounded_box(ax, (0.42, 0.69), 0.16, 0.16, "28 日残差\n经验 q80 保护", BLUE, BLUE_EDGE),
        "milp": rounded_box(ax, (0.63, 0.69), 0.15, 0.16, "日前 MILP\n最小化计划费", BLUE, BLUE_EDGE),
        "freeze": rounded_box(ax, (0.83, 0.69), 0.15, 0.16, "冻结 144 段\n计划购电量", GREEN, GREEN_EDGE),
        "execute": rounded_box(ax, (0.80, 0.30), 0.18, 0.16, "真实供需逐段揭示\n储能反馈执行", BLUE, BLUE_EDGE),
        "settle": rounded_box(ax, (0.55, 0.30), 0.18, 0.16, "记录充放电、SOC\n与应急费用", GREEN, GREEN_EDGE),
        "update": rounded_box(ax, (0.30, 0.30), 0.18, 0.16, "传递日末 SOC\n更新历史残差", PURPLE, PURPLE_EDGE),
    }

    # Horizontal planning chain.
    top = ["data", "forecast", "protect", "milp", "freeze"]
    for a, b in zip(top[:-1], top[1:]):
        x1, y1, w1, h1 = nodes[a]
        x2, y2, _, h2 = nodes[b]
        arrow(ax, (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2))

    # The second row runs right-to-left so the daily loop stays outside nodes.
    x, y, w, h = nodes["freeze"]
    x2, y2, w2, h2 = nodes["execute"]
    arrow(ax, (x + w / 2, y), (x2 + w2 / 2, y2 + h2))
    arrow(ax, (nodes["execute"][0], 0.38), (nodes["settle"][0] + nodes["settle"][2], 0.38))
    arrow(ax, (nodes["settle"][0], 0.38), (nodes["update"][0] + nodes["update"][2], 0.38))
    arrow(ax, (nodes["update"][0], 0.38), (0.20, 0.38))
    arrow(ax, (0.20, 0.38), (nodes["forecast"][0] + nodes["forecast"][2] / 2, nodes["forecast"][1]), rad=-0.18)
    ax.text(0.695, 0.49, "若电池仍不足：按 5p 应急购电", fontsize=7.1, color=RED_EDGE, ha="center", transform=ax.transAxes)
    ax.text(0.18, 0.29, "下一日", fontsize=7.0, color=PURPLE_EDGE, transform=ax.transAxes)

    ax.text(
        0.02,
        0.08,
        "关键边界：当天计划在 0:00 冻结；真实 SOC 跨日继承；名义规划量与实际充放电量分开记录。",
        fontsize=7.5,
        color=MUTED,
        transform=ax.transAxes,
    )


def panel_b(ax):
    ax.set_axis_off()
    ax.text(-0.025, 1.04, "b", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)
    ax.text(0.0, 1.04, "单个 10 分钟时段的实际反馈", transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK)

    rounded_box(ax, (0.03, 0.38), 0.17, 0.26, "计划购电 q\n真实净需求 n\n当前 SOC E", PURPLE, PURPLE_EDGE)
    decision(ax, (0.33, 0.51), 0.19, 0.22, "s=q−n\n≥ 0?" )
    rounded_box(ax, (0.43, 0.66), 0.22, 0.18, "富余：按功率、容量\n和效率充电；余量记 w", BLUE, BLUE_EDGE)
    rounded_box(ax, (0.43, 0.20), 0.22, 0.18, "不足：按库存和\n功率上限放电", BLUE, BLUE_EDGE)
    decision(ax, (0.74, 0.29), 0.18, 0.25, "电池仍\n不足?" )
    rounded_box(ax, (0.81, 0.07), 0.16, 0.14, "是：应急购电\n价格 5p", RED, RED_EDGE, fontsize=7.4)
    rounded_box(ax, (0.79, 0.57), 0.18, 0.16, "更新 E(t+1)\n进入下一时段", PURPLE, PURPLE_EDGE, fontsize=7.4)

    arrow(ax, (0.20, 0.51), (0.235, 0.51))
    arrow(ax, (0.425, 0.60), (0.43, 0.70))
    arrow(ax, (0.425, 0.42), (0.43, 0.29))
    arrow(ax, (0.65, 0.75), (0.79, 0.65))
    arrow(ax, (0.65, 0.29), (0.67, 0.29))
    arrow(ax, (0.72, 0.195), (0.81, 0.14), color=RED_EDGE)
    arrow(ax, (0.805, 0.34), (0.82, 0.57), color=GREEN_EDGE)
    arrow(ax, (0.89, 0.21), (0.89, 0.57), color=RED_EDGE)
    ax.text(
        0.03,
        -0.06,
        "能量平衡：q + d + e = n + c + w；状态递推：E(t+1)=E(t)+0.9c−d/0.9。",
        fontsize=7.5,
        color=MUTED,
        transform=ax.transAxes,
    )


def main():
    fig, axes = plt.subplots(2, 1, figsize=(7.2047, 5.0394), gridspec_kw={"height_ratios": [1.15, 0.85]})
    fig.patch.set_facecolor("white")
    panel_a(axes[0])
    panel_b(axes[1])
    fig.subplots_adjust(left=0.035, right=0.99, top=0.94, bottom=0.08, hspace=0.34)

    require_matplotlib_panel_alignment(
        fig,
        json_out=f"{OUT}.alignment.json",
        overlay_svg=f"{OUT}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(f"{OUT}.svg", facecolor="white")
    fig.savefig(f"{OUT}.pdf", facecolor="white")
    fig.savefig(f"{OUT}.png", dpi=600, facecolor="white")
    fig.savefig(f"{OUT}.tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
