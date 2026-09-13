from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "问题四求解流程"
STEM = "问题四求解流程"

SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


FONT_FAMILY = "Microsoft YaHei"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "SimHei", "Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# Restrained functional palette. Red is reserved for rejection or supply deficit.
INK = "#273238"
MUTED = "#68747A"
GRID = "#B9C2C6"
BUS = "#4B5961"
BLUE = "#4C7180"
BLUE_FILL = "#EFF4F5"
PURPLE = "#686274"
PURPLE_FILL = "#F3F1F4"
GREEN = "#557866"
GREEN_FILL = "#EEF4F0"
OCHRE = "#8B7148"
OCHRE_FILL = "#F6F2EA"
RED = "#98574F"
RED_FILL = "#F7F0EE"

SOFT_SHADOW = [
    pe.SimplePatchShadow(
        offset=(1.0, -1.0),
        shadow_rgbFace=(0.20, 0.24, 0.26),
        alpha=0.13,
        rho=0.98,
    ),
    pe.Normal(),
]


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    edge: str = BUS,
    face: str = "white",
    lw: float = 1.1,
    fontsize: float = 6.6,
    weight: str = "normal",
    radius: float = 0.65,
    color: str = INK,
    linespacing: float = 1.25,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    patch.set_path_effects(SOFT_SHADOW)
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=linespacing,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BUS,
    lw: float = 1.05,
    style: str = "-|>",
    dashed: bool = False,
    zorder: int = 2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8.2,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "-",
        connectionstyle="arc3",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def polyline_arrow(ax, points, *, color=BUS, lw=1.05, dashed=False, zorder=2):
    for start, end in zip(points[:-2], points[1:-1]):
        arrow(
            ax,
            start,
            end,
            color=color,
            lw=lw,
            style="-",
            dashed=dashed,
            zorder=zorder,
        )
    arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        lw=lw,
        dashed=dashed,
        zorder=zorder,
    )


def stage_region(ax, y: float, h: float, number: str, label: str, accent: str):
    ax.add_patch(
        Rectangle(
            (21.5, y),
            76.0,
            h,
            linewidth=0.9,
            edgecolor=GRID,
            facecolor="none",
            linestyle=(0, (4, 3)),
            zorder=0,
        )
    )
    cy = y + h / 2
    ax.plot([17.0, 21.5], [cy, cy], color=BUS, lw=1.35, zorder=1)
    ax.scatter([17.0], [cy], s=38, color=accent, edgecolor="white", linewidth=0.8, zorder=4)
    ax.text(3.0, cy + 1.4, number, ha="left", va="center", color=accent, fontsize=7.0, fontweight="bold")
    ax.text(3.0, cy - 1.2, label, ha="left", va="center", color=INK, fontsize=7.5, fontweight="bold")
    return cy


def diamond(ax, cx, cy, w, h, text, *, edge=GREEN, face=GREEN_FILL, fontsize=6.1):
    patch = Polygon(
        [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.15,
        zorder=3,
    )
    patch.set_path_effects(SOFT_SHADOW)
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        color=edge,
        fontsize=fontsize,
        fontweight="bold",
        linespacing=1.16,
        rotation=0,
        rotation_mode="anchor",
        zorder=4,
    )
    return patch


def branch_header(ax, x, y, w, text, accent):
    patch = FancyBboxPatch(
        (x, y),
        w,
        4.1,
        boxstyle="round,pad=0.12,rounding_size=0.55",
        linewidth=0,
        facecolor=accent,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + 2.05, text, ha="center", va="center", color="white", fontsize=7.0, fontweight="bold", zorder=4)


def rolling_timeline(ax):
    xs = [62.0, 71.2, 80.4, 89.6]
    labels = ["0:00", "6:00", "12:00", "18:00"]
    sublabels = ["原计划", "109段", "73段", "37段"]
    ax.plot([xs[0], xs[-1]], [96.0, 96.0], color=OCHRE, lw=1.35, zorder=1)
    for x, label, sublabel in zip(xs, labels, sublabels):
        circle = Circle((x, 96.0), radius=1.25, facecolor="white", edgecolor=OCHRE, linewidth=1.2, zorder=3)
        circle.set_path_effects(SOFT_SHADOW)
        ax.add_patch(circle)
        ax.text(x, 98.0, label, ha="center", va="bottom", fontsize=5.9, color=INK, fontweight="bold")
        ax.text(x, 93.9, sublabel, ha="center", va="top", fontsize=5.5, color=OCHRE)
    for start, end in zip(xs[:-1], xs[1:]):
        arrow(ax, (start + 1.5, 96.0), (end - 1.5, 96.0), color=OCHRE, lw=1.0)


def core_model_box(ax):
    x, y, w, h = 43.0, 105.0, 50.5, 8.2
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.20,rounding_size=0.70",
        linewidth=1.25,
        edgecolor=PURPLE,
        facecolor=PURPLE_FILL,
        zorder=3,
    )
    patch.set_path_effects(SOFT_SHADOW)
    ax.add_patch(patch)
    ax.text(x + 1.8, y + h - 1.7, "统一储能 MILP", ha="left", va="center", fontsize=7.4, fontweight="bold", color=PURPLE)
    ax.plot([x + 1.8, x + w - 1.8], [y + h - 2.8, y + h - 2.8], color="#D1CBD4", lw=0.75)
    ax.text(
        x + 2.0,
        y + 3.5,
        "决策：q, c_bar, d_bar, w_bar, E_bar, z",
        ha="left",
        va="center",
        fontsize=5.9,
        color=INK,
    )
    ax.text(
        x + 2.0,
        y + 1.35,
        "约束：供需平衡 · SOC递推 · 1200—10800 kWh · 功率边界 · 充放电互斥 · 自由末态",
        ha="left",
        va="center",
        fontsize=5.55,
        color=MUTED,
    )
    return patch


def draw_figure():
    fig = plt.figure(figsize=(7.09, 10.45))
    ax = fig.add_axes([0.025, 0.016, 0.95, 0.972])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 149)
    ax.axis("off")

    ax.text(50, 146.0, "问题四求解流程", ha="center", va="center", fontsize=14.0, fontweight="bold", color=INK)

    stage_1 = stage_region(ax, 133.0, 10.0, "01", "数据与价格映射", BLUE)
    stage_2 = stage_region(ax, 118.0, 13.0, "02", "公共初始化", BLUE)
    stage_3 = stage_region(ax, 78.0, 38.0, "03", "两支重优化", PURPLE)
    stage_4 = stage_region(ax, 58.0, 18.0, "04", "实际反馈", GREEN)
    stage_5 = stage_region(ax, 32.0, 24.0, "05", "同口径评价", OCHRE)
    stage_6 = stage_region(ax, 2.0, 28.0, "06", "方案与交付", GREEN)

    arrow(ax, (17.0, stage_1 + 2.0), (17.0, stage_6 - 2.0), color=BUS, lw=1.75)

    # 01 Inputs and the exact delivery-price clock.
    rounded_box(ax, 24.5, 135.0, 18.0, 6.0, "附件2与冻结档案\n负荷 · 光伏 · 原策略", edge=BLUE, face=BLUE_FILL, weight="bold")
    rounded_box(ax, 47.0, 135.0, 18.0, 6.0, "附件4波动电价\n144段价格已知", edge=OCHRE, face=OCHRE_FILL, weight="bold")
    rounded_box(ax, 69.5, 135.0, 24.0, 6.0, "统一10 min时间轴\n午夜价取前一源行末段", edge=BLUE, face=BLUE_FILL, weight="bold", fontsize=6.3)
    arrow(ax, (21.5, stage_1), (24.1, stage_1), color=BLUE)
    arrow(ax, (42.9, 138.0), (46.6, 138.0), color=BLUE)
    arrow(ax, (65.4, 138.0), (69.1, 138.0), color=OCHRE)
    ax.text(56.0, 133.7, "价格不预测、不平滑", ha="center", va="center", fontsize=5.7, color=OCHRE, fontweight="bold")

    # 02 One common January trajectory makes the six strategies comparable.
    rounded_box(ax, 24.5, 121.0, 20.0, 7.0, "公共一月初始化\n单条共享物理轨迹", edge=BLUE, face=BLUE_FILL, weight="bold")
    rounded_box(ax, 49.0, 121.0, 20.0, 7.0, "正式起点 2025-02-01\nSOC = 6097.354410 kWh", edge=BLUE, face=BLUE_FILL, weight="bold", fontsize=6.15)
    rounded_box(ax, 73.5, 121.0, 20.0, 7.0, "正式评价至12-31\n48096个自然日区间", edge=BLUE, face=BLUE_FILL, weight="bold", fontsize=6.15)
    arrow(ax, (21.5, stage_2), (24.1, stage_2), color=BLUE)
    arrow(ax, (44.9, 124.5), (48.6, 124.5), color=BLUE)
    arrow(ax, (69.4, 124.5), (73.1, 124.5), color=BLUE)

    # 03 Shared robust net demand and two information-permission branches.
    rounded_box(
        ax,
        24.5,
        105.0,
        15.0,
        8.2,
        "保护净需求 n_pro(t)\n= [L_hat(t)−V_hat(t)]dt\n+ rho(t)，rho为残差分位数",
        edge=PURPLE,
        face=PURPLE_FILL,
        weight="bold",
        fontsize=5.35,
    )
    core_model_box(ax)
    arrow(ax, (21.5, 109.1), (24.1, 109.1), color=PURPLE)
    arrow(ax, (39.9, 109.1), (42.6, 109.1), color=PURPLE)

    branch_header(ax, 24.5, 98.0, 30.0, "4-2  |  Q42：每日一次优化", BLUE)
    branch_header(ax, 58.5, 98.0, 35.0, "4-3  |  Q43：日内滚动优化", OCHRE)

    rounded_box(
        ax,
        25.0,
        88.0,
        29.0,
        7.2,
        "0:00读取问题二预测 + W28/q80\n144段 MILP：min C0 = sum[p(t)q0(t)]",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
        fontsize=5.95,
    )
    rounded_box(
        ax,
        25.0,
        80.2,
        29.0,
        5.2,
        "Q42：日内固定 q_eff = q0\nF42：冻结旧购电量作回放对照",
        edge=BLUE,
        face="white",
        fontsize=5.9,
    )
    arrow(ax, (39.5, 97.6), (39.5, 95.6), color=BLUE)
    arrow(ax, (39.5, 87.6), (39.5, 85.8), color=BLUE)

    rolling_timeline(ax)
    rounded_box(
        ax,
        59.0,
        86.0,
        34.0,
        6.3,
        "每个节点：冻结已执行段 → 更新预测与真实SOC\n剩余时域 MILP：min Cr = sum[p(t)(q_r(t)+0.5a(t))]",
        edge=OCHRE,
        face=OCHRE_FILL,
        weight="bold",
        fontsize=5.65,
    )
    rounded_box(
        ax,
        59.0,
        79.2,
        22.0,
        4.3,
        "同一价格、保护和SOC下评分\n新费用低 0.0001 元以上？",
        edge=OCHRE,
        face="white",
        fontsize=5.45,
    )
    rounded_box(ax, 84.0, 82.0, 9.0, 3.7, "接受 q_r", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=5.7)
    rounded_box(ax, 84.0, 77.6, 9.0, 3.7, "保留 q_old", edge=RED, face=RED_FILL, weight="bold", fontsize=5.55)
    arrow(ax, (76.0, 95.6), (76.0, 92.7), color=OCHRE)
    arrow(ax, (76.0, 85.6), (76.0, 83.9), color=OCHRE)
    arrow(ax, (81.4, 81.3), (83.6, 83.8), color=GREEN)
    ax.text(82.0, 84.8, "是", ha="center", va="center", fontsize=5.7, color=GREEN, fontweight="bold")
    arrow(ax, (81.4, 80.1), (83.6, 79.5), color=RED)
    ax.text(82.3, 78.8, "否", ha="center", va="center", fontsize=5.7, color=RED, fontweight="bold")
    ax.text(60.0, 76.8, "S0：原0点负载 + Linear + q75   |   S2：F2修正 + Linear + 自身q75", ha="left", va="center", fontsize=5.5, color=MUTED)

    # 04 Realised operation is evaluated outside the deterministic nominal MILP.
    rounded_box(ax, 24.5, 64.0, 17.0, 6.0, "执行 q_eff\n读取实际净需求 n(t)", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.1)
    diamond(ax, 49.0, 67.0, 10.0, 9.0, "q_eff−n(t)\n>= 0？", edge=GREEN, face=GREEN_FILL, fontsize=6.1)
    rounded_box(ax, 58.0, 68.5, 14.0, 5.0, "富余：充电\n其余记未使用", edge=GREEN, face=GREEN_FILL, fontsize=5.9)
    rounded_box(ax, 58.0, 60.5, 14.0, 5.0, "缺额：放电\n不足则应急购电", edge=RED, face=RED_FILL, fontsize=5.9)
    rounded_box(ax, 78.0, 63.8, 15.5, 6.4, "更新真实SOC\n记录 c、d、e、w", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.15)
    rounded_box(ax, 32.0, 58.9, 20.0, 3.4, "SOC与午夜承诺跨日连续传递", edge=GREEN, face="white", fontsize=5.55)
    arrow(ax, (21.5, stage_4), (24.1, stage_4), color=GREEN)
    arrow(ax, (41.9, 67.0), (43.6, 67.0), color=GREEN)
    arrow(ax, (54.4, 69.2), (57.6, 70.5), color=GREEN)
    ax.text(55.8, 71.5, "是", ha="center", va="center", fontsize=5.8, color=GREEN, fontweight="bold")
    arrow(ax, (54.4, 64.8), (57.6, 63.0), color=RED)
    ax.text(55.8, 62.7, "否", ha="center", va="center", fontsize=5.8, color=RED, fontweight="bold")
    arrow(ax, (72.4, 71.0), (77.6, 68.2), color=GREEN)
    arrow(ax, (72.4, 63.0), (77.6, 65.6), color=RED)
    polyline_arrow(ax, [(85.8, 63.4), (85.8, 60.6), (52.4, 60.6)], color=GREEN, lw=0.9, dashed=True, zorder=1)

    # 05 Cash accounting and fixed-plan counterfactuals use the same realised state rules.
    rounded_box(
        ax,
        24.5,
        48.0,
        33.0,
        5.6,
        "4-2现金费用\nC42 = sum[p(t)(q0(t) + 5e(t))]",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
        fontsize=6.0,
    )
    rounded_box(
        ax,
        60.5,
        48.0,
        33.0,
        5.6,
        "4-3现金费用\nC43 = sum[p(t)(q_eff(t) + 0.5|q_eff(t)−q0(t)| + 5e(t))]",
        edge=OCHRE,
        face=OCHRE_FILL,
        weight="bold",
        fontsize=5.4,
    )
    rounded_box(ax, 24.5, 40.0, 20.5, 5.2, "F42  vs  Q42\n节省 115,595.68 元", edge=BLUE, face="white", weight="bold", fontsize=6.0)
    rounded_box(ax, 48.8, 40.0, 20.5, 5.2, "F43_S0  vs  Q43_S0\n节省 120,292.72 元", edge=OCHRE, face="white", weight="bold", fontsize=5.85)
    rounded_box(ax, 73.0, 40.0, 20.5, 5.2, "F43_S2  vs  Q43_S2\n节省 110,164.02 元", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=5.85)
    rounded_box(
        ax,
        24.5,
        33.8,
        69.0,
        3.7,
        "同步核对：普通购电费 · 调整费 · 应急费 · 应急电量 · 期末库存    |    费用下降不代表应急风险全面改善",
        edge=RED,
        face=RED_FILL,
        fontsize=5.7,
        weight="bold",
        color=RED,
    )
    arrow(ax, (21.5, stage_5), (24.1, stage_5), color=OCHRE)
    arrow(ax, (41.0, 47.6), (34.8, 45.6), color=BLUE)
    arrow(ax, (77.0, 47.6), (83.2, 45.6), color=OCHRE)

    # 06 Frozen final candidates and reproducible spreadsheet handoff.
    rounded_box(
        ax,
        24.5,
        20.5,
        20.5,
        6.5,
        "4-2最终方案：Q42\n总费用 14,347,907.89 元",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
        fontsize=6.25,
    )
    rounded_box(
        ax,
        48.5,
        20.5,
        21.5,
        6.5,
        "4-3最终方案：Q43_S2\n总费用 13,972,808.91 元",
        edge=GREEN,
        face=GREEN_FILL,
        weight="bold",
        fontsize=6.25,
    )
    rounded_box(
        ax,
        73.5,
        20.5,
        20.0,
        6.5,
        "Q43_S2 比 Q42 少\n375,098.98 元（2.6143%）",
        edge=GREEN,
        face="white",
        weight="bold",
        fontsize=6.0,
    )
    rounded_box(
        ax,
        24.5,
        11.6,
        69.0,
        5.8,
        "写入附件5副本：result4-2.xlsx / result4-3.xlsx  →  保存后重读  →  与CSV逐段核对",
        edge=OCHRE,
        face=OCHRE_FILL,
        weight="bold",
        fontsize=6.25,
    )
    rounded_box(
        ax,
        31.0,
        4.0,
        56.0,
        4.6,
        "费用容差 0.0001 元 · 能量容差 0.000001 kWh · 仅报告2025-02-01至12-31",
        edge=GREEN,
        face=GREEN_FILL,
        fontsize=5.9,
    )
    arrow(ax, (21.5, stage_6), (24.1, stage_6), color=GREEN)
    arrow(ax, (45.4, 23.8), (48.1, 23.8), color=GREEN)
    arrow(ax, (70.4, 23.8), (73.1, 23.8), color=GREEN)
    arrow(ax, (59.0, 20.1), (59.0, 17.8), color=OCHRE)
    arrow(ax, (59.0, 11.2), (59.0, 9.0), color=GREEN)

    fig.canvas.draw()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / STEM
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment-overlay.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )

    fig.savefig(str(base) + ".svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(str(base) + ".pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        str(base) + ".tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
