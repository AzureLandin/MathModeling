from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
MIRROR_DIR = Path(__file__).resolve().parent
STEM = "问题一MILP建模求解与基线对照流程图"

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


# Restrained report palette: graphite structure with muted functional accents.
INK = "#263238"
MUTED = "#68747C"
GRID = "#BCC4C8"
BUS = "#46545D"
BLUE = "#496C7A"
BLUE_FILL = "#F0F4F5"
PURPLE = "#625F70"
PURPLE_FILL = "#F3F2F4"
GREEN = "#557866"
GREEN_FILL = "#EFF4F1"
OCHRE = "#8A7047"
OCHRE_FILL = "#F6F2EA"
RED = "#98574F"

SOFT_SHADOW = [
    pe.SimplePatchShadow(
        offset=(1.1, -1.1),
        shadow_rgbFace=(0.20, 0.24, 0.26),
        alpha=0.14,
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
    lw: float = 1.2,
    fontsize: float = 7.2,
    weight: str = "normal",
    radius: float = 0.8,
    zorder: int = 3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.22,rounding_size={radius}",
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
        color=INK,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.35,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BUS,
    lw: float = 1.2,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    dashed: bool = False,
    zorder: int = 2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def stage_region(ax, y: float, h: float, number: str, label: str, accent: str):
    # Dashed region begins to the right of the stage bus, keeping the left rail clean.
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
    ax.plot([17.0, 21.5], [cy, cy], color=BUS, lw=1.4, zorder=1)
    ax.scatter([17.0], [cy], s=40, color=accent, edgecolor="white", linewidth=0.8, zorder=4)
    ax.text(
        3.0,
        cy + 1.4,
        number,
        ha="left",
        va="center",
        color=accent,
        fontsize=7.2,
        fontweight="bold",
        rotation=0,
        rotation_mode="anchor",
    )
    ax.text(
        3.0,
        cy - 1.2,
        label,
        ha="left",
        va="center",
        color=INK,
        fontsize=8.2,
        fontweight="bold",
        rotation=0,
        rotation_mode="anchor",
    )
    return cy


def draw_figure():
    fig = plt.figure(figsize=(7.09, 7.25))
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        50,
        96.5,
        "问题一求解流程",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=INK,
    )

    stage_1 = stage_region(ax, 77.5, 14.0, "01", "数据准备", BLUE)
    stage_2 = stage_region(ax, 52.0, 23.5, "02", "模型构建", PURPLE)
    stage_3 = stage_region(ax, 27.0, 23.0, "03", "求解校验", GREEN)
    stage_4 = stage_region(ax, 4.0, 21.0, "04", "结果输出", OCHRE)

    # Stage bus and downward direction.
    arrow(ax, (17.0, stage_1 + 2.3), (17.0, stage_4 - 2.0), color=BUS, lw=1.8)

    # 01 Data preparation.
    rounded_box(ax, 26.0, 81.0, 18.0, 7.0, "读取附件 1\n电价 · 负荷 · 光伏", edge=BLUE, face=BLUE_FILL)
    rounded_box(ax, 51.0, 81.0, 18.0, 7.0, "统一时间尺度\n144 时段 · 10 min", edge=BLUE, face=BLUE_FILL)
    rounded_box(ax, 76.0, 81.0, 17.0, 7.0, "生成模型输入\n功率换算为电量", edge=BLUE, face=BLUE_FILL)
    arrow(ax, (21.5, stage_1), (25.6, stage_1), color=BLUE)
    arrow(ax, (44.4, 84.5), (50.6, 84.5), color=BLUE)
    arrow(ax, (69.4, 84.5), (75.6, 84.5), color=BLUE)

    # 02 Parameterized deterministic MILP. The objective is displayed explicitly.
    model_x, model_y, model_w, model_h = 25.0, 55.0, 69.0, 17.0
    model_patch = FancyBboxPatch(
        (model_x, model_y),
        model_w,
        model_h,
        boxstyle="round,pad=0.28,rounding_size=0.8",
        linewidth=1.35,
        edgecolor=PURPLE,
        facecolor=PURPLE_FILL,
        zorder=2,
    )
    model_patch.set_path_effects(SOFT_SHADOW)
    ax.add_patch(model_patch)
    ax.text(59.5, 69.4, "参数化确定性 MILP", ha="center", va="center", fontsize=9.2, fontweight="bold", color=PURPLE)
    ax.plot([29.0, 90.0], [66.8, 66.8], color="#CFC4D8", lw=0.8)
    ax.text(29.0, 64.5, "决策变量", ha="left", va="center", fontsize=6.6, fontweight="bold", color=MUTED)
    ax.text(40.0, 64.5, "购电 q(t)；充电 P_ch(t)；放电 P_dis(t)；储能 E(t)；互斥变量 u(t)", ha="left", va="center", fontsize=6.8, color=INK)
    ax.text(29.0, 61.3, "目标函数", ha="left", va="center", fontsize=6.8, fontweight="bold", color=PURPLE)
    ax.text(47.0, 61.3, "min C = Σ p(t) q(t),  t = 1, …, 144", ha="left", va="center", fontsize=8.4, fontweight="bold", color=PURPLE)
    ax.text(29.0, 58.1, "约束条件", ha="left", va="center", fontsize=6.6, fontweight="bold", color=MUTED)
    ax.text(
        40.0,
        58.1,
        "供需平衡 · 状态递推 · 容量边界 · 功率边界 · 充放电互斥 · 初末状态相等",
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK,
    )
    arrow(ax, (21.5, stage_2), (24.6, stage_2), color=PURPLE)
    arrow(ax, (84.5, 80.7), (84.5, 72.4), color=PURPLE)

    # 03 Solve and verify. Baseline is a parallel reference, not part of the MILP.
    rounded_box(ax, 25.5, 42.1, 17.0, 5.2, "代入附件 1 参数", edge=BLUE, face=BLUE_FILL, fontsize=7.0, weight="bold")
    rounded_box(ax, 49.0, 42.1, 17.0, 5.2, "MILP · HiGHS 求解", edge=PURPLE, face=PURPLE_FILL, fontsize=7.0, weight="bold")
    rounded_box(ax, 72.5, 42.1, 16.0, 5.2, "约束与最优性核验", edge=GREEN, face=GREEN_FILL, fontsize=6.8, weight="bold")
    rounded_box(ax, 37.0, 31.0, 20.0, 5.2, "无储能 baseline\nP_ch(t) = P_dis(t) = 0", edge=OCHRE, face=OCHRE_FILL, fontsize=6.8, weight="bold")

    # Decision node is centered within the verification row rather than pushed to the page edge.
    diamond_center = (80.5, 34.5)
    diamond = Polygon(
        [(80.5, 39.1), (85.5, 34.5), (80.5, 29.9), (75.5, 34.5)],
        closed=True,
        facecolor=GREEN_FILL,
        edgecolor=GREEN,
        linewidth=1.25,
        zorder=3,
    )
    diamond.set_path_effects(SOFT_SHADOW)
    ax.add_patch(diamond)
    ax.text(*diamond_center, "通过？", ha="center", va="center", fontsize=7.0, fontweight="bold", color=GREEN, rotation=0, rotation_mode="anchor", zorder=4)

    arrow(ax, (21.5, stage_3), (25.1, stage_3), color=GREEN)
    arrow(ax, (42.9, 44.7), (48.6, 44.7), color=PURPLE)
    arrow(ax, (66.4, 44.7), (72.1, 44.7), color=GREEN)
    arrow(ax, (80.5, 42.1), (80.5, 39.2), color=GREEN)
    arrow(ax, (34.0, 41.8), (42.3, 36.4), color=OCHRE)

    # Rejected solutions return to model construction. Both labels remain horizontal.
    # Rejection path exits to the right, folds upward, then points left into the model.
    arrow(ax, (85.7, 34.5), (96.0, 34.5), color=RED, lw=1.15, style="-", dashed=True)
    arrow(ax, (96.0, 34.5), (96.0, 62.8), color=RED, lw=1.15, style="-", dashed=True)
    arrow(ax, (96.0, 62.8), (93.8, 62.8), color=RED, lw=1.15, dashed=True)
    ax.text(87.2, 37.4, "否", ha="left", va="center", fontsize=6.8, fontweight="bold", color=RED, rotation=0, rotation_mode="anchor")
    arrow(ax, (80.5, 29.6), (70.0, 22.7), color=GREEN)
    ax.text(84.0, 28.5, "是", ha="left", va="center", fontsize=6.8, fontweight="bold", color=GREEN, rotation=0, rotation_mode="anchor")

    # 04 Comparable economic outputs.
    rounded_box(ax, 57.5, 18.0, 25.0, 4.7, "结果对照", edge=GREEN, face=GREEN_FILL, fontsize=7.4, weight="bold")
    rounded_box(ax, 25.5, 8.0, 20.0, 6.4, "优化前总购电费用\n48,052.046591 元", edge=OCHRE, face=OCHRE_FILL, fontsize=7.0, weight="bold")
    rounded_box(ax, 51.0, 8.0, 20.0, 6.4, "优化后总购电费用\n35,126.948589 元", edge=PURPLE, face=PURPLE_FILL, fontsize=7.0, weight="bold")
    rounded_box(ax, 76.5, 8.0, 17.0, 6.4, "优化费用\n12,925.098002 元\n（26.8981%）", edge=GREEN, face=GREEN_FILL, fontsize=6.8, weight="bold")
    arrow(ax, (70.0, 17.7), (35.5, 14.7), color=OCHRE, connectionstyle="angle3,angleA=-90,angleB=180")
    arrow(ax, (70.0, 17.7), (61.0, 14.7), color=PURPLE, connectionstyle="angle3,angleA=-90,angleB=180")
    arrow(ax, (70.0, 17.7), (85.0, 14.7), color=GREEN, connectionstyle="angle3,angleA=-90,angleB=0")
    arrow(ax, (21.5, stage_4), (25.1, stage_4), color=OCHRE)

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
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # Keep the earlier figure-assets folder synchronized with the delivery bundle.
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        shutil.copy2(str(base) + suffix, MIRROR_DIR / f"{STEM}{suffix}")


if __name__ == "__main__":
    draw_figure()
