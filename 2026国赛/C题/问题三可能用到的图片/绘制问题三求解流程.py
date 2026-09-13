from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "问题三求解流程"
STEM = "问题三求解流程"

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


# Low-saturation functional palette. Red is reserved for deficit/rejection.
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
    lw: float = 1.15,
    fontsize: float = 6.8,
    weight: str = "normal",
    radius: float = 0.7,
    color: str = INK,
    linespacing: float = 1.28,
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
    lw: float = 1.1,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    dashed: bool = False,
    zorder: int = 2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "-",
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def polyline_arrow(ax, points, *, color=BUS, lw=1.1, dashed=False, zorder=2):
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
    ax.text(
        3.0,
        cy + 1.4,
        number,
        ha="left",
        va="center",
        color=accent,
        fontsize=7.0,
        fontweight="bold",
    )
    ax.text(
        3.0,
        cy - 1.2,
        label,
        ha="left",
        va="center",
        color=INK,
        fontsize=7.7,
        fontweight="bold",
    )
    return cy


def diamond(ax, cx, cy, w, h, text, *, edge=GREEN, face=GREEN_FILL, fontsize=6.5):
    patch = Polygon(
        [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
        closed=True,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
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
        linespacing=1.2,
        rotation=0,
        rotation_mode="anchor",
        zorder=4,
    )
    return patch


def model_box(ax, x, y, w, h):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.22,rounding_size=0.75",
        linewidth=1.3,
        edgecolor=PURPLE,
        facecolor=PURPLE_FILL,
        zorder=3,
    )
    patch.set_path_effects(SOFT_SHADOW)
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 1.6,
        "0:00 原始计划 MILP（144 段）",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color=PURPLE,
        zorder=4,
    )
    ax.plot([x + 2.2, x + w - 2.2], [y + h - 3.0, y + h - 3.0], color="#D1CBD4", lw=0.75)
    ax.text(x + 2.8, y + 7.1, "决策变量", ha="left", va="center", fontsize=5.9, fontweight="bold", color=MUTED)
    ax.text(x + 12.0, y + 7.1, "q0, c_bar, d_bar, w_bar, E_bar, z", ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(x + 2.8, y + 4.5, "目标函数", ha="left", va="center", fontsize=5.9, fontweight="bold", color=PURPLE)
    ax.text(x + 12.0, y + 4.5, "min C0 = Σ p(t)q0(t)", ha="left", va="center", fontsize=7.2, fontweight="bold", color=PURPLE)
    ax.text(x + 2.8, y + 1.9, "约束条件", ha="left", va="center", fontsize=5.9, fontweight="bold", color=MUTED)
    ax.text(
        x + 12.0,
        y + 1.9,
        "供需平衡 · SOC递推 · 充放电功率/容量边界 · 互斥 · 自由末态",
        ha="left",
        va="center",
        fontsize=5.9,
        color=INK,
    )
    return patch


def draw_figure():
    fig = plt.figure(figsize=(7.09, 9.45))
    ax = fig.add_axes([0.025, 0.018, 0.95, 0.965])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.axis("off")

    ax.text(
        50,
        126.8,
        "问题三求解流程",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=INK,
    )

    stage_1 = stage_region(ax, 111.5, 11.5, "01", "数据与时间映射", BLUE)
    stage_2 = stage_region(ax, 91.2, 18.3, "02", "四节点预测", PURPLE)
    stage_3 = stage_region(ax, 74.0, 15.3, "03", "原始计划", PURPLE)
    stage_4 = stage_region(ax, 43.5, 28.5, "04", "滚动调整", OCHRE)
    stage_5 = stage_region(ax, 23.5, 18.0, "05", "实际反馈", GREEN)
    stage_6 = stage_region(ax, 2.0, 19.5, "06", "评价与输出", OCHRE)

    arrow(ax, (17.0, stage_1 + 2.0), (17.0, stage_6 - 2.0), color=BUS, lw=1.75)

    # 01 Data sources and the one unambiguous physical timeline.
    rounded_box(ax, 25.0, 114.0, 18.0, 6.2, "附件 1、2\n电价 · 负荷 · 实测", edge=BLUE, face=BLUE_FILL)
    rounded_box(ax, 49.0, 114.0, 18.0, 6.2, "附件 3\n四时点光伏预报", edge=BLUE, face=BLUE_FILL)
    rounded_box(
        ax,
        73.0,
        114.0,
        20.5,
        6.2,
        "真实时间轴映射\n10 min · 跨日衔接",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
    )
    arrow(ax, (21.5, stage_1), (24.6, stage_1), color=BLUE)
    arrow(ax, (43.4, 117.1), (48.6, 117.1), color=BLUE)
    arrow(ax, (67.4, 117.1), (72.6, 117.1), color=BLUE)

    # 02 Four publication-time forecasts and causal protection.
    rounded_box(
        ax,
        25.0,
        97.0,
        16.0,
        7.0,
        "信息集\n00:00 · 06:00\n12:00 · 18:00",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
    )
    rounded_box(
        ax,
        47.0,
        101.0,
        20.0,
        5.8,
        "F2 负载预测\n日内残差线性修正",
        edge=PURPLE,
        face=PURPLE_FILL,
        weight="bold",
    )
    rounded_box(
        ax,
        47.0,
        93.2,
        20.0,
        5.8,
        "Linear 光伏预测\n整点线性插值积分",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
    )
    rounded_box(
        ax,
        73.0,
        96.0,
        20.5,
        7.5,
        "W = 28 日 · q75 保护\n仅用已实现的同类发布误差\n保护净需求 n_pro(r,t)",
        edge=PURPLE,
        face=PURPLE_FILL,
        weight="bold",
        fontsize=6.4,
    )
    arrow(ax, (21.5, stage_2), (24.6, stage_2), color=PURPLE)
    arrow(ax, (41.4, 100.5), (46.6, 102.0), color=PURPLE)
    arrow(ax, (41.4, 97.0), (46.6, 95.0), color=BLUE)
    arrow(ax, (67.4, 102.0), (72.6, 99.5), color=PURPLE)
    arrow(ax, (67.4, 95.0), (72.6, 97.0), color=BLUE)

    # 03 Midnight original commitment.
    rounded_box(
        ax,
        24.5,
        77.0,
        17.0,
        8.5,
        "真实 SOC(00:00)\n+ 前日午夜承诺\n估算 E_bar(00:10)",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
        fontsize=6.3,
    )
    model_box(ax, 45.0, 76.0, 48.5, 11.2)
    arrow(ax, (21.5, stage_3), (24.1, stage_3), color=PURPLE)
    arrow(ax, (41.9, 81.2), (44.6, 81.2), color=PURPLE)
    arrow(ax, (82.5, 95.6), (82.5, 87.5), color=PURPLE)

    # 04 Receding-horizon revisions. Candidate generation and acceptance use distinct criteria.
    rounded_box(
        ax,
        24.5,
        63.0,
        17.0,
        6.3,
        "到达更新时刻 r\nr = 06:00 / 12:00 / 18:00",
        edge=OCHRE,
        face=OCHRE_FILL,
        weight="bold",
        fontsize=6.3,
    )
    rounded_box(
        ax,
        46.0,
        63.0,
        18.0,
        6.3,
        "冻结已执行区间\n读取最新预测与真实 SOC",
        edge=BLUE,
        face=BLUE_FILL,
        weight="bold",
        fontsize=6.2,
    )
    rounded_box(
        ax,
        69.0,
        61.8,
        23.0,
        8.7,
        "剩余时域 MILP\n109 / 73 / 37 段\nmin Σ[pq_r + 0.5pa]\na >= |q_r − q0|",
        edge=PURPLE,
        face=PURPLE_FILL,
        weight="bold",
        fontsize=6.1,
    )
    rounded_box(
        ax,
        25.0,
        48.0,
        25.0,
        8.2,
        "同口径预测费用比较\nC_hat_r(q) = Σ[pq + 0.5p|q−q0|\n+ 5p e_hat(q)]",
        edge=OCHRE,
        face=OCHRE_FILL,
        weight="bold",
        fontsize=6.2,
    )
    diamond(ax, 59.0, 52.1, 10.5, 10.0, "新候选\n更省？", edge=GREEN, face=GREEN_FILL, fontsize=6.3)
    rounded_box(ax, 70.0, 56.4, 13.5, 5.0, "接受 q_r\n覆盖未来计划", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.2)
    rounded_box(ax, 70.0, 45.4, 13.5, 5.0, "保留 q_old\n不作调整", edge=RED, face=RED_FILL, weight="bold", fontsize=6.2)
    rounded_box(ax, 86.0, 50.1, 9.0, 5.5, "形成\nq_eff", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.5)

    arrow(ax, (21.5, stage_4), (24.1, stage_4), color=OCHRE)
    arrow(ax, (41.9, 66.2), (45.6, 66.2), color=OCHRE)
    arrow(ax, (64.4, 66.2), (68.6, 66.2), color=PURPLE)
    polyline_arrow(ax, [(80.5, 61.4), (80.5, 58.1), (50.4, 54.0)], color=OCHRE)
    arrow(ax, (50.4, 52.1), (53.5, 52.1), color=OCHRE)
    arrow(ax, (64.4, 54.7), (69.6, 58.9), color=GREEN)
    ax.text(65.8, 57.4, "是", ha="center", va="center", fontsize=6.2, fontweight="bold", color=GREEN)
    arrow(ax, (64.4, 49.5), (69.6, 47.9), color=RED)
    ax.text(66.2, 47.1, "否", ha="center", va="center", fontsize=6.2, fontweight="bold", color=RED)
    arrow(ax, (83.9, 58.9), (87.2, 55.9), color=GREEN)
    arrow(ax, (83.9, 47.9), (85.6, 51.4), color=RED)

    # Dashed return means the same procedure is repeated only at the next legal update.
    polyline_arrow(
        ax,
        [(95.4, 52.9), (96.4, 52.9), (96.4, 70.6), (33.0, 70.6), (33.0, 69.7)],
        color=OCHRE,
        lw=0.95,
        dashed=True,
        zorder=1,
    )
    ax.text(83.0, 70.9, "下一合法更新节点", ha="center", va="bottom", fontsize=5.7, color=OCHRE)

    # 05 Physical execution uses realised net demand; it is not part of the deterministic MILP.
    rounded_box(ax, 24.5, 31.1, 15.5, 6.0, "执行当前 q_eff\n读取实际净需求 n(t)", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.2)
    diamond(ax, 48.0, 34.1, 10.5, 9.8, "q_eff−n(t)\n>= 0？", edge=GREEN, face=GREEN_FILL, fontsize=6.3)
    rounded_box(ax, 59.0, 35.8, 13.5, 5.7, "富余：充电\n其余记未使用", edge=GREEN, face=GREEN_FILL, fontsize=6.2)
    rounded_box(ax, 59.0, 29.0, 13.5, 5.2, "缺额：放电\n不足则应急购电", edge=RED, face=RED_FILL, fontsize=6.2)
    rounded_box(ax, 78.0, 31.2, 15.5, 6.4, "更新真实 SOC\n记录 c, d, e, w", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.3)
    rounded_box(ax, 47.0, 24.0, 28.0, 4.0, "真实 SOC 与午夜承诺跨日连续传递", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.1)
    arrow(ax, (21.5, stage_5), (24.1, stage_5), color=GREEN)
    arrow(ax, (40.4, 34.1), (42.5, 34.1), color=GREEN)
    arrow(ax, (53.4, 36.5), (58.6, 38.6), color=GREEN)
    ax.text(55.2, 39.0, "是", ha="center", va="center", fontsize=6.1, fontweight="bold", color=GREEN)
    arrow(ax, (53.4, 31.7), (58.6, 31.6), color=RED)
    ax.text(55.2, 30.8, "否", ha="center", va="center", fontsize=6.1, fontweight="bold", color=RED)
    arrow(ax, (72.9, 38.6), (77.6, 35.8), color=GREEN)
    arrow(ax, (72.9, 31.6), (77.6, 33.0), color=RED)
    arrow(ax, (85.8, 31.0), (74.6, 27.0), color=GREEN)
    arrow(ax, (90.5, 50.3), (85.8, 38.0), color=GREEN)

    # 06 Cash evaluation and the frozen S2 delivery choice.
    rounded_box(
        ax,
        24.5,
        14.0,
        69.0,
        5.2,
        "实际现金费用  C = Σ[pq_eff + 0.5p|q_eff−q0| + 5pe]   ·   自然日 / 模板日双账本核对",
        edge=OCHRE,
        face=OCHRE_FILL,
        weight="bold",
        fontsize=6.5,
    )
    rounded_box(ax, 24.5, 4.2, 20.0, 6.3, "S0 / L75\n13,353,822.04 元", edge=OCHRE, face=OCHRE_FILL, weight="bold", fontsize=6.6)
    rounded_box(ax, 50.0, 4.2, 20.0, 6.3, "S2（最终采用）\n13,330,377.96 元", edge=PURPLE, face=PURPLE_FILL, weight="bold", fontsize=6.6)
    rounded_box(ax, 75.5, 4.2, 18.0, 6.3, "节省 23,444.08 元\n填入 result3.xlsx", edge=GREEN, face=GREEN_FILL, weight="bold", fontsize=6.4)
    arrow(ax, (21.5, stage_6), (24.1, stage_6), color=OCHRE)
    arrow(ax, (41.0, 13.6), (34.5, 10.8), color=OCHRE)
    arrow(ax, (58.8, 13.6), (60.0, 10.8), color=PURPLE)
    arrow(ax, (70.5, 7.4), (75.1, 7.4), color=GREEN)

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
