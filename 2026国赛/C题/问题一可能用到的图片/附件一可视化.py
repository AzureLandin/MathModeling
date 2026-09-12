"""附件一数据可视化绘制脚本 (简约出版版)。
包含图表：
1. 光伏与负载预测功率曲线 (小区负载 vs 光伏发电预测功率)
2. 分时电价折线图

设计规范：
- 符合国赛/美赛/研电建模论文顶刊规范
- 纯白背景、无多余杂色、学术内向刻度线
- 纯中文表头、纯中文图例 (绝无公式字母)
- 输出 300 DPI 超清 PNG 与无损矢量 SVG
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import font_manager

# -------------------------------------------------------------------------
# 1. 路径自动解析
# -------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output"
CSV_FILE = OUTPUT_DIR / "q1_intervals.csv"


# -------------------------------------------------------------------------
# 2. 学术样式配置
# -------------------------------------------------------------------------
def setup_academic_style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    cn_font = "Microsoft YaHei"
    for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if f in installed:
            cn_font = f
            break

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [cn_font, "SimHei", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#2D3748",
        "axes.linewidth": 0.85,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.pad": 4.5,
        "ytick.major.pad": 4.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 12.5,
        "legend.fontsize": 10.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })


# -------------------------------------------------------------------------
# 3. 数据加载
# -------------------------------------------------------------------------
def load_annex1_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到数据文件: {CSV_FILE}")

    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    dt = 1.0 / 6.0
    starts = np.array([float(r["start_hour"]) for r in rows])
    ends = np.array([float(r["end_hour"]) for r in rows])
    centers = starts + dt / 2.0

    load_kw = np.array([float(r["load_kw"]) for r in rows])
    pv_kw = np.array([float(r["pv_kw"]) for r in rows])
    price = np.array([float(r["price_yuan_per_kwh"]) for r in rows])

    # 阶梯曲线拓展
    t_stair = []
    price_stair = []
    load_stair = []
    pv_stair = []
    for s, e, pr, l, pv in zip(starts, ends, price, load_kw, pv_kw):
        t_stair.extend([s, e])
        price_stair.extend([pr, pr])
        load_stair.extend([l, l])
        pv_stair.extend([pv, pv])

    return {
        "centers": centers,
        "load_kw": load_kw,
        "pv_kw": pv_kw,
        "price": price,
        "t_stair": np.array(t_stair),
        "price_stair": np.array(price_stair),
        "load_stair": np.array(load_stair),
        "pv_stair": np.array(pv_stair),
    }


# -------------------------------------------------------------------------
# 4. 图1：光伏和负载预测
# -------------------------------------------------------------------------
def plot_pv_and_load(data: dict):
    clr_load_line = "#1F4E79"     # 小区负载：深海蓝
    clr_load_fill = "#2980B9"     # 负载浅蓝阴影
    clr_pv_line = "#D97706"       # 光伏预测：阳光琥珀金
    clr_pv_fill = "#F59E0B"       # 光伏金黄浅影

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)

    # 1. 小区负载曲线
    f_load = ax.fill_between(data["centers"], 0, data["load_kw"], color=clr_load_fill, alpha=0.15, zorder=1)
    l_load, = ax.plot(data["centers"], data["load_kw"], color=clr_load_line, linewidth=2.0, zorder=3)

    # 2. 光伏发电预测功率曲线
    f_pv = ax.fill_between(data["centers"], 0, data["pv_kw"], color=clr_pv_fill, alpha=0.22, zorder=2)
    l_pv, = ax.plot(data["centers"], data["pv_kw"], color=clr_pv_line, linewidth=2.0, zorder=4)

    # 坐标轴与范围
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8500)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_xlabel("时刻", labelpad=6)
    ax.set_ylabel("功率 / kW", labelpad=8)
    ax.set_title("光伏和负载预测", pad=32, fontsize=13.0, weight="bold")

    # 图例：纯中文无字母
    handles = [
        (f_load, l_load),
        (f_pv, l_pv),
    ]
    labels = ["小区负载", "光伏发电预测功率"]

    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=10.0,
        columnspacing=3.0,
        handlelength=2.4,
    )

    plt.tight_layout()

    targets = [
        CURRENT_DIR / "光伏和负载预测.png",
        CURRENT_DIR / "光伏和负载预测.svg",
        OUTPUT_DIR / "光伏和负载预测.png",
        OUTPUT_DIR / "光伏和负载预测.svg",
    ]
    for p in targets:
        if p.suffix == ".png":
            fig.savefig(p, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(p, bbox_inches="tight")
        print(f"已生成: {p}")

    plt.close(fig)


# -------------------------------------------------------------------------
# 5. 图2：分时电价折线图
# -------------------------------------------------------------------------
def plot_tou_price(data: dict):
    clr_price_line = "#6C3483"    # 典雅学术紫
    clr_price_fill = "#A569BD"    # 浅紫底衬

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)

    t = data["t_stair"]
    price = data["price_stair"]

    # 阶梯折线与半透明底色
    f_price = ax.fill_between(t, 0, price, step="pre", color=clr_price_fill, alpha=0.12, zorder=1)
    l_price, = ax.step(t, price, where="pre", color=clr_price_line, linewidth=1.9, zorder=3)

    # 坐标轴与范围
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.6)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_xlabel("时刻", labelpad=6)
    ax.set_ylabel("电价 / (元/kWh)", labelpad=8)
    ax.set_title("分时电价折线图", pad=32, fontsize=13.0, weight="bold")

    # 图例：纯中文无字母
    ax.legend(
        [(f_price, l_price)],
        ["分时电价"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=10.0,
        handlelength=2.4,
    )

    plt.tight_layout()

    targets = [
        CURRENT_DIR / "分时电价折线图.png",
        CURRENT_DIR / "分时电价折线图.svg",
        OUTPUT_DIR / "分时电价折线图.png",
        OUTPUT_DIR / "分时电价折线图.svg",
    ]
    for p in targets:
        if p.suffix == ".png":
            fig.savefig(p, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(p, bbox_inches="tight")
        print(f"已生成: {p}")

    plt.close(fig)


def main():
    setup_academic_style()
    data = load_annex1_data()
    plot_pv_and_load(data)
    plot_tou_price(data)


if __name__ == "__main__":
    main()
