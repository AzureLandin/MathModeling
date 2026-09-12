"""问题一：全天调度总览图"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import font_manager

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output"
CSV_FILE = OUTPUT_DIR / "q1_intervals.csv"

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


def load_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到调度数据文件: {CSV_FILE}")

    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    dt = 1.0 / 6.0  # 10 分钟 = 1/6 小时
    starts = np.array([float(r["start_hour"]) for r in rows])
    ends = np.array([float(r["end_hour"]) for r in rows])
    
    pur_kwh = np.array([float(r["purchase_kwh"]) for r in rows])
    chg_kwh = np.array([float(r["charge_kwh"]) for r in rows])
    dis_kwh = np.array([float(r["discharge_kwh"]) for r in rows])

    # 换算为调度功率 (kW)
    p_pur = pur_kwh / dt
    p_chg = chg_kwh / dt
    p_dis = dis_kwh / dt

    # 严格物理阶梯插值 (保持真实调度阶梯跳变，绝无斜坡假象)
    t_stair = []
    pur_stair = []
    chg_stair = []
    dis_stair = []
    
    for s, e, p, c, d in zip(starts, ends, p_pur, p_chg, p_dis):
        t_stair.extend([s, e])
        pur_stair.extend([p, p])
        chg_stair.extend([c, c])
        dis_stair.extend([d, d])

    return {
        "t_stair": np.array(t_stair),
        "pur_stair": np.array(pur_stair),
        "chg_stair": np.array(chg_stair),
        "dis_stair": np.array(dis_stair),
    }


def plot_dispatch_overview(data: dict):
    # 学术期刊标准色彩方案
    clr_pur_line = "#1F4E79"      # 购电功率：深海蓝
    clr_chg_line = "#196F3D"      # 充电功率：翡翠绿边框
    clr_chg_fill = "#27AE60"      # 充电功率：半透明填充
    clr_dis_line = "#922B21"      # 放电功率：砖红边框
    clr_dis_fill = "#C0392B"      # 放电功率：半透明填充

    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)
    
    t = data["t_stair"]
    pur = data["pur_stair"]
    chg = data["chg_stair"]
    dis = data["dis_stair"]

    # 1. 充电功率 (正向阶梯填充)
    f_chg = ax.fill_between(t, 0, chg, step="pre", color=clr_chg_fill, alpha=0.38, zorder=2)
    l_chg, = ax.step(t, chg, where="pre", color=clr_chg_line, linewidth=1.15, zorder=3)

    # 2. 放电功率 (负向阶梯填充，负值体现能量反哺微网)
    f_dis = ax.fill_between(t, 0, -dis, step="pre", color=clr_dis_fill, alpha=0.38, zorder=2)
    l_dis, = ax.step(t, -dis, where="pre", color=clr_dis_line, linewidth=1.15, zorder=3)

    # 3. 购电功率 (深蓝阶梯主线)
    l_pur, = ax.step(t, pur, where="pre", color=clr_pur_line, linewidth=1.75, zorder=4)

    # 零基准线
    ax.axhline(0, color="#2C3E50", linewidth=1.0, linestyle="-", zorder=3)

    # 坐标轴刻度与标签
    ax.set_xlim(0, 24)
    ax.set_ylim(-5600, 9500)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.set_xlabel("时刻", labelpad=6)
    ax.set_ylabel("功率 / kW（放电为负）", labelpad=8)
    ax.set_title("全天调度总览", pad=32, fontsize=13.0, weight="bold")

    # 图例：纯中文无字母，置于绘图区上方居中
    handles = [
        (f_chg, l_chg),
        (f_dis, l_dis),
        l_pur,
    ]
    labels = ["充电功率", "放电功率", "购电功率"]
    
    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        fontsize=10.0,
        columnspacing=2.5,
        handlelength=2.2,
    )

    plt.tight_layout()

    # 导出保存 (同时保存在本文件夹与 output 文件夹)
    targets = [
        CURRENT_DIR / "全天调度总览.png",
        CURRENT_DIR / "全天调度总览.svg",
        OUTPUT_DIR / "全天调度总览.png",
        OUTPUT_DIR / "全天调度总览.svg",
    ]
    
    for t_path in targets:
        if t_path.suffix == ".png":
            fig.savefig(t_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(t_path, bbox_inches="tight")
        print(f"已生成: {t_path}")

    plt.close(fig)


def main():
    setup_academic_style()
    data = load_data()
    plot_dispatch_overview(data)


if __name__ == "__main__":
    main()
