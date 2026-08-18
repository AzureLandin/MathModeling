# -*- coding: utf-8 -*-
"""公共绘图工具：字体、主题、配色、输出函数。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
P1_FIG = FIG_DIR / "p1_figures"
P2_FIG = FIG_DIR / "p2_figures"
P3_FIG = FIG_DIR / "p3_figures"

DPI = 300

# 统一配色
C_WD = "#1F77B4"       # 工作日 蓝
C_WE = "#FF7F0E"       # 周末 橙
C_PRE = "#7F7F7F"      # 调度前 灰
C_POST = "#2CA02C"     # 调度后 绿
C_GRID = "#D62728"     # 电网上限 红
C_REC = "#E31A1C"      # 推荐解 红
C_BG_VALLEY = "#E3F2FD"
C_BG_PEAK = "#FFEBEE"
C_BG_MID = "#FFF9C4"
C_BG_OTHER = "#F5F5F5"

# 时段集合（与问题3一致）
VALLEY_HOURS = [0, 1, 2, 3, 4, 5, 6]
PEAK_HOURS = [11, 12, 13, 16, 17, 18, 19, 20, 21, 22]
MID_HOURS = [7, 8, 9, 10, 14, 15]
OTHER_HOURS = [23]

HOUR_LABELS = [f"{h:02d}-{(h+1)%24:02d}" for h in range(24)]


def setup_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"


def ensure_dirs():
    for d in (P1_FIG, P2_FIG, P3_FIG):
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, path: Path, pdf=True):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def add_time_bg(ax):
    """在24小时曲线上添加峰/平/谷背景色。"""
    for t in VALLEY_HOURS:
        ax.axvspan(t - 0.5, t + 0.5, alpha=0.10, color=C_BG_VALLEY, zorder=0)
    for t in PEAK_HOURS:
        ax.axvspan(t - 0.5, t + 0.5, alpha=0.10, color=C_BG_PEAK, zorder=0)
    for t in MID_HOURS:
        ax.axvspan(t - 0.5, t + 0.5, alpha=0.08, color=C_BG_MID, zorder=0)
    ax.axvspan(22.5, 23.5, alpha=0.05, color=C_BG_OTHER, zorder=0)


def bar_with_labels(ax, x, vals, width=0.6, fmt="{:.0f}", **kwargs):
    """柱状图 + 柱顶标注。"""
    bars = ax.bar(x, vals, width, **kwargs)
    for bar, v in zip(bars, vals):
        if v != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=7)
    return bars
