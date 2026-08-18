# -*- coding: utf-8 -*-
"""
问题3 · Step 4：图表输出（任务单 §9.4）
=======================================
生成图件（figures/p3_*/，统一 DPI=200，统一配色与图例规范见 figures/p3_绘图说明.md）：

    A. p3_curve_region{i}_{wd|we}.png （20 张）
       —— 每个区域 × 日期类型一张折线叠加图：调度前负荷 / 调度后负荷 / 电网最大允许负荷；
          背景色块区分低谷（蓝）、高峰（红）、平段及非调度时段（黄/灰），含坐标轴刻度。

    B. p3_city_aggregate_{wd|we}.png （2 张）
       —— 10 区域逐时求和的“调度前 / 调度后”全市汇总曲线，用于展示总体削峰填谷趋势
          （同时叠加 10 区域电网上限之和作为参考）。

    C. p3_improvement_bar.jpg（或 .png）（1–2 张）
       —— 峰谷差改善率与方差改善率的 10 区域对比柱状图（工作日 vs 周末并排分组，双面板）。

数据源：results/p3_results/p3_curves_all.csv、p3_表3_调度效果评价.csv；
        results/p3_data/time_sets.json、G_limit.csv。

配色规范（全篇统一）：
    调度前: #8C8C8C(灰,虚线)   调度后: #0B5FA4(深蓝,实线)   电网上限: #D62728(红,点划线)
    背景色块: 低谷 #E3F2FD(浅蓝) / 高峰 #FFEBEE(浅红) / 平段及非调度 #FFF9C4/#F5F5F5

用法：python problem3_figures.py   （依赖 matplotlib；Windows 字体 msyh/simhei 自动探测）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "p3_data"
RES_DIR = ROOT / "results" / "p3_results"
OUT_DIR = ROOT / "figures" / "p3_figures"

DPI = 200
C_PRE, C_POST, C_GRID = "#8C8C8C", "#0B5FA4", "#D62728"
BG_VALLEY, BG_PEAK, BG_OTHER1, BG_OTHER2 = "#E3F2FD", "#FFEBEE", "#FFF9C4", "#F5F5F5"

HOUR_LABELS = [f"{h:02d}" for h in range(24)]


def setup_cjk_font() -> None:
    """优先微软雅黑→SimHei，找不到则退回 DejaVu（图内中文可能缺字）。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc", "/mnt/c/Windows/Fonts/msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf", "/mnt/c/Windows/Fonts/simhei.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 使用中文字体：{Path(p).name}")
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


def shade_periods(ax, TS: dict) -> None:
    """在轴上按时段集合绘制背景色块：谷(蓝)/峰(红)/平与剩余(黄)/非调度(灰)。"""
    colors = {t: BG_VALLEY for t in TS["valley_hours"]}
    for t in TS["peak_hours"]:
        colors[t] = BG_PEAK if True else None                # 高峰→浅红（含23点单独处理）
    for t in TS["middle_hours"]:
        colors[t] = BG_OTHER1                                # 平段→浅黄
    for t in TS["unchanged_other_hours"]:
        colors[t] = BG_OTHER2                                # 23-00非调度→浅灰
    for h in range(24):
        ax.axvspan(h - 0.5, h + 0.5, color=colors.get(h, "white"), alpha=0.9, zorder=0)


def style_ax(ax: plt.Axes) -> None:
    ax.set_xlim(-0.6, 23.6)
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels(HOUR_LABELS, rotation=45, fontsize=8)
    ax.grid(axis="y", alpha=0.35, zorder=0)


def legend_period(ax: plt.Axes) -> None:
    handle = [Patch(color=BG_VALLEY, label="低谷 00–07"), Patch(color=BG_PEAK, label="高峰 11–14/16–23(23点非调度)"),
              Patch(color=BG_OTHER1, label="平段 07–11/14–16")]
    ax.legend(handles=handle, loc="upper left", fontsize=8, framealpha=0.7, ncols=3)


def main() -> None:
    global TS_CACHE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_cjk_font()

    with open(DATA_DIR / "time_sets.json", encoding="utf-8") as f:
        TS = json.load(f)
    curves = pd.read_csv(RES_DIR / "p3_curves_all.csv")
    t3 = pd.read_csv(RES_DIR / "p3_表3_调度效果评价.csv")

    dt_name = {"wd": "工作日", "we": "周末"}

    # ---------------- A. 20 张区域级“前/后/上限”图 ----------------
    for i in range(1, 11):
        for dkey in ("wd", "we"):
            sub = curves[(curves["区域"] == i) & (curves["日期类型"] == dt_name[dkey])].sort_values("hour")
            assert len(sub) == 24
            fig, ax = plt.subplots(figsize=(10.5, 4.6))
            shade_periods(ax, TS)
            x = sub["hour"].to_numpy()
            ax.plot(x, sub["调度前kW"], color=C_PRE, ls="--", lw=1.8, label="调度前充电负荷")
            ax.plot(x, sub["调度后kW"], color=C_POST, ls="-", lw=2.4, label="调度后充电负荷（峰−20%，谷+转移）")
            ax.plot(x, sub["电网上限kW"], color=C_GRID, ls="-.", lw=1.6, alpha=0.9, label="电网最大允许负荷 G_i,t")
            ymax = float(sub[["调度前kW", "调度后kW", "电网上限kW"]].to_numpy().max()) * 1.08
            ax.set_ylim(0, ymax)
            style_ax(ax)
            # 合并时段色块说明与业务曲线图例（同一图例框内）
            hh, ll = ax.get_legend_handles_labels()
            period_handles = [Patch(color=BG_VALLEY), Patch(color=BG_PEAK), Patch(color=BG_OTHER1)]
            period_labels = ["低谷 00–07", "高峰时段(23点非调度)", "平段 07–11/14–16"]
            ax.legend(handles=period_handles + list(hh), labels=period_labels + ll, loc="upper left", fontsize=8, framealpha=0.75)
            ax.set_ylabel("充电负荷 / kW")
            ax.set_xlabel("小时（起点时刻）")
            dtag = dt_name[dkey]
            _ip = int(sub["调度后kW"].to_numpy().argmax())   # sub.index为全局行标签，须用位置argmax
            _ht = HOUR_LABELS[int(sub.iloc[_ip, sub.columns.get_loc("hour")])]
            ax.set_title(f"区域 {i} · {dtag} 分时电价调度前后负荷曲线与电网上限\n"
                             f"(调度后最大负荷 {sub['调度后kW'].iloc[_ip]:.0f} kW @ {_ht}; "
                             f"调度前峰值 {sub['调度前kW'].max():.0f} kW)")
            fig.tight_layout()
            out = OUT_DIR / f"p3_curve_region{i}_{dkey}.png"
            fig.savefig(out, dpi=DPI)
            plt.close(fig)
    print(f"[OK] A 组：20 张区域曲线图 → {OUT_DIR}")

    # ---------------- B. 全市汇总（10 区域求和） × 2 ----------------
    Gm = pd.read_csv(DATA_DIR / "G_limit.csv").drop(columns=["区域"]).to_numpy(float)
    for dkey in ("wd", "we"):
        sub = curves[curves["日期类型"] == dt_name[dkey]].groupby("hour")[["调度前kW", "调度后kW"]].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        shade_periods(ax, TS)
        x = sub["hour"].to_numpy() - 0.5   # 对齐背景格中心
        ax.plot(sub["hour"], Gm.sum(axis=0), color=C_GRID, ls="-.", lw=1.4, alpha=0.85, label="10 区域电网上限之和（参考）")
        ax.plot(sub["hour"], sub["调度前kW"], color=C_PRE, ls="--", lw=1.8, label="全市 10 区域合计 · 调度前")
        ax.plot(sub["hour"], sub["调度后kW"], color=C_POST, ls="-", lw=2.6, marker="o", ms=3.5,
                label="全市 10 区域合计 · 调度后（分时电价引导）")
        i_p = int(np.argmax(sub["调度后kW"].to_numpy()))
        ax.annotate(f"最高 {sub['调度后kW'].iloc[i_p]:,.0f} kW", (i_p, sub['调度后kW'].iloc[i_p]),
                    textcoords="offset points", xytext=(28, -6), fontsize=9)
        style_ax(ax)
        h1, l1 = ax.get_legend_handles_labels()
        period_handles = [Patch(color=BG_VALLEY), Patch(color=BG_PEAK), Patch(color=BG_OTHER1)]
        period_labels = ["低谷时段", "高峰时段", "平段/非调度"]
        ax.legend(handles=period_handles + list(h1), labels=period_labels + l1, loc="upper left", fontsize=8)
        ax.set_ylabel("合计充电负荷 / kW")
        ax.set_xlabel("小时（起点时刻）")
        gap_pre = sub["调度前kW"].max() - sub["调度前kW"].min()
        gap_post = sub["调度后kW"].max() - sub["调度后kW"].min()
        ax.set_title(f"全市 10 区域合计负荷曲线 · {dt_name[dkey]}（峰谷差 {gap_pre:,.0f} → {gap_post:,.0f} kW，改善 {(gap_pre-gap_post)/gap_pre*100:.1f}%）")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"p3_city_aggregate_{dkey}.png", dpi=DPI)
        plt.close(fig)
    print("[OK] B 组：2 张全市汇总图")

    # ---------------- C. 改善率区域对比柱状图（双面板） ----------------
    regs = list(range(1, 11))
    def val(dt, col):
        return t3[t3["日期类型"] == dt].set_index("区域").loc[regs, col].to_numpy(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    xpos = np.arange(len(regs)); wdt = 0.4
    for ax, col in zip((ax1, ax2), ("峰谷差改善率%", "方差改善率%")):
        v_wd, v_we = val("工作日", col), val("周末", col)
        b1 = ax.bar(xpos - wdt / 2, v_wd, wdt, color=C_POST, alpha=0.95, label="工作日")
        b2 = ax.bar(xpos + wdt / 2, v_we, wdt, color="#F4A261", edgecolor="black", linewidth=0.4, label="周末")
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + max(v_wd.max(), v_we.max())*0.012,
                        f"%.1f" % round(float(r.get_height()), 1), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xpos); ax.set_xticklabels([str(r) for r in regs])
        ax.grid(axis="y", alpha=0.35)
        upper = max(v_wd.max(), v_we.max()) * 1.22
        ax.set_ylim(0, upper)
    ax1.set_ylabel("改善率 / %")
    ax1.set_title("峰谷差改善率（区域对比）", fontsize=11)
    fig.axes[1].set_title("方差改善率（区域对比）", fontsize=11)
    fig.legend(loc="upper right", bbox_to_anchor=(0.99, 1.02), ncol=2, fontsize=9)
    ax1.set_xlabel("区域编号")
    fig.suptitle("问题3 · 分时电价调度效果：峰谷差 / 方差改善率（工作日 vs 周末，n=10 区域）", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p3_improvement_bar.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("[OK] C 组：改善率对比柱状图 ×1\n全部图表输出完成 →", OUT_DIR)


TS_CACHE = None
if __name__ == "__main__":
    main()
