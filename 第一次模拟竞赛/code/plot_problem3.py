# -*- coding: utf-8 -*-
"""问题3 制图脚本。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_common import *

setup_font()
ensure_dirs()

RESULTS = ROOT / "results" / "p3_results"
P3_DATA = ROOT / "p3_data"


def load(name):
    return pd.read_csv(RESULTS / f"{name}.csv", encoding="utf-8-sig")


def load_merged():
    with open(RESULTS / "subproblem_results_merged.json", encoding="utf-8") as f:
        return json.load(f)


def post_load(L, z, HH, VH, ratio):
    Lp = L.copy()
    Lp[HH] *= (1.0 - ratio)
    for k, t in enumerate(VH):
        Lp[t] += z[k]
    return Lp


# ── P3-F01 调度前峰谷差对比 ──
def p3_f01():
    df = load("p3_02_调度前峰谷特征").sort_values("区域")
    wd = df[df["日期类型"] == "工作日"]
    we = df[df["日期类型"] == "周末"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    w = 0.35
    ax.bar(x - w/2, wd["调度前峰谷差_kW"], w, label="工作日", color=C_WD, edgecolor="black", linewidth=0.3)
    ax.bar(x + w/2, we["调度前峰谷差_kW"], w, label="周末", color=C_WE, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in range(10)])
    ax.set_xlabel("区域编号"); ax.set_ylabel("调度前峰谷差 / kW")
    ax.set_title("P3-F01  调度前峰谷差 工作日—周末对比")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, P3_FIG / "p3_f01_调度前峰谷差_工作日周末对比.png")


# ── P3-F02 转移量与低谷接纳能力 ──
def p3_f02():
    df = load("p3_03_转移量与低谷接纳能力").sort_values("区域")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dt, color in [(ax1, "工作日", C_WD), (ax2, "周末", C_WE)]:
        sub = df[df["日期类型"] == dt]
        x = np.arange(10)
        w = 0.35
        ax.bar(x - w/2, sub["规定转移量_M_kWh"], w, label="规定转移量M", color=C_REC, edgecolor="black", linewidth=0.3)
        ax.bar(x + w/2, sub["低谷可接纳能力_B_kWh"], w, label="低谷接纳能力B", color=C_POST, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in range(10)])
        ax.set_xlabel("区域编号"); ax.set_ylabel("负荷量 / kWh")
        ax.set_title(f"{dt}"); ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("P3-F02  高峰转移量与低谷接纳能力 (min B/M≈4.67)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, P3_FIG / "p3_f02_转移量与低谷接纳能力.png")


# ── P3-F03/04 全市调度前后曲线 ──
def p3_f03_04():
    df = load("p3_08_全市逐时汇总曲线")
    summary = load("p3_08b_全市调度效果汇总")

    for dt, fig_name, delta_info in [
        ("工作日", "p3_f03_全市工作日调度前后曲线", "14057→9669 kW, -31.2%"),
        ("周末", "p3_f04_全市周末调度前后曲线", "11896→10484 kW, -11.9%"),
    ]:
        sub = df[df["日期类型"] == dt]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        add_time_bg(ax)
        ax.plot(sub["小时序号"], sub["调度前全市负荷_kW"], "o--", color=C_PRE, markersize=3, label="调度前", linewidth=1.5)
        ax.plot(sub["小时序号"], sub["调度后全市负荷_kW"], "^-", color=C_POST, markersize=3, label="调度后", linewidth=2)
        ax.annotate(delta_info, xy=(0.02, 0.95), xycoords="axes fraction", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5), va="top")
        ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([HOUR_LABELS[t] for t in range(0, 24, 2)], fontsize=8)
        ax.set_xlabel("时段"); ax.set_ylabel("全市充电负荷 / kW")
        ax.set_title(f"P3-{'F03' if dt=='工作日' else 'F04'}  全市{dt}调度前后负荷曲线")
        ax.legend(); ax.grid(linestyle="--", alpha=0.3)
        save_fig(fig, P3_FIG / f"{fig_name}.png")


# ── P3-F05 区域峰谷差改善率 ──
def p3_f05():
    df = load("p3_06_调度效果评价_最终合并").sort_values("区域")
    wd = df[df["日期类型"] == "工作日"]
    we = df[df["日期类型"] == "周末"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    w = 0.35
    ax.bar(x - w/2, wd["峰谷差改善率_pct"], w, label="工作日", color=C_WD, edgecolor="black", linewidth=0.3)
    ax.bar(x + w/2, we["峰谷差改善率_pct"], w, label="周末", color=C_WE, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    # 标注区域8周末=0
    ax.annotate("区域8周末=0%", xy=(7 + w/2, 0), xytext=(5, -5), fontsize=8, color=C_REC,
                arrowprops=dict(arrowstyle="->", color=C_REC))
    ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in range(10)])
    ax.set_xlabel("区域编号"); ax.set_ylabel("峰谷差改善率 / %")
    ax.set_title("P3-F05  区域峰谷差改善率")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, P3_FIG / "p3_f05_区域峰谷差改善率.png")


# ── P3-F06 区域方差改善率 ──
def p3_f06():
    df = load("p3_06_调度效果评价_最终合并").sort_values("区域")
    wd = df[df["日期类型"] == "工作日"]
    we = df[df["日期类型"] == "周末"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    w = 0.35
    ax.bar(x - w/2, wd["方差改善率_pct"], w, label="工作日", color=C_WD, edgecolor="black", linewidth=0.3)
    ax.bar(x + w/2, we["方差改善率_pct"], w, label="周末", color=C_WE, edgecolor="black", linewidth=0.3)
    # 标注区域10工作日最高
    r10_wd = wd[wd["区域"] == 10]["方差改善率_pct"].values[0]
    ax.annotate(f"区域10: {r10_wd:.1f}%", xy=(9 - w/2, r10_wd), xytext=(7, r10_wd + 3),
                fontsize=8, arrowprops=dict(arrowstyle="->", color=C_WD), color=C_WD)
    ax.set_xticks(x); ax.set_xticklabels([f"{i+1}" for i in range(10)])
    ax.set_xlabel("区域编号"); ax.set_ylabel("方差改善率 / %")
    ax.set_title("P3-F06  区域负荷方差改善率")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, P3_FIG / "p3_f06_区域负荷方差改善率.png")


# ── P3-F07 调度后负荷率与安全裕度 ──
def p3_f07():
    df = load("p3_06_调度效果评价_最终合并").sort_values("区域")
    wd = df[df["日期类型"] == "工作日"]
    we = df[df["日期类型"] == "周末"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(10); w = 0.35

    ax1.bar(x - w/2, wd["调度后最大负荷率_pct"], w, label="工作日", color=C_WD, edgecolor="black", linewidth=0.3)
    ax1.bar(x + w/2, we["调度后最大负荷率_pct"], w, label="周末", color=C_WE, edgecolor="black", linewidth=0.3)
    ax1.axhline(100, color=C_REC, linestyle="--", linewidth=1.5, label="100%上限")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{i+1}" for i in range(10)])
    ax1.set_ylabel("最大负荷率 / %"); ax1.set_title("调度后最大负荷率")
    ax1.legend(fontsize=8); ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2.bar(x - w/2, wd["最小安全裕度_kW"], w, label="工作日", color=C_WD, edgecolor="black", linewidth=0.3)
    ax2.bar(x + w/2, we["最小安全裕度_kW"], w, label="周末", color=C_WE, edgecolor="black", linewidth=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{i+1}" for i in range(10)])
    ax2.set_ylabel("最小安全裕度 / kW"); ax2.set_title("最小安全裕度")
    ax2.legend(fontsize=8); ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("P3-F07  调度后负荷率与安全裕度（全部安全）", fontsize=12)
    fig.tight_layout()
    save_fig(fig, P3_FIG / "p3_f07_调度后负荷率与安全裕度.png")


# ── P3-F08/09/10 典型区域调度曲线 ──
def _plot_region_curve(region, dt, fig_name, annotations):
    df = load("p3_05_调度前后曲线长表_合并后")
    sub = df[(df["区域"] == region) & (df["日期类型"] == dt)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    add_time_bg(ax)
    ax.plot(sub["小时序号"], sub["调度前负荷_kW"], "o--", color=C_PRE, markersize=3, label="调度前", linewidth=1.5)
    ax.plot(sub["小时序号"], sub["调度后负荷_kW"], "^-", color=C_POST, markersize=3, label="调度后", linewidth=2)
    ax.plot(sub["小时序号"], sub["最大允许负荷_kW"], ":", color=C_GRID, linewidth=1.5, label="最大允许负荷")
    for ann in annotations:
        ax.annotate(ann["text"], xy=ann["xy"], xytext=ann.get("xytext", (0, 20)),
                    textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color=ann.get("color", "black")),
                    color=ann.get("color", "black"),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7))
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([HOUR_LABELS[t] for t in range(0, 24, 2)], fontsize=8)
    ax.set_xlabel("时段"); ax.set_ylabel("负荷 / kW")
    ax.set_title(f"P3-{fig_name}  区域{region}{dt}调度曲线")
    ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.3)
    save_fig(fig, P3_FIG / f"{fig_name}_区域{region}{dt}调度曲线.png")


def p3_f08():
    _plot_region_curve(4, "工作日", "p3_f08", [
        {"text": "峰谷差改善35.0%\n方差改善67.5%", "xy": (12, 1600), "color": C_POST},
    ])

def p3_f09():
    _plot_region_curve(8, "周末", "p3_f09", [
        {"text": "峰值在平段10-11时\nRΔ=0%, Rσ=41.4%", "xy": (10, 600), "color": C_WD},
    ])

def p3_f10():
    _plot_region_curve(9, "周末", "p3_f10", [
        {"text": "原始峰2285kW@12-13\n削减后1828kW", "xy": (12, 2200), "color": C_REC},
        {"text": "调度后最大2038kW@14-15\n裕度614kW", "xy": (14, 2000), "color": C_POST},
    ])


# ── P3-F11 低谷水位填充分配示例 ──
def p3_f11():
    df_curve = load("p3_05_调度前后曲线长表_合并后")
    df_z = load("p3_04_低谷最优分配")
    # 选区域4工作日
    sub = df_curve[(df_curve["区域"] == 4) & (df_curve["日期类型"] == "工作日")]
    z_row = df_z[(df_z["区域"] == 4) & (df_z["日期类型"] == "工作日")]

    valley_data = sub[sub["时段类别"] == "低谷"]
    orig_load = valley_data["调度前负荷_kW"].values
    z_vals = [z_row[f"z_{h:02d}_{h+1:02d}_kW"].values[0] for h in VALLEY_HOURS]
    post_load_v = valley_data["调度后负荷_kW"].values
    g_limit = valley_data["最大允许负荷_kW"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(7)
    ax.bar(x, orig_load, 0.5, label="原始低谷负荷", color=C_PRE, edgecolor="black", linewidth=0.3)
    ax.bar(x, z_vals, 0.5, bottom=orig_load, label="接收转移量 z", color=C_WD, edgecolor="black", linewidth=0.3)
    for i in range(7):
        ax.text(x[i], post_load_v[i] + 10, f"{post_load_v[i]:.0f}", ha="center", va="bottom", fontsize=7)
    ax.plot(x, g_limit, "D-", color=C_GRID, markersize=5, label="最大允许负荷", linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{h:02d}-{h+1:02d}" for h in VALLEY_HOURS])
    ax.set_xlabel("低谷时段"); ax.set_ylabel("负荷 / kW")
    ax.set_title("P3-F11  区域4工作日 低谷水位填充分配")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, P3_FIG / "p3_f11_低谷水位填充分配示例.png")


# ── P3-F12 补算前后对比 ──
def p3_f12():
    # 从原始求解结果和补算结果中提取5个场景
    with open(P3_DATA / "time_sets.json") as f:
        ts = json.load(f)
    VH = list(ts["valley_hours"])
    HH = list(ts["peak_hours"])
    RATIO = float(ts["transfer_ratio_peak"])

    L_wd = pd.read_csv(P3_DATA / "L_pre_wd.csv").drop(columns=["区域"]).to_numpy(dtype=float)
    L_we = pd.read_csv(P3_DATA / "L_pre_we.csv").drop(columns=["区域"]).to_numpy(dtype=float)

    # 加载原始和合并后的结果
    with open(ROOT / "results" / "p3_solve" / "subproblem_results.json") as f:
        orig = json.load(f)
    merged = load_merged()

    cases = [(2, "we"), (5, "wd"), (7, "wd"), (8, "wd"), (10, "we")]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for idx, (region, dt) in enumerate(cases):
        ax = axes[idx]
        # 找原始和合并后的 z
        r_orig = next(r for r in orig if r["region"] == region and r["date_type"] == dt)
        r_merged = next(r for r in merged if r["region"] == region and r["date_type"] == dt)
        z_lp = np.array(r_orig["z_valley"])
        z_qp = np.array(r_merged["z_valley"])

        x = np.arange(7)
        w = 0.35
        ax.bar(x - w/2, z_lp, w, label="LP", color=C_PRE, edgecolor="black", linewidth=0.3)
        ax.bar(x + w/2, z_qp, w, label="QP", color=C_POST, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels([f"{h}" for h in VALLEY_HOURS], fontsize=7)
        ax.set_title(f"区域{region}{'工作日' if dt=='wd' else '周末'}", fontsize=9)
        if idx == 0:
            ax.set_ylabel("分配量 / kW")
        ax.legend(fontsize=7); ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("P3-F12  二阶段补算 LP vs QP 低谷分配对比", fontsize=12)
    fig.tight_layout()
    save_fig(fig, P3_FIG / "p3_f12_二阶段补算LP与QP低谷分配对比.png")


def main():
    print("[问题3] 制图...")
    for fn in [p3_f01, p3_f02, p3_f03_04, p3_f05, p3_f06, p3_f07, p3_f08, p3_f09, p3_f10, p3_f11, p3_f12]:
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__}: 跳过 ({e})")
    print("[问题3] 制图完成")


if __name__ == "__main__":
    main()
