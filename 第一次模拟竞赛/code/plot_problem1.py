# -*- coding: utf-8 -*-
"""问题1 制图脚本。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_common import *

setup_font()
ensure_dirs()

RESULTS = ROOT / "results" / "p1_results"


def load(name):
    return pd.read_csv(RESULTS / f"{name}.csv", encoding="utf-8-sig")


# ── P1-F02 工作日周末全天需求对比 ──
def p1_f02():
    df = load("p1_01_区域全天负荷汇总与排序").sort_values("区域")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(10)
    w = 0.35
    ax.bar(x - w/2, df["工作日全天负荷_kWh"], w, label="工作日", color=C_WD, edgecolor="black", linewidth=0.3)
    ax.bar(x + w/2, df["周末全天负荷_kWh"], w, label="周末", color=C_WE, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r)}" for r in df["区域"]])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("全天充电需求 / (kWh/day)")
    ax.set_title("P1-F02  工作日与周末全天需求对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_fig(fig, P1_FIG / "p1_f02_工作日周末全天需求对比.png")


# ── P1-F04 全市三类典型日曲线 ──
def p1_f04():
    df = load("p1_02b_全市分时负荷汇总")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df["小时序号"], df["全市工作日负荷_kWh"], "o-", color=C_WD, markersize=3, label="工作日", linewidth=1.5)
    ax.plot(df["小时序号"], df["全市周末负荷_kWh"], "s-", color=C_WE, markersize=3, label="周末", linewidth=1.5)
    ax.plot(df["小时序号"], df["全市综合典型日负荷_kWh"], "^-", color="#2CA02C", markersize=3, label="综合典型日", linewidth=2)

    # 标注峰谷
    star = df["全市综合典型日负荷_kWh"].values
    peak_t = int(np.argmax(star))
    valley_t = int(np.argmin(star))
    ax.annotate(f"峰值 {star[peak_t]:.0f}\n{HOUR_LABELS[peak_t]}", xy=(peak_t, star[peak_t]),
                xytext=(peak_t - 3, star[peak_t] + 500), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax.annotate(f"谷值 {star[valley_t]:.0f}\n{HOUR_LABELS[valley_t]}", xy=(valley_t, star[valley_t]),
                xytext=(valley_t + 2, star[valley_t] + 1500), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="blue"), color="blue")

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([HOUR_LABELS[t] for t in range(0, 24, 2)], fontsize=8)
    ax.set_xlabel("时段")
    ax.set_ylabel("全市充电负荷 / kW")
    ax.set_title("P1-F04  全市工作日、周末与综合典型日 24 小时曲线")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.3)
    save_fig(fig, P1_FIG / "p1_f04_全市工作日周末综合典型日曲线.png")


# ── P1-F08 LOOCV 预测对比散点图 ──
def p1_f08():
    df = load("p1_05_岭回归LOOCV逐区域预测")
    if "LOOCV预测值_kWh" not in df.columns:
        print("  P1-F08: 跳过（无LOOCV预测列）")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    actual = df["实际综合日均负荷_kWh"].values
    pred = df["LOOCV预测值_kWh"].values
    ax.scatter(actual, pred, s=80, color=C_WD, edgecolors="black", linewidth=0.5, zorder=5)
    for i, r in df.iterrows():
        ax.annotate(f"区域{int(r['区域'])}", (r["实际综合日均负荷_kWh"], r["LOOCV预测值_kWh"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    lim = [min(actual.min(), pred.min()) * 0.9, max(actual.max(), pred.max()) * 1.1]
    ax.plot(lim, lim, "--", color=C_PRE, linewidth=1, label="y=x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("实际综合日均需求 / (kWh/day)")
    ax.set_ylabel("LOOCV 预测值 / (kWh/day)")
    ax.set_title("P1-F08  岭回归 LOOCV 实际值—预测值")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.35)
    save_fig(fig, P1_FIG / "p1_f08_岭回归LOOCV预测对比.png")


def main():
    print("[问题1] 制图...")
    for fn in [p1_f02, p1_f04, p1_f08]:
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__}: 跳过 ({e})")
    print("[问题1] 制图完成")


if __name__ == "__main__":
    main()


