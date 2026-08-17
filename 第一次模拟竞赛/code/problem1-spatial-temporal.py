# -*- coding: utf-8 -*-
"""
problem1-spatial-temporal.py
问题一：区域空间、24 小时时段、工作日/周末分布分析

输出：
  results/表1–表8
  results/问题一_结果汇总.xlsx
  figures/图1–图4
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem1_utils import FIGURES_DIR, RESULTS_DIR, save_table

BASE_DIR = Path(__file__).resolve().parent.parent
ATTACH_DIR = BASE_DIR / "附件"

for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

HOUR_LABELS = [
    "00-01", "01-02", "02-03", "03-04", "04-05", "05-06",
    "06-07", "07-08", "08-09", "09-10", "10-11", "11-12",
    "12-13", "13-14", "14-15", "15-16", "16-17", "17-18",
    "18-19", "19-20", "20-21", "21-22", "22-23", "23-00",
]

plt.rcParams["font.sans-serif"] = [
    "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


def fmt_hour_list(indices_1based: list[int]) -> str:
    return "、".join(HOUR_LABELS[i - 1] for i in indices_1based)


def argmax_all(arr: np.ndarray) -> list[int]:
    m = np.max(arr)
    return (np.where(np.isclose(arr, m))[0] + 1).tolist()


def argmin_all(arr: np.ndarray) -> list[int]:
    m = np.min(arr)
    return (np.where(np.isclose(arr, m))[0] + 1).tolist()


def read_hourly_matrix(path: Path, sheet: str) -> np.ndarray:
    df = pd.read_excel(path, sheet_name=sheet, header=0)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    if df.iloc[:, 0].tolist() != list(range(1, 11)):
        raise ValueError(f"{path.name}/{sheet} 区域编号不对应")
    mat = df.iloc[:, 1:25].astype(float).to_numpy()
    if mat.shape != (10, 24):
        raise ValueError(f"{sheet} 形状应为(10,24)，实际{mat.shape}")
    if np.any(mat < 0):
        raise ValueError(f"{sheet} 存在负负荷")
    return mat


def load_loads() -> dict:
    path3 = ATTACH_DIR / "附件3 市主城区 10 区域分时段充电负荷.xlsx"
    load_wd = read_hourly_matrix(path3, "工作日分时段充电负荷数据")
    load_we = read_hourly_matrix(path3, "周末充电负荷数据（修改后）")
    return {
        "region": np.arange(1, 11),
        "load_wd": load_wd,
        "load_we": load_we,
    }


def calc_spatial(data: dict) -> dict:
    Q_wd = data["load_wd"].sum(axis=1)
    Q_we = data["load_we"].sum(axis=1)
    Q_bar = (5 * Q_wd + 2 * Q_we) / 7.0
    order = np.argsort(-Q_bar)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, 11)

    table1 = pd.DataFrame({
        "区域": data["region"],
        "工作日全天负荷_Qwd": Q_wd,
        "周末全天负荷_Qwe": Q_we,
        "典型日均负荷_Qbar": Q_bar,
        "降序排名": ranks,
    })
    save_table(table1, "表1_区域典型日均负荷")

    mu_Q = float(np.mean(Q_bar))
    sigma_Q = float(np.sqrt(np.mean((Q_bar - mu_Q) ** 2)))
    table2 = pd.DataFrame([{
        "mu_Q": mu_Q,
        "Delta_Q": float(Q_bar.max() - Q_bar.min()),
        "R_Q": float(Q_bar.max() / Q_bar.min()),
        "sigma_Q": sigma_Q,
        "CV_percent": float(sigma_Q / mu_Q * 100.0),
        "max_region": int(data["region"][np.argmax(Q_bar)]),
        "min_region": int(data["region"][np.argmin(Q_bar)]),
        "max_Qbar": float(Q_bar.max()),
        "min_Qbar": float(Q_bar.min()),
    }])
    save_table(table2, "表2_区域空间差异指标")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    q_sorted = Q_bar[order]
    regions_sorted = data["region"][order]
    bars = ax.bar([str(r) for r in regions_sorted], q_sorted, color="#2E86AB",
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, q_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("区域编号（按典型日均负荷降序）")
    ax.set_ylabel("典型日均负荷 (kWh/d)")
    ax.set_title("图1 区域典型日均充电负荷")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(FIGURES_DIR / "图1_区域典型日均充电负荷柱状图.png")
    plt.close(fig)
    return {"Q_wd": Q_wd, "Q_we": Q_we, "Q_bar": Q_bar}


def calc_weekday_hourly(data: dict) -> None:
    load_wd = data["load_wd"]
    L_bar = load_wd.mean(axis=0)
    t_max_list = argmax_all(L_bar)
    t_min_list = argmin_all(L_bar)
    L_max, L_min = float(L_bar.max()), float(L_bar.min())

    table3 = pd.DataFrame({
        "时段": HOUR_LABELS,
        "平均负荷_Lt": L_bar,
        "是否峰值时段": ["是" if (i + 1) in t_max_list else "否" for i in range(24)],
        "是否谷值时段": ["是" if (i + 1) in t_min_list else "否" for i in range(24)],
    })
    save_table(table3, "表3_工作日分时平均负荷")
    save_table(pd.DataFrame([{
        "峰值时段": fmt_hour_list(t_max_list),
        "峰值负荷": L_max,
        "谷值时段": fmt_hour_list(t_min_list),
        "谷值负荷": L_min,
        "峰谷差_DeltaL": L_max - L_min,
        "峰谷差率_rho_percent": (L_max - L_min) / L_max * 100.0,
    }]), "表4_工作日峰谷指标")

    peak_hours, peak_loads, peak_idx = [], [], []
    for i in range(10):
        idxs = argmax_all(load_wd[i])
        peak_hours.append(fmt_hour_list(idxs))
        peak_loads.append(float(load_wd[i].max()))
        peak_idx.append(idxs)
    save_table(pd.DataFrame({
        "区域": data["region"],
        "峰值时段_ti_star": peak_hours,
        "峰值负荷": peak_loads,
    }), "表5_区域工作日峰值时段")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(24)
    ax.plot(x, L_bar, "o-", color="#E63946", linewidth=2, markersize=5)
    for t in t_max_list:
        ax.scatter(t - 1, L_bar[t - 1], s=120, c="red", zorder=5, marker="^")
        ax.annotate(f"峰 {HOUR_LABELS[t-1]}\n{L_bar[t-1]:.1f}",
                    xy=(t - 1, L_bar[t - 1]), xytext=(8, 12),
                    textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray"))
    for t in t_min_list:
        ax.scatter(t - 1, L_bar[t - 1], s=120, c="blue", zorder=5, marker="v")
        ax.annotate(f"谷 {HOUR_LABELS[t-1]}\n{L_bar[t-1]:.1f}",
                    xy=(t - 1, L_bar[t - 1]), xytext=(8, -28),
                    textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xticks(x)
    ax.set_xticklabels(HOUR_LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("时段")
    ax.set_ylabel("平均充电负荷 (kWh)")
    ax.set_title("图2 工作日平均24小时充电负荷曲线")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(FIGURES_DIR / "图2_工作日平均24小时充电负荷曲线.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(load_wd, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels([str(r) for r in data["region"]])
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(HOUR_LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("时段")
    ax.set_ylabel("区域")
    ax.set_title("图3 区域工作日负荷热力图")
    fig.colorbar(im, ax=ax).set_label("充电负荷 (kWh)")
    for i in range(10):
        for t in peak_idx[i]:
            ax.plot(t - 1, i, "w*", markersize=8)
    fig.savefig(FIGURES_DIR / "图3_区域工作日负荷热力图.png")
    plt.close(fig)


def calc_wd_we(data: dict, spatial: dict) -> None:
    Q_wd, Q_we = spatial["Q_wd"], spatial["Q_we"]
    d = Q_wd - Q_we
    t_scipy, p_scipy = stats.ttest_rel(Q_wd, Q_we)
    s_d = float(np.std(d, ddof=1))
    d_bar = float(np.mean(d))
    t_manual = d_bar / (s_d / np.sqrt(10))
    conclusion = "拒绝H0：存在显著差异" if p_scipy < 0.05 else "不能拒绝H0：差异未达显著水平"
    save_table(pd.DataFrame([{
        "工作日均值": float(np.mean(Q_wd)),
        "周末均值": float(np.mean(Q_we)),
        "平均差_dbar": d_bar,
        "差值标准差_sd": s_d,
        "t值": float(t_scipy),
        "t值_手工": float(t_manual),
        "自由度": 9,
        "双侧p值": float(p_scipy),
        "alpha0.05结论": conclusion,
    }]), "表6_工作日周末配对检验")

    L_wd = data["load_wd"].mean(axis=0)
    L_we = data["load_we"].mean(axis=0)
    save_table(pd.DataFrame({
        "时段": HOUR_LABELS,
        "工作日平均负荷": L_wd,
        "周末平均负荷": L_we,
        "差值_周末减工作日": L_we - L_wd,
    }), "表7_工作日周末分时负荷对比")

    t_max_wd_list, t_max_we_list = argmax_all(L_wd), argmax_all(L_we)
    t_max_wd, t_max_we = t_max_wd_list[0], t_max_we_list[0]
    delta_t = t_max_we - t_max_wd
    peak_wd, peak_we = float(L_wd.max()), float(L_we.max())
    save_table(pd.DataFrame([{
        "工作日峰值时段": fmt_hour_list(t_max_wd_list),
        "周末峰值时段": fmt_hour_list(t_max_we_list),
        "delta_t": delta_t,
        "工作日峰值": peak_wd,
        "周末峰值": peak_we,
        "r_max_percent": (peak_we - peak_wd) / peak_wd * 100.0,
        "说明_delta_t": "周末峰值提前" if delta_t < 0 else ("周末峰值延后" if delta_t > 0 else "峰值时段一致"),
    }]), "表8_峰值平移与强度变化")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(24)
    ax.plot(x, L_wd, "o-", color="#E63946", linewidth=2, markersize=4, label="工作日")
    ax.plot(x, L_we, "s-", color="#2A9D8F", linewidth=2, markersize=4, label="周末")
    ax.scatter(t_max_wd - 1, peak_wd, s=120, c="#E63946", marker="^", zorder=5)
    ax.scatter(t_max_we - 1, peak_we, s=120, c="#2A9D8F", marker="^", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(HOUR_LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("时段")
    ax.set_ylabel("平均充电负荷 (kWh)")
    ax.set_title("图4 工作日与周末平均24小时负荷对比")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.savefig(FIGURES_DIR / "图4_工作日周末平均24小时负荷对比.png")
    plt.close(fig)


def main():
    print("problem1-spatial-temporal: 空间/时段/工作日周末分析")
    data = load_loads()
    spatial = calc_spatial(data)
    calc_weekday_hourly(data)
    calc_wd_we(data, spatial)
    print("完成 -> results/表1-表8, figures/图1-图4")


if __name__ == "__main__":
    main()
