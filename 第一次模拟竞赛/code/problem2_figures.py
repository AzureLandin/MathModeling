# -*- coding: utf-8 -*-
"""
问题二图表生成。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from problem2_utils import COLORS, FIGURES_DIR, N_REGIONS, setup_plot_style, ensure_dirs


def plot_pareto_3d(solutions_ma: list[dict], solutions_mb: list[dict]) -> None:
    """图1：M-A、M-B 的三维 Pareto 前沿散点图。"""
    setup_plot_style()
    ensure_dirs()

    fig = plt.figure(figsize=(14, 6))

    for idx, (sols, name, color) in enumerate([
        (solutions_ma, "M-A", COLORS["ma"]),
        (solutions_mb, "M-B", COLORS["mb"]),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        if sols:
            f1 = [s["f1"] for s in sols]
            f2 = [s["C_city"] for s in sols]
            f3 = [s["f3"] for s in sols]
            ax.scatter(f1, f2, f3, c=color, s=12, alpha=0.6, edgecolors="none")
        ax.set_xlabel("总成本/万元", fontsize=9)
        ax.set_ylabel("覆盖率", fontsize=9)
        ax.set_zlabel("目标3", fontsize=9)
        ax.set_title(f"{name} Pareto 前沿 ({len(sols)} 解)", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "图1_Pareto前沿_3D.png")
    plt.close(fig)


def plot_stacked_bar(recommended: dict) -> None:
    """图2：最终方案各区域新增快慢充桩堆叠柱状图。"""
    setup_plot_style()
    ensure_dirs()

    sol = recommended["solution"]
    model = recommended["model"]
    x = sol["x"]
    y = sol["y"]
    regions = np.arange(1, N_REGIONS + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    w = 0.55
    ax.bar(regions, x, w, label="新增快充桩", color=COLORS["primary"], edgecolor="black", linewidth=0.3)
    ax.bar(regions, y, w, bottom=x, label="新增慢充桩", color=COLORS["tertiary"], edgecolor="black", linewidth=0.3)

    for i in range(N_REGIONS):
        total = x[i] + y[i]
        if total > 0:
            ax.text(regions[i], total + 0.5, str(int(total)), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("区域编号")
    ax.set_ylabel("新增桩数/台")
    ax.set_title(f"图2  最终方案（{model}）各区域新增快慢充桩")
    ax.set_xticks(regions)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图2_新增快慢充桩堆叠柱状图.png")
    plt.close(fig)


def plot_pressure_bar(recommended: dict) -> None:
    """图3：各区域新增接入压力率柱状图，标出均值。"""
    setup_plot_style()
    ensure_dirs()

    sol = recommended["solution"]
    model = recommended["model"]
    nu = sol["nu"]
    nubar = sol["nubar"]
    regions = np.arange(1, N_REGIONS + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [COLORS["peak"] if v > nubar else COLORS["primary"] for v in nu]
    ax.bar(regions, nu, 0.55, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(nubar, color=COLORS["secondary"], linestyle="--", linewidth=1.5, label=f"均值 ν̄={nubar:.4f}")

    for i in range(N_REGIONS):
        ax.text(regions[i], nu[i] + 0.0005, f"{nu[i]:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("区域编号")
    ax.set_ylabel("新增接入压力率 ν_i")
    ax.set_title(f"图3  最终方案（{model}）各区域新增接入压力率")
    ax.set_xticks(regions)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图3_新增接入压力率柱状图.png")
    plt.close(fig)


def plot_pressure_comparison(recommended_a: dict, recommended_b: dict) -> None:
    """图4：M-A 与 M-B 的压力分布对比折线图。"""
    setup_plot_style()
    ensure_dirs()

    nu_a = recommended_a["solution"]["nu"]
    nu_b = recommended_b["solution"]["nu"]
    regions = np.arange(1, N_REGIONS + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(regions, nu_a, "o-", color=COLORS["ma"], linewidth=2, markersize=6, label="M-A")
    ax.plot(regions, nu_b, "s--", color=COLORS["mb"], linewidth=2, markersize=6, label="M-B")

    ax.set_xlabel("区域编号")
    ax.set_ylabel("新增接入压力率 ν_i")
    ax.set_title("图4  M-A 与 M-B 各区域新增接入压力对比")
    ax.set_xticks(regions)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图4_压力分布对比折线图.png")
    plt.close(fig)


def plot_topsis_sensitivity(
    solutions: list[dict],
    model: str,
    entropy_idx: int,
    equal_idx: int,
    cost_idx: int,
) -> None:
    """图5：TOPSIS 权重敏感性对比图。"""
    setup_plot_style()
    ensure_dirs()

    if not solutions:
        return

    K = len(solutions)
    f1 = [s["f1"] for s in solutions]
    cc = [s["C_city"] for s in solutions]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(f1, cc, c=COLORS["neutral"], s=20, alpha=0.4, label="Pareto 候选")

    markers = [
        (entropy_idx, "熵权-TOPSIS", COLORS["peak"], "*"),
        (equal_idx, "等权-TOPSIS", COLORS["primary"], "D"),
        (cost_idx, "偏成本-TOPSIS", COLORS["tertiary"], "s"),
    ]
    for idx, label, color, marker in markers:
        if 0 <= idx < K:
            ax.scatter(f1[idx], cc[idx], c=color, s=150, marker=marker,
                       edgecolors="black", linewidth=1, zorder=5, label=label)

    ax.set_xlabel("总成本/万元")
    ax.set_ylabel("全市面积加权覆盖率")
    ax.set_title(f"图5  {model} TOPSIS 权重敏感性对比")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / f"图5_TOPSIS敏感性_{model}.png")
    plt.close(fig)
