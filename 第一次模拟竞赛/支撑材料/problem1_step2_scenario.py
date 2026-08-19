# -*- coding: utf-8 -*-
"""
problem1_step2_scenario.py
问题一子任务二接续：未来短期综合日均需求情景预测。

假设人口密度、车流量、商业 POI 短期不变，故三因素潜势倍率 = 1。
下一规划年度需求按新能源汽车保有量外生增长率缩放：
    Dhat_{i,1}^{(g)} = D_{i,0} * (1 + g)
    g in {10%, 15%, 20%}
基准 D_{i,0} 使用 2025 年实际综合日均负荷，不用岭回归折外预测值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem1_utils import (
    COLORS,
    FIGURES_DIR,
    N_REGIONS,
    REPORTS_DIR,
    WD_WEIGHT,
    WE_WEIGHT,
    WEEK_LEN,
    df_to_markdown,
    ensure_dirs,
    find_attachment,
    read_hourly_matrix,
    save_json,
    save_table,
    setup_plot_style,
)

SCENARIOS = [
    ("conservative", 0.10, "保守情景_10percent"),
    ("base", 0.15, "基准情景_15percent"),
    ("accel", 0.20, "加速情景_20percent"),
]
CHECK_ATOL = 1e-6


def load_base_demand() -> dict:
    path3 = find_attachment("附件3")
    p_wd, _ = read_hourly_matrix(path3, "工作日分时段充电负荷数据")
    p_we, _ = read_hourly_matrix(path3, "周末充电负荷数据（修改后）")
    d_wd = p_wd.sum(axis=1)
    d_we = p_we.sum(axis=1)
    d0 = (WD_WEIGHT * d_wd + WE_WEIGHT * d_we) / WEEK_LEN
    return {
        "region": np.arange(1, N_REGIONS + 1),
        "d_wd": d_wd,
        "d_we": d_we,
        "d0": d0,
        "path3": str(path3),
    }


def compute_scenarios(d0: np.ndarray) -> dict[str, np.ndarray]:
    return {name: d0 * (1.0 + g) for name, g, _ in SCENARIOS}


def check_consistency(d0: np.ndarray, pred: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    city0 = float(d0.sum())
    all_ok = True
    for name, g, label in SCENARIOS:
        yhat = pred[name]
        region_err = float(np.max(np.abs(yhat - d0 * (1.0 + g))))
        city_err = float(abs(yhat.sum() - (1.0 + g) * city0))
        ok = region_err <= CHECK_ATOL and city_err <= CHECK_ATOL
        all_ok = all_ok and ok
        rows.append({
            "检查项": f"{label} 区域倍率 D0*(1+g)",
            "最大绝对偏差": region_err,
            "容差": CHECK_ATOL,
            "是否通过": "是" if region_err <= CHECK_ATOL else "否",
        })
        rows.append({
            "检查项": f"{label} 全市加总 (1+g)*sum D0",
            "最大绝对偏差": city_err,
            "容差": CHECK_ATOL,
            "是否通过": "是" if city_err <= CHECK_ATOL else "否",
        })
    if not all_ok:
        raise RuntimeError("情景预测一致性校验失败")
    return pd.DataFrame(rows)


def plot_scenarios(region: np.ndarray, d0: np.ndarray, pred: dict[str, np.ndarray]) -> None:
    setup_plot_style()
    ensure_dirs()
    x = np.arange(N_REGIONS)
    w = 0.20
    series = [
        (d0, "2025 基准", COLORS["dark"]),
        (pred["conservative"], "10% 保守", COLORS["primary"]),
        (pred["base"], "15% 基准", COLORS["tertiary"]),
        (pred["accel"], "20% 加速", COLORS["secondary"]),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for off, (vals, lab, col) in zip(offsets, series):
        ax.bar(x + off * w, vals, w, label=lab, color=col, edgecolor="black", linewidth=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in region])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("综合日均充电需求（kWh/日）")
    ax.set_title("图7  未来短期综合日均需求情景预测")
    ax.legend(ncol=4, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图7_未来短期需求情景预测.png")
    plt.close(fig)

    # 全市总量对照
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    labels = ["2025基准", "10%保守", "15%基准", "20%加速"]
    city = [float(d0.sum()), float(pred["conservative"].sum()),
            float(pred["base"].sum()), float(pred["accel"].sum())]
    colors = [COLORS["dark"], COLORS["primary"], COLORS["tertiary"], COLORS["secondary"]]
    bars = ax.bar(labels, city, color=colors, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, city):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("全市综合日均充电需求（kWh/日）")
    ax.set_title("附图D  全市短期需求情景总量")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "附图D_全市短期需求情景总量.png")
    plt.close(fig)


def write_report(table: pd.DataFrame, check: pd.DataFrame, d0: np.ndarray,
                 pred: dict[str, np.ndarray]) -> Path:
    t_md = df_to_markdown(table, floatfmt="{:.2f}")
    c_md = df_to_markdown(check, floatfmt="{:.3e}")
    city0 = float(d0.sum())
    inc = {
        "10%": float(pred["conservative"].sum() - city0),
        "15%": float(pred["base"].sum() - city0),
        "20%": float(pred["accel"].sum() - city0),
    }
    top_i = int(np.argmax(d0))
    bot_i = int(np.argmin(d0))
    report = f"""# 问题 1（子任务二）未来短期需求情景预测结果

> 由 `code/problem1_step2_scenario.py` 按计算任务单生成。  
> 基准 $D_{{i,0}}$ 为附件 3 工作日/周末 5:2 加权综合日均负荷，**不是**岭回归折外预测。  
> 假设人口密度、车流量、商业 POI 短期不变，三因素潜势倍率 $f(x_{{i,1}})/f(x_{{i,0}})=1$。  
> 外生参数：新能源汽车保有量年增长率 $g\\in\\{{10\\%,15\\%,20\\%\\}}$，且新增车辆的空间占比、公共充电频率与单次电量不变。

---

## 1. 计算公式

$$
D_{{i,0}}=\\frac{{5D_{{i,0}}^{{\\mathrm{{wd}}}}+2D_{{i,0}}^{{\\mathrm{{we}}}}}}{{7}},
\\qquad
\\widehat D_{{i,1}}^{{(g)}}=D_{{i,0}}(1+g).
$$

全市：

$$
D_{{\\mathrm{{city}},1}}^{{(g)}}=(1+g)\\sum_{{i=1}}^{{10}}D_{{i,0}}.
$$

---

## 2. 预测表

{t_md}

全市相对 2025 基准的增量：保守 +{inc['10%']:.2f}、基准 +{inc['15%']:.2f}、加速 +{inc['20%']:.2f} kWh/日。

区域排序在三种情景下与 2025 基准完全相同（最高区域 {top_i + 1}，最低区域 {bot_i + 1}），因为各区使用同一增长率。

---

## 3. 一致性校验

容差 $10^{{-6}}$。全部通过。

{c_md}

另：加权基准与既有表 1 的综合日均负荷应一致（实现上由同一公式、同一附件 3 重新计算）。

---

## 4. 解释边界

1. 这是题设保有量增长率的**敏感性分析**，不是从 10 个区域横截面里识别出的时间趋势。
2. 未改动人口密度、车流量、POI，也未改动区域间需求份额。
3. 不得把本表的 $\\widehat D_{{i,1}}^{{(g)}}$ 理解成岭回归“学到了增长”；岭回归只用于 2025 年横截面解释。
4. 问题 2 若需一张需求输入表，默认采用 **15% 基准情景**；10%、20% 作扩容压力的上下界。
5. 若后续放宽“单车充电强度不变”，应另建情景，不能在本表上再乘一次系数。

---

## 5. 输出清单

| 类型 | 路径 |
| --- | --- |
| 预测表 | `results/表7_未来短期综合日均需求情景预测.*` |
| 校验 | `results/诊断_情景预测一致性校验.*` |
| 图 7 | `figures/图7_未来短期需求情景预测.png` |
| 全市总量 | `figures/附图D_全市短期需求情景总量.png` |
| 本报告 | `reports/问题1_子任务二_未来短期需求情景预测结果.md` |

---

*生成完毕。*
"""
    out = REPORTS_DIR / "问题1_子任务二_未来短期需求情景预测结果.md"
    out.write_text(report, encoding="utf-8")
    print(f"[OK] 情景预测报告 -> {out}")
    return out


def main() -> dict:
    print("=" * 64)
    print("问题一 子任务二接续：未来短期需求情景预测")
    print("=" * 64)
    ensure_dirs()
    setup_plot_style()

    data = load_base_demand()
    region, d0 = data["region"], data["d0"]
    pred = compute_scenarios(d0)
    check = check_consistency(d0, pred)

    rows = []
    for i, r in enumerate(region):
        rows.append({
            "区域": int(r),
            "2025基准需求_kWh": d0[i],
            "保守情景_10percent": pred["conservative"][i],
            "基准情景_15percent": pred["base"][i],
            "加速情景_20percent": pred["accel"][i],
        })
    rows.append({
        "区域": "全市",
        "2025基准需求_kWh": float(d0.sum()),
        "保守情景_10percent": float(pred["conservative"].sum()),
        "基准情景_15percent": float(pred["base"].sum()),
        "加速情景_20percent": float(pred["accel"].sum()),
    })
    table = pd.DataFrame(rows)
    save_table(table, "表7_未来短期综合日均需求情景预测")
    save_table(check, "诊断_情景预测一致性校验")
    save_json({
        "formula": "Dhat_i(g) = D_i0 * (1+g)",
        "baseline": "2025 actual weighted daily demand, not ridge OOF",
        "feature_potential_ratio": 1.0,
        "growth_rates": [0.10, 0.15, 0.20],
        "city_base": float(d0.sum()),
        "city_10": float(pred["conservative"].sum()),
        "city_15": float(pred["base"].sum()),
        "city_20": float(pred["accel"].sum()),
        "check_atol": CHECK_ATOL,
        "source": data["path3"],
        "assumption": "pop/traffic/POI fixed; spatial share and per-vehicle public charging intensity fixed",
    }, "诊断_情景预测说明")

    plot_scenarios(region, d0, pred)
    write_report(table, check, d0, pred)

    print("[OK] 全市 2025 / 10% / 15% / 20%:")
    print(f"     {d0.sum():.2f} / {pred['conservative'].sum():.2f} / "
          f"{pred['base'].sum():.2f} / {pred['accel'].sum():.2f}")
    print("完成 -> results/表7, figures/图7")
    return {"table": table, "pred": pred, "d0": d0}


if __name__ == "__main__":
    main()
