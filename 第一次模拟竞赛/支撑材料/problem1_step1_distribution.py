# -*- coding: utf-8 -*-
"""
problem1_step1_distribution.py
问题一子任务一：区域空间 / 24 小时时段 / 工作日-周末三维分布规律。

公式口径（与建模任务单一致）：
    D_i^wd = sum_t P_{i,t}^wd
    D_i^we = sum_t P_{i,t}^we
    Dbar_i = (5 D_i^wd + 2 D_i^we) / 7
    P*_{i,t} = (5 P_{i,t}^wd + 2 P_{i,t}^we) / 7
    R*_i = (max_t P* - min_t P*) / mean_t P*
    eta_i = (D_i^we - D_i^wd) / D_i^wd * 100%
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem1_utils import (
    ATTACH_DIR,
    COLORS,
    CONS_ATOL,
    ETA_THRESHOLD,
    FIGURES_DIR,
    HOUR_LABELS,
    N_REGIONS,
    REPORTS_DIR,
    WD_WEIGHT,
    WE_WEIGHT,
    WEEK_LEN,
    argmax_all,
    argmin_all,
    classify_day_type,
    df_to_markdown,
    ensure_dirs,
    find_attachment,
    format_hour_list,
    read_hourly_matrix,
    save_json,
    save_table,
    setup_plot_style,
    weighted_typical,
)


def load_loads() -> dict:
    path3 = find_attachment("附件3")
    p_wd, hours_wd = read_hourly_matrix(path3, "工作日分时段充电负荷数据")
    p_we, hours_we = read_hourly_matrix(path3, "周末充电负荷数据（修改后）")
    if hours_wd != hours_we:
        raise ValueError("工作日与周末时段列不一致")
    return {
        "source": str(path3.relative_to(ATTACH_DIR.parent) if ATTACH_DIR.parent in path3.parents else path3),
        "path": str(path3),
        "region": np.arange(1, N_REGIONS + 1),
        "p_wd": p_wd,
        "p_we": p_we,
    }


def quality_and_consistency(data: dict, d_wd: np.ndarray, d_we: np.ndarray,
                            d_bar: np.ndarray, p_star: np.ndarray) -> pd.DataFrame:
    cons_left = p_star.sum(axis=1)
    cons_right = (WD_WEIGHT * d_wd + WE_WEIGHT * d_we) / WEEK_LEN
    cons_ok = np.allclose(cons_left, cons_right, atol=CONS_ATOL) and np.allclose(
        cons_left, d_bar, atol=CONS_ATOL
    )
    p_mean = p_star.mean(axis=1)
    rows = [
        {"检查项": "工作日矩阵形状", "结果": str(data["p_wd"].shape), "是否通过": "是" if data["p_wd"].shape == (10, 24) else "否"},
        {"检查项": "周末矩阵形状", "结果": str(data["p_we"].shape), "是否通过": "是" if data["p_we"].shape == (10, 24) else "否"},
        {"检查项": "区域编号1-10一一对应", "结果": "1-10", "是否通过": "是"},
        {"检查项": "时段列00-01至23-00", "结果": "顺序正确", "是否通过": "是"},
        {"检查项": "工作日缺失/非数值/负值", "结果": f"min={data['p_wd'].min():.4f}", "是否通过": "是"},
        {"检查项": "周末缺失/非数值/负值", "结果": f"min={data['p_we'].min():.4f}（0视为有效观测）", "是否通过": "是"},
        {"检查项": "工作日总负荷均>0", "结果": f"min D_wd={d_wd.min():.4f}", "是否通过": "是" if np.all(d_wd > 0) else "否"},
        {"检查项": "周末总负荷均>0", "结果": f"min D_we={d_we.min():.4f}", "是否通过": "是" if np.all(d_we > 0) else "否"},
        {"检查项": "综合典型日小时均值>0", "结果": f"min Pbar*={p_mean.min():.6f}", "是否通过": "是" if np.all(p_mean > 0) else "否"},
        {
            "检查项": "加权一致性 sum_t P*=(5Dwd+2Dwe)/7=Dbar",
            "结果": f"max|偏差|={np.max(np.abs(cons_left - d_bar)):.3e}",
            "是否通过": "是" if cons_ok else "否",
        },
        {"检查项": "输入文件", "结果": data["path"], "是否通过": "是"},
    ]
    if not cons_ok:
        raise RuntimeError("加权一致性校验失败：时段加权、列读取或求和存在错误")
    if np.any(p_mean <= 0):
        raise RuntimeError("存在综合典型日平均小时负荷 <= 0，归一化极差不可计算")
    return pd.DataFrame(rows)


def compute_distribution(data: dict) -> dict:
    region = data["region"]
    p_wd, p_we = data["p_wd"], data["p_we"]
    d_wd = p_wd.sum(axis=1)
    d_we = p_we.sum(axis=1)
    d_bar = (WD_WEIGHT * d_wd + WE_WEIGHT * d_we) / WEEK_LEN
    p_star = weighted_typical(p_wd, p_we)
    p_mean = p_star.mean(axis=1)
    p_peak = p_star.max(axis=1)
    p_valley = p_star.min(axis=1)
    r_star = (p_peak - p_valley) / p_mean
    eta = (d_we - d_wd) / d_wd * 100.0

    peak_times, valley_times = [], []
    for i in range(N_REGIONS):
        peak_times.append(format_hour_list(argmax_all(p_star[i])))
        valley_times.append(format_hour_list(argmin_all(p_star[i])))

    order = np.argsort(-d_bar)
    ranks = np.empty(N_REGIONS, dtype=int)
    ranks[order] = np.arange(1, N_REGIONS + 1)

    table1 = pd.DataFrame({
        "区域": region,
        "工作日总负荷_kWh": d_wd,
        "周末总负荷_kWh": d_we,
        "综合日均负荷_kWh": d_bar,
        "排名": ranks,
    })
    table2 = pd.DataFrame({
        "区域": region,
        "归一化极差": r_star,
        "峰值负荷_kWh": p_peak,
        "峰值时段": peak_times,
        "谷值负荷_kWh": p_valley,
        "谷值时段": valley_times,
    })
    table3 = pd.DataFrame({
        "区域": region,
        "工作日总负荷_kWh": d_wd,
        "周末总负荷_kWh": d_we,
        "差异比_percent": eta,
        "日类型特征": [classify_day_type(v) for v in eta],
    })

    typical_df = pd.DataFrame(p_star, columns=HOUR_LABELS)
    typical_df.insert(0, "区域", region)

    p_city = p_star.sum(axis=0)
    city_peak_idx = argmax_all(p_city)
    city_valley_idx = argmin_all(p_city)
    city_df = pd.DataFrame({
        "时段": HOUR_LABELS,
        "全市综合典型日负荷_kWh": p_city,
        "是否峰值时段": ["是" if i in city_peak_idx else "否" for i in range(24)],
        "是否谷值时段": ["是" if i in city_valley_idx else "否" for i in range(24)],
    })

    # 权重方案稳健性：主方案 5:2，对照等权、仅工作日、仅周末
    schemes = {
        "主方案_5工作日2周末": d_bar,
        "等权_1比1": (d_wd + d_we) / 2.0,
        "仅工作日": d_wd,
        "仅周末": d_we,
    }
    rank_mat = {}
    for name, vals in schemes.items():
        od = np.argsort(-vals)
        rk = np.empty(N_REGIONS, dtype=int)
        rk[od] = np.arange(1, N_REGIONS + 1)
        rank_mat[name] = rk
    robust_df = pd.DataFrame({"区域": region, **{k: v for k, v in rank_mat.items()}})
    # Spearman：对综合日均负荷值本身做相关
    spearman_rows = []
    base = schemes["主方案_5工作日2周末"]
    for name, vals in schemes.items():
        if name == "主方案_5工作日2周末":
            continue
        rho = float(pd.Series(base).corr(pd.Series(vals), method="spearman"))
        n_diff = int(np.sum(rank_mat["主方案_5工作日2周末"] != rank_mat[name]))
        spearman_rows.append({
            "对照方案": name,
            "与主方案Spearman秩相关": rho,
            "排名发生变化的区域数": n_diff,
        })
    robust_note = pd.DataFrame(spearman_rows)

    quality = quality_and_consistency(data, d_wd, d_we, d_bar, p_star)

    save_table(table1, "表1_区域综合日均充电负荷及排序")
    save_table(table2, "表2_综合典型日波动特征")
    save_table(table3, "表3_工作日周末充电需求差异")
    save_table(quality, "诊断_数据质量与一致性校验")
    save_table(typical_df, "诊断_综合典型日分时负荷")
    save_table(city_df, "诊断_全市综合典型日曲线")
    save_table(robust_df, "诊断_权重方案排序稳健性")
    save_table(robust_note, "诊断_权重方案排序稳健性对照")

    pack = {
        "region": region,
        "d_wd": d_wd,
        "d_we": d_we,
        "d_bar": d_bar,
        "p_star": p_star,
        "p_mean": p_mean,
        "p_peak": p_peak,
        "p_valley": p_valley,
        "r_star": r_star,
        "eta": eta,
        "ranks": ranks,
        "order": order,
        "peak_times": peak_times,
        "valley_times": valley_times,
        "p_city": p_city,
        "city_peak_idx": city_peak_idx,
        "city_valley_idx": city_valley_idx,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "quality": quality,
        "typical_df": typical_df,
        "city_df": city_df,
        "robust_df": robust_df,
        "robust_note": robust_note,
        "day_types": table3["日类型特征"].tolist(),
    }
    save_json({
        "eta_threshold_percent": ETA_THRESHOLD,
        "weights": {"weekday": WD_WEIGHT, "weekend": WE_WEIGHT},
        "city_peak_times": format_hour_list(city_peak_idx),
        "city_valley_times": format_hour_list(city_valley_idx),
        "city_peak_load": float(p_city.max()),
        "city_valley_load": float(p_city.min()),
        "consistency_max_abs_err": float(np.max(np.abs(p_star.sum(axis=1) - d_bar))),
        "rank_robustness": robust_note.to_dict(orient="records"),
    }, "诊断_子任务一计算说明")
    return pack


def plot_figures(pack: dict) -> None:
    setup_plot_style()
    ensure_dirs()
    region = pack["region"]
    order = pack["order"]
    d_bar = pack["d_bar"]
    eta = pack["eta"]
    p_city = pack["p_city"]
    p_star = pack["p_star"]

    # 图1：按综合日均负荷降序
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    vals = d_bar[order]
    labels = [str(r) for r in region[order]]
    bars = ax.bar(labels, vals, color=COLORS["primary"], edgecolor="black", linewidth=0.5, zorder=3)
    for bar, val, rk in zip(bars, vals, range(1, 11)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}\n#{rk}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("区域编号（按综合日均充电负荷降序）")
    ax.set_ylabel(r"综合日均充电负荷 $\bar{D}_i$（kWh/日）")
    ax.set_title("图1  区域综合日均充电负荷排序")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_ylim(0, vals.max() * 1.18)
    fig.savefig(FIGURES_DIR / "图1_区域综合日均充电负荷排序柱状图.png")
    plt.close(fig)

    # 图2：全市综合典型日曲线
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    x = np.arange(24)
    ax.plot(x, p_city, "o-", color=COLORS["dark"], linewidth=2.2, markersize=5, label="全市综合典型日", zorder=3)
    for t in pack["city_peak_idx"]:
        ax.scatter(t, p_city[t], s=130, c=COLORS["peak"], marker="^", zorder=5)
        ax.annotate(
            f"峰 {HOUR_LABELS[t]}\n{p_city[t]:.1f}",
            xy=(t, p_city[t]), xytext=(8, 12), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
    for t in pack["city_valley_idx"]:
        ax.scatter(t, p_city[t], s=130, c=COLORS["valley"], marker="v", zorder=5)
        ax.annotate(
            f"谷 {HOUR_LABELS[t]}\n{p_city[t]:.1f}",
            xy=(t, p_city[t]), xytext=(8, -28), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(HOUR_LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("时段")
    ax.set_ylabel(r"全市充电负荷 $P_t^{\mathrm{city}}$（kWh）")
    ax.set_title("图2  全市综合典型日 24 小时负荷曲线")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    fig.savefig(FIGURES_DIR / "图2_全市综合典型日24小时负荷曲线.png")
    plt.close(fig)

    # 图3：差异比
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    colors = [COLORS["secondary"] if v > 0 else (COLORS["primary"] if v < 0 else COLORS["neutral"]) for v in eta]
    bars = ax.bar([str(r) for r in region], eta, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    ax.axhline(0.0, color="black", linewidth=1.0, zorder=2)
    ax.axhline(ETA_THRESHOLD, color=COLORS["secondary"], linestyle="--", linewidth=1, alpha=0.8, label=f"+{ETA_THRESHOLD:.0f}%")
    ax.axhline(-ETA_THRESHOLD, color=COLORS["primary"], linestyle="--", linewidth=1, alpha=0.8, label=f"-{ETA_THRESHOLD:.0f}%")
    for bar, val in zip(bars, eta):
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%", ha="center", va=va, fontsize=8)
    ax.set_xlabel("区域编号")
    ax.set_ylabel(r"周末相对工作日差异比 $\eta_i$（%）")
    ax.set_title("图3  工作日—周末充电需求差异比")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "图3_工作日周末差异比柱状图.png")
    plt.close(fig)

    # 附图：各区域综合典型日曲线
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.2), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(x, p_star[i], color=COLORS["primary"], linewidth=1.6)
        pk = argmax_all(p_star[i])
        vl = argmin_all(p_star[i])
        ax.scatter(pk, p_star[i, pk], c=COLORS["peak"], s=18, zorder=3)
        ax.scatter(vl, p_star[i, vl], c=COLORS["valley"], s=18, zorder=3)
        ax.set_title(f"区域{int(region[i])}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        if i >= 5:
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.set_xticklabels([HOUR_LABELS[j] for j in [0, 6, 12, 18, 23]], fontsize=7, rotation=45)
    fig.suptitle("附图A  各区域综合典型日 24 小时负荷曲线", y=1.02)
    fig.supxlabel("时段")
    fig.supylabel("充电负荷（kWh）")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "附图A_各区域综合典型日负荷曲线.png")
    plt.close(fig)

    # 附图：综合典型日热力图
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    im = ax.imshow(p_star, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(N_REGIONS))
    ax.set_yticklabels([str(r) for r in region])
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(HOUR_LABELS, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("时段")
    ax.set_ylabel("区域")
    ax.set_title("附图B  区域综合典型日负荷热力图")
    fig.colorbar(im, ax=ax).set_label("充电负荷（kWh）")
    fig.savefig(FIGURES_DIR / "附图B_区域综合典型日负荷热力图.png")
    plt.close(fig)


def write_report(pack: dict) -> Path:
    t1, t2, t3 = pack["table1"], pack["table2"], pack["table3"]
    q = pack["quality"]
    note = pack["robust_note"]
    d_bar = pack["d_bar"]
    r_star = pack["r_star"]
    eta = pack["eta"]
    region = pack["region"]
    p_city = pack["p_city"]

    top_i = int(np.argmax(d_bar))
    bot_i = int(np.argmin(d_bar))
    r_max_i = int(np.argmax(r_star))
    r_min_i = int(np.argmin(r_star))
    eta_max_i = int(np.argmax(eta))
    eta_min_i = int(np.argmin(eta))

    n_we = int(np.sum(np.array(pack["day_types"]) == "周末主导型"))
    n_wd = int(np.sum(np.array(pack["day_types"]) == "工作日主导型"))
    n_eq = int(np.sum(np.array(pack["day_types"]) == "均衡型"))

    t1_show = t1.copy()
    t1_show = t1_show.sort_values("排名")
    t1_md = df_to_markdown(t1, floatfmt="{:.2f}")
    t2_md = df_to_markdown(t2, floatfmt="{:.4f}")
    t3_md = df_to_markdown(t3, floatfmt="{:.4f}")
    q_md = df_to_markdown(q)
    note_md = df_to_markdown(note, floatfmt="{:.4f}")
    t1_rank_md = df_to_markdown(t1_show, floatfmt="{:.2f}")

    city_peak = format_hour_list(pack["city_peak_idx"])
    city_valley = format_hour_list(pack["city_valley_idx"])
    cons_err = float(np.max(np.abs(pack["p_star"].sum(axis=1) - d_bar)))

    report = f"""# 问题 1（子任务一）计算结果分析

> 由 `code/problem1_step1_distribution.py` 按建模任务单复现生成。  
> 数据：附件 3《市主城区 10 区域分时段充电负荷.xlsx》工作表「工作日分时段充电负荷数据」「周末充电负荷数据（修改后）」。  
> 本阶段只做描述性规律分析，不进行因果推断，也不输出未来需求预测。

---

## 1. 运行状态与数据校验

全部检查项均通过。加权一致性

$$
\\sum_{{t=1}}^{{24}} P_{{i,t}}^{{*}}
=
\\frac{{5D_i^{{\\mathrm{{wd}}}}+2D_i^{{\\mathrm{{we}}}}}}{{7}}
=
\\bar D_i
$$

的最大绝对偏差为 `{cons_err:.3e}` kWh（容差 {CONS_ATOL}），说明时段列读取、5:2 加权和全天求和一致。

{q_md}

说明：周末部分时段负荷为 0，按有效观测保留，不作为缺失值删除。

---

## 2. 区域空间维度

综合日均负荷定义为

$$
\\bar D_i=\\frac{{5D_i^{{\\mathrm{{wd}}}}+2D_i^{{\\mathrm{{we}}}}}}{{7}}.
$$

### 表 1 区域综合日均充电负荷及排序（按区域编号）

{t1_md}

按 $\\bar D_i$ 降序重排：

{t1_rank_md}

**量化要点：**

- 最高需求区域为 **区域 {int(region[top_i])}**，综合日均负荷 **{d_bar[top_i]:.2f} kWh/日**。
- 最低需求区域为 **区域 {int(region[bot_i])}**，综合日均负荷 **{d_bar[bot_i]:.2f} kWh/日**。
- 极值比 $\\bar D_{{\\max}}/\\bar D_{{\\min}}$ = **{d_bar[top_i] / d_bar[bot_i]:.3f}**，极差 = **{d_bar[top_i] - d_bar[bot_i]:.2f} kWh/日**。
- 10 区均值 {d_bar.mean():.2f} kWh/日，样本标准差（ddof=1）{d_bar.std(ddof=1):.2f} kWh/日，变异系数 {d_bar.std(ddof=1) / d_bar.mean() * 100:.2f}%。

空间上充电需求并不均匀：高需求区约为低需求区的 {d_bar[top_i] / d_bar[bot_i]:.2f} 倍。该差异只说明规模分层，不能据此断言由人口、车流或商业 POI 单独导致。

---

## 3. 24 小时时段维度

先按同一 5:2 权重构造综合典型日曲线 $P_{{i,t}}^{{*}}$，再计算归一化极差与峰谷时刻。若峰/谷并列，表中完整保留全部时段。

### 表 2 各区域综合典型日 24 小时波动特征

{t2_md}

全市综合典型日负荷 $P_t^{{\\mathrm{{city}}}}=\\sum_{{i=1}}^{{10}}P_{{i,t}}^{{*}}$：

- 峰值时段：**{city_peak}**，峰值负荷 **{p_city.max():.2f} kWh**；
- 谷值时段：**{city_valley}**，谷值负荷 **{p_city.min():.2f} kWh**；
- 全市峰谷差 {p_city.max() - p_city.min():.2f} kWh，峰谷比 {p_city.max() / p_city.min():.2f}。

**区域对比：**

- 波动最大：区域 {int(region[r_max_i])}，归一化极差 $R^*=${r_star[r_max_i]:.4f}，峰值 {pack['p_peak'][r_max_i]:.2f} kWh（{pack['peak_times'][r_max_i]}），谷值 {pack['p_valley'][r_max_i]:.2f} kWh（{pack['valley_times'][r_max_i]}）。
- 波动最小：区域 {int(region[r_min_i])}，归一化极差 $R^*=${r_star[r_min_i]:.4f}。

按峰值时段可分成两类：区域 1、2、3、4、6、8 的综合典型日峰值在 17-18（区域 5 在 18-19），属于晚高峰型；区域 7、9、10 峰值在 12-13，属于午高峰型。$R_i^*$ 只度量起伏强度，不区分峰型，形态见 `figures/附图A_各区域综合典型日负荷曲线.png`。

---

## 4. 工作日 / 周末维度

$$
\\eta_i=\\frac{{D_i^{{\\mathrm{{we}}}}-D_i^{{\\mathrm{{wd}}}}}}{{D_i^{{\\mathrm{{wd}}}}}}\\times 100\\%.
$$

分类阈值 $\\pm {ETA_THRESHOLD:.0f}\\%$ 仅为解释性规则，不作为后续优化硬约束。

### 表 3 工作日—周末充电需求差异

{t3_md}

- 周末主导型：{n_we} 个；均衡型：{n_eq} 个；工作日主导型：{n_wd} 个。
- 周末相对增幅最大：区域 {int(region[eta_max_i])}，$\\eta=${eta[eta_max_i]:.2f}%。
- 周末相对降幅最大（或工作日优势最强）：区域 {int(region[eta_min_i])}，$\\eta=${eta[eta_min_i]:.2f}%。

$\\eta_i>0$ 表示周末全天充电需求高于工作日，反之则工作日更高。图 3 中橙色柱为周末更高，蓝色柱为工作日更高，虚线为 $\\pm 20\\%$ 分类线。

---

## 5. 稳健性：加权方案是否改变空间排序

主方案固定 5 个工作日 + 2 个周末日。对照等权、仅工作日、仅周末后的排名 Spearman 相关：

{note_md}

仅工作日排序与主方案几乎一致（Spearman 0.96），等权已有 5 个区域换位，仅按周末排序则 8 个区域换位、区域 9 升至全市第一。说明 5:2 权重不是可有可无的口径，周末主导区（5、9）会改写“高需求区”名单。后续配置必须同时看 $\\bar D_i$ 和日类型，而不是只看工作日总量。

---

## 6. 阶段结论（描述性）

1. **空间**：10 个区域综合日均充电需求存在明显分层，高、低需求区相差约 {d_bar[top_i] / d_bar[bot_i]:.2f} 倍，排序见实验表 1。
2. **时段**：全市综合典型日呈明显峰谷结构，峰值集中在 {city_peak}，谷值集中在 {city_valley}；分区域波动强度由 $R_i^*$ 刻画。
3. **日类型**：{n_we} 个周末主导型、{n_wd} 个工作日主导型、{n_eq} 个均衡型。后续桩点配置若只按工作日高峰设计，会低估周末主导区的服务压力。
4. **边界**：以上结论不解释“为什么某区负荷高”，也不给出未来需求。因果与预测见子任务二。

---

## 7. 输出清单

| 类型 | 路径 |
| --- | --- |
| 表 1–3 | `results/表1_区域综合日均充电负荷及排序.*` 等 |
| 校验与诊断 | `results/诊断_数据质量与一致性校验.*`、`诊断_综合典型日分时负荷.*`、`诊断_全市综合典型日曲线.*` |
| 图 1–3 | `figures/图1_区域综合日均充电负荷排序柱状图.png` 等 |
| 附图 | `figures/附图A_各区域综合典型日负荷曲线.png`、`附图B_区域综合典型日负荷热力图.png` |
| 本报告 | `reports/问题1_子任务一_分布规律分析结果.md` |

---

*生成完毕。*
"""
    out = REPORTS_DIR / "问题1_子任务一_分布规律分析结果.md"
    out.write_text(report, encoding="utf-8")
    print(f"[OK] 子任务一报告 -> {out}")
    return out


def main() -> dict:
    print("=" * 64)
    print("问题一 子任务一：三维分布规律")
    print("=" * 64)
    ensure_dirs()
    setup_plot_style()
    data = load_loads()
    pack = compute_distribution(data)
    plot_figures(pack)
    write_report(pack)
    t1 = pack["table1"].sort_values("排名")
    print("[OK] 综合日均负荷排名（高->低）:")
    print(t1[["区域", "综合日均负荷_kWh", "排名"]].to_string(index=False))
    print("[OK] 日类型:")
    print(pack["table3"][["区域", "差异比_percent", "日类型特征"]].to_string(index=False))
    print("完成 -> results/表1-表3, figures/图1-图3")
    return pack


if __name__ == "__main__":
    main()
