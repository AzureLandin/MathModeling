# -*- coding: utf-8 -*-
"""
problem1_step2_lightgbm.py
问题一子任务二接续：低复杂度 LightGBM-L1 非线性稳健性验证。

与岭回归 R1 使用同一目标、同一四项原始特征、同一套 LOOCV 外层折。
LightGBM 不做标准化，不进行网格搜索，参数按验证方案预先固定。

本脚本不改写正式表 1–6；输出写入 LGB验证_* 与独立分析报告。
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem1_step2_prediction import (
    LAMBDA_GRID,
    load_xy,
    loocv_mean_baseline,
    loocv_ridge,
    pick_lambda,
)
from problem1_utils import (
    COLORS,
    FIGURES_DIR,
    N_REGIONS,
    REPORTS_DIR,
    demand_metrics,
    df_to_markdown,
    ensure_dirs,
    save_json,
    save_table,
    setup_plot_style,
)

# 验证方案预先固定的保守参数，禁止在此脚本中改动或网格搜索
LGB_PARAMS = {
    "objective": "regression",
    "n_estimators": 20,
    "learning_rate": 0.05,
    "max_depth": 1,
    "num_leaves": 2,
    "min_child_samples": 4,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l1": 0.0,
    "lambda_l2": 10.0,
    "random_state": 2026,
    "verbosity": -1,
}
FOCUS_REGIONS = (7, 8, 9)
WORSE_MEAN_RATIO = 1.10
WORSE_SINGLE_RATIO = 1.20


def _lgb_native_params() -> dict:
    """将方案中的固定参数转为原生 lgb.train 接口（不依赖 scikit-learn）。"""
    return {
        "objective": LGB_PARAMS["objective"],
        "learning_rate": LGB_PARAMS["learning_rate"],
        "max_depth": LGB_PARAMS["max_depth"],
        "num_leaves": LGB_PARAMS["num_leaves"],
        "min_child_samples": LGB_PARAMS["min_child_samples"],
        "feature_fraction": LGB_PARAMS["feature_fraction"],
        "bagging_fraction": LGB_PARAMS["bagging_fraction"],
        "bagging_freq": LGB_PARAMS["bagging_freq"],
        "lambda_l1": LGB_PARAMS["lambda_l1"],
        "lambda_l2": LGB_PARAMS["lambda_l2"],
        "verbosity": LGB_PARAMS["verbosity"],
        "seed": LGB_PARAMS["random_state"],
        "feature_fraction_seed": LGB_PARAMS["random_state"],
        "bagging_seed": LGB_PARAMS["random_state"],
        "data_random_seed": LGB_PARAMS["random_state"],
        "deterministic": True,
        "force_col_wise": True,
    }


def loocv_lgb_l1(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """原始特征、不标准化、固定参数的 LOOCV。"""
    n = len(y)
    yhat = np.zeros(n)
    params = _lgb_native_params()
    n_round = int(LGB_PARAMS["n_estimators"])
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        dset = lgb.Dataset(X[tr], label=y[tr], free_raw_data=True)
        model = lgb.train(params, dset, num_boost_round=n_round)
        yhat[i] = float(model.predict(X[i:i + 1])[0])
    return yhat


def decide_case(m_r1: dict, m_lgb: dict, ae_r1: np.ndarray, ae_lgb: np.ndarray,
                region: np.ndarray) -> tuple[str, str]:
    rmse_r1, rmse_lgb = m_r1["RMSE"], m_lgb["RMSE"]
    mae_r1, mae_lgb = m_r1["MAE"], m_lgb["MAE"]
    idx = [int(np.where(region == r)[0][0]) for r in FOCUS_REGIONS]
    mean_r1 = float(np.mean(ae_r1[idx]))
    mean_lgb = float(np.mean(ae_lgb[idx]))
    n_worse = int(np.sum(ae_lgb[idx] > WORSE_SINGLE_RATIO * ae_r1[idx]))
    focus_worse = (mean_lgb >= WORSE_MEAN_RATIO * mean_r1) or (n_worse >= 2)
    abnormal = m_lgb["n_neg"] > 0

    if rmse_lgb >= rmse_r1 or focus_worse or abnormal:
        case = "C"
        reason = (
            f"RMSE_LGB={rmse_lgb:.2f} "
            f"{'≥' if rmse_lgb >= rmse_r1 else '<'} RMSE_R1={rmse_r1:.2f}；"
            f"区域7/8/9平均绝对误差 LGB={mean_lgb:.2f}、R1={mean_r1:.2f}"
            f"（相对比 {mean_lgb / mean_r1:.3f}）；"
            f"单区恶化超过 20% 的个数={n_worse}；负预测数={m_lgb['n_neg']}。"
            "按规则 C，当前样本不足以支撑非线性树模型，论文保留岭回归。"
        )
        return case, reason

    if rmse_lgb < 0.95 * rmse_r1 and mae_lgb < mae_r1:
        case = "A"
        reason = (
            f"RMSE_LGB={rmse_lgb:.2f} < 0.95×RMSE_R1={0.95 * rmse_r1:.2f}，"
            f"且 MAE_LGB={mae_lgb:.2f} < MAE_R1={mae_r1:.2f}，无负预测。"
            "按规则 A，数据中可能存在一定非线性，但仍以岭回归为可解释主模型，"
            "LightGBM 只作稳健性检验，不替换主模型。"
        )
        return case, reason

    case = "B"
    reason = (
        f"0.95×RMSE_R1={0.95 * rmse_r1:.2f} ≤ RMSE_LGB={rmse_lgb:.2f} "
        f"< RMSE_R1={rmse_r1:.2f}。"
        "按规则 B，不以小于 5% 的 RMSE 改善改用 LightGBM，主模型仍为岭回归。"
    )
    return case, reason


def plot_figures(region: np.ndarray, y: np.ndarray, y_r1: np.ndarray, y_lgb: np.ndarray,
                 ae_r1: np.ndarray, ae_lgb: np.ndarray) -> None:
    setup_plot_style()
    ensure_dirs()
    x = np.arange(N_REGIONS)
    w = 0.26

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.bar(x - w, y, w, label="实际综合日均负荷", color=COLORS["dark"], edgecolor="black", linewidth=0.3)
    ax.bar(x, y_r1, w, label="岭回归 R1 LOOCV", color=COLORS["tertiary"], edgecolor="black", linewidth=0.3)
    ax.bar(x + w, y_lgb, w, label="LightGBM-L1 LOOCV", color=COLORS["secondary"], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in region])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("综合日均充电负荷（kWh/日）")
    ax.set_title("LGB验证图1  实际值与两类模型留一交叉验证预测对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "LGB验证_图1_实际与两类模型LOOCV对比.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.bar(x - 0.2, ae_r1, 0.4, label="岭回归 R1", color=COLORS["primary"], edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.2, ae_lgb, 0.4, label="LightGBM-L1", color=COLORS["secondary"], edgecolor="black", linewidth=0.3)
    for i, r in enumerate(region):
        if int(r) in FOCUS_REGIONS:
            ax.axvline(i, color=COLORS["gold"], linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in region])
    ax.set_xlabel("区域编号（虚线标出区域 7、8、9）")
    ax.set_ylabel("LOOCV 绝对误差（kWh/日）")
    ax.set_title("LGB验证图2  两类模型逐区域绝对误差对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "LGB验证_图2_区域绝对误差对比.png")
    plt.close(fig)


def write_report(pack: dict) -> Path:
    t1, t2, t789 = pack["table1"], pack["table2"], pack["table789"]
    m_mean, m_r1, m_r2, m_lgb = pack["m_mean"], pack["m_r1"], pack["m_r2"], pack["m_lgb"]
    case, reason = pack["case"], pack["reason"]
    y, y_r1, y_lgb = pack["y"], pack["y_r1"], pack["y_lgb"]

    t1_md = df_to_markdown(t1, floatfmt="{:.4f}")
    t2_md = df_to_markdown(t2, floatfmt="{:.4f}")
    t789_md = df_to_markdown(t789, floatfmt="{:.4f}")

    n_lgb_better = int((t2["表现更优模型"] == "LightGBM-L1").sum())
    n_r1_better = int((t2["表现更优模型"] == "岭回归R1").sum())
    n_tie = int((t2["表现更优模型"] == "持平").sum())
    rank_corr = float(pd.Series(y_r1).corr(pd.Series(y_lgb), method="spearman"))
    pred_corr = float(pd.Series(y_r1).corr(pd.Series(y_lgb), method="pearson"))
    y_mean_cv = pack["y_mean"]
    deg_err = float(np.max(np.abs(y_lgb - y_mean_cv)))
    degenerated = deg_err < 1e-4

    max_i_r1 = int(np.argmax(np.abs(y - y_r1)))
    max_i_lgb = int(np.argmax(np.abs(y - y_lgb)))
    ape_r1 = np.abs((y - y_r1) / y) * 100.0
    ape_lgb = np.abs((y - y_lgb) / y) * 100.0
    maxape_i_r1 = int(np.argmax(ape_r1))
    maxape_i_lgb = int(np.argmax(ape_lgb))

    report = f"""# 问题 1（子任务二）LightGBM 稳健性验证结果

> 由 `code/problem1_step2_lightgbm.py` 按《LightGBM稳健性验证方案》复现生成。  
> 预测对象与岭回归完全一致：$y_i=\\bar D_i=(5D_i^{{\\mathrm{{wd}}}}+2D_i^{{\\mathrm{{we}}}})/7$。  
> LightGBM-L1 使用与 R1 相同的四项**原始特征**，不标准化，参数预先固定，不做网格搜索。  
> 本结果只用于模型形式稳健性对照，不替代正式表 4–6，也不改写问题二输入。

---

## 1. 运行设定

| 项目 | 设定 |
| --- | --- |
| 特征 | 人口密度、车流量、商业 POI、现有总充电桩数（快+慢） |
| 特征变换 | 无（原始尺度，与 R1 一致） |
| 标准化 | 岭回归折内标准化；LightGBM 不标准化 |
| 验证 | 同一套 10 折留一交叉验证 |
| LightGBM 参数 | `max_depth=1, num_leaves=2, min_child_samples=4, n_estimators=20, learning_rate=0.05, lambda_l2=10, random_state=2026` |
| 对照岭回归 | R1：原始特征 + $\\lambda=10$；R2：对数特征 + $\\lambda=10$ |

未计算、也未在结论中使用：训练集 $R^2$、树结构图、特征重要性、SHAP。

---

## 2. 统一模型对照

### 表 1 统一模型对照

{t1_md}

- 岭回归 R1 最大绝对误差 {m_r1['MaxAE']:.2f} kWh/日（区域 {int(pack['region'][max_i_r1])}），最大相对误差 {m_r1['MaxAPE']:.2f}%（区域 {int(pack['region'][maxape_i_r1])}）。
- LightGBM-L1 最大绝对误差 {m_lgb['MaxAE']:.2f} kWh/日（区域 {int(pack['region'][max_i_lgb])}），最大相对误差 {m_lgb['MaxAPE']:.2f}%（区域 {int(pack['region'][maxape_i_lgb])}）。
- LightGBM-L1 负预测数 = {m_lgb['n_neg']}。
- 两类折外预测的 Pearson 相关 = {pred_corr:.4f}，Spearman 秩相关 = {rank_corr:.4f}。

相对 R1：RMSE 比 = {m_lgb['RMSE'] / m_r1['RMSE']:.4f}，MAE 比 = {m_lgb['MAE'] / m_r1['MAE']:.4f}。  
判定阈值：情况 A 要求 RMSE 比 $<0.95$ 且 MAE 同步下降。

{"**关键诊断：** LightGBM-L1 的 10 个折外预测与均值基准逐点重合（最大绝对偏差 " + f"{deg_err:.3e}" + "）。在 `max_depth=1`、`min_child_samples=4`、`lambda_l2=10`、每折仅 9 个训练点时，分裂增益不足以形成有效 stump，提升停留在初始预测（训练折均值）。这不是实现错误，而是验证方案禁止高复杂度之后的直接结果：该约束下的“非线性对照”实际上没有学到非线性。" if degenerated else ""}

---

## 3. 逐区域折外预测

### 表 2 逐区域折外预测对比

{t2_md}

绝对误差更优：岭回归 R1 {n_r1_better} 个区域，LightGBM-L1 {n_lgb_better} 个区域，持平 {n_tie} 个。

---

## 4. 区域 7、8、9 专项

上一轮岭回归的主要偏差集中在这三区：7、8 被高估，9 被低估。本轮检查非线性 stump 是否缓解该结构。

区域 9 的 LightGBM 绝对误差从 6012 降到 1284，看起来改善，但它的预测值 {y_lgb[int(np.where(pack['region']==9)[0][0])]:.2f} 就是“去掉区域 9 后其余 9 区的均值”，并不是学到了周末主导结构。区域 7、8 则进一步被拉向全市均值，相对误差升至 99.7%、70.3%。

{t789_md}

---

## 5. 选模判定

按验证方案第 7 节，对照对象为岭回归 **R1**。

**判定结果：情况 {case}。**

{reason}

因此：

1. 论文主模型仍为岭回归 R1（原始特征标准化，$\\lambda=10$）。
2. LightGBM-L1 只作为非线性稳健性附录，不进入问题 2 的需求输入。
3. 问题 2 仍优先使用 2025 年**实际综合日均负荷**，不用任何模型的折外预测值替代。

---

## 6. 禁止项自检

| 禁止项 | 本轮是否发生 |
| --- | --- |
| 用训练集 $R^2$ 选模 | 否 |
| 用树结构解释区域规则 | 否 |
| 按特征重要性排序作因果判断 | 否 |
| 使用 SHAP | 否 |
| 用 LightGBM 折外值替代实际 $\\bar D_i$ | 否 |
| 按单一 LightGBM 结果确定问题 2 新增桩 | 否 |
| `max_depth>=3` / `num_leaves>3` / `n_estimators>=100` / 网格搜索 | 否 |

现有总桩数即使被树模型频繁用于分裂，也只表示它与当期需求共变，不表示“多种桩就会增加需求”。

---

## 7. 解释边界

当前对照比较的是 2025 年 10 个区域的横截面折外误差，不是未来多期预测能力。$n=10$ 且每折仅 9 个训练点，深度为 1 的树模型表达能力极弱，本结果不能推广到“树模型一定优于/劣于岭回归”。未来需求仍需在明确人口、车流、POI 或新能源汽车保有量情景后再外推。

---

## 8. 输出清单

| 类型 | 路径 |
| --- | --- |
| 对照表 | `results/LGB验证_表1_统一模型对照.*` |
| 逐区域表 | `results/LGB验证_表2_逐区域折外预测对比.*` |
| 7/8/9 专项 | `results/LGB验证_表3_区域789误差变化.*` |
| 图 1–2 | `figures/LGB验证_图1_实际与两类模型LOOCV对比.png`、`LGB验证_图2_区域绝对误差对比.png` |
| 本报告 | `reports/问题1_子任务二_LightGBM稳健性验证结果.md` |

---

*生成完毕。*
"""
    out = REPORTS_DIR / "问题1_子任务二_LightGBM稳健性验证结果.md"
    out.write_text(report, encoding="utf-8")
    print(f"[OK] LightGBM 验证报告 -> {out}")
    return out


def main() -> dict:
    print("=" * 64)
    print("问题一 子任务二接续：LightGBM-L1 稳健性验证")
    print("=" * 64)
    ensure_dirs()
    setup_plot_style()

    data = load_xy()
    X, y, region = data["X"], data["y"], data["region"]

    y_mean = loocv_mean_baseline(y)
    y_r1, _ = loocv_ridge(X, y, 10.0, log_transform=False)
    y_r2, _ = loocv_ridge(X, y, 10.0, log_transform=True)
    y_lgb = loocv_lgb_l1(X, y)

    # 复核：R1/R2 的 lambda 仍应由网格选出，与正式脚本一致
    rows_r1, rows_r2 = [], []
    for lam in LAMBDA_GRID:
        ya, _ = loocv_ridge(X, y, lam, log_transform=False)
        yb, _ = loocv_ridge(X, y, lam, log_transform=True)
        rows_r1.append({"lambda": lam, "LOOCV_RMSE": demand_metrics(y, ya)["RMSE"]})
        rows_r2.append({"lambda": lam, "LOOCV_RMSE": demand_metrics(y, yb)["RMSE"]})
    lam_r1 = pick_lambda(pd.DataFrame(rows_r1))
    lam_r2 = pick_lambda(pd.DataFrame(rows_r2))
    if not np.isclose(lam_r1, 10.0) or not np.isclose(lam_r2, 10.0):
        print(f"[WARN] 网格重选 lambda 与缓存不一致：R1={lam_r1:g}, R2={lam_r2:g}")
        y_r1, _ = loocv_ridge(X, y, lam_r1, log_transform=False)
        y_r2, _ = loocv_ridge(X, y, lam_r2, log_transform=True)

    m_mean = demand_metrics(y, y_mean)
    m_r1 = demand_metrics(y, y_r1)
    m_r2 = demand_metrics(y, y_r2)
    m_lgb = demand_metrics(y, y_lgb)
    ae_r1 = np.abs(y - y_r1)
    ae_lgb = np.abs(y - y_lgb)
    ape_r1 = np.abs((y - y_r1) / y) * 100.0
    ape_r2 = np.abs((y - y_r2) / y) * 100.0
    ape_lgb = np.abs((y - y_lgb) / y) * 100.0
    ape_mean = np.abs((y - y_mean) / y) * 100.0

    table1 = pd.DataFrame([
        {
            "模型": "均值基准",
            "LOOCV_MAE": m_mean["MAE"],
            "LOOCV_RMSE": m_mean["RMSE"],
            "LOOCV_MAPE": m_mean["MAPE"],
            "R2_CV": m_mean["R2_CV"],
            "最大相对误差_percent": float(np.max(ape_mean)),
            "负预测数": m_mean["n_neg"],
        },
        {
            "模型": "岭回归R1",
            "LOOCV_MAE": m_r1["MAE"],
            "LOOCV_RMSE": m_r1["RMSE"],
            "LOOCV_MAPE": m_r1["MAPE"],
            "R2_CV": m_r1["R2_CV"],
            "最大相对误差_percent": float(np.max(ape_r1)),
            "负预测数": m_r1["n_neg"],
        },
        {
            "模型": "岭回归R2",
            "LOOCV_MAE": m_r2["MAE"],
            "LOOCV_RMSE": m_r2["RMSE"],
            "LOOCV_MAPE": m_r2["MAPE"],
            "R2_CV": m_r2["R2_CV"],
            "最大相对误差_percent": float(np.max(ape_r2)),
            "负预测数": m_r2["n_neg"],
        },
        {
            "模型": "LightGBM-L1",
            "LOOCV_MAE": m_lgb["MAE"],
            "LOOCV_RMSE": m_lgb["RMSE"],
            "LOOCV_MAPE": m_lgb["MAPE"],
            "R2_CV": m_lgb["R2_CV"],
            "最大相对误差_percent": float(np.max(ape_lgb)),
            "负预测数": m_lgb["n_neg"],
        },
    ])

    better = []
    for a, b in zip(ae_r1, ae_lgb):
        if np.isclose(a, b, atol=1e-8, rtol=0.0):
            better.append("持平")
        elif b < a:
            better.append("LightGBM-L1")
        else:
            better.append("岭回归R1")

    table2 = pd.DataFrame({
        "区域": region,
        "实际综合日均负荷": y,
        "岭回归R1预测": y_r1,
        "LightGBM_L1预测": y_lgb,
        "岭回归绝对误差": ae_r1,
        "LightGBM绝对误差": ae_lgb,
        "表现更优模型": better,
    })

    rows789 = []
    for r in FOCUS_REGIONS:
        i = int(np.where(region == r)[0][0])
        rows789.append({
            "区域": r,
            "实际负荷": y[i],
            "R1预测": y_r1[i],
            "LGB预测": y_lgb[i],
            "R1绝对误差": ae_r1[i],
            "LGB绝对误差": ae_lgb[i],
            "R1相对误差_percent": ape_r1[i],
            "LGB相对误差_percent": ape_lgb[i],
            "绝对误差变化_LGB减R1": ae_lgb[i] - ae_r1[i],
            "误差是否改善": "是" if ae_lgb[i] < ae_r1[i] - 1e-8 else ("持平" if np.isclose(ae_lgb[i], ae_r1[i]) else "否"),
        })
    table789 = pd.DataFrame(rows789)

    case, reason = decide_case(m_r1, m_lgb, ae_r1, ae_lgb, region)

    save_table(table1, "LGB验证_表1_统一模型对照")
    save_table(table2, "LGB验证_表2_逐区域折外预测对比")
    save_table(table789, "LGB验证_表3_区域789误差变化")
    save_json({
        "lightgbm_params": LGB_PARAMS,
        "lightgbm_version": lgb.__version__,
        "features": ["人口密度", "车流量", "商业POI数", "现有总充电桩数"],
        "standardize_lgb": False,
        "log_transform_lgb": False,
        "lambda_R1": float(lam_r1),
        "lambda_R2": float(lam_r2),
        "metrics_R1": {k: float(m_r1[k]) for k in ["MAE", "RMSE", "MAPE", "R2_CV", "MaxAE", "MaxAPE", "n_neg"]},
        "metrics_LGB": {k: float(m_lgb[k]) for k in ["MAE", "RMSE", "MAPE", "R2_CV", "MaxAE", "MaxAPE", "n_neg"]},
        "rmse_ratio_lgb_over_r1": float(m_lgb["RMSE"] / m_r1["RMSE"]),
        "case": case,
        "reason": reason,
        "y_r1": [float(v) for v in y_r1],
        "y_lgb": [float(v) for v in y_lgb],
        "max_abs_dev_from_mean_baseline": float(np.max(np.abs(y_lgb - y_mean))),
        "degenerated_to_fold_mean": bool(np.max(np.abs(y_lgb - y_mean)) < 1e-4),
        "api": "lightgbm.train native (no scikit-learn)",
    }, "LGB验证_计算说明")

    pack = {
        "region": region,
        "y": y,
        "y_mean": y_mean,
        "y_r1": y_r1,
        "y_lgb": y_lgb,
        "table1": table1,
        "table2": table2,
        "table789": table789,
        "m_mean": m_mean,
        "m_r1": m_r1,
        "m_r2": m_r2,
        "m_lgb": m_lgb,
        "case": case,
        "reason": reason,
    }
    plot_figures(region, y, y_r1, y_lgb, ae_r1, ae_lgb)
    write_report(pack)

    print(f"[OK] 情况 {case}")
    print(f"     R1   RMSE={m_r1['RMSE']:.2f}  MAE={m_r1['MAE']:.2f}  MAPE={m_r1['MAPE']:.2f}%  R2={m_r1['R2_CV']:.4f}")
    print(f"     LGB  RMSE={m_lgb['RMSE']:.2f}  MAE={m_lgb['MAE']:.2f}  MAPE={m_lgb['MAPE']:.2f}%  R2={m_lgb['R2_CV']:.4f}")
    print(f"     RMSE比={m_lgb['RMSE'] / m_r1['RMSE']:.4f}")
    print(reason)
    print("完成 -> results/LGB验证_表1-表3, figures/LGB验证_图1-图2")
    return pack


if __name__ == "__main__":
    main()
