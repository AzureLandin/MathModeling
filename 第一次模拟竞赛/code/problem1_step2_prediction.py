# -*- coding: utf-8 -*-
"""
problem1_step2_prediction.py
问题一子任务二：多因素综合日均充电负荷预测。

主模型：对数变换 + 折内标准化 + 岭回归 + 留一交叉验证。
对照：原始特征岭回归（R1）、可选 LightGBM（低复杂度）。

岭回归按任务单闭式解实现，截距不惩罚：
    先对训练折计算 mu, sigma，得到 Z；
    y 中心化后
    beta = (Z^T Z + lambda I)^{-1} Z^T y_c
    beta0 = mean(y_train)
标准化使用总体标准差 ddof=0（与常见 StandardScaler 一致），
mu/sigma 仅由当前训练折估计，避免泄漏。
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
    demand_metrics,
    df_to_markdown,
    ensure_dirs,
    find_attachment,
    read_hourly_matrix,
    read_region_features,
    save_json,
    save_table,
    setup_plot_style,
)

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
FEATURE_NAMES = ["人口密度", "车流量", "商业POI数", "现有总充电桩数"]
FEATURE_KEYS = ["x1_pop_density", "x2_traffic", "x3_poi", "x4_n_total"]
MECHANISM_SIGN = {
    "人口密度": "+",
    "车流量": "+",
    "商业POI数": "+",
    "现有总充电桩数": "+",
}
MECHANISM_NOTE = {
    "人口密度": "人口越密，潜在充电用户通常越多，预期系数为正。",
    "车流量": "过境与到访车辆越多，公共充电机会通常越多，预期系数为正。",
    "商业POI数": "商业活动越密集，停留充电需求通常越高，预期系数为正。",
    "现有总充电桩数": "现有桩数与当期需求往往同向（供给跟随或诱导需求），预期系数为正；该变量存在内生性，不能解释为“多种桩就会增加需求”。",
}


def load_xy() -> dict:
    path1 = find_attachment("附件 1")
    path3 = find_attachment("附件3")
    df1 = read_region_features(path1)
    p_wd, _ = read_hourly_matrix(path3, "工作日分时段充电负荷数据")
    p_we, _ = read_hourly_matrix(path3, "周末充电负荷数据（修改后）")
    d_wd = p_wd.sum(axis=1)
    d_we = p_we.sum(axis=1)
    y = (WD_WEIGHT * d_wd + WE_WEIGHT * d_we) / WEEK_LEN

    pop = df1.iloc[:, 3].astype(float).to_numpy()
    traffic = df1.iloc[:, 4].astype(float).to_numpy()
    poi = df1.iloc[:, 5].astype(float).to_numpy()
    n_fast = df1.iloc[:, 7].astype(float).to_numpy()
    n_slow = df1.iloc[:, 8].astype(float).to_numpy()
    n_total = n_fast + n_slow
    n_listed = df1.iloc[:, 6].astype(float).to_numpy()
    if not np.allclose(n_total, n_listed, atol=1e-8):
        raise ValueError("快充+慢充与“现有充电桩数量”不一致")
    if np.any(np.stack([pop, traffic, poi, n_fast, n_slow]) < 0):
        raise ValueError("特征存在负值")

    X = np.column_stack([pop, traffic, poi, n_total])
    feat_df = pd.DataFrame({
        "区域": np.arange(1, N_REGIONS + 1),
        "y_综合日均负荷": y,
        "人口密度": pop,
        "车流量": traffic,
        "商业POI数": poi,
        "快充桩数": n_fast,
        "慢充桩数": n_slow,
        "现有总充电桩数": n_total,
    })
    return {
        "region": np.arange(1, N_REGIONS + 1),
        "y": y,
        "X": X,
        "feat_df": feat_df,
        "path1": str(path1),
        "path3": str(path3),
    }


def standardize_train(X_tr: np.ndarray, X_te: np.ndarray | None = None, ddof: int = 0):
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0, ddof=ddof)
    sigma = np.where(np.abs(sigma) < 1e-12, 1.0, sigma)
    z_tr = (X_tr - mu) / sigma
    if X_te is None:
        return z_tr, mu, sigma
    return z_tr, (X_te - mu) / sigma, mu, sigma


def fit_ridge_closed(Z: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """截距不惩罚的岭回归闭式解。Z 为已标准化特征。"""
    y = np.asarray(y, dtype=float)
    beta0 = float(y.mean())
    y_c = y - beta0
    p = Z.shape[1]
    a = Z.T @ Z + lam * np.eye(p)
    b = Z.T @ y_c
    beta = np.linalg.solve(a, b)
    return beta0, beta


def predict_ridge(beta0: float, beta: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return beta0 + Z @ beta


def loocv_ridge(X: np.ndarray, y: np.ndarray, lam: float, log_transform: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    n = len(y)
    yhat = np.zeros(n)
    coefs = []
    X_use = np.log(X + 1.0) if log_transform else X.copy()
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        z_tr, z_te, _, _ = standardize_train(X_use[tr], X_use[i:i + 1])
        beta0, beta = fit_ridge_closed(z_tr, y[tr], lam)
        yhat[i] = float(predict_ridge(beta0, beta, z_te)[0])
        coefs.append(beta)
    return yhat, coefs


def loocv_mean_baseline(y: np.ndarray) -> np.ndarray:
    n = len(y)
    yhat = np.zeros(n)
    for i in range(n):
        yhat[i] = float(np.mean(np.delete(y, i)))
    return yhat


def loocv_lgb(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if not HAS_LGB:
        raise RuntimeError("未安装 lightgbm")
    n = len(y)
    yhat = np.zeros(n)
    X_use = np.log(X + 1.0)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        model = lgb.LGBMRegressor(
            max_depth=2,
            num_leaves=3,
            min_child_samples=3,
            n_estimators=50,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_use[tr], y[tr])
        yhat[i] = float(model.predict(X_use[i:i + 1])[0])
    return yhat


def pearson_corr(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)


def vif_scores(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    out = np.zeros(p)
    for j in range(p):
        yj = X[:, j]
        z = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        coef, *_ = np.linalg.lstsq(z, yj, rcond=None)
        yhat = z @ coef
        ss_res = float(np.sum((yj - yhat) ** 2))
        ss_tot = float(np.sum((yj - yj.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[j] = float(1.0 / (1.0 - r2)) if r2 < 1.0 - 1e-12 else float("inf")
    return out


def pick_lambda(grid_df: pd.DataFrame) -> float:
    """以 LOOCV-RMSE 最小为准；若并列取更大 lambda（更强正则，小样本更稳）。"""
    min_rmse = float(grid_df["LOOCV_RMSE"].min())
    close = grid_df[np.isclose(grid_df["LOOCV_RMSE"], min_rmse, atol=1e-12)]
    return float(close["lambda"].max())


def fit_full(X: np.ndarray, y: np.ndarray, lam: float, log_transform: bool):
    X_use = np.log(X + 1.0) if log_transform else X.copy()
    z, mu, sigma = standardize_train(X_use)
    beta0, beta = fit_ridge_closed(z, y, lam)
    yhat = predict_ridge(beta0, beta, z)
    return {
        "beta0": beta0,
        "beta": beta,
        "mu": mu,
        "sigma": sigma,
        "yhat_insample": yhat,
        "X_use": X_use,
        "Z": z,
    }


def sign_text(v: float) -> str:
    if v > 0:
        return "正"
    if v < 0:
        return "负"
    return "零"


def interpret_coef(name: str, v: float) -> str:
    expected = MECHANISM_SIGN[name]
    actual = "+" if v > 0 else ("-" if v < 0 else "0")
    match = "与机理预期同向" if actual == expected else "与机理预期不符"
    return f"{MECHANISM_NOTE[name]} 标准化系数={v:.4f}（{sign_text(v)}），{match}。"


def compute_collinearity(X: np.ndarray) -> pd.DataFrame:
    X_log = np.log(X + 1.0)
    rows = []
    for tag, mat in (("原始", X), ("对数", X_log)):
        c = pearson_corr(mat)
        vif = vif_scores(mat)
        z, _, _ = standardize_train(mat)
        cond = float(np.linalg.cond(z))
        for i, name in enumerate(FEATURE_NAMES):
            rows.append({
                "特征尺度": tag,
                "特征": name,
                "VIF": vif[i],
                "与人口密度相关": c[i, 0],
                "与车流量相关": c[i, 1],
                "与商业POI相关": c[i, 2],
                "与总桩数相关": c[i, 3],
                "标准化设计阵条件数": cond,
            })
    return pd.DataFrame(rows)


def plot_prediction(pack: dict) -> None:
    setup_plot_style()
    ensure_dirs()
    region = pack["region"]
    y = pack["y"]
    best = pack["best"]
    yhat = pack["yhat_best"]

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    x = np.arange(N_REGIONS)
    w = 0.38
    ax.bar(x - w / 2, y, w, label="实际综合日均负荷", color=COLORS["dark"], edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, yhat, w, label="LOOCV 预测", color=COLORS["tertiary"], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in region])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("综合日均充电负荷（kWh/日）")
    ax.set_title(f"图4  实际值与{best['模型']}留一交叉验证预测对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图4_实际与LOOCV预测对比.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    g1, g2 = pack["grid_r1"], pack["grid_r2"]
    ax.plot(g1["lambda"], g1["LOOCV_RMSE"], "o-", color=COLORS["primary"], linewidth=2, label="R1 RMSE（原始特征）")
    ax.plot(g2["lambda"], g2["LOOCV_RMSE"], "s--", color=COLORS["secondary"], linewidth=1.8, label="R2 RMSE（对数特征）")
    ax.set_xscale("log")
    ax.set_xlabel(r"正则化参数 $\lambda$")
    ax.set_ylabel("LOOCV-RMSE（kWh/日）")
    ax.set_title("图5  岭回归 $\\lambda$ 灵敏度（留一交叉验证）")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.savefig(FIGURES_DIR / "图5_岭回归lambda灵敏度.png")
    plt.close(fig)

    ape = np.abs((y - yhat) / y) * 100.0
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    colors = [COLORS["secondary"] if v == ape.max() else COLORS["primary"] for v in ape]
    ax.bar([str(r) for r in region], ape, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("区域编号")
    ax.set_ylabel("LOOCV 相对误差（%）")
    ax.set_title("图6  逐区域留一交叉验证相对误差")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图6_逐区域LOOCV相对误差.png")
    plt.close(fig)

    # 相关热力图（对数特征）
    X_log = np.log(pack["X"] + 1.0)
    corr = pearson_corr(X_log)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(FEATURE_NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("附图C  对数特征 Pearson 相关")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIGURES_DIR / "附图C_对数特征相关系数热力图.png")
    plt.close(fig)


def write_report(pack: dict) -> Path:
    t4, t5, t6 = pack["table4"], pack["table5"], pack["table6"]
    best = pack["best"]
    y = pack["y"]
    yhat = pack["yhat_best"]
    m = pack["metrics_best"]
    grid = pack["grid_best"]
    ablation = pack["ablation"]
    stab = pack["stability"]
    colin = pack["colin"]

    t4_md = df_to_markdown(t4, floatfmt="{:.4f}")
    t5_md = df_to_markdown(t5, floatfmt="{:.4f}")
    t6_md = df_to_markdown(t6, floatfmt="{:.4f}")
    grid_md = df_to_markdown(grid, floatfmt="{:.4f}")
    ab_md = df_to_markdown(ablation, floatfmt="{:.4f}")
    st_md = df_to_markdown(stab)
    colin_md = df_to_markdown(colin, floatfmt="{:.4f}")

    max_i = int(np.argmax(np.abs(y - yhat)))
    r2_note = (
        "留一交叉验证 $R^2_{\\mathrm{CV}}$ 为正，说明折外预测优于“预测训练集均值”。"
        if m["R2_CV"] > 0
        else "留一交叉验证 $R^2_{\\mathrm{CV}}$ 为负，说明折外预测整体不优于均值基准；小样本下该结果应视为过拟合或特征外推失败的信号，不得用样本内拟合优度替代。"
    )
    rec = pack["recommendation"]

    report = f"""# 问题 1（子任务二）计算结果分析

> 由 `code/problem1_step2_prediction.py` 按建模任务单复现生成。  
> 目标：$y_i=\\bar D_i=(5D_i^{{\\mathrm{{wd}}}}+2D_i^{{\\mathrm{{we}}}})/7$，单位 kWh/日。  
> 特征：人口密度、车流量、商业 POI 数、现有总充电桩数（快充+慢充）。不把总桩数与快/慢充同时入模。  
> 验证：仅使用留一交叉验证（LOOCV）。标准化均值与标准差只在训练折上计算。

---

## 1. 相关性与共线性

{colin_md}

VIF $>10$ 或标准化设计阵条件数很大时，普通最小二乘系数会不稳定，这是采用岭回归的直接理由。对数变换后若相关结构仍然很强，正则化仍然必要。

---

## 2. 模型性能比较

候选 $\\lambda\\in\\{{0.01,0.1,1,10,100\\}}$，按该模型自身的 LOOCV-RMSE 选参。均值基准用于判断“模型是否真的学到了跨区域差异”。

### 表 4 模型性能比较

{t4_md}

**入选主模型：{best['模型']}**（特征变换 = {best['特征变换']}，参数 = {best['最优参数']}）。

选择依据：在 R1/R2 中优先 LOOCV-RMSE 更低者；若 R2 同时系数更稳（见第 4 节），则与任务单默认推荐一致，采用对数特征岭回归。LightGBM 仅作低复杂度非线性对照，不用训练集 $R^2$ 评判。

{r2_note}

主模型在 $\\lambda$ 网格上的 LOOCV：

{grid_md}

---

## 3. 逐区域交叉验证预测

### 表 5 实际值与交叉验证预测值

{t5_md}

- LOOCV-MAE = {m['MAE']:.2f} kWh/日，RMSE = {m['RMSE']:.2f}，MAPE = {m['MAPE']:.2f}%，$R^2_{{\\mathrm{{CV}}}}$ = {m['R2_CV']:.4f}。
- 最大绝对误差出现在 **区域 {int(pack['region'][max_i])}**（{m['MaxAE']:.2f} kWh/日，相对误差 {m['MaxAPE']:.2f}%）。
- 负预测个数：{m['n_neg']}。

高误差区（相对误差 $\\ge 40\\%$）见上表。区域 7、8 的四项规模特征偏低，模型把它们往全市均值拉，倾向高估；区域 9 的人口密度、车流量、POI 和桩数都低，但周末负荷接近工作日的两倍，静态特征无法刻画这种日类型结构，倾向低估。后续若把折外点直接当作“已校准需求”送入问题二，会扭曲这些区域的配置。问题二建议优先使用 **实际综合日均负荷**，全样本拟合值仅作对照基准。

---

## 4. 最终模型回归系数

全样本拟合仅用于报告方向与方程，**不参与选模**。标准化系数对应“入模特征经 Z-score 后提高 1 个标准差”的边际关联。主模型为 {best['模型']}，入模特征为{' $\\ln(x+1)$ ' if pack['log_best'] else '原始尺度 '}后再标准化。

### 表 6 最终模型回归系数

{t6_md}

截距 $\\hat\\beta_0$ = {pack['full']['beta0']:.4f}。

方程（{best['模型']}，全样本）：

$$
\\hat y_i
=
{pack['full']['beta0']:.4f}
{''.join(f"{'+' if b >= 0 else '-'} {abs(b):.4f}\\,z_{{i{j+1}}}" for j, b in enumerate(pack['full']['beta']))}
$$

其中 $z_{{ij}}$ 为{' $\\ln(x_{ij}+1)$ ' if pack['log_best'] else '原始特征 $x_{ij}$ '}经全样本标准化后的值。

---

## 5. 稳定性检查

{st_md}

去掉现有总桩数后的对照（同一变换、同一 $\\lambda$ 网格重选）：

{ab_md}

解读规则（与任务单第 9 节对应）：

1. 若不同 $\\lambda$ 下 RMSE 变化平缓，则选参不过度敏感；若 0.01 与 100 之间指标剧烈跳动，应报告小样本不稳定性。
2. 去掉总桩数后若误差明显变差，说明该变量贡献了拟合能力，但因其内生性，解释时应谨慎。
3. 系数符号与“人口/车流/POI 越高需求越高”的基本机理对照见表 6。
4. 出现负预测则主模型不具备直接交付给规划模块的可行性。
5. 最大误差是否集中于单一区域，见表 5。
6. 若启用 LightGBM，其区域排序与岭回归的 Spearman 相关过低，则非线性对照与线性主模型冲突，不能只看某一方。

---

## 6. 推荐结论

{rec}

当前 $\\hat y_i$ 是 **2025 年横截面需求水平的估计**，不是未来三年的外推。任务单明确：若短期未来特征值未知，不擅自引入增长率；未来增长在问题 4 中结合新能源汽车保有量另行修正。

---

## 7. 输出清单

| 类型 | 路径 |
| --- | --- |
| 表 4–6 | `results/表4_模型性能比较.*` 等 |
| 诊断 | `results/诊断_特征相关与共线性.*`、`诊断_岭回归lambda网格LOOCV.*`、`诊断_去总桩数消融.*`、`诊断_稳定性检查.*` |
| 图 4–6 | `figures/图4_实际与LOOCV预测对比.png` 等 |
| 本报告 | `reports/问题1_子任务二_多因素预测分析结果.md` |

---

*生成完毕。*
"""
    out = REPORTS_DIR / "问题1_子任务二_多因素预测分析结果.md"
    out.write_text(report, encoding="utf-8")
    print(f"[OK] 子任务二报告 -> {out}")
    return out


def build_recommendation(table4: pd.DataFrame, metrics: dict, table6: pd.DataFrame,
                         has_lgb: bool, rank_corr: float | None) -> str:
    r1 = table4[table4["模型"] == "R1"].iloc[0]
    r2 = table4[table4["模型"] == "R2"].iloc[0]
    lines = []
    if r2["LOOCV_RMSE"] <= r1["LOOCV_RMSE"]:
        lines.append(
            f"R2（对数特征）的 LOOCV-RMSE={r2['LOOCV_RMSE']:.2f}，不高于 R1 的 {r1['LOOCV_RMSE']:.2f}，"
            f"按任务单采用对数变换 + 标准化 + 岭回归。"
        )
    else:
        lines.append(
            f"R1（原始特征）的 LOOCV-RMSE={r1['LOOCV_RMSE']:.2f} 低于 R2 的 {r2['LOOCV_RMSE']:.2f}。"
            f"任务单默认推荐 R2，但交叉验证误差显示原始尺度更稳；正文主结果仍报告两者，最终交付以误差更低者为准，并在论文中同时给出对照。"
        )
    if metrics["R2_CV"] < 0:
        lines.append("主模型 $R^2_{CV}<0$，不能把它当作高精度预测器，只可作为有正则约束的横截面平滑估计，并在问题二中做敏感性讨论。")
    if metrics["n_neg"] > 0:
        lines.append("出现负预测，不得直接把折外预测值送入后续优化模型。")
    signs_ok = all(
        (row["标准化回归系数"] > 0 and MECHANISM_SIGN[row["特征"]] == "+")
        or (row["标准化回归系数"] < 0 and MECHANISM_SIGN[row["特征"]] == "-")
        or row["标准化回归系数"] == 0
        for _, row in table6.iterrows()
    )
    if signs_ok:
        lines.append("四个标准化系数符号均与基本机理同向。")
    else:
        lines.append("存在与基本机理反向的系数，解释时应强调这是小样本偏相关，而不是政策效应。")
    if has_lgb and rank_corr is not None:
        if rank_corr < 0.6:
            lines.append(f"LightGBM 与主模型预测排序 Spearman={rank_corr:.3f}，存在明显冲突，不以树模型替换岭回归。")
        else:
            lines.append(f"LightGBM 与主模型预测排序 Spearman={rank_corr:.3f}，排序大体一致。")
    elif not has_lgb:
        lines.append("当前运行环境未安装 lightgbm，非线性对照未计算；主结论不依赖该对照。")
    return " ".join(lines)


def main() -> dict:
    print("=" * 64)
    print("问题一 子任务二：多因素岭回归预测")
    print("=" * 64)
    ensure_dirs()
    setup_plot_style()
    data = load_xy()
    X, y, region = data["X"], data["y"], data["region"]
    save_table(data["feat_df"], "诊断_预测特征与目标")

    colin = compute_collinearity(X)
    save_table(colin, "诊断_特征相关与共线性")

    # 网格：R1 / R2
    rows_r1, rows_r2 = [], []
    cv_r1, cv_r2 = {}, {}
    for lam in LAMBDA_GRID:
        y1, _ = loocv_ridge(X, y, lam, log_transform=False)
        y2, coefs2 = loocv_ridge(X, y, lam, log_transform=True)
        cv_r1[lam], cv_r2[lam] = y1, y2
        m1, m2 = demand_metrics(y, y1), demand_metrics(y, y2)
        coef_std = float(np.std(np.vstack(coefs2), axis=0).mean())
        rows_r1.append({"lambda": lam, **{f"LOOCV_{k}": m1[k] for k in ["MAE", "RMSE", "MAPE", "R2_CV"]}})
        rows_r2.append({
            "lambda": lam,
            **{f"LOOCV_{k}": m2[k] for k in ["MAE", "RMSE", "MAPE", "R2_CV"]},
            "折间系数平均标准差": coef_std,
            "负预测数": m2["n_neg"],
        })
    grid_r1 = pd.DataFrame(rows_r1)
    grid_r2 = pd.DataFrame(rows_r2)
    save_table(grid_r1, "诊断_岭回归R1_lambda网格")
    save_table(grid_r2, "诊断_岭回归lambda网格LOOCV")

    lam_r1 = pick_lambda(grid_r1)
    lam_r2 = pick_lambda(grid_r2)
    y_r1, y_r2 = cv_r1[lam_r1], cv_r2[lam_r2]
    m_r1, m_r2 = demand_metrics(y, y_r1), demand_metrics(y, y_r2)
    y_mean = loocv_mean_baseline(y)
    m_mean = demand_metrics(y, y_mean)

    table4_rows = [
        {
            "模型": "均值基准",
            "特征变换": "无",
            "最优参数": "训练折均值",
            "LOOCV_MAE": m_mean["MAE"],
            "LOOCV_RMSE": m_mean["RMSE"],
            "LOOCV_MAPE": m_mean["MAPE"],
            "R2_CV": m_mean["R2_CV"],
        },
        {
            "模型": "R1",
            "特征变换": "原始特征标准化",
            "最优参数": f"lambda={lam_r1:g}",
            "LOOCV_MAE": m_r1["MAE"],
            "LOOCV_RMSE": m_r1["RMSE"],
            "LOOCV_MAPE": m_r1["MAPE"],
            "R2_CV": m_r1["R2_CV"],
        },
        {
            "模型": "R2",
            "特征变换": "对数特征标准化",
            "最优参数": f"lambda={lam_r2:g}",
            "LOOCV_MAE": m_r2["MAE"],
            "LOOCV_RMSE": m_r2["RMSE"],
            "LOOCV_MAPE": m_r2["MAPE"],
            "R2_CV": m_r2["R2_CV"],
        },
    ]

    # 正式 LightGBM-L1 对照改由 problem1_step2_lightgbm.py 按验证方案固定参数执行，
    # 避免此处用另一套超参写入表4。
    y_lgb = None
    m_lgb = None
    rank_corr = None

    table4 = pd.DataFrame(table4_rows)

    # 主模型：R1/R2 中 RMSE 更低者；并列取 R2（任务单默认）
    if m_r2["RMSE"] <= m_r1["RMSE"]:
        best_name, best_row, yhat_best, m_best, lam_best, log_best = "R2", table4[table4["模型"] == "R2"].iloc[0], y_r2, m_r2, lam_r2, True
    else:
        best_name, best_row, yhat_best, m_best, lam_best, log_best = "R1", table4[table4["模型"] == "R1"].iloc[0], y_r1, m_r1, lam_r1, False

    table5 = pd.DataFrame({
        "区域": region,
        "实际综合日均负荷": y,
        "LOOCV预测值": yhat_best,
        "绝对误差": np.abs(y - yhat_best),
        "相对误差_percent": np.abs((y - yhat_best) / y) * 100.0,
    })

    full = fit_full(X, y, lam_best, log_transform=log_best)
    table6 = pd.DataFrame({
        "特征": FEATURE_NAMES,
        "标准化回归系数": full["beta"],
        "系数正负": [sign_text(v) for v in full["beta"]],
        "解释": [interpret_coef(n, v) for n, v in zip(FEATURE_NAMES, full["beta"])],
    })

    # 消融：去掉总桩数
    X3 = X[:, :3]
    grid_ab = []
    cv_ab = {}
    for lam in LAMBDA_GRID:
        ya, _ = loocv_ridge(X3, y, lam, log_transform=log_best)
        cv_ab[lam] = ya
        ma = demand_metrics(y, ya)
        grid_ab.append({"lambda": lam, "LOOCV_RMSE": ma["RMSE"], "LOOCV_MAE": ma["MAE"], "LOOCV_MAPE": ma["MAPE"], "R2_CV": ma["R2_CV"]})
    grid_ab = pd.DataFrame(grid_ab)
    lam_ab = pick_lambda(grid_ab)
    y_ab = cv_ab[lam_ab]
    m_ab = demand_metrics(y, y_ab)
    ablation = pd.DataFrame([
        {"方案": "四特征主模型", "lambda": lam_best, "LOOCV_RMSE": m_best["RMSE"], "LOOCV_MAE": m_best["MAE"], "LOOCV_MAPE": m_best["MAPE"], "R2_CV": m_best["R2_CV"]},
        {"方案": "去掉现有总桩数", "lambda": lam_ab, "LOOCV_RMSE": m_ab["RMSE"], "LOOCV_MAE": m_ab["MAE"], "LOOCV_MAPE": m_ab["MAPE"], "R2_CV": m_ab["R2_CV"]},
    ])
    save_table(ablation, "诊断_去总桩数消融")

    grid_best = grid_r2 if log_best else grid_r1
    rmse_vals = grid_best["LOOCV_RMSE"].to_numpy()
    rel_swing = float((rmse_vals.max() - rmse_vals.min()) / rmse_vals.min() * 100.0)

    baseline = pd.DataFrame({
        "区域": region,
        "实际综合日均负荷": y,
        "LOOCV预测值": yhat_best,
        "全样本拟合值_短期基准": full["yhat_insample"],
        "样本内绝对误差": np.abs(y - full["yhat_insample"]),
    })
    save_table(baseline, "诊断_短期基准需求")
    max_i = int(np.argmax(np.abs(y - yhat_best)))
    coef_fold_std = None
    _, coefs_best = loocv_ridge(X, y, lam_best, log_transform=log_best)
    coef_fold_std = np.std(np.vstack(coefs_best), axis=0)

    stab = pd.DataFrame([
        {"检查项": "不同lambda的RMSE相对摆动", "结果": f"{rel_swing:.2f}%", "判定": "平缓" if rel_swing < 30 else "较敏感"},
        {"检查项": "去掉总桩数后RMSE变化", "结果": f"{m_ab['RMSE'] - m_best['RMSE']:+.2f}", "判定": "明显变差" if m_ab["RMSE"] > m_best["RMSE"] * 1.05 else "变化有限"},
        {"检查项": "系数符号与机理", "结果": "；".join(f"{n}:{sign_text(v)}" for n, v in zip(FEATURE_NAMES, full["beta"])), "判定": "见报告正文"},
        {"检查项": "负预测个数", "结果": str(m_best["n_neg"]), "判定": "通过" if m_best["n_neg"] == 0 else "不通过"},
        {"检查项": "最大绝对误差区域", "结果": f"区域{int(region[max_i])}，AE={m_best['MaxAE']:.2f}", "判定": "记录"},
        {"检查项": "LightGBM与主模型排序相关", "结果": "未计算" if rank_corr is None else f"{rank_corr:.4f}", "判定": "—" if rank_corr is None else ("冲突" if rank_corr < 0.6 else "大体一致")},
        {"检查项": "LOOCV折间系数平均标准差", "结果": f"{float(coef_fold_std.mean()):.4f}", "判定": "记录"},
    ])
    save_table(stab, "诊断_稳定性检查")

    rec = build_recommendation(table4, m_best, table6, HAS_LGB, rank_corr)

    save_table(table4, "表4_模型性能比较")
    save_table(table5, "表5_实际值与交叉验证预测值")
    save_table(table6, "表6_最终模型回归系数")

    pack = {
        "region": region,
        "y": y,
        "X": X,
        "table4": table4,
        "table5": table5,
        "table6": table6,
        "best": dict(best_row),
        "yhat_best": yhat_best,
        "metrics_best": m_best,
        "grid_r1": grid_r1,
        "grid_r2": grid_r2,
        "grid_best": grid_best,
        "log_best": log_best,
        "ablation": ablation,
        "stability": stab,
        "colin": colin,
        "full": full,
        "recommendation": rec,
        "lam_r1": lam_r1,
        "lam_r2": lam_r2,
    }
    save_json({
        "best_model": best_name,
        "lambda_R1": lam_r1,
        "lambda_R2": lam_r2,
        "metrics": {k: float(m_best[k]) for k in ["MAE", "RMSE", "MAPE", "R2_CV", "MaxAE", "MaxAPE", "n_neg"]},
        "beta0": float(full["beta0"]),
        "beta": [float(v) for v in full["beta"]],
        "feature_means_log_or_raw": [float(v) for v in full["mu"]],
        "feature_stds_ddof0": [float(v) for v in full["sigma"]],
        "lightgbm_available": HAS_LGB,
        "rank_corr_lgb": rank_corr,
        "standardize_ddof": 0,
        "intercept_unpenalized": True,
        "recommendation": rec,
    }, "诊断_子任务二模型说明")

    plot_prediction(pack)
    write_report(pack)
    print(f"[OK] 主模型 {best_name}, lambda={lam_best:g}")
    print(f"     LOOCV RMSE={m_best['RMSE']:.2f}, MAE={m_best['MAE']:.2f}, MAPE={m_best['MAPE']:.2f}%, R2={m_best['R2_CV']:.4f}")
    print("完成 -> results/表4-表6, figures/图4-图6")
    return pack


if __name__ == "__main__":
    main()
