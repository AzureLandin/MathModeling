# -*- coding: utf-8 -*-
"""
problem1-plsr.py
问题一：偏最小二乘回归（PLSR）短期日均充电需求预测

输出：
  results/表9–表13
  results/问题一_结果汇总.xlsx（与时空分析表一并汇总）
  figures/图5–图7
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem1_utils import FIGURES_DIR, RESULTS_DIR, save_table

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
ATTACH_DIR = BASE_DIR / "附件"
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

for d in (RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FAST_KW = 120.0
SLOW_KW = 7.0
K_GRID = [1, 2, 3]
LAMBDA_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
TIE_REL = 0.01
PRED_EQ_TOL = 1e-6

FEATURE_NAMES = ["X1", "X2", "X3", "X4", "X5"]
FEATURE_MEANINGS = {
    "X1": "估算人口规模 D_pop*A",
    "X2": "日车流量 V",
    "X3": "商业POI数 B",
    "X4": "充电覆盖率 Acover/A*100%",
    "X5": "供给能力 120*Nf+7*Ns",
}

plt.rcParams["font.sans-serif"] = [
    "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


def zscore_train(X_tr: np.ndarray, X_te: np.ndarray | None = None):
    mu = X_tr.mean(axis=0)
    s = X_tr.std(axis=0, ddof=1)
    s = np.where(np.abs(s) < 1e-12, 1.0, s)
    X_tr_s = (X_tr - mu) / s
    if X_te is None:
        return X_tr_s, mu, s
    return X_tr_s, (X_te - mu) / s, mu, s


def demand_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ape = np.abs((y_true - y_pred) / y_true) * 100.0
    return {
        "R2": float(r2),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "MAPE": float(np.mean(ape)),
        "MaxAPE": float(np.max(ape)),
        "MAPE_78": float(np.mean(ape[[6, 7]])),
        "Bsum": float((y_pred.sum() - y_true.sum()) / y_true.sum() * 100.0),
        "n_neg": int(np.sum(y_pred < 0)),
        "ape": ape,
    }


def read_region_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=0)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    if df.iloc[:, 0].tolist() != list(range(1, 11)):
        raise ValueError(f"{path.name} 区域编号异常: {df.iloc[:, 0].tolist()}")
    return df


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
        raise ValueError(f"{path.name}/{sheet} 形状应为(10,24)，实际{mat.shape}")
    if np.any(mat < 0):
        raise ValueError(f"{path.name}/{sheet} 存在负值")
    return mat


def load_and_validate() -> dict:
    path1 = ATTACH_DIR / "附件 1 市主城区 10 个典型区域基础数据.xlsx"
    path3 = ATTACH_DIR / "附件3 市主城区 10 区域分时段充电负荷.xlsx"
    df1 = read_region_table(path1)
    area = df1.iloc[:, 1].astype(float).to_numpy()
    cover = df1.iloc[:, 2].astype(float).to_numpy()
    pop_density = df1.iloc[:, 3].astype(float).to_numpy()
    traffic = df1.iloc[:, 4].astype(float).to_numpy()
    poi = df1.iloc[:, 5].astype(float).to_numpy()
    n_fast = df1.iloc[:, 7].astype(float).to_numpy()
    n_slow = df1.iloc[:, 8].astype(float).to_numpy()
    if np.any(area <= 0):
        raise ValueError("区域面积必须全部大于0")
    if np.any(np.array([cover, pop_density, traffic, poi, n_fast, n_slow]) < 0):
        raise ValueError("存在负值字段")
    load_wd = read_hourly_matrix(path3, "工作日分时段充电负荷数据")
    load_we = read_hourly_matrix(path3, "周末充电负荷数据（修改后）")
    Q_wd = load_wd.sum(axis=1)
    Q_we = load_we.sum(axis=1)
    Y = (5 * Q_wd + 2 * Q_we) / 7.0
    X = np.column_stack([
        pop_density * area,
        traffic,
        poi,
        cover / area * 100.0,
        FAST_KW * n_fast + SLOW_KW * n_slow,
    ])
    feat_df = pd.DataFrame({
        "区域": np.arange(1, 11),
        "Y_Qbar": Y,
        "X1_估算人口": X[:, 0],
        "X2_车流量": X[:, 1],
        "X3_POI": X[:, 2],
        "X4_覆盖率percent": X[:, 3],
        "X5_供给能力_kW": X[:, 4],
    })
    feat_df.to_csv(PROCESSED_DIR / "plsr_features.csv", index=False, encoding="utf-8-sig")
    print("[OK] 数据读取与校验通过")
    return {"region": np.arange(1, 11), "Y": Y, "X": X, "feat_df": feat_df}


def fit_pls(X_std: np.ndarray, y: np.ndarray, k: int) -> PLSRegression:
    model = PLSRegression(n_components=k, scale=False, max_iter=500)
    model.fit(X_std, y)
    return model


def pls_predict(model: PLSRegression, X_std: np.ndarray) -> np.ndarray:
    pred = model.predict(X_std)
    return np.asarray(pred, dtype=float).ravel()


def pls_reconstruct_from_std_features(model: PLSRegression, X_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """由标准化原特征还原预测：intercept + X_std @ beta。"""
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 2:
        # sklearn>=1.3: (n_targets, n_features)
        if coef.shape[0] == 1:
            beta = coef.ravel()
        elif coef.shape[1] == 1:
            beta = coef.ravel()
        else:
            beta = coef[0]
    else:
        beta = coef.ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    y_hat = intercept + X_std @ beta
    return y_hat, beta, intercept


def compute_vip(model: PLSRegression) -> np.ndarray:
    W = np.asarray(model.x_weights_, dtype=float)  # (p, K)
    T = np.asarray(model.x_scores_, dtype=float)   # (n, K)
    c = np.asarray(model.y_loadings_, dtype=float).ravel()  # (K,)
    p, k = W.shape
    ssy = np.sum(T ** 2, axis=0) * (c ** 2)
    if np.sum(ssy) <= 0:
        return np.ones(p)
    w_norm_sq = (W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-15)) ** 2
    vip = np.sqrt(p * (w_norm_sq * ssy).sum(axis=1) / ssy.sum())
    return vip


def vip_level(v: float) -> str:
    if v > 1.0:
        return "相对重要"
    if v >= 0.8:
        return "中等贡献"
    return "贡献较弱"


def loocv_pls(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    n = len(y)
    yhat = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        X_tr_s, X_te_s, _, _ = zscore_train(X[tr], X[i:i + 1])
        model = fit_pls(X_tr_s, y[tr], k)
        yhat[i] = pls_predict(model, X_te_s)[0]
    return yhat


def loocv_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = len(y)
    yhat = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        X_tr_s, X_te_s, _, _ = zscore_train(X[tr], X[i:i + 1])
        model = Ridge(alpha=lam, fit_intercept=True, random_state=RANDOM_SEED)
        model.fit(X_tr_s, y[tr])
        yhat[i] = float(model.predict(X_te_s)[0])
    return yhat


def pick_k(table1: pd.DataFrame) -> tuple[int, str]:
    min_rmse = float(table1["CV_RMSE"].min())
    close = table1[table1["CV_RMSE"] <= min_rmse * (1.0 + TIE_REL)].copy()
    # 默认取 RMSE 最小；若相对差距<=1%，优先更少潜变量，除非辅助指标明显恶化
    best_k = int(table1.loc[table1["CV_RMSE"].idxmin(), "K"])
    reason = f"CV-RMSE 最小出现在 K={best_k}（{min_rmse:.4f}）。"
    if len(close) >= 2:
        smaller = close.sort_values("K").iloc[0]
        best_row = table1[table1["K"] == best_k].iloc[0]
        # 较少潜变量相对最优：MAE/MAPE/MAPE78 恶化不超过 5% 相对幅度则采用更少 K
        mae_ok = float(smaller["CV_MAE"]) <= float(best_row["CV_MAE"]) * 1.05
        mape_ok = float(smaller["CV_MAPE"]) <= float(best_row["CV_MAPE"]) * 1.05
        mape78_ok = float(smaller["CV_MAPE78"]) <= float(best_row["CV_MAPE78"]) * 1.05
        if mae_ok and mape_ok and mape78_ok:
            chosen = int(smaller["K"])
            reason += (
                f" K={int(smaller['K'])} 与最小 CV-RMSE 相对差距不超过 1%，"
                f"且 CV-MAE/MAPE/MAPE7,8 未明显恶化，按规则优先更少潜变量，取 K*={chosen}。"
            )
            return chosen, reason
        reason += (
            f" 虽有其他 K 落入 1% 阈值，但更少潜变量的辅助误差明显较差，仍取 K*={best_k}。"
        )
    return best_k, reason


def pick_ridge_lambdas(ridge_tbl: pd.DataFrame) -> dict:
    min_rmse = float(ridge_tbl["CV_RMSE"].min())
    strict_row = ridge_tbl.loc[ridge_tbl["CV_RMSE"].idxmin()]
    close = ridge_tbl[ridge_tbl["CV_RMSE"] <= min_rmse * (1.0 + TIE_REL)].copy()
    approx_row = close.sort_values("lambda", ascending=False).iloc[0]
    return {
        "strict_lambda": float(strict_row["lambda"]),
        "strict_row": strict_row,
        "approx_lambda": float(approx_row["lambda"]),
        "approx_row": approx_row,
        "same": np.isclose(float(strict_row["lambda"]), float(approx_row["lambda"])),
    }


def decide_replace(pls_m: dict, ridge_m: dict) -> tuple[bool, str]:
    conds = []
    better_rmse = pls_m["RMSE"] < ridge_m["RMSE"]
    conds.append(f"CV-RMSE：PLSR {pls_m['RMSE']:.2f} vs 岭回归 {ridge_m['RMSE']:.2f}，{'更低' if better_rmse else '未更低'}")
    mae_ok = pls_m["MAE"] <= ridge_m["MAE"] * 1.05
    mape_ok = pls_m["MAPE"] <= ridge_m["MAPE"] * 1.05
    conds.append(f"CV-MAE/MAPE 未明显恶化：MAE {pls_m['MAE']:.2f}/{ridge_m['MAE']:.2f}，MAPE {pls_m['MAPE']:.2f}/{ridge_m['MAPE']:.2f}")
    mape78_ok = pls_m["MAPE_78"] <= ridge_m["MAPE_78"] * 1.02
    conds.append(f"区域7/8：PLSR {pls_m['MAPE_78']:.2f}% vs 岭回归 {ridge_m['MAPE_78']:.2f}%")
    maxape_ok = pls_m["MaxAPE"] <= ridge_m["MaxAPE"] * 1.05
    conds.append(f"MaxAPE：PLSR {pls_m['MaxAPE']:.2f}% vs 岭回归 {ridge_m['MaxAPE']:.2f}%")
    pos_ok = pls_m["n_neg"] == 0
    bsum_ok = abs(pls_m["Bsum"]) <= max(5.0, abs(ridge_m["Bsum"]) + 2.0)
    conds.append(f"负预测数={pls_m['n_neg']}，Bsum={pls_m['Bsum']:.2f}%")
    replace = better_rmse and mae_ok and mape_ok and mape78_ok and maxape_ok and pos_ok and bsum_ok
    return replace, "；".join(conds)


def plot_k_curves(table1: pd.DataFrame, k_star: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].plot(table1["K"], table1["CV_RMSE"], "o-", color="#264653", linewidth=2)
    axes[1].plot(table1["K"], table1["CV_MAPE"], "s-", color="#E76F51", linewidth=2)
    row = table1[table1["K"] == k_star].iloc[0]
    axes[0].scatter([k_star], [row["CV_RMSE"]], s=90, c="#E9C46A", zorder=5)
    axes[1].scatter([k_star], [row["CV_MAPE"]], s=90, c="#E9C46A", zorder=5)
    axes[0].set_ylabel("CV-RMSE (kWh/d)")
    axes[1].set_ylabel("CV-MAPE (%)")
    for ax, title in zip(axes, ["CV-RMSE 随潜变量数变化", "CV-MAPE 随潜变量数变化"]):
        ax.set_xticks(K_GRID)
        ax.set_xlabel("潜变量数 K")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("图5 PLSR潜变量数量与CV指标", y=1.02)
    fig.savefig(FIGURES_DIR / "图5_PLSR潜变量数量与CV指标.png")
    plt.close(fig)


def plot_comparisons(region, y, pls_cv, ridge_cv) -> None:
    x = np.arange(10)
    width = 0.26
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.bar(x - width, y, width, label="实际典型日均需求", color="#264653", edgecolor="black", linewidth=0.3)
    ax.bar(x, pls_cv, width, label="PLSR折外预测", color="#2A9D8F", edgecolor="black", linewidth=0.3)
    ax.bar(x + width, ridge_cv, width, label="岭回归折外预测", color="#E9C46A", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in region])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("日均充电需求 (kWh/d)")
    ax.set_title("PLSR验证图2 PLSR与岭回归逐区域折外预测")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "PLSR验证_图2_PLSR与岭回归逐区域折外预测.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ape_p = np.abs((y - pls_cv) / y) * 100.0
    ape_r = np.abs((y - ridge_cv) / y) * 100.0
    ax.bar(x - 0.2, ape_p, 0.4, label="PLSR", color="#2A9D8F", edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.2, ape_r, 0.4, label="岭回归", color="#E9C46A", edgecolor="black", linewidth=0.3)
    ax.axhline(25, color="gray", linestyle="--", linewidth=1, label="25%参考线")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in region])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("折外相对误差 (%)")
    ax.set_title("PLSR验证图3 PLSR与岭回归折外相对误差")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "PLSR验证_图3_PLSR与岭回归折外相对误差.png")
    plt.close(fig)


def plot_vip(vip_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    colors = ["#E76F51" if v > 1 else ("#E9C46A" if v >= 0.8 else "#8AB17D") for v in vip_df["VIP"]]
    ax.barh(vip_df["特征"] + " " + vip_df["含义"], vip_df["VIP"], color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(1.0, color="#E63946", linestyle="--", linewidth=1, label="VIP=1")
    ax.axvline(0.8, color="gray", linestyle=":", linewidth=1, label="VIP=0.8")
    ax.set_xlabel("VIP")
    ax.set_title("图6 VIP变量重要性")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图6_VIP变量重要性.png")
    plt.close(fig)


def plot_pred_compare(region, y, y_full) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    x = np.arange(len(region))
    width = 0.38
    ax.bar(x - width / 2, y, width, label="实际典型日均需求", color="#264653", edgecolor="black", linewidth=0.3)
    ax.bar(x + width / 2, y_full, width, label="PLSR短期预测", color="#2A9D8F", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in region])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("日均充电需求 (kWh/d)")
    ax.set_title("图7 实际典型需求与PLSR短期预测对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "图7_实际与PLSR预测需求对比.png")
    plt.close(fig)


def write_analysis(pack: dict) -> Path:
    t1, t2, t3, t4, t5, t6 = pack["table1"], pack["table2"], pack["table3"], pack["table4"], pack["table5"], pack["table6"]
    info = pack["info"]
    k_star = info["K_star"]
    rec = info["recommendation"]

    t1_md = ["| K | CV-R² | CV-RMSE | CV-MAE | CV-MAPE | CV-MaxAPE | CV-MAPE7,8 | CV-Bsum | 负预测数 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in t1.iterrows():
        mark = " ←K*" if int(r["K"]) == k_star else ""
        t1_md.append(
            f"| {int(r['K'])}{mark} | {r['CV_R2']:.4f} | {r['CV_RMSE']:.2f} | {r['CV_MAE']:.2f} | "
            f"{r['CV_MAPE']:.2f} | {r['CV_MaxAPE']:.2f} | {r['CV_MAPE78']:.2f} | "
            f"{r['CV_Bsum']:.2f} | {int(r['负预测数'])} |"
        )

    t3_md = ["| 模型 | 超参数 | CV-R² | CV-RMSE | CV-MAE | CV-MAPE | CV-MaxAPE | CV-MAPE7,8 | CV-Bsum | 负预测 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in t3.iterrows():
        t3_md.append(
            f"| {r['模型']} | {r['超参数']} | {r['CV_R2']:.4f} | {r['CV_RMSE']:.2f} | "
            f"{r['CV_MAE']:.2f} | {r['CV_MAPE']:.2f} | {r['CV_MaxAPE']:.2f} | "
            f"{r['CV_MAPE78']:.2f} | {r['CV_Bsum']:.2f} | {int(r['负预测数'])} |"
        )

    t2_md = ["| 区域 | 实际需求 | PLSR折外 | 相对误差(%) | 重点校核 |",
             "| ---: | ---: | ---: | ---: | --- |"]
    for _, r in t2.iterrows():
        t2_md.append(
            f"| {int(r['区域'])} | {r['实际需求']:.2f} | {r['PLSR折外预测']:.2f} | "
            f"{r['相对误差']:.2f} | {r['是否重点校核区域']} |"
        )

    t5_md = ["| 特征 | 含义 | VIP | 等级 |",
             "| --- | --- | ---: | --- |"]
    for _, r in t5.iterrows():
        t5_md.append(f"| {r['特征']} | {r['含义']} | {r['VIP']:.4f} | {r['重要性等级']} |")

    pred_md = ["| 区域 | 实际 | 折外预测 | 折外相对误差(%) | 全样本拟合 | 样本内相对误差(%) | 短期预测 |",
               "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in t6.iterrows():
        pred_md.append(
            f"| {int(r['区域'])} | {r['实际典型日均需求']:.2f} | {r['LOOCV折外预测']:.2f} | "
            f"{r['折外相对误差']:.2f} | {r['全样本拟合值']:.2f} | {r['样本内相对误差']:.2f} | "
            f"{r['短期预测值']:.2f} |"
        )

    beta = info["std_feature_coefficients"]
    intercept = info["intercept"]
    W = np.array(info["x_weights"])
    theta = info["latent_regression_coefficients"]
    terms = [f"{intercept:.6f}"]
    for name, c in beta.items():
        sign = "+" if c >= 0 else "-"
        terms.append(f" {sign} {abs(c):.6f} {name}^*")
    eq = "".join(terms)

    latent_lines = []
    for k in range(k_star):
        parts = []
        for j, name in enumerate(FEATURE_NAMES):
            w = W[j][k]
            sign = "+" if w >= 0 else "-"
            if j == 0 and w >= 0:
                parts.append(f"{w:.6f} {name}^*")
            else:
                parts.append(f" {sign} {abs(w):.6f} {name}^*")
        latent_lines.append(f"$T_{{i{k+1}}}=" + "".join(parts) + "$")

    latent_reg_parts = [f"{intercept:.6f}"]
    for k, th in enumerate(theta):
        sign = "+" if th >= 0 else "-"
        latent_reg_parts.append(f" {sign} {abs(th):.6f} T_{{i{k+1}}}")
    latent_reg_eq = "".join(latent_reg_parts)

    rec_text = "建议采用 PLSR 替换岭回归作为问题一最终短期需求模型。" if rec["use_plsr"] else \
        "建议**保留岭回归**作为问题一最终短期需求模型；PLSR 仅作为共线性处理的对照验证。"

    q2_note = ""
    if rec["use_plsr"]:
        q2_note = "若论文采用 PLSR，供问题二使用的短期预测值见下表“短期预测”列（全样本方程，非折外预测）。"
    else:
        q2_note = "因不建议替换岭回归，问题二仍应使用既有 M2 岭回归全样本预测值；下表 PLSR 短期预测仅作对照。"

    report = f"""# 问题一 PLSR 模型计算结果分析

> 由 `code/problem1-plsr.py` 生成。  
> 预测目标与五个解释变量与 M2 原尺度总量岭回归完全相同。标准化为折内/全样本 ddof=1 的 Z-score；`PLSRegression(n_components=K, scale=False)`。  
> LOOCV 指标用于选 K 与相对比较，**不是**稳定的未来泛化精度。

---

## 一、潜变量数量选择（问题 1、2）

{chr(10).join(t1_md)}

{info['k_selection_reason']}

最优潜变量数 **$K^*={k_star}$**。曲线见图1。不得用样本内 $R^2$ 选 $K$。

---

## 二、与岭回归公平比较（问题 3--6）

岭回归在同一数据、同一五特征、同一 LOOCV 划分和折内标准化下重算：

- 严格最小 CV-RMSE：λ = {info['ridge_strict_lambda']:g}
- “CV-RMSE 差不超过 1% 时选更强惩罚”：λ = {info['ridge_approx_lambda']:g}

主比较使用严格最小 CV-RMSE 的岭回归。

{chr(10).join(t3_md)}

**PLSR 是否降低折外误差？** {info['compare_text']}

**区域 7、8：** PLSR 的 CV-MAPE7,8 = {info['loocv_metrics']['MAPE_78']:.2f}%，岭回归（严格）为 {info['ridge_strict_metrics']['MAPE_78']:.2f}%。

**新的高误差区域或负预测：** PLSR 负预测数 = {info['loocv_metrics']['n_neg']}。逐区域折外结果：

{chr(10).join(t2_md)}

**最终建议：** {rec_text}

判断依据：{rec['detail']}

---

## 三、最终 PLSR 方程（问题 7、9）

实现方式：先对五个特征做全样本 Z-score（ddof=1），再拟合 `PLSRegression(K={k_star}, scale=False)`；目标 $Y$ 不人工标准化。标准化原特征方程与 `predict()` 的最大绝对偏差为 {info['equation_max_abs_err']:.3e}（要求 ≤ 1e-6）。

截距 $\\hat\\beta_0={intercept:.6f}$

标准化原特征方程：

$$
\\hat Y_i = {eq}
$$

潜变量：

{chr(10).join(f"- {line}" for line in latent_lines)}

潜变量回归： $\\hat Y_i={latent_reg_eq}$

**折外指标（用于选 $K$ / 比较）：**

| 指标 | 数值 |
| --- | ---: |
| CV-$R^2$ | {info['loocv_metrics']['R2']:.6f} |
| CV-RMSE | {info['loocv_metrics']['RMSE']:.4f} |
| CV-MAE | {info['loocv_metrics']['MAE']:.4f} |
| CV-MAPE (%) | {info['loocv_metrics']['MAPE']:.4f} |
| CV-MaxAPE (%) | {info['loocv_metrics']['MaxAPE']:.4f} |
| CV-MAPE7,8 (%) | {info['loocv_metrics']['MAPE_78']:.4f} |
| CV-$B_{{sum}}$ (%) | {info['loocv_metrics']['Bsum']:.4f} |

**全样本拟合指标（仅说明现有 10 区拟合，不参与选 $K$）：**

| 指标 | 数值 |
| --- | ---: |
| 样本内 $R^2$ | {info['insample_metrics']['R2']:.6f} |
| 样本内 RMSE | {info['insample_metrics']['RMSE']:.4f} |
| 样本内 MAPE (%) | {info['insample_metrics']['MAPE']:.4f} |
| 样本内 MaxAPE (%) | {info['insample_metrics']['MaxAPE']:.4f} |
| 样本内 MAPE7,8 (%) | {info['insample_metrics']['MAPE_78']:.4f} |
| 样本内 $B_{{sum}}$ (%) | {info['insample_metrics']['Bsum']:.4f} |

---

## 四、VIP 变量重要性（问题 8）

{chr(10).join(t5_md)}

VIP>1 视为相对重要，0.8--1 为中等，<0.8 较弱。VIP 只表示当前 PLSR 中的预测贡献，不是因果效应，也不据此自动删变量。图4给出条形图。

在当前五个总量/供给特征下，VIP 较高的变量对应与区域充电总量共变更强的规模与供给信息；覆盖率若 VIP 偏低，说明其在控制规模变量后的增量贡献有限。

---

## 五、短期需求预测（问题 10）

{q2_note}

{chr(10).join(pred_md)}

全市实际总需求 {t6['实际典型日均需求'].sum():.2f} kWh/d，PLSR 全样本拟合总需求 {t6['全样本拟合值'].sum():.2f} kWh/d。

---

## 六、解释边界

1. n=10 的 LOOCV 只用于相对比较，存在选择乐观偏差。
2. 标准化均在折内完成，无全样本预标准化泄漏。
3. PLSR 缓解共线性，但不能弥补私桩、停车时长、车辆类型等缺失信息。
4. 覆盖率与供给能力可能与同期需求互相影响，系数和 VIP 不是因果增量。
5. 若 PLSR 折外没有明显优于岭回归，应保留更简洁或更稳的模型，而不是继续加复杂度。

---

## 七、输出清单

- `results/PLSR验证_表1` 至 `表6`
- `results/PLSR验证_最终模型说明.json`
- `figures/PLSR验证_图1` 至 `图4`
- `code/problem1-plsr.py`

---

*生成完毕。*
"""
    out = REPORTS_DIR / "问题一_PLSR模型计算结果分析.md"
    out.write_text(report, encoding="utf-8")
    print(f"[OK] 分析报告已写入 {out}")
    return out


def main():
    print("=" * 64)
    print("问题一 PLSR 验证开始")
    print("=" * 64)
    data = load_and_validate()
    X, y, region = data["X"], data["Y"], data["region"]

    # ---- PLSR LOOCV over K ----
    rows1 = []
    pls_cv_by_k = {}
    for k in K_GRID:
        yhat = loocv_pls(X, y, k)
        pls_cv_by_k[k] = yhat
        m = demand_metrics(y, yhat)
        rows1.append({
            "K": k,
            "CV_R2": m["R2"],
            "CV_RMSE": m["RMSE"],
            "CV_MAE": m["MAE"],
            "CV_MAPE": m["MAPE"],
            "CV_MaxAPE": m["MaxAPE"],
            "CV_MAPE78": m["MAPE_78"],
            "CV_Bsum": m["Bsum"],
            "负预测数": m["n_neg"],
        })
    table1 = pd.DataFrame(rows1)
    save_table(table1, "表9_PLSR潜变量数量LOOCV")
    k_star, k_reason = pick_k(table1)
    print(f"[OK] PLSR LOOCV 完成，K*={k_star}")

    # ---- Ridge LOOCV over lambda ----
    ridge_rows = []
    ridge_cv_by_lam = {}
    for lam in LAMBDA_GRID:
        yhat = loocv_ridge(X, y, lam)
        ridge_cv_by_lam[lam] = yhat
        m = demand_metrics(y, yhat)
        ridge_rows.append({
            "lambda": lam,
            "CV_R2": m["R2"],
            "CV_RMSE": m["RMSE"],
            "CV_MAE": m["MAE"],
            "CV_MAPE": m["MAPE"],
            "CV_MaxAPE": m["MaxAPE"],
            "CV_MAPE78": m["MAPE_78"],
            "CV_Bsum": m["Bsum"],
            "负预测数": m["n_neg"],
        })
    ridge_tbl = pd.DataFrame(ridge_rows)
    # 岭回归仅作内部对照，不写入正式表号
    ridge_pick = pick_ridge_lambdas(ridge_tbl)
    print(f"[OK] 岭回归 LOOCV 完成，严格λ={ridge_pick['strict_lambda']:g}，近似λ={ridge_pick['approx_lambda']:g}")

    pls_cv = pls_cv_by_k[k_star]
    ridge_cv = ridge_cv_by_lam[ridge_pick["strict_lambda"]]
    ridge_cv_approx = ridge_cv_by_lam[ridge_pick["approx_lambda"]]
    pls_m = demand_metrics(y, pls_cv)
    ridge_m = demand_metrics(y, ridge_cv)
    ridge_m_approx = demand_metrics(y, ridge_cv_approx)

    # 表2
    ape = pls_m["ape"]
    focus = ["是" if (r in (7, 8) or ape[i] >= 25) else "否" for i, r in enumerate(region)]
    table2 = pd.DataFrame({
        "区域": region,
        "实际需求": y,
        "PLSR折外预测": pls_cv,
        "绝对误差": np.abs(y - pls_cv),
        "相对误差": ape,
        "是否重点校核区域": focus,
    })
    save_table(table2, "表10_PLSR逐区域折外预测")

    # 表3
    table3_rows = [
        {
            "模型": "最优PLSR",
            "超参数": f"K={k_star}",
            "CV_R2": pls_m["R2"], "CV_RMSE": pls_m["RMSE"], "CV_MAE": pls_m["MAE"],
            "CV_MAPE": pls_m["MAPE"], "CV_MaxAPE": pls_m["MaxAPE"],
            "CV_MAPE78": pls_m["MAPE_78"], "CV_Bsum": pls_m["Bsum"], "负预测数": pls_m["n_neg"],
        },
        {
            "模型": "岭回归_严格最小CVRMSE",
            "超参数": f"lambda={ridge_pick['strict_lambda']:g}",
            "CV_R2": ridge_m["R2"], "CV_RMSE": ridge_m["RMSE"], "CV_MAE": ridge_m["MAE"],
            "CV_MAPE": ridge_m["MAPE"], "CV_MaxAPE": ridge_m["MaxAPE"],
            "CV_MAPE78": ridge_m["MAPE_78"], "CV_Bsum": ridge_m["Bsum"], "负预测数": ridge_m["n_neg"],
        },
    ]
    if not ridge_pick["same"]:
        table3_rows.append({
            "模型": "岭回归_1percent更强惩罚",
            "超参数": f"lambda={ridge_pick['approx_lambda']:g}",
            "CV_R2": ridge_m_approx["R2"], "CV_RMSE": ridge_m_approx["RMSE"],
            "CV_MAE": ridge_m_approx["MAE"], "CV_MAPE": ridge_m_approx["MAPE"],
            "CV_MaxAPE": ridge_m_approx["MaxAPE"], "CV_MAPE78": ridge_m_approx["MAPE_78"],
            "CV_Bsum": ridge_m_approx["Bsum"], "负预测数": ridge_m_approx["n_neg"],
        })
    table3 = pd.DataFrame(table3_rows)

    use_plsr, compare_detail = decide_replace(pls_m, ridge_m)

    # ---- 全样本 PLSR ----
    X_std, mu, s = zscore_train(X)
    final_model = fit_pls(X_std, y, k_star)
    y_full = pls_predict(final_model, X_std)
    y_recon, beta, intercept = pls_reconstruct_from_std_features(final_model, X_std)
    max_eq_err = float(np.max(np.abs(y_full - y_recon)))
    if max_eq_err > PRED_EQ_TOL:
        raise RuntimeError(f"标准化原特征方程与 predict() 不一致，max abs err={max_eq_err}")
    print(f"[OK] 方程复核通过，max abs err={max_eq_err:.3e}")

    in_m = demand_metrics(y, y_full)
    vip = compute_vip(final_model)
    W = np.asarray(final_model.x_weights_, dtype=float)
    T = np.asarray(final_model.x_scores_, dtype=float)
    # 潜变量回归系数：对中心化 Y 用 T 回归，再加截距
    y_c = y - y.mean()
    theta, *_ = np.linalg.lstsq(T, y_c, rcond=None)
    # sklearn 的 intercept 已含 Y 均值

    # 表4 参数长表
    param_rows = []
    for name, val in zip(FEATURE_NAMES, mu):
        param_rows.append({"参数类型": "特征均值", "名称": name, "数值": float(val)})
    for name, val in zip(FEATURE_NAMES, s):
        param_rows.append({"参数类型": "样本标准差_ddof1", "名称": name, "数值": float(val)})
    param_rows.append({"参数类型": "截距", "名称": "beta0", "数值": intercept})
    for name, val in zip(FEATURE_NAMES, beta):
        param_rows.append({"参数类型": "标准化原特征系数", "名称": name, "数值": float(val)})
    for k in range(k_star):
        param_rows.append({"参数类型": "潜变量回归系数", "名称": f"theta_T{k+1}", "数值": float(theta[k])})
        for j, name in enumerate(FEATURE_NAMES):
            param_rows.append({"参数类型": f"潜变量权重_T{k+1}", "名称": name, "数值": float(W[j, k])})
    table4 = pd.DataFrame(param_rows)
    save_table(table4, "表11_最终PLSR参数")

    table5 = pd.DataFrame({
        "特征": FEATURE_NAMES,
        "含义": [FEATURE_MEANINGS[n] for n in FEATURE_NAMES],
        "VIP": vip,
        "重要性等级": [vip_level(v) for v in vip],
    }).sort_values("VIP", ascending=False).reset_index(drop=True)
    save_table(table5, "表12_VIP变量重要性")

    table6 = pd.DataFrame({
        "区域": region,
        "实际典型日均需求": y,
        "LOOCV折外预测": pls_cv,
        "折外相对误差": ape,
        "全样本拟合值": y_full,
        "样本内相对误差": np.abs((y - y_full) / y) * 100.0,
        "短期预测值": y_full,
    })
    save_table(table6, "表13_最终PLSR区域需求")

    plot_k_curves(table1, k_star)
    plot_vip(table5)
    plot_pred_compare(region, y, y_full)

    info = {
        "K_star": k_star,
        "k_selection_reason": k_reason,
        "feature_definitions": FEATURE_MEANINGS,
        "feature_means": {n: float(v) for n, v in zip(FEATURE_NAMES, mu)},
        "feature_stds_ddof1": {n: float(v) for n, v in zip(FEATURE_NAMES, s)},
        "x_weights": W.tolist(),
        "x_scores": T.tolist(),
        "latent_regression_coefficients": [float(v) for v in theta],
        "std_feature_coefficients": {n: float(v) for n, v in zip(FEATURE_NAMES, beta)},
        "intercept": intercept,
        "VIP": {n: float(v) for n, v in zip(FEATURE_NAMES, vip)},
        "loocv_metrics": {k: float(pls_m[k]) for k in ["R2", "RMSE", "MAE", "MAPE", "MaxAPE", "MAPE_78", "Bsum", "n_neg"]},
        "insample_metrics": {k: float(in_m[k]) for k in ["R2", "RMSE", "MAE", "MAPE", "MaxAPE", "MAPE_78", "Bsum"]},
        "loocv_predictions": [float(v) for v in pls_cv],
        "insample_predictions": [float(v) for v in y_full],
        "ridge_strict_lambda": ridge_pick["strict_lambda"],
        "ridge_approx_lambda": ridge_pick["approx_lambda"],
        "ridge_strict_metrics": {k: float(ridge_m[k]) for k in ["R2", "RMSE", "MAE", "MAPE", "MaxAPE", "MAPE_78", "Bsum", "n_neg"]},
        "ridge_approx_metrics": {k: float(ridge_m_approx[k]) for k in ["R2", "RMSE", "MAE", "MAPE", "MaxAPE", "MAPE_78", "Bsum", "n_neg"]},
        "equation_max_abs_err": max_eq_err,
        "recommendation": {"use_plsr": bool(use_plsr), "detail": compare_detail},
        "compare_text": compare_detail,
        "implementation": "manual Z-score ddof=1 inside each fold / full sample; PLSRegression(scale=False); Y not manually standardized",
        "random_seed": RANDOM_SEED,
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    with open(RESULTS_DIR / "表11_最终PLSR模型说明.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print(f"K*={k_star}  建议采用PLSR={use_plsr}")
    print(f"PLSR CV-RMSE={pls_m['RMSE']:.2f}  岭回归(λ={ridge_pick['strict_lambda']:g}) CV-RMSE={ridge_m['RMSE']:.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
