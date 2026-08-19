# -*- coding: utf-8 -*-
"""
问题一公共工具：路径、读表、校验、存盘、绘图风格。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ATTACH_DIR = BASE_DIR / "附件"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"
SUMMARY_XLSX = RESULTS_DIR / "问题一_结果汇总.xlsx"

HOUR_LABELS = [
    "00-01", "01-02", "02-03", "03-04", "04-05", "05-06",
    "06-07", "07-08", "08-09", "09-10", "10-11", "11-12",
    "12-13", "13-14", "14-15", "15-16", "16-17", "17-18",
    "18-19", "19-20", "20-21", "21-22", "22-23", "23-00",
]

N_REGIONS = 10
WD_WEIGHT = 5
WE_WEIGHT = 2
WEEK_LEN = 7
ETA_THRESHOLD = 20.0  # 解释性分类阈值（%）
CONS_ATOL = 1e-8

# 统一配色
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#E76F51",
    "tertiary": "#2A9D8F",
    "peak": "#E63946",
    "valley": "#457B9D",
    "neutral": "#6C757D",
    "gold": "#E9C46A",
    "dark": "#264653",
}

TABLE_CATALOG = [
    "表1_区域综合日均充电负荷及排序",
    "表2_综合典型日波动特征",
    "表3_工作日周末充电需求差异",
    "表4_模型性能比较",
    "表5_实际值与交叉验证预测值",
    "表6_最终模型回归系数",
    "表7_未来短期综合日均需求情景预测",
    "诊断_数据质量与一致性校验",
    "诊断_综合典型日分时负荷",
    "诊断_全市综合典型日曲线",
    "诊断_权重方案排序稳健性",
    "诊断_权重方案排序稳健性对照",
    "诊断_预测特征与目标",
    "诊断_特征相关与共线性",
    "诊断_岭回归lambda网格LOOCV",
    "诊断_岭回归R1_lambda网格",
    "诊断_去总桩数消融",
    "诊断_短期基准需求",
    "诊断_稳定性检查",
    "诊断_情景预测一致性校验",
]


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = False


def find_attachment(prefix: str) -> Path:
    """按文件名前缀定位附件，忽略 Excel 锁文件 ~$*。"""
    cands = [
        p for p in ATTACH_DIR.glob("*.xlsx")
        if p.name.startswith(prefix) and not p.name.startswith("~$")
    ]
    if not cands:
        raise FileNotFoundError(f"未找到附件：{prefix}*.xlsx（目录 {ATTACH_DIR}）")
    return sorted(cands, key=lambda p: len(p.name))[0]


def save_table(df: pd.DataFrame, stem: str) -> None:
    ensure_dirs()
    df.to_csv(RESULTS_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    df.to_excel(RESULTS_DIR / f"{stem}.xlsx", index=False)
    rebuild_summary_workbook()


def save_json(obj: dict, stem: str) -> None:
    ensure_dirs()
    with open(RESULTS_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rebuild_summary_workbook() -> None:
    ensure_dirs()
    index_rows = []
    frames: list[tuple[str, pd.DataFrame]] = []
    for i, stem in enumerate(TABLE_CATALOG, start=1):
        csv_path = RESULTS_DIR / f"{stem}.csv"
        exists = csv_path.exists()
        if stem.startswith("表"):
            label = stem.split("_", 1)[0]
        else:
            label = f"诊断{i - 6}" if i > 6 else stem
        index_rows.append({
            "序号": i,
            "表号或类型": label,
            "文件名": stem,
            "是否已生成": "是" if exists else "否",
        })
        if exists:
            sheet = label[:31]
            frames.append((sheet, pd.read_csv(csv_path)))

    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(index_rows).to_excel(writer, sheet_name="目录", index=False)
        used = {"目录"}
        for sheet, df in frames:
            name = sheet
            k = 2
            while name in used:
                name = f"{sheet[:28]}_{k}"
                k += 1
            used.add(name)
            df.to_excel(writer, sheet_name=name, index=False)


def format_hour_list(indices_0based: list[int]) -> str:
    return "、".join(HOUR_LABELS[i] for i in indices_0based)


def argmax_all(arr: np.ndarray, atol: float = 1e-10) -> list[int]:
    m = float(np.max(arr))
    return np.where(np.isclose(arr, m, atol=atol, rtol=0.0))[0].tolist()


def argmin_all(arr: np.ndarray, atol: float = 1e-10) -> list[int]:
    m = float(np.min(arr))
    return np.where(np.isclose(arr, m, atol=atol, rtol=0.0))[0].tolist()


def classify_day_type(eta_percent: float, thr: float = ETA_THRESHOLD) -> str:
    if eta_percent >= thr:
        return "周末主导型"
    if eta_percent <= -thr:
        return "工作日主导型"
    return "均衡型"


def read_hourly_matrix(path: Path, sheet: str) -> tuple[np.ndarray, list[str]]:
    """读取 10×24 分时矩阵，校验区域编号与时段列顺序。"""
    raw = pd.read_excel(path, sheet_name=sheet, header=0)
    raw = raw.dropna(subset=[raw.columns[0]]).copy()
    raw = raw[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()
    raw.iloc[:, 0] = raw.iloc[:, 0].astype(int)
    raw = raw.sort_values(raw.columns[0]).reset_index(drop=True)

    regions = raw.iloc[:, 0].tolist()
    if regions != list(range(1, N_REGIONS + 1)):
        raise ValueError(f"{path.name}/{sheet} 区域编号应为1-10，实际为 {regions}")

    hour_cols = [str(c).strip() for c in raw.columns[1:25]]
    if hour_cols != HOUR_LABELS:
        raise ValueError(
            f"{path.name}/{sheet} 时段列顺序不符。\n期望: {HOUR_LABELS}\n实际: {hour_cols}"
        )

    block = raw.iloc[:, 1:25]
    if block.isna().any().any():
        raise ValueError(f"{path.name}/{sheet} 存在缺失值")
    numeric = block.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"{path.name}/{sheet} 存在非数值")
    mat = numeric.to_numpy(dtype=float)
    if mat.shape != (N_REGIONS, 24):
        raise ValueError(f"{sheet} 形状应为(10,24)，实际 {mat.shape}")
    if np.any(mat < 0):
        raise ValueError(f"{path.name}/{sheet} 存在负负荷")
    return mat, hour_cols


def read_region_features(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=0)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    if df.iloc[:, 0].tolist() != list(range(1, N_REGIONS + 1)):
        raise ValueError(f"{path.name} 区域编号异常: {df.iloc[:, 0].tolist()}")
    return df


def weighted_typical(p_wd: np.ndarray, p_we: np.ndarray) -> np.ndarray:
    return (WD_WEIGHT * p_wd + WE_WEIGHT * p_we) / WEEK_LEN


def demand_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ape = np.abs(err / y_true) * 100.0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(ape)),
        "R2_CV": float(r2),
        "MaxAE": float(np.max(np.abs(err))),
        "MaxAPE": float(np.max(ape)),
        "n_neg": int(np.sum(y_pred < 0)),
        "ape": ape,
        "ae": np.abs(err),
    }


def _fmt_md_cell(v, floatfmt: str) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isfinite(v) and abs(v - round(v)) < 1e-9 and abs(v) < 1e12:
            return str(int(round(v)))
        return floatfmt.format(float(v))
    return str(v)


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    aligns = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            aligns.append("---:")
        else:
            aligns.append("---")
    lines.append("| " + " | ".join(aligns) + " |")
    for _, row in df.iterrows():
        cells = [_fmt_md_cell(row[c], floatfmt) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
