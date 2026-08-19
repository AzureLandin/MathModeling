# -*- coding: utf-8 -*-
"""
问题二公共工具：路径、配色、绘图风格、存盘。
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

# 固定参数
C_F = 6.0       # 万元/快充桩
C_S = 0.8       # 万元/慢充桩
P_F = 120.0     # kW/快充桩
P_S = 7.0       # kW/慢充桩
S_F = 80.0      # 车次/(快充桩·日)
S_S = 20.0      # 车次/(慢充桩·日)
COVERAGE_MIN = 0.90
GROWTH_RATE = 0.15

# NSGA-II 超参数
POP_SIZE = 240
N_GEN = 500
SEEDS = [2026, 2027, 2028, 2029, 2030]

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
    "ma": "#2E86AB",
    "mb": "#E76F51",
}


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


def save_json(obj: dict, stem: str) -> None:
    ensure_dirs()
    with open(RESULTS_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, (bool, np.bool_)):
                cells.append(str(v))
            elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                if np.isfinite(v) and abs(v - round(v)) < 1e-9 and abs(v) < 1e12:
                    cells.append(str(int(round(v))))
                else:
                    cells.append(floatfmt.format(float(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
