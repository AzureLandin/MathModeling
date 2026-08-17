# -*- coding: utf-8 -*-
"""问题一结果输出：单表 csv/xlsx + 汇总工作簿。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
SUMMARY_XLSX = RESULTS_DIR / "问题一_结果汇总.xlsx"

# 问题一统一表号：1-8 分布分析，9-13 PLSR 预测
TABLE_CATALOG = [
    "表1_区域典型日均负荷",
    "表2_区域空间差异指标",
    "表3_工作日分时平均负荷",
    "表4_工作日峰谷指标",
    "表5_区域工作日峰值时段",
    "表6_工作日周末配对检验",
    "表7_工作日周末分时负荷对比",
    "表8_峰值平移与强度变化",
    "表9_PLSR潜变量数量LOOCV",
    "表10_PLSR逐区域折外预测",
    "表11_最终PLSR参数",
    "表12_VIP变量重要性",
    "表13_最终PLSR区域需求",
]


def save_table(df: pd.DataFrame, stem: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    df.to_excel(RESULTS_DIR / f"{stem}.xlsx", index=False)
    rebuild_summary_workbook()


def rebuild_summary_workbook() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []
    frames: list[tuple[str, pd.DataFrame]] = []
    for i, stem in enumerate(TABLE_CATALOG, start=1):
        csv_path = RESULTS_DIR / f"{stem}.csv"
        exists = csv_path.exists()
        index_rows.append({
            "表号": f"表{i}",
            "文件名": stem,
            "内容": stem.split("_", 1)[1] if "_" in stem else stem,
            "是否已生成": "是" if exists else "否",
        })
        if exists:
            frames.append((f"表{i}", pd.read_csv(csv_path)))

    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(index_rows).to_excel(writer, sheet_name="目录", index=False)
        for sheet, df in frames:
            df.to_excel(writer, sheet_name=sheet, index=False)
