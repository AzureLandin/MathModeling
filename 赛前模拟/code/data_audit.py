"""NIPT 附件的只读数据体检。

输入：项目根目录的 附件.xlsx（男胎检测数据、女胎检测数据）。
输出：results/data_audit.json。
方法：字段名去除首尾空白；逐表统计数据类型、缺失、数值分位数、孕妇重复观测和标签分布。
不做缺失插补、异常删除或数值变换。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "附件.xlsx"
OUTPUT_PATH = PROJECT_DIR / "results" / "data_audit.json"
SHEETS = ("男胎检测数据", "女胎检测数据")
QUANTILES = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]


def json_value(value: object) -> object:
    """将 pandas/numpy 标量安全地转为 JSON 值。"""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def audit_sheet(frame: pd.DataFrame) -> dict[str, object]:
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    numeric = frame.select_dtypes(include="number")
    field_summary = []
    for column in frame.columns:
        series = frame[column]
        item: dict[str, object] = {
            "field": column,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_nonmissing": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            item["quantiles"] = {
                str(q): json_value(v)
                for q, v in series.quantile(QUANTILES).items()
            }
        else:
            item["top_values"] = {
                str(k): int(v)
                for k, v in series.value_counts(dropna=False).head(10).items()
            }
        field_summary.append(item)

    mother_counts = frame["孕妇代码"].value_counts(dropna=True)
    label_column = "染色体的非整倍体"
    health_column = "胎儿是否健康"
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "n_mothers": int(mother_counts.size),
        "observations_per_mother": {
            "min": int(mother_counts.min()),
            "median": float(mother_counts.median()),
            "max": int(mother_counts.max()),
            "mothers_with_multiple_observations": int((mother_counts > 1).sum()),
        },
        "label_distribution": {
            str(k): int(v)
            for k, v in frame[label_column].fillna("<缺失>").value_counts().items()
        },
        "health_distribution": {
            str(k): int(v)
            for k, v in frame[health_column].fillna("<缺失>").value_counts().items()
        },
        "numeric_correlation": {
            row: {column: json_value(value) for column, value in values.items()}
            for row, values in numeric.corr(numeric_only=True).items()
        },
        "fields": field_summary,
    }


def main() -> None:
    report = {
        "input_path": str(INPUT_PATH),
        "rules": [
            "字段名仅去除首尾空白；不改变原始观测值。",
            "缺失率基于原始行；重复行指所有字段完全一致的记录。",
            "相关矩阵为 Pearson 相关，仅用于初步体检，不能替代重复测量回归。",
        ],
        "sheets": {},
    }
    for sheet in SHEETS:
        report["sheets"][sheet] = audit_sheet(pd.read_excel(INPUT_PATH, sheet_name=sheet))
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
