"""将 NIPT 原始表转换为可建模的标准化数据集。

输入：项目根目录的 附件.xlsx。
输出：results/male_prepared.csv、results/female_prepared.csv 和
      results/preprocess_summary.json。
核心处理：严格解析“检测孕周”为连续周数；按 身高(cm)、体重(kg)重算 BMI
用于一致性核验；为男胎构造 4% Y 浓度达标指示变量。
不删除重复测量、不插补缺失值、不改写原始字段；后续模型须自行限定有效样本。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "附件.xlsx"
RESULTS_DIR = PROJECT_DIR / "results"
SHEET_MAP = {"男胎检测数据": "male", "女胎检测数据": "female"}
WEEK_PATTERN = re.compile(r"^\s*(\d+)w(?:\+(\d+))?\s*$", flags=re.IGNORECASE)


def parse_gestational_week(value: object) -> float | None:
    """将如 12w+3、20w 的孕周编码转为连续周数；不符合格式时返回缺失。"""
    match = WEEK_PATTERN.match(str(value))
    if match is None:
        return None
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    if not 0 <= days <= 6:
        return None
    return weeks + days / 7


def prepare(frame: pd.DataFrame, sex: str) -> tuple[pd.DataFrame, dict[str, object]]:
    data = frame.copy()
    data.columns = data.columns.astype(str).str.strip()
    data["gestational_week"] = data["检测孕周"].map(parse_gestational_week)
    data["bmi_recomputed"] = data["体重"] / (data["身高"] / 100) ** 2
    data["bmi_difference"] = data["孕妇BMI"] - data["bmi_recomputed"]
    data["mother_observation_count"] = data.groupby("孕妇代码")["孕妇代码"].transform("size")
    data["observation_order"] = data.groupby("孕妇代码")["检测抽血次数"].rank(
        method="first"
    ).astype(int)
    if sex == "male":
        data["y_pass_4pct"] = (data["Y染色体浓度"] >= 0.04).astype("int8")

    summary: dict[str, object] = {
        "rows": int(len(data)),
        "mothers": int(data["孕妇代码"].nunique()),
        "invalid_gestational_week_rows": int(data["gestational_week"].isna().sum()),
        "gestational_week_range": [
            float(data["gestational_week"].min()),
            float(data["gestational_week"].max()),
        ],
        "bmi_missing_rows": int(data["孕妇BMI"].isna().sum()),
        "bmi_recomputation_abs_difference": {
            "median": float(data["bmi_difference"].abs().median()),
            "max": float(data["bmi_difference"].abs().max()),
        },
    }
    if sex == "male":
        summary["y_pass_4pct_rate"] = float(data["y_pass_4pct"].mean())
    return data, summary


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    summary: dict[str, object] = {
        "input": str(INPUT_PATH),
        "rules": [
            "孕周按 week + day/7 转换为连续周数。",
            "身高按 cm、体重按 kg 重算 BMI，仅用于核验，不替换原 BMI。",
            "男胎 4% 达标变量定义为 Y染色体浓度 >= 0.04。",
            "保留每条原始检测记录；不对缺失值插补。",
        ],
        "datasets": {},
    }
    for sheet, sex in SHEET_MAP.items():
        raw = pd.read_excel(INPUT_PATH, sheet_name=sheet)
        prepared, dataset_summary = prepare(raw, sex)
        output_path = RESULTS_DIR / f"{sex}_prepared.csv"
        prepared.to_csv(output_path, index=False, encoding="utf-8-sig")
        summary["datasets"][sex] = dataset_summary
    (RESULTS_DIR / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Prepared datasets and summary written to results/.")


if __name__ == "__main__":
    main()
