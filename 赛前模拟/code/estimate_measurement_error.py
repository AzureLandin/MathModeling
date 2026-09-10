"""从同日同次抽血的男胎复测估计 Y 染色体浓度技术误差。

输入：results/male_prepared.csv。
输出：results/technical_replicates.csv 与 results/technical_error_summary.json。
规则：仅保留同一孕妇、检测日期和检测抽血次数相同且恰有两条记录的事件；
      若两次测量误差独立同方差，则单次测量标准差估计为 sd(差异)/sqrt(2)。
限制：该估计基于有限复测事件，描述的是本附件内的技术变异，不能替代外部临床标定。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "results" / "male_prepared.csv"
RESULTS_DIR = PROJECT_DIR / "results"
PAIR_KEYS = ["孕妇代码", "检测日期", "检测抽血次数"]
THRESHOLD = 0.04


def extract_pairs(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, event in data.groupby(PAIR_KEYS, sort=False):
        if len(event) != 2:
            continue
        first, second = event.iloc[0], event.iloc[1]
        y_first = float(first["Y染色体浓度"])
        y_second = float(second["Y染色体浓度"])
        rows.append(
            {
                "mother_id": keys[0],
                "test_date": int(keys[1]),
                "draw_count": int(keys[2]),
                "gestational_week_first": float(first["gestational_week"]),
                "gestational_week_second": float(second["gestational_week"]),
                "y_first": y_first,
                "y_second": y_second,
                "difference_second_minus_first": y_second - y_first,
                "absolute_difference": abs(y_second - y_first),
                "threshold_discordant": bool((y_first >= THRESHOLD) != (y_second >= THRESHOLD)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    pairs = extract_pairs(data)
    if pairs.empty:
        raise RuntimeError("No paired technical replicates were found.")
    pairs.to_csv(RESULTS_DIR / "technical_replicates.csv", index=False, encoding="utf-8-sig")
    differences = pairs["difference_second_minus_first"]
    ttest = ttest_1samp(differences, popmean=0.0)
    sigma = float(differences.std(ddof=1) / np.sqrt(2))
    summary = {
        "pair_definition": "same mother, test date, and draw count; exactly two records",
        "n_pair_events": int(len(pairs)),
        "difference_scale": "proportion (multiply by 100 for percentage points)",
        "mean_difference": float(differences.mean()),
        "sd_difference": float(differences.std(ddof=1)),
        "estimated_single_measurement_sigma": sigma,
        "estimated_single_measurement_sigma_percentage_points": 100 * sigma,
        "mean_absolute_difference": float(pairs["absolute_difference"].mean()),
        "threshold_discordant_events": int(pairs["threshold_discordant"].sum()),
        "threshold_discordant_rate": float(pairs["threshold_discordant"].mean()),
        "mean_difference_ttest_pvalue": float(ttest.pvalue),
        "limitations": [
            "Pairs are inferred from metadata and may not perfectly represent independent technical replicates.",
            "The sample of pair events is small; this is a scenario parameter, not a clinical calibration constant.",
        ],
    }
    (RESULTS_DIR / "technical_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Technical replicate error summary written to results/.")


if __name__ == "__main__":
    main()
