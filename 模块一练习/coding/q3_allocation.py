from pathlib import Path

import numpy as np
import pandas as pd


DELIVERY = Path(__file__).resolve().parents[1]
SOURCE = DELIVERY / "data" / "processed" / "q1_teacher_scores.csv"
QUOTA_OUTPUT = DELIVERY / "results" / "q3_quota.csv"
ORDER_OUTPUT = DELIVERY / "results" / "q3_selection_order.csv"
SCHEDULE_OUTPUT = DELIVERY / "results" / "q3_pick_schedule.csv"
TOTAL = 175


def round_half_up(values):
    return np.floor(np.asarray(values) + 0.5).astype(int)


def main():
    data = pd.read_csv(SOURCE, encoding="utf-8-sig")
    data = data.sort_values(["综合得分", "累计指导时间", "教师编号"], ascending=[False, False, True]).reset_index(drop=True)
    data["选队次序"] = np.arange(1, 36)
    data["百分位"] = (35 - data["选队次序"]) / 34 * 100
    high = data["百分位"] >= 70
    middle = (data["百分位"] >= 40) & ~high
    data["配额组别"] = np.select([high, middle], ["A", "B"], default="C")
    data["组下限"] = data["配额组别"].map({"A": 7, "B": 5, "C": 3})
    data["组上限"] = data["配额组别"].map({"A": 10, "B": 7, "C": 4})
    data["连续配额"] = np.select(
        [high, middle],
        [7 + 3 * (data["百分位"] - 70) / 30, 5 + 2 * (data["百分位"] - 40) / 30],
        default=3 + data["百分位"] / 40,
    )
    data["初始配额"] = round_half_up(data["连续配额"])
    data["最终配额"] = data["初始配额"].clip(lower=data["组下限"], upper=data["组上限"])

    excess = int(data["最终配额"].sum() - TOTAL)
    for index in data.sort_values(["综合得分", "累计指导时间", "教师编号"], ascending=[True, True, False]).index:
        while excess and data.at[index, "最终配额"] > data.at[index, "组下限"]:
            data.at[index, "最终配额"] -= 1
            excess -= 1
    if excess:
        raise RuntimeError(f"配额修正失败，仍超出{excess}队")

    rows, pick = [], 1
    for round_number in range(1, 11):
        for teacher in data.itertuples(index=False):
            if teacher.最终配额 >= round_number:
                rows.append({"pick_index": pick, "round": round_number, "teacher_id": teacher.教师编号})
                pick += 1
    schedule = pd.DataFrame(rows)
    assert data["最终配额"].sum() == TOTAL and len(schedule) == TOTAL
    assert data["最终配额"].between(3, 10).all()
    actual = schedule["teacher_id"].value_counts().sort_index()
    expected = data.set_index("教师编号")["最终配额"].sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False)

    columns = ["教师编号", "综合得分", "累计指导时间", "百分位", "配额组别", "连续配额", "初始配额", "最终配额", "选队次序"]
    data[columns].to_csv(QUOTA_OUTPUT, index=False, encoding="utf-8-sig", float_format="%.6f")
    data[["选队次序", "教师编号", "综合得分", "配额组别", "最终配额"]].to_csv(
        ORDER_OUTPUT, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    schedule.to_csv(SCHEDULE_OUTPUT, index=False, encoding="utf-8-sig")
    print("配额分布:", data["最终配额"].value_counts().sort_index().to_dict())
    print("配额合计:", int(data["最终配额"].sum()))


if __name__ == "__main__":
    main()
