import re
from pathlib import Path

import numpy as np
import pandas as pd


DELIVERY = Path(__file__).resolve().parents[1]
SOURCE = DELIVERY / "data" / "raw" / "附件1 指导教师信息与选队意愿.xls"
OUTPUT = DELIVERY / "data" / "processed" / "q1_teacher_scores.csv"
AWARDS = ["国家级一等奖", "国家级二等奖", "省级一等奖", "省级二等奖"]
AWARD_WEIGHTS = np.array([0.6049, 0.2801, 0.0693, 0.0458])


def extract_awards(text):
    result = {award: 0 for award in AWARDS}
    for scope, section in re.findall(r"(国家级|省级)(.*?)(?=国家级|省级|$)", str(text)):
        for rank in ("一等奖", "二等奖"):
            match = re.search(rf"{rank}(\d+)项", section)
            if match:
                result[f"{scope}{rank}"] = int(match.group(1))
    return result


def normalize(values):
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min())


def main():
    raw = pd.read_excel(SOURCE, header=None)
    data = raw.iloc[2:37, :8].copy()
    data.columns = ["教师编号", "性别", "学位", "职称", "主要研究方向", "累计指导时间", "获奖情况", "选队条件"]
    data["教师编号"] = data["教师编号"].astype(int)
    data["累计指导时间"] = data["累计指导时间"].astype(float)
    awards = data.pop("获奖情况").apply(extract_awards).apply(pd.Series)
    data = pd.concat([data, awards], axis=1)

    award_matrix = data[AWARDS].to_numpy(dtype=float)
    data["加权获奖成果"] = award_matrix @ AWARD_WEIGHTS
    data["指导效率"] = data["加权获奖成果"] / data["累计指导时间"]
    normalized_awards = np.column_stack([normalize(award_matrix[:, column]) for column in range(4)])
    normalized_result = normalized_awards @ AWARD_WEIGHTS
    data["综合得分"] = 100 * (normalize(data["指导效率"]) / 3 + 2 * normalized_result / 3)

    data = data.sort_values(["综合得分", "累计指导时间", "教师编号"], ascending=[False, False, True]).reset_index(drop=True)
    data["排名"] = np.arange(1, len(data) + 1)
    data["指导水平"] = np.select(
        [data["排名"] <= 7, data["排名"] <= 17, data["排名"] <= 28],
        ["I", "II", "III"], default="IV"
    )
    columns = ["排名", "教师编号", "性别", "学位", "职称", "主要研究方向", "累计指导时间",
               *AWARDS, "加权获奖成果", "指导效率", "综合得分", "指导水平", "选队条件"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data[columns].to_csv(OUTPUT, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(data[["排名", "教师编号", "综合得分", "指导水平"]].to_string(index=False))


if __name__ == "__main__":
    main()
