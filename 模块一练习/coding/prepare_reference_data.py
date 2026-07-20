from pathlib import Path

import pandas as pd


DELIVERY = Path(__file__).resolve().parents[1]
OUTPUT = DELIVERY / "data" / "processed"
RAW = DELIVERY / "data" / "raw"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    teacher_raw = pd.read_excel(RAW / "附件1 指导教师信息与选队意愿.xls", header=None)
    teachers = teacher_raw.iloc[2:37, :8].copy()
    teachers.columns = ["教师编号", "性别", "学位", "职称", "主要研究方向", "累计指导时间", "累计获奖情况", "选队条件"]
    teachers.to_csv(OUTPUT / "teacher_reference.csv", index=False, encoding="utf-8-sig")

    team_raw = pd.read_excel(RAW / "附件2 建模队信息.xls", header=None)
    rows = []
    for source_row in range(3, 178):
        row = team_raw.iloc[source_row]
        result = {"队号": int(row.iloc[0])}
        for member, start in enumerate((1, 7, 13), 1):
            for name, offset in zip(("性别", "特长", "专业", "年级", "参赛经历", "专业排名"), range(6)):
                result[f"队员{member}_{name}"] = row.iloc[start + offset]
        rows.append(result)
    pd.DataFrame(rows).to_csv(OUTPUT / "team_reference.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"年份": 2020, "国家级一等奖": 29, "国家级二等奖": 80, "省级一等奖": 754, "省级二等奖": 1053},
        {"年份": 2021, "国家级一等奖": 24, "国家级二等奖": 95, "省级一等奖": 755, "省级二等奖": 1097},
        {"年份": 2022, "国家级一等奖": 31, "国家级二等奖": 60, "省级一等奖": 614, "省级二等奖": 875},
        {"年份": 2023, "国家级一等奖": 29, "国家级二等奖": 94, "省级一等奖": 792, "省级二等奖": 1161},
    ]).to_csv(OUTPUT / "award_reference.csv", index=False, encoding="utf-8-sig")
    print("参考数据已生成:", OUTPUT)


if __name__ == "__main__":
    main()
