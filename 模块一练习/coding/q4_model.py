from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent
TEACHER_SOURCE = ROOT / "2026A题附件" / "附件1 指导教师信息与选队意愿.xls"
TEAM_SOURCE = ROOT / "2026A题附件" / "附件2 建模队信息.xls"
OUTPUT = ROOT / "第四问成果"

# 严格沿用“已完成”问题3报告中的排序和最终配额。
SELECTION_ORDER = [7, 2, 29, 21, 23, 17, 4, 13, 22, 9, 16, 32, 27, 28,
                   19, 35, 11, 14, 31, 25, 24, 33, 30, 18, 34, 20, 10, 26,
                   8, 12, 15, 1, 6, 5, 3]
QUOTAS = {teacher: (10 if teacher in {7, 2} else
                    7 if teacher in {29, 21, 23, 17, 4, 13, 22, 9, 16} else
                    5 if teacher in {32, 27, 28, 19, 35, 11, 14, 31, 25, 24} else 3)
          for teacher in SELECTION_ORDER}
Q3_SCORES = [85.72, 77.10, 58.84, 54.91, 52.73, 41.04, 40.36, 39.02, 33.36,
             31.90, 30.66, 30.52, 28.40, 23.86, 19.29, 19.29, 17.93, 17.34,
             16.06, 15.52, 15.04, 14.67, 14.45, 13.51, 13.51, 12.52, 12.35,
             11.42, 10.69, 10.34, 5.70, 5.01, 5.01, 1.59, 0.00]

PREFERRED_MAJORS = {
    1: {"土木", "环工"}, 2: {"计算机", "应数"}, 6: {"计算机", "应数"},
    9: {"自动化", "电气", "土木"}, 12: {"计算机", "应数"},
    18: {"土木", "环工"}, 20: {"计算机", "土木"},
    23: {"电气", "自动化", "土木"}, 29: {"计算机", "应数"},
    31: {"应数", "计算机"}, 32: {"应数", "环工"}, 35: {"电气", "土木"},
}


def load_teachers():
    raw = pd.read_excel(TEACHER_SOURCE, header=None)
    teachers = raw.iloc[2:37, [0, 4, 7]].copy()
    teachers.columns = ["教师编号", "研究方向", "选队条件"]
    teachers["教师编号"] = teachers["教师编号"].astype(int)
    score_map = dict(zip(SELECTION_ORDER, Q3_SCORES))
    teachers["第三问得分"] = teachers["教师编号"].map(score_map)
    teachers["选队次序"] = teachers["教师编号"].map(
        {teacher: order for order, teacher in enumerate(SELECTION_ORDER, 1)}
    )
    teachers["指导队数"] = teachers["教师编号"].map(QUOTAS)
    return teachers.sort_values("选队次序").reset_index(drop=True)


def load_teams():
    raw = pd.read_excel(TEAM_SOURCE, header=None)
    rows = []
    for source_row in range(3, 178):
        row = raw.iloc[source_row]
        members = []
        for start in (1, 7, 13):
            rank, total = map(int, str(row.iloc[start + 5]).split("/"))
            members.append({
                "性别": str(row.iloc[start]), "特长": str(row.iloc[start + 1]),
                "专业": str(row.iloc[start + 2]), "年级": int(row.iloc[start + 3]),
                "参赛经历": int(row.iloc[start + 4]), "排名比": rank / total,
                "专业排名": str(row.iloc[start + 5]),
            })
        rows.append({"队号": int(row.iloc[0]), "队员": members})
    return rows


def parse_requirements(text):
    requirements = []
    if "参赛" in text:
        if "均具有参赛经历" in text or "均有参赛经历" in text or "队员均有参赛经历" in text:
            requirements.append(("三人均有参赛经历", lambda t: sum(m["参赛经历"] > 0 for m in t) / 3))
        elif "至少两人" in text or "两人以上" in text or "队员至少两人" in text:
            requirements.append(("至少两人有参赛经历", lambda t: min(sum(m["参赛经历"] > 0 for m in t) / 2, 1)))
        elif "至少一人" in text or "至少有一人" in text or "至少一名" in text:
            requirements.append(("至少一人有参赛经历", lambda t: min(sum(m["参赛经历"] > 0 for m in t), 1)))

    rank_match = re.search(r"排名[^，、；]*?(\d+)%", text)
    if rank_match and "至少一人专业排名" not in text:
        threshold = int(rank_match.group(1)) / 100
        requirements.append((f"三人排名前{rank_match.group(1)}%", lambda t, p=threshold: sum(m["排名比"] <= p for m in t) / 3))
    elif "至少一人专业排名前20%" in text:
        requirements.append(("至少一人排名前20%", lambda t: min(sum(m["排名比"] <= .2 for m in t), 1)))

    if "均为二年级或以上" in text or "均为二年级以上" in text or (
        "二年级以上" in text and "至少一人" not in text and "至少一名" not in text
    ):
        requirements.append(("三人均为二年级以上", lambda t: sum(m["年级"] >= 2 for m in t) / 3))
    elif "至少一人为二年级或以上" in text or "至少一人二年级或以上" in text:
        requirements.append(("至少一人为二年级以上", lambda t: min(sum(m["年级"] >= 2 for m in t), 1)))

    if "队员专业不同" in text or "专业不同" in text:
        requirements.append(("三人专业不同", lambda t: 1 if len({m["专业"] for m in t}) == 3 else (0.5 if len({m["专业"] for m in t}) == 2 else 0)))
    elif "专业不全相同" in text:
        requirements.append(("专业不全相同", lambda t: 1 if len({m["专业"] for m in t}) >= 2 else 0))
    return requirements


def team_quality(members):
    rank_quality = np.mean([1 - m["排名比"] for m in members])
    experience = np.mean([min(m["参赛经历"], 2) / 2 for m in members])
    skills = len({m["特长"] for m in members} & {"建模", "编程", "写作"}) / 3
    grade = np.mean([(m["年级"] - 1) / 2 for m in members])
    return 100 * (0.35 * rank_quality + 0.30 * experience + 0.20 * skills + 0.15 * grade)


def evaluate_pair(teacher_id, condition, members):
    requirements = parse_requirements(condition)
    values = [function(members) for _, function in requirements]
    condition_rate = float(np.mean(values)) if values else 1.0
    full_satisfaction = all(value >= 1 - 1e-12 for value in values)
    preference_parts = []
    preferred = PREFERRED_MAJORS.get(teacher_id)
    if preferred:
        preference_parts.append(sum(m["专业"] in preferred for m in members) / 3)
    if "二年级或以上优先" in condition:
        preference_parts.append(sum(m["年级"] >= 2 for m in members) / 3)
    preference_rate = float(np.mean(preference_parts)) if preference_parts else 1.0
    quality = team_quality(members)
    score = 70 * condition_rate + 10 * preference_rate + 0.20 * quality
    return {
        "匹配得分": score, "条件满足率": condition_rate,
        "条件完全满足": full_satisfaction, "专业偏好满足率": preference_rate,
        "队伍质量": quality,
        "条件明细": "；".join(f"{name}:{value:.2f}" for (name, _), value in zip(requirements, values)) or "无限制",
    }


def build_score_table(teachers, teams):
    records = []
    team_lookup = {team["队号"]: team for team in teams}
    for teacher in teachers.itertuples(index=False):
        for team_id, team in team_lookup.items():
            records.append({"教师编号": teacher.教师编号, "队号": team_id,
                            **evaluate_pair(teacher.教师编号, teacher.选队条件, team["队员"])})
    return pd.DataFrame(records)


def build_schedule():
    schedule = []
    pick_index = 1
    remaining = QUOTAS.copy()
    while sum(remaining.values()):
        for teacher in SELECTION_ORDER:
            if remaining[teacher] > 0:
                schedule.append((pick_index, teacher))
                remaining[teacher] -= 1
                pick_index += 1
    return schedule


def sequential_assignment(score_table):
    lookup = score_table.set_index(["教师编号", "队号"])
    unassigned = set(score_table["队号"].unique())
    rows = []
    for pick_index, teacher in build_schedule():
        candidates = []
        for team in unassigned:
            pair = lookup.loc[(teacher, team)]
            candidates.append((bool(pair["条件完全满足"]), pair["匹配得分"], pair["条件满足率"],
                               pair["专业偏好满足率"], pair["队伍质量"], -team, team))
        chosen_team = max(candidates)[-1]
        pair = lookup.loc[(teacher, chosen_team)]
        rows.append({"分配序号": pick_index, "教师编号": teacher, "队号": chosen_team, **pair.to_dict()})
        unassigned.remove(chosen_team)
    return pd.DataFrame(rows)


def global_assignment(score_table):
    slots = [teacher for teacher in SELECTION_ORDER for _ in range(QUOTAS[teacher])]
    teams = sorted(score_table["队号"].unique())
    lookup = score_table.set_index(["教师编号", "队号"])
    # 大常数保证先最大化完全满足条件的配对数，再最大化匹配得分。
    utility = np.array([[1_000_000 * bool(lookup.loc[(teacher, team), "条件完全满足"]) +
                         lookup.loc[(teacher, team), "匹配得分"] for team in teams] for teacher in slots])
    slot_index, team_index = linear_sum_assignment(-utility)
    rows = []
    for s, t in zip(slot_index, team_index):
        teacher, team = slots[s], teams[t]
        rows.append({"教师编号": teacher, "队号": team, **lookup.loc[(teacher, team)].to_dict()})
    return pd.DataFrame(rows)


def enrich_assignment(assignment, teachers, teams):
    teacher_data = teachers.set_index("教师编号")
    team_lookup = {team["队号"]: team for team in teams}
    rows = []
    for row in assignment.itertuples(index=False):
        members = team_lookup[row.队号]["队员"]
        output = row._asdict()
        output.update({
            "选队次序": teacher_data.at[row.教师编号, "选队次序"],
            "教师研究方向": teacher_data.at[row.教师编号, "研究方向"],
            "教师选队条件": teacher_data.at[row.教师编号, "选队条件"],
            "队员专业": "/".join(m["专业"] for m in members),
            "队员年级": "/".join(str(m["年级"]) for m in members),
            "参赛经历": "/".join(str(m["参赛经历"]) for m in members),
            "专业排名": "/".join(m["专业排名"] for m in members),
            "队员特长": "/".join(m["特长"] for m in members),
        })
        rows.append(output)
    return pd.DataFrame(rows)


def summarize(assignment, teachers):
    summary = assignment.groupby("教师编号").agg(
        实际队数=("队号", "size"), 平均匹配得分=("匹配得分", "mean"),
        完全满足队数=("条件完全满足", "sum"), 平均条件满足率=("条件满足率", "mean"),
        平均专业偏好满足率=("专业偏好满足率", "mean"), 平均队伍质量=("队伍质量", "mean"),
    ).reset_index()
    summary = teachers.merge(summary, on="教师编号", validate="one_to_one")
    summary["完全满足比例"] = summary["完全满足队数"] / summary["实际队数"]
    return summary.sort_values("选队次序")


def validate(assignment, summary):
    assert len(assignment) == 175
    assert assignment["队号"].nunique() == 175
    assert set(assignment["队号"]) == set(range(1, 176))
    actual = summary.set_index("教师编号")["实际队数"].to_dict()
    assert actual == QUOTAS
    assert summary["实际队数"].between(3, 10).all()


def main():
    OUTPUT.mkdir(exist_ok=True)
    teachers, teams = load_teachers(), load_teams()
    score_table = build_score_table(teachers, teams)
    sequential = sequential_assignment(score_table)
    benchmark = global_assignment(score_table)
    sequential = enrich_assignment(sequential, teachers, teams)
    benchmark = enrich_assignment(benchmark, teachers, teams)
    summary = summarize(sequential, teachers)
    validate(sequential, summary)
    validate(benchmark, summarize(benchmark, teachers))

    sequential.to_csv(OUTPUT / "第四问最佳分配方案.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    summary.to_csv(OUTPUT / "第四问教师匹配汇总.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    benchmark.to_csv(OUTPUT / "全局优化对照方案.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    score_table.to_csv(OUTPUT / "教师队伍匹配得分矩阵.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    def metrics(data):
        return {"平均匹配得分": data["匹配得分"].mean(),
                "条件完全满足率": data["条件完全满足"].mean(),
                "平均条件满足率": data["条件满足率"].mean(),
                "平均队伍质量": data["队伍质量"].mean()}
    comparison = pd.DataFrame([{"方案": "第三问轮转顺序方案", **metrics(sequential)},
                               {"方案": "全局容量优化对照", **metrics(benchmark)}])
    comparison.to_csv(OUTPUT / "方案对比.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print(comparison.to_string(index=False))
    print("\n配额分布:", summary["实际队数"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
