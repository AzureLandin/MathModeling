from pathlib import Path


DELIVERY = Path(__file__).resolve().parents[1]
import q4_model as model


model.TEACHER_SOURCE = DELIVERY / "data" / "raw" / "附件1 指导教师信息与选队意愿.xls"
model.TEAM_SOURCE = DELIVERY / "data" / "raw" / "附件2 建模队信息.xls"
model.OUTPUT = DELIVERY / "results"


if __name__ == "__main__":
    model.main()
