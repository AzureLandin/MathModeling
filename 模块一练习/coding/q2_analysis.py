from pathlib import Path


DELIVERY = Path(__file__).resolve().parents[1]
import q2_model as model


model.TEACHER_SOURCE = DELIVERY / "data" / "raw" / "附件1 指导教师信息与选队意愿.xls"
model.SCORE_SOURCE = DELIVERY / "data" / "processed" / "q1_teacher_scores.csv"
model.DATA_OUTPUT = DELIVERY / "data" / "processed" / "q2_model_data.csv"
model.IMPORTANCE_OUTPUT = DELIVERY / "results" / "q2_factor_importance.csv"
model.STATS_OUTPUT = DELIVERY / "results" / "q2_category_statistics.csv"
model.FIGURE_OUTPUT = DELIVERY / "results" / "q2_factor_importance.pdf"


if __name__ == "__main__":
    model.main()
