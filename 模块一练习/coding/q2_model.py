from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
TEACHER_SOURCE = ROOT / "2026A题附件" / "附件1 指导教师信息与选队意愿.xls"
SCORE_SOURCE = ROOT / "第一问教师评分结果.csv"
DATA_OUTPUT = ROOT / "第二问建模数据.csv"
IMPORTANCE_OUTPUT = ROOT / "第二问因素重要性.csv"
STATS_OUTPUT = ROOT / "第二问分类统计.csv"
FIGURE_OUTPUT = ROOT / "第二问因素重要性.pdf"

RAW_FEATURES = ["性别", "学位", "职称", "主要研究方向"]
FEATURES = ["性别编码", "学位编码", "职称编码", "研究方向编码"]
FACTOR_NAMES = {
    "性别编码": "性别",
    "学位编码": "学位",
    "职称编码": "职称",
    "研究方向编码": "研究方向",
}
DIRECTION_GROUPS = {
    "泛函分析": (0, "基础数学与方程类"),
    "模糊数学": (0, "基础数学与方程类"),
    "微分方程": (0, "基础数学与方程类"),
    "概率统计": (0, "基础数学与方程类"),
    "优化、图与网络": (1, "优化运筹类"),
    "优化与决策": (1, "优化运筹类"),
    "运筹与控制": (1, "优化运筹类"),
    "算法设计与分析": (2, "算法数据类"),
    "数据分析与可视化": (2, "算法数据类"),
    "时间序列": (2, "算法数据类"),
    "数值计算": (2, "算法数据类"),
    "半群与代数": (3, "代数密码类"),
    "密码学": (3, "代数密码类"),
    "图像处理与小波分析": (4, "图像处理类"),
}
LEVEL_ORDER = ["I", "II", "III", "IV"]
RANDOM_STATE = 2026


def build_model(seed):
    encoder = ColumnTransformer(
        [("direction", OneHotEncoder(handle_unknown="ignore"), ["研究方向编码"])],
        remainder="passthrough",
    )
    forest = RandomForestRegressor(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("encoder", encoder), ("forest", forest)])


def load_data():
    raw = pd.read_excel(TEACHER_SOURCE, header=None)
    teachers = raw.iloc[2:37, [0, 1, 2, 3, 4]].copy()
    teachers.columns = ["教师编号", *RAW_FEATURES]
    teachers["教师编号"] = teachers["教师编号"].astype(int)
    teachers["性别编码"] = teachers["性别"].map({"女": 0, "男": 1})
    teachers["学位编码"] = teachers["学位"].map({"硕士": 0, "博士": 1})
    teachers["职称编码"] = teachers["职称"].map({"讲师": 1, "副教授": 2, "教授": 3})
    teachers["研究方向编码"] = teachers["主要研究方向"].map(
        {direction: group[0] for direction, group in DIRECTION_GROUPS.items()}
    )
    teachers["研究方向大类"] = teachers["主要研究方向"].map(
        {direction: group[1] for direction, group in DIRECTION_GROUPS.items()}
    )
    if teachers[FEATURES].isna().any().any():
        missing = teachers.loc[teachers[FEATURES].isna().any(axis=1), RAW_FEATURES]
        raise ValueError(f"存在未定义编码的教师属性：\n{missing}")
    teachers[FEATURES] = teachers[FEATURES].astype(int)
    scores = pd.read_csv(SCORE_SOURCE, encoding="utf-8-sig")
    return teachers.merge(
        scores[["教师编号", "综合得分", "指导水平"]], on="教师编号", validate="one_to_one"
    )


def cross_validated_analysis(data):
    x = data[FEATURES].copy()
    y = data["综合得分"].to_numpy(dtype=float)
    splitter = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    prediction_sum = np.zeros(len(data))
    prediction_count = np.zeros(len(data))
    importance = {feature: [] for feature in FEATURES}
    rng = np.random.default_rng(RANDOM_STATE)

    for fold, (train_index, test_index) in enumerate(splitter.split(x), start=1):
        model = build_model(RANDOM_STATE + fold)
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        prediction_sum[test_index] += prediction
        prediction_count[test_index] += 1
        baseline_mse = mean_squared_error(y_test, prediction)

        for feature in FEATURES:
            deltas = []
            for _ in range(10):
                permuted = x_test.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                permuted_mse = mean_squared_error(y_test, model.predict(permuted))
                deltas.append(permuted_mse - baseline_mse)
            importance[feature].append(np.mean(deltas))

    oof_prediction = prediction_sum / prediction_count
    metrics = {
        "MAE": mean_absolute_error(y, oof_prediction),
        "RMSE": np.sqrt(mean_squared_error(y, oof_prediction)),
        "R2": r2_score(y, oof_prediction),
    }
    raw_importance = pd.DataFrame(
        {
            "因素": [FACTOR_NAMES[feature] for feature in FEATURES],
            "置换后MSE增量": [np.mean(importance[feature]) for feature in FEATURES],
            "标准差": [np.std(importance[feature], ddof=1) for feature in FEATURES],
        }
    )
    positive = raw_importance["置换后MSE增量"].clip(lower=0)
    raw_importance["相对重要性"] = positive / positive.sum()
    raw_importance = raw_importance.sort_values("相对重要性", ascending=False).reset_index(drop=True)
    raw_importance["排序"] = np.arange(1, len(raw_importance) + 1)
    return oof_prediction, metrics, raw_importance


def category_statistics(data):
    rows = []
    for feature in ["性别", "学位", "职称", "研究方向大类"]:
        for category, group in data.groupby(feature, sort=False):
            level_counts = group["指导水平"].value_counts()
            rows.append(
                {
                    "因素": feature,
                    "类别": category,
                    "人数": len(group),
                    "平均得分": group["综合得分"].mean(),
                    "得分标准差": group["综合得分"].std(ddof=1) if len(group) > 1 else np.nan,
                    **{f"{level}级人数": int(level_counts.get(level, 0)) for level in LEVEL_ORDER},
                }
            )
    return pd.DataFrame(rows)


def create_figure(importance):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    plot_data = importance.sort_values("相对重要性")
    colors = ["#8fa9a0", "#72928b", "#4f7770", "#285b55"]
    bars = axis.barh(plot_data["因素"], 100 * plot_data["相对重要性"], color=colors)
    axis.set_xlabel("交叉验证分组置换相对重要性（%）")
    axis.set_xlim(0, max(50, 100 * plot_data["相对重要性"].max() * 1.18))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in 100 * plot_data["相对重要性"]], padding=4)
    figure.tight_layout()
    figure.savefig(FIGURE_OUTPUT, bbox_inches="tight")
    plt.close(figure)


def main():
    data = load_data()
    predictions, metrics, importance = cross_validated_analysis(data)
    data["交叉验证预测得分"] = predictions
    data["预测绝对误差"] = np.abs(data["综合得分"] - predictions)
    stats = category_statistics(data)

    data.to_csv(DATA_OUTPUT, index=False, encoding="utf-8-sig", float_format="%.6f")
    importance.to_csv(IMPORTANCE_OUTPUT, index=False, encoding="utf-8-sig", float_format="%.6f")
    stats.to_csv(STATS_OUTPUT, index=False, encoding="utf-8-sig", float_format="%.6f")
    create_figure(importance)

    print("交叉验证指标:", {name: round(value, 6) for name, value in metrics.items()})
    print("\n因素重要性:\n", importance.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n分类统计:\n", stats.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
