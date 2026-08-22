"""Problem 1: cross-analysis of named clusters and TOPSIS pressure levels."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "problem1_evaluation_clustering"
OUT = ROOT / "results" / "problem1_cross_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260822
N_PERM = 9999

CLUSTER_NAMES = {
    "I类：低综合压力型",
    "II类：高规模—高强度工业型",
    "III类：中高规模—工业主导型",
}
LEVEL_ORDER = ["I-高压力", "II-较高压力", "III-中压力", "IV-低压力"]
LEVEL_CN = {
    "I-高压力": "I级：高压力",
    "II-较高压力": "II级：较高压力",
    "III-中压力": "III级：中压力",
    "IV-低压力": "IV级：低压力",
}


def cramers_v(table: pd.DataFrame) -> float:
    obs = table.to_numpy(dtype=float)
    n = obs.sum()
    expected = obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / n
    chi2 = ((obs - expected) ** 2 / expected).sum()
    return float(np.sqrt((chi2 / n) / min(obs.shape[0] - 1, obs.shape[1] - 1)))


def chi2_stat(table: pd.DataFrame) -> float:
    obs = table.to_numpy(dtype=float)
    n = obs.sum()
    expected = obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / n
    return float(((obs - expected) ** 2 / expected).sum())


def permutation_p(clusters: np.ndarray, levels: np.ndarray, n_perm: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs_table = pd.crosstab(clusters, levels)
    obs = chi2_stat(obs_table)
    values = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        values[b] = chi2_stat(pd.crosstab(clusters, rng.permutation(levels)))
    p = (1.0 + np.sum(values >= obs)) / (n_perm + 1.0)
    return obs, float(p)


def named_clusters(assignments: pd.DataFrame, score_col: str) -> pd.DataFrame:
    assignments = assignments.copy()
    means = assignments.groupby("cluster")[score_col].mean().sort_values()
    ordered_names = ["I类：低综合压力型", "III类：中高规模—工业主导型", "II类：高规模—高强度工业型"]
    mapping = {int(cluster): ordered_names[i] for i, cluster in enumerate(means.index)}
    assignments["cluster_name"] = assignments["cluster"].map(mapping)
    assignments["cluster_name_order"] = assignments["cluster_name"].map({name: i for i, name in enumerate(ordered_names, start=1)})
    return assignments


def analyze(rep: str, score_file: str, score_col: str, level_col: str, level_cn_col: str, output_tag: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scores = pd.read_csv(BASE / score_file, encoding="utf-8-sig")
    assignments = pd.read_csv(BASE / "cluster_assignments.csv", encoding="utf-8-sig")
    assignments = assignments[(assignments["representation"] == rep) & (assignments["method"] == "Kmeans") & (assignments["k"] == 3)][["province", "cluster"]]
    scores = scores[["province", score_col, level_col]].rename(columns={level_col: "pressure_level"})
    d = scores.merge(assignments, on="province", validate="one_to_one")
    d = named_clusters(d, score_col)
    d["pressure_level_cn"] = d["pressure_level"].map(LEVEL_CN)
    d.to_csv(OUT / f"named_assignments_{output_tag}.csv", index=False, encoding="utf-8-sig")

    table = pd.crosstab(d["cluster_name"], d["pressure_level"])
    table = table.reindex(columns=LEVEL_ORDER, fill_value=0)
    table.to_csv(OUT / f"cluster_pressure_crosstab_{output_tag}.csv", encoding="utf-8-sig")

    row = table.div(table.sum(axis=1), axis=0)
    row.columns = [LEVEL_CN[c] for c in row.columns]
    row.to_csv(OUT / f"cluster_pressure_row_proportions_{output_tag}.csv", encoding="utf-8-sig")

    stats = d.groupby("cluster_name").agg(
        province_count=("province", "size"),
        score_mean=(score_col, "mean"),
        score_median=(score_col, "median"),
        score_min=(score_col, "min"),
        score_max=(score_col, "max"),
    ).reset_index()
    stats["dominant_pressure_level"] = [table.loc[name].idxmax() for name in stats["cluster_name"]]
    stats["dominant_level_share"] = [table.loc[name].max() / table.loc[name].sum() for name in stats["cluster_name"]]
    stats.to_csv(OUT / f"cluster_pressure_summary_{output_tag}.csv", index=False, encoding="utf-8-sig")

    chi2, p = permutation_p(d["cluster_name"].to_numpy(), d["pressure_level"].to_numpy(), N_PERM, SEED)
    association = {
        "representation": rep,
        "cluster_method": "Kmeans",
        "k": 3,
        "score": score_col,
        "pressure_level": level_col,
        "chi2_stat": chi2,
        "cramers_v": cramers_v(table),
        "permutation_p": p,
        "permutations": N_PERM,
    }
    return d, table, association


def main() -> None:
    d4, t4, a4 = analyze("4d_direct", "topsis_scores_4d.csv", "topsis_score_4d", "provisional_level_4d", "provisional_level_4d", "4d_primary")
    d3, t3, a3 = analyze("3d_balanced", "topsis_scores_3d_balanced.csv", "topsis_score_3d_balanced", "provisional_level_3d", "provisional_level_3d", "3d_robustness")
    pd.DataFrame([a4, a3]).to_csv(OUT / "cluster_pressure_association.csv", index=False, encoding="utf-8-sig")

    # Policy subtypes for the primary 4D result.
    policy = d4[["province", "cluster_name", "pressure_level", "pressure_level_cn", "topsis_score_4d"]].copy()
    def subtype(row):
        c, l = row["cluster_name"], row["pressure_level"]
        if c == "I类：低综合压力型":
            return "低压力维持/绿色示范"
        if c == "II类：高规模—高强度工业型":
            return "重点减排/深度工业脱碳"
        return {
            "I-高压力": "工业主导—高压力重点治理",
            "II-较高压力": "工业主导—较高压力结构优化",
            "III-中压力": "工业主导—中压力技术改造",
            "IV-低压力": "工业主导—低压力协同发展",
        }[l]
    policy["policy_subtype"] = policy.apply(subtype, axis=1)
    policy.sort_values("topsis_score_4d", ascending=False).to_csv(OUT / "primary_policy_subtypes_4d.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "primary": "4d_direct TOPSIS + 4d_direct Kmeans K=3",
        "robustness": "3d_balanced TOPSIS + 3d_balanced Kmeans K=3",
        "cluster_names": sorted(CLUSTER_NAMES),
        "level_order": LEVEL_ORDER,
        "permutation_seed": SEED,
        "permutations": N_PERM,
        "image_generation": False,
    }
    (OUT / "cross_analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("4D primary crosstab")
    print(t4.to_string())
    print("3D robustness crosstab")
    print(t3.to_string())
    print(pd.DataFrame([a4, a3]).to_string(index=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
