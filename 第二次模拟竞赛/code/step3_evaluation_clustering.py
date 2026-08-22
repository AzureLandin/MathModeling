"""Problem 1: CRITIC-TOPSIS and K-means/Ward validation without figures.

Inputs are the four already-standardized pressure indicators from preprocessing.
The script compares a direct four-dimensional representation with a balanced
three-dimensional representation (scale, efficiency, industrial structure),
then evaluates K-means and Ward clustering for K=3,4,5.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "problem1_preprocessed" / "problem1_features_standardized_2022.csv"
RAW_INPUT = ROOT / "results" / "problem1_preprocessed" / "problem1_province_features_2022.csv"
OUT = ROOT / "results" / "problem1_evaluation_clustering"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260822
K_VALUES = [3, 4, 5]
N_INIT_MAIN = 200
N_INIT_STABILITY = 50
N_STABILITY = 100
MAX_ITER = 500

FOUR_COLS = [
    "z_emission_total",
    "z_carbon_intensity",
    "z_per_capita_emission",
    "z_industrial_emission_share",
]
FOUR_CN = {
    "z_emission_total": "碳排放总量",
    "z_carbon_intensity": "碳排放强度",
    "z_per_capita_emission": "人均碳排放",
    "z_industrial_emission_share": "工业碳排放占比",
}
THREE_COLS = ["scale_score", "efficiency_score", "industrial_structure_score"]
THREE_CN = {
    "scale_score": "规模维度",
    "efficiency_score": "效率维度",
    "industrial_structure_score": "工业结构维度",
}


def squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    first = int(rng.integers(0, n))
    chosen = [first]
    d2 = ((X - X[first]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        if total <= 0:
            candidates = [i for i in range(n) if i not in chosen]
            chosen.append(int(rng.choice(candidates)))
        else:
            probs = d2 / total
            nxt = int(rng.choice(n, p=probs))
            while nxt in chosen:
                nxt = int(rng.choice(n, p=probs))
            chosen.append(nxt)
        d2 = np.minimum(d2, ((X - X[chosen[-1]]) ** 2).sum(axis=1))
    return X[chosen].copy()


def kmeans_once(X: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float, int]:
    centers = kmeans_plus_plus(X, k, rng)
    labels = np.full(len(X), -1, dtype=int)
    for it in range(MAX_ITER):
        dist = squared_distances(X, centers)
        new_labels = dist.argmin(axis=1)
        new_centers = np.empty_like(centers)
        for c in range(k):
            members = X[new_labels == c]
            if len(members) == 0:
                # Reinitialize an empty centroid at the point currently farthest
                # from its assigned center.
                farthest = int(np.argmax(dist.min(axis=1)))
                new_centers[c] = X[farthest]
            else:
                new_centers[c] = members.mean(axis=0)
        shift = float(np.sqrt(((new_centers - centers) ** 2).sum()))
        centers = new_centers
        if np.array_equal(new_labels, labels) or shift <= 1e-10:
            labels = new_labels
            break
        labels = new_labels
    sse = float(((X - centers[labels]) ** 2).sum())
    return labels, centers, sse, it + 1


def kmeans_best(X: np.ndarray, k: int, seed: int, n_init: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    best = None
    for r in range(n_init):
        rng = np.random.default_rng(seed + 1009 * r + 17 * k)
        candidate = kmeans_once(X, k, rng)
        if best is None or candidate[2] < best[2]:
            best = candidate
    assert best is not None
    return best


def ward_labels(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > k:
        best_pair = None
        best_delta = float("inf")
        for a in range(len(clusters)):
            A = np.asarray(clusters[a], dtype=int)
            mu_a = X[A].mean(axis=0)
            for b in range(a + 1, len(clusters)):
                B = np.asarray(clusters[b], dtype=int)
                mu_b = X[B].mean(axis=0)
                delta = (len(A) * len(B) / (len(A) + len(B))) * float(((mu_a - mu_b) ** 2).sum())
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (a, b)
        a, b = best_pair
        clusters[a].extend(clusters[b])
        del clusters[b]
    labels = np.empty(len(X), dtype=int)
    centers = np.empty((k, X.shape[1]), dtype=float)
    for c, members in enumerate(clusters):
        labels[members] = c
        centers[c] = X[members].mean(axis=0)
    sse = float(((X - centers[labels]) ** 2).sum())
    return labels, centers, sse


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(np.maximum(d2, 0.0))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    D = pairwise_distances(X)
    vals = []
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        other_means = []
        for c in sorted(set(labels)):
            if c == labels[i]:
                continue
            mask = labels == c
            other_means.append(D[i, mask].mean())
        b = min(other_means) if other_means else 0.0
        vals.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(vals))


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    n, _ = X.shape
    groups = sorted(set(labels))
    k = len(groups)
    overall = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in groups:
        members = X[labels == c]
        center = members.mean(axis=0)
        between += len(members) * float(((center - overall) ** 2).sum())
        within += float(((members - center) ** 2).sum())
    return float((between / (k - 1)) / (within / (n - k))) if k > 1 and n > k and within > 0 else float("nan")


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    groups = sorted(set(labels))
    centers = np.vstack([X[labels == c].mean(axis=0) for c in groups])
    scatters = np.array([np.linalg.norm(X[labels == c] - centers[i], axis=1).mean() for i, c in enumerate(groups)])
    C = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    ratios = np.full_like(C, -np.inf)
    for i in range(len(groups)):
        for j in range(len(groups)):
            if i != j and C[i, j] > 0:
                ratios[i, j] = (scatters[i] + scatters[j]) / C[i, j]
    return float(np.mean(np.max(ratios, axis=1)))


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=int)
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2
    nij = comb(table).sum()
    ai_sum = comb(table.sum(axis=1)).sum()
    bi_sum = comb(table.sum(axis=0)).sum()
    n = len(a)
    total = comb(n)
    expected = ai_sum * bi_sum / total if total else 0.0
    max_index = 0.5 * (ai_sum + bi_sum)
    denom = max_index - expected
    return float((nij - expected) / denom) if denom != 0 else 1.0


def critic_weights(X: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    sigma = X.std(axis=0, ddof=1)
    corr = np.corrcoef(X, rowvar=False)
    conflict = (1.0 - corr).sum(axis=1)
    information = sigma * conflict
    weights = information / information.sum()
    out = pd.DataFrame({
        "feature": names,
        "feature_cn": [FOUR_CN.get(x, THREE_CN.get(x, x)) for x in names],
        "std": sigma,
        "conflict_sum": conflict,
        "critic_information": information,
        "critic_weight": weights,
    })
    return out, weights


def topsis_pressure(X: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # All indicators are pressure-type: max is the worst/positive-pressure ideal,
    # min is the best/negative-pressure ideal.
    weighted = X * weights[None, :]
    positive = weighted.max(axis=0)
    negative = weighted.min(axis=0)
    d_positive = np.sqrt(((weighted - positive) ** 2).sum(axis=1))
    d_negative = np.sqrt(((weighted - negative) ** 2).sum(axis=1))
    score = d_negative / (d_positive + d_negative)
    return score, d_positive, d_negative


def stability_for_kmeans(X: np.ndarray, k: int, base_labels: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed + 777 * k)
    aris = []
    for b in range(N_STABILITY):
        noise = rng.normal(loc=0.0, scale=0.02, size=X.shape)
        labels, _, _, _ = kmeans_best(X + noise, k, seed + 5000 + b, N_INIT_STABILITY)
        aris.append(adjusted_rand_index(base_labels, labels))
    aris = np.asarray(aris)
    return {
        "stability_mean_ari": float(aris.mean()),
        "stability_median_ari": float(np.median(aris)),
        "stability_min_ari": float(aris.min()),
        "stability_q10_ari": float(np.quantile(aris, 0.10)),
        "stability_n": N_STABILITY,
    }


def assign_levels(score: pd.Series) -> pd.Series:
    # Provisional four-level pressure grading by rank quartiles; final thresholds
    # should be fixed only after selecting the final representation.
    ranks = score.rank(method="first", ascending=True)
    return pd.cut(ranks, bins=[0, 7.5, 15, 22.5, 30], labels=["IV-低压力", "III-中压力", "II-较高压力", "I-高压力"], include_lowest=True)


def main() -> None:
    std = pd.read_csv(INPUT, encoding="utf-8-sig")
    raw = pd.read_csv(RAW_INPUT, encoding="utf-8-sig")
    if len(std) != 30 or std["province"].nunique() != 30:
        raise ValueError("标准化表不是30个唯一省份")
    if set(std.province) != set(raw.province):
        raise ValueError("标准化表与原始表省份集合不一致")

    X4 = std[FOUR_COLS].to_numpy(dtype=float)
    # Balanced 3D representation: preserve all four original features while
    # preventing the two efficiency indicators from receiving two full axes.
    X3 = np.column_stack([
        std["z_emission_total"].to_numpy(float),
        std[["z_carbon_intensity", "z_per_capita_emission"]].mean(axis=1).to_numpy(float),
        std["z_industrial_emission_share"].to_numpy(float),
    ])

    w4_df, w4 = critic_weights(X4, FOUR_COLS)
    w4_df.to_csv(OUT / "critic_weights_4d.csv", index=False, encoding="utf-8-sig")
    score4, dp4, dn4 = topsis_pressure(X4, w4)
    score4_df = pd.DataFrame({
        "province": std["province"],
        "topsis_score_4d": score4,
        "distance_to_pressure_ideal_4d": dp4,
        "distance_to_low_pressure_ideal_4d": dn4,
    }).sort_values("topsis_score_4d", ascending=False)
    score4_df["provisional_level_4d"] = assign_levels(score4_df["topsis_score_4d"])
    score4_df.to_csv(OUT / "topsis_scores_4d.csv", index=False, encoding="utf-8-sig")

    # For the 3D balanced representation, use equal dimension weights to make
    # scale, efficiency and industrial structure equally important.
    w3 = np.ones(3) / 3.0
    w3_df = pd.DataFrame({
        "feature": THREE_COLS,
        "feature_cn": [THREE_CN[x] for x in THREE_COLS],
        "critic_weight": w3,
        "weight_note": "维度间等权；效率维度由强度和人均排放均值构成",
    })
    w3_df.to_csv(OUT / "critic_weights_3d_balanced.csv", index=False, encoding="utf-8-sig")
    score3, dp3, dn3 = topsis_pressure(X3, w3)
    score3_df = pd.DataFrame({
        "province": std["province"],
        "topsis_score_3d_balanced": score3,
        "distance_to_pressure_ideal_3d": dp3,
        "distance_to_low_pressure_ideal_3d": dn3,
    }).sort_values("topsis_score_3d_balanced", ascending=False)
    score3_df["provisional_level_3d"] = assign_levels(score3_df["topsis_score_3d_balanced"])
    score3_df.to_csv(OUT / "topsis_scores_3d_balanced.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    assignment_rows = []
    center_rows = []
    stability_rows = []
    representations = [("4d_direct", X4, score4), ("3d_balanced", X3, score3)]
    for rep_name, X, score in representations:
        for k in K_VALUES:
            km_labels, km_centers, km_sse, km_iter = kmeans_best(X, k, SEED, N_INIT_MAIN)
            wd_labels, wd_centers, wd_sse = ward_labels(X, k)
            km_stability = stability_for_kmeans(X, k, km_labels, SEED)
            for method, labels, centers, sse, iterations in [
                ("Kmeans", km_labels, km_centers, km_sse, km_iter),
                ("Ward", wd_labels, wd_centers, wd_sse, np.nan),
            ]:
                metric_rows.append({
                    "representation": rep_name,
                    "method": method,
                    "k": k,
                    "sse": sse,
                    "silhouette": silhouette_score(X, labels),
                    "calinski_harabasz": calinski_harabasz(X, labels),
                    "davies_bouldin": davies_bouldin(X, labels),
                    "iterations": iterations,
                    "kmeans_stability_mean_ari": km_stability["stability_mean_ari"] if method == "Kmeans" else np.nan,
                    "kmeans_stability_q10_ari": km_stability["stability_q10_ari"] if method == "Kmeans" else np.nan,
                    "kmeans_stability_min_ari": km_stability["stability_min_ari"] if method == "Kmeans" else np.nan,
                })
                for i, p in enumerate(std["province"]):
                    assignment_rows.append({
                        "province": p,
                        "representation": rep_name,
                        "method": method,
                        "k": k,
                        "cluster": int(labels[i]) + 1,
                        "topsis_score_4d": float(score4[i]),
                        "topsis_score_3d_balanced": float(score3[i]),
                    })
                for c in range(k):
                    center_rows.append({
                        "representation": rep_name,
                        "method": method,
                        "k": k,
                        "cluster": c + 1,
                        **{f"center_{j+1}": float(v) for j, v in enumerate(centers[c])},
                        "cluster_size": int((labels == c).sum()),
                    })
            metric_rows[-2]["kmeans_ward_ari"] = adjusted_rand_index(km_labels, wd_labels)
            metric_rows[-1]["kmeans_ward_ari"] = adjusted_rand_index(km_labels, wd_labels)
            stability_rows.append({"representation": rep_name, "k": k, **km_stability})

    pd.DataFrame(metric_rows).to_csv(OUT / "cluster_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(assignment_rows).to_csv(OUT / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(center_rows).to_csv(OUT / "cluster_centers.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stability_rows).to_csv(OUT / "kmeans_stability.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "input_standardized": str(INPUT),
        "input_raw": str(RAW_INPUT),
        "seed": SEED,
        "k_values": K_VALUES,
        "n_init_main": N_INIT_MAIN,
        "n_stability": N_STABILITY,
        "representation_4d": FOUR_COLS,
        "representation_3d": THREE_COLS,
        "image_generation": False,
        "grading": "provisional quartile rank levels; not final until representation and cluster model are selected",
    }
    (OUT / "evaluation_clustering_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CRITIC 4D weights")
    print(w4_df.to_string(index=False))
    print("Cluster metrics")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
