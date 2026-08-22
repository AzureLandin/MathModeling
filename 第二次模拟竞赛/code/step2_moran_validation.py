"""Problem 1: Moran spatial validation without generating figures.

This script uses the preprocessed four-feature table and a transparent, manually
encoded province adjacency matrix. The adjacency file is not supplied with the
contest attachments, so the matrix is treated as a provisional analysis input
and is exported for audit and later replacement by an official GIS-derived
matrix if available.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "problem1_preprocessed" / "problem1_province_features_2022.csv"
OUT = ROOT / "results" / "problem1_spatial_validation"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260822
N_PERM = 9999

PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州",
    "云南", "陕西", "甘肃", "青海", "宁夏", "新疆",
]

# Province-level land-border adjacency for the 30-province sample.
# Tibet and other regions outside attachment 2 are intentionally omitted.
# Hainan is connected to Guangdong as an island-neighbor fallback so that its
# row is not isolated in the spatial weights matrix; this assumption is reported.
ADJ = {
    "北京": {"天津", "河北"},
    "天津": {"北京", "河北"},
    "河北": {"北京", "天津", "山西", "内蒙古", "辽宁", "河南", "山东"},
    "山西": {"河北", "内蒙古", "陕西", "河南"},
    "内蒙古": {"黑龙江", "吉林", "辽宁", "河北", "山西", "陕西", "宁夏", "甘肃"},
    "辽宁": {"河北", "内蒙古", "吉林"},
    "吉林": {"辽宁", "黑龙江", "内蒙古"},
    "黑龙江": {"吉林", "内蒙古"},
    "上海": {"江苏", "浙江"},
    "江苏": {"山东", "安徽", "浙江", "上海"},
    "浙江": {"上海", "江苏", "安徽", "江西", "福建"},
    "安徽": {"江苏", "浙江", "江西", "湖北", "河南"},
    "福建": {"浙江", "江西", "广东"},
    "江西": {"浙江", "福建", "广东", "湖南", "湖北", "安徽"},
    "山东": {"河北", "江苏", "河南"},
    "河南": {"河北", "山西", "山东", "安徽", "湖北", "陕西"},
    "湖北": {"河南", "安徽", "江西", "湖南", "重庆", "陕西"},
    "湖南": {"湖北", "江西", "广东", "广西", "贵州", "重庆"},
    "广东": {"福建", "江西", "湖南", "广西", "海南"},
    "广西": {"广东", "湖南", "贵州", "云南"},
    "海南": {"广东"},
    "重庆": {"湖北", "湖南", "四川", "贵州", "陕西"},
    "四川": {"重庆", "贵州", "云南", "陕西", "甘肃", "青海"},
    "贵州": {"四川", "重庆", "湖南", "广西", "云南"},
    "云南": {"四川", "贵州", "广西"},
    "陕西": {"内蒙古", "山西", "河南", "湖北", "重庆", "四川", "甘肃", "宁夏"},
    "甘肃": {"内蒙古", "宁夏", "陕西", "四川", "青海", "新疆"},
    "青海": {"甘肃", "四川", "新疆"},
    "宁夏": {"内蒙古", "陕西", "甘肃"},
    "新疆": {"甘肃", "青海"},
}

FEATURES = {
    "emission_total_mtco2": "碳排放总量",
    "carbon_intensity_tco2_per_10k_yuan": "碳排放强度",
    "per_capita_emission_tco2_per_person": "人均碳排放",
    "industrial_emission_share": "工业碳排放占比",
}


def build_weight_matrix(provinces: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    unknown = set(provinces) - set(PROVINCES)
    if unknown:
        raise ValueError(f"输入表存在未定义省份: {sorted(unknown)}")
    if set(provinces) != set(PROVINCES):
        missing = set(PROVINCES) - set(provinces)
        raise ValueError(f"输入表缺少省份: {sorted(missing)}")
    idx = {p: i for i, p in enumerate(provinces)}
    W = np.zeros((len(provinces), len(provinces)), dtype=float)
    for p, neighbors in ADJ.items():
        for q in neighbors:
            if p not in idx or q not in idx:
                continue
            W[idx[p], idx[q]] = 1.0
    if not np.allclose(W, W.T):
        raise ValueError("邻接矩阵不是对称矩阵，请检查邻接表")
    degree = W.sum(axis=1)
    if np.any(degree <= 0):
        bad = [provinces[i] for i, d in enumerate(degree) if d <= 0]
        raise ValueError(f"存在无邻接省份: {bad}")
    W = W / degree[:, None]
    audit = pd.DataFrame({
        "province": provinces,
        "neighbor_count": degree.astype(int),
        "neighbors": ["、".join(sorted(ADJ[p])) for p in provinces],
    })
    return W, audit


def moran_stat(x: np.ndarray, W: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    denom = float(z @ z)
    if denom <= 0:
        return float("nan")
    n = len(x)
    s0 = float(W.sum())
    return float((n / s0) * (z @ W @ z) / denom)


def global_permutation(x: np.ndarray, W: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    observed = moran_stat(x, W)
    values = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        values[b] = moran_stat(rng.permutation(x), W)
    expected = -1.0 / (len(x) - 1)
    centered = np.abs(values - expected)
    p_two = (1.0 + np.sum(centered >= abs(observed - expected))) / (n_perm + 1.0)
    p_high = (1.0 + np.sum(values >= observed)) / (n_perm + 1.0)
    p_low = (1.0 + np.sum(values <= observed)) / (n_perm + 1.0)
    return {
        "moran_I": observed,
        "expected_I": expected,
        "perm_mean": float(values.mean()),
        "perm_std": float(values.std(ddof=1)),
        "p_two_sided": float(p_two),
        "p_upper": float(p_high),
        "p_lower": float(p_low),
        "n_permutations": n_perm,
        "significant_05": bool(p_two < 0.05),
        "direction": "正空间自相关" if observed > expected else "负空间自相关/弱集聚",
    }


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def local_moran(x: np.ndarray, W: np.ndarray, provinces: list[str], n_perm: int, rng: np.random.Generator) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    z = x - x.mean()
    m2 = float(np.mean(z * z))
    if m2 <= 0:
        raise ValueError("局部Moran输入变量无方差")
    z_std = z / math.sqrt(m2)
    lag = W @ z_std
    observed = z_std * lag
    out = []
    for i, p in enumerate(provinces):
        perm_values = np.empty(n_perm, dtype=float)
        for b in range(n_perm):
            zp = rng.permutation(z_std)
            perm_values[b] = zp[i] * (W[i] @ zp)
        # Two-sided permutation p-value around the permutation mean.
        pval = (1.0 + np.sum(np.abs(perm_values - perm_values.mean()) >= abs(observed[i] - perm_values.mean()))) / (n_perm + 1.0)
        out.append({
            "province": p,
            "local_moran_I": float(observed[i]),
            "local_p_two_sided": float(pval),
            "z_value": float(z_std[i]),
            "spatial_lag_z": float(lag[i]),
            "quadrant": (
                "High-High" if z_std[i] > 0 and lag[i] > 0 else
                "Low-Low" if z_std[i] < 0 and lag[i] < 0 else
                "High-Low" if z_std[i] > 0 and lag[i] < 0 else
                "Low-High"
            ),
            "significant_05": bool(pval < 0.05),
        })
    return pd.DataFrame(out)


def main() -> None:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    if len(df) != 30 or df["province"].nunique() != 30:
        raise ValueError("预处理表不是30个唯一省份")
    provinces = list(df["province"])
    W, audit = build_weight_matrix(provinces)
    audit.to_csv(OUT / "spatial_adjacency_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(W, index=provinces, columns=provinces).to_csv(OUT / "spatial_weight_matrix_row_standardized.csv", encoding="utf-8-sig")

    rng = np.random.default_rng(SEED)
    global_rows = []
    local_frames = []
    for col, label in FEATURES.items():
        x = df[col].to_numpy(dtype=float)
        result = global_permutation(x, W, N_PERM, rng)
        result.update({"feature": col, "feature_cn": label, "n": len(x), "weight_type": "manual province adjacency + Hainan-Guangdong island fallback"})
        global_rows.append(result)
        local = local_moran(x, W, provinces, N_PERM, rng)
        local.insert(0, "feature", col)
        local.insert(1, "feature_cn", label)
        local_frames.append(local)
    global_df = pd.DataFrame(global_rows)[[
        "feature", "feature_cn", "n", "weight_type", "moran_I", "expected_I", "perm_mean", "perm_std",
        "p_two_sided", "p_upper", "p_lower", "n_permutations", "significant_05", "direction"
    ]]
    global_df.to_csv(OUT / "global_moran_results.csv", index=False, encoding="utf-8-sig")
    local_df = pd.concat(local_frames, ignore_index=True)
    local_df["p_fdr_bh"] = bh_fdr(local_df["local_p_two_sided"].to_numpy())
    local_df["significant_fdr_05"] = local_df["p_fdr_bh"] < 0.05
    local_df.to_csv(OUT / "local_moran_results.csv", index=False, encoding="utf-8-sig")

    # Weight-matrix sensitivity: remove the provisional Hainan-Guangdong link
    # and compare with the primary row-standardized matrix and binary matrix.
    W_no_hainan = W.copy()
    h = provinces.index("海南")
    W_no_hainan[h, :] = 0.0
    W_no_hainan[:, h] = 0.0
    row_sum = W_no_hainan.sum(axis=1)
    W_no_hainan = np.divide(W_no_hainan, row_sum[:, None], out=np.zeros_like(W_no_hainan), where=row_sum[:, None] > 0)
    W_binary = (W > 0).astype(float)
    sensitivity_rows = []
    for matrix_name, matrix in [("primary_row_standardized", W), ("no_hainan_row_standardized", W_no_hainan), ("primary_binary", W_binary)]:
        local_rng = np.random.default_rng(SEED)
        for col, label in FEATURES.items():
            x = df[col].to_numpy(dtype=float)
            result = global_permutation(x, matrix, N_PERM, local_rng)
            sensitivity_rows.append({
                "matrix": matrix_name,
                "feature": col,
                "feature_cn": label,
                "moran_I": result["moran_I"],
                "p_two_sided": result["p_two_sided"],
                "significant_05": result["significant_05"],
            })
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "global_moran_weight_sensitivity.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "input": str(INPUT),
        "n_provinces": len(provinces),
        "seed": SEED,
        "n_permutations": N_PERM,
        "matrix": "symmetric binary province adjacency, row-standardized; Hainan connected to Guangdong as island-neighbor fallback",
        "features": FEATURES,
        "image_generation": False,
    }
    (OUT / "spatial_validation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(global_df.to_string(index=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()

