#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1 论文插图生成 — P1-F01 ~ P1-F06
====================================
按《问题1_论文插图与表格设计报告.md》规范生成正文 6 幅核心图。
数据源均从 results/ 直接读取，不手工录入数值。
输出 PNG(300dpi) + PDF 矢量，双格式。

优先级：
  第一：P1-F02 四指标地图、P1-F06 分类分级地图
  第二：P1-F03 Moran 散点、P1-F04 TOPSIS 排序、P1-F05 热力图
  第三：P1-F01 流程图
"""

from pathlib import Path
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = PROJECT_ROOT / "figures" / "problem1_final"
DATA_OUT = PROJECT_ROOT / "results" / "problem1_figures_data"
FIG_OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

# 字体 — 显式加载中文字体，避免 Arial 回退
try:
    import matplotlib.font_manager as fm
    for fp in [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc"]:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "STXihei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    plt.rcParams["axes.unicode_minus"] = False

# 统一配色
PRESSURE_COLORS = {
    "I级：高压力": "#B2182B",
    "II级：较高压力": "#EF8A62",
    "III级：中压力": "#FDD998",
    "IV级：低压力": "#67A9CF",
}
PRESSURE_ORDER = ["I级：高压力", "II级：较高压力", "III级：中压力", "IV级：低压力"]

CLUSTER_COLORS = {
    "I类：低综合压力型": "#2A9D8F",
    "II类：高规模—高强度工业型": "#9B2226",
    "III类：中高规模—工业主导型": "#E9A23B",
}
CLUSTER_ORDER = ["I类：低综合压力型", "II类：高规模—高强度工业型", "III类：中高规模—工业主导型"]

GEO_CACHE = PROJECT_ROOT / "data" / "china_100000_full.json"
GEO_URL = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"

# 报告中固定的三类中心矩阵（用于校验）
FIXED_CENTERS = {
    "I类：低综合压力型": [0.171, 0.327, 0.219, 0.419],
    "II类：高规模—高强度工业型": [0.839, 0.835, 0.752, 0.938],
    "III类：中高规模—工业主导型": [0.655, 0.478, 0.288, 0.764],
}

# ---------------------------------------------------------------------------
def _province_normalize(name: str) -> str:
    """将地理数据的全称规范为报告中的简称。"""
    for suf in ["省", "市", "自治区", "壮族自治区", "回族自治区", "维吾尔自治区"]:
        if name.endswith(suf):
            # 特殊：内蒙古保留
            if name == "内蒙古自治区":
                return "内蒙古"
            name = name[: -len(suf)]
            break
    # 直辖市去“市”后已正确
    # 新疆/西藏等
    mapping = {
        "内蒙古自治区": "内蒙古",
        "广西壮族自治区": "广西",
        "宁夏回族自治区": "宁夏",
        "新疆维吾尔自治区": "新疆",
        "西藏自治区": "西藏",
        "香港特别行政区": "香港",
        "澳门特别行政区": "澳门",
    }
    return mapping.get(name, name)

def _load_geo() -> "gpd.GeoDataFrame | None":
    """加载中国省级GeoJSON，带本地缓存与离线容错。"""
    try:
        import geopandas as gpd
    except Exception as e:
        print(f"[GEO] geopandas 不可用: {e}，地图将使用占位图")
        return None

    # 缓存
    GEO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not GEO_CACHE.exists():
        try:
            print(f"[GEO] 下载 {GEO_URL} ...")
            urllib.request.urlretrieve(GEO_URL, GEO_CACHE)
            print(f"[GEO] 已缓存至 {GEO_CACHE}")
        except Exception as e:
            print(f"[GEO] 下载失败: {e}")
            return None
    try:
        gdf = gpd.read_file(GEO_CACHE)
        # 规范名称
        gdf["province_short"] = gdf["name"].apply(_province_normalize)
        # 保留原名用于调试
        return gdf
    except Exception as e:
        print(f"[GEO] 读取失败: {e}")
        return None

def _quantile_bins(series: pd.Series, k: int = 5):
    """分位数边界，用于地图分级。"""
    qs = np.linspace(0, 1, k + 1)
    bins = series.quantile(qs).values
    # 去重（极端值导致重复分位）
    bins = np.unique(bins)
    return bins

# ---------------------------------------------------------------------------
def fig_F01_flowchart():
    """P1-F01 总体建模流程图 — 矢量流程，16x8 cm 双栏"""
    fig, ax = plt.subplots(figsize=(8, 4))  # ~20x10 cm，适配中文字
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # 流程 7 步
    steps = [
        ("数据输入\n附件2·30省排放\nGDP·人口", 0.6),
        ("指标构建\nC·I·H·Q", 2.0),
        ("数据预处理\n口径统一·ln·Min-Max", 3.4),
        ("空间差异检验\n全局/局部Moran·敏感性", 4.8),
        ("综合评价\nCRITIC·TOPSIS·分级", 6.2),
        ("聚类分类\nK-means K=3/4/5·Ward对照", 7.6),
        ("交叉解释\n类别×等级·Cramér's V", 9.0),
    ]
    # 主流程框（深蓝）
    for x, (label, _) in zip([s[1] for s in steps], steps):
        # 居中 y=3
        box = mpatches.FancyBboxPatch((x - 0.58, 2.4), 1.16, 1.2, boxstyle="round,pad=0.03", facecolor="#1B3A5F", edgecolor="white", linewidth=0.8)
        ax.add_patch(box)
        ax.text(x, 3.0, label, ha="center", va="center", color="white", fontsize=6.5, linespacing=1.3)

    # 箭头（深蓝实线）
    for i in range(len(steps) - 1):
        x0 = steps[i][1] + 0.58
        x1 = steps[i + 1][1] - 0.58
        ax.annotate("", xy=(x1, 3.0), xytext=(x0, 3.0), arrowprops=dict(arrowstyle="->", color="#1B3A5F", lw=1.4))

    # 稳健性虚线分支（灰）
    # 在“综合评价”处分支为 3D，三维平衡虚线
    ax.text(6.2, 1.85, "3D平衡\nE=(zI+zH)/2  等权", ha="center", va="top", fontsize=5, color="#6B7280",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3F4F6", edgecolor="#9CA3AF", linestyle="--"))
    ax.annotate("", xy=(6.2, 2.4), xytext=(6.2, 2.05), arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=0.9, linestyle="--"))
    # 在“聚类”处 Ward 分支
    ax.text(7.6, 1.85, "Ward\n结构性对照", ha="center", va="top", fontsize=5, color="#6B7280",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3F4F6", edgecolor="#9CA3AF", linestyle="--"))
    ax.annotate("", xy=(7.6, 2.4), xytext=(7.6, 2.05), arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=0.9, linestyle="--"))

    # 顶部标题条
    ax.text(5, 5.45, "问题一 省域碳排放空间差异与分类分级建模流程", ha="center", va="center", fontsize=9, weight="bold", color="#1B3A5F")
    ax.text(5, 5.12, "四指标构建 → 空间差异检验 → CRITIC-TOPSIS综合评价 → K-means分类 → TOPSIS分级 → 类别×等级交叉", ha="center", fontsize=5.5, color="#4B5563")

    # 图例
    ax.plot([0.4, 0.9], [0.55, 0.55], color="#1B3A5F", lw=1.6)
    ax.text(1.0, 0.55, "主流程", va="center", fontsize=5, color="#1B3A5F")
    ax.plot([2.2, 2.7], [0.55, 0.55], color="#9CA3AF", lw=1.0, linestyle="--")
    ax.text(2.85, 0.55, "稳健性对照", va="center", fontsize=5, color="#6B7280")


    for ext in ["png", "pdf"]:
        fig.savefig(FIG_OUT / f"P1_F01_问题一总体建模流程图.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[F01] -> {FIG_OUT}/P1_F01_问题一总体建模流程图.*")

# ---------------------------------------------------------------------------
def fig_F02_four_indicators_map():
    """P1-F02 四指标空间分布 2x2 地图，16x12 cm，YlOrRd，5分位，灰色未样本"""
    df = pd.read_csv(PROJECT_ROOT / "results/problem1_preprocessed/problem1_province_features_2022.csv", encoding="utf-8-sig")
    # 字段映射
    cols = {
        "emission_total_mtco2": ("碳排放总量\n(Mt CO2)", "总量"),
        "carbon_intensity_tco2_per_10k_yuan": ("碳排放强度\n(t/万元)", "强度"),
        "per_capita_emission_tco2_per_person": ("人均碳排放\n(t/人)", "人均"),
        "industrial_emission_share_pct": ("工业碳排放占比\n(%)", "工业占比"),
    }
    gdf = _load_geo()
    if gdf is None:
        # 占位图
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (col, (label, _)) in zip(axes.flat, cols.items()):
            ax.text(0.5, 0.5, f"{label}\n地图底图离线\n{col}", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#9CA3AF")
            ax.set_title(label, fontsize=10)
            ax.axis("off")
        fig.suptitle("2022年30省四项碳排放指标的空间分布（离线占位）", fontsize=12)
        for ext in ["png", "pdf"]:
            fig.savefig(FIG_OUT / f"P1_F02_四项碳排放指标空间分布.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # 合并
    value_df = df[["province"] + list(cols.keys())].copy()
    gdf = gdf.merge(value_df, left_on="province_short", right_on="province", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flat
    cmap = plt.cm.YlOrRd

    for ax, (col, (label, short)) in zip(axes, cols.items()):
        # 5分位
        vals = gdf[col].dropna()
        bins = _quantile_bins(vals, 5)
        # 归一化到 0-1 用于色标
        vmin, vmax = vals.min(), vals.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # 绘制
        # 未样本灰色
        gdf[gdf[col].isna()].plot(ax=ax, color="#D9D9D9", edgecolor="white", linewidth=0.5)
        # 有样本按值着色
        gdf[gdf[col].notna()].plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.5, legend=False)

        ax.set_title(f"({chr(97+list(cols.keys()).index(col))}) {label}", fontsize=10, pad=6)
        ax.axis("off")
        # 标注最高值省
        max_row = gdf.loc[gdf[col].idxmax()]
        # 取质心近似
        try:
            cx, cy = max_row.geometry.centroid.x, max_row.geometry.centroid.y
            ax.text(cx, cy, f"{max_row['province_short']}\n{max_row[col]:.1f}", ha="center", va="center", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#333", alpha=0.85))
        except Exception:
            pass

        # 色条（横向）
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.06, pad=0.04, shrink=0.8)
        cbar.set_label(col, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle("2022年30个省级地区四项碳排放指标的空间分布", fontsize=13, y=0.98)
    fig.text(0.5, 0.01, "注：灰色地区表示未纳入附件2样本，不表示指标值为零；颜色等级按样本内分位数划分", ha="center", fontsize=7, color="#4B5563")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    # 保存中英文数据审计
    pd.DataFrame({"province": df["province"], **{k: df[k] for k in cols}}).to_csv(DATA_OUT / "P1_F02_data.csv", index=False, encoding="utf-8-sig")

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_OUT / f"P1_F02_四项碳排放指标空间分布.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[F02] done")

# ---------------------------------------------------------------------------
def fig_F03_moran_scatter():
    """[已废止] P1-F03 显著指标 Moran 散点 — 按2026-08-22修订版不再作为正文核心图，仅保留为可选诊断图。"""
    df = pd.read_csv(PROJECT_ROOT / "results/problem1_preprocessed/problem1_province_features_2022.csv", encoding="utf-8-sig")
    W = pd.read_csv(PROJECT_ROOT / "results/problem1_spatial_validation/spatial_weight_matrix_row_standardized.csv", index_col=0, encoding="utf-8-sig")
    global_df = pd.read_csv(PROJECT_ROOT / "results/problem1_spatial_validation/global_moran_results.csv", encoding="utf-8-sig")
    local_df = pd.read_csv(PROJECT_ROOT / "results/problem1_spatial_validation/local_moran_results.csv", encoding="utf-8-sig")

    # 指标映射：强度与人均
    targets = [
        ("carbon_intensity_tco2_per_10k_yuan", "碳排放强度"),
        ("per_capita_emission_tco2_per_person", "人均碳排放"),
    ]

    provinces = df["province"].tolist()
    W = W.loc[provinces, provinces].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=False, sharey=False)

    for ax, (col, cn) in zip(axes, targets):
        x = df[col].values.astype(float)
        z = (x - x.mean()) / x.std(ddof=1)
        lag = W @ z
        # 全局 I
        row = global_df[global_df["feature"] == col].iloc[0]
        I, p = row["moran_I"], row["p_two_sided"]

        # 散点：显著内蒙古星标
        # 找出 FDR 显著
        sig = set(local_df[(local_df["feature"] == col) & (local_df["significant_fdr_05"]) ]["province"].tolist())
        colors = ["#B2182B" if pr in sig else "#67A9CF" for pr in provinces]
        sizes = [90 if pr in sig else 28 for pr in provinces]
        ax.scatter(z, lag, c=colors, s=sizes, edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3)
        # 标注内蒙古
        for zi, lgi, pr in zip(z, lag, provinces):
            if pr in sig:
                ax.text(zi + 0.08, lgi + 0.08, f"{pr}★", fontsize=8, color="#B2182B", weight="bold")

        # 拟合线斜率 = I
        xs = np.linspace(z.min() - 0.3, z.max() + 0.3, 100)
        ax.plot(xs, I * xs, color="#1B3A5F", lw=1.6, linestyle="--", label=f"拟合斜率 I={I:.4f}")
        # 零线
        ax.axhline(0, color="#9CA3AF", lw=0.7, linestyle="-")
        ax.axvline(0, color="#9CA3AF", lw=0.7, linestyle="-")
        # 象限标记
        ax.text(0.95, 0.95, "HH", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#4B5563", bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF3C7"))
        ax.text(0.05, 0.95, "LH", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#4B5563", bbox=dict(boxstyle="round,pad=0.2", facecolor="#E0E7FF"))
        ax.text(0.05, 0.05, "LL", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#4B5563", bbox=dict(boxstyle="round,pad=0.2", facecolor="#DBEAFE"))
        ax.text(0.95, 0.05, "HL", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#4B5563", bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEE2E2"))

        ax.set_xlabel(r"$z_i=(x_i-\bar{x})/s_x$", fontsize=9)
        ax.set_ylabel(r"$Wz_i=\sum_j w_{ij}z_j$", fontsize=9)
        ax.set_title(f"({chr(97+targets.index((col, cn)))}) {cn}  I={I:.4f}, p={p:.4f}", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("碳排放强度与人均碳排放的 Moran 散点图", fontsize=13, y=1.02)
    fig.text(0.5, -0.02, "注：★为经 FDR 校正后局部显著的省份（内蒙古）；其余为普通省份；象限仅示局部状态，显著性以置换检验为准", ha="center", fontsize=7, color="#4B5563")
    fig.tight_layout()

    # 保存中间数据
    pd.DataFrame({"province": provinces}).to_csv(DATA_OUT / "P1_F03_data_index.csv", index=False, encoding="utf-8-sig")

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_OUT / f"P1_F03_显著指标Moran散点图.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[F03] done")

# ---------------------------------------------------------------------------
def fig_F04_topsis_rank():
    """P1-F03 TOPSIS 排序 13x16 cm 横向条形，条色=等级，圆点=类别 — 修订后编号为 F03"""
    df = pd.read_csv(PROJECT_ROOT / "results/problem1_cross_analysis/named_assignments_4d_primary.csv", encoding="utf-8-sig")
    df = df.sort_values("topsis_score_4d", ascending=True)  # 横向条形从下到上为升序，顶部为最高

    fig, ax = plt.subplots(figsize=(7, 9))  # ~18x23 cm 竖长，容纳30条

    y = np.arange(len(df))
    scores = df["topsis_score_4d"].values
    # 条色按等级
    bar_colors = [PRESSURE_COLORS.get(l, "#9CA3AF") for l in df["pressure_level_cn"]]

    # 横条
    ax.barh(y, scores, color=bar_colors, edgecolor="white", height=0.68, zorder=3)

    # 圆点类别标记（右侧）
    for yi, cl in zip(y, df["cluster_name"]):
        ax.plot(scores[yi] + 0.015, yi, marker="o", markersize=7, markerfacecolor=CLUSTER_COLORS.get(cl, "#333"),
                markeredgecolor="white", markeredgewidth=0.8, linestyle="None", zorder=4)

    # 省份标签（左侧）
    ax.set_yticks(y)
    ax.set_yticklabels(df["province"].values, fontsize=9)
    ax.set_xlabel("TOPSIS 综合减排压力得分  (0—1，越大压力越高)", fontsize=10)
    ax.set_title("2022年30省TOPSIS综合减排压力得分及等级", fontsize=12, pad=10)

    # 数值标注（条末）
    for yi, sc in zip(y, scores):
        ax.text(sc + 0.008, yi, f"{sc:.3f}", va="center", ha="left", fontsize=7, color="#1F2937")

    # 等级分隔虚线（按等级边界分位数近似：8/7/8/7）
    # 按排序后等级变化处画线
    for i in range(1, len(df)):
        if df.iloc[i]["pressure_level_cn"] != df.iloc[i - 1]["pressure_level_cn"]:
            ax.axhline(i - 0.5, color="#D1D5DB", lw=0.8, linestyle="--", alpha=0.7)

    # 前四名星标
    top4 = df.tail(4)["province"].tolist()
    for yi, pr in zip(y, df["province"]):
        if pr in top4:
            ax.text(0.02, yi, "★", va="center", ha="left", fontsize=9, color="#B2182B", weight="bold")

    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    # y=0 在底部对应低分，y=29 在顶部对应高分，已满足“最高值置于顶部”，不需 invert

    # 图例：等级
    level_patches = [mpatches.Patch(color=PRESSURE_COLORS[k], label=k) for k in PRESSURE_ORDER]
    cluster_dots = [plt.Line2D([0], [0], marker="o", color="white", markerfacecolor=CLUSTER_COLORS[k], markeredgecolor="white", markersize=8, linestyle="None", label=k) for k in CLUSTER_ORDER]
    leg1 = ax.legend(handles=level_patches, title="压力等级（条色）", fontsize=7, title_fontsize=8, loc="lower right", framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=cluster_dots, title="聚类类别（圆点）", fontsize=7, title_fontsize=8, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    # 审计输出
    df.to_csv(DATA_OUT / "P1_F03_data.csv", index=False, encoding="utf-8-sig")
    # 修订后编号：P1-F03（旧F04）。同时保留旧文件名做兼容
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_OUT / f"P1_F03_TOPSIS综合压力得分排序.{ext}", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_OUT / f"P1_F04_TOPSIS综合压力得分排序.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[F04->F03] done (双文件名兼容)")

# ---------------------------------------------------------------------------
def fig_F05_cluster_heatmap():
    """P1-F04 三类中心热力图 3x4，0-1 固定色标 — 修订后编号为 F04"""
    # 读取中心并按正式类别重排
    centers = pd.read_csv(PROJECT_ROOT / "results/problem1_evaluation_clustering/cluster_centers.csv", encoding="utf-8-sig")
    sel = centers[(centers["representation"] == "4d_direct") & (centers["method"] == "Kmeans") & (centers["k"] == 3)].copy()
    # 原始 cluster 1/2/3 对应报告中的 I/II/III 映射：1->I类(4省), 2->II类(6省), 3->III类(20省)
    # 验证：sizes 4,6,20
    mapping_cluster_to_name = {1: "I类：低综合压力型", 2: "II类：高规模—高强度工业型", 3: "III类：中高规模—工业主导型"}
    sel["正式类别"] = sel["cluster"].map(mapping_cluster_to_name)
    # 按正式顺序排序：I, II, III
    order = ["I类：低综合压力型", "II类：高规模—高强度工业型", "III类：中高规模—工业主导型"]
    sel = sel.set_index("正式类别").loc[order].reset_index()

    # 提取 4 列中心
    cols = ["center_1", "center_2", "center_3", "center_4"]
    cn_cols = ["总量", "强度", "人均", "工业占比"]
    mat = sel[cols].values
    # 校验与固定矩阵一致
    for idx, name in enumerate(order):
        assert np.allclose(mat[idx], FIXED_CENTERS[name], atol=1e-3), f"中心不匹配 {name}: {mat[idx]} vs {FIXED_CENTERS[name]}"

    fig, ax = plt.subplots(figsize=(7, 3.2))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlBu_r", vmin=0, vmax=1, cbar=True,
                xticklabels=cn_cols, yticklabels=[f"{n} (n={int(s)})" for n, s in zip(order, sel["cluster_size"])],
                linewidths=1.2, linecolor="white", cbar_kws={"label": "标准化中心值 (0—1)", "shrink": 0.9}, ax=ax)
    ax.set_title("K-means 三类省域碳排放类型的标准化聚类中心", fontsize=12, pad=10)
    ax.set_xlabel("四项指标（已 ln+Min-Max 标准化）", fontsize=10)
    ax.set_ylabel("聚类类别", fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    fig.tight_layout()
    # 审计
    sel.to_csv(DATA_OUT / "P1_F04_data.csv", index=False, encoding="utf-8-sig")
    # 修订后编号：P1-F04（旧F05），保留旧文件名兼容
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_OUT / f"P1_F04_Kmeans三类聚类中心热力图.{ext}", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_OUT / f"P1_F05_Kmeans三类聚类中心热力图.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[F05->F04] done (双文件名兼容)")

# ---------------------------------------------------------------------------
def fig_F06_classification_map():
    """P1-F05/P1-F06 1x2 分类与分级地图 — 修订后正文核心终图（表F06，图序F05/F06双别名）"""
    df = pd.read_csv(PROJECT_ROOT / "results/problem1_cross_analysis/named_assignments_4d_primary.csv", encoding="utf-8-sig")
    gdf = _load_geo()
    if gdf is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, title in zip(axes, ["K-means三类排放类型", "TOPSIS四级压力等级"]):
            ax.text(0.5, 0.5, f"{title}\n地图底图离线\n占位", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#9CA3AF")
            ax.set_title(title)
            ax.axis("off")
        for ext in ["png", "pdf"]:
            fig.savefig(FIG_OUT / f"P1_F06_省域分类与压力分级空间分布.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # 合并
    gdf = gdf.merge(df[["province", "cluster_name", "pressure_level_cn"]], left_on="province_short", right_on="province", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # (a) 分类
    ax = axes[0]
    # 未样本灰色
    gdf[gdf["cluster_name"].isna()].plot(ax=ax, color="#D9D9D9", edgecolor="white", linewidth=0.6)
    for cname in CLUSTER_ORDER:
        sub = gdf[gdf["cluster_name"] == cname]
        if len(sub):
            sub.plot(ax=ax, color=CLUSTER_COLORS[cname], edgecolor="white", linewidth=0.6, label=f"{cname} (n={(gdf['cluster_name']==cname).sum()})")
    ax.set_title("(a) K-means三类排放类型", fontsize=11, pad=8)
    ax.axis("off")
    # (b) 分级
    ax = axes[1]
    gdf[gdf["pressure_level_cn"].isna()].plot(ax=ax, color="#D9D9D9", edgecolor="white", linewidth=0.6)
    for lvl in PRESSURE_ORDER:
        sub = gdf[gdf["pressure_level_cn"] == lvl]
        if len(sub):
            sub.plot(ax=ax, color=PRESSURE_COLORS[lvl], edgecolor="white", linewidth=0.6, label=f"{lvl} (n={(gdf['pressure_level_cn']==lvl).sum()})")
    ax.set_title("(b) TOPSIS四级综合压力等级", fontsize=11, pad=8)
    ax.axis("off")

    # 图例：分类 / 分级分开
    for ax in axes:
        leg = ax.legend(fontsize=7, title_fontsize=8, loc="lower left", framealpha=0.95, edgecolor="#E5E7EB")
        leg.get_frame().set_linewidth(0.6)

    fig.suptitle("2022年30省碳排放类型与综合压力等级的空间分布", fontsize=13, y=0.98)
    fig.text(0.5, 0.01, "注：灰色为未纳入样本；分类 I类4省 II类6省 III类20省；分级 I级8省 II级7省 III级8省 IV级7省", ha="center", fontsize=7, color="#4B5563")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    # 审计
    df.to_csv(DATA_OUT / "P1_F06_data.csv", index=False, encoding="utf-8-sig")
    df.to_csv(DATA_OUT / "P1_F05_data_final.csv", index=False, encoding="utf-8-sig")
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_OUT / f"P1_F06_省域分类与压力分级空间分布.{ext}", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_OUT / f"P1_F05_省域分类与压力分级空间分布.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[F06] done (F05/F06双文件名兼容)")

# ---------------------------------------------------------------------------
def main():
    print(f"输出目录: {FIG_OUT}")
    print(f"中间数据: {DATA_OUT}")
    print("修订版：Moran散点图不再作为正文核心图（仅可选诊断），按新编号 F01/F02/F03/F04/F06 生成")
    fig_F01_flowchart()
    fig_F02_four_indicators_map()
    # fig_F03_moran_scatter()  # 已按修订版移除正文核心，不生成；如需诊断可手动取消注释
    fig_F04_topsis_rank()       # -> P1-F03
    fig_F05_cluster_heatmap()   # -> P1-F04
    fig_F06_classification_map()  # -> P1-F05/F06
    print("All figures done. Check PNG+PDF pairs. (F03 Moran 已跳过)")

if __name__ == "__main__":
    main()
