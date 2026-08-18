# -*- coding: utf-8 -*-
"""
问题3 · Step 1：数据读取与口径统一
==================================
功能：
    1. 读取附件3（10 区域 × 24 小时 × 工作日/周末的充电负荷，单位 kWh，因 Δt=1h 数值上等价于 kW）；
    2. 读取附件4（10 区域 × 24 小时的电网最大允许负荷 G_{i,t}，单位 kW）；
    3. 固定题给的分时电价时段集合（谷 / 平 / 峰 / 非调度），不参与优化；
    4. 清洗：去除末尾说明行、统一列顺序、校验数值完整性与非负性。

输出：
    results/p3_data/L_pre_wd.csv   工作日调度前负荷 (10×24, kW)
    results/p3_data/L_pre_we.csv   周末   调度前负荷 (10×24, kW)
    results/p3_data/G_limit.csv    电网最大允许负荷 (10×24, kW)
    results/p3_data/time_sets.json 时段集合与元信息

运行依赖：numpy, pandas（见 code/requirements.txt）
用法：python problem3_data_preprocess.py   （在仓库根目录或任意位置均可，脚本自动定位项目根目录）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 路径与常量
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # 项目根目录（含“附件/”）
ATTACH_DIR = ROOT / "附件"
F_LOAD = ATTACH_DIR / "附件3 市主城区 10 区域分时段充电负荷.xlsx"   # 调度前充电负荷
F_GRID = ATTACH_DIR / "附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx"  # 电网上限
OUT_DIR = ROOT / "p3_data"

DT_HOURS = 24                                    # 一天的小时数（时段长度 Δt=1h）
REGIONS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # 区域编号

HOUR_COLS = [f"{h:02d}-{(h + 1) % 24:02d}" for h in range(DT_HOURS)]
assert HOUR_COLS[-1] == "23-00", "小时列命名应为 00-01 ... 23-00"

# -----------------------------------------------------------------------------
# 题给分时电价时段（外生政策，不得作为决策变量）。按附件5原文：
#   低谷 00:00-07:00；平段 07:00-11:00 / 14:00-16:00；高峰 11:00-14:00 / 16:00-23:00
# 注意：任务单 §2.3 文字写“高峰(11个小时)”，但按其列出的时段与附件5，
#       高峰实为 {11,12,13} ∪ {16..22} = 10 小时；23:00-00:00 为非调度时段。
# -----------------------------------------------------------------------------
VALLEY_HOURS: list[int] = [0, 1, 2, 3, 4, 5, 6]              # 谷段 7h → z 的决策位
PEAK_HOURS: list[int] = [11, 12, 13, 16, 17, 18, 19, 20, 21, 22]   # 峰段 10h，等比例削减20%
MIDDLE_HOURS: list[int] = [7, 8, 9, 10, 14, 15]              # 平段 6h，保持不变
UNCHANGED_OTHER: list[int] = [23]                            # 非峰谷的剩余时段（23-00），保持不变

TRANSFER_RATIO_PEAK = 0.20   # 题设：高峰时段 20% 负荷可转移至低谷


def _to_hour_index(col: str) -> int:
    """把 'HH-MM' 形式的列名转换为起始小时索引（0-23）。"""
    return int(col[:2])


def load_dataframe(raw: pd.DataFrame, region_col: str, name: str) -> np.ndarray:
    """把工作表清洗为 shape=(10,24) 的 float64 矩阵，行序对应 REGIONS。

    参数
    ----
    raw : 附件原始 DataFrame（第一列为区域编号/名称）
    region_col : 区域列名
    name : 数据集名称（用于断言信息）
    """
    hour_cols = [c for c in HOUR_COLS if c in raw.columns]
    assert len(hour_cols) == DT_HOURS, f"{name}: 小时列不完整，缺少 {set(HOUR_COLS)-set(raw.columns)}"

    sub = raw[raw[region_col].notna()].copy()
    sub = sub[sub[region_col].isin(REGIONS)]
    # 若区域列为字符串（如附件4的“1”..“10”），统一转 int
    if sub[region_col].dtype == object:
        sub[region_col] = pd.to_numeric(sub[region_col])

    assert sorted(sub[region_col].astype(int).tolist()) == REGIONS, \
        f"{name}: 区域编号应恰好为 {REGIONS}，实际 {sorted(sub[region_col].unique())}"
    mat = sub.set_index(region_col).reindex(REGIONS)[hour_cols].to_numpy(dtype=float)

    assert np.isfinite(mat).all(), f"{name}: 存在 NaN/Inf"
    assert (mat >= 0).all(), f"{name}: 存在负值负荷"
    return mat


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 附件3：工作日 / 周末负荷 ----------------
    wd_raw = pd.read_excel(F_LOAD, sheet_name=0)
    we_raw = pd.read_excel(F_LOAD, sheet_name=1)
    L_wd = load_dataframe(wd_raw, "区域", "附件3-工作日")   # (10,24), kW（数值=kWh，Δt=1h）
    L_we = load_dataframe(we_raw, "区域", "附件3-周末")

    # ---------------- 附件4：电网最大允许负荷（不分日期类型） ----------------
    grid_raw = pd.read_excel(F_GRID, sheet_name=0)
    G = load_dataframe(grid_raw, "区域名称", "附件4")       # (10,24), kW

    # 保存为规范的宽表 CSV
    df_hdr = pd.DataFrame({"区域": REGIONS})
    for h in range(DT_HOURS):
        df_hdr[HOUR_COLS[h]] = [round(float(x), 6) for x in L_wd[:, h]]
    df_hdr.to_csv(OUT_DIR / "L_pre_wd.csv", index=False, encoding="utf-8-sig")

    df_we = pd.DataFrame({"区域": REGIONS})
    for h in range(DT_HOURS):
        df_we[HOUR_COLS[h]] = [round(float(x), 6) for x in L_we[:, h]]
    df_we.to_csv(OUT_DIR / "L_pre_we.csv", index=False, encoding="utf-8-sig")

    df_g = pd.DataFrame({"区域": REGIONS})
    for h in range(DT_HOURS):
        df_g[HOUR_COLS[h]] = [round(float(x), 6) for x in G[:, h]]
    df_g.to_csv(OUT_DIR / "G_limit.csv", index=False, encoding="utf-8-sig")

    # ---------------- 时段集合元信息（后续步骤共用，避免索引错位） ----------------
    meta = {
        "hour_columns": HOUR_COLS,
        "valley_hours": VALLEY_HOURS,          # t∈V：接收 z_{i,t}
        "peak_hours": PEAK_HOURS,              # t∈H：等比例削减 20%
        "middle_hours": MIDDLE_HOURS,          # t∈M：保持原负荷
        "unchanged_other_hours": UNCHANGED_OTHER,   # 23-00：非峰谷，保持原负荷
        "transfer_ratio_peak": TRANSFER_RATIO_PEAK,
        "covered_count": len(VALLEY_HOURS) + len(PEAK_HOURS) + len(MIDDLE_HOURS) + len(UNCHANGED_OTHER),
    }
    assert meta["covered_count"] == DT_HOURS and \
        len(set([*meta["valley_hours"], *meta["peak_hours"], *meta["middle_hours"], *meta["unchanged_other_hours"]])) == DT_HOURS, \
        "时段集合必须恰好覆盖 24h 且互不重叠"
    with open(OUT_DIR / "time_sets.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------------- 控制台摘要 ----------------
    print("[OK] 数据预处理完成：")
    print(f"     L_pre_wd : {L_wd.shape} kW，全天总量(区域均值/日): {L_wd.sum(axis=1).mean():.0f} kWh")
    print(f"     L_pre_we : {L_we.shape} kW，全天总量(区域均值/日): {L_we.sum(axis=1).mean():.0f} kWh")
    print(f"     G_limit  : {G.shape} kW")
    print(f"     时段集合: 谷{meta['valley_hours']} | 峰{PEAK_HOURS[:4]}...{PEAK_HOURS[-3:]}(共{len(PEAK_HOURS)}h) "
          f"| 平{MIDDLE_HOURS} | 非调度{UNCHANGED_OTHER}")


if __name__ == "__main__":
    main()
