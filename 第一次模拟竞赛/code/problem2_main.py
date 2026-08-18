# -*- coding: utf-8 -*-
"""
problem2_main.py
问题二验证主程序：数据构造、可行性预检、NSGA-II、熵权 TOPSIS、敏感性与出图。

用法：
  python problem2_main.py           # 基准规模 300×500，10 个随机种子
  python problem2_main.py --quick   # 快速核验 80×60，3 个种子
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problem2_nsga2 import (
    default_ref_point,
    front_to_dataframe,
    hv_of_front,
    igd_of_front,
    last50_hv_rel_change,
    merge_fronts,
    milp_min_cost,
    neighborhood_check,
    solve_nsga2,
)
from problem2_topsis import equal_weight_topsis, pick_representatives, topsis_rank
from problem2_utils import (
    ATTACH_DIR,
    C_F,
    C_S,
    FIGURES_DIR,
    N_REGION,
    PALETTE,
    RESULTS_DIR,
    audit_plan,
    coverage,
    derived_parameter_tables,
    load_problem2_data,
    make_diag_m0,
    make_diag_m1,
    make_strict_m0,
    make_strict_m1,
    min_delta_for_region,
    objectives,
    plan_detail,
    precheck_region,
    precheck_scenario,
    save_xlsx,
)

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["font.sans-serif"] = [
    "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

SEEDS_FULL = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="问题二验证主程序")
    p.add_argument("--quick", action="store_true", help="使用较小种群与代数做连通性核验")
    return p.parse_args()


def xy_from_row(front_df: pd.DataFrame, idx: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([int(front_df.loc[idx, f"x{i+1}"]) for i in range(N_REGION)], dtype=int)
    y = np.array([int(front_df.loc[idx, f"y{i+1}"]) for i in range(N_REGION)], dtype=int)
    return x, y


def summarize_plan(data, scen, x, y, extra: dict | None = None) -> dict:
    f1, f2, f3, S, rho = objectives(data, x, y, scen.beta, scen.alpha)
    aud = audit_plan(data, x, y, scen)
    out = {
        "新增快充合计": int(np.sum(x)),
        "新增慢充合计": int(np.sum(y)),
        "总投资成本": float(f1),
        "需求加权覆盖率": float(f2),
        "最低区域覆盖率": float(S.min()),
        "最大区域电网压力": float(rho.max()),
        "区域压力方差": float(np.mean((rho - rho.mean()) ** 2)),
        "F3": float(f3),
        "快充是否全零": bool(np.all(x == 0)),
        "最大约束违反量": aud["max_any_violation"],
        "复核通过": aud["pass"],
    }
    if extra:
        out.update(extra)
    return out


def run_multi_seed(data, scen, seeds, pop, n_gen, tag: str):
    ref = default_ref_point(data, scen)
    results = []
    print(f"\n=== {tag}: pop={pop}, gen={n_gen}, seeds={seeds} ===")
    print(f"情景说明: {scen.note}")
    for seed in seeds:
        print(f"  运行 seed={seed} ...", flush=True)
        res = solve_nsga2(data, scen, pop_size=pop, n_gen=n_gen, seed=seed, ref_point=ref)
        print(
            f"    可行Pareto={len(res.X)}, 用时={res.runtime_s:.1f}s, 备注={res.note or 'ok'}",
            flush=True,
        )
        results.append(res)
    return results, ref


def m0_degeneration_report(front_df: pd.DataFrame) -> dict:
    if front_df.empty:
        return {"n_pareto": 0, "all_slow": None, "fast_always_zero": None, "n_unique_F": 0}
    all_slow = bool(front_df["快充是否全零"].all())
    n_unique_obj = int(front_df[["F1_投资成本", "F2_需求加权覆盖率", "F3_电网目标"]].round(8).drop_duplicates().shape[0])
    return {
        "n_pareto": int(len(front_df)),
        "all_slow": all_slow,
        "fast_always_zero": all_slow,
        "n_unique_F": n_unique_obj,
        "fast_total_max": int(front_df["新增快充合计"].max()),
        "fast_total_min": int(front_df["新增快充合计"].min()),
    }


def build_representative_table(data, scen, front_df, picks, topsis_df) -> pd.DataFrame:
    labels = {
        "topsis": "TOPSIS推荐",
        "min_cost": "最低成本",
        "max_cover": "最高覆盖率",
        "min_grid": "最低电网压力",
    }
    w = topsis_df.attrs.get("weights", np.array([np.nan, np.nan, np.nan]))
    rows = []
    for key, lab in labels.items():
        idx = picks[key]
        x, y = xy_from_row(front_df, idx)
        sid = int(front_df.loc[idx, "方案编号"])
        c_row = topsis_df.loc[topsis_df["方案编号"] == sid]
        cval = float(c_row["贴近度_C"].iloc[0]) if len(c_row) else np.nan
        rec = summarize_plan(data, scen, x, y, {
            "方案类型": lab,
            "Pareto方案编号": sid,
            "熵权_w1": float(w[0]),
            "熵权_w2": float(w[1]),
            "熵权_w3": float(w[2]),
            "TOPSIS贴近度": cval,
        })
        rec["x"] = x.tolist()
        rec["y"] = y.tolist()
        rows.append(rec)
    return pd.DataFrame(rows)


def sensitivity_cell(data, beta, delta, tau, alpha, use_mix, pop, n_gen, seed, lift_delta: bool = False):
    if use_mix:
        # δ 扫描保持原始取值以暴露不可行；β/τ/α 扫描与主模型一致，必要时上调个别区域 δ
        scen = make_diag_m1(data, beta=beta, delta=delta, tau=tau, alpha=alpha, lift_infeasible_delta=lift_delta)
    else:
        scen = make_diag_m0(data, beta=beta, tau=tau, alpha=alpha)
    chk = precheck_scenario(data, scen)
    feasible_regions = bool(chk["feasible"].all())
    n_infeas = int((~chk["feasible"]).sum())
    infeas_ids = chk.loc[~chk["feasible"], "region"].tolist()
    if not feasible_regions:
        return {
            "feasible": False,
            "n_infeasible_regions": n_infeas,
            "infeasible_regions": infeas_ids,
            "n_pareto": 0,
            "cost": np.nan,
            "fast": np.nan,
            "slow": np.nan,
            "piles": np.nan,
            "F2": np.nan,
            "max_rho": np.nan,
            "note": scen.note,
        }
    res = solve_nsga2(data, scen, pop_size=pop, n_gen=n_gen, seed=seed)
    if not res.feasible:
        return {
            "feasible": False,
            "n_infeasible_regions": 0,
            "infeasible_regions": [],
            "n_pareto": 0,
            "cost": np.nan,
            "fast": np.nan,
            "slow": np.nan,
            "piles": np.nan,
            "F2": np.nan,
            "max_rho": np.nan,
            "note": res.note,
        }
    front = front_to_dataframe(data, scen, res.X, res.F)
    td = topsis_rank(front["F1_投资成本"].to_numpy(), front["F2_需求加权覆盖率"].to_numpy(), front["F3_电网目标"].to_numpy())
    picks = pick_representatives(front, td)
    x, y = xy_from_row(front, picks["topsis"])
    f1, f2, f3, S, rho = objectives(data, x, y, scen.beta, scen.alpha)
    return {
        "feasible": True,
        "n_infeasible_regions": 0,
        "infeasible_regions": [],
        "n_pareto": int(len(front)),
        "cost": float(f1),
        "fast": int(x.sum()),
        "slow": int(y.sum()),
        "piles": int(x.sum() + y.sum()),
        "F2": float(f2),
        "max_rho": float(rho.max()),
        "note": scen.note,
        "entropy_extreme": bool(np.max(td.attrs["weights"]) > 0.9),
    }


def plot_all(data, scen_m1, front_m0, front_m1, hist_best, rec_x, rec_y, sens_frames, picks, topsis_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 三维 Pareto
    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    if not front_m0.empty:
        ax.scatter(
            front_m0["F1_投资成本"], front_m0["F2_需求加权覆盖率"], front_m0["F3_电网目标"],
            c=PALETTE["m0"], s=22, alpha=0.65, label="M0 诊断前沿",
        )
    if not front_m1.empty:
        ax.scatter(
            front_m1["F1_投资成本"], front_m1["F2_需求加权覆盖率"], front_m1["F3_电网目标"],
            c=PALETTE["m1"], s=28, alpha=0.8, label="M1 诊断前沿",
        )
        labels = {"topsis": "TOPSIS", "min_cost": "最低成本", "max_cover": "最高覆盖", "min_grid": "最低压力"}
        markers = {"topsis": "*", "min_cost": "D", "max_cover": "s", "min_grid": "^"}
        for key, lab in labels.items():
            r = front_m1.loc[picks[key]]
            ax.scatter(
                [r["F1_投资成本"]], [r["F2_需求加权覆盖率"]], [r["F3_电网目标"]],
                s=140 if key == "topsis" else 70, marker=markers[key],
                c=PALETTE["cost"] if key != "topsis" else PALETTE["fast"],
                edgecolors="k", linewidths=0.6, label=lab, zorder=5,
            )
    ax.set_xlabel("F1 投资成本 / 万元")
    ax.set_ylabel("F2 需求加权覆盖率")
    ax.set_zlabel("F3 电网压力目标")
    ax.set_title("问题二 Pareto 三维前沿（诊断可行模型）")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIGURES_DIR / "问题二_Pareto三维前沿.png")
    plt.close(fig)

    # 2. 两两投影
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    pairs = [
        ("F1_投资成本", "F2_需求加权覆盖率", "投资成本 / 万元", "需求加权覆盖率"),
        ("F1_投资成本", "F3_电网目标", "投资成本 / 万元", "电网压力目标"),
        ("F2_需求加权覆盖率", "F3_电网目标", "需求加权覆盖率", "电网压力目标"),
    ]
    for ax, (a, b, xl, yl) in zip(axes, pairs):
        if not front_m0.empty:
            ax.scatter(front_m0[a], front_m0[b], c=PALETTE["m0"], s=18, alpha=0.6, label="M0")
        if not front_m1.empty:
            ax.scatter(front_m1[a], front_m1[b], c=PALETTE["m1"], s=22, alpha=0.8, label="M1")
            r = front_m1.loc[picks["topsis"]]
            ax.scatter([r[a]], [r[b]], marker="*", s=160, c=PALETTE["fast"], edgecolors="k", zorder=5, label="TOPSIS")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)
    fig.suptitle("问题二 三目标两两投影")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题二_两两目标投影.png")
    plt.close(fig)

    # 3. HV 收敛
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    if hist_best:
        gens = [h["gen"] for h in hist_best]
        ax.plot(gens, [h["hv"] for h in hist_best], color=PALETTE["accent"], lw=1.8, label="Hypervolume")
        ax2 = ax.twinx()
        ax2.plot(gens, [h["feas_ratio"] for h in hist_best], color=PALETTE["cover"], lw=1.2, ls="--", label="可行比例")
        ax2.set_ylabel("可行个体比例")
        ax2.set_ylim(-0.05, 1.05)
        lines, labels = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(lines + l2, labels + lb2, loc="lower right", fontsize=8)
    ax.set_xlabel("迭代代数")
    ax.set_ylabel("Hypervolume")
    ax.set_title("问题二 Hypervolume 收敛曲线（M1 诊断，代表性种子）")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "问题二_Hypervolume收敛曲线.png")
    plt.close(fig)

    # 4. 新增快慢充配置
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    idx = np.arange(N_REGION)
    w = 0.38
    ax.bar(idx - w / 2, rec_x, width=w, color=PALETTE["fast"], edgecolor="k", lw=0.4, label="新增快充")
    ax.bar(idx + w / 2, rec_y, width=w, color=PALETTE["slow"], edgecolor="k", lw=0.4, label="新增慢充")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("新增桩数 / 台")
    ax.set_title("问题二 TOPSIS 推荐方案的区域新增快慢充配置")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for i, (xf, ys) in enumerate(zip(rec_x, rec_y)):
        ax.text(i - w / 2, xf, str(int(xf)), ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, ys, str(int(ys)), ha="center", va="bottom", fontsize=7)
    fig.savefig(FIGURES_DIR / "问题二_新增快慢充区域配置.png")
    plt.close(fig)

    # 5. 覆盖率对比
    S1 = coverage(data, rec_x, rec_y, scen_m1.beta)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(idx - w / 2, data.S0, width=w, color=PALETTE["before"], edgecolor="k", lw=0.4, label="规划前")
    ax.bar(idx + w / 2, S1, width=w, color=PALETTE["after"], edgecolor="k", lw=0.4, label="规划后")
    ax.axhline(0.9, color=PALETTE["fast"], ls="--", lw=1.1, label="90% 覆盖底线")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("面积覆盖率")
    ax.set_ylim(0, 1.08)
    ax.set_title("问题二 规划前后面积覆盖率对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "问题二_规划前后覆盖率对比.png")
    plt.close(fig)

    # 6. 电网压力对比
    from problem2_utils import growth
    rho1 = growth(data, rec_x, rec_y) * data.rho0
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(idx - w / 2, data.rho0, width=w, color=PALETTE["before"], edgecolor="k", lw=0.4, label="规划前")
    ax.bar(idx + w / 2, rho1, width=w, color=PALETTE["grid"], edgecolor="k", lw=0.4, label="规划后")
    ax.axhline(1.0, color=PALETTE["fast"], ls="--", lw=1.1, label="τ=1.00 上限")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])
    ax.set_xlabel("区域编号")
    ax.set_ylabel("区域最大电网压力 ρ")
    ax.set_title("问题二 规划前后电网最大压力对比")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(FIGURES_DIR / "问题二_规划前后电网压力对比.png")
    plt.close(fig)

    # 7. 敏感性热力图
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
    specs = [
        (sens_frames["beta"], "beta", "覆盖效率 β", axes[0, 0]),
        (sens_frames["delta"], "delta", "快慢比允许偏差 δ", axes[0, 1]),
        (sens_frames["tau"], "tau", "电网安全裕度 τ", axes[1, 0]),
        (sens_frames["alpha"], "alpha", "电网目标系数 α", axes[1, 1]),
    ]
    metrics = ["cost", "fast", "F2", "max_rho"]
    metric_names = ["投资成本", "新增快充", "加权覆盖率", "最大压力"]
    for df, key, title, ax in specs:
        mat = []
        ytick = []
        for _, row in df.iterrows():
            ytick.append(f"{key}={row[key]}")
            vec = []
            for m in metrics:
                val = row[m]
                vec.append(np.nan if pd.isna(val) else float(val))
            mat.append(vec)
        arr = np.asarray(mat, dtype=float)
        # 列内标准化便于同图比较；不可行保持空白
        arr_n = arr.copy()
        for j in range(arr_n.shape[1]):
            col = arr_n[:, j]
            if np.all(np.isnan(col)):
                continue
            lo, hi = np.nanmin(col), np.nanmax(col)
            if hi - lo < 1e-12:
                arr_n[:, j] = np.where(np.isnan(col), np.nan, 0.5)
            else:
                arr_n[:, j] = (col - lo) / (hi - lo)
        sns.heatmap(
            arr_n, ax=ax, cmap="YlGnBu", vmin=0, vmax=1, annot=arr,
            fmt=".3g", xticklabels=metric_names, yticklabels=ytick,
            cbar_kws={"label": "列内归一化"}, linewidths=0.4, linecolor="white",
        )
        ax.set_title(title)
    fig.suptitle("问题二 敏感性分析热力图（不可行情景为空白）")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "问题二_敏感性分析热力图.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.quick:
        pop, n_gen = 80, 60
        seeds = [42, 43, 44]
        sens_pop, sens_gen = 60, 40
    else:
        pop, n_gen = 300, 500
        seeds = SEEDS_FULL
        sens_pop, sens_gen = 200, 250

    t_all = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("读取附件与问题一预测 ...", flush=True)
    data = load_problem2_data()
    derived = derived_parameter_tables(data)

    scen_m0_s = make_strict_m0(data)
    scen_m1_s = make_strict_m1(data, delta=0.10)
    chk_m0_s = precheck_scenario(data, scen_m0_s)
    chk_m1_s = precheck_scenario(data, scen_m1_s)
    d5 = min_delta_for_region(data, 4, 1.0, 1.0)
    d9 = min_delta_for_region(data, 8, 1.0, 1.0)

    scen_m0 = make_diag_m0(data)
    scen_m1 = make_diag_m1(data, delta=0.10, lift_infeasible_delta=True)
    chk_m0 = precheck_scenario(data, scen_m0)
    chk_m1 = precheck_scenario(data, scen_m1)

    info_sheets = {
        "数据校验": derived["数据校验"],
        "派生参数": derived["派生参数"],
        "M0严格预检": chk_m0_s,
        "M1严格预检": chk_m1_s,
        "M0诊断预检": chk_m0,
        "M1诊断预检": chk_m1,
        "冲突摘要": pd.DataFrame([
            {
                "项目": "M0严格是否全市可行",
                "结果": "是" if chk_m0_s["feasible"].all() else "否",
                "说明": "；".join(chk_m0_s.loc[~chk_m0_s["feasible"], "conflict"].astype(str)),
            },
            {
                "项目": "M1严格是否全市可行",
                "结果": "是" if chk_m1_s["feasible"].all() else "否",
                "说明": "；".join(chk_m1_s.loc[~chk_m1_s["feasible"], "conflict"].astype(str)),
            },
            {
                "项目": "区域5恢复可行的最小δ",
                "结果": d5 if d5 is not None else "不存在",
                "说明": "覆盖90%+电网+结构同时成立",
            },
            {
                "项目": "区域9恢复90%覆盖的最小δ",
                "结果": d9 if d9 is not None else "不存在",
                "说明": "区域9电网裕度不足以支撑90%面积覆盖",
            },
            {
                "项目": "现状服务约束是否已满足",
                "结果": "全部满足" if np.all(data.K0 >= data.Dhat) else "存在缺口",
                "说明": "不得为激活该约束而修改80/20车次参数",
            },
            {
                "项目": "M0诊断覆盖下界_区域9",
                "结果": float(scen_m0.cover_req[8]),
                "说明": scen_m0.note,
            },
            {
                "项目": "M1诊断覆盖下界_区域9",
                "结果": float(scen_m1.cover_req[8]),
                "说明": scen_m1.note,
            },
            {
                "项目": "M1诊断区域5的δ",
                "结果": float(scen_m1.delta_vec()[4]),
                "说明": "仅在基准δ无整数解时上调",
            },
        ]),
    }
    save_xlsx(RESULTS_DIR / "问题二_基础数据与派生参数.xlsx", info_sheets)
    print("已写入 问题二_基础数据与派生参数.xlsx", flush=True)

    print("\n严格模型可行性：", flush=True)
    print("  M0", bool(chk_m0_s["feasible"].all()), chk_m0_s.loc[~chk_m0_s["feasible"], ["region", "conflict"]].to_dict("records"))
    print("  M1", bool(chk_m1_s["feasible"].all()), chk_m1_s.loc[~chk_m1_s["feasible"], ["region", "conflict"]].to_dict("records"))
    print("  区域5最小δ", d5, "区域9最小δ(90%)", d9)
    print("  诊断M0可行", bool(chk_m0["feasible"].all()), "诊断M1可行", bool(chk_m1["feasible"].all()))

    milp_m0 = milp_min_cost(data, scen_m0)
    milp_m1 = milp_min_cost(data, scen_m1)
    print("  MILP最低成本 M0", milp_m0["success"], milp_m0["cost"], "M1", milp_m1["success"], milp_m1["cost"])

    # M0 诊断：用于结构退化检查，种子可少于 M1
    m0_seeds = seeds[: max(3, min(5, len(seeds)))]
    res_m0, ref_m0 = run_multi_seed(data, scen_m0, m0_seeds, pop, n_gen, "M0_diag")
    Xm0, Fm0 = merge_fronts(res_m0)
    front_m0 = front_to_dataframe(data, scen_m0, Xm0, Fm0)
    deg = m0_degeneration_report(front_m0)
    # 结构支配检验：把快充等量改成慢充后，是否严格变优
    if not front_m0.empty:
        n_dom = 0
        n_mixed = 0
        for idx in front_m0.index:
            x, y = xy_from_row(front_m0, idx)
            if np.all(x == 0):
                continue
            n_mixed += 1
            y2 = y + x
            x2 = np.zeros_like(x)
            from problem2_utils import constraint_violations, total_cv
            if total_cv(constraint_violations(data, x2, y2, scen_m0)) > 1e-10:
                continue
            f1a, f2a, f3a, _, _ = objectives(data, x, y, scen_m0.beta, scen_m0.alpha)
            f1b, f2b, f3b, _, _ = objectives(data, x2, y2, scen_m0.beta, scen_m0.alpha)
            Fa = np.array([f1a, -f2a, f3a])
            Fb = np.array([f1b, -f2b, f3b])
            if np.all(Fb <= Fa + 1e-9) and np.any(Fb < Fa - 1e-9):
                n_dom += 1
        deg["n_mixed"] = n_mixed
        deg["n_mixed_dominated_by_allslow"] = n_dom
        deg["structural_slow_dominates"] = bool(n_mixed > 0 and n_dom == n_mixed)
    print("  M0 退化检查:", deg)
    save_xlsx(RESULTS_DIR / "问题二_M0_Pareto解集.xlsx", {
        "合并前沿": front_m0,
        "退化检查": pd.DataFrame([deg]),
        "MILP最低成本": pd.DataFrame([{"success": milp_m0["success"], "cost": milp_m0["cost"],
                                      "x": None if milp_m0["x"] is None else milp_m0["x"].tolist(),
                                      "y": None if milp_m0["y"] is None else milp_m0["y"].tolist()}]),
    })

    res_m1, ref_m1 = run_multi_seed(data, scen_m1, seeds, pop, n_gen, "M1_diag")
    Xm1, Fm1 = merge_fronts(res_m1)
    front_m1 = front_to_dataframe(data, scen_m1, Xm1, Fm1)
    print(f"  M1 合并去重 Pareto 数量 = {len(front_m1)}")
    if front_m1.empty:
        raise RuntimeError("M1 诊断模型未得到可行 Pareto 解，停止 TOPSIS。请检查预检。")

    topsis_df = topsis_rank(
        front_m1["F1_投资成本"].to_numpy(),
        front_m1["F2_需求加权覆盖率"].to_numpy(),
        front_m1["F3_电网目标"].to_numpy(),
    )
    eq_df = equal_weight_topsis(
        front_m1["F1_投资成本"].to_numpy(),
        front_m1["F2_需求加权覆盖率"].to_numpy(),
        front_m1["F3_电网目标"].to_numpy(),
    )
    w = topsis_df.attrs["weights"]
    entropy_extreme = bool(np.max(w) > 0.9)
    picks = pick_representatives(front_m1, topsis_df)
    rec_x, rec_y = xy_from_row(front_m1, picks["topsis"])

    save_xlsx(RESULTS_DIR / "问题二_M1_Pareto解集.xlsx", {
        "合并前沿": front_m1,
        "MILP最低成本": pd.DataFrame([{
            "success": milp_m1["success"], "cost": milp_m1["cost"],
            "message": milp_m1["message"],
            "x": None if milp_m1["x"] is None else milp_m1["x"].tolist(),
            "y": None if milp_m1["y"] is None else milp_m1["y"].tolist(),
        }]),
    })
    save_xlsx(RESULTS_DIR / "问题二_TOPSIS排序.xlsx", {
        "熵权TOPSIS": topsis_df,
        "等权对照": eq_df,
        "权重": pd.DataFrame([{
            "w1_成本": w[0], "w2_覆盖": w[1], "w3_电网": w[2],
            "是否极端(>0.9)": entropy_extreme,
        }]),
    })

    rec_detail = plan_detail(data, rec_x, rec_y, scen_m1)
    rec_audit = audit_plan(data, rec_x, rec_y, scen_m1)
    nb = neighborhood_check(data, scen_m1, rec_x, rec_y)
    rec_sum = summarize_plan(data, scen_m1, rec_x, rec_y, {
        "熵权_w1": float(w[0]),
        "熵权_w2": float(w[1]),
        "熵权_w3": float(w[2]),
        "TOPSIS贴近度": float(topsis_df.iloc[0]["贴近度_C"]),
        "全部约束最大违反量": rec_audit["max_any_violation"],
        "邻域支配解数量": nb["n_dominating"],
        "模型说明": scen_m1.note,
    })
    save_xlsx(RESULTS_DIR / "问题二_推荐配置方案.xlsx", {
        "区域配置": rec_detail,
        "全市汇总": pd.DataFrame([rec_sum]),
        "约束复核": pd.DataFrame([rec_audit]),
        "邻域复核": pd.DataFrame([{k: v for k, v in nb.items() if k != "dominating"}]),
    })

    cmp_df = build_representative_table(data, scen_m1, front_m1, picks, topsis_df)
    # M0 vs M1 对照：各自 TOPSIS 或最低成本
    compare_rows = []
    if not front_m0.empty:
        td0 = topsis_rank(front_m0["F1_投资成本"].to_numpy(), front_m0["F2_需求加权覆盖率"].to_numpy(), front_m0["F3_电网目标"].to_numpy())
        p0 = pick_representatives(front_m0, td0)
        x0, y0 = xy_from_row(front_m0, p0["topsis"])
        compare_rows.append({"模型": "M0诊断-TOPSIS", **summarize_plan(data, scen_m0, x0, y0)})
        x0c, y0c = xy_from_row(front_m0, p0["min_cost"])
        compare_rows.append({"模型": "M0诊断-最低成本", **summarize_plan(data, scen_m0, x0c, y0c)})
    compare_rows.append({"模型": "M1诊断-TOPSIS", **summarize_plan(data, scen_m1, rec_x, rec_y)})
    if milp_m0["success"]:
        compare_rows.append({"模型": "M0-MILP最低成本", **summarize_plan(data, scen_m0, milp_m0["x"], milp_m0["y"])})
    if milp_m1["success"]:
        compare_rows.append({"模型": "M1-MILP最低成本", **summarize_plan(data, scen_m1, milp_m1["x"], milp_m1["y"])})
    save_xlsx(RESULTS_DIR / "问题二_代表性方案对比.xlsx", {
        "四套方案": cmp_df,
        "M0与M1对照": pd.DataFrame(compare_rows),
    })

    # 稳定性
    hv_list, n_list, cost_list, fast_list, slow_list, rec_keys = [], [], [], [], [], []
    ref_front = Fm1
    seed_rows = []
    for r in res_m1:
        hv_list.append(hv_of_front(r.F, ref_m1))
        n_list.append(len(r.X))
        igd = igd_of_front(r.F, ref_front)
        if len(r.X):
            fd = front_to_dataframe(data, scen_m1, r.X, r.F)
            td = topsis_rank(fd["F1_投资成本"].to_numpy(), fd["F2_需求加权覆盖率"].to_numpy(), fd["F3_电网目标"].to_numpy())
            pk = pick_representatives(fd, td)
            xx, yy = xy_from_row(fd, pk["topsis"])
            cost_list.append(float((C_F * xx + C_S * yy).sum()))
            fast_list.append(int(xx.sum()))
            slow_list.append(int(yy.sum()))
            rec_keys.append(tuple(xx.tolist() + yy.tolist()))
        else:
            cost_list.append(np.nan)
            fast_list.append(0)
            slow_list.append(0)
            rec_keys.append(None)
        seed_rows.append({
            "seed": r.seed,
            "n_pareto": len(r.X),
            "hv": hv_list[-1],
            "igd": igd,
            "runtime_s": r.runtime_s,
            "hv_rel_last50": last50_hv_rel_change(r.history),
            "topsis_cost": cost_list[-1],
            "topsis_fast": fast_list[-1],
            "topsis_slow": slow_list[-1],
        })
    valid_keys = [k for k in rec_keys if k is not None]
    if valid_keys:
        from collections import Counter
        cnt = Counter(valid_keys)
        repeat_rate = cnt.most_common(1)[0][1] / len(valid_keys)
    else:
        repeat_rate = 0.0
    stability = {
        "pop_size": pop,
        "n_gen": n_gen,
        "seeds": seeds,
        "hv_mean": float(np.nanmean(hv_list)),
        "hv_std": float(np.nanstd(hv_list, ddof=1)) if len(hv_list) > 1 else 0.0,
        "n_pareto_mean": float(np.mean(n_list)),
        "n_pareto_std": float(np.std(n_list, ddof=1)) if len(n_list) > 1 else 0.0,
        "merged_pareto": int(len(front_m1)),
        "topsis_cost_mean": float(np.nanmean(cost_list)),
        "topsis_cost_range": float(np.nanmax(cost_list) - np.nanmin(cost_list)),
        "fast_mean": float(np.nanmean(fast_list)),
        "fast_std": float(np.nanstd(fast_list, ddof=1)) if len(fast_list) > 1 else 0.0,
        "slow_mean": float(np.nanmean(slow_list)),
        "slow_range": float(np.nanmax(slow_list) - np.nanmin(slow_list)),
        "recommendation_mode_rate": float(repeat_rate),
        "entropy_weights": w.tolist(),
        "entropy_extreme": entropy_extreme,
        "m0_degeneration": deg,
        "strict_m0_feasible": bool(chk_m0_s["feasible"].all()),
        "strict_m1_feasible": bool(chk_m1_s["feasible"].all()),
        "region5_min_delta": d5,
        "region9_min_delta_for_90cover": d9,
        "diag_note": scen_m1.note,
        "milp_m0_cost": milp_m0["cost"],
        "milp_m1_cost": milp_m1["cost"],
        "neighborhood_dominating": nb["n_dominating"],
        "runtime_total_hint_s": None,
        "seed_table": seed_rows,
    }
    (RESULTS_DIR / "问题二_算法稳定性.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 敏感性
    print("\n=== 敏感性分析 ===", flush=True)
    beta_rows, delta_rows, tau_rows, alpha_rows = [], [], [], []
    for beta in (0.6, 0.8, 1.0):
        print(f"  beta={beta}", flush=True)
        cell = sensitivity_cell(data, beta, 0.10, 1.00, 0.5, True, sens_pop, sens_gen, 42, lift_delta=True)
        cell["beta"] = beta
        beta_rows.append(cell)
        print("   ", {k: cell[k] for k in ("feasible", "cost", "fast", "slow", "infeasible_regions")})
    for delta in (0.05, 0.10, 0.15):
        print(f"  delta={delta}", flush=True)
        cell = sensitivity_cell(data, 1.0, delta, 1.00, 0.5, True, sens_pop, sens_gen, 42, lift_delta=False)
        cell["delta"] = delta
        delta_rows.append(cell)
        print("   ", {k: cell[k] for k in ("feasible", "cost", "fast", "slow", "infeasible_regions")})
    for tau in (0.85, 0.90, 1.00):
        print(f"  tau={tau}", flush=True)
        cell = sensitivity_cell(data, 1.0, 0.10, tau, 0.5, True, sens_pop, sens_gen, 42, lift_delta=True)
        cell["tau"] = tau
        tau_rows.append(cell)
        print("   ", {k: cell[k] for k in ("feasible", "cost", "fast", "slow", "infeasible_regions")})
    for alpha in (0.3, 0.5, 0.7):
        print(f"  alpha={alpha}", flush=True)
        cell = sensitivity_cell(data, 1.0, 0.10, 1.00, alpha, True, sens_pop, sens_gen, 42, lift_delta=True)
        cell["alpha"] = alpha
        alpha_rows.append(cell)
        print("   ", {k: cell[k] for k in ("feasible", "cost", "fast", "F2", "max_rho")})

    def _sens_df(rows, key):
        df = pd.DataFrame(rows)
        df["infeasible_regions"] = df["infeasible_regions"].apply(lambda v: ",".join(map(str, v)) if isinstance(v, list) else v)
        return df

    sens_frames = {
        "beta": _sens_df(beta_rows, "beta"),
        "delta": _sens_df(delta_rows, "delta"),
        "tau": _sens_df(tau_rows, "tau"),
        "alpha": _sens_df(alpha_rows, "alpha"),
    }
    save_xlsx(RESULTS_DIR / "问题二_敏感性分析.xlsx", sens_frames)

    # 代表性种子历史：选 HV 中位数的一次
    hist_best = []
    ok_hist = [r for r in res_m1 if r.history]
    if ok_hist:
        hvs = [h["hv"] for h in ok_hist[-1].history]
        hist_best = max(ok_hist, key=lambda r: r.history[-1]["hv"] if r.history else -1).history

    print("\n绘图 ...", flush=True)
    plot_all(data, scen_m1, front_m0, front_m1, hist_best, rec_x, rec_y, sens_frames, picks, topsis_df)

    elapsed = time.perf_counter() - t_all
    stability["runtime_total_hint_s"] = elapsed
    (RESULTS_DIR / "问题二_算法稳定性.json").write_text(
        json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n======== 验证摘要 ========", flush=True)
    print("M0严格可行:", bool(chk_m0_s["feasible"].all()))
    print("M1严格可行:", bool(chk_m1_s["feasible"].all()))
    print("M0诊断Pareto:", len(front_m0), "全慢充:", deg.get("all_slow"))
    print("M1诊断Pareto:", len(front_m1))
    print("推荐新增快/慢:", int(rec_x.sum()), int(rec_y.sum()), "成本", rec_sum["总投资成本"])
    print("熵权", w.tolist(), "贴近度", rec_sum["TOPSIS贴近度"])
    print("复核", rec_audit["pass"], "最大违反", rec_audit["max_any_violation"])
    print("总用时 %.1f s" % elapsed)
    print("输出目录:", RESULTS_DIR, FIGURES_DIR)


if __name__ == "__main__":
    main()
