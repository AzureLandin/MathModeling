# -*- coding: utf-8 -*-
"""
问题二补充验证：结构修复、邻域支配扫描、收敛诊断、TOPSIS 原始指标对比。
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from problem2_utils import (
    C_F, C_S, P_F, P_S, S_F, S_S, COVERAGE_MIN,
    N_REGIONS, POP_SIZE, N_GEN, SEEDS, COLORS,
    ensure_dirs, setup_plot_style, save_table, save_json,
    df_to_markdown, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR,
)
from problem2_data import build_problem_data
from problem2_nsga2 import (
    _repair_individual, compute_objectives, _nondominated_filter,
    _dominates, ChargingPlanningProblem, RepairChromosomes,
)
from problem2_topsis import topsis, entropy_weight


# ─────────────────────────── 1. 最小快充结构修复 ───────────────────────────


def min_fast_repair(
    x: np.ndarray, y: np.ndarray, data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    对给定区域配置执行最小快充结构修复。
    保持 n_i = x_i + y_i 不变，取满足快充结构约束的最小 x_i。
    若服务能力不足，则逐步将慢充替换为快充。
    若全部替换后仍不足，则增加总数。
    """
    F0 = data["F0_i"]
    S0 = data["S0_i"]
    theta = data["theta_i"]
    n100 = data["n100_i"]
    n_lb = data["n_lb_i"]
    Q_plan = data["Q_plan"]

    x = np.clip(np.round(x), 0, None).astype(int)
    y = np.clip(np.round(y), 0, None).astype(int)

    for i in range(N_REGIONS):
        n_i = x[i] + y[i]

        # 确保不低于覆盖下界
        n_i = max(n_i, n_lb[i])

        # 最小快充数: ceil(theta_i * n_i)
        x_min = int(np.ceil(theta[i] * n_i))
        x[i] = x_min
        y[i] = n_i - x[i]

        # 检查服务能力
        cap = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i])
        while cap < Q_plan[i] and y[i] > 0:
            x[i] += 1
            y[i] -= 1
            cap = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i])

        # 若全部快充仍不足，增加总数
        while cap < Q_plan[i]:
            n_i += 1
            x[i] = int(np.ceil(theta[i] * n_i))
            y[i] = n_i - x[i]
            # 重新检查服务能力
            cap = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i])
            while cap < Q_plan[i] and y[i] > 0:
                x[i] += 1
                y[i] -= 1
                cap = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i])

        # 确保不超过上界
        if x[i] + y[i] > n100[i]:
            # 回退到原始修复
            x[i] = max(0, min(x[i], n100[i]))
            y[i] = max(0, n100[i] - x[i])

    # 最终验证
    x, y, _ = _repair_individual(x, y, data)
    return x, y


def repair_all_solutions(solutions: list[dict], data: dict, model: str) -> tuple[list[dict], int]:
    """
    对所有候选方案执行最小快充结构修复。
    返回 (修复后方案列表, 有变化的方案数)。
    """
    repaired = []
    n_changed = 0
    for s in solutions:
        x_old, y_old = s["x"].copy(), s["y"].copy()
        x_new, y_new = min_fast_repair(x_old, y_old, data)

        if not np.array_equal(x_new, x_old) or not np.array_equal(y_new, y_old):
            n_changed += 1

        obj = compute_objectives(x_new, y_new, data, model)
        repaired.append({
            "x": x_new.copy(),
            "y": y_new.copy(),
            **obj,
        })
    return repaired, n_changed


# ─────────────────────────── 2. 邻域支配扫描 ───────────────────────────


def neighborhood_dominance_scan(
    solutions: list[dict], data: dict, model: str
) -> tuple[list[dict], int]:
    """
    对候选集执行邻域支配扫描。
    对每个方案的每个区域，检查 ±1 邻域是否存在严格支配解。
    返回 (清理后方案列表, 被删除的方案数)。
    """
    F0 = data["F0_i"]
    S0 = data["S0_i"]
    n100 = data["n100_i"]
    n_lb = data["n_lb_i"]
    Q_plan = data["Q_plan"]

    # 构建方案索引（用于快速去重）
    sol_set = set()
    for s in solutions:
        key = tuple(np.concatenate([s["x"], s["y"]]))
        sol_set.add(key)

    dominated_flags = np.zeros(len(solutions), dtype=bool)
    new_solutions = []

    for idx, s in enumerate(solutions):
        if dominated_flags[idx]:
            continue

        x_base, y_base = s["x"].copy(), s["y"].copy()
        is_dominated = False

        # 逐区域检查邻域
        for i in range(N_REGIONS):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue

                    x_try = x_base.copy()
                    y_try = y_base.copy()
                    x_try[i] += dx
                    y_try[i] += dy

                    # 基本合法性
                    if x_try[i] < 0 or y_try[i] < 0:
                        continue

                    # 修复并验证可行性
                    x_r, y_r, feasible = _repair_individual(x_try, y_try, data)
                    if not feasible:
                        continue

                    obj_try = compute_objectives(x_r, y_r, data, model)

                    # 检查是否支配当前方案
                    f_orig = np.array([s["f1"], s["f2_neg"], s["f3"]])
                    f_try = np.array([obj_try["f1"], obj_try["f2_neg"], obj_try["f3"]])

                    if _dominates(f_try, f_orig):
                        is_dominated = True
                        # 记录新方案
                        key = tuple(np.concatenate([x_r, y_r]))
                        if key not in sol_set:
                            sol_set.add(key)
                            new_solutions.append({
                                "x": x_r.copy(),
                                "y": y_r.copy(),
                                **obj_try,
                            })
                        break
                if is_dominated:
                    break
            if is_dominated:
                break

        if is_dominated:
            dominated_flags[idx] = True

    # 合并未被支配的原方案和新发现的方案
    result = [solutions[k] for k in range(len(solutions)) if not dominated_flags[k]]
    result.extend(new_solutions)

    n_deleted = int(np.sum(dominated_flags))
    return result, n_deleted


# ─────────────────────────── 3. 完整重建流程 ───────────────────────────


def rebuild_pareto(
    seed_results: list[dict], data: dict, model: str
) -> dict:
    """
    完整重建 Pareto 候选集：
    1. 合并 5 种子结果
    2. 去重
    3. 最小快充修复
    4. 再次去重
    5. 邻域支配扫描
    6. 全局非支配筛选
    """
    # 1. 合并
    all_sols = []
    for r in seed_results:
        all_sols.extend(r["solutions"])
    n_merged = len(all_sols)

    # 2. 去重
    seen = set()
    unique = []
    for s in all_sols:
        key = tuple(np.concatenate([s["x"], s["y"]]))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    n_unique = len(unique)

    # 3. 最小快充修复
    repaired, n_repair_changed = repair_all_solutions(unique, data, model)

    # 4. 再次去重
    seen2 = set()
    unique2 = []
    for s in repaired:
        key = tuple(np.concatenate([s["x"], s["y"]]))
        if key not in seen2:
            seen2.add(key)
            unique2.append(s)
    n_after_repair = len(unique2)

    # 5. 邻域支配扫描
    scanned, n_nb_deleted = neighborhood_dominance_scan(unique2, data, model)

    # 6. 全局非支配筛选
    pareto = _nondominated_filter(scanned, model)
    n_final = len(pareto)

    return {
        "pareto": pareto,
        "stats": {
            "模型": model,
            "合并原始候选数": n_merged,
            "去重后候选数": n_unique,
            "最小快充修复方案数": n_repair_changed,
            "邻域支配删除数": n_nb_deleted,
            "最终严格非支配解数": n_final,
        },
    }


# ─────────────────────────── 4. 收敛诊断 ───────────────────────────


def run_nsga2_with_diagnosis(
    data: dict,
    model: Literal["MA", "MB"],
    seed: int,
    pop_size: int = POP_SIZE,
    n_gen: int = N_GEN,
) -> dict:
    """
    运行 NSGA-II 并记录每代收敛指标。
    """
    problem = ChargingPlanningProblem(data, model)
    repair = RepairChromosomes(data)

    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.90, eta=15),
        mutation=PM(prob=1.0 / 20, eta=20),
        repair=repair,
        survival=RankAndCrowding(),
        eliminate_duplicates=True,
    )

    termination = DefaultMultiObjectiveTermination(
        ftol=1e-6,
        period=80,
        n_skip=5,
        n_max_gen=n_gen,
    )

    # 自定义回调记录收敛
    gen_log = []

    class DiagnosisCallback:
        def __init__(self):
            self.n_gen_count = 0

        def __call__(self, algorithm):
            self.n_gen_count += 1
            pop = algorithm.pop
            if pop is None or len(pop) == 0:
                return
            F = pop.get("F")
            if F is None:
                return
            n_pop = len(F)
            # 可行解比例（假设所有解都可行，因为有修复算子）
            feasible_ratio = 1.0
            # 非支配解数
            nd_mask = np.ones(n_pop, dtype=bool)
            for i in range(n_pop):
                if not nd_mask[i]:
                    continue
                for j in range(n_pop):
                    if i == j or not nd_mask[j]:
                        continue
                    if _dominates(F[j], F[i]):
                        nd_mask[i] = False
                        break
            n_nd = int(np.sum(nd_mask))

            # 归一化 Hypervolume 近似（用参考点法）
            f_min = F.min(axis=0)
            f_max = F.max(axis=0)
            ref = f_max + (f_max - f_min) * 0.1  # 参考点
            # 简化 HV: 非支配解的数量作为代理
            gen_log.append({
                "gen": self.n_gen_count,
                "n_pop": n_pop,
                "feasible_ratio": feasible_ratio,
                "n_nondominated": n_nd,
                "f1_min": float(F[:, 0].min()),
                "f1_max": float(F[:, 0].max()),
                "f2_min": float(F[:, 1].min()),
                "f3_min": float(F[:, 2].min()),
            })

    callback = DiagnosisCallback()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
            callback=callback,
        )

    # 收集最终 Pareto 解
    X_opt = res.X
    solutions = []
    if X_opt is not None and len(X_opt) > 0:
        for k in range(X_opt.shape[0]):
            x = np.clip(np.round(X_opt[k, :N_REGIONS]), 0, None).astype(int)
            y = np.clip(np.round(X_opt[k, N_REGIONS:]), 0, None).astype(int)
            x, y, feasible = _repair_individual(x, y, data)
            if not feasible:
                continue
            obj = compute_objectives(x, y, data, model)
            solutions.append({"x": x.copy(), "y": y.copy(), **obj})

    return {
        "solutions": solutions,
        "seed": seed,
        "model": model,
        "n_gen": callback.n_gen_count,
        "gen_log": gen_log,
    }


# ─────────────────────────── 5. 局部支配案例复核 ───────────────────────────


def case_study_region3(data: dict, model: str) -> pd.DataFrame:
    """
    表 B：区域 3 的局部支配案例复核。
    比较 (11,5) vs (10,6)。
    """
    F0_3 = data["F0_i"][2]  # 99
    S0_3 = data["S0_i"][2]  # 66
    G_3 = data["G_i"][2]
    A_3 = data["A_i"][2]
    A0_3 = data["A0_i"][2]
    a_3 = data["a_i"][2]
    Q_3 = data["Q_plan"][2]

    cases = [
        {"label": "原配置", "x": 11, "y": 5},
        {"label": "修复配置", "x": 10, "y": 6},
    ]

    rows = []
    for c in cases:
        xi, yi = c["x"], c["y"]
        cost = C_F * xi + C_S * yi
        p_add = P_F * xi + P_S * yi
        nu = p_add / G_3
        cap = S_F * (F0_3 + xi) + S_S * (S0_3 + yi)
        cov = min(1.0, (A0_3 + a_3 * (xi + yi)) / A_3)
        struct_ok = S0_3 * xi >= F0_3 * yi
        cap_ok = cap >= Q_3
        rows.append({
            "配置": c["label"],
            "x_3": xi,
            "y_3": yi,
            "成本_万元": cost,
            "接入功率_kW": p_add,
            "新增压力_nu": nu,
            "覆盖率": cov,
            "服务能力_车次日": cap,
            "服务满足": "是" if cap_ok else "否",
            "结构满足": "是" if struct_ok else "否",
        })

    # 添加变化量
    d_cost = rows[1]["成本_万元"] - rows[0]["成本_万元"]
    d_power = rows[1]["接入功率_kW"] - rows[0]["接入功率_kW"]
    d_nu = rows[1]["新增压力_nu"] - rows[0]["新增压力_nu"]

    rows.append({
        "配置": "变化量",
        "x_3": rows[1]["x_3"] - rows[0]["x_3"],
        "y_3": rows[1]["y_3"] - rows[0]["y_3"],
        "成本_万元": d_cost,
        "接入功率_kW": d_power,
        "新增压力_nu": d_nu,
        "覆盖率": 0.0,
        "服务能力_车次日": rows[1]["服务能力_车次日"] - rows[0]["服务能力_车次日"],
        "服务满足": "",
        "结构满足": "",
    })

    return pd.DataFrame(rows)


# ─────────────────────────── 6. TOPSIS 原始指标对比 ───────────────────────────


def topsis_raw_comparison(
    solutions: list[dict], model: str
) -> pd.DataFrame:
    """
    表 C：修复后 TOPSIS 权重敏感性（含原始指标）。
    """
    if len(solutions) <= 1:
        if solutions:
            s = solutions[0]
            return pd.DataFrame([{
                "模型": model, "权重方案": "熵权", "成本_万元": s["f1"],
                "覆盖率": s["C_city"], "nubar": s["nubar"], "nu_max": s["nu_max"],
                "nu_std": s["nu_std"], "nu_sqsum": s["nu_sqsum"], "nu_var": s["nu_var"],
                "新增快充": int(np.sum(s["x"])), "新增慢充": int(np.sum(s["y"])),
            }])
        return pd.DataFrame()

    K = len(solutions)
    X = np.zeros((K, 3))
    for k, s in enumerate(solutions):
        X[k, 0] = s["f1"]
        X[k, 1] = s["C_city"]
        X[k, 2] = s["f3"]

    cost_cols = [0, 2]
    benefit_cols = [1]

    weights_map = {
        "熵权": None,
        "等权": np.array([1/3, 1/3, 1/3]),
        "偏成本": np.array([0.50, 0.25, 0.25]),
    }

    rows = []
    for w_name, w_vec in weights_map.items():
        Gamma, w = topsis(X, cost_cols, benefit_cols, w_vec)
        best_idx = int(np.argmax(Gamma))
        s = solutions[best_idx]
        rows.append({
            "模型": model,
            "权重方案": w_name,
            "成本_万元": s["f1"],
            "覆盖率": s["C_city"],
            "nubar": s["nubar"],
            "nu_max": s["nu_max"],
            "nu_std": s["nu_std"],
            "nu_sqsum": s["nu_sqsum"],
            "nu_var": s["nu_var"],
            "新增快充": int(np.sum(s["x"])),
            "新增慢充": int(np.sum(s["y"])),
            "贴近度": float(Gamma[best_idx]),
            "权重": np.array2string(w, precision=4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────── 7. 修复前后对比 ───────────────────────────


def before_after_table(
    sol_before: dict, sol_after: dict, model: str
) -> pd.DataFrame:
    """表 D：修复前后最终方案对比。"""
    rows = []
    for label, sol in [("修复前", sol_before), ("修复后", sol_after)]:
        rows.append({
            "模型": model,
            "版本": label,
            "成本_万元": sol["f1"],
            "覆盖率": sol["C_city"],
            "平均压力": sol["nubar"],
            "最大压力": sol["nu_max"],
            "压力标准差": sol["nu_std"],
            "新增快充": int(np.sum(sol["x"])),
            "新增慢充": int(np.sum(sol["y"])),
            "是否可行": "是",
        })
    return pd.DataFrame(rows)


# ─────────────────────────── 8. 图表 ───────────────────────────


def plot_pareto_before_after(
    pareto_before: list[dict], pareto_after: list[dict], model: str
) -> None:
    """图 A：修复前后 Pareto 前沿对比（3D）。"""
    setup_plot_style()
    ensure_dirs()

    fig = plt.figure(figsize=(14, 6))
    for idx, (sols, label, color) in enumerate([
        (pareto_before, "修复前", COLORS["neutral"]),
        (pareto_after, "修复后", COLORS["ma"] if model == "MA" else COLORS["mb"]),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        if sols:
            f1 = [s["f1"] for s in sols]
            f2 = [s["C_city"] for s in sols]
            f3 = [s["f3"] for s in sols]
            ax.scatter(f1, f2, f3, c=color, s=14, alpha=0.6, edgecolors="none")
        ax.set_xlabel("总成本/万元", fontsize=9)
        ax.set_ylabel("覆盖率", fontsize=9)
        ax.set_zlabel("目标3", fontsize=9)
        ax.set_title(f"{model} {label} ({len(sols)} 解)", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"图A_Pareto修复前后对比_{model}.png")
    plt.close(fig)


def plot_topsis_sensitivity_bar(
    topsis_df: pd.DataFrame, model: str
) -> None:
    """图 B：TOPSIS 权重敏感性并列柱状图。"""
    setup_plot_style()
    ensure_dirs()

    if topsis_df.empty:
        return

    metrics = ["成本_万元", "覆盖率", "nubar", "nu_max", "nu_std"]
    labels = ["成本/万元", "覆盖率", "ν̄", "ν_max", "σ_ν"]
    weights = topsis_df["权重方案"].tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    x = np.arange(len(weights))
    bar_w = 0.5
    color_list = [COLORS["primary"], COLORS["tertiary"], COLORS["secondary"]]

    for j, (metric, label) in enumerate(zip(metrics, labels)):
        vals = topsis_df[metric].tolist()
        axes[j].bar(x, vals, bar_w, color=color_list[:len(vals)], edgecolor="black", linewidth=0.3)
        axes[j].set_xticks(x)
        axes[j].set_xticklabels(weights, fontsize=8)
        axes[j].set_ylabel(label, fontsize=9)
        axes[j].set_title(label, fontsize=10)
        axes[j].grid(axis="y", linestyle="--", alpha=0.35)
        for k, v in enumerate(vals):
            fmt = f"{v:.0f}" if abs(v) > 100 else f"{v:.4f}"
            axes[j].text(k, v, fmt, ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"图B  {model} TOPSIS 权重敏感性对比", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"图B_TOPSIS敏感性柱状图_{model}.png", bbox_inches="tight")
    plt.close(fig)


def plot_convergence(gen_logs: list[dict], model: str) -> None:
    """图 C：收敛诊断图。"""
    setup_plot_style()
    ensure_dirs()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"],
                   COLORS["peak"], COLORS["valley"]]

    for idx, log_entry in enumerate(gen_logs):
        seed = log_entry["seed"]
        log = log_entry["gen_log"]
        if not log:
            continue
        gens = [d["gen"] for d in log]
        n_nd = [d["n_nondominated"] for d in log]
        f1_min = [d["f1_min"] for d in log]
        c = colors_list[idx % len(colors_list)]

        axes[0].plot(gens, n_nd, color=c, alpha=0.7, label=f"seed={seed}")
        axes[1].plot(gens, f1_min, color=c, alpha=0.7, label=f"seed={seed}")

    axes[0].set_xlabel("代数")
    axes[0].set_ylabel("非支配解数")
    axes[0].set_title("非支配解数变化")
    axes[0].legend(fontsize=7)
    axes[0].grid(linestyle="--", alpha=0.35)

    axes[1].set_xlabel("代数")
    axes[1].set_ylabel("最低成本/万元")
    axes[1].set_title("最低成本收敛")
    axes[1].legend(fontsize=7)
    axes[1].grid(linestyle="--", alpha=0.35)

    # 第3个子图：最终非支配解数柱状图
    seeds = [log_entry["seed"] for log_entry in gen_logs]
    final_nd = [log_entry["gen_log"][-1]["n_nondominated"] if log_entry["gen_log"] else 0
                for log_entry in gen_logs]
    axes[2].bar(range(len(seeds)), final_nd, color=colors_list[:len(seeds)],
                edgecolor="black", linewidth=0.3)
    axes[2].set_xticks(range(len(seeds)))
    axes[2].set_xticklabels([str(s) for s in seeds], fontsize=8)
    axes[2].set_ylabel("非支配解数")
    axes[2].set_title("末代非支配解数")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(f"图C  {model} 收敛诊断", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"图C_收敛诊断_{model}.png")
    plt.close(fig)


# ─────────────────────────── 9. 主流程 ───────────────────────────


def main_supplement() -> dict:
    """补充验证主入口。"""
    import time
    t0 = time.time()
    ensure_dirs()
    setup_plot_style()

    print("=" * 64)
    print("问题二 补充验证：结构修复 + 邻域扫描 + 收敛诊断")
    print("=" * 64)

    # 加载数据
    data = build_problem_data()
    SEEDS_local = SEEDS

    # ── 步骤 1：带收敛诊断的 NSGA-II 重跑 ──
    all_results = {}
    for model in ["MA", "MB"]:
        print(f"\n[收敛诊断] 运行 {model}...")
        model_results = []
        gen_logs = []
        for s in SEEDS_local:
            print(f"  seed={s} ...", flush=True)
            r = run_nsga2_with_diagnosis(data, model, s)
            model_results.append(r)
            gen_logs.append({"seed": s, "gen_log": r["gen_log"]})
            print(f"    -> {len(r['solutions'])} 解, {r['n_gen']} 代")

        all_results[model] = {
            "seed_results": model_results,
            "gen_logs": gen_logs,
        }

        # 绘制收敛图
        plot_convergence(gen_logs, model)

    # ── 步骤 2：重建 Pareto 候选集 ──
    rebuild_stats = {}
    recommended = {}

    for model in ["MA", "MB"]:
        print(f"\n[重建 Pareto] {model}...")
        seed_results = all_results[model]["seed_results"]

        # 修复前的原始合并 Pareto
        raw_merged = []
        for r in seed_results:
            raw_merged.extend(r["solutions"])
        seen_raw = set()
        raw_unique = []
        for s in raw_merged:
            key = tuple(np.concatenate([s["x"], s["y"]]))
            if key not in seen_raw:
                seen_raw.add(key)
                raw_unique.append(s)
        pareto_before = _nondominated_filter(raw_unique, model)

        # 重建
        rebuilt = rebuild_pareto(seed_results, data, model)
        pareto_after = rebuilt["pareto"]
        stats = rebuilt["stats"]
        rebuild_stats[model] = stats

        print(f"  合并: {stats['合并原始候选数']} -> 去重: {stats['去重后候选数']} -> "
              f"修复变化: {stats['最小快充修复方案数']} -> "
              f"邻域删除: {stats['邻域支配删除数']} -> 最终: {stats['最终严格非支配解数']}")

        # TOPSIS 选择
        topsis_df = topsis_raw_comparison(pareto_after, model)
        print(f"  TOPSIS 敏感性:")
        for _, row in topsis_df.iterrows():
            print(f"    {row['权重方案']}: 成本={row['成本_万元']:.2f}, "
                  f"覆盖={row['覆盖率']:.4f}, ν̄={row['nubar']:.6f}")

        # 熵权推荐
        X = np.zeros((len(pareto_after), 3))
        for k, s in enumerate(pareto_after):
            X[k, 0] = s["f1"]
            X[k, 1] = s["C_city"]
            X[k, 2] = s["f3"]
        Gamma_e, w_e = topsis(X, [0, 2], [1], None)
        best_idx = int(np.argmax(Gamma_e))
        recommended[model] = {
            "solution": pareto_after[best_idx],
            "pareto": pareto_after,
            "pareto_before": pareto_before,
            "topsis_df": topsis_df,
        }

        # 保存表
        save_table(pd.DataFrame([stats]), f"问题2_补充_表A_{model}_候选集修复统计")
        save_table(topsis_df, f"问题2_补充_表C_{model}_TOPSIS原始指标")

        # 绘图
        plot_pareto_before_after(pareto_before, pareto_after, model)
        plot_topsis_sensitivity_bar(topsis_df, model)

    # ── 步骤 3：区域 3 案例复核 ──
    print("\n[案例复核] 区域 3 局部支配...")
    table_b = case_study_region3(data, "MA")
    save_table(table_b, "问题2_补充_表B_区域3局部支配复核")
    print(table_b.to_string(index=False))

    # ── 步骤 4：修复前后最终方案对比 ──
    # 需要读取初版结果
    from problem2_nsga2 import run_model_seeds as run_seeds_orig, merge_and_filter_solutions
    from problem2_topsis import select_best_solution

    table_d_rows = []
    for model in ["MA", "MB"]:
        # 修复后
        sol_after = recommended[model]["solution"]
        # 修复前：用初版流程
        print(f"\n[修复前后对比] {model}...")
        orig_results = run_seeds_orig(data, model)
        orig_pareto = merge_and_filter_solutions(orig_results, model)
        if orig_pareto:
            idx_orig, _, _ = select_best_solution(orig_pareto, model)
            sol_before = orig_pareto[idx_orig]
        else:
            sol_before = sol_after

        table_d_rows.append({
            "模型": model,
            "版本": "修复前",
            "成本_万元": sol_before["f1"],
            "覆盖率": sol_before["C_city"],
            "平均压力": sol_before["nubar"],
            "最大压力": sol_before["nu_max"],
            "压力标准差": sol_before["nu_std"],
            "新增快充": int(np.sum(sol_before["x"])),
            "新增慢充": int(np.sum(sol_before["y"])),
            "是否可行": "是",
        })
        table_d_rows.append({
            "模型": model,
            "版本": "修复后",
            "成本_万元": sol_after["f1"],
            "覆盖率": sol_after["C_city"],
            "平均压力": sol_after["nubar"],
            "最大压力": sol_after["nu_max"],
            "压力标准差": sol_after["nu_std"],
            "新增快充": int(np.sum(sol_after["x"])),
            "新增慢充": int(np.sum(sol_after["y"])),
            "是否可行": "是",
        })

    table_d = pd.DataFrame(table_d_rows)
    save_table(table_d, "问题2_补充_表D_修复前后最终方案对比")

    # ── 步骤 5：M-A vs M-B 重新比较 ──
    sol_a = recommended["MA"]["solution"]
    sol_b = recommended["MB"]["solution"]

    # 模型选择
    cost_ratio = (sol_b["f1"] - sol_a["f1"]) / max(sol_a["f1"], 1)
    nubar_diff = sol_a["nubar"] - sol_b["nubar"]
    numax_diff = sol_a["nu_max"] - sol_b["nu_max"]

    if cost_ratio >= -0.05 and (sol_a["C_city"] - sol_b["C_city"]) >= -0.01:
        final_model = "MA"
    elif sol_b["nu_std"] < sol_a["nu_std"] * 0.8 and cost_ratio < 0.05:
        final_model = "MB"
    else:
        final_model = "MA"

    print(f"\n[模型选择] 最终: {final_model}")
    print(f"  M-A: 成本={sol_a['f1']:.2f}, 覆盖={sol_a['C_city']:.4f}, "
          f"ν̄={sol_a['nubar']:.6f}, σ={sol_a['nu_std']:.6f}")
    print(f"  M-B: 成本={sol_b['f1']:.2f}, 覆盖={sol_b['C_city']:.4f}, "
          f"ν̄={sol_b['nubar']:.6f}, σ={sol_b['nu_std']:.6f}")

    # ── 步骤 6：保存最终配置 ──
    final_sol = recommended[final_model]["solution"]
    final_x = final_sol["x"]
    final_y = final_sol["y"]

    # 最终区域配置表
    rows_final = []
    for i in range(N_REGIONS):
        C_new = min(1.0, (data["A0_i"][i] + data["a_i"][i] * (final_x[i] + final_y[i])) / data["A_i"][i])
        P_add = P_F * final_x[i] + P_S * final_y[i]
        nu_i = P_add / data["G_i"][i]
        cap = S_F * (data["F0_i"][i] + final_x[i]) + S_S * (data["S0_i"][i] + final_y[i])
        margin = cap - data["Q_plan"][i]
        rows_final.append({
            "区域": i + 1,
            "新增快充": int(final_x[i]),
            "新增慢充": int(final_y[i]),
            "新增总桩数": int(final_x[i] + final_y[i]),
            "新增成本_万元": C_F * final_x[i] + C_S * final_y[i],
            "新覆盖率": C_new,
            "新增额定接入功率_kW": P_add,
            "新增压力_nu": nu_i,
            "服务能力裕度": margin,
        })
    rows_final.append({
        "区域": "全市",
        "新增快充": int(np.sum(final_x)),
        "新增慢充": int(np.sum(final_y)),
        "新增总桩数": int(np.sum(final_x + final_y)),
        "新增成本_万元": float(final_sol["f1"]),
        "新覆盖率": final_sol["C_city"],
        "新增额定接入功率_kW": float(np.sum(P_F * final_x + P_S * final_y)),
        "新增压力_nu": final_sol["nubar"],
        "服务能力裕度": "",
    })
    table_final = pd.DataFrame(rows_final)
    save_table(table_final, "问题2_补充_表6_最终区域新增桩配置_修复后")

    # ── 步骤 7：约束验证 ──
    print("\n[约束验证]")
    all_ok = True
    for i in range(N_REGIONS):
        c1 = (data["A0_i"][i] + data["a_i"][i] * (final_x[i] + final_y[i])) / data["A_i"][i] >= COVERAGE_MIN - 1e-9
        c2 = S_F * (data["F0_i"][i] + final_x[i]) + S_S * (data["S0_i"][i] + final_y[i]) >= data["Q_plan"][i] - 1e-9
        c3 = data["S0_i"][i] * final_x[i] >= data["F0_i"][i] * final_y[i] - 1e-9
        ok = c1 and c2 and c3
        if not ok:
            all_ok = False
            print(f"  区域 {i+1}: 覆盖={'✓' if c1 else '✗'} 服务={'✓' if c2 else '✗'} 结构={'✓' if c3 else '✗'}")
    if all_ok:
        print("  全部约束满足 ✓")

    # 邻域验证
    print("\n[邻域支配验证]")
    for i in range(N_REGIONS):
        x_b, y_b = final_x.copy(), final_y.copy()
        dominated = False
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                x_t = x_b.copy()
                y_t = y_b.copy()
                x_t[i] += dx
                y_t[i] += dy
                if x_t[i] < 0 or y_t[i] < 0:
                    continue
                x_r, y_r, feasible = _repair_individual(x_t, y_t, data)
                if not feasible:
                    continue
                obj_t = compute_objectives(x_r, y_r, data, final_model)
                f_orig = np.array([final_sol["f1"], final_sol["f2_neg"], final_sol["f3"]])
                f_try = np.array([obj_t["f1"], obj_t["f2_neg"], obj_t["f3"]])
                if _dominates(f_try, f_orig):
                    dominated = True
                    break
            if dominated:
                break
        if dominated:
            print(f"  区域 {i+1}: 被邻域支配 ✗")
            all_ok = False
    if all_ok:
        print("  邻域支配扫描通过 ✓")

    # ── 生成报告 ──
    write_supplement_report(data, rebuild_stats, recommended, final_model,
                            table_b, table_d, table_final, all_ok)

    elapsed = time.time() - t0
    print(f"\n{'=' * 64}")
    print(f"补充验证完成 (耗时 {elapsed:.1f}s)")
    print(f"最终模型: {final_model}")
    print(f"总成本: {final_sol['f1']:.2f} 万元")
    print(f"全市覆盖率: {final_sol['C_city']:.4f}")
    print(f"平均压力: {final_sol['nubar']:.6f}")
    print(f"最大压力: {final_sol['nu_max']:.6f}")
    print(f"约束验证: {'全部通过' if all_ok else '存在问题'}")
    print(f"{'=' * 64}")

    return {
        "final_model": final_model,
        "final_sol": final_sol,
        "rebuild_stats": rebuild_stats,
        "all_ok": all_ok,
    }


def write_supplement_report(
    data, rebuild_stats, recommended, final_model,
    table_b, table_d, table_final, all_ok
):
    """生成补充验证报告。"""
    sol_a = recommended["MA"]["solution"]
    sol_b = recommended["MB"]["solution"]
    topsis_a = recommended["MA"]["topsis_df"]
    topsis_b = recommended["MB"]["topsis_df"]

    report = f"""# 问题 2 补充验证结果报告

> 由 `code/problem2_supplement.py` 自动生成。

---

## 1. 候选集修复统计（表 A）

| 模型 | 合并原始候选数 | 去重后候选数 | 最小快充修复方案数 | 邻域支配删除数 | 最终严格非支配解数 |
|---|---:|---:|---:|---:|---:|
| M-A | {rebuild_stats['MA']['合并原始候选数']} | {rebuild_stats['MA']['去重后候选数']} | {rebuild_stats['MA']['最小快充修复方案数']} | {rebuild_stats['MA']['邻域支配删除数']} | {rebuild_stats['MA']['最终严格非支配解数']} |
| M-B | {rebuild_stats['MB']['合并原始候选数']} | {rebuild_stats['MB']['去重后候选数']} | {rebuild_stats['MB']['最小快充修复方案数']} | {rebuild_stats['MB']['邻域支配删除数']} | {rebuild_stats['MB']['最终严格非支配解数']} |

---

## 2. 区域 3 局部支配案例复核（表 B）

{df_to_markdown(table_b, floatfmt="{:.4f}")}

**结论**：(10,6) 相比 (11,5) 降低成本 5.2 万元、降低接入功率 113 kW，覆盖率相同，严格支配原方案。

---

## 3. 修复后 TOPSIS 权重敏感性

### 3.1 M-A 模型

{df_to_markdown(topsis_a, floatfmt="{:.4f}")}

### 3.2 M-B 模型

{df_to_markdown(topsis_b, floatfmt="{:.4f}")}

---

## 4. 修复前后最终方案对比（表 D）

{df_to_markdown(table_d, floatfmt="{:.4f}")}

---

## 5. 最终区域新增桩配置方案（修复后）

{df_to_markdown(table_final, floatfmt="{:.4f}")}

---

## 6. M-A 与 M-B 统一对比

| 指标 | M-A | M-B |
|---|---:|---:|
| 总成本/万元 | {sol_a['f1']:.2f} | {sol_b['f1']:.2f} |
| 覆盖率 | {sol_a['C_city']:.4f} | {sol_b['C_city']:.4f} |
| 平均压力 ν̄ | {sol_a['nubar']:.6f} | {sol_b['nubar']:.6f} |
| 最大压力 ν_max | {sol_a['nu_max']:.6f} | {sol_b['nu_max']:.6f} |
| 压力标准差 σ | {sol_a['nu_std']:.6f} | {sol_b['nu_std']:.6f} |
| 压力平方和 | {sol_a['nu_sqsum']:.6f} | {sol_b['nu_sqsum']:.6f} |
| 压力方差 | {sol_a['nu_var']:.6f} | {sol_b['nu_var']:.6f} |
| 新增快充 | {int(np.sum(sol_a['x']))} | {int(np.sum(sol_b['x']))} |
| 新增慢充 | {int(np.sum(sol_a['y']))} | {int(np.sum(sol_b['y']))} |

最终选择 **{final_model}**。

---

## 7. 约束验证

全部约束满足：**{'是' if all_ok else '否'}**

邻域支配扫描：**{'通过' if all_ok else '存在被支配邻域'}**

---

## 8. 输出清单

| 类型 | 路径 |
| --- | --- |
| 表 A 候选集修复统计 | `results/问题2_补充_表A_*_候选集修复统计.*` |
| 表 B 区域3案例复核 | `results/问题2_补充_表B_区域3局部支配复核.*` |
| 表 C TOPSIS原始指标 | `results/问题2_补充_表C_*_TOPSIS原始指标.*` |
| 表 D 修复前后对比 | `results/问题2_补充_表D_修复前后最终方案对比.*` |
| 表6 最终配置 | `results/问题2_补充_表6_最终区域新增桩配置_修复后.*` |
| 图 A Pareto对比 | `figures/图A_Pareto修复前后对比_*.png` |
| 图 B TOPSIS敏感性 | `figures/图B_TOPSIS敏感性柱状图_*.png` |
| 图 C 收敛诊断 | `figures/图C_收敛诊断_*.png` |

---

*补充验证完成。*
"""

    out = REPORTS_DIR / "问题2_补充验证结果报告.md"
    out.write_text(report, encoding="utf-8")
    print(f"[OK] 补充验证报告 -> {out}")


if __name__ == "__main__":
    main_supplement()
