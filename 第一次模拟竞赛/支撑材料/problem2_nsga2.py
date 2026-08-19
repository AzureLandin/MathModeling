# -*- coding: utf-8 -*-
"""
问题二 NSGA-II 优化：M-A（压力平方和）与 M-B（压力方差）。
使用 pymoo 实现整数编码 + 约束修复。
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from problem2_utils import (
    C_F, C_S, P_F, P_S, S_F, S_S, COVERAGE_MIN,
    N_REGIONS, POP_SIZE, N_GEN,
)


# ─────────────────────────── 约束修复 ───────────────────────────


def _repair_individual(
    x: np.ndarray, y: np.ndarray, data: dict
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    对单个个体 (x_i, y_i) 执行约束修复。
    返回修复后的 (x, y, feasible)。
    """
    F0 = data["F0_i"]
    S0 = data["S0_i"]
    N0 = data["N0_i"]
    A_i = data["A_i"]
    A0 = data["A0_i"]
    a = data["a_i"]
    theta = data["theta_i"]
    n100 = data["n100_i"]
    n_lb = data["n_lb_i"]
    Q_plan = data["Q_plan"]

    x = np.clip(np.round(x), 0, None).astype(int)
    y = np.clip(np.round(y), 0, None).astype(int)

    for _ in range(5):  # 最多修复5轮
        changed = False

        for i in range(N_REGIONS):
            # 1) 总数上界
            total = x[i] + y[i]
            if total > n100[i]:
                # 按较大基因优先削减
                if x[i] >= y[i]:
                    x[i] = min(x[i], n100[i])
                    y[i] = n100[i] - x[i]
                else:
                    y[i] = min(y[i], n100[i])
                    x[i] = n100[i] - y[i]
                changed = True

            # 2) 覆盖率下界
            need_cover = n_lb[i]
            if x[i] + y[i] < need_cover:
                deficit = need_cover - (x[i] + y[i])
                # 优先补慢充（成本低）
                y[i] += deficit
                changed = True

            # 3) 服务能力约束
            cap = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i])
            if cap < Q_plan[i]:
                deficit = Q_plan[i] - cap
                # 优先补快充（服务能力效率高）
                n_fast = int(np.ceil(deficit / S_F))
                x[i] += n_fast
                changed = True

            # 4) 快充结构约束: S0_i * x_i >= F0_i * y_i
            if S0[i] * x[i] < F0[i] * y[i]:
                # 需要增加 x 或减少 y
                # S0*x >= F0*y  =>  x >= F0*y/S0
                if S0[i] > 0:
                    x_min = int(np.ceil(F0[i] * y[i] / S0[i]))
                    if x_min > x[i]:
                        x[i] = x_min
                        changed = True
                else:
                    # S0=0: 必须 y=0
                    if y[i] > 0:
                        y[i] = 0
                        changed = True

            # 5) 再次检查总数上界
            total = x[i] + y[i]
            if total > n100[i]:
                # 优先削减慢充
                excess = total - n100[i]
                cut_y = min(y[i], excess)
                y[i] -= cut_y
                excess -= cut_y
                x[i] -= excess
                changed = True

        if not changed:
            break

    # 最终检查可行性
    feasible = True
    for i in range(N_REGIONS):
        if x[i] + y[i] < n_lb[i]:
            feasible = False
            break
        cap = S_F * (F0[i] + x[i]) + S_S * (S0[i] + y[i])
        if cap < Q_plan[i]:
            feasible = False
            break
        if S0[i] * x[i] < F0[i] * y[i]:
            feasible = False
            break
        if x[i] + y[i] > n100[i]:
            feasible = False
            break
        if x[i] < 0 or y[i] < 0:
            feasible = False
            break

    return x, y, feasible


class RepairChromosomes(Repair):
    """pymoo 修复算子：对每个个体执行约束修复。"""

    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def _do(self, problem, X, **kwargs):
        n = X.shape[0]
        X_new = X.copy()
        for k in range(n):
            x_vals = X_new[k, :N_REGIONS]
            y_vals = X_new[k, N_REGIONS:]
            x_r, y_r, _ = _repair_individual(x_vals, y_vals, self.data)
            X_new[k, :N_REGIONS] = x_r
            X_new[k, N_REGIONS:] = y_r
        return X_new


# ─────────────────────────── 目标函数 ───────────────────────────


def compute_objectives(
    x: np.ndarray, y: np.ndarray, data: dict, model: Literal["MA", "MB"]
) -> dict:
    """
    计算所有目标与辅助指标。
    返回字典：f1, f2_neg, f3, nubar, nu_max, nu_std, nu_cv, nu_sqsum, nu_var, C_city
    """
    A_i = data["A_i"]
    A0 = data["A0_i"]
    a = data["a_i"]
    G = data["G_i"]

    # 目标1：总成本
    f1 = float(np.sum(C_F * x + C_S * y))

    # 目标2：全市面积加权覆盖率（取负用于最小化）
    C_i = np.minimum(1.0, (A0 + a * (x + y)) / A_i)
    C_city = float(np.sum(A_i * C_i) / np.sum(A_i))
    f2_neg = -C_city

    # 目标3：新增接入压力率
    nu = (P_F * x + P_S * y) / G
    nubar = float(np.mean(nu))
    nu_max = float(np.max(nu))
    nu_std = float(np.std(nu))
    nu_cv = nu_std / nubar if nubar > 1e-12 else 0.0
    nu_sqsum = float(np.sum(nu ** 2))
    nu_var = float(np.var(nu))

    if model == "MA":
        f3 = nu_sqsum
    else:
        f3 = nu_var

    return {
        "f1": f1,
        "f2_neg": f2_neg,
        "f3": f3,
        "C_city": C_city,
        "nubar": nubar,
        "nu_max": nu_max,
        "nu_std": nu_std,
        "nu_cv": nu_cv,
        "nu_sqsum": nu_sqsum,
        "nu_var": nu_var,
        "nu": nu,
    }


# ─────────────────────────── pymoo Problem ───────────────────────────


class ChargingPlanningProblem(Problem):
    """20 维整数多目标优化问题。"""

    def __init__(self, data: dict, model: Literal["MA", "MB"]):
        self.pdata = data
        self.ptype = model
        n100 = data["n100_i"]
        xl = np.zeros(2 * N_REGIONS)
        xu = np.tile(n100, 2).astype(float)
        super().__init__(
            n_var=2 * N_REGIONS,
            n_obj=3,
            n_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.zeros((n, 3))
        for k in range(n):
            x = X[k, :N_REGIONS]
            y = X[k, N_REGIONS:]
            obj = compute_objectives(x, y, self.pdata, self.ptype)
            F[k, 0] = obj["f1"]
            F[k, 1] = obj["f2_neg"]
            F[k, 2] = obj["f3"]
        out["F"] = F


# ─────────────────────────── 运行 NSGA-II ───────────────────────────


def run_nsga2(
    data: dict,
    model: Literal["MA", "MB"],
    seed: int,
    pop_size: int = POP_SIZE,
    n_gen: int = N_GEN,
) -> dict:
    """
    运行一次 NSGA-II，返回 Pareto 解集及评估结果。
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
        )

    # 收集 Pareto 解
    X_opt = res.X  # (n_pop, 20) 或 None
    if X_opt is None or len(X_opt) == 0:
        return {"solutions": [], "seed": seed, "model": model, "n_gen": res.algorithm.n_gen}

    solutions = []
    for k in range(X_opt.shape[0]):
        x = np.clip(np.round(X_opt[k, :N_REGIONS]), 0, None).astype(int)
        y = np.clip(np.round(X_opt[k, N_REGIONS:]), 0, None).astype(int)

        # 修复四舍五入可能引入的微小违反
        x, y, feasible = _repair_individual(x, y, data)
        if not feasible:
            continue

        obj = compute_objectives(x, y, data, model)
        solutions.append({
            "x": x.copy(),
            "y": y.copy(),
            **obj,
        })

    return {
        "solutions": solutions,
        "seed": seed,
        "model": model,
        "n_gen": int(res.algorithm.n_gen) if hasattr(res.algorithm, "n_gen") else n_gen,
    }


def run_model_seeds(
    data: dict,
    model: Literal["MA", "MB"],
    seeds: list[int] | None = None,
) -> list[dict]:
    """对一个模型运行所有随机种子。"""
    if seeds is None:
        from problem2_utils import SEEDS
        seeds = SEEDS
    results = []
    for s in seeds:
        print(f"  [NSGA-II] model={model}, seed={s} ...", flush=True)
        r = run_nsga2(data, model, s)
        print(f"    -> {len(r['solutions'])} feasible solutions")
        results.append(r)
    return results


def merge_and_filter_solutions(results: list[dict], model: Literal["MA", "MB"]) -> list[dict]:
    """
    合并多次运行结果：去重 + 再次非支配筛选。
    """
    all_sols = []
    for r in results:
        all_sols.extend(r["solutions"])

    if not all_sols:
        return []

    # 去重（基于 x, y 的整数配置）
    seen = set()
    unique = []
    for s in all_sols:
        key = tuple(np.concatenate([s["x"], s["y"]]))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    # 非支配筛选
    pareto = _nondominated_filter(unique, model)
    return pareto


def _nondominated_filter(solutions: list[dict], model: Literal["MA", "MB"]) -> list[dict]:
    """对方案列表进行非支配筛选。"""
    n = len(solutions)
    if n <= 1:
        return solutions

    # 构造目标矩阵 (f1, -C_city, f3) 均最小化
    F = np.zeros((n, 3))
    for k, s in enumerate(solutions):
        F[k, 0] = s["f1"]
        F[k, 1] = s["f2_neg"]
        F[k, 2] = s["f3"]

    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(i + 1, n):
            if dominated[j]:
                continue
            if _dominates(F[i], F[j]):
                dominated[j] = True
            elif _dominates(F[j], F[i]):
                dominated[i] = True
                break

    return [solutions[k] for k in range(n) if not dominated[k]]


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a 是否支配 b（所有目标 <= 且至少一个 <）。"""
    return np.all(a <= b) and np.any(a < b)
