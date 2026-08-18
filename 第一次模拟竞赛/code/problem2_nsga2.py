# -*- coding: utf-8 -*-
"""
问题二 NSGA-II 求解与精确复核。

约束处理采用 Deb 可行性优先（pymoo 默认 CV 比较），
不使用超大罚函数。交叉变异后修复为非负整数。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.optimize import Bounds, LinearConstraint, milp

from problem2_utils import (
    C_F,
    C_S,
    EPS,
    N_REGION,
    P_F,
    P_S,
    Q_F,
    Q_S,
    Problem2Data,
    Scenario,
    enumerate_region_pairs,
    evaluate_batch,
    grid_add_cap,
    objectives,
    useful_bounds,
    zmin_cover,
    zsat_cover,
)


class ChargingProblem(Problem):
    """20 维非负整数编码：(x_1..x_10, y_1..y_10)。"""

    def __init__(self, data: Problem2Data, scen: Scenario):
        self.bundle = data
        self.scen = scen
        self.x_max, self.y_max = useful_bounds(data, scen)
        xl = np.zeros(2 * N_REGION)
        xu = np.concatenate([self.x_max, self.y_max]).astype(float)
        super().__init__(
            n_var=2 * N_REGION,
            n_obj=3,
            n_ieq_constr=4,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        zsat = zsat_cover(data, scen.beta)
        self.feasible_pairs = [
            enumerate_region_pairs(data, i, scen, z_hi=int(zsat[i]))
            for i in range(N_REGION)
        ]

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = evaluate_batch(self.bundle, X, self.scen)
        out["F"] = F
        out["G"] = G


class IntegerClipRepair(Repair):
    """交叉变异后取整并裁剪到变量边界。"""

    def _do(self, problem, X, **kwargs):
        X = np.rint(np.asarray(X, dtype=float))
        X = np.minimum(np.maximum(X, problem.xl), problem.xu)
        return X.astype(int)


class FeasiblePairSampling(Sampling):
    """按区域从整数可行对中独立抽样；无可行对时退回 (0, 0)。"""

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for k in range(n_samples):
            for i in range(N_REGION):
                pairs = problem.feasible_pairs[i]
                if not pairs:
                    x, y = 0, 0
                elif k == 0:
                    x, y = min(pairs, key=lambda xy: C_F * xy[0] + C_S * xy[1])
                elif k == 1:
                    x, y = max(pairs, key=lambda xy: (xy[0] + xy[1], -xy[0]))
                elif k == 2:
                    x, y = min(pairs, key=lambda xy: (P_F * xy[0] + P_S * xy[1], C_F * xy[0] + C_S * xy[1]))
                else:
                    x, y = pairs[int(np.random.randint(0, len(pairs)))]
                X[k, i] = x
                X[k, N_REGION + i] = y
        return X


class MetricsCallback(Callback):
    """记录每代可行比例、非支配规模、HV 与目标极值。"""

    def __init__(self, ref_point: np.ndarray):
        super().__init__()
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.hv_ind = HV(ref_point=self.ref_point)
        self.rows: list[dict] = []

    def notify(self, algorithm):
        pop = algorithm.pop
        F = np.asarray(pop.get("F"), dtype=float)
        G = np.asarray(pop.get("G"), dtype=float)
        X = np.asarray(pop.get("X"), dtype=int)
        cv = np.maximum(G, 0.0).sum(axis=1)
        feas = cv <= 1e-12
        n_feas = int(feas.sum())
        feas_ratio = n_feas / max(len(pop), 1)
        uniq = len({tuple(row.tolist()) for row in X})
        dup_ratio = 1.0 - uniq / max(len(pop), 1)
        if n_feas > 0:
            Ff = F[feas]
            nd = _nondominated_mask(Ff)
            n_nd = int(nd.sum())
            try:
                hv = float(self.hv_ind(Ff[nd]))
            except Exception:
                hv = 0.0
            fmin = Ff.min(axis=0)
            fmax = Ff.max(axis=0)
        else:
            n_nd = 0
            hv = 0.0
            fmin = np.full(3, np.nan)
            fmax = np.full(3, np.nan)
        self.rows.append({
            "gen": int(algorithm.n_iter),
            "feas_ratio": feas_ratio,
            "n_nd": n_nd,
            "hv": hv,
            "dup_ratio": dup_ratio,
            "F1_min": float(fmin[0]) if np.isfinite(fmin[0]) else np.nan,
            "F1_max": float(fmax[0]) if np.isfinite(fmax[0]) else np.nan,
            "F2_max": float(-fmin[1]) if np.isfinite(fmin[1]) else np.nan,
            "F2_min": float(-fmax[1]) if np.isfinite(fmax[1]) else np.nan,
            "F3_min": float(fmin[2]) if np.isfinite(fmin[2]) else np.nan,
            "F3_max": float(fmax[2]) if np.isfinite(fmax[2]) else np.nan,
        })


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    """最小化问题的非支配掩码。"""
    n = F.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        better = np.all(F <= F[i] + 1e-12, axis=1) & np.any(F < F[i] - 1e-12, axis=1)
        if np.any(better):
            nd[i] = False
    return nd


def default_ref_point(data: Problem2Data, scen: Scenario) -> np.ndarray:
    """固定参考点，保证不同随机种子的 HV 可比较。"""
    x_max, y_max = useful_bounds(data, scen)
    f1_hi = float((C_F * x_max + C_S * y_max).sum()) * 1.1 + 10.0
    f2_lo = 0.80
    f3_hi = 1.05
    return np.array([f1_hi, -f2_lo, f3_hi], dtype=float)


@dataclass
class NSGAResult:
    X: np.ndarray
    F: np.ndarray
    history: list[dict]
    runtime_s: float
    seed: int
    n_eval: int
    feasible: bool
    note: str = ""
    extra: dict = field(default_factory=dict)


def solve_nsga2(
    data: Problem2Data,
    scen: Scenario,
    pop_size: int = 300,
    n_gen: int = 500,
    seed: int = 42,
    ref_point: np.ndarray | None = None,
    verbose: bool = False,
) -> NSGAResult:
    problem = ChargingProblem(data, scen)
    n_empty = sum(1 for p in problem.feasible_pairs if len(p) == 0)
    if n_empty > 0:
        empty = [int(data.region[i]) for i, p in enumerate(problem.feasible_pairs) if len(p) == 0]
        return NSGAResult(
            X=np.zeros((0, 20), dtype=int),
            F=np.zeros((0, 3)),
            history=[],
            runtime_s=0.0,
            seed=seed,
            n_eval=0,
            feasible=False,
            note=f"预检无可行整数解，区域 {empty}，未启动 NSGA-II",
        )

    if ref_point is None:
        ref_point = default_ref_point(data, scen)
    cb = MetricsCallback(ref_point)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeasiblePairSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=IntegerClipRepair()),
        mutation=PM(prob=1.0 / 20.0, eta=20, vtype=float, repair=IntegerClipRepair()),
        eliminate_duplicates=True,
    )
    t0 = time.perf_counter()
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        callback=cb,
        verbose=verbose,
        copy_algorithm=False,
    )
    runtime = time.perf_counter() - t0
    if res.X is None:
        return NSGAResult(
            X=np.zeros((0, 20), dtype=int),
            F=np.zeros((0, 3)),
            history=cb.rows,
            runtime_s=runtime,
            seed=seed,
            n_eval=int(getattr(getattr(res, "algorithm", None), "evaluator", type("E", (), {"n_eval": 0})()).n_eval),
            feasible=False,
            note="NSGA-II 未返回解集",
            extra={"ref_point": ref_point.tolist()},
        )
    X = np.rint(np.asarray(res.X, dtype=float)).astype(int)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    F = np.asarray(res.F, dtype=float)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    G = np.asarray(res.G if res.G is not None else np.zeros((len(X), 4)), dtype=float)
    if G.ndim == 1:
        G = G.reshape(1, -1)
    feas = np.maximum(G, 0.0).sum(axis=1) <= 1e-10
    X, F = X[feas], F[feas]
    X, F = deduplicate_front(X, F)
    return NSGAResult(
        X=X,
        F=F,
        history=cb.rows,
        runtime_s=runtime,
        seed=seed,
        n_eval=int(res.algorithm.evaluator.n_eval) if res.algorithm is not None else pop_size * n_gen,
        feasible=len(X) > 0,
        note="" if len(X) > 0 else "NSGA-II 结束时无可行非支配解",
        extra={"ref_point": ref_point.tolist()},
    )


def deduplicate_front(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """决策向量完全一致时只保留一条。"""
    if len(X) == 0:
        return X, F
    seen = {}
    keep = []
    for i, row in enumerate(X):
        key = tuple(int(v) for v in row.tolist())
        if key in seen:
            continue
        seen[key] = i
        keep.append(i)
    keep = np.asarray(keep, dtype=int)
    Xk, Fk = X[keep], F[keep]
    nd = _nondominated_mask(Fk)
    return Xk[nd], Fk[nd]


def merge_fronts(results: list[NSGAResult]) -> tuple[np.ndarray, np.ndarray]:
    xs = [r.X for r in results if len(r.X)]
    fs = [r.F for r in results if len(r.F)]
    if not xs:
        return np.zeros((0, 20), dtype=int), np.zeros((0, 3))
    return deduplicate_front(np.vstack(xs), np.vstack(fs))


def front_to_dataframe(data: Problem2Data, scen: Scenario, X: np.ndarray, F: np.ndarray):
    import pandas as pd

    rows = []
    for k in range(len(X)):
        x, y = X[k, :N_REGION], X[k, N_REGION:]
        f1, f2, f3, S, rho = objectives(data, x, y, scen.beta, scen.alpha)
        rows.append({
            "方案编号": k + 1,
            "F1_投资成本": f1,
            "F2_需求加权覆盖率": f2,
            "F3_电网目标": f3,
            "新增快充合计": int(x.sum()),
            "新增慢充合计": int(y.sum()),
            "新增总桩数": int(x.sum() + y.sum()),
            "最低区域覆盖率": float(S.min()),
            "最大区域压力": float(rho.max()),
            "压力方差": float(np.mean((rho - rho.mean()) ** 2)),
            "快充是否全零": bool(np.all(x == 0)),
            **{f"x{i+1}": int(x[i]) for i in range(N_REGION)},
            **{f"y{i+1}": int(y[i]) for i in range(N_REGION)},
        })
    return pd.DataFrame(rows)


def milp_min_cost(data: Problem2Data, scen: Scenario) -> dict:
    """最低成本端点的整数线性规划精确求解（HiGHS）。"""
    _, padd, _ = grid_add_cap(data, scen.tau)
    zmin = zmin_cover(data, scen.beta, scen.cover_req)
    x_max, y_max = useful_bounds(data, scen)
    c = np.concatenate([np.full(N_REGION, C_F), np.full(N_REGION, C_S)])
    lb = np.zeros(2 * N_REGION)
    ub = np.concatenate([x_max.astype(float), y_max.astype(float)])
    A_rows = []
    b_lb = []
    b_ub = []
    for i in range(N_REGION):
        row_z = np.zeros(2 * N_REGION)
        row_z[i] = 1.0
        row_z[N_REGION + i] = 1.0
        A_rows.append(row_z)
        b_lb.append(float(zmin[i]))
        b_ub.append(np.inf)

        row_svc = np.zeros(2 * N_REGION)
        row_svc[i] = Q_F
        row_svc[N_REGION + i] = Q_S
        A_rows.append(row_svc)
        b_lb.append(float(data.Dhat[i] - data.K0[i]))
        b_ub.append(np.inf)

        row_g = np.zeros(2 * N_REGION)
        row_g[i] = P_F
        row_g[N_REGION + i] = P_S
        A_rows.append(row_g)
        b_lb.append(-np.inf)
        b_ub.append(float(padd[i]))

        if scen.use_mix and scen.mix_enable[i]:
            dlt = float(scen.delta_vec()[i])
            rlo = data.r0[i] - dlt
            rhi = data.r0[i] + dlt
            # (Nf+x) - rlo*(N+x+y) >= 0
            row_lo = np.zeros(2 * N_REGION)
            row_lo[i] = 1.0 - rlo
            row_lo[N_REGION + i] = -rlo
            A_rows.append(row_lo)
            b_lb.append(float(rlo * data.Ntot[i] - data.Nf[i]))
            b_ub.append(np.inf)
            # rhi*(N+x+y) - (Nf+x) >= 0
            row_hi = np.zeros(2 * N_REGION)
            row_hi[i] = rhi - 1.0
            row_hi[N_REGION + i] = rhi
            A_rows.append(row_hi)
            b_lb.append(float(data.Nf[i] - rhi * data.Ntot[i]))
            b_ub.append(np.inf)

    A = np.vstack(A_rows)
    cons = LinearConstraint(A, np.array(b_lb), np.array(b_ub))
    res = milp(
        c,
        integrality=np.ones(2 * N_REGION),
        bounds=Bounds(lb, ub),
        constraints=cons,
        options={"time_limit": 30.0},
    )
    out = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "cost": None,
        "x": None,
        "y": None,
    }
    if res.success and res.x is not None:
        z = np.rint(res.x).astype(int)
        out["x"] = z[:N_REGION]
        out["y"] = z[N_REGION:]
        out["cost"] = float((C_F * out["x"] + C_S * out["y"]).sum())
    return out


def neighborhood_check(
    data: Problem2Data,
    scen: Scenario,
    x0: np.ndarray,
    y0: np.ndarray,
) -> dict:
    """
    推荐方案邻域复核：
      1) 逐区域 3×3 局部扰动；
      2) 全市同步 (±1,0)/(0,±1) 四方向。
    检查是否存在可行且支配当前方案的邻居。
    """
    from problem2_utils import constraint_violations, total_cv

    x0 = np.asarray(x0, dtype=int).copy()
    y0 = np.asarray(y0, dtype=int).copy()
    f1_0, f2_0, f3_0, _, _ = objectives(data, x0, y0, scen.beta, scen.alpha)
    F0 = np.array([f1_0, -f2_0, f3_0])

    dominating = []
    n_checked = 0
    n_feas = 0

    def consider(x, y, tag):
        nonlocal n_checked, n_feas
        if np.any(x < 0) or np.any(y < 0):
            return
        n_checked += 1
        viol = constraint_violations(data, x, y, scen)
        if total_cv(viol) > 1e-10:
            return
        n_feas += 1
        f1, f2, f3, _, _ = objectives(data, x, y, scen.beta, scen.alpha)
        F = np.array([f1, -f2, f3])
        if np.all(F <= F0 + 1e-9) and np.any(F < F0 - 1e-9):
            dominating.append({
                "tag": tag,
                "F1": f1,
                "F2": f2,
                "F3": f3,
                "x": x.tolist(),
                "y": y.tolist(),
            })

    for i in range(N_REGION):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                x = x0.copy()
                y = y0.copy()
                x[i] += dx
                y[i] += dy
                consider(x, y, f"region{i+1}_dx{dx}_dy{dy}")

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        consider(x0 + dx, y0 + dy, f"all_dx{dx}_dy{dy}")

    return {
        "n_checked": n_checked,
        "n_feasible_neighbors": n_feas,
        "n_dominating": len(dominating),
        "dominating": dominating[:20],
    }


def hv_of_front(F: np.ndarray, ref_point: np.ndarray) -> float:
    if len(F) == 0:
        return 0.0
    try:
        return float(HV(ref_point=np.asarray(ref_point, dtype=float))(F))
    except Exception:
        return 0.0


def igd_of_front(F: np.ndarray, ref_front: np.ndarray) -> float:
    if len(F) == 0 or len(ref_front) == 0:
        return float("nan")
    try:
        return float(IGD(ref_front)(F))
    except Exception:
        return float("nan")


def last50_hv_rel_change(history: list[dict]) -> float:
    if len(history) < 51:
        return float("nan")
    hv = np.array([h["hv"] for h in history], dtype=float)
    a, b = hv[-51], hv[-1]
    if abs(a) < 1e-12:
        return float("nan") if abs(b) < 1e-12 else float("inf")
    return abs(b - a) / abs(a)
