# -*- coding: utf-8 -*-
"""
问题3 · Step 2：低谷负荷分配优化（两阶段求解）与可行性预检
==========================================================

模型说明（对区域 i、日期类型 d∈{wd,we}，共 10×2=20 个独立子问题）
------------------------------------------------------------------
决策变量：
    z_k ≥ 0 (k=0..6) ：高峰转移至低谷时段 V[k] 的增量负荷（kW；Δt=1h，数值=kWh）
    U ：调度后全天最大负荷上界；V ：下限（仅用于线性化峰谷差目标/约束）

调度后负荷（固定规则 + 决策部分）：
    L̃_t = 0.8·L_t        t ∈ H（高峰，等比例削减 20%）
    L̃_t = L_t + z_k      t = V[k] ∈ V（低谷，接收待优化转移量）
    L̃_t = L_t            t ∈ M ∪ {23}（平段及非调度时段，保持原负荷）
全天电量自动守恒：Σ L̃ = Σ L（0.8·峰值部分被削下的 20% 恰好等于 Σ z）。

第一阶段 LP（min U − V），变量 x=(z_1..z_7, U, V)，scipy.linprog(method='highs')：
    min        c·x,  c = (0,…,0,+1,−1)
    s.t.       −(L_t + z_k) …… 逐时上下界（线性）：
                 t∉V : −U ≤ −Lbase_t           （即 U ≥ Lbase_t）
                       V ≤ Lbase_t
                 t=V[k]:  z_k − U ≤ −L_t       （即 U ≥ L_t+z_k）
                          V − z_k ≤ L_t        （即 V ≤ L_t+z_k，注意符号）
                         注：上行正确形式为 V ≤ L_t + z_k ⇔ (−1)·z_k? 见下代码注释，以数值核验兜底。
               Σ_{k} z_k = M_used（等式）
               0 ≤ z_k ≤ cap_k, cap_k = max(G_{V[k]} − L_{V[k]}, 0)   # 隐含低谷安全约束

第二阶段 QP（次目标：最小化全天方差），变量同上，scipy.minimize(method='SLSQP')：
    min      (1/24) · Σ_t (L̃_t(z) − L̄)^2 ，L̄ = mean(L)（守恒 ⇒ 调度后同均值）
    s.t.     U − V ≤ Δ* + 1e−6      （不恶化第一阶段最优峰谷差；线性，Jacobian 光滑）
             第一阶段全部约束
优势：因 L̃_t 对 z、U、V 均为仿射函数，二阶段全程无 max/min 非光滑项。

可行性预检（任务单 §7.2 / §7.3）
--------------------------------
    B = Σ_{k} cap_k；若 B < M − tol → “低谷接纳不足”，取 M_used = B（不突破电网上限、
    不擅自转平段），r_actual = M_used / Σ_L(H) × 100%。

输出：results/p3_solve/subproblem_results.json（20 个子问题的完整求解记录）
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "p3_data"
OUT_DIR = ROOT / "results" / "p3_solve"

TOL_FEA = 1e-6      # 可行性容差（§7.2）
TOL_EQ = 1e-6       # 正确性核验容差（§10.1）


@dataclass
class SubResult:
    region: int                       # 区域编号 i
    date_type: str                    # 'wd' | 'we'
    peak_total_kW: float = 0.0        # Σ_{t∈H} L_t（高峰负荷总量，kW口径）
    M_kW: float = 0.0                 # 规定转移量 0.2·Σ_H L
    B_kW: float = 0.0                 # 低谷可接纳量 Σ_V max(G−L,0)
    M_used_kW: float = 0.0            # 实际采用转移量（可行=M，否则=B）
    feasible_20pct: bool = True       # B ≥ M ?
    infeas_reason: str = ""           # 异常原因说明
    r_actual_pct: float = 0.0         # 实际转移比例 % (M_used / Σ_H L)
    delta_star_kW: float = 0.0        # 第一阶段最优峰谷差 U*−V*
    stage2_enabled: bool = False      # 二阶段次目标是否生效
    z_valley: list[float] = field(default_factory=list)   # 7 维低谷分配向量（kW口径）
    status_lp: str = ""               # HiGHS 状态
    status_qp: str = ""               # SLSQP 状态（启用时）
    lp_obj: float = 0.0               # LP 最优目标值 (=Δ*)
    qp_obj_kW2: float = 0.0           # QP 最优方差 (kW^2)，等效 /24 口径见报告
    solve_time_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---- 模块级时段上下文（main() 载入 time_sets.json 后填充）----
TS: dict = {}
VH: list[int] = []          # 谷段小时列表（7）
HH: list[int] = []          # 峰段小时列表（10）
RATIO_PEAK: float = 0.2


def post_load(L: np.ndarray, z: np.ndarray) -> np.ndarray:
    """按调度规则构造调度后 24h 负荷曲线。L:(24,) 调度前；z:(7,) 谷段分配。"""
    Lp = L.copy()
    Lp[HH] *= (1.0 - RATIO_PEAK)                 # t∈H：削减 20%
    for k, t in enumerate(VH):                   # t∈V：叠加增量
        Lp[t] += z[k]
    return Lp                                    # t∈M∪{23}：保持原值


def solve_one(region: int, d_type: str, L: np.ndarray, G: np.ndarray) -> SubResult:
    """求解单个 (i,d) 子问题。L:(24,) kW；G:(24,) kW。返回完整记录。"""
    res = SubResult(region=region, date_type=d_type)
    t0 = time.perf_counter()
    nV = len(VH)

    # ---------------- A. 转移量与接纳能力（§5.1 / §7.2）----------------
    peak_sum = float(L[HH].sum())                # Σ_{t∈H} L_t
    res.peak_total_kW = round(peak_sum, 6)
    M = RATIO_PEAK * peak_sum                    # 规定转移量 kW
    cap = np.maximum(G[np.array(VH)] - L[np.array(VH)], 0.0)   # 谷段逐时接纳能力
    B = float(cap.sum())
    res.M_kW, res.B_kW = round(M, 6), round(B, 6)
    res.feasible_20pct = (B + TOL_FEA >= M)
    if not res.feasible_20pct:
        res.infeas_reason = "低谷接纳不足（任务单§7.3：不擅自转平段，取 B 为可行转移量）"
    res.M_used_kW = round(B, 6) if not res.feasible_20pct else M
    # r_actual_pct 在 D 节统一计算

    # ---------------- B. 第一阶段 LP: min U−V ----------------
    nvar = nV + 2                                # x = (z_0..z_6, U, V)
    IU, IV = nV, nV + 1
    c = np.zeros(nvar); c[IU], c[IV] = 1.0, -1.0

    def zidx(t: int) -> int:                     # 小时 t 对应的谷段 k（非谷段返回 −1）
        return VH.index(t) if t in VH else -1

    A_rows, b_rows = [], []
    for t in range(24):                          # 逐时 U/V 包络约束：V ≤ L̃_t ≤ U
        k = zidx(t)
        Lt = float(L[t])
        if k < 0:                                # t∉V：L̃_t = L_t（峰段 ×0.8、平/23h 不变）
            Lb = (1.0 - RATIO_PEAK) * Lt if t in HH else Lt
            A_rows.append(np.array([*[0.0] * nV, -1.0, 0.0]))
            b_rows.append(-Lb)                    # −U ≤ −Lb ⇔ U ≥ Lb
            A_rows.append(np.array([*[0.0]*nV, 0.0, 1.0]))
            b_rows.append(Lb)                    # V ≤ Lb
        else:                                    # t = VH[k]：L̃_t = L_t + z_k
            rA = np.full(nvar, 0.0); rA[k], rA[IU] = 1.0, -1.0
            A_rows.append(rA); b_rows.append(-Lt)             # U ≥ L_t+z_k ⇔ z_k−U ≤ −L_t
            rB = np.full(nvar, 0.0); rB[k], rB[IV] = -1.0, 1.0
            A_rows.append(rB); b_rows.append(Lt)              # V ≤ L_t+z_k ⇔ V−z_k ≤ L_t
    A_ub = np.array(A_rows, dtype=float); b_ub = np.array(b_rows, dtype=float)

    Ae = np.zeros((1, nvar)); Ae[0, :nV] = 1.0; be = np.array([res.M_used_kW])
    bounds = [(0.0, float(cap[k])) for k in range(nV)] + [(-np.inf, np.inf), (-np.inf, np.inf)]

    lp = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=Ae, b_eq=be, bounds=bounds, method="highs")
    res.status_lp = "Optimal" if lp.status == 0 else f"{lp.status}:{str(lp.message)[:90]}"
    res.lp_obj = float(lp.fun) if lp.status == 0 else float("nan")

    z_sol = np.zeros(nV)
    delta_star = float(lp.x[IU] - lp.x[IV]) if lp.status == 0 else float("nan")
    res.delta_star_kW = round(delta_star, 6)
    if lp.status == 0:
        z_sol = np.clip(lp.x[:nV], 0.0, cap)

        # ---------------- C. 第二阶段 QP（次目标：最小化方差，SLSQP）----------------
        Lbar = float(L.mean())                   # 守恒 ⇒ 调度后均值不变
        var0 = float(np.mean((post_load(L, z_sol) - Lbar) ** 2))

        def qp_obj(v):                           # v=(z,U,V)；目标仅依赖 z
            zz = v[:nV]
            Lp = post_load(L, zz)
            return float(np.mean((Lp - Lbar) ** 2))

        con_gap = {"type": "ineq",                                            # U−V ≤ Δ*+tol ⇒ f(v)=Δ*+tol−(U−V)≥0
                   "fun": lambda v: delta_star + TOL_FEA - (v[IU] - v[IV]),
                   "jac": lambda v: np.array([*[0.0]*nV, -1.0, 1.0])}


        x0 = np.r_[z_sol, lp.x[IU], lp.x[IV]]
        sol = minimize(qp_obj, x0, jac=None, method="SLSQP",
                       bounds=bounds, constraints=[con_gap] + [
                           {"type": "eq", "fun": lambda v: float(v[:nV].sum() - res.M_used_kW),
                            "jac": lambda v: np.r_[np.ones(nV), 0.0, 0.0]}],
                       options={"maxiter": 500, "ftol": 1e-14})

        if sol.success:
            zq = np.clip(sol.x[:nV], 0.0, cap)
            Lp_q = post_load(L, zq)                         # 峰值差核验（防 SLSQP 残差漂出）
            gap_ok = (Lp_q.max() - Lp_q.min()) <= delta_star + 1e-5
            mass_ok = abs(zq.sum() - res.M_used_kW) < max(1e-6, 1e-9 * M) and zq.max() < cap.max() + 1e-8
            if gap_ok and mass_ok:
                # 残差均摊保证等式精确；再裁回上界内（上界通常很松，均摊量极小）
                zq = zq + (res.M_used_kW - zq.sum()) / nV
                z_sol = np.clip(zq, 0.0, cap)
                if abs(delta_star - float((Lp_q.max() - Lp_q.min()))) <= 1e-4:
                    res.stage2_enabled = True
        else:
            res.status_qp = f"fallback_LP:{str(sol.message)[:60]}"

    # ---------------- D. 结果向量与核验 ----------------
    if abs(z_sol.sum() - res.M_used_kW) > 1e-9:             # 兜底：线性均分修正残差（不超容差时几乎不动）
        z_sol = z_sol + (res.M_used_kW - z_sol.sum()) / nV
    z_final = np.clip(z_sol, 0.0, None)
    res.z_valley = [round(float(v), 9) for v in z_final]   # 保留9位小数，使量化误差(≤7×5e−10)远小于 §10.1 的 1e−6 容差

    Lp = post_load(L, z_final)
    assert abs(Lp.sum() - L.sum()) < TOL_EQ, "电量守恒破坏（解内断言）"
    res.qp_obj_kW2 = round(float(np.mean((Lp - float(L.mean())) ** 2)), 6)
    res.r_actual_pct = round(res.M_used_kW / peak_sum * 100.0, 6) if peak_sum > 0 else 0.0

    res.solve_time_s = round(time.perf_counter() - t0, 4)
    return res


def main() -> list[dict]:
    """运行全部 10×2=20 个子问题并输出汇总 JSON。"""
    global TS, VH, HH, RATIO_PEAK

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "time_sets.json", encoding="utf-8") as f:
        TS = json.load(f)
    VH = list(TS["valley_hours"]); HH = list(TS["peak_hours"])
    RATIO_PEAK = float(TS["transfer_ratio_peak"])

    L_wd = pd.read_csv(DATA_DIR / "L_pre_wd.csv").drop(columns=["区域"]).to_numpy(dtype=float)   # (10,24)
    L_we = pd.read_csv(DATA_DIR / "L_pre_we.csv").drop(columns=["区域"]).to_numpy(dtype=float)
    Gm = pd.read_csv(DATA_DIR / "G_limit.csv").drop(columns=["区域"]).to_numpy(dtype=float)

    data = {"wd": L_wd, "we": L_we}
    results: list[SubResult] = []
    for i in range(10):
        for d_name in ("wd", "we"):
            r = solve_one(i + 1, d_name, data[d_name][i], Gm[i])
            results.append(r)
            print(f"[solve] region={r.region:2d} {d_name}: feasible={r.feasible_20pct!s:5} "
                  f"M={r.M_kW:9.2f} B={r.B_kW:9.2f} Δ*={r.delta_star_kW:8.2f}kW  "
                  f"stage2={'ON ' if r.stage2_enabled else 'off'}  [{r.status_lp}/{r.status_qp or '-'}] {r.solve_time_s}s")

    out = [asdict(r) for r in results]
    with open(OUT_DIR / "subproblem_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 20 个子问题完成 → {OUT_DIR / 'subproblem_results.json'}")
    return out


if __name__ == "__main__":
    main()
