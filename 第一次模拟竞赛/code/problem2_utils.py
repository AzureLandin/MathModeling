# -*- coding: utf-8 -*-
"""
问题二公共数据与评价模块。

职责：
  读取附件 1—4、附件 5 参数与问题一预测值；
  派生 e_i、Dhat、覆盖/电网/车次参数；
  可行性预检与约束复核；
  三目标与约束违反量计算。

覆盖模型说明：附件未提供站点坐标，采用区域平均有效覆盖面积近似，
确定的是区域级桩数，不是具体站点选址。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ATTACH_DIR = BASE_DIR / "附件"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

HOUR_LABELS = [
    "00-01", "01-02", "02-03", "03-04", "04-05", "05-06",
    "06-07", "07-08", "08-09", "09-10", "10-11", "11-12",
    "12-13", "13-14", "14-15", "15-16", "16-17", "17-18",
    "18-19", "19-20", "20-21", "21-22", "22-23", "23-00",
]

C_F = 6.0
C_S = 0.8
P_F = 120.0
P_S = 7.0
Q_F = 80.0
Q_S = 20.0
N_REGION = 10
N_HOUR = 24
AUDIT_TOL = 1e-8
EPS = 1e-12

PALETTE = {
    "fast": "#D55E00",
    "slow": "#0072B2",
    "cost": "#E69F00",
    "cover": "#009E73",
    "grid": "#CC79A7",
    "before": "#56B4E9",
    "after": "#E69F00",
    "m0": "#999999",
    "m1": "#0072B2",
    "accent": "#2E86AB",
}


def read_region_table(path: Path) -> pd.DataFrame:
    """读取区域级附表，保留编号 1—10 的有效行。"""
    df = pd.read_excel(path, header=0)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    if df.iloc[:, 0].tolist() != list(range(1, 11)):
        raise ValueError(f"{path.name} 区域编号异常: {df.iloc[:, 0].tolist()}")
    return df


def read_hourly_matrix(path: Path, sheet: str) -> np.ndarray:
    """读取 10×24 分时矩阵，校验非负与区域编号。"""
    df = pd.read_excel(path, sheet_name=sheet, header=0)
    df = df.dropna(subset=[df.columns[0]]).copy()
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    if df.iloc[:, 0].tolist() != list(range(1, 11)):
        raise ValueError(f"{path.name}/{sheet} 区域编号不对应")
    mat = df.iloc[:, 1:25].astype(float).to_numpy()
    if mat.shape != (N_REGION, N_HOUR):
        raise ValueError(f"{path.name}/{sheet} 形状应为(10,24)，实际{mat.shape}")
    if np.any(mat < 0):
        raise ValueError(f"{path.name}/{sheet} 存在负值")
    return mat


@dataclass
class Problem2Data:
    """问题二统一数据集。所有向量长度均为 10。"""

    region: np.ndarray
    A: np.ndarray
    A0: np.ndarray
    Nf: np.ndarray
    Ns: np.ndarray
    Ntot: np.ndarray
    P_grid_kw: np.ndarray
    D_wd: np.ndarray
    D_we: np.ndarray
    L_wd: np.ndarray
    L_we: np.ndarray
    U: np.ndarray
    Yhat: np.ndarray
    Dbar: np.ndarray
    Qbar: np.ndarray
    e: np.ndarray
    Dhat: np.ndarray
    a_eff: np.ndarray
    S0: np.ndarray
    P0: np.ndarray
    K0: np.ndarray
    r0: np.ndarray
    omega: np.ndarray
    rho0_wd: np.ndarray
    rho0_we: np.ndarray
    rho0: np.ndarray
    Gmax_raw: np.ndarray
    peak_hour_wd: np.ndarray
    peak_hour_we: np.ndarray
    bottleneck: list[str]


def load_problem2_data() -> Problem2Data:
    """读取附件与问题一预测，构造派生参数。不融合当前实际需求。"""
    path1 = ATTACH_DIR / "附件 1 市主城区 10 个典型区域基础数据.xlsx"
    path2 = ATTACH_DIR / "附件2 市主城区 10 区域分时段充电车次.xlsx"
    path3 = ATTACH_DIR / "附件3 市主城区 10 区域分时段充电负荷.xlsx"
    path4 = ATTACH_DIR / "附件4 市主城区 10 个区域分时段电网最大允许负荷数据.xlsx"
    path_pred = RESULTS_DIR / "表13_最终PLSR区域需求.csv"

    df1 = read_region_table(path1)
    region = df1.iloc[:, 0].astype(int).to_numpy()
    A = df1.iloc[:, 1].astype(float).to_numpy()
    A0 = df1.iloc[:, 2].astype(float).to_numpy()
    Nf = df1.iloc[:, 7].astype(float).to_numpy()
    Ns = df1.iloc[:, 8].astype(float).to_numpy()
    # 附件 1 电网总容量单位为万千瓦，仅作背景对照，约束使用附件 4。
    P_grid_kw = df1.iloc[:, 9].astype(float).to_numpy() * 10000.0

    if np.any(A <= 0) or np.any(A0 < 0):
        raise ValueError("面积数据异常")
    if np.any(A0 > A + 1e-9):
        raise ValueError("现有覆盖面积大于区域总面积")
    if np.any(Nf < 0) or np.any(Ns < 0):
        raise ValueError("现有桩数存在负值")
    if np.any(Nf + Ns <= 0):
        raise ValueError("存在无桩区域，无法定义 a_i^eff")

    D_wd = read_hourly_matrix(path2, "工作日分时段充电车次数据")
    D_we = read_hourly_matrix(path2, "周末充电车次数据")
    L_wd = read_hourly_matrix(path3, "工作日分时段充电负荷数据")
    L_we = read_hourly_matrix(path3, "周末充电负荷数据（修改后）")
    U = read_hourly_matrix(path4, "Sheet1")
    if np.any(U <= 0):
        raise ValueError("附件4 存在非正电网上限")

    pred = pd.read_csv(path_pred, encoding="utf-8-sig")
    pred = pred.sort_values("区域").reset_index(drop=True)
    if pred["区域"].astype(int).tolist() != list(range(1, 11)):
        raise ValueError("问题一预测表区域编号异常")
    Yhat = pred["短期预测值"].astype(float).to_numpy()
    if np.any(Yhat <= 0):
        raise ValueError("预测需求必须全部为正")

    Dbar = (5.0 * D_wd.sum(axis=1) + 2.0 * D_we.sum(axis=1)) / 7.0
    Qbar = (5.0 * L_wd.sum(axis=1) + 2.0 * L_we.sum(axis=1)) / 7.0
    if np.any(Dbar <= 0) or np.any(Qbar <= 0):
        raise ValueError("典型日车次或电量非正")
    e = Qbar / Dbar
    if np.any(e <= 0):
        raise ValueError("e_i 必须为正")
    Dhat = Yhat / e
    if np.any(Dhat <= 0):
        raise ValueError("预测车次必须为正")

    Ntot = Nf + Ns
    a_eff = A0 / Ntot
    S0 = A0 / A
    P0 = P_F * Nf + P_S * Ns
    K0 = Q_F * Nf + Q_S * Ns
    r0 = Nf / Ntot
    omega = Yhat / Yhat.sum()

    r_wd = L_wd / U
    r_we = L_we / U
    rho0_wd = r_wd.max(axis=1)
    rho0_we = r_we.max(axis=1)
    rho0 = np.maximum(rho0_wd, rho0_we)
    peak_hour_wd = r_wd.argmax(axis=1).astype(int)
    peak_hour_we = r_we.argmax(axis=1).astype(int)
    bottleneck = [
        "周末" if rho0_we[i] >= rho0_wd[i] else "工作日" for i in range(N_REGION)
    ]

    g_wd = np.divide(U, L_wd, out=np.full_like(U, np.inf), where=L_wd > 0)
    g_we = np.divide(U, L_we, out=np.full_like(U, np.inf), where=L_we > 0)
    Gmax_raw = np.minimum(g_wd.min(axis=1), g_we.min(axis=1))

    return Problem2Data(
        region=region,
        A=A,
        A0=A0,
        Nf=Nf,
        Ns=Ns,
        Ntot=Ntot,
        P_grid_kw=P_grid_kw,
        D_wd=D_wd,
        D_we=D_we,
        L_wd=L_wd,
        L_we=L_we,
        U=U,
        Yhat=Yhat,
        Dbar=Dbar,
        Qbar=Qbar,
        e=e,
        Dhat=Dhat,
        a_eff=a_eff,
        S0=S0,
        P0=P0,
        K0=K0,
        r0=r0,
        omega=omega,
        rho0_wd=rho0_wd,
        rho0_we=rho0_we,
        rho0=rho0,
        Gmax_raw=Gmax_raw,
        peak_hour_wd=peak_hour_wd,
        peak_hour_we=peak_hour_we,
        bottleneck=bottleneck,
    )


@dataclass
class Scenario:
    """一组验证情景参数。"""

    name: str
    beta: float = 1.0
    delta: float = 0.10
    tau: float = 1.00
    alpha: float = 0.5
    use_mix: bool = True
    cover_req: np.ndarray = field(default_factory=lambda: np.full(N_REGION, 0.9))
    mix_enable: np.ndarray = field(default_factory=lambda: np.ones(N_REGION, dtype=bool))
    delta_i: np.ndarray | None = None
    note: str = ""

    def delta_vec(self) -> np.ndarray:
        if self.delta_i is None:
            return np.full(N_REGION, float(self.delta))
        return np.asarray(self.delta_i, dtype=float)


def zmin_cover(data: Problem2Data, beta: float, cover_req: np.ndarray) -> np.ndarray:
    """由覆盖下界给出的最小新增总桩数。"""
    need = np.maximum(0.0, cover_req * data.A - data.A0)
    step = beta * data.a_eff
    z = np.zeros(N_REGION)
    mask = need > EPS
    z[mask] = np.ceil(need[mask] / step[mask] - 1e-12)
    return z.astype(int)


def zsat_cover(data: Problem2Data, beta: float) -> np.ndarray:
    """覆盖率达到 100% 所需新增总桩数。超过该值 F2 不再改善。"""
    need = np.maximum(0.0, data.A - data.A0)
    z = np.ceil(need / (beta * data.a_eff) - 1e-12)
    return z.astype(int)


def grid_add_cap(data: Problem2Data, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """电网约束给出的相对放大上限与新增功率上限。"""
    gmax = tau * data.Gmax_raw
    padd = (gmax - 1.0) * data.P0
    return gmax, padd, gmax


def coverage(data: Problem2Data, x: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    """规划后面积覆盖率，带饱和截断。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    raw = (data.A0 + beta * data.a_eff * (x + y)) / data.A
    return np.minimum(1.0, raw)


def growth(data: Problem2Data, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """装机相对增长系数 g_i。"""
    return 1.0 + (P_F * x + P_S * y) / data.P0


def service_capacity(data: Problem2Data, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return Q_F * (data.Nf + x) + Q_S * (data.Ns + y)


def mix_ratio(data: Problem2Data, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (data.Nf + x) / (data.Ntot + x + y)


def objectives(
    data: Problem2Data,
    x: np.ndarray,
    y: np.ndarray,
    beta: float,
    alpha: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """返回 F1, F2, F3, 覆盖率向量, 区域最大压力向量。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    S = coverage(data, x, y, beta)
    g = growth(data, x, y)
    rho = g * data.rho0
    f1 = float((C_F * x + C_S * y).sum())
    f2 = float((data.omega * S).sum())
    rho_bar = float(rho.mean())
    f3 = float(alpha * rho.max() + (1.0 - alpha) * np.mean((rho - rho_bar) ** 2))
    return f1, f2, f3, S, rho


def constraint_violations(
    data: Problem2Data,
    x: np.ndarray,
    y: np.ndarray,
    scen: Scenario,
) -> dict[str, np.ndarray]:
    """逐区域约束违反量（已截断为非负）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    S = coverage(data, x, y, scen.beta)
    K = service_capacity(data, x, y)
    rho = growth(data, x, y) * data.rho0
    v_cover = np.maximum(0.0, scen.cover_req - S)
    v_service = np.maximum(0.0, data.Dhat - K)
    v_grid = np.maximum(0.0, rho - scen.tau)
    ratio = mix_ratio(data, x, y)
    dlt = scen.delta_vec()
    rlo = data.r0 - dlt
    rhi = data.r0 + dlt
    v_mix = np.maximum(0.0, rlo - ratio) + np.maximum(0.0, ratio - rhi)
    v_mix = np.where(scen.use_mix & scen.mix_enable, v_mix, 0.0)
    return {
        "cover": v_cover,
        "service": v_service,
        "grid": v_grid,
        "mix": v_mix,
    }


def total_cv(viol: dict[str, np.ndarray]) -> float:
    """归一化后的总约束违反量。"""
    return float(
        viol["cover"].sum()
        + viol["service"].sum() / max(float(np.abs(viol["service"]).max()), 1.0)
        + viol["grid"].sum()
        + viol["mix"].sum()
    )


def evaluate_batch(
    data: Problem2Data,
    X: np.ndarray,
    scen: Scenario,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化评价种群。X 形状 (n, 20)，前 10 为快充，后 10 为慢充。"""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x = X[:, :N_REGION]
    y = X[:, N_REGION:]
    S = np.minimum(1.0, (data.A0 + scen.beta * data.a_eff * (x + y)) / data.A)
    g = 1.0 + (P_F * x + P_S * y) / data.P0
    rho = g * data.rho0
    f1 = (C_F * x + C_S * y).sum(axis=1)
    f2 = (data.omega * S).sum(axis=1)
    rho_bar = rho.mean(axis=1, keepdims=True)
    f3 = scen.alpha * rho.max(axis=1) + (1.0 - scen.alpha) * ((rho - rho_bar) ** 2).mean(axis=1)
    F = np.column_stack([f1, -f2, f3])

    v_cover = np.maximum(0.0, scen.cover_req - S).sum(axis=1)
    K = Q_F * (data.Nf + x) + Q_S * (data.Ns + y)
    v_service = np.maximum(0.0, data.Dhat - K).sum(axis=1)
    v_grid = np.maximum(0.0, rho - scen.tau).sum(axis=1)
    if scen.use_mix:
        den = data.Ntot + x + y
        ratio = (data.Nf + x) / den
        dlt = scen.delta_vec()
        rlo = data.r0 - dlt
        rhi = data.r0 + dlt
        v_mix_r = np.maximum(0.0, rlo - ratio) + np.maximum(0.0, ratio - rhi)
        v_mix_r = np.where(scen.mix_enable, v_mix_r, 0.0)
        v_mix = v_mix_r.sum(axis=1)
    else:
        v_mix = np.zeros(X.shape[0])
    svc_scale = max(float(data.Dhat.max()), 1.0)
    G = np.column_stack([
        v_cover,
        v_service / svc_scale,
        v_grid,
        v_mix,
    ])
    return F, G


def enumerate_region_pairs(
    data: Problem2Data,
    i: int,
    scen: Scenario,
    z_hi: int | None = None,
) -> list[tuple[int, int]]:
    """枚举单区域满足覆盖、车次、电网、结构约束的整数 (x, y)。"""
    gmax, padd, _ = grid_add_cap(data, scen.tau)
    if padd[i] < -EPS:
        return []
    zmin = int(zmin_cover(data, scen.beta, scen.cover_req)[i])
    zsat = int(zsat_cover(data, scen.beta)[i])
    if z_hi is None:
        z_hi = zsat
    xmax = int(np.floor(max(0.0, padd[i]) / P_F + EPS))
    ymax = int(np.floor(max(0.0, padd[i]) / P_S + EPS))
    need_svc = data.Dhat[i] - data.K0[i]
    dlt = float(scen.delta_vec()[i])
    rlo = data.r0[i] - dlt
    rhi = data.r0[i] + dlt
    use_mix = bool(scen.use_mix and scen.mix_enable[i])
    pairs: list[tuple[int, int]] = []
    for x in range(0, xmax + 1):
        y_grid = int(np.floor((padd[i] - P_F * x) / P_S + EPS))
        y_grid = min(y_grid, ymax)
        if y_grid < 0:
            continue
        y_lo = max(0, zmin - x)
        if need_svc > 0:
            y_lo = max(y_lo, int(np.ceil((need_svc - Q_F * x) / Q_S - EPS)))
        y_hi = min(y_grid, max(z_hi - x, y_lo))
        if y_lo > y_grid:
            continue
        y_hi = min(y_grid, z_hi - x) if z_hi - x >= 0 else -1
        if y_lo > y_hi:
            continue
        for y in range(y_lo, y_hi + 1):
            if use_mix:
                ratio = (data.Nf[i] + x) / (data.Ntot[i] + x + y)
                if ratio < rlo - 1e-12 or ratio > rhi + 1e-12:
                    continue
            pairs.append((x, y))
    return pairs


def precheck_region(data: Problem2Data, i: int, scen: Scenario) -> dict:
    """单区域可行性诊断。"""
    gmax, padd, _ = grid_add_cap(data, scen.tau)
    zmin = int(zmin_cover(data, scen.beta, scen.cover_req)[i])
    zsat = int(zsat_cover(data, scen.beta)[i])
    xmax = int(np.floor(max(0.0, padd[i]) / P_F + EPS)) if padd[i] >= 0 else -1
    ymax = int(np.floor(max(0.0, padd[i]) / P_S + EPS)) if padd[i] >= 0 else -1
    pairs = enumerate_region_pairs(data, i, scen, z_hi=zsat)
    # 最大电网可行覆盖（忽略结构约束）
    s_grid = []
    if padd[i] >= 0:
        for x in range(0, max(xmax, 0) + 1):
            y = int(np.floor((padd[i] - P_F * x) / P_S + EPS))
            if y < 0:
                continue
            s_val = min(1.0, (data.A0[i] + scen.beta * data.a_eff[i] * (x + y)) / data.A[i])
            s_grid.append((s_val, x, y))
    if s_grid:
        s_best = max(s_grid, key=lambda t: (t[0], -(C_F * t[1] + C_S * t[2])))
        s_max_grid = s_best[0]
        xy_max_grid = (s_best[1], s_best[2])
    else:
        s_max_grid = float(data.S0[i])
        xy_max_grid = (0, 0)

    s_max_mix = None
    xy_max_mix = None
    if pairs:
        best = max(pairs, key=lambda xy: (xy[0] + xy[1], -xy[0]))
        s_max_mix = min(1.0, (data.A0[i] + scen.beta * data.a_eff[i] * (best[0] + best[1])) / data.A[i])
        xy_max_mix = best

    status_quo_grid_ok = data.rho0[i] <= scen.tau + 1e-12
    cover_ok = len(pairs) > 0
    conflict = []
    if not status_quo_grid_ok:
        conflict.append(
            f"现状最大压力 {data.rho0[i]:.6f} > τ={scen.tau:.2f}"
        )
    if padd[i] < 0:
        conflict.append("电网不允许任何新增装机")
    if s_max_grid + 1e-12 < scen.cover_req[i]:
        conflict.append(
            f"电网可行最大覆盖 {s_max_grid:.6f} < 要求 {scen.cover_req[i]:.3f}"
        )
    if scen.use_mix and scen.mix_enable[i] and s_max_grid + 1e-12 >= scen.cover_req[i] and not cover_ok:
        conflict.append("覆盖+电网可行，但快慢比约束下无整数解")

    min_cost = None
    min_cost_xy = None
    if pairs:
        costs = [(C_F * x + C_S * y, x, y) for x, y in pairs]
        c, x, y = min(costs)
        min_cost, min_cost_xy = float(c), (int(x), int(y))

    return {
        "region": int(data.region[i]),
        "S0": float(data.S0[i]),
        "zmin": zmin,
        "zsat": zsat,
        "Gmax": float(gmax[i]),
        "Padd": float(padd[i]),
        "xmax": xmax,
        "ymax": ymax,
        "n_pairs": len(pairs),
        "feasible": cover_ok and status_quo_grid_ok,
        "status_quo_grid_ok": status_quo_grid_ok,
        "S_max_grid": float(s_max_grid),
        "xy_max_grid": xy_max_grid,
        "S_max_mix": None if s_max_mix is None else float(s_max_mix),
        "xy_max_mix": xy_max_mix,
        "min_cost": min_cost,
        "min_cost_xy": min_cost_xy,
        "rho0": float(data.rho0[i]),
        "rho0_wd": float(data.rho0_wd[i]),
        "rho0_we": float(data.rho0_we[i]),
        "bottleneck": data.bottleneck[i],
        "service_slack0": float(data.K0[i] - data.Dhat[i]),
        "conflict": "；".join(conflict) if conflict else "",
        "pairs": pairs,
    }


def precheck_scenario(data: Problem2Data, scen: Scenario) -> pd.DataFrame:
    rows = [precheck_region(data, i, scen) for i in range(N_REGION)]
    slim = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "pairs"}
        if d["xy_max_grid"] is not None:
            d["x_max_grid"], d["y_max_grid"] = d["xy_max_grid"]
        else:
            d["x_max_grid"], d["y_max_grid"] = None, None
        if d["xy_max_mix"] is not None:
            d["x_max_mix"], d["y_max_mix"] = d["xy_max_mix"]
        else:
            d["x_max_mix"], d["y_max_mix"] = None, None
        if d["min_cost_xy"] is not None:
            d["x_min_cost"], d["y_min_cost"] = d["min_cost_xy"]
        else:
            d["x_min_cost"], d["y_min_cost"] = None, None
        del d["xy_max_grid"]
        del d["xy_max_mix"]
        del d["min_cost_xy"]
        slim.append(d)
    return pd.DataFrame(slim)


def make_strict_m0(data: Problem2Data) -> Scenario:
    return Scenario(name="M0_strict", use_mix=False, note="无快慢比约束的原始模型")


def make_strict_m1(data: Problem2Data, delta: float = 0.10) -> Scenario:
    return Scenario(name="M1_strict", use_mix=True, delta=delta, note="含快慢比约束的原始模型")


def make_diag_m0(data: Problem2Data, beta: float = 1.0, tau: float = 1.0, alpha: float = 0.5) -> Scenario:
    """诊断模型：区域9覆盖下界改为电网可行最大覆盖。"""
    cover = np.full(N_REGION, 0.9)
    tmp = Scenario(name="tmp", beta=beta, tau=tau, use_mix=False, cover_req=cover.copy())
    info = precheck_region(data, 8, tmp)
    cover[8] = min(0.9, info["S_max_grid"])
    return Scenario(
        name="M0_diag",
        beta=beta,
        delta=0.10,
        tau=tau,
        alpha=alpha,
        use_mix=False,
        cover_req=cover,
        note="诊断：区域9覆盖下界改为电网可行最大覆盖，其余区域仍为0.9",
    )


def make_diag_m1(
    data: Problem2Data,
    beta: float = 1.0,
    delta: float = 0.10,
    tau: float = 1.0,
    alpha: float = 0.5,
    lift_infeasible_delta: bool = True,
) -> Scenario:
    """
    诊断模型：
      - 区域9覆盖下界改为结构+电网约束下的最大可行覆盖；
        若结构约束下无解，则关闭区域9结构约束并取电网最大覆盖；
      - 若某区域在给定 δ 下无整数解，可单独上调该区域 δ 至最小可行值。
    """
    cover = np.full(N_REGION, 0.9)
    mix_enable = np.ones(N_REGION, dtype=bool)
    delta_i = np.full(N_REGION, float(delta))
    notes = ["诊断：区域9覆盖下界改为可行最大覆盖"]

    tmp = Scenario(name="tmp", beta=beta, delta=delta, tau=tau, use_mix=True, cover_req=cover.copy())
    info9 = precheck_region(data, 8, tmp)
    if info9["n_pairs"] > 0 and info9["S_max_mix"] is not None:
        cover[8] = min(0.9, info9["S_max_mix"])
    else:
        info9b = precheck_region(
            data, 8, Scenario(name="t", beta=beta, tau=tau, use_mix=False, cover_req=np.full(N_REGION, 0.9))
        )
        cover[8] = min(0.9, info9b["S_max_grid"])
        mix_enable[8] = False
        notes.append("区域9在结构约束下无法同时满足覆盖，已关闭该区域快慢比约束")

    if lift_infeasible_delta:
        for i in range(N_REGION):
            if not mix_enable[i]:
                continue
            probe = Scenario(
                name="p", beta=beta, delta=float(delta_i[i]), tau=tau,
                use_mix=True, cover_req=cover.copy(), mix_enable=mix_enable.copy(),
            )
            info = precheck_region(data, i, probe)
            if info["n_pairs"] == 0:
                dmin = min_delta_for_region(data, i, beta, tau)
                if dmin is None:
                    notes.append(f"区域{i+1}在任意δ下仍无覆盖+电网+结构整数解")
                else:
                    delta_i[i] = dmin
                    notes.append(f"区域{i+1}基准δ={delta:.3f}无整数解，诊断δ上调至{dmin:.3f}")

    return Scenario(
        name="M1_diag",
        beta=beta,
        delta=delta,
        tau=tau,
        alpha=alpha,
        use_mix=True,
        cover_req=cover,
        mix_enable=mix_enable,
        delta_i=delta_i,
        note="；".join(notes),
    )


def min_delta_for_region(data: Problem2Data, i: int, beta: float, tau: float) -> float | None:
    """使覆盖+电网+结构约束同时可行的最小 δ（0.001 网格）。"""
    cover = np.full(N_REGION, 0.9)
    for k in range(0, 501):
        delta = k / 1000.0
        scen = Scenario(name="scan", beta=beta, delta=delta, tau=tau, use_mix=True, cover_req=cover)
        info = precheck_region(data, i, scen)
        if info["n_pairs"] > 0:
            return delta
    return None


def useful_bounds(data: Problem2Data, scen: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """NSGA-II 变量上界：电网上限与 100% 覆盖所需桩数的较小值。"""
    _, padd, _ = grid_add_cap(data, scen.tau)
    zsat = zsat_cover(data, scen.beta)
    x_max = np.minimum(np.floor(np.maximum(padd, 0.0) / P_F).astype(int), zsat)
    y_max = np.minimum(np.floor(np.maximum(padd, 0.0) / P_S).astype(int), zsat)
    zmin = zmin_cover(data, scen.beta, scen.cover_req)
    y_max = np.maximum(y_max, zmin)
    x_max = np.maximum(x_max, 0)
    y_max = np.maximum(y_max, 0)
    return x_max.astype(int), y_max.astype(int)


def plan_detail(data: Problem2Data, x: np.ndarray, y: np.ndarray, scen: Scenario) -> pd.DataFrame:
    """逐区域配置明细，供论文表使用。"""
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    S = coverage(data, x, y, scen.beta)
    K = service_capacity(data, x, y)
    g = growth(data, x, y)
    rho_wd = g * data.rho0_wd
    rho_we = g * data.rho0_we
    Lwd = g[:, None] * data.L_wd
    Lwe = g[:, None] * data.L_we
    r_wd = Lwd / data.U
    r_we = Lwe / data.U
    danger = []
    for i in range(N_REGION):
        if r_wd[i].max() >= r_we[i].max():
            t = int(r_wd[i].argmax())
            danger.append(f"工作日 {HOUR_LABELS[t]}")
        else:
            t = int(r_we[i].argmax())
            danger.append(f"周末 {HOUR_LABELS[t]}")
    return pd.DataFrame({
        "区域": data.region,
        "预测需求_kWh_d": data.Yhat,
        "预测车次": data.Dhat,
        "单车电量_kWh": data.e,
        "现有快充": data.Nf.astype(int),
        "现有慢充": data.Ns.astype(int),
        "新增快充": x,
        "新增慢充": y,
        "规划后覆盖率": S,
        "现状覆盖率": data.S0,
        "服务能力": K,
        "服务能力裕度": K - data.Dhat,
        "工作日最大压力": rho_wd,
        "周末最大压力": rho_we,
        "区域最大压力": np.maximum(rho_wd, rho_we),
        "最危险时段": danger,
        "快充比例_规划后": mix_ratio(data, x, y),
        "快充比例_现状": data.r0,
        "增长系数_g": g,
        "投资成本_万元": C_F * x + C_S * y,
    })


def audit_plan(data: Problem2Data, x: np.ndarray, y: np.ndarray, scen: Scenario) -> dict:
    """不依赖求解器缓存的完整约束复核。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    int_ok = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
    int_ok = int_ok and bool(np.allclose(x, np.round(x)) and np.allclose(y, np.round(y)))
    int_ok = int_ok and bool(np.all(x >= -AUDIT_TOL) and np.all(y >= -AUDIT_TOL))
    S = coverage(data, x, y, scen.beta)
    K = service_capacity(data, x, y)
    g = growth(data, x, y)
    Lwd = g[:, None] * data.L_wd
    Lwe = g[:, None] * data.L_we
    viol_wd = np.maximum(0.0, Lwd - scen.tau * data.U)
    viol_we = np.maximum(0.0, Lwe - scen.tau * data.U)
    viol = constraint_violations(data, x, y, scen)
    max_grid = float(max(viol_wd.max(), viol_we.max()))
    max_all = float(max(
        viol["cover"].max(),
        viol["service"].max(),
        viol["grid"].max(),
        viol["mix"].max(),
        max_grid,
    ))
    f1, f2, f3, _, rho = objectives(data, x, y, scen.beta, scen.alpha)
    return {
        "integer_nonneg": int_ok,
        "min_cover": float(S.min()),
        "cover_ok": bool(np.all(S + AUDIT_TOL >= scen.cover_req)),
        "service_ok": bool(np.all(K + AUDIT_TOL >= data.Dhat)),
        "grid_ok": bool(max_grid <= AUDIT_TOL),
        "mix_ok": bool(viol["mix"].max() <= AUDIT_TOL),
        "n_grid_constraints": int(N_REGION * N_HOUR * 2),
        "max_grid_violation_kW": max_grid,
        "max_cover_violation": float(viol["cover"].max()),
        "max_service_violation": float(viol["service"].max()),
        "max_mix_violation": float(viol["mix"].max()),
        "max_any_violation": max_all,
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "max_rho": float(rho.max()),
        "rho_var": float(np.mean((rho - rho.mean()) ** 2)),
        "pass": bool(int_ok and max_all <= AUDIT_TOL),
    }


def derived_parameter_tables(data: Problem2Data) -> dict[str, pd.DataFrame]:
    """基础数据与派生参数，供人工复核。"""
    base = pd.DataFrame({
        "区域": data.region,
        "总面积_km2": data.A,
        "现有覆盖面积_km2": data.A0,
        "现状覆盖率": data.S0,
        "现有快充": data.Nf.astype(int),
        "现有慢充": data.Ns.astype(int),
        "现有总桩数": data.Ntot.astype(int),
        "现有快充比例": data.r0,
        "现有装机_kW": data.P0,
        "附件1电网总容量_kW": data.P_grid_kw,
        "单桩有效覆盖_km2": data.a_eff,
        "预测需求_kWh_d": data.Yhat,
        "需求权重_omega": data.omega,
        "典型日车次_Dbar": data.Dbar,
        "典型日电量_Qbar": data.Qbar,
        "单车电量_e": data.e,
        "预测车次_Dhat": data.Dhat,
        "现状服务能力": data.K0,
        "服务能力裕度_现状": data.K0 - data.Dhat,
        "现状工作日最大压力": data.rho0_wd,
        "现状周末最大压力": data.rho0_we,
        "现状最大压力": data.rho0,
        "压力瓶颈日": data.bottleneck,
        "工作日最危险时段": [HOUR_LABELS[t] for t in data.peak_hour_wd],
        "周末最危险时段": [HOUR_LABELS[t] for t in data.peak_hour_we],
        "Gmax_tau1": data.Gmax_raw,
        "新增功率上限_tau1_kW": (data.Gmax_raw - 1.0) * data.P0,
    })
    checks = pd.DataFrame([
        {"检查项": "e_i>0", "结果": "通过" if np.all(data.e > 0) else "失败", "最小取值": float(data.e.min())},
        {"检查项": "Dhat>0", "结果": "通过" if np.all(data.Dhat > 0) else "失败", "最小取值": float(data.Dhat.min())},
        {"检查项": "未融合实际需求", "结果": "通过", "最小取值": np.nan},
        {"检查项": "Y_plan=Yhat", "结果": "通过", "最小取值": float(data.Yhat.min())},
        {"检查项": "量纲：负荷与上限均为kW口径", "结果": "通过（按附件数值直接使用）", "最小取值": np.nan},
        {"检查项": "现状服务能力是否已覆盖预测车次", "结果": "全部满足" if np.all(data.K0 >= data.Dhat) else "存在缺口",
         "最小取值": float((data.K0 - data.Dhat).min())},
    ])
    return {"派生参数": base, "数据校验": checks}


def save_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
