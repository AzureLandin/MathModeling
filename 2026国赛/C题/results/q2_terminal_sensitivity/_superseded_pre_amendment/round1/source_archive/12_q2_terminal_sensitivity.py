"""Q2 sensitivity of the fixed nominal day-end target: 4800 / 6000 / 7200 kWh.

Run (math_modeling env):
    E:/Anaconda/envs/math_modeling/python.exe code/12_q2_terminal_sensitivity.py --stage register
    E:/Anaconda/envs/math_modeling/python.exe code/12_q2_terminal_sensitivity.py --stage run
    E:/Anaconda/envs/math_modeling/python.exe code/12_q2_terminal_sensitivity.py --stage all

What is held fixed (inherited from the frozen snapshot, not re-implemented here):
    predictor        L[k-7,t] with fallback to L[k-1,t] when history < 7 days; V[k-1,t]
    protection       per-slot empirical q80 of the net-demand residual over the most recent
                     W=28 valid history days; sorted ascending, take index ceil(0.8*m)-1;
                     fewer than 7 valid days -> zero correction; negative corrections and
                     negative net demand preserved, never clipped
    physics          dt=1/6 h, eta_c=eta_d=0.9, state 1200..10800 kWh, 5000 kW bus limit
    kernel           the original SciPy/HiGHS MILP used by the frozen 08 module, same matrix
                     build, variable order, relative gap target 1e-9 and 120 s time limit;
                     not replaced by an LP and with no secondary objective
    warm-up          the original lag January run; every group forks from the common
                     2025-02-01 state 8801.462273333342 kWh and then carries its own actual
                     day-end state forward, with no daily reset
    settlement       the 144-slot plan is frozen at 00:00; actual surplus charges, actual
                     deficit discharges, the remaining deficit is bought at 5x the slot price;
                     every planned kWh is paid for; unused energy carries no extra penalty
    information      attachment-1 prices and attachment-2 history only; no attachment 3/4 and
                     no same-day future realisation is used for any decision

Only the right-hand side of the nominal terminal equality E_bar[144] = h_k changes.
4800 / 6000 / 7200 are pre-registered sensitivity candidates, not optima; the grid is frozen
before the run and is not widened afterwards to chase savings.

Outputs
    results/q2_terminal_sensitivity/       tables, per-group dispatch, validation, registration
    figures/q2_terminal_sensitivity/       figures
    reports/问题二_固定名义末态敏感性实验结果报告.md
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_terminal_sensitivity'
FIG = ROOT / 'figures/q2_terminal_sensitivity'
ARCHIVE = OUT / 'source_archive'
SNAP = ROOT / 'results/q2_revision_audit_20260911/source_snapshot'
REFERENCE_RUN = ROOT / 'results/q2_terminal_experiment_20260911'
LICENSE_TOTAL = 14158360.487139747            # existing T_q80_6000, 334-day cash total
LICENSE_COST_TOL = 1e-4                       # yuan,  task-book tolerance
LICENSE_ENERGY_TOL = 1e-6                     # kWh,   task-book tolerance
DEFAULT_TARGET = 6000.0
CANDIDATES = (4800.0, 6000.0, 7200.0)
COMMON_INITIAL = 8801.462273333342
POLICIES = [
    dict(id='T_q80_4800', kind='fixed', theta=0.8, W=28, J=None, terminal=4800.0),
    dict(id='T_q80_6000', kind='fixed', theta=0.8, W=28, J=None, terminal=6000.0),
    dict(id='T_q80_7200', kind='fixed', theta=0.8, W=28, J=None, terminal=7200.0),
]
ENERGY_COLUMNS = ['planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
                  'state_start_kWh', 'state_end_kWh', 'nominal_state_end_kWh',
                  'net_forecast_kWh', 'residual_adjustment_kWh', 'protected_net_kWh']
COST_COLUMNS = ['planned_cost_yuan', 'emergency_cost_yuan']

PLAN_TEXT = """# 问题二：固定名义日末目标敏感性实验方案（事前登记）

登记日期：2026-09-11。本文件在任何敏感性策略运行之前写入，登记候选网格、固定条件、
通过标准与输出位置；结果写入后不修改本文件的候选与参数。

## 1. 问题分析

已完成的终端 2x2 对照表明：在固定 q80 保护下，把名义日末目标由“当天实际日初电量”改为
固定 6000 kWh，334 天现金总费用下降 179,318.81 元（-1.2507%），11 个月全部改善。该结果
只说明 6000 优于“跟随实际日初”，不说明 6000 在目标值上最优，也不说明收益不依赖单一数值。

本实验要回答的问题：在预测、保护、物理参数、MILP 内核、实际反馈与收费规则全部不变的
条件下，名义日末目标在 6000 附近变化（4800 / 6000 / 7200）会如何改变 334 天实际现金
总费用？收益是集中在一个特殊数值上的巧合，还是在邻近目标上同向？

机制上的两个方向相反的作用：较低的目标会压低次日规划的库存、减少高价时段的过度购电与
循环损耗；但它也可能削弱次晨缓冲，抬高清晨 0—10 时的应急购电。两者孰强属于待检验假说，
不预设低目标必然更省。

## 2. 数据预处理

- 仅使用附件 1 的 144 点日内电价和附件 2 的 365x144 负载、光伏实际功率；冻结并核对
  SHA256、字段形状与日期。不使用附件 3、4。
- 每段 dt=1/6 小时。功率单位 kW，全部流量与电池内部储电量单位 kWh，电价单位元/kWh。
  样本继续解释为前 10 分钟区间的代表功率，不称为段初已知真值。
- 不平滑、不删点、不归一化。净需求 n=(L-V)*dt 允许为负，保留光伏富余与充电机会。
- 预测器固定为 Lhat[k,t]=L[k-7,t]、Vhat[k,t]=V[k-1,t]；负载历史不足 7 天回退昨日同段；
  首日沿用原公共冷启动，不生成伪预测。
- 残差 eps[j,t]=n[j,t]-nhat[j,t] 只来自当时发出的预测。每天 0:00 仅取 j<k 的有效日期。
- q80 组逐时段取最近至多 28 个有效日期的经验分位数，排序后取第 ceil(0.8m) 个值；有效
  日期不足 7 个则零修正并记录原因。保护需求 ntilde=nhat+r，r 与 ntilde 均不截零。

## 3. 模型建立

### 3.1 事前候选登记

| 组 | 历史残差保护 | 名义日末目标 h_k | 说明 |
|---|---|---|---|
| T_q80_4800 | 每段最近 28 日残差 q80 | 4800 kWh | 低于公共初态与 6000 的邻近候选 |
| T_q80_6000 | 同上 | 6000 kWh | 已登记对照，本实验的复现锚点与比较基准 |
| T_q80_7200 | 同上 | 7200 kWh | 高于 6000 的邻近候选 |

三个数值是事前指定的敏感性候选，来源为：6000 是题面给定的 2025-01-01 初始电量、也是旧
方案已用的终端敏感性值；4800 与 7200 是围绕它的等距邻近值（步长 1200 kWh，约为可用区间
1200—10800 的 1/8）。它们不是全年搜索得到的最优参数，也不因结果好坏而增减或改名。

### 3.2 名义模型（只改末态等式右端）

单日省略日期下标。名义变量带横线，z 为二进制充电模式。

min sum_t p_t q_t

q_t + dbar_t = ntilde_t + cbar_t + wbar_t

Ebar_{t+1} = Ebar_t + 0.9 cbar_t - dbar_t/0.9

0 <= cbar_t <= M zbar_t,  0 <= dbar_t <= M (1-zbar_t),  zbar_t in {0,1},  M = 5000/6

q_t, wbar_t >= 0,  1200 <= Ebar_t <= 10800,  Ebar_0 = E_{k,0},  Ebar_144 = h_k

原规则 h_k = E_{k,0}（当天实际日初电量）；本实验 h_k 取上表固定值。h_k 是名义规划目标，
不是对真实库存的强制重置；实际日末可以偏离目标，不做事后裁剪。

### 3.3 实际执行与收费

计划 q 在每天 0:00 冻结。实际控制沿用当前区间内代表功率的快速反馈：富余在功率与容量
允许内充电，缺额在功率与最低电量允许内放电，剩余缺口按 5 倍价应急购电。内部状态按
E_{t+1}=E_t+0.9c_t-d_t/0.9 递推并交给次日。

日费用 C_k = sum_t p_t q_t + 5 sum_t p_t e_t。普通购电全部按计划付款，未使用电量无额外
罚金，无售电收入。共同 1 月预运行从 6000 kWh 出发；三组 2025-02-01 共同初态
8801.462273333342 kWh，之后各自连续继承实际末态，不每日重置。

## 4. 模型求解与评价

### 4.1 算法

调用 math_modeling 环境下 scipy.optimize.milp/HiGHS，采用冻结快照中的原 MILP 内核：
每次 721 个连续变量 + 144 个二进制变量，矩阵结构、变量顺序、相对 gap 目标 1e-9、
时限 120 秒与既有实现一致，不替换为 LP、不添加二级目标。每组 334 次日前求解；
实际反馈每段常数复杂度。

### 4.2 执行顺序

1. 复现 T_q80_6000，核对 334 天总费用 14158360.487139747 元，并与既有
   results/q2_terminal_experiment_20260911/T_q80_6000/dispatch.csv 逐段比较。
2. 费用容差 1e-4 元，物理量容差 1e-6 kWh。若不一致，先定位原因并暂停新组结论。
3. 复现通过后，顺序无关地运行 4800 与 7200 两组。
4. 由独立脚本从逐段明细重算费用、能量与对照差额，再生成报告。

### 4.3 主指标与辅助指标

主指标为 2—12 月 334 天实际现金总费用（计划费 + 5 倍应急费）。同时报告计划费、应急费、
应急电量、应急事件数、未使用电量、储能损耗、日末满电天数、逐月差额、最差日期与
0—10 时应急费。共同期初与各不相同的期末库存必须披露；nu=median(p)/0.9 的期末存量调整
只作诊断，不混入实际应付费用，也不能用年末估值替代每日终端规则的运行实验。

### 4.4 通过标准

- 复现 T_q80_6000：总费用容差 1e-4 元，逐段物理量容差 1e-6 kWh；
- 每段供需平衡、状态递推、功率、容量、非负、互斥的最大违反不超过 1e-6 kWh；
- 累计费用与能量核账不超过 1e-4；日间状态连续，无日重置；
- 名义末态确实满足登记的 h_k（误差 < 1e-6 kWh）；
- 三组原点预测与残差修正逐段一致（保护条件固定，不随状态变化）；
- 无求解失败、无回退；
- 未来扰动不改变当日 0:00 的预测、修正与计划。

### 4.5 负面结果与边界

若低目标或高目标更贵，保留负面结论，不追加新目标值使结果变好。目标在邻近取值上优于
6000 也不等于该值为全局最优，更不声称问题二已取得全局最优控制。本年度数据此前已参与
诊断，故本实验属于因果滚动回测，不是独立盲测。本轮不实施备用控制、MPC、场景优化或
第三、四问；未实现单策略逐日断点恢复。

## 5. 输出

results/q2_terminal_sensitivity/：注册文件、三组逐段明细、汇总与对照表、月度与小时诊断、
验证记录；figures/q2_terminal_sensitivity/：图表；source_archive/：实际运行的源码。
原附件、第一问结果、旧 Baseline 与既有终端实验输出均不覆盖。
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def module():
    """Load the frozen snapshot of 08. Identical bytes to the verified current source."""
    spec = importlib.util.spec_from_file_location('frozen_q2_sens',
                                                 SNAP / 'code/08_q2_quantile_experiment.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.OUT = OUT
    return loaded


def build_archive(m, load, pv, target):
    """Same MILP call as the frozen module, with the nominal terminal target parametrised."""
    class TerminalArchive(m.Archive):
        def solve_plan(self, protected_net_kWh, price, initial):
            self.solve_calls += 1
            (q, c, d, w, state), info = m.core.solve(
                np.asarray(protected_net_kWh) / m.DT, np.zeros(144), price,
                integer=True, initial_kWh=initial, terminal_kWh=float(target))
            assert abs(state[-1] - float(target)) < 1e-6
            return q, state, info
    return TerminalArchive(load, pv, len(load))


# --------------------------------------------------------------------------------------
# stage 1: pre-registration
# --------------------------------------------------------------------------------------
def stage_register():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = ['code/02_q1_baseline.py', 'code/05_q2_baseline.py',
              'code/08_q2_quantile_experiment.py', 'code/12_q2_terminal_sensitivity.py',
              '附件/附件1.xlsx', '附件/附件2.xlsx', 'results/audit/audit_summary.json']
    hashes = {rel: digest(ROOT / rel) for rel in inputs}
    snapshot_hashes = {rel: digest(SNAP / rel) for rel in
                       ['code/08_q2_quantile_experiment.py', 'code/05_q2_baseline.py',
                        'code/02_q1_baseline.py']}
    frozen_matches_current = hashes['code/08_q2_quantile_experiment.py'] == \
        snapshot_hashes['code/08_q2_quantile_experiment.py']
    (OUT / 'experiment_plan.md').write_text(PLAN_TEXT, encoding='utf-8')
    record = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(), status='registered',
        question='How does the fixed nominal day-end target 4800/6000/7200 kWh change the '
                 '334-day realized cash cost, holding predictor, protection, physics, MILP '
                 'kernel, actual feedback and settlement rules fixed?',
        candidates_kWh=list(CANDIDATES),
        candidate_justification='6000 is the problem-given initial energy and the previously '
                                'used terminal sensitivity value; 4800 and 7200 are equidistant '
                                'neighbours (step 1200 kWh, 1/8 of the 1200-10800 range). '
                                'Pre-declared, not year-optimised.',
        grid_is_frozen=True,
        policies=POLICIES,
        fixed_conditions=dict(
            predictor='L[k-7,t], fallback L[k-1,t] when history < 7 days; V[k-1,t]',
            protection='per-slot empirical q80 over the most recent W=28 valid residual days; '
                       'sorted ascending, index ceil(0.8*m)-1; <7 valid days -> zero correction; '
                       'negative corrections and negative net demand preserved',
            physics='dt=1/6 h, eta_c=eta_d=0.9, state 1200-10800 kWh, 5000 kW bus limit',
            kernel='frozen 08 SciPy/HiGHS MILP, same matrix build, variable order, '
                   'mip_rel_gap=1e-9, time_limit=120 s, no LP substitution, no secondary objective',
            warmup='original lag January run; common 2025-02-01 state 8801.462273333342 kWh',
            terminal_rule='E_bar[144] = h_k with h_k the only changed quantity',
            settlement='plan frozen at 00:00 and paid in full; deficit at 5x slot price; '
                       'unused energy free; no selling, no curtailment penalty, no depreciation',
            information='attachment 1 price + attachment 2 history only; no attachment 3/4; '
                        'no same-day future realisation'),
        primary_metric='334-day realized cash cost = planned purchase cost + 5x emergency cost',
        reference=dict(policy='T_q80_6000', total_cost_yuan=LICENSE_TOTAL,
                       source='results/q2_terminal_experiment_20260911/T_q80_6000/dispatch.csv'),
        tolerances=dict(cost_yuan=LICENSE_COST_TOL, energy_kWh=LICENSE_ENERGY_TOL,
                        physical_kWh=1e-6, reconciliation=1e-4),
        pass_criteria=PASS_CRITERIA,
        not_claimed=['6000 or any candidate is a generalisable optimum',
                     'the problem-2 model is globally optimal',
                     'this is an independent held-out test'],
        outputs=dict(results=str(OUT.relative_to(ROOT)), figures=str(FIG.relative_to(ROOT))),
        input_sha256=hashes, snapshot_sha256=snapshot_hashes,
        frozen_snapshot_matches_current_source=frozen_matches_current,
    )
    signature = hashlib.sha256(json.dumps(
        dict(candidates=list(CANDIDATES), policies=POLICIES,
             input_sha256=hashes, snapshot_sha256=snapshot_hashes),
        sort_keys=True).encode()).hexdigest()
    record['signature'] = signature
    save(OUT / 'registration.json', record)
    print(f'registered: candidates={list(CANDIDATES)} signature={signature[:16]}', flush=True)
    print(f'frozen snapshot matches current source: {frozen_matches_current}', flush=True)
    return record


PASS_CRITERIA = dict(
    reproduce_6000_cost_tolerance_yuan=LICENSE_COST_TOL,
    reproduce_6000_segment_energy_tolerance_kWh=LICENSE_ENERGY_TOL,
    max_physical_violation_kWh=1e-6,
    max_reconciliation_yuan_or_kWh=1e-4,
    day_boundary_continuity_kWh=1e-9,
    nominal_terminal_error_kWh=1e-6,
    solver_failures=0,
)


def assert_registered(signature):
    path = OUT / 'registration.json'
    assert path.exists(), 'run --stage register first'
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['signature'] == signature, 'registration and current inputs disagree'
    return saved


# --------------------------------------------------------------------------------------
# stage 2: run
# --------------------------------------------------------------------------------------
def run_one(policy, load, pv, price, warm, signature):
    m = module()
    folder = OUT / policy['id']
    marker = folder / 'completed.json'
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved.get('signature') == signature:
            assert all(digest(folder / k) == v for k, v in saved['files'].items()), \
                'stored result files changed since completion'
            print(f'    {policy["id"]}: reusing registered completion (signature matches)',
                  flush=True)
            return saved['result'], True
        raise RuntimeError(f'{folder} exists with a different signature; refusing to overwrite')
    folder.mkdir(parents=True, exist_ok=True)
    bundle = dict(archive=build_archive(m, load, pv, policy['terminal']),
                  price=price, dates=m.bm.DATES, warmup_states=warm,
                  nu=float(np.median(price) / .9), a_monthly=None, load=load, pv=pv, stop_day=None)
    result = m.run_policy(policy, bundle)
    result['totals']['terminal_rule'] = policy['terminal']
    selection = result.pop('selection')
    if selection:
        pd.DataFrame(selection).to_csv(folder / 'selection_log.csv', index=False)
    assert result['totals']['solver_failures'] == 0, 'solver failure recorded'
    assert result['totals']['fallback_days'] == 0, 'protection fallback recorded'
    assert max(result['validation']['checks'].values()) < 1e-6, result['validation']['checks']
    files = {p.name: digest(p) for p in folder.glob('*.csv')}
    files['validation.json'] = digest(folder / 'validation.json')
    save(marker, dict(signature=signature, result=result, files=files))
    return result, False


def compare_segments(name):
    """Segment-by-segment comparison against the already registered T_q80_6000 output."""
    old_path = REFERENCE_RUN / name / 'dispatch.csv'
    new_path = OUT / name / 'dispatch.csv'
    a = pd.read_csv(old_path, low_memory=False)
    b = pd.read_csv(new_path, low_memory=False)
    assert a[['date', 'slot']].equals(b[['date', 'slot']]), 'date/slot axis differs'
    assert a.theta.astype(str).equals(b.theta.astype(str)), 'selected theta differs'
    energy = {c: float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in ENERGY_COLUMNS}
    cost = {c: float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in COST_COLUMNS}
    totals_old = float(a.planned_cost_yuan.sum() + a.emergency_cost_yuan.sum())
    totals_new = float(b.planned_cost_yuan.sum() + b.emergency_cost_yuan.sum())
    return dict(
        compared_file=str(old_path.relative_to(ROOT)), segments=int(len(b)),
        total_cost_old_yuan=totals_old, total_cost_new_yuan=totals_new,
        total_cost_abs_diff_yuan=abs(totals_old - totals_new),
        license_total_yuan=LICENSE_TOTAL,
        license_total_abs_diff_yuan=abs(totals_new - LICENSE_TOTAL),
        max_energy_diff_kWh=energy, max_cost_diff_yuan=cost,
        max_energy_diff_overall_kWh=max(energy.values()),
        max_cost_diff_overall_yuan=max(cost.values()),
        axis_and_theta_match=True,
        energy_within_tolerance=bool(max(energy.values()) <= LICENSE_ENERGY_TOL),
        cost_within_tolerance=bool(max(cost.values()) <= LICENSE_COST_TOL),
        license_total_within_tolerance=bool(abs(totals_new - LICENSE_TOTAL) <= LICENSE_COST_TOL))


def independent(name, target):
    """Re-derive every cost and physical quantity from the saved raw columns."""
    folder = OUT / name
    d = pd.read_csv(folder / 'dispatch.csv', low_memory=False)
    q = d.planned_kWh.to_numpy()
    c = d.charge_kWh.to_numpy()
    b = d.discharge_kWh.to_numpy()
    e = d.emergency_kWh.to_numpy()
    w = d.unused_kWh.to_numpy()
    s0 = d.state_start_kWh.to_numpy()
    s1 = d.state_end_kWh.to_numpy()
    p = d.price_yuan_kWh.to_numpy()
    load = d.load_kW.to_numpy()
    pv = d.pv_kW.to_numpy()
    nominal = d.nominal_state_end_kWh.to_numpy()
    fee = p * q + 5 * p * e
    planned_cost = float((p * q).sum())
    emergency_cost = float((5 * p * e).sum())
    loss = float((0.1 * c + (1 / 0.9 - 1) * b).sum())
    physical = dict(
        balance_kWh=float(np.max(np.abs(q + pv / 6 + b + e - load / 6 - c - w))),
        state_kWh=float(np.max(np.abs((s1 - s0) - (0.9 * c - b / 0.9)))),
        continuity_kWh=float(np.max(np.abs(s0[1:] - s1[:-1]))),
        bounds_kWh=float(max(0.0, 1200 - min(s0.min(), s1.min()),
                             max(s0.max(), s1.max()) - 10800)),
        power_kWh=float(max(0.0, c.max() - 5000 / 6, b.max() - 5000 / 6)),
        nonnegative_kWh=float(max(0.0, -min(q.min(), c.min(), b.min(), e.min(), w.min()))),
        mutex_kWh=float(np.minimum(c, b).max()),
        nominal_terminal_kWh=float(np.max(np.abs(nominal - float(target)))),
        initial_state_kWh=float(abs(s0[0] - COMMON_INITIAL)),
    )
    energy_identity = float(np.sum(q + e + pv / 6 - load / 6 - w - (0.1 * c + (1 / 0.9 - 1) * b))
                            - (s1[-1] - s0[0]))
    daily = pd.read_csv(folder / 'daily_summary.csv')
    monthly = pd.read_csv(folder / 'monthly_summary.csv')
    events = pd.read_csv(folder / 'emergency_events.csv')
    accounting = dict(
        daily_total_cost_yuan=float(abs(fee.sum() - daily.total_cost_yuan.sum())),
        daily_planned_cost_yuan=float(abs(planned_cost - daily.planned_cost_yuan.sum())),
        daily_emergency_cost_yuan=float(abs(emergency_cost - daily.emergency_cost_yuan.sum())),
        monthly_total_cost_yuan=float(abs(fee.sum() - monthly.total_cost_yuan.sum())),
        monthly_planned_cost_yuan=float(abs(planned_cost - monthly.planned_cost_yuan.sum())),
        monthly_emergency_cost_yuan=float(abs(emergency_cost - monthly.emergency_cost_yuan.sum())),
        event_energy_kWh=float(abs(events.emergency_kWh.sum() - e.sum())),
        event_cost_yuan=float(abs(events.emergency_cost_yuan.sum() - emergency_cost)),
        energy_identity_kWh=abs(energy_identity),
    )
    morning = d.slot < 60
    return dict(
        strategy_id=name, terminal_target_kWh=float(target),
        planned_cost_yuan=planned_cost, emergency_cost_yuan=emergency_cost,
        total_cost_yuan=planned_cost + emergency_cost,
        planned_kWh=float(q.sum()), emergency_kWh=float(e.sum()), unused_kWh=float(w.sum()),
        charge_kWh=float(c.sum()), discharge_kWh=float(b.sum()), loss_kWh=loss,
        morning_0_10_emergency_cost_yuan=float(fee[morning].sum() - (p * q)[morning].sum()),
        morning_0_10_emergency_kWh=float(e[morning].sum()),
        emergency_slots=int((e > 1e-6).sum()),
        emergency_days=int(d.assign(a=e > 1e-6).groupby('date').a.any().sum()),
        emergency_events=int(len(events)),
        full_end_days=int((s1 >= 10800 - 1e-6).sum()),
        empty_slots=int((s1 <= 1200 + 1e-6).sum()),
        mean_initial_kWh=float(s0.mean()), initial_kWh=float(s0[0]), final_kWh=float(s1[-1]),
        segments=int(len(d)), days=int(d.date.nunique()),
        physical=physical, accounting=accounting,
    )


def causal_checks(load, pv, price, warm):
    """Future data must not change today's 00:00 forecast, correction or plan."""
    m = module()
    rows = []
    for target in CANDIDATES:
        for k in (31, 200):
            perturbed_load = load.copy()
            perturbed_pv = pv.copy()
            perturbed_load[k:] = load[k:] * 2 + 1234
            perturbed_pv[k:] = pv[k:] * 0.1
            a = build_archive(m, load, pv, target)
            b = build_archive(m, perturbed_load, perturbed_pv, target)
            initial = float(warm[k])
            q1, _, info1, _, r1, _, _ = a.plan_day(k, 0.8, 28, price, initial)
            q2, _, info2, _, r2, _, _ = b.plan_day(k, 0.8, 28, price, initial)
            row = dict(target_kWh=target, day_index=k,
                       day=str(m.bm.DATES[k].date()),
                       forecast_error_kWh=float(np.max(np.abs(a.n_forecast[k] - b.n_forecast[k]))),
                       correction_error_kWh=float(np.max(np.abs(r1 - r2))),
                       plan_error_kWh=float(np.max(np.abs(q1 - q2))),
                       solver_fallback=bool(info1['fallback'] or info2['fallback']))
            row['passed'] = bool(row['forecast_error_kWh'] == 0 and row['correction_error_kWh'] == 0
                                 and row['plan_error_kWh'] == 0 and not row['solver_fallback'])
            assert row['passed'], row
            rows.append(row)
    return rows


def shared_protection_check():
    """The protection input must be identical across groups: only the terminal rule changed."""
    frames = {p['id']: pd.read_csv(OUT / p['id'] / 'dispatch.csv', low_memory=False)
              for p in POLICIES}
    base = frames['T_q80_6000']
    out = {}
    for name, frame in frames.items():
        out[name] = {c: float(np.max(np.abs(frame[c].to_numpy() - base[c].to_numpy())))
                     for c in ['net_forecast_kWh', 'residual_adjustment_kWh', 'protected_net_kWh']}
    out['all_zero'] = bool(all(v == 0.0 for name, d in out.items() if isinstance(d, dict)
                               for v in d.values()))
    return out


def contrasts(summary, baseline='T_q80_6000'):
    frame = summary.set_index('strategy_id')
    reference = frame.loc[baseline]
    rows = []
    for sid in frame.index:
        if sid == baseline:
            continue
        row = dict(contrast=f'{sid} minus {baseline}', treatment=sid, control=baseline)
        for column in ['planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                       'planned_kWh', 'emergency_kWh', 'unused_kWh', 'charge_kWh',
                       'discharge_kWh', 'loss_kWh', 'morning_0_10_emergency_cost_yuan',
                       'morning_0_10_emergency_kWh', 'emergency_slots', 'emergency_days',
                       'emergency_events', 'full_end_days', 'empty_slots',
                       'mean_initial_kWh', 'initial_kWh', 'final_kWh', 'adjusted_cost_yuan']:
            row['delta_' + column] = float(frame.loc[sid, column] - reference[column])
        row['delta_percent'] = 100 * row['delta_total_cost_yuan'] / reference['total_cost_yuan']
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_tables():
    totals, contrasts_rows = {}, []
    for p in POLICIES:
        daily = pd.read_csv(OUT / p['id'] / 'daily_summary.csv')
        monthly = daily.assign(month=daily.date.str[:7]).groupby('month').agg(
            planned_cost_yuan=('planned_cost_yuan', 'sum'),
            emergency_cost_yuan=('emergency_cost_yuan', 'sum'),
            total_cost_yuan=('total_cost_yuan', 'sum'),
            emergency_kWh=('emergency_kWh', 'sum'),
            unused_kWh=('unused_kWh', 'sum'))
        totals[p['id']] = monthly
        monthly.to_csv(OUT / p['id'] / 'monthly_totals.csv', encoding='utf-8-sig')
    base = totals['T_q80_6000']
    for sid, monthly in totals.items():
        frame = monthly.copy()
        frame['delta_total_vs_6000_yuan'] = monthly.total_cost_yuan - base.total_cost_yuan
        frame['delta_emergency_vs_6000_yuan'] = monthly.emergency_cost_yuan - base.emergency_cost_yuan
        frame['delta_unused_vs_6000_kWh'] = monthly.unused_kWh - base.unused_kWh
        frame.insert(0, 'strategy_id', sid)
        contrasts_rows.append(frame)
    combined = pd.concat(contrasts_rows)
    combined.to_csv(OUT / 'monthly_contrasts.csv', index=False, encoding='utf-8-sig')
    return combined


def daily_extremes():
    frames = {p['id']: pd.read_csv(OUT / p['id'] / 'daily_summary.csv') for p in POLICIES}
    base = frames['T_q80_6000'].set_index('date')
    rows = []
    for sid, frame in frames.items():
        if sid == 'T_q80_6000':
            continue
        merged = frame.set_index('date').join(
            base[['total_cost_yuan', 'emergency_cost_yuan', 'unused_kWh']], rsuffix='_6000')
        merged['delta_total'] = merged.total_cost_yuan - merged.total_cost_yuan_6000
        merged['delta_emergency'] = merged.emergency_cost_yuan - merged.emergency_cost_yuan_6000
        merged['delta_unused'] = merged.unused_kWh - merged.unused_kWh_6000
        ordered = merged.sort_values('delta_total')
        for label, part in [('worst_5', ordered.tail(5).iloc[::-1]), ('best_5', ordered.head(5))]:
            for date, row in part.iterrows():
                rows.append(dict(strategy_id=sid, group=label, date=date,
                                 delta_total_cost_yuan=float(row.delta_total),
                                 delta_emergency_cost_yuan=float(row.delta_emergency),
                                 delta_unused_kWh=float(row.delta_unused),
                                 total_cost_yuan=float(row.total_cost_yuan)))
        merged.reset_index()[['date', 'delta_total', 'delta_emergency', 'delta_unused']].to_csv(
            OUT / sid / 'daily_contrast_vs_6000.csv', index=False, encoding='utf-8-sig')
    return pd.DataFrame(rows)


def hourly_table():
    rows = []
    for p in POLICIES:
        d = pd.read_csv(OUT / p['id'] / 'dispatch.csv', low_memory=False)
        d = d.assign(hour=d.slot // 6)
        hourly = d.groupby('hour').agg(planned_kWh=('planned_kWh', 'sum'),
                                       emergency_kWh=('emergency_kWh', 'sum'),
                                       unused_kWh=('unused_kWh', 'sum'),
                                       emergency_cost_yuan=('emergency_cost_yuan', 'sum')).reset_index()
        hourly.insert(0, 'strategy_id', p['id'])
        rows.append(hourly)
    combined = pd.concat(rows)
    combined.to_csv(OUT / 'hourly.csv', index=False, encoding='utf-8-sig')
    return combined


def figures(summary, monthly, hourly, extremes):
    FIG.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': .25,
                         'axes.axisbelow': True, 'font.size': 9})
    order = [p['id'] for p in POLICIES]
    labels = ['4800', '6000', '7200']
    frame = summary.set_index('strategy_id').loc[order]

    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    x = np.arange(3)
    planned = frame.planned_cost_yuan.to_numpy() / 1e6
    emergency = frame.emergency_cost_yuan.to_numpy() / 1e6
    ax.bar(x, planned, .55, label='Planned purchase cost', color='#2e75b6')
    ax.bar(x, emergency, .55, bottom=planned, label='Emergency cost (5x)', color='#c00000')
    for i in range(3):
        ax.text(i, planned[i] + emergency[i] + .05, f'{planned[i] + emergency[i]:.3f}',
                ha='center', fontsize=8)
    ax.set(xticks=x, ylabel='Cost (million CNY)',
           title='Fixed nominal day-end target sensitivity: realized 334-day cost')
    ax.set_xticklabels([f'{v} kWh' for v in labels])
    ax.legend(loc='lower right')
    fig.savefig(FIG / 'fig1_terminal_target_costs.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2), layout='constrained')
    pivot = monthly.pivot_table(index='month', columns='strategy_id',
                                values='delta_total_vs_6000_yuan')
    for sid in order:
        if sid in pivot:
            ax.plot(pivot.index, pivot[sid].to_numpy() / 1e3, marker='o', ms=3.5, lw=1.4,
                    label=sid)
    ax.axhline(0, color='black', lw=.8)
    ax.set(title='Monthly realized cost difference versus the 6000 kWh target',
           xlabel='Month of 2025', ylabel='Difference (thousand CNY)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    fig.savefig(FIG / 'fig2_monthly_contrast.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), layout='constrained')
    for ax, column, title, unit in [
            (axes[0], 'final_kWh', 'Year-end stored energy', 'kWh'),
            (axes[1], 'unused_kWh', 'Unused energy', 'kWh'),
            (axes[2], 'loss_kWh', 'Storage loss', 'kWh'),
            (axes[3], 'full_end_days', 'Days ending at the 10800 kWh cap', 'days')]:
        ax.bar(labels, frame[column].to_numpy(), color=['#c00000', '#2e75b6', '#7f6000'])
        ax.set(title=title, ylabel=unit)
        ax.tick_params(axis='x', rotation=0)
    fig.savefig(FIG / 'fig3_endpoint_and_waste.png')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), layout='constrained')
    for sid in order:
        block = hourly[hourly.strategy_id == sid]
        axes[0].plot(block.hour, block.emergency_cost_yuan / 1e3, marker='o', ms=3, lw=1.3,
                     label=sid)
        axes[1].plot(block.hour, block.emergency_kWh, marker='o', ms=3, lw=1.3, label=sid)
    axes[0].axvspan(0, 10, color='gray', alpha=.12)
    axes[0].text(.3, axes[0].get_ylim()[1] * .9, '0-10 h', fontsize=8, color='gray')
    axes[0].set(ylabel='Emergency cost (thousand CNY)',
                title='Emergency purchase by hour of day (the shaded band is the 0-10 h morning window)')
    axes[1].set(ylabel='Emergency energy (kWh)', xlabel='Hour of day', xticks=range(24))
    for ax in axes:
        ax.legend()
    fig.savefig(FIG / 'fig4_hourly_emergency.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for ax, sid in zip(axes, ['T_q80_4800', 'T_q80_7200']):
        block = extremes[extremes.strategy_id == sid]
        worst = block[block.group == 'worst_5']
        best = block[block.group == 'best_5']
        ax.barh(worst.date, worst.delta_total_cost_yuan, color='#c00000', label='worst 5 days')
        ax.barh(best.date, best.delta_total_cost_yuan, color='#2e75b6', label='best 5 days')
        ax.axvline(0, color='black', lw=.8)
        ax.set(title=f'{sid} vs 6000: extreme days', xlabel='Daily cost difference (CNY)')
        ax.tick_params(axis='y', labelsize=7)
        ax.legend()
    fig.savefig(FIG / 'fig5_worst_dates.png')
    plt.close(fig)
    return sorted(p.name for p in FIG.glob('*.png'))


def archive_sources(hashes):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for rel in ['code/12_q2_terminal_sensitivity.py']:
        shutil.copy2(ROOT / rel, ARCHIVE / Path(rel).name)
    for rel in ['code/02_q1_baseline.py', 'code/05_q2_baseline.py',
                'code/08_q2_quantile_experiment.py']:
        shutil.copy2(SNAP / rel, ARCHIVE / ('snapshot_' + Path(rel).name))
    save(ARCHIVE / 'source_sha256.json', hashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['register', 'run', 'all'], default='all')
    args = parser.parse_args()
    assert Path(sys.prefix).name == 'math_modeling', sys.prefix
    np.random.seed(20260910)

    registration = stage_register()
    signature = registration['signature']
    if args.stage == 'register':
        return
    assert_registered(signature)

    started = time.perf_counter()
    m = module()
    load, pv, price, source_hashes = m.bm.read_sources()
    warm, warm_frame, warm_end, warm_checks = m.run_warmup(m.Archive(load, pv, len(load)),
                                                          price, m.bm.DATES)
    assert abs(warm_end - COMMON_INITIAL) < 1e-6, warm_end
    warm_frame.to_csv(OUT / 'warmup_january.csv', index=False)
    nu = float(np.median(price) / 0.9)
    archive_sources(registration['input_sha256'])
    print(f'common 2025-02-01 state = {warm_end:.12f} kWh (reference {COMMON_INITIAL})', flush=True)

    results, reused = {}, {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(run_one, p, load, pv, price, warm, signature): p['id']
                for p in POLICIES}
        for job in as_completed(jobs):
            result, was_reused = job.result()
            sid = jobs[job]
            results[sid] = result
            reused[sid] = was_reused
            print(f'    {sid}: total={result["totals"]["total_cost_yuan"]:.6f} '
                  f'({result["totals"]["solver_seconds"]:.1f}s solver)', flush=True)

    reproduction = compare_segments('T_q80_6000')
    save(OUT / 'reproduction_vs_existing.json', reproduction)
    print(f'reproduction of T_q80_6000: total={reproduction["total_cost_new_yuan"]:.9f} '
          f'license={LICENSE_TOTAL:.9f} diff={reproduction["license_total_abs_diff_yuan"]:.3e} '
          f'ok={reproduction["license_total_within_tolerance"]}', flush=True)
    if not (reproduction['license_total_within_tolerance']
            and reproduction['energy_within_tolerance']
            and reproduction['cost_within_tolerance']):
        save(OUT / 'run_manifest.json', dict(status='aborted_reproduction_failed',
                                             reproduction=reproduction))
        raise SystemExit('6000 group did not reproduce; new-group conclusions withheld.')

    independent_rows = [independent(p['id'], p['terminal']) for p in POLICIES]
    summary = pd.DataFrame(independent_rows)
    summary['adjusted_cost_yuan'] = summary.total_cost_yuan - nu * (
        summary.final_kWh - summary.initial_kWh)
    summary['nu_yuan_per_kWh'] = nu
    summary.to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig')

    contrast_frame = contrasts(summary)
    contrast_frame.to_csv(OUT / 'contrasts.csv', index=False, encoding='utf-8-sig')
    monthly = monthly_tables()
    extremes = daily_extremes()
    extremes.to_csv(OUT / 'extreme_dates.csv', index=False, encoding='utf-8-sig')
    hourly = hourly_table()
    shared = shared_protection_check()
    save(OUT / 'shared_protection_check.json', shared)
    perturbation = causal_checks(load, pv, price, warm)
    save(OUT / 'future_perturbation_checks.json', perturbation)

    checks = dict(
        max_physical_violation_kWh=max(max(r['physical'][k] for k in
                                          ['balance_kWh', 'state_kWh', 'bounds_kWh', 'power_kWh',
                                           'nonnegative_kWh', 'mutex_kWh'])
                                      for r in independent_rows),
        max_day_continuity_kWh=max(r['physical']['continuity_kWh'] for r in independent_rows),
        max_nominal_terminal_error_kWh=max(r['physical']['nominal_terminal_kWh']
                                           for r in independent_rows),
        max_initial_offset_kWh=max(r['physical']['initial_state_kWh'] for r in independent_rows),
        max_accounting_error=max(max(r['accounting'].values()) for r in independent_rows),
        max_energy_identity_kWh=max(r['accounting']['energy_identity_kWh']
                                    for r in independent_rows),
        segments_all_48096=bool((summary.segments == 48096).all()),
        days_all_334=bool((summary.days == 334).all()),
        protection_identical_across_groups=shared['all_zero'],
        future_perturbation_all_passed=bool(all(r['passed'] for r in perturbation)),
        solver_failures_total=int(sum(r['totals']['solver_failures'] for r in results.values())),
        fallback_days_total=int(sum(r['totals']['fallback_days'] for r in results.values())),
        max_mip_gap=max(r['totals']['max_mip_gap'] for r in results.values()),
    )
    checks['physical_within_tolerance'] = bool(checks['max_physical_violation_kWh'] <= 1e-6)
    checks['accounting_within_tolerance'] = bool(checks['max_accounting_error'] <= 1e-4)
    checks['continuity_within_tolerance'] = bool(checks['max_day_continuity_kWh'] <= 1e-9)
    checks['nominal_terminal_within_tolerance'] = bool(
        checks['max_nominal_terminal_error_kWh'] <= 1e-6)
    save(OUT / 'validation.json', dict(
        thresholds=dict(physical_kWh=1e-6, accounting=1e-4, continuity_kWh=1e-9,
                        nominal_terminal_kWh=1e-6, cost_yuan=LICENSE_COST_TOL,
                        energy_kWh=LICENSE_ENERGY_TOL),
        reproduction=reproduction, checks=checks,
        warmup=dict(state_kWh=warm_end, checks={k: float(v) for k, v in warm_checks.items()}),
        per_group_physical={r['strategy_id']: r['physical'] for r in independent_rows},
        per_group_accounting={r['strategy_id']: r['accounting'] for r in independent_rows},
        future_perturbation=perturbation, shared_protection=shared))

    figure_names = figures(summary, monthly, hourly, extremes)
    protected_hashes = json.loads((ROOT / 'results/q2_revision_audit_20260911/'
                                   'protected_before.json').read_text(encoding='utf-8'))
    protected_now = {rel: digest(ROOT / rel) for rel in
                     ['附件/附件1.xlsx', '附件/附件2.xlsx']}
    save(OUT / 'run_manifest.json', dict(
        status='complete',
        registered_utc=registration['registered_utc'],
        finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter() - started,
        signature=signature, executable=sys.executable, python=sys.version,
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        candidates_kWh=list(CANDIDATES), policies=POLICIES,
        reproduction=reproduction, validation=checks, figures=figure_names,
        reused_completed={k: bool(v) for k, v in reused.items()},
        input_sha256=registration['input_sha256'],
        source_archive=[p.name for p in sorted(ARCHIVE.iterdir())],
        attachment_sha256_now=protected_now,
        note='Only the nominal day-end target changes. 4800/6000/7200 are pre-registered '
             'sensitivity candidates, not optima. No within-policy daily checkpointing.'))
    print(summary[['strategy_id', 'planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                   'final_kWh', 'unused_kWh', 'loss_kWh',
                   'morning_0_10_emergency_cost_yuan']].to_string(index=False), flush=True)
    print(f'wall seconds = {time.perf_counter() - started:.1f}', flush=True)


if __name__ == '__main__':
    main()
