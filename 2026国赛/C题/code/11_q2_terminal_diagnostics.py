"""Independently audit and summarize the registered Q2 terminal-target experiment.

Inputs: completed 10 outputs and its frozen attachment 1/2 sources.
Outputs: results/q2_terminal_diagnostics_20260911, figures of the same name,
and reports/问题二_终端目标对照实验结果.md. Does not change dispatch outputs.
Run: conda run -n math_modeling python code/11_q2_terminal_diagnostics.py
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/q2_terminal_experiment_20260911'
OUT = ROOT / 'results/q2_terminal_diagnostics_20260911'
FIG = ROOT / 'figures/q2_terminal_diagnostics_20260911'
REPORT = ROOT / 'reports/问题二_终端目标对照实验结果.md'
IDS = ['A_base', 'B_fixed_q80', 'T_base_6000', 'T_q80_6000', 'B_adaptive']
PAIRS = [('no_correction', 'T_base_6000', 'A_base'),
         ('q80', 'T_q80_6000', 'B_fixed_q80')]
DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
DT, ETA, CAP, EMIN, EMAX = 1 / 6, .9, 5000 / 6, 1200., 10800.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def table(frame, columns):
    lines = ['| ' + ' | '.join(label for _, label in columns) + ' |',
             '|' + '|'.join('---' for _ in columns) + '|']
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f'{value:,.2f}' if isinstance(value, (float, np.floating)) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def audit_policy(sid, source_l, source_v, price, initial):
    frame = pd.read_csv(RUN / sid / 'dispatch.csv', low_memory=False)
    raw_daily = pd.read_csv(RUN / sid / 'daily_summary.csv')
    raw_month = pd.read_csv(RUN / sid / 'monthly_summary.csv')
    events = pd.read_csv(RUN / sid / 'emergency_events.csv')
    assert len(frame) == 334 * 144
    expected = pd.date_range('2025-02-01', periods=334 * 144, freq='10min')
    assert pd.DatetimeIndex(frame.start_time).equals(expected)
    assert pd.DatetimeIndex(frame.end_time).equals(expected + pd.Timedelta(minutes=10))
    assert pd.DatetimeIndex(frame.issue_time).equals(expected.normalize())
    assert np.array_equal(frame.slot, np.tile(np.arange(144), 334))
    assert np.array_equal(frame.date, expected.strftime('%Y-%m-%d'))
    assert np.allclose(frame.load_kW, source_l[31:].ravel(), rtol=0, atol=1e-9)
    assert np.allclose(frame.pv_kW, source_v[31:].ravel(), rtol=0, atol=1e-9)
    assert np.array_equal(frame.price_yuan_kWh, np.tile(price, 334))
    q, c, d, e, w, s0, s1 = [frame[col].to_numpy() for col in [
        'planned_kWh', 'charge_kWh', 'discharge_kWh', 'emergency_kWh', 'unused_kWh',
        'state_start_kWh', 'state_end_kWh']]
    assert np.isfinite(np.column_stack([q, c, d, e, w, s0, s1])).all()
    loss = (1 - ETA) * c + (1 / ETA - 1) * d
    frame['loss_kWh'] = loss
    frame['planned_fee'] = frame.price_yuan_kWh * q
    frame['emergency_fee'] = 5 * frame.price_yuan_kWh * e
    frame['total_fee'] = frame.planned_fee + frame.emergency_fee
    frame['morning_fee'] = np.where(frame.slot < 60, frame.emergency_fee, 0.)
    checks = {
        'balance_kWh': float(np.max(np.abs(q + frame.pv_kW * DT + d + e - frame.load_kW * DT - c - w))),
        'state_kWh': float(np.max(np.abs(s1 - s0 - ETA * c + d / ETA))),
        'continuity_kWh': float(np.max(np.abs(s0[1:] - s1[:-1]))),
        'initial_kWh': float(abs(s0[0] - initial)),
        'bounds_kWh': float(max(0, EMIN - min(s0.min(), s1.min()), max(s0.max(), s1.max()) - EMAX)),
        'power_kWh': float(max(0, c.max() - CAP, d.max() - CAP)),
        'nonnegative_kWh': float(max(0, -min(q.min(), c.min(), d.min(), e.min(), w.min()))),
        'mutex_kWh': float(np.maximum(0, np.minimum(c, d)).max()),
    }
    nominal = frame.groupby('date').agg(initial=('state_start_kWh', 'first'),
                                        terminal=('nominal_state_end_kWh', 'last'))
    target = 6000. if sid.startswith('T_') else nominal.initial
    checks['nominal_target_kWh'] = float(np.abs(nominal.terminal - target).max())
    assert max(checks.values()) < 1e-6, (sid, checks)
    daily = frame.groupby('date', sort=True).agg(
        initial_kWh=('state_start_kWh', 'first'), final_kWh=('state_end_kWh', 'last'),
        planned_cost_yuan=('planned_fee', 'sum'), emergency_cost_yuan=('emergency_fee', 'sum'),
        total_cost_yuan=('total_fee', 'sum'), planned_kWh=('planned_kWh', 'sum'),
        emergency_kWh=('emergency_kWh', 'sum'), unused_kWh=('unused_kWh', 'sum'),
        charge_kWh=('charge_kWh', 'sum'), discharge_kWh=('discharge_kWh', 'sum'),
        loss_kWh=('loss_kWh', 'sum'), morning_emergency_cost_yuan=('morning_fee', 'sum'))
    daily['full_end'] = (daily.final_kWh >= EMAX - 1e-6).astype(int)
    aggregation = {col: 'sum' for col in daily}
    aggregation.update(initial_kWh='first', final_kWh='last')
    monthly = daily.groupby(daily.index.str[:7]).agg(aggregation)
    accounting = {}
    for col in ['planned_cost_yuan', 'emergency_cost_yuan', 'total_cost_yuan',
                'planned_kWh', 'emergency_kWh', 'unused_kWh', 'charge_kWh', 'discharge_kWh']:
        accounting['daily_' + col] = float(np.abs(daily[col] - raw_daily.set_index('date')[col]).max())
        accounting['monthly_' + col] = float(np.abs(monthly[col] - raw_month.set_index('month')[col]).max())
    accounting['event_energy_kWh'] = abs(float(events.emergency_kWh.sum() - e.sum()))
    accounting['event_cost_yuan'] = abs(float(events.emergency_cost_yuan.sum() - frame.emergency_fee.sum()))
    accounting['energy_identity_kWh'] = abs(float(np.sum(q + e + (frame.pv_kW - frame.load_kW) * DT - w - loss) - (s1[-1] - s0[0])))
    assert max(accounting.values()) < 1e-4, (sid, accounting)
    totals = {col: float(daily[col].sum()) for col in daily if col not in ['initial_kWh', 'final_kWh', 'full_end']}
    nu = float(np.median(price) / ETA)
    totals.update(strategy_id=sid, full_end_days=int(daily.full_end.sum()),
                  full_slots=int((s1 >= EMAX - 1e-6).sum()), empty_slots=int((s1 <= EMIN + 1e-6).sum()),
                  mean_initial_kWh=float(daily.initial_kWh.mean()), initial_kWh=float(s0[0]), final_kWh=float(s1[-1]),
                  adjusted_cost_yuan=float(frame.total_fee.sum() - nu * (s1[-1] - s0[0])),
                  emergency_slots=int((e > 1e-6).sum()), emergency_events=len(events),
                  emergency_days=int((daily.emergency_kWh > 1e-6).sum()))
    # Associate each evaluated midnight with the following 00:00-10:00, excluding Dec 31's unknown continuation.
    overnight = pd.DataFrame({'date': daily.index[1:], 'previous_final_kWh': daily.final_kWh.to_numpy()[:-1],
                              'initial_kWh': daily.initial_kWh.to_numpy()[1:],
                              'morning_emergency_cost_yuan': daily.morning_emergency_cost_yuan.to_numpy()[1:]})
    assert np.max(np.abs(overnight.previous_final_kWh - overnight.initial_kWh)) < 1e-6
    overnight.insert(0, 'strategy_id', sid)
    return frame, daily, monthly, totals, overnight, dict(physical=checks, accounting=accounting)


def figures(summary, monthly, daily):
    colors = ['#20735d', '#b64943']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fixed = summary.set_index('strategy_id').loc[IDS[:4]]
    labels = ['Base / cycle', 'q80 / cycle', 'Base / 6000', 'q80 / 6000']
    axes[0].bar(labels, fixed.planned_cost_yuan / 1e4, color=colors[0], label='Planned')
    axes[0].bar(labels, fixed.emergency_cost_yuan / 1e4, bottom=fixed.planned_cost_yuan / 1e4,
                color=colors[1], label='Emergency')
    axes[0].set_ylabel('Cost (10,000 yuan)')
    axes[0].set_title('Actual cost, Feb-Dec 2025', pad=32)
    axes[0].tick_params(axis='x', labelrotation=20)
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    for (key, _, _), color in zip(PAIRS, colors):
        part = monthly[monthly.contrast == key]
        axes[1].plot(part.month.str[5:], part.delta_total_cost_yuan / 1e4, marker='o', color=color,
                     label='No correction' if key == 'no_correction' else 'q80 correction')
    axes[1].axhline(0, color='#777777', linewidth=.8)
    axes[1].set(xlabel='Month (2025)', ylabel='6000 minus cycle (10,000 yuan)', title='Matched terminal effect')
    axes[1].legend()
    fig.savefig(FIG / 'cost_and_monthly.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for ax, (_, treated, control) in zip(axes, PAIRS):
        for sid, color in zip([control, treated], colors):
            ax.plot(pd.to_datetime(daily[sid].index), daily[sid].final_kWh, linewidth=1, color=color, label=sid)
        ax.axhline(6000, color='#777777', linewidth=.7, linestyle='--', label='Nominal target: 6000 kWh')
        ax.set(ylabel='Actual day-end energy (kWh)', ylim=(1000, 11200))
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Actual terminal states are carried forward, not reset')
    axes[1].set_xlabel('Date (2025)')
    fig.savefig(FIG / 'actual_terminal_states.png', dpi=180)
    plt.close(fig)


def terminal_boundaries(experiment, frozen):
    zeros = np.zeros((2, 144))
    planner = experiment.archive(frozen, zeros, zeros, '6000')
    rows = []
    for initial in [EMIN, EMAX]:
        q, nominal, info = planner.solve_plan(np.zeros(144), np.ones(144), initial)
        c, d, e, w, states, checks = frozen.bm.control(q, zeros[0], zeros[0], initial)
        expected_purchase = max(0., 6000. - initial) / ETA
        assert abs(nominal[-1] - 6000.) < 1e-6
        assert abs(q.sum() - expected_purchase) < 1e-6
        assert abs(states[-1] - max(initial, 6000.)) < 1e-6
        assert np.max(e) < 1e-6 and max(checks.values()) < 1e-6
        rows.append(dict(initial_kWh=initial, nominal_terminal_kWh=float(nominal[-1]),
                         actual_terminal_kWh=float(states[-1]), planned_kWh=float(q.sum()),
                         expected_planned_kWh=expected_purchase, passed=True))
    return rows


def main():
    assert Path(sys.prefix).name == 'math_modeling'
    manifest = read_json(RUN / 'run_manifest.json')
    assert manifest['status'] == 'complete', 'Run 10 must finish before diagnostics.'
    for rel, expected_hash in manifest['result_sha256'].items():
        assert sha(RUN / rel) == expected_hash, rel
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    source_archive = OUT / 'source_snapshot'
    source_archive.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), source_archive / Path(__file__).name)
    spec = importlib.util.spec_from_file_location('terminal_run', ROOT / 'code/10_q2_terminal_experiment.py')
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    frozen = experiment.module()
    load, pv, price, _ = frozen.bm.read_sources()
    boundaries = terminal_boundaries(experiment, frozen)
    initial = read_json(RUN / 'config.json')['evaluation_initial_kWh']
    frames, daily, months, totals, checks, overnight = {}, {}, {}, [], {}, []
    for sid in IDS:
        frames[sid], daily[sid], months[sid], row, night, checks[sid] = audit_policy(sid, load, pv, price, initial)
        totals.append(row)
        overnight.append(night)
    summary = pd.DataFrame(totals)
    reference = pd.read_csv(RUN / 'comparison.csv').set_index('strategy_id')
    for row in totals:
        assert abs(row['total_cost_yuan'] - reference.loc[row['strategy_id'], 'total_cost_yuan']) < 1e-4
    shared = ['load_forecast_kW', 'pv_forecast_kW', 'net_forecast_kWh']
    for sid in IDS:
        assert np.max(np.abs(frames[sid][shared] - frames['A_base'][shared]).to_numpy()) < 1e-9
    for _, treatment, control in PAIRS:
        assert np.array_equal(frames[treatment].residual_adjustment_kWh, frames[control].residual_adjustment_kWh)
    comparisons, monthly_rows, daily_rows = [], [], []
    numeric = [key for key in summary.columns if key != 'strategy_id']
    indexed = summary.set_index('strategy_id')
    for key, treatment, control in PAIRS:
        row = {'contrast': key, 'treatment': treatment, 'control': control}
        row.update({'delta_' + col: float(indexed.loc[treatment, col] - indexed.loc[control, col]) for col in numeric})
        row['delta_percent'] = 100 * row['delta_total_cost_yuan'] / indexed.loc[control, 'total_cost_yuan']
        row['improved_months'] = int((months[treatment].total_cost_yuan < months[control].total_cost_yuan - 1e-4).sum())
        row['improved_days'] = int((daily[treatment].total_cost_yuan < daily[control].total_cost_yuan - 1e-4).sum())
        comparisons.append(row)
        for label, collection, rows in [('month', months, monthly_rows), ('date', daily, daily_rows)]:
            delta = collection[treatment].subtract(collection[control])
            for date, values in delta.iterrows():
                rows.append({'contrast': key, label: date,
                             **{'delta_' + col: float(values[col]) for col in ['total_cost_yuan', 'planned_cost_yuan',
                                 'emergency_cost_yuan', 'unused_kWh', 'morning_emergency_cost_yuan']}})
    contrasts = pd.DataFrame(comparisons)
    monthly = pd.DataFrame(monthly_rows)
    daily_delta = pd.DataFrame(daily_rows)
    interaction = {col: float(contrasts.iloc[1][col] - contrasts.iloc[0][col])
                   for col in contrasts if col.startswith('delta_') and col != 'delta_percent'}
    # Complete monthly fees are recomputed here; raw per-policy delta_vs_A fields from 10 are placeholders.
    pd.concat([value.assign(strategy_id=sid).rename_axis('month').reset_index()
               for sid, value in months.items()]).to_csv(OUT / 'monthly_totals.csv', index=False)
    summary.to_csv(OUT / 'summary.csv', index=False)
    contrasts.to_csv(OUT / 'contrasts.csv', index=False)
    monthly.to_csv(OUT / 'monthly_contrasts.csv', index=False)
    daily_delta.to_csv(OUT / 'daily_contrasts.csv', index=False)
    pd.concat(overnight).to_csv(OUT / 'overnight_continuity.csv', index=False)
    extreme = pd.concat([part.nsmallest(5, 'delta_total_cost_yuan').assign(extreme='best')
                         for _, part in daily_delta.groupby('contrast')] +
                        [part.nlargest(5, 'delta_total_cost_yuan').assign(extreme='worst')
                         for _, part in daily_delta.groupby('contrast')])
    extreme.to_csv(OUT / 'extreme_dates.csv', index=False)
    selected = pd.concat([value.loc[DATES].assign(strategy_id=sid).rename_axis('date').reset_index()
                          for sid, value in daily.items()])
    selected.to_csv(OUT / 'selected_dates_summary.csv', index=False)
    dispatch = pd.concat([frame[frame.date.isin(DATES)] for frame in frames.values()])
    dispatch.to_csv(OUT / 'selected_dates_dispatch.csv', index=False)
    dispatch['block'] = dispatch.slot // 24
    storage = dispatch.groupby(['strategy_id', 'date', 'block']).agg(
        charge_kWh=('charge_kWh', 'sum'), discharge_kWh=('discharge_kWh', 'sum'),
        initial_kWh=('state_start_kWh', 'first'), final_kWh=('state_end_kWh', 'last')).reset_index()
    storage.to_csv(OUT / 'selected_dates_storage_4h.csv', index=False)
    fixed_q80 = contrasts[contrasts.contrast == 'q80'].iloc[0]
    saving = float(indexed.loc['A_base', 'total_cost_yuan'] - indexed.loc['T_q80_6000', 'total_cost_yuan'])
    saving_percent = 100 * saving / indexed.loc['A_base', 'total_cost_yuan']
    limit_rows = pd.read_csv(RUN / 'limits.csv')
    reproduction = read_json(ROOT / 'results/q2_revision_audit_20260911/reproduction.json')
    causal = read_json(RUN / 'future_perturbation_checks.json')
    figures(summary, monthly, daily)
    overall_max = max(max(c['physical'].values()) for c in checks.values())
    account_max = max(max(c['accounting'].values()) for c in checks.values())
    lines = ['# 问题二：终端目标对照实验结果', '',
             '日期：2026-09-11。状态：登记的2×2对照已完成；本报告由11脚本读取冻结输出并独立核算生成。', '',
             '## 1. 问题分析与结论', '',
             f'固定q80时，将名义末态从当日实际初态改为6000 kWh，2—12月现金总费用变化{fixed_q80.delta_total_cost_yuan:,.2f}元'
             f'（{fixed_q80.delta_percent:+.4f}%），11个月中{int(fixed_q80.improved_months)}个月费用降低。'
             '负值表示节费；这不是对其他末态目标或其他控制器的最优性证明。', '',
             '实验只改名义终端约束，公共1月、点预测、历史修正、MILP内核、实际贪心反馈和收费规则保持一致。'
             '不同策略2月以后各自继承真实状态；这些状态差是终端规则作用的后果。', '',
             f'T_q80_6000相对原无保护Baseline总共节费{saving:,.2f}元（{saving_percent:.4f}%）。'
             '其中保护与终端同时变化，不能把这一全部收益归为终端调整；终端自身的匹配收益为前述17.93万元。', '',
             '## 2. 数据预处理与核验口径', '',
             '直接对照冻结附件1电价、附件2实际功率；时间为前10分钟代表功率，功率乘1/6小时转换电量。'
             '不平滑、不删负净需求。每组334天、48096段，2月1日共同初态8801.462273333342 kWh。'
             '0—10时定义为slot 0—59；次晨衔接单独保存2月2日至12月31日共333对午夜，末日以后没有真值，不补造延续。', '',
             '## 3. 模型与求解', '',
             r'$q$为冻结普通购电量；本节的$c,d,w,\bar E,z$分别为名义母线充电、放电、未使用电量、内部储电和二进制充电模式，'
             r'$\varepsilon$为实际净需求减当时预测的残差。实际轨迹另外用当天观测与反馈计算，其中$e$为应急购电量。', '',
             r'日前目标为$\min\sum_t p_tq_t$；历史校准需求$\widetilde n_{k,t}=\widehat n_{k,t}+Q_{0.8}(\varepsilon_{j,t}:k-28\le j<k)$。'
             'q80使用排序后第ceil(0.8m)个值，有效历史不足7日回退零修正；无修正组直接使用点预测。负修正不截断。', '',
             r'约束为$q_t+d_t=\widetilde n_{k,t}+c_t+w_t$、$\bar E_{t+1}=\bar E_t+0.9c_t-d_t/0.9$，'
             r'$0\le c_t\le(5000/6)z_t$、$0\le d_t\le(5000/6)(1-z_t)$、$z_t\in\{0,1\}$、'
             r'$1200\le\bar E_t\le10800$；$q_t,w_t\ge0$。初态为当日已知实际储电量；末态分别为当日实际初态或6000 kWh。', '',
             '全部电量单位kWh，电价元/kWh；实际反馈按已观测当期净供需充放电，应急购电按5倍价计费。'
             '普通购电全部付款，未使用电量无额外罚金；实际末态不强制达到名义目标。', '',
             '环境math_modeling，SciPy/HiGHS的scipy.optimize.milp，分支割框架；每次名义规划相对gap目标1e-9、时限120秒。'
             '无随机采样或全年参数搜索，达到求解器最优性条件后停止。'
             f'五组实际最大名义MIP gap为{reference.max_mip_gap.max():.3e}，失败{int(reference.solver_failures.sum())}次、回退{int(reference.fallback_days.sum())}天。'
             '这只证明名义MILP求解精度，不能证明随机控制全局最优。', '',
             '## 4. 求解结果', '',
             table(summary, [('strategy_id', '策略'), ('planned_cost_yuan', '计划费/元'),
                             ('emergency_cost_yuan', '应急费/元'), ('total_cost_yuan', '总费/元'),
                             ('unused_kWh', '未使用电量/kWh'), ('full_end_days', '日末满电天数')]), '',
             'B_adaptive仅用于旧行为复现，不属于2×2因子实验；不能将它和T_q80_6000的差额归为纯终端效应。', '',
             table(contrasts, [('contrast', '保护条件'), ('delta_planned_cost_yuan', '计划费差/元'),
                               ('delta_emergency_cost_yuan', '应急费差/元'), ('delta_total_cost_yuan', '总费差/元'),
                               ('delta_percent', '总费变化/%'), ('improved_months', '改善月数')]), '',
             f"差中差为{interaction['delta_total_cost_yuan']:,.2f}元，即(q80下终端效应)减(无修正下终端效应)。"
             '这是相同数据环境下的交互描述，不是随机试验的总体统计推断。', '',
             '### 4.1 全部月份', '',
             table(monthly.pivot(index='month', columns='contrast', values='delta_total_cost_yuan').reset_index(),
                   [('month', '月份'), ('no_correction', '无修正：6000减cycle/元'), ('q80', 'q80：6000减cycle/元')]), '',
             '### 4.2 存量、次晨费用与损耗', '',
             table(summary, [('strategy_id', '策略'), ('mean_initial_kWh', '平均日初/kWh'), ('final_kWh', '年末/kWh'),
                             ('morning_emergency_cost_yuan', '0—10时应急费/元'), ('loss_kWh', '损耗/kWh'),
                             ('adjusted_cost_yuan', '存量估值后费用/元')]), '',
             r'存量估值仅作诊断：$C_{adj}=C-\nu(E_T-E_0)$，$\nu=\operatorname{median}(p)/0.9=0.89705556$元/kWh。'
             '它不是题目应付费用或已知真实价值函数；不能用年末估值替代每日终端规则的运行实验。', '',
             f'q80匹配组的平均日初电量降低{-fixed_q80.delta_mean_initial_kWh:,.2f} kWh，'
             f'未使用电量减少{-fixed_q80.delta_unused_kWh:,.2f} kWh，损耗减少{-fixed_q80.delta_loss_kWh:,.2f} kWh；'
             f'0—10时应急费变化{fixed_q80.delta_morning_emergency_cost_yuan:,.2f}元。'
             '这些结果支持高库存延续带来额外计划购电和循环损耗的机制，但不意味所有库存都没有风险缓冲价值。'
             f'按同一估值修正后，终端自身仍节费{-fixed_q80.delta_adjusted_cost_yuan:,.2f}元，年末余额差不足以解释主要收益。', '',
             '### 4.3 限制类型与不利日期', '',
             '限制标签来自事件时段末态和实际充放电：state_only为仅储电边界触发，power_only为仅功率触发，both为同时触发。'
             '这些是共现诊断，不能单独证明扩容或加功率的反事实收益。', '',
             table(limit_rows[limit_rows.strategy_id.isin(['B_fixed_q80', 'T_q80_6000'])],
                   [('strategy_id', '策略'), ('event', '事件'), ('limit', '触边类型'), ('slots', '时段数'),
                    ('energy_kWh', '电量/kWh'), ('emergency_cost_yuan', '应急费/元')]), '',
             table(extreme[(extreme.contrast == 'q80') & (extreme.extreme == 'worst')],
                   [('date', 'q80终端改动最差日期'), ('delta_total_cost_yuan', '总费差/元'),
                    ('delta_emergency_cost_yuan', '应急费差/元'), ('delta_unused_kWh', '未使用电量差/kWh')]), '',
             '指定3/20、6/21、9/23、12/21全部结果见selected_dates_summary.csv、selected_dates_dispatch.csv和selected_dates_storage_4h.csv。', '',
             '## 5. 验证与适用边界', '',
             f'独立物理核验最大误差{overall_max:.3e} kWh，小于1e-6；费用与累计能量核账最大数值误差{account_max:.3e}，对应单位为元或kWh，小于1e-4。'
             '共同初态、跨日连续、名义末态、负载/光伏原始输入、时段轴、费用汇总和事件汇总均通过。'
             '五组原点预测一致，两个匹配对照中的残差修正量逐段一致。', '',
             '新增无负载、无光伏、单位电价边界：1200 kWh初态时购买4800/0.9 kWh，名义和实际均到6000；'
             '10800 kWh初态时购电为0，名义末态6000而实际末态仍为10800。两例均通过，说明名义目标未被用来强制重置实际库存。'
             '以上是人工构造的程序边界测试，不是附件观测结果。', '',
             f'当前冻结版本的A_base、B_fixed_q80、B_adaptive均与旧轨迹完成逐段复现；'
             f'最大比较误差{max(max(v["max_differences"].values()) for v in reproduction.values()):.3e}。'
             f'另完成{len(causal)}个未来扰动案例，当日预测、修正、候选评分、选参和计划均未改变。'
             '案例检验与代码中的历史截断共同提供因果依据，不宣称覆盖所有程序路径。', '',
             '版本限制：原始08源文件未归档，旧清单哈希与当前代码不一致；此次复现验证当前冻结版本能重现旧行为，不能恢复未知旧源码。'
             '上一轮首轮改进文档的版本争议由本次证据补充，不重写原运行清单。', '',
             '原10脚本各策略monthly_summary.csv中delta_total_cost_yuan_vs_A未传入A基准，全部为0占位；'
             '本报告完全由费用重新相减，使用本目录monthly_totals.csv、monthly_contrasts.csv；原列不作证据。', '',
             '当前年数据此前已参与诊断，本结果属于因果滚动回测，不是独立未见测试。6000与q80为固定候选而非已证最优参数。'
             '名义MILP多解的二级择优尚未实施，因此结果是在当前固定求解器选择规则下取得的。'
             '未实施动态备用、MPC、场景优化或第三、四问。', '',
             '## 6. 当前判断与下一步', '',
             '登记的终端实验已闭环：在固定q80、原控制器与当前求解器规则下，6000名义末态在全部11个月降低费用，'
             f'334天中{int(fixed_q80.improved_days)}天更省钱，应急費未增加。可将T_q80_6000作为下一轮候选基准，'
             '保留原Baseline及B_adaptive作为已验证对照。第二问尚未锁定最终模型或正式填表。', '',
             '下一步优先在不变预测、q80和反馈控制下检验6000附近有限目标的敏感性，先固定候选集合再运行，'
             '以判断收益是否依赖单一数值；如需每日选择目标，只允许过去日期费用选参。'
             '随后单独检查名义多解的统一二级规则或储能备用，避免同时改变保护参数与控制器。'
             '更低的应急费或更少的弃余都不能代替现金总费用评价。上述后续实验尚未实施。', '',
             '## 7. 文件与复现', '',
             '- 原始运行：results/q2_terminal_experiment_20260911/；源快照与复现：results/q2_revision_audit_20260911/。',
             '- 本次独立汇总：results/q2_terminal_diagnostics_20260911/；图表：figures/q2_terminal_diagnostics_20260911/。',
             '- 复现运行：conda run -n math_modeling python code/10_q2_terminal_experiment.py。',
             '- 复现核验与本报告：conda run -n math_modeling python code/11_q2_terminal_diagnostics.py。',
             '- 附件/附件5/result2.xlsx尚未填写；第一问和旧实验输出保持原记录。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    save_json(OUT / 'validation.json', dict(policies=checks, source_results_hashes_verified=True, terminal_boundaries=boundaries,
               shared_forecasts=True, paired_corrections_equal=True, interaction=interaction,
               saving_vs_baseline_yuan=saving, saving_vs_baseline_percent=saving_percent,
               input_manifest_sha256=sha(RUN / 'run_manifest.json'), script_sha256=sha(Path(__file__)),
               artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in list(OUT.glob('*.csv')) + list(FIG.glob('*.png')) + [REPORT]}))
    print(summary[['strategy_id', 'total_cost_yuan', 'unused_kWh', 'full_end_days']].to_string(index=False))
    print(contrasts[['contrast', 'delta_total_cost_yuan', 'delta_percent', 'improved_months']].to_string(index=False))
    print('Independent audit passed. Report:', REPORT)


if __name__ == '__main__':
    main()
