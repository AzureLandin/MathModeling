"""Read saved Q3 artifacts only. No training or optimization. Use math_modeling."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/q3_rolling_baseline'
OUT = ROOT / 'results/q3_baseline_report_review'


def main():
    assert Path(sys.prefix).name == 'math_modeling'
    result = {'scope': 'Read-only saved ledgers; no solver, training or upstream audit', 'groups': {}}
    days = {}
    for group in ('B0', 'B1', 'B2'):
        f = pd.read_csv(SOURCE / f'{group}_dispatch.csv', parse_dates=['interval_start'])
        a = f.iloc[:-1]
        p,q,q0,e = (f[c] for c in ('price_yuan_kWh','q_eff_kWh','q0_kWh','emergency_kWh'))
        cost = p*q + .5*p*abs(q-q0) + 5*p*e
        balance = q + f.discharge_kWh + e - f.net_kWh - f.charge_kWh - f.unused_kWh
        state_error = f.state_end_kWh-f.state_start_kWh-.9*f.charge_kWh+f.discharge_kWh/.9
        continuity = f.state_start_kWh.to_numpy()[1:]-f.state_end_kWh.to_numpy()[:-1]
        assert len(f)==48097 and not f.interval_start.duplicated().any()
        assert f.interval_start.diff().dropna().eq(pd.Timedelta(minutes=10)).all()
        assert abs(balance).max()<1e-6 and abs(state_error).max()<1e-6 and abs(continuity).max()<1e-6
        assert abs(cost-f.total_cost_yuan).max()<1e-4
        assert f.state_end_kWh.between(1200-1e-6,10800+1e-6).all()
        # Independent greedy replay from each saved actual initial state.
        surplus=q-f.net_kWh
        c=np.minimum.reduce([surplus.clip(lower=0).to_numpy(),np.full(len(f),5000/6),
                             ((10800-f.state_start_kWh)/.9).clip(lower=0).to_numpy()])
        d=np.minimum.reduce([(-surplus).clip(lower=0).to_numpy(),np.full(len(f),5000/6),
                             (.9*(f.state_start_kWh-1200)).clip(lower=0).to_numpy()])
        assert np.max(abs(c-f.charge_kWh))<1e-6 and np.max(abs(d-f.discharge_kWh))<1e-6
        result['groups'][group] = dict(total=float(cost.iloc[:-1].sum()),
            emergency_kWh=float(a.emergency_kWh.sum()), emergency_intervals=int((a.emergency_kWh>1e-6).sum()),
            initial=float(f.state_start_kWh.iloc[0]), final=float(a.state_end_kWh.iloc[-1]),
            cash_residual=float(abs(cost-f.total_cost_yuan).max()),energy_residual=float(abs(balance).max()))
        days[group]=a.groupby(a.interval_start.dt.strftime('%Y-%m-%d')).total_cost_yuan.sum()
    decisions=pd.read_csv(SOURCE/'B2_revision_decisions.csv')
    assert np.array_equal(decisions.accepted,decisions.new_predicted_total_yuan<decisions.old_predicted_total_yuan-1e-4)
    result['updates']=dict(count=len(decisions),accepted=int(decisions.accepted.sum()),
        rejected_strictly_more_expensive=int(((~decisions.accepted)&(decisions.delta_predicted_yuan>1e-4)).sum()),
        rejected_near_equal=int(((~decisions.accepted)&(abs(decisions.delta_predicted_yuan)<=1e-4)).sum()))
    for newer,older in [('B1','B0'),('B2','B1'),('B2','B0')]:
        delta=days[newer]-days[older]
        months=delta.groupby(delta.index.str[:7]).sum()
        result[f'{newer}-{older}']=dict(delta=float(delta.sum()),cheaper_days=int((delta<-1e-4).sum()),
            cheaper_months=int((months<-1e-4).sum()))
    b2=pd.read_csv(SOURCE/'B2_dispatch.csv',parse_dates=['interval_start'])
    onday=b2[b2.interval_start.dt.strftime('%Y-%m-%d')=='2025-06-21']
    selected=onday.iloc[35:]
    result['A2_saved_slice_problem']=dict(first_target=str(selected.interval_start.iloc[0]),
        last_target=str(selected.interval_start.iloc[-1]), count=len(selected),
        final_effective_differs_from_original_intervals=int((abs(selected.q_eff_kWh-selected.q0_kWh)>1e-6).sum()),
        expected_first='2025-06-21 06:00:00',expected_last='2025-06-22 00:00:00')
    result['status']='saved_cash_physics_feedback_and_selection_checks_passed'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
