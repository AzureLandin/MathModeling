"""Read saved LightGBM outputs only; no fitting, optimization or experiment imports."""
from pathlib import Path
import hashlib
import io
import json
import math
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/q2_lightgbm_residual'
OUT = ROOT / 'results/q2_lightgbm_report_review'
HASHES = {}


def read_bytes(path):
    path = Path(path)
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(data).hexdigest()
    return data


def csv(path):
    return pd.read_csv(io.BytesIO(read_bytes(path)))


def js(path):
    return json.loads(read_bytes(path))


def main():
    assert Path(sys.prefix).name == 'math_modeling'
    summary = {'scope': 'Saved-output arithmetic and archive review only; no training or MILP replay.',
               'groups': {}, 'q80_review': {}}
    reg, manifest = js(SRC/'registration.json'), js(SRC/'run_manifest.json')
    summary['registration_matches_run'] = reg['signature'] == manifest['signature']
    summary['source_matches_registration'] = hashlib.sha256(read_bytes(
        ROOT/'code/19_q2_lightgbm_residual_experiment.py')).hexdigest() == reg['code_sha256']
    summary['manifest_clock_duration_seconds'] = (
        pd.Timestamp(manifest['finished_utc'])-pd.Timestamp(manifest['started_utc'])).total_seconds()
    summary['manifest_perf_duration_seconds'] = manifest['wall_seconds']
    comparison = csv(SRC/'comparison.csv').set_index('strategy_id')
    days = {}
    for sid in comparison.index:
        d = csv(SRC/sid/'dispatch.csv')
        a = csv(SRC/sid/'forecast_residuals.csv').sort_values(['date', 'slot'])
        q, e, c, v, w = (d[n].to_numpy() for n in
                        ['planned_kWh', 'emergency_kWh', 'charge_kWh', 'discharge_kWh', 'unused_kWh'])
        p = d.price_yuan_kWh.to_numpy()
        n = (d.load_kW.to_numpy()-d.pv_kW.to_numpy())/6
        s0, s1 = d.state_start_kWh.to_numpy(), d.state_end_kWh.to_numpy()
        loss = .1*c+(1/.9-1)*v
        cost = q*p+5*e*p
        active = (e > 1e-6).reshape(-1,144)
        g = dict(rows=len(d), total_cost_yuan=float(cost.sum()),
                 planned_cost_yuan=float((q*p).sum()), emergency_cost_yuan=float((5*e*p).sum()),
                 cost_vs_summary=float(cost.sum()-comparison.loc[sid,'total_cost_yuan']),
                 balance_error_kWh=float(np.max(np.abs(q+e+v-n-c-w))),
                 state_error_kWh=float(np.max(np.abs(s1-s0-.9*c+v/.9))),
                 continuity_error_kWh=float(np.max(np.abs(s0[1:]-s1[:-1]))),
                 energy_error_kWh=float(np.sum(q+e-n-w-loss)-(s1[-1]-s0[0])),
                 emergency_slots=int(active.sum()), emergency_days=int(active.any(axis=1).sum()),
                 emergency_events=int((active & ~np.concatenate([np.zeros((len(active),1),bool),active[:,:-1]],axis=1)).sum()),
                 initial_kWh=float(s0[0]), final_kWh=float(s1[-1]),
                 planned_kWh=float(q.sum()), emergency_kWh=float(e.sum()), unused_kWh=float(w.sum()),
                 loss_kWh=float(loss.sum()), metrics={})
        for target, actual, pred in [
            ('load',d.load_kW,d.load_forecast_kW),('pv',d.pv_kW,d.pv_forecast_kW),
            ('net',d.load_kW-d.pv_kW,d.load_forecast_kW-d.pv_forecast_kW)]:
            err = actual.to_numpy()-pred.to_numpy()
            g['metrics'][target] = dict(mae=float(np.mean(abs(err))),rmse=float(np.sqrt(np.mean(err**2))))
        summary['groups'][sid] = g
        days[sid] = pd.Series(cost).groupby(d.date).sum()
        dates = a.date.drop_duplicates().to_numpy()
        actual_l, actual_pv = [a[col].to_numpy().reshape(-1,144) for col in ['load_kW','pv_kW']]
        pred_l, pred_pv = [a[col].to_numpy().reshape(-1,144) for col in ['load_forecast_kW','pv_forecast_kW']]
        missing = {}
        for name, arr in [('load',pred_l),('pv',pred_pv)]:
            bad = np.flatnonzero(~np.isfinite(arr[1:]).all(axis=1))+1
            missing[name] = [str(dates[i]) for i in bad]
        eps = (actual_l-actual_pv-pred_l+pred_pv)/6
        # Diagnostic repair of only missing issued forecasts under the task-book fallback.
        # Not a new dispatch or a claim about repaired cash costs.
        fixed_l, fixed_pv = pred_l.copy(), pred_pv.copy()
        for k in range(1,len(dates)):
            base_l = actual_l[k-7] if k>=7 else actual_l[k-1]
            fixed_l[k] = np.where(np.isfinite(fixed_l[k]),fixed_l[k],base_l)
            base_pv = actual_pv[k-1].copy()
            if k>=7:
                ix = np.flatnonzero(actual_pv[k-7:k].max(axis=0)>1)
                if len(ix):
                    lo,hi=max(0,int(ix[0])-3),min(143,int(ix[-1])+3)
                    base_pv[:lo]=0; base_pv[hi+1:]=0
            fixed_pv[k] = np.where(np.isfinite(fixed_pv[k]),fixed_pv[k],base_pv)
        fixed_eps = (actual_l-actual_pv-fixed_l+fixed_pv)/6
        rows=[]
        for k in range(31,len(dates)):
            lo=max(1,k-28); raw=eps[lo:k]; rank=math.ceil(.8*len(raw))-1
            recorded=d.loc[d.date==dates[k],'residual_adjustment_kWh'].to_numpy()
            fixed_q=np.sort(fixed_eps[lo:k],axis=0)[rank]
            raw_q=np.sort(raw,axis=0)[rank]
            if not np.isfinite(raw).all() or np.max(abs(recorded-fixed_q))>1e-6:
                rows.append(dict(date=str(dates[k]), counted_dates=len(raw),
                                 finite_dates_min=int(np.isfinite(raw).sum(axis=0).min()),
                                 finite_dates_max=int(np.isfinite(raw).sum(axis=0).max()),
                                 selected_one_based_rank=rank+1,
                                 raw_sort_matches_saved_max_error=float(np.max(abs(recorded-raw_q))),
                                 changed_slots_after_fallback=int((abs(recorded-fixed_q)>1e-6).sum()),
                                 max_q80_difference_kWh=float(np.max(abs(recorded-fixed_q))),
                                 delta_q80_sum_kWh=float((fixed_q-recorded).sum())))
        summary['q80_review'][sid] = dict(missing_forecast_dates=missing, affected_dates=rows)
    base, test = 'G1_ridge_gate','G2_lightgbm_gate'
    delta = days[test]-days[base]
    monthly = delta.groupby(delta.index.str[:7]).sum()
    summary['primary'] = dict(delta_total_cost_yuan=float(delta.sum()),
                              delta_pct=float(delta.sum()/summary['groups'][base]['total_cost_yuan']*100),
                              improved_days=int((delta<0).sum()), improved_months=int((monthly<0).sum()),
                              monthly=monthly.to_dict())
    summary['primary']['mae_change_pct'] = {target:100*(
        summary['groups'][test]['metrics'][target]['mae']/summary['groups'][base]['metrics'][target]['mae']-1)
        for target in ['load','pv','net']}
    summary['source_hashes'] = dict(HASHES)
    changed = [rel for rel,h in HASHES.items() if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h]
    summary['inputs_changed_during_review'] = changed
    assert not changed, changed
    OUT.mkdir(exist_ok=True)
    (OUT/'review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
