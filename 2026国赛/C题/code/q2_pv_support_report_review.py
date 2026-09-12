"""Read-only review of saved PV-support outputs; no training or optimisation."""
from pathlib import Path
import hashlib
import json
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/q2_pv_support_features'
OUT = ROOT / 'results/q2_pv_support_report_review'
OUT.mkdir(exist_ok=True)
hashes = {}
def read(name):
    p = SRC / name
    hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p).sort_values(['date', 'slot']).reset_index(drop=True)
def worst(x):
    return float(np.max(np.abs(np.asarray(x))))

summary = pd.read_csv(SRC / 'summary.csv').set_index('strategy_id')
frames, rows, controls, comparisons = {}, {}, {}, {}
for pred in ['P0', 'P1']:
    archive = read(f'pv_forecast_archive_{pred}.csv')
    assert len(archive) == 52560 and not archive.duplicated(['date', 'slot']).any()
    forecast = archive[['load_forecast_kW','pv_forecast_kW']].to_numpy().reshape(365,144,2)
    assert np.isfinite(forecast[1:]).all()
    # Truth taken from the frozen G4 archive; this review does not reread Excel.
    truth = pd.read_csv(ROOT / 'results/q2_lightgbm_residual/G4_lightgbm_fixed/forecast_residuals.csv').sort_values(['date','slot'])
    actual = truth[['load_kW','pv_kW']].to_numpy().reshape(365,144,2)
    eps = (actual[:,:,0]-actual[:,:,1]-forecast[:,:,0]+forecast[:,:,1])/6
    for level in [80,75]:
        gid = f'{pred}_q{level}'
        f = frames[gid] = read(f'{gid}/dispatch.csv')
        assert len(f) == 48096 and not f.duplicated(['date','slot']).any()
        assert np.all(f.residual_days == 28)
        q,c,d,e,w = [f[col].to_numpy() for col in ['planned_kWh','charge_kWh','discharge_kWh','emergency_kWh','unused_kWh']]
        s0,s1 = f.state_start_kWh.to_numpy(), f.state_end_kWh.to_numpy()
        n = (f.load_kW-f.pv_kW).to_numpy()/6
        price = f.price_yuan_kWh.to_numpy()
        loss = .1*c+(1/.9-1)*d
        cost = price*q+5*price*e
        adj = np.stack([np.sort(eps[k-28:k],axis=0)[(28*level+99)//100-1] for k in range(31,365)])
        replay_c=np.minimum.reduce([np.maximum(q-n,0),np.full(len(f),5000/6),(10800-s0)/.9])
        replay_d=np.minimum.reduce([np.maximum(n-q,0),np.full(len(f),5000/6),.9*(s0-1200)])
        active=e>1e-6
        day_active=active.reshape(334,144)
        metrics={
            'total_cost_yuan':float(cost.sum()), 'planned_cost_yuan':float((price*q).sum()),
            'emergency_cost_yuan':float((5*price*e).sum()),
            'balance_max_kWh':worst(q+e+d-n-c-w),
            'recursion_max_kWh':worst(s1-s0-.9*c+d/.9),
            'continuity_max_kWh':worst(s0[1:]-s1[:-1]),
            'feedback_max_kWh':max(worst(c-replay_c),worst(d-replay_d),worst(e-np.maximum(n-q-replay_d,0)),worst(w-np.maximum(q-n-replay_c,0))),
            'energy_error_kWh':float(np.sum(q+e-n-w-loss)-(s1[-1]-s0[0])),
            'quantile_max_kWh':worst(adj.ravel()-f.residual_adjustment_kWh),
            'cost_vs_summary_yuan':float(cost.sum()-summary.loc[gid,'total_cost_yuan']),
            'emergency_days':int(day_active.any(axis=1).sum()),
            'emergency_slots':int(active.sum()),
            'emergency_events':int((day_active & ~np.concatenate([np.zeros((334,1),bool),day_active[:,:-1]],axis=1)).sum()),
            'final_kWh':float(s1[-1]),
            'pv_mae_kW':float(np.abs(f.pv_kW-f.pv_forecast_kW).mean()),
            'net_mae_kW':float(np.abs((f.load_kW-f.pv_kW)-(f.load_forecast_kW-f.pv_forecast_kW)).mean())}
        assert max(metrics[x] for x in ['balance_max_kWh','recursion_max_kWh','continuity_max_kWh','feedback_max_kWh','quantile_max_kWh'])<1e-6
        assert abs(metrics['cost_vs_summary_yuan'])<1e-4 and abs(metrics['energy_error_kWh'])<1e-4
        rows[gid]=metrics
        if pred=='P0':
            ref=pd.read_csv(ROOT/f'results/q2_quantile_level_scan/L_q{level}/dispatch.csv').sort_values(['date','slot']).reset_index(drop=True)
            cols=['planned_kWh','charge_kWh','discharge_kWh','emergency_kWh','unused_kWh','state_start_kWh','state_end_kWh','residual_adjustment_kWh']
            controls[gid]=worst(f[cols].to_numpy()-ref[cols].to_numpy())
            assert controls[gid]<1e-6
for level in [80,75]:
    a,b=frames[f'P1_q{level}'],frames[f'P0_q{level}']
    delta=(a.planned_cost_yuan+a.emergency_cost_yuan-b.planned_cost_yuan-b.emergency_cost_yuan)
    monthly=delta.groupby(a.date.str[:7]).sum()
    total=float(delta.sum())
    comparisons[str(level)]={'delta_cost_yuan':total,'monthly_delta_yuan':monthly.to_dict(),'august_share_of_net_saving':float(monthly.loc['2025-08']/total),'delta_excluding_august_yuan':float(total-monthly.loc['2025-08']),'same_load_max_kW':worst(a.load_forecast_kW-b.load_forecast_kW),'delta_final_kWh':float(a.state_end_kWh.iloc[-1]-b.state_end_kWh.iloc[-1])}
manifest=json.loads((SRC/'run_manifest.json').read_text(encoding='utf-8'))
reg=json.loads((SRC/'registration.json').read_text(encoding='utf-8'))
elapsed=(datetime.fromisoformat(manifest['finished_utc'])-datetime.fromisoformat(manifest['started_utc'])).total_seconds()
windows=pd.read_csv(SRC/'support_windows.csv')
result={'scope':'Saved-output review only; no Excel reread, independent training, future perturbation or MILP re-solve.', 'groups':rows,'controls_max_kWh':controls,'comparisons':comparisons,'manifest_elapsed_seconds':elapsed,'manifest_wall_seconds':manifest['wall_seconds'],'registration_signature_matches':reg['signature']==manifest['signature'],'source_hash_matches':hashlib.sha256((ROOT/'code/23_q2_pv_support_features_experiment.py').read_bytes()).hexdigest()==manifest['input_sha256']['code/23_q2_pv_support_features_experiment.py'],'first_window_record':windows.iloc[0].to_dict(),'input_hashes':hashes}
(OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'},ensure_ascii=False,indent=2))
