"""Saved-result reception only; math_modeling; no training or policy rerun."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/q3_intraday_load_forward'
OUT=ROOT/'results/q3_intraday_load_forward_review'
manifest=json.loads((D/'run_manifest.json').read_text(encoding='utf-8-sig'))
hashes={p:hashlib.sha256((D/p).read_bytes()).hexdigest()==h for p,h in manifest['outputs'].items()}
assert all(hashes.values())
p=pd.read_csv(D/'prediction_pairs.csv')
log=pd.read_csv(D/'parameter_log.csv')
with np.load(ROOT/'results/q3_bias_correction_diagnostic/bias_forecast_archive.npz') as z:
    issued=z['issued_load']; truth=z['truth_load']; residual=truth-issued
with np.load(D/'forecast_archive.npz') as z:
    forecasts=z['forecast']
for g in range(3):
    assert np.array_equal(np.isnan(forecasts[g,:,0,:]),np.isnan(issued))
    assert np.allclose(forecasts[g,:,0,:],issued,atol=1e-8,rtol=0,equal_nan=True)
k=p.k.to_numpy(int);h=p.h.to_numpy(int);v=p.version.to_numpy(int)
assert np.allclose(p.actual_kW,truth[k,h],atol=1e-8,rtol=0)
assert np.allclose(p.f0_kW,issued[k,h],atol=1e-8,rtol=0)
for g in range(3):
    assert np.allclose(p[f'f{g}_kW'],forecasts[g,k,v,h],atol=1e-8,rtol=0)
assert len(p)==73146 and p.in_A.sum()==36072
assert not p[p.in_A].duplicated(['k','h']).any()
# Independently reconstruct the valid prior-day sets for every saved parameter group.
max_parameter_error=0.; checked=0;base=pd.Timestamp('2025-01-01')
for (r,kind,j), block in log.groupby(['publication_hour','kind','j']):
    s=int(r)*6
    b=residual[:,s-6:s].mean(axis=1)
    target=144 if kind=='tail' else s+6*int(j)
    width=1 if kind=='tail' else 6
    y=residual[:,target:target+width].mean(axis=1)
    for row in block.itertuples():
        hist=np.arange(max(0,row.k-28),row.k)
        hist=hist[np.isfinite(b[hist])&np.isfinite(y[hist])]
        dates=[] if pd.isna(row.history_dates) else [(pd.Timestamp(t)-base).days for t in row.history_dates.split('|')]
        assert np.array_equal(hist,dates) and row.m==len(hist)
        assert all(d*144+target+width <= row.k*144+s for d in hist)
        if row.k in [31,171,364] and (kind=='tail' or j==0):
            assert len(hist)>=14
            a,beta=np.linalg.lstsq(np.column_stack([np.ones(len(hist)),b[hist]]),y[hist],rcond=None)[0]
            expected=[a,beta,y[hist].mean(),a+beta*b[row.k]]
            saved=[row.intercept_a_kW,row.beta,row.delta1_kW,row.delta2_kW]
            max_parameter_error=max(max_parameter_error,float(np.max(np.abs(np.array(expected)-saved))))
            checked+=1
assert max_parameter_error<1e-8

def stats(frame):
    out={}
    for g in range(3):
        e=(frame.actual_kW-frame[f'f{g}_kW']).to_numpy()
        out[f'F{g}']={'n':len(e),'mae_kW':float(np.abs(e).mean()),'rmse_kW':float(np.sqrt((e*e).mean())),
                    'bias_kW':float(e.mean())}
    return out
A=p[p.in_A].copy()
for g in range(3): A[f'ae{g}']=abs(A.actual_kW-A[f'f{g}_kW'])
day=A.groupby('date')[[f'ae{g}' for g in range(3)]].mean()
delta=day.ae2-day.ae0
best5=delta.nsmallest(5)
monthly=A.groupby('month')[[f'ae{g}' for g in range(3)]].mean()
allfit=log[log.fitted]; formal=allfit[allfit.k>=31]
result={'scope':'saved-prediction and history review; not policy or full independent forecast audit',
        'hash_checks':hashes,'signature':manifest['signature'],
        'history_groups_checked':len(log),'parameter_spot_checks':checked,'max_parameter_error':max_parameter_error,
        'A':stats(A),'B':stats(p),'tail_all_versions':stats(p[p.is_tail]),
        'tail_by_node':{int(r):stats(b) for r,b in p[p.is_tail].groupby('publication_hour')},
        'node_A':{int(r):stats(b) for r,b in A.groupby('publication_hour')},
        'monthly_F2_minus_F0':(monthly.ae2-monthly.ae0).to_dict(),
        'monthly_F2_minus_F1':(monthly.ae2-monthly.ae1).to_dict(),
        'days_better_vs_F0':int((delta<0).sum()),'day_delta_median':float(delta.median()),
        'best5_dates':best5.to_dict(),
        'best5_share_of_net_improvement':float(best5.sum()/delta.sum()),
        'remaining329_day_mean_delta_kW':float(delta.drop(best5.index).mean()),
        'beta_negative_all_fitted':float((allfit.beta<0).mean()),
        'beta_negative_formal_fitted':float((formal.beta<0).mean()),
        'formal_fit_m_range':[int(formal.m.min()),int(formal.m.max())],
        'formal_beta_range':[float(formal.beta.min()),float(formal.beta.max())]}
OUT.mkdir(exist_ok=True)
(OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))
