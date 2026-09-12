"""Read-only reception check; run only in math_modeling; no model/dispatch execution."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/q3_intraday_load_residual_diagnostic'
OUT = ROOT / 'results/q3_intraday_load_residual_review'
manifest = json.loads((D/'run_manifest.json').read_text(encoding='utf-8-sig'))
hashes = {p: hashlib.sha256((D/p).read_bytes()).hexdigest() == h for p,h in manifest['outputs'].items()}
assert all(hashes.values())
with np.load(ROOT/'results/q3_bias_correction_diagnostic/bias_forecast_archive.npz') as a:
    residual = a['truth_load'] - a['issued_load']
pairs = pd.read_csv(D/'hourly_pairs.csv')
overall = pd.read_csv(D/'overall_summary.csv')
monthly = pd.read_csv(D/'monthly_summary.csv')
centered = pd.read_csv(D/'centered_summary.csv')
errors = {'b_kW':0., 'y_kW':0., 'summary':0.}
for p in pairs.itertuples():
    s = int(p.publication_hour)*6
    b = residual[p.k,s-6:s].mean()
    h = 144 if p.kind == 'midnight_tail' else s+6*p.j
    y = residual[p.k,h:h+(1 if p.kind == 'midnight_tail' else 6)].mean()
    errors['b_kW'] = max(errors['b_kW'],abs(b-p.b_kW))
    errors['y_kW'] = max(errors['y_kW'],abs(y-p.y_kW))

def check(block,row,group=None):
    x,y = block.b_kW.to_numpy(),block.y_kW.to_numpy()
    if group:
        x = (block.b_kW-block.groupby(group).b_kW.transform('mean')).to_numpy()
        y = (block.y_kW-block.groupby(group).y_kW.transform('mean')).to_numpy()
    xc,yc=x-x.mean(),y-y.mean()
    beta = np.dot(xc,yc)/np.dot(xc,xc)
    r = np.dot(xc,yc)/np.sqrt(np.dot(xc,xc)*np.dot(yc,yc))
    for name,val in [('pearson_r',r),('beta',beta)]:
        errors['summary']=max(errors['summary'],abs(val-float(row[name])))
keys=['publication_hour','kind','j']
for key,block in pairs.groupby(keys):
    mask = (overall[keys]==key).all(axis=1)
    check(block,overall.loc[mask].iloc[0])
    for month,b in block.groupby('month'):
        m=(monthly[keys]==key).all(axis=1)&(monthly.month==month)
        check(b,monthly.loc[m].iloc[0])
    for g in ['month','weekday']:
        m=(centered[keys]==key).all(axis=1)&(centered.demean==g)
        check(block,centered.loc[m].iloc[0],g)
assert max(errors.values()) < 1e-8
near=monthly[(monthly.kind=='hour')&(monthly.j<6)]
minrow=monthly.loc[monthly.pearson_r.idxmin()]
result={
    'scope':'saved-data pairing and descriptive summaries; not independent model-chain audit',
    'signature':manifest['signature'],'hash_checks':hashes,'max_absolute_errors':errors,
    'pair_rows':len(pairs),'overall_cells':len(overall),
    'overall_r_range':[float(overall.pearson_r.min()),float(overall.pearson_r.max())],
    'centered_r_min':centered.groupby('demean').pearson_r.min().to_dict(),
    'cells_with_negative_month':int((monthly.groupby(keys).pearson_r.min()<0).sum()),
    'minimum_monthly_cell':minrow.to_dict(),
    'near_negative_month_hour_cells':near[near.pearson_r<0].groupby('publication_hour').size().to_dict(),
    'near_distinct_negative_months':near[near.pearson_r<0].groupby('publication_hour').month.nunique().to_dict(),
    'r_by_hour_18':overall[(overall.publication_hour==18)&(overall.kind=='hour')].sort_values('j').pearson_r.tolist(),
}
OUT.mkdir(exist_ok=True)
(OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))
