from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
root=Path.cwd(); d=root/'results/q3_intraday_load_cost'; m=json.loads((d/'run_manifest.json').read_text(encoding='utf-8-sig'))
h={p:hashlib.sha256((d/p).read_bytes()).hexdigest()==v for p,v in m['outputs'].items()}; assert all(h.values())
s=pd.read_csv(d/'summary.csv'); c=pd.read_csv(d/'contrasts.csv'); daily=pd.read_csv(d/'daily.csv'); monthly=pd.read_csv(d/'monthly.csv')
a=s.set_index('strategy_id'); assert abs((a.loc['S2_load_q75','natural_total_yuan']-a.loc['S0_L75','natural_total_yuan'])-c.loc[c.item=='total','delta_yuan'].iloc[0])<1e-8
for item in ['ordinary','adjustment','emergency']:
 dlt=float(a.loc['S2_load_q75',item+'_cost_yuan']-a.loc['S0_L75',item+'_cost_yuan'])
 assert abs(dlt-c.loc[c.item==item,'delta_yuan'].iloc[0])<1e-8
assert int((daily.delta_yuan<0).sum())==143 and int((daily.delta_yuan>0).sum())==191
assert int((monthly.delta_yuan<0).sum())==5 and int((monthly.delta_yuan>0).sum())==6
out={'hash_checks':h,'signature':m['signature'],'summary_rows':len(s),'max_physics':float(s.physics_residual_kWh.abs().max()),'max_cash_recalc':float(max(abs(c.delta_yuan.sum()-c.loc[c.item=='total','delta_yuan'].iloc[0]),0)),'daily_better':143,'daily_worse':191,'monthly_better':5,'monthly_worse':6,'delta_yuan':float(c.loc[c.item=='total','delta_yuan'].iloc[0]),'decomposition':{x:float(c.loc[c.item==x,'delta_yuan'].iloc[0]) for x in ['ordinary','adjustment','emergency']}}
outdir=root/'results/q3_intraday_load_cost_review';outdir.mkdir(exist_ok=True);(outdir/'review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
