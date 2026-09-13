"""Q1 interval-start interpretation; preserve official template order, midnight E=6000."""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/q1_start_time_20260912'
TEMPLATE=Path('C:/Users/Azure/Downloads/CUMCM2026Problems/C题/附件/附件5/result1.xlsx')

def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'附件/附件1.xlsx'
    w=openpyxl.load_workbook(source,read_only=True,data_only=True)
    rows=list(w.active.values)[1:];w.close()
    assert len(rows)==144
    raw=np.asarray([r[1:4] for r in rows],dtype=float)
    # Explicit daily-periodic boundary: source last row is midnight, not 23:50.
    order=np.r_[143,np.arange(143)]
    price,load,pv=raw[order].T
    spec=importlib.util.spec_from_file_location('q1',ROOT/'code/02_q1_baseline.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    (q,c,d,unused,state),info=m.solve(load,pv,price,integer=True,initial_kWh=6000,terminal_kWh=6000)
    (_,_,_,_,_),lp=m.solve(load,pv,price,integer=False,initial_kWh=6000,terminal_kWh=6000)
    assert abs(info['cost_yuan']-lp['cost_yuan'])<1e-6
    w=openpyxl.load_workbook(TEMPLATE,read_only=True,data_only=True)
    labels=[r[0] for r in list(w.worksheets[0].values)[1:]];w.close()
    label=lambda i:f'{i//60:02d}:{i%60:02d}'
    frame=pd.DataFrame(dict(start_minute=np.arange(144)*10,source_excel_row=order+2,
        interval=[label(t)+'-'+label(t+10) for t in range(0,1440,10)],
        price_yuan_kWh=price,load_kW=load,pv_kW=pv,grid_kWh=q,charge_kWh=c,
        discharge_kWh=d,unused_kWh=unused,state_start_kWh=state[:-1],state_end_kWh=state[1:],cost_yuan=price*q))
    frame.to_csv(OUT/'dispatch_chronological.csv',index=False,encoding='utf-8-sig')
    output_order=np.r_[np.arange(1,144),0]
    tf=frame.iloc[output_order].reset_index(drop=True)
    tf.insert(0,'template_interval',labels)
    tf.to_csv(OUT/'dispatch_template_order.csv',index=False,encoding='utf-8-sig')
    groups=[[float(c[i:i+24].sum()),float(d[i:i+24].sum())] for i in range(0,144,24)]
    payload=dict(template=str(TEMPLATE),purchases=[[float(x)] for x in q[output_order]],
                 storage=groups,endpoints=[[float(state[0])],[float(state[-1])]],labels=labels)
    (OUT/'workbook_payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    info.update(lp_cost_yuan=lp['cost_yuan'],lp_gap_yuan=info['cost_yuan']-lp['cost_yuan'],
        midnight_source_row=145,periodic_midnight_assumption=True,
        template_period='00:10 to next 00:10; repeat same daily profile',
        physical_day='00:00 to 24:00',template_period_start_kWh=float(state[1]),template_period_end_kWh=float(state[1]),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,TEMPLATE,ROOT/'code/02_q1_baseline.py',Path(__file__)]})
    (OUT/'validation.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:info[k] for k in ['cost_yuan','grid_kWh','initial_kWh','final_kWh','template_period_start_kWh','lp_gap_yuan','checks']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
