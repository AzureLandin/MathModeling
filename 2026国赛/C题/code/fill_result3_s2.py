from pathlib import Path
from copy import copy
from datetime import datetime
import shutil, hashlib, json
import pandas as pd
from openpyxl import load_workbook

ROOT=Path.cwd(); template=ROOT/'附件/附件5/result3.xlsx'; backup=ROOT/'results/template_backups/result3_before_s2.xlsx'
backup.parent.mkdir(exist_ok=True)
if not backup.exists(): shutil.copy2(template,backup)
out=ROOT/'附件/附件5/result3.xlsx'
df=pd.read_csv(ROOT/'results/q3_intraday_load_cost/S2_load_q75/dispatch.csv',parse_dates=['interval_start','interval_end'])
df=df.sort_values('interval_start').reset_index(drop=True)
natural=df[df.interval_start < pd.Timestamp('2026-01-01')].copy()
tail=df[df.interval_start >= pd.Timestamp('2026-01-01')].iloc[0]
# date -> natural 144 intervals, with the template plan row interpreted as 00:10..next 00:10.
groups={d: g.sort_values('interval_start').reset_index(drop=True) for d,g in natural.groupby(natural.interval_start.dt.normalize())}
wb=load_workbook(template)
# first two sheets: original commitment q0 and final effective purchase q_eff
for si,col in [(0,'q0_kWh'),(1,'q_eff_kWh')]:
 ws=wb.worksheets[si]
 for day_idx in range(334):
  day=pd.Timestamp('2025-02-01')+pd.Timedelta(days=day_idx); nextday=day+pd.Timedelta(days=1)
  vals=list(groups[day].iloc[1:][col].astype(float))+([float(groups[nextday].iloc[0][col])] if nextday in groups else [float(tail[col])])
  r=2+day_idx
  for j,v in enumerate(vals, start=2): ws.cell(r,j).value=v
  ws.cell(r,146).value=sum(vals); ws.cell(r,147).value=float(sum(vals[i]*groups[day].iloc[1:]["price_yuan_kWh"].astype(float).iloc[i] for i in range(143)) + vals[-1]*float((groups[nextday].iloc[0] if nextday in groups else tail)['price_yuan_kWh']))
# storage sheet: six 4-hour blocks based on each natural date's 144 intervals.
ws=wb.worksheets[2]
# clear existing data rows, then write exactly 334*6 rows while preserving row 2 styles.
for day_idx in range(334):
 day=pd.Timestamp('2025-02-01')+pd.Timedelta(days=day_idx); g=groups[day]
 for block in range(6):
  r=2+day_idx*6+block
  if r>ws.max_row: ws.insert_rows(r)
  seg=g.iloc[block*24:(block+1)*24]
  ws.cell(r,1).value=day.to_pydatetime() if block==0 else None
  ws.cell(r,2).value=f'{block*4:02d}:00-{(block+1)*4:02d}:00'
  ws.cell(r,3).value=float(seg.charge_kWh.sum()); ws.cell(r,4).value=float(seg.discharge_kWh.sum())
  ws.cell(r,5).value='0:00' if block==0 else ('24:00' if block==1 else None)
  ws.cell(r,6).value=float(g.state_start_kWh.iloc[0]) if block==0 else (float(g.state_end_kWh.iloc[-1]) if block==1 else None)
# emergency events from natural rows; retain only positive emergency purchases.
ws=wb.worksheets[3]
for r in range(2,ws.max_row+1):
 for c in range(1,4): ws.cell(r,c).value=None
pos=natural[natural.emergency_kWh>1e-9]
for i,(_,row) in enumerate(pos.iterrows(),start=2):
 if i>ws.max_row: ws.insert_rows(i)
 ws.cell(i,1).value=pd.Timestamp(row.interval_start).to_pydatetime().date()
 ws.cell(i,2).value=f"{pd.Timestamp(row.interval_start).strftime('%H:%M')}-{pd.Timestamp(row.interval_end).strftime('%H:%M')}"
 ws.cell(i,3).value=float(row.emergency_kWh)
# common formatting for added rows and numeric cells
for ws in wb.worksheets[:3]:
 for row in ws.iter_rows(min_row=2):
  if row[0].value is not None and hasattr(row[0].value,'year'): row[0].number_format='yyyy/mm/dd'
  for cell in row[1:]:
   if isinstance(cell.value,(int,float)): cell.number_format='0.000000'
for row in ws.iter_rows(min_row=2):
 if hasattr(row[0].value,'year'): row[0].number_format='yyyy/mm/dd'
 for cell in row[1:]:
  if isinstance(cell.value,(int,float)): cell.number_format='0.000000'
wb.save(out)
# readback validations
rb=load_workbook(out,data_only=True)
checks={}
for si,col in [(0,'q0_kWh'),(1,'q_eff_kWh')]:
 vals=[]
 for r in range(2,336): vals.extend([rb.worksheets[si].cell(r,c).value for c in range(2,146)])
 expected=list(natural.iloc[1:][col].astype(float))+[float(tail[col])]
 # expected order plan rows: each day slots 1..143 then next slot0
 exp=[]
 for day in groups:
  exp.extend(groups[day].iloc[1:][col].astype(float).tolist())
  nd=day+pd.Timedelta(days=1); exp.append(float(groups[nd].iloc[0][col]) if nd in groups else float(tail[col]))
 checks[col+'_max_abs_error']=max(abs(float(a)-float(b)) for a,b in zip(vals,exp))
checks['storage_rows']=rb.worksheets[2].max_row; checks['emergency_rows']=rb.worksheets[3].max_row
checks['emergency_total_error']=abs(sum(float(rb.worksheets[3].cell(r,3).value or 0) for r in range(2,rb.worksheets[3].max_row+1))-float(natural.emergency_kWh.sum()))
checks['sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
assert checks['q0_kWh_max_abs_error']<1e-9 and checks['q_eff_kWh_max_abs_error']<1e-9 and checks['emergency_total_error']<1e-8
(ROOT/'results/q3_s2_delivery_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
