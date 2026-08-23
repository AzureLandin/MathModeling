from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROVINCES = ['Beijing','Tianjin','Hebei','Shanxi','Inner Mongolia','Liaoning','Jilin','Heilongjiang','Shanghai','Jiangsu','Zhejiang','Anhui','Fujian','Jiangxi','Shandong','Henan','Hubei','Hunan','Guangdong','Guangxi','Hainan','Chongqing','Sichuan','Guizhou','Yunnan','Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang']
GDP_2022 = dict(zip(PROVINCES,[41610.9,16311.3,42370.4,25642.6,23158.6,28975.1,13070.2,15901.0,44652.8,122875.6,77715.4,45045.0,53109.9,32074.7,87435.1,61345.1,53734.9,48670.4,129118.6,26300.9,6818.2,29129.0,56749.8,20164.6,28954.2,32772.7,11201.6,3610.1,5069.6,17741.3]))
POP_2022 = dict(zip(PROVINCES,[2184,1363,7420,3481,2401,4197,2348,3099,2475,8515,6577,6127,4188,4528,10163,9872,5844,6604,12657,5047,4027,3213,8374,3856,4693,3956,2492,595,728,2587]))

def annualize_2025(df):
    d=df[df.Sector=='Total'].copy(); d['year']=d.Date.dt.year; d['month']=d.Date.dt.month
    hist=d[d.year<=2024].groupby('year')['CO2 (Mt)'].sum()
    frac=d[d.year<=2024].query('month<=9').groupby('year')['CO2 (Mt)'].sum()/hist
    y25=d[d.year==2025].query('month<=9')['CO2 (Mt)'].sum()/frac.mean()
    return float(y25), float(frac.mean()), float(frac.std(ddof=1))

def load_inputs(input_root:Path):
    csv_path=input_root/'附件1-中国2019年-2025年碳排放数据.csv'
    xlsx_path=input_root/'附件2-2022年30个省份排放清单.xlsx'
    d=pd.read_csv(csv_path, parse_dates=['Date'])
    assert len(d)==17255 and d.Date.nunique()==2465
    assert set(d.Sector.unique())=={'Domestic Aviation','Ground Transport','Industry','International Aviation','Power','Residential','Total'}
    d['year']=d.Date.dt.year; d['month']=d.Date.dt.month
    check=d.pivot_table(index='Date',columns='Sector',values='CO2 (Mt)',aggfunc='sum')
    # 附件1 Total 与国内五部门一致；International Aviation 单列不计入 Total
    sector_sum=check[['Domestic Aviation','Ground Transport','Industry','Power','Residential']].sum(axis=1)
    max_err=float(np.max(np.abs(sector_sum-check['Total'])))
    assert max_err<1e-5, max_err
    xls=pd.ExcelFile(xlsx_path); province_rows=[]
    coal_cols=['Raw_Coal','CleanedCoal','Other_Washed_Coal','Briquettes','Coke','Coke_Oven_Gas','Other_Gas','Other_Coking_Products']
    oil_cols=['Crude_Oil','Gasoline','Kerosene','Diesel_Oil','Fuel_Oil','LPG','Refinery_Gas','Other_Petroleum_Products']
    gas_cols=['Natural_Gas']
    for s in xls.sheet_names:
        if s=='NOTE': continue
        raw=pd.read_excel(xlsx_path,sheet_name=s,header=None)
        cols=raw.iloc[0].tolist(); row=raw.iloc[3].tolist(); rec=dict(zip(cols,row))
        assert 'Scope_1_Total' in rec and rec['Scope_1_Total'] is not None
        province_rows.append({'province':s.replace('2022',''),'total_mt':float(rec['Scope_1_Total'] or 0),
            'coal_mt':sum(float(rec.get(c,0) or 0) for c in coal_cols), 'oil_mt':sum(float(rec.get(c,0) or 0) for c in oil_cols),
            'gas_mt':sum(float(rec.get(c,0) or 0) for c in gas_cols), 'process_mt':float(rec.get('Process',0) or 0),
            'other_mt':sum(float(rec.get(c,0) or 0) for c in ['Scope_2_Heat','Scope_2_Electricity','Other_Energy'])})
    prov=pd.DataFrame(province_rows)
    name_map={'Shanghai':'Shanghai','Yunnan':'Yunnan','InnerMongolia':'Inner Mongolia','Beijing':'Beijing','Jilin':'Jilin','Sichuan':'Sichuan','Tianjin':'Tianjin','Ningxia':'Ningxia','Anhui':'Anhui','Shandong':'Shandong','Shanxi':'Shanxi','Guangdong':'Guangdong','Guangxi':'Guangxi','Xinjiang':'Xinjiang','Jiangsu':'Jiangsu','Jiangxi':'Jiangxi','Hebei':'Hebei','Henan':'Henan','Zhejiang':'Zhejiang','Hainan':'Hainan','Hubei':'Hubei','Hunan':'Hunan','Gansu':'Gansu','Fujian':'Fujian','Guizhou':'Guizhou','Liaoning':'Liaoning','Chongqing':'Chongqing','Shaanxi':'Shaanxi','Qinghai':'Qinghai','Heilongjiang':'Heilongjiang'}
    prov['province']=prov.province.map(name_map)
    assert prov.province.notna().all() and len(prov)==30
    prov['gdp_2022_100m']=prov.province.map(GDP_2022); prov['pop_2022_10k']=prov.province.map(POP_2022)
    assert prov.gdp_2022_100m.notna().all() and prov.pop_2022_10k.notna().all()
    prov['share_pct']=100*prov.total_mt/prov.total_mt.sum(); prov['per_capita_t']=100*prov.total_mt/prov.pop_2022_10k
    prov['intensity_t_per_10k_yuan']=100*prov.total_mt/prov.gdp_2022_100m
    prov['coal_share_pct']=100*prov.coal_mt/prov.total_mt; prov['process_share_pct']=100*prov.process_mt/prov.total_mt
    el_cols=['intensity_t_per_10k_yuan','coal_share_pct','process_share_pct']; prov['economic_linkage_index']=StandardScaler().fit_transform(prov[el_cols]).mean(axis=1)
    return d,prov,max_err
