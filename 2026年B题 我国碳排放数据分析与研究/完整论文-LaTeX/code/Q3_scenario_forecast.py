from __future__ import annotations

import numpy as np
import pandas as pd

def forecast_scenarios(mdl,scaler,anchor_2025,annual,residual_sd,outdir):
    y25=anchor_2025; gdp25=134.9084*1.05; pop25=140.8; coal25=.524; clean25=.296
    scen_defs={'基准':(0.6,0.6,[.045,.040,.035]),'低碳':(1.0,1.0,[.045,.040,.035]),'强化低碳':(1.4,1.4,[.043,.038,.032])}
    rows=[]
    for name,(coal_step,clean_step,growths) in scen_defs.items():
        gdp=gdp25; pop=pop25; coal=coal25; clean=clean25
        for year in range(2025,2046):
            if year>2025:
                gr=growths[0] if year<=2030 else growths[1] if year<=2035 else growths[2]
                gdp*=1+gr; pop*=1+0.001; coal=max(.12,coal-coal_step/100); clean=min(.78,clean+clean_step/100)
            X=np.log([[pop,gdp,coal,clean]]); pred=float(np.exp(mdl.predict(scaler.transform(X))[0]))
            rows.append({'scenario':name,'year':year,'gdp_trillion':gdp,'population_million':pop,'coal_share':coal,'clean_share':clean,'pred_mt_raw':pred})
    f=pd.DataFrame(rows); scale=y25/f[f.year==2025].pred_mt_raw.mean(); f['pred_mt']=f.pred_mt_raw*scale; f['intensity_mt_per_trillion_yuan']=f.pred_mt/f.gdp_trillion
    f['intensity_base_2025']=f.groupby('scenario')['intensity_mt_per_trillion_yuan'].transform('first'); f['intensity_reduction_pct']=100*(1-f.intensity_mt_per_trillion_yuan/f.intensity_base_2025); f['total_yoy_pct']=f.groupby('scenario').pred_mt.pct_change()*100; f['row_type']=np.where(f.year==2025,'anchor','forecast')
    horizon=np.maximum(1,(f.year-2025).to_numpy()); f['lower_mt']=np.maximum(0,f.pred_mt-1.96*residual_sd*np.sqrt(horizon)); f['upper_mt']=f.pred_mt+1.96*residual_sd*np.sqrt(horizon)
    f['path_valid']=(f.gdp_trillion>0)&(f.population_million>0)&(f.coal_share.between(0,1))&(f.clean_share.between(0,1))&((f.coal_share+f.clean_share)<=1)&(f.pred_mt>=0)
    return f,scale
