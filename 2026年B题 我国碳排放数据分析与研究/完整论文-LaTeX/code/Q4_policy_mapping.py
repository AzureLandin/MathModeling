from __future__ import annotations

import numpy as np
import pandas as pd

def build_q4_outputs(prov, annual, forecast):
    """Build reproducible policy flags and phase indicators from q1--q3 outputs."""
    thresholds={'intensity_median':float(prov.intensity_t_per_10k_yuan.median()),'coal_share_median':float(prov.coal_share_pct.median()),'linkage_median':float(prov.economic_linkage_index.median()),'total_median':float(prov.total_mt.median())}
    policy=prov[['province','class_name','priority','total_mt','intensity_t_per_10k_yuan','coal_share_pct','economic_linkage_index','process_share_pct']].copy()
    policy['总量控制']=policy.priority.eq('重点治理')
    policy['能效改造']=policy.intensity_t_per_10k_yuan.ge(thresholds['intensity_median'])
    policy['煤炭替代']=policy.coal_share_pct.ge(thresholds['coal_share_median'])
    policy['市场机制']=policy.economic_linkage_index.ge(thresholds['linkage_median'])
    policy['需求侧治理']=policy.total_mt.ge(thresholds['total_median']) & policy.coal_share_pct.lt(thresholds['coal_share_median'])
    policy_cols=['总量控制','能效改造','煤炭替代','市场机制','需求侧治理']
    policy['政策工具数']=policy[policy_cols].sum(axis=1).astype(int)
    policy['政策规则']='总量:重点治理；能效:强度≥中位数；煤炭:煤炭占比≥中位数；市场:经济联系≥中位数；需求侧:总量≥中位数且煤炭占比<中位数'
    class_policy=policy.groupby('class_name')[policy_cols].mean().mul(100).reset_index()
    class_policy[policy_cols]=class_policy[policy_cols].round(1)
    class_policy['省份数量']=policy.groupby('class_name').size().reindex(class_policy.class_name).to_numpy()
    sector_priority=annual[annual.sector!='Total'].groupby('sector',as_index=False).total_mt.sum().sort_values('total_mt',ascending=False)
    sector_priority['share_pct']=100*sector_priority.total_mt/sector_priority.total_mt.sum(); sector_priority['rank']=np.arange(1,len(sector_priority)+1)
    phase_rows=[]; phase_defs=[('2026--2030',2030),('2031--2035',2035),('2036--2045',2045)]
    for scenario,g in forecast.groupby('scenario'):
        base=float(g.loc[g.year==2025,'intensity_mt_per_trillion_yuan'].iloc[0])
        for phase,end_year in phase_defs:
            row=g.loc[g.year==end_year].iloc[0]
            phase_rows.append({'scenario':scenario,'phase':phase,'end_year':end_year,'total_mt':float(row.pred_mt),'intensity_mt_per_trillion_yuan':float(row.intensity_mt_per_trillion_yuan),'intensity_index_2025':float(100*row.intensity_mt_per_trillion_yuan/base),'intensity_reduction_pct':float(row.intensity_reduction_pct),'coal_share_pct':float(100*row.coal_share),'clean_share_pct':float(100*row.clean_share)})
    return policy,class_policy,pd.DataFrame(phase_rows),sector_priority,thresholds
