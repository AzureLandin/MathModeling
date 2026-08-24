from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler

from All_common_utilities import COLORS, savefig

def make_figures(annual,prov,spatial,coef,bt,forecast,q4_class_policy,q4_phase_targets,q4_sector_priority,outdir):
    outdir.mkdir(exist_ok=True,parents=True)
    # raw q1
    p=prov.sort_values('total_mt',ascending=False).head(20); fig,ax=plt.subplots(); ax.barh(p.province.iloc[::-1],p.total_mt.iloc[::-1],color=COLORS['blue']); ax.set_xlabel('2022 排放总量 (Mt CO2)'); ax.set_ylabel('省份'); ax.set_title('省级排放规模（前20）'); savefig(fig,outdir/'raw_q1_province_scale')
    # process q1
    fig,ax=plt.subplots(); q=pd.qcut(prov.coal_share_pct,3,labels=['低煤炭','中煤炭','高煤炭']); palette={'低煤炭':COLORS['teal'],'中煤炭':COLORS['orange'],'高煤炭':COLORS['red']}
    for cat in q.cat.categories:
        g=prov[q==cat]; ax.scatter(g.intensity_t_per_10k_yuan,g.per_capita_t,c=palette[cat],s=42,label=cat,edgecolor='white',lw=.4)
    ax.set_xlabel('排放强度 (t/万元GDP)'); ax.set_ylabel('人均排放 (t/person)'); ax.set_title('规模—效率—人口压力关系'); ax.legend(frameon=False,title='煤炭排放占比分组'); savefig(fig,outdir/'process_q1_indicator_space')
    # result q1
    cp=prov.groupby(['cluster','class_name'])[['total_mt','per_capita_t','intensity_t_per_10k_yuan','coal_share_pct','process_share_pct']].mean(); Z=StandardScaler().fit_transform(cp); fig,ax=plt.subplots(figsize=(6.5,3.5)); cmap=matplotlib.colormaps['RdBu_r']; norm=Normalize(-2,2); nrows,ncols=Z.shape
    for r in range(nrows):
        for c in range(ncols):
            ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=cmap(norm(Z[r,c])),edgecolor='white',lw=.6)); ax.text(c,r,f'{Z[r,c]:.1f}',ha='center',va='center',fontsize=7)
    ax.set_xlim(-.5,ncols-.5); ax.set_ylim(nrows-.5,-.5); ax.set_yticks(range(nrows)); ax.set_yticklabels([f'{i} {n}' for i,n in cp.index]); ax.set_xticks(range(ncols)); ax.set_xticklabels(['总量','人均','强度','煤炭','过程'],rotation=25,ha='right'); ax.set_title('三类省份指标画像'); savefig(fig,outdir/'result_q1_cluster_profile',size=(6.5,3.5))
    # raw q2
    sec=annual[annual.sector!='Total'].pivot(index='year',columns='sector',values='total_mt'); fig,ax=plt.subplots(); ax.stackplot(sec.index,sec.T.values,labels=sec.columns,alpha=.88); ax.set_ylabel('年度排放 (Mt CO2)'); ax.set_xlabel('年份'); ax.set_title('全国分部门排放变化'); ax.legend(ncol=3,frameon=False,loc='upper left'); savefig(fig,outdir/'raw_q2_annual_sectors')
    # process q2
    fig,ax=plt.subplots(); c=coef.sort_values('standardized_coefficient'); ax.barh(c.factor,c.standardized_coefficient,color=[COLORS['red'] if x<0 else COLORS['teal'] for x in c.standardized_coefficient]); ax.axvline(0,color='black',lw=.7); ax.set_xlabel('标准化岭系数'); ax.set_title('驱动因素方向与相对权重'); savefig(fig,outdir/'process_q2_coefficients')
    # result q2
    fig,ax=plt.subplots(); ax.plot(bt.year,bt.total_mt,'o-',label='观测',color=COLORS['gray']); ax.plot(bt.year,bt.pred_mt,'s--',label='拟合/回测',color=COLORS['orange']); ax.set_xlabel('年份'); ax.set_ylabel('年度排放 (Mt CO2)'); ax.set_title('STIRPAT-ridge 回测拟合'); ax.legend(frameon=False); savefig(fig,outdir/'result_q2_backtest')
    # raw q3
    hist=annual[annual.sector=='Total'].groupby('year').total_mt.sum(); fig,ax=plt.subplots(1,2,figsize=(6.5,3.0)); ax[0].plot(hist.index,hist.values,'o-',color=COLORS['blue']); ax[0].axvline(2025,color=COLORS['red'],ls='--'); ax[0].set_title('历史排放总量'); ax[0].set_ylabel('Mt CO2'); ax[1].plot(hist.index,hist.values/np.array([99.0865,101.3567,114.9237,121.0207,126.0582,134.9084,140.0])[:len(hist)],'o-',color=COLORS['teal']); ax[1].set_title('历史排放强度代理'); ax[1].set_ylabel('Mt/万亿元 GDP'); [a.set_xlabel('年份') for a in ax]; savefig(fig,outdir/'raw_q3_historical_trend',size=(6.5,3.0))
    # process q3
    path=forecast[forecast.year>=2025].drop_duplicates(['scenario','year']); fig,ax=plt.subplots();
    for s,g in path.groupby('scenario'): ax.plot(g.year,g.coal_share*100,label=s+' 煤炭',lw=1.8); ax.plot(g.year,g.clean_share*100,ls='--',label=s+' 清洁',lw=1.2)
    ax.set_xlabel('年份'); ax.set_ylabel('能源结构比重 (%)'); ax.set_title('情景外生路径'); ax.legend(ncol=2,frameon=False,fontsize=6); savefig(fig,outdir/'process_q3_scenario_paths')
    # result q3
    fig,ax=plt.subplots(1,2,figsize=(6.5,3.0));
    for s,g in path.groupby('scenario'): ax[0].plot(g.year,g.pred_mt,label=s,lw=2); ax[1].plot(g.year,g.intensity_mt_per_trillion_yuan,label=s,lw=2)
    ax[0].set_title('总量预测'); ax[0].set_ylabel('Mt CO2'); ax[1].set_title('强度预测'); ax[1].set_ylabel('Mt/万亿元 GDP'); [a.set_xlabel('年份') for a in ax]; ax[0].legend(frameon=False,fontsize=6); savefig(fig,outdir/'result_q3_forecast',size=(6.5,3.0))
    # raw q4: measured cumulative sector emissions
    sectors=q4_sector_priority.sort_values('total_mt'); fig,ax=plt.subplots(); ax.barh(sectors.sector,sectors.total_mt,color=COLORS['purple']); ax.set_xlabel('2019—2025累计排放 (Mt CO2)'); ax.set_title('部门治理优先级原始证据'); savefig(fig,outdir/'raw_q4_sector_share')
    # process q4: policy coverage computed from province-level indicators
    policy_cols=['总量控制','能效改造','煤炭替代','市场机制','需求侧治理']; mat=q4_class_policy.set_index('class_name')[policy_cols]/100.0; fig,ax=plt.subplots(figsize=(6.5,3.0)); cmap=matplotlib.colormaps['Blues']; nrows,ncols=mat.shape
    for r in range(nrows):
        for c in range(ncols):
            value=float(mat.iloc[r,c]); ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=cmap(.15+.8*value),edgecolor='white')); ax.text(c,r,f'{value:.0%}',ha='center',va='center',fontsize=8)
    ax.set_xlim(-.5,ncols-.5); ax.set_ylim(nrows-.5,-.5); ax.set_xticks(range(ncols)); ax.set_xticklabels(mat.columns,rotation=25,ha='right'); ax.set_yticks(range(nrows)); ax.set_yticklabels(mat.index); ax.set_title('省域类别的政策工具覆盖率'); savefig(fig,outdir/'process_q4_policy_matrix',size=(6.5,3.0))
    # result q4: calculated phase-end intensity indices under each scenario
    fig,ax=plt.subplots(figsize=(6.5,3.0)); order=['2026--2030','2031--2035','2036--2045']
    for scenario,g in q4_phase_targets.groupby('scenario'):
        g=g.set_index('phase').loc[order].reset_index(); ax.plot(g.phase,g.intensity_index_2025,'o-',lw=2,label=scenario)
    ax.set_ylim(0,110); ax.set_ylabel('排放强度指数（2025=100）'); ax.set_title('三情景阶段末排放强度指数'); ax.legend(frameon=False); savefig(fig,outdir/'result_q4_roadmap',size=(6.5,3.0))
