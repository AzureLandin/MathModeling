from __future__ import annotations
import argparse, hashlib, json, math, os, sys, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score

SKILL_ROOT = Path(r'C:\Users\Azure\Downloads\math_modeling_test\math-modeling-skill-v1.2.0')
sys.path.insert(0, str(SKILL_ROOT / 'tools' / 'figure' / 'scripts'))
from export_figure import export_figure

SEED = 20260822
RNG = np.random.default_rng(SEED)

PROVINCES = ['Beijing','Tianjin','Hebei','Shanxi','Inner Mongolia','Liaoning','Jilin','Heilongjiang','Shanghai','Jiangsu','Zhejiang','Anhui','Fujian','Jiangxi','Shandong','Henan','Hubei','Hunan','Guangdong','Guangxi','Hainan','Chongqing','Sichuan','Guizhou','Yunnan','Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang']
GDP_2022 = dict(zip(PROVINCES,[41610.9,16311.3,42370.4,25642.6,23158.6,28975.1,13070.2,15901.0,44652.8,122875.6,77715.4,45045.0,53109.9,32074.7,87435.1,61345.1,53734.9,48670.4,129118.6,26300.9,6818.2,29129.0,56749.8,20164.6,28954.2,32772.7,11201.6,3610.1,5069.6,17741.3]))
POP_2022 = dict(zip(PROVINCES,[2184,1363,7420,3481,2401,4197,2348,3099,2475,8515,6577,6127,4188,4528,10163,9872,5844,6604,12657,5047,4027,3213,8374,3856,4693,3956,2492,595,728,2587]))
# 国家年度驱动变量由项目 data/全国年度驱动变量_来源数据.csv 提供；该文件记录单位、年份和来源。
ADJ = {
'Beijing':['Tianjin','Hebei'], 'Tianjin':['Beijing','Hebei'], 'Hebei':['Beijing','Tianjin','Shanxi','Inner Mongolia','Liaoning','Shandong','Henan'],
'Shanxi':['Hebei','Inner Mongolia','Shaanxi','Henan'], 'Inner Mongolia':['Hebei','Shanxi','Liaoning','Jilin','Heilongjiang','Ningxia','Shaanxi'],
'Liaoning':['Hebei','Inner Mongolia','Jilin'], 'Jilin':['Liaoning','Inner Mongolia','Heilongjiang'], 'Heilongjiang':['Jilin','Inner Mongolia'],
'Shanghai':['Jiangsu','Zhejiang'], 'Jiangsu':['Shandong','Anhui','Zhejiang','Shanghai'], 'Zhejiang':['Shanghai','Jiangsu','Anhui','Jiangxi','Fujian'],
'Anhui':['Henan','Hubei','Jiangsu','Zhejiang','Jiangxi'], 'Fujian':['Zhejiang','Jiangxi','Guangdong'], 'Jiangxi':['Hubei','Anhui','Zhejiang','Fujian','Hunan','Guangdong'],
'Shandong':['Hebei','Jiangsu','Henan','Anhui'], 'Henan':['Hebei','Shanxi','Shaanxi','Hubei','Anhui','Shandong'], 'Hubei':['Henan','Shaanxi','Chongqing','Hunan','Jiangxi','Anhui'],
'Hunan':['Hubei','Chongqing','Guizhou','Guangxi','Jiangxi','Guangdong'], 'Guangdong':['Fujian','Jiangxi','Hunan','Guangxi','Hainan'], 'Guangxi':['Yunnan','Guizhou','Hunan','Guangdong','Hainan'],
'Hainan':['Guangdong','Guangxi'], 'Chongqing':['Shaanxi','Hubei','Hunan','Guizhou','Sichuan'], 'Sichuan':['Qinghai','Gansu','Shaanxi','Chongqing','Guizhou','Yunnan'],
'Guizhou':['Sichuan','Chongqing','Yunnan','Guangxi','Hunan'], 'Yunnan':['Sichuan','Guizhou','Guangxi'], 'Shaanxi':['Inner Mongolia','Shanxi','Henan','Hubei','Chongqing','Sichuan','Gansu','Ningxia'],
'Gansu':['Inner Mongolia','Ningxia','Shaanxi','Sichuan','Qinghai','Xinjiang'], 'Qinghai':['Gansu','Sichuan','Xinjiang'], 'Ningxia':['Inner Mongolia','Shaanxi','Gansu'], 'Xinjiang':['Gansu','Qinghai']}


class SignConstrainedRidge:
    """STIRPAT-ridge with theory-consistent signs: population/GDP/coal nonnegative, clean energy nonpositive."""
    def __init__(self, alpha=1.0): self.alpha=float(alpha)
    def fit(self, X, y):
        X=np.asarray(X,float); y=np.asarray(y,float); X1=np.c_[np.ones(len(X)),X]
        p=X1.shape[1]
        def obj(b): return float(np.sum((y-X1@b)**2)+self.alpha*np.sum(b[1:]**2))
        bounds=[(None,None),(0,None),(0,None),(0,None),(None,0)]
        res=minimize(obj,np.zeros(p),method='L-BFGS-B',bounds=bounds)
        if not res.success: raise RuntimeError(res.message)
        self.intercept_=float(res.x[0]); self.coef_=np.asarray(res.x[1:],float); return self
    def predict(self,X): return self.intercept_+np.asarray(X,float)@self.coef_
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','Arial Unicode MS','DejaVu Sans'],'axes.unicode_minus':False,'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'figure.dpi':120})
COLORS = {'blue':'#2166AC','teal':'#1B9E77','orange':'#D95F02','purple':'#7570B3','red':'#B2182B','gray':'#666666','light':'#F0F0F0'}

def savefig(fig, path, size=(6.5,4.2)):
    export_figure(fig, str(path), formats=['svg','png'], size_inches=size, dpi=300, grayscale_preview=False, tight=True)
    from PIL import Image
    qa=Path(path).parent.parent/'figure_qa'; qa.mkdir(exist_ok=True,parents=True)
    Image.open(str(path)+'.png').convert('L').save(qa/(Path(path).name+'_grayscale.png'),dpi=(300,300))
    plt.close(fig)

def sha256(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

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

def moran(x, W):
    x=np.asarray(x,float); z=x-x.mean(); s0=W.sum(); return len(x)/s0*(z@W@z)/(z@z)

def spatial_tests(prov):
    idx={p:i for i,p in enumerate(PROVINCES)}; W=np.zeros((len(PROVINCES),len(PROVINCES)))
    for p,ns in ADJ.items():
        for q in ns:
            if p in idx and q in idx: W[idx[p],idx[q]]=1
    W=np.maximum(W,W.T); rows=[]
    for col in ['total_mt','per_capita_t','intensity_t_per_10k_yuan','coal_share_pct','process_share_pct']:
        x=prov.set_index('province').loc[PROVINCES,col].to_numpy(); I=moran(x,W); null=[]
        for _ in range(999): null.append(moran(RNG.permutation(x),W))
        p=(1+sum(abs(v)>=abs(I) for v in null))/(len(null)+1)
        rows.append({'indicator':col,'moran_I':I,'perm_p':p,'null_mean':np.mean(null),'null_sd':np.std(null)})
    # quartile groups for a non-parametric global difference diagnostic
    prov['scale_quartile']=pd.qcut(prov.total_mt,4,labels=False,duplicates='drop')+1
    kw=[]
    for col in ['per_capita_t','intensity_t_per_10k_yuan','coal_share_pct','process_share_pct']:
        groups=[g[col].to_numpy() for _,g in prov.groupby('scale_quartile',observed=True)]
        h,p=stats.kruskal(*groups); kw.append({'indicator':col,'kruskal_H':h,'kruskal_p':p})
    return pd.DataFrame(rows),pd.DataFrame(kw),W

def classify(prov):
    features=['total_mt','per_capita_t','intensity_t_per_10k_yuan','coal_share_pct','process_share_pct']
    X=np.log1p(prov[features].to_numpy()); Z=StandardScaler().fit_transform(X)
    pca=PCA(n_components=2,random_state=SEED); X2=pca.fit_transform(Z)
    Zlink=linkage(Z,method='ward')
    silhouette_rows=[]
    for k in range(2,7):
        lab=fcluster(Zlink,t=k,criterion='maxclust')
        silhouette_rows.append({'k':k,'silhouette':float(silhouette_score(Z,lab))})
    stability=pd.DataFrame(silhouette_rows)
    # Deterministic selection rule: maximize silhouette; break ties with smaller k.
    selected_k=int(stability.sort_values(['silhouette','k'],ascending=[False,True]).iloc[0].k)
    stability['selected']=stability.k.eq(selected_k)
    stability['selection_rule']='max silhouette; ties choose smaller k'
    labels=fcluster(Zlink,t=selected_k,criterion='maxclust')
    cent=pd.DataFrame(Z,columns=features).assign(cluster=labels).groupby('cluster')[features].mean()
    order=cent['total_mt'].sort_values(ascending=False).index.tolist(); remap={old:i+1 for i,old in enumerate(order)}
    prov['cluster']=pd.Series(labels,index=prov.index).map(remap).astype(int)
    cent2=prov.groupby('cluster')[features].mean().sort_index(); med=prov[features].median(); names={}
    for k,row in cent2.iterrows():
        high_total=row.total_mt>=med.total_mt; high_int=row.intensity_t_per_10k_yuan>=med.intensity_t_per_10k_yuan; high_coal=row.coal_share_pct>=med.coal_share_pct
        if high_total and (high_int or high_coal): base='高规模高压力型'
        elif high_coal: base='煤炭结构压力型'
        elif high_int: base='效率偏弱型'
        else: base='低规模相对低碳型'
        names[k]=base
        if list(names.values()).count(base)>1: names[k]=base+'（结构差异簇）'
    prov['class_name']=prov.cluster.map(names)
    prov['priority']=((prov.total_mt>=prov.total_mt.quantile(.8)) | (prov.intensity_t_per_10k_yuan>=prov.intensity_t_per_10k_yuan.quantile(.8))).map({True:'重点治理',False:'常规提升'})
    decision={'selected_k':selected_k,'rule':'max silhouette; ties choose smaller k','best_silhouette':float(stability.loc[stability.selected,'silhouette'].iloc[0])}
    return prov,pca,X2,Zlink,names,features,stability,decision
def drivers_and_model(annual, driver_path):
    """Fit the national model with fold-specific scaling for every validation split."""
    required=['year','gdp_trillion','population_million','coal_share','clean_share','source']
    dr=pd.read_csv(driver_path)
    if sorted(dr.columns.tolist()) != sorted(required):
        raise ValueError(f'驱动变量文件字段应为 {required}，实际为 {dr.columns.tolist()}')
    dr=dr[required].sort_values('year').reset_index(drop=True)
    if dr.year.duplicated().any() or dr[required[:-1]].isna().any().any():
        raise ValueError('驱动变量文件存在重复年份或缺失值')
    y=annual[(annual.year<=2024)&(annual.sector=='Total')][['year','total_mt']].copy()
    m=y.merge(dr,on='year',validate='one_to_one').sort_values('year').reset_index(drop=True)
    feats=['population_million','gdp_trillion','coal_share','clean_share']
    X=np.log(m[feats]); yy=np.log(m.total_mt)
    alphas=np.logspace(-4,4,80); best=None
    # 留一年法：每一折仅以训练年份拟合标准化器，杜绝测试年份特征泄漏。
    for a in alphas:
        errs=[]
        for i in range(len(m)):
            train=np.arange(len(m))!=i
            fold_scaler=StandardScaler().fit(X.iloc[train])
            mdl_i=SignConstrainedRidge(alpha=a).fit(fold_scaler.transform(X.iloc[train]),yy.iloc[train])
            errs.append(float(np.exp(mdl_i.predict(fold_scaler.transform(X.iloc[[i]]))[0])))
        mae=mean_absolute_error(m.total_mt,errs)
        if best is None or mae<best[0]: best=(mae,a)
    ridge_lambda=float(best[1])
    # 最终模型使用全样本标准化器；该对象仅服务于未来情景预测，不参与回测评分。
    scaler=StandardScaler().fit(X); Xz=scaler.transform(X)
    mdl=SignConstrainedRidge(alpha=ridge_lambda).fit(Xz,yy); fitted=np.exp(mdl.predict(Xz))
    coef=pd.DataFrame({'factor':['population','GDP','coal_share','clean_share'],'standardized_coefficient':mdl.coef_,'abs_importance':np.abs(mdl.coef_)})
    coef['importance_pct']=100*coef.abs_importance/coef.abs_importance.sum()
    # 严格扩展窗口一步回测：每个预测年份只能使用此前样本拟合标准化器和模型。
    rolling=[]; baseline=[]
    for i in range(3,len(m)):
        train=np.arange(i)
        fold_scaler=StandardScaler().fit(X.iloc[train])
        mdl_i=SignConstrainedRidge(alpha=ridge_lambda).fit(fold_scaler.transform(X.iloc[train]),yy.iloc[train])
        rolling.append(float(np.exp(mdl_i.predict(fold_scaler.transform(X.iloc[[i]]))[0])))
        trend_i=Ridge(alpha=ridge_lambda).fit(m.year.iloc[train].to_numpy().reshape(-1,1),yy.iloc[train])
        baseline.append(float(np.exp(trend_i.predict([[m.year.iloc[i]]])[0])))
    bt=m[['year','total_mt']].copy(); bt['pred_mt']=fitted; bt['residual_mt']=bt.total_mt-bt.pred_mt; bt['ape_pct']=100*np.abs(bt.residual_mt)/bt.total_mt
    bt['expanding_pred_mt']=np.nan; bt.loc[3:,'expanding_pred_mt']=rolling; bt['trend_baseline_mt']=np.nan; bt.loc[3:,'trend_baseline_mt']=baseline
    valid=bt.dropna(subset=['expanding_pred_mt'])
    pred_metrics={'ridge_lambda':ridge_lambda,'n_years':int(len(m)),'n_parameters_including_intercept':5,'fitted_MAE_Mt':float(mean_absolute_error(bt.total_mt,bt.pred_mt)),'fitted_RMSE_Mt':float(mean_squared_error(bt.total_mt,bt.pred_mt)**0.5),'fitted_MAPE_pct':float(bt.ape_pct.mean()),'expanding_MAE_Mt':float(mean_absolute_error(valid.total_mt,valid.expanding_pred_mt)),'trend_baseline_MAE_Mt':float(mean_absolute_error(valid.total_mt,valid.trend_baseline_mt)),'residual_sd_Mt':float(bt.residual_mt.std(ddof=1)),'validation_preprocessing':'StandardScaler fitted on each training fold only'}
    return dr,m,mdl,scaler,coef,bt,pred_metrics
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
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--input-root',required=True); ap.add_argument('--seed',type=int,default=SEED); ap.add_argument('--minimal',action='store_true'); args=ap.parse_args()
    global RNG; RNG=np.random.default_rng(args.seed)
    project=Path(args.project_root); input_root=Path(args.input_root); results=project/'results'; figures=project/'figures'; data_dir=project/'data'
    d,prov,max_err=load_inputs(input_root); y25,frac,frac_sd=annualize_2025(d)
    annual=d.groupby(['year','Sector'],as_index=False)['CO2 (Mt)'].sum().rename(columns={'Sector':'sector','CO2 (Mt)':'total_mt'}); annual=annual[annual.year<=2024].copy()
    ann25=pd.DataFrame([{'year':2025,'sector':'Total','total_mt':y25}]); annual=pd.concat([annual,ann25],ignore_index=True)
    spatial,kw,W=spatial_tests(prov); prov,pc,X2,Zlink,names,features,stability,cluster_decision=classify(prov)
    dr,m,mdl,scaler,coef,bt,pred_metrics=drivers_and_model(annual,project/'data'/'全国年度驱动变量_来源数据.csv'); forecast,cal=forecast_scenarios(mdl,scaler,y25,annual,pred_metrics['residual_sd_Mt'],results)
    q4_province,q4_class,q4_phase,q4_sector,q4_thresholds=build_q4_outputs(prov,annual,forecast)
    if args.minimal:
        print(json.dumps({'rows_input':len(d),'provinces':len(prov),'annualized_2025_mt':y25,'ridge_lambda':pred_metrics['ridge_lambda'],'expanding_backtest_mae_mt':pred_metrics['expanding_MAE_Mt'],'forecast_rows':len(forecast)},ensure_ascii=False)); return
    results.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True); data_dir.mkdir(parents=True,exist_ok=True); [p.unlink() for p in figures.glob('*_grayscale.png')]
    q4_province.to_csv(results/'问题4_省级政策映射.csv',index=False,encoding='utf-8-sig'); q4_class.to_csv(results/'问题4_类别政策覆盖率.csv',index=False,encoding='utf-8-sig'); q4_phase.to_csv(results/'问题4_阶段情景指标.csv',index=False,encoding='utf-8-sig'); q4_sector.to_csv(results/'问题4_部门优先级.csv',index=False,encoding='utf-8-sig'); stability.to_csv(results/'问题1_聚类决策.csv',index=False,encoding='utf-8-sig')
    # Save tables
    d.to_csv(data_dir/'附件1_清洗后.csv',index=False,encoding='utf-8-sig'); prov.to_csv(results/'问题1_省级指标与分类.csv',index=False,encoding='utf-8-sig'); spatial.to_csv(results/'问题1_Moran检验.csv',index=False,encoding='utf-8-sig'); kw.to_csv(results/'问题1_分组差异检验.csv',index=False,encoding='utf-8-sig'); pd.DataFrame({'component':['PC1','PC2'],'explained_variance_ratio':pc.explained_variance_ratio_}).to_csv(results/'问题1_PCA.csv',index=False); stability.to_csv(results/'问题1_聚类稳定性.csv',index=False); coef.to_csv(results/'问题2_驱动因素系数.csv',index=False,encoding='utf-8-sig'); bt.to_csv(results/'问题2_回测.csv',index=False,encoding='utf-8-sig'); dr.to_csv(data_dir/'external_driver_data.csv',index=False,encoding='utf-8-sig'); forecast.to_csv(results/'问题3_2026_2045_三情景预测.csv',index=False,encoding='utf-8-sig'); forecast.groupby('scenario').agg(path_valid=('path_valid','all'),min_coal=('coal_share','min'),max_clean=('clean_share','max'),max_energy_sum=('coal_share',lambda x: float((x+forecast.loc[x.index,'clean_share']).max()))).reset_index().to_csv(results/'问题3_情景约束检查.csv',index=False,encoding='utf-8-sig');
    metrics={'input_rows':len(d),'unique_dates':int(d.Date.nunique()),'units':{'emissions':'Mt CO2','gdp':'trillion yuan','forecast_intensity':'Mt CO2 per trillion yuan GDP','q1_intensity':'t CO2 per 10,000 yuan GDP','energy_shares':'fraction in [0,1]'},'total_consistency_max_abs_error':max_err,'annualized_2025_mt':y25,'jan_sep_fraction_mean':frac,'jan_sep_fraction_sd':frac_sd ,'spatial_weight_type':'land-border plus island bridge (Hainan-Guangdong/Guangxi)','spatial_moran':spatial.to_dict('records'),'kruskal':kw.to_dict('records'),'pca_explained':pc.explained_variance_ratio_.tolist(),'cluster_silhouette':stability.to_dict('records'),'cluster_decision':cluster_decision,'cluster_names':names,'q4_policy_model':{'thresholds':q4_thresholds,'province_policy_rows':int(len(q4_province)),'class_policy_rows':int(len(q4_class)),'phase_rows':int(len(q4_phase)),'sector_rows':int(len(q4_sector))},'model_selection':'STIRPAT-ridge retained as the energy-structure model; trend ridge is a pure-error baseline and is not used for q3','model_metrics':pred_metrics,'calibration_factor_2025':cal,'peak_by_scenario':{s:{'peak_year':int(g.loc[g.pred_mt.idxmax(),'year']),'peak_mt':float(g.pred_mt.max()),'intensity_2045':float(g.loc[g.year==2045,'intensity_mt_per_trillion_yuan'].iloc[0])} for s,g in forecast.groupby('scenario')}}
    (results/'关键指标.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    make_figures(annual,prov,spatial,coef,bt,forecast,q4_class,q4_phase,q4_sector,figures)
    manifest={'seed':args.seed,'created_at':datetime.now().isoformat(),'python':sys.version,'input_files':{str(p):sha256(p) for p in [input_root/'附件1-中国2019年-2025年碳排放数据.csv',input_root/'附件2-2022年30个省份排放清单.xlsx',project/'data'/'全国年度驱动变量_来源数据.csv']},'parameters':{'moran_permutations':999,'cluster_k':cluster_decision['selected_k'],'cluster_selection_rule':cluster_decision['rule'],'scenario_defs':'see carbon_model.py','annualization':'mean 2019-2024 Jan-Sep share'},'command':f'cd /d "{project}" && "E:\\Anaconda\\envs\\math_modeling\\python.exe" carbon_model.py --project-root "{project}" --input-root "{input_root}" --seed {args.seed}'}
    (results/'复现清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','input_rows':len(d),'provinces':len(prov),'annualized_2025_mt':y25,'backtest':pred_metrics,'peaks':metrics['peak_by_scenario']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()















