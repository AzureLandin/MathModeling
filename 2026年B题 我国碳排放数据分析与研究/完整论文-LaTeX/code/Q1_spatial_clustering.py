from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from All_data_processing import PROVINCES

SEED = 20260822
RNG = np.random.default_rng(SEED)
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

def set_seed(seed: int) -> None:
    """Reset the permutation generator used by Question 1."""
    global RNG
    RNG = np.random.default_rng(seed)

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
