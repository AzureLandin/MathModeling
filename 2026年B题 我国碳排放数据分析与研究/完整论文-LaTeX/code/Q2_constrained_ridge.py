from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

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
