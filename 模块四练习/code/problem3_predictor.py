"""
问题三求解器：基于偏差序列ARIMA的供应方案预测
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def build_deviation_series(order_data, supply_data, indices):
    deviation_series = []
    valid_weeks = []
    
    for idx in indices:
        mask = order_data[idx, :] > 0
        O = order_data[idx, mask]
        S = supply_data[idx, mask]
        
        D = S - O
        deviation_series.append(D)
        valid_weeks.append(len(D))
    
    return deviation_series, valid_weeks


def fit_supplier_model(D_series, min_obs=30, sig_level=0.05):
    N = len(D_series)
    mean_D = np.mean(D_series)
    
    if N < min_obs:
        forecast = np.full(24, mean_D)
        residuals = D_series - mean_D
        return "均值(短序列)", forecast, residuals, {
            'adf_p': np.nan, 'lb_p': np.nan, 'aic': np.nan,
            'order': 'N/A', 'mean_D': mean_D, 'N': N
        }
    
    try:
        adf_result = adfuller(D_series, maxlag=20)
        adf_p = adf_result[1]
    except:
        adf_p = 1.0
    
    d = 0 if adf_p < sig_level else 1
    
    if d == 1:
        D_diff = np.diff(D_series)
    else:
        D_diff = D_series
    
    try:
        lb_result = acorr_ljungbox(D_diff, lags=min(10, len(D_diff)-1))
        lb_p = lb_result.iloc[-1, 1]
    except:
        lb_p = 0.0
    
    if lb_p > sig_level:
        forecast = np.full(24, mean_D)
        residuals = D_series - mean_D
        return f"均值(白噪声,d={d})", forecast, residuals, {
            'adf_p': adf_p, 'lb_p': lb_p, 'aic': np.nan,
            'order': 'N/A', 'mean_D': mean_D, 'N': N
        }
    
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for p in range(4):
        for q in range(4):
            try:
                model = ARIMA(D_series, order=(p, d, q))
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
                    best_model = fit
            except:
                continue
    
    if best_model is None:
        forecast = np.full(24, mean_D)
        residuals = D_series - mean_D
        return f"均值(ARIMA失败,d={d})", forecast, residuals, {
            'adf_p': adf_p, 'lb_p': lb_p, 'aic': np.nan,
            'order': 'N/A', 'mean_D': mean_D, 'N': N
        }
    
    forecast = best_model.forecast(steps=24)
    residuals = best_model.resid
    
    return f"ARIMA{best_order}", forecast, residuals, {
        'adf_p': adf_p, 'lb_p': lb_p, 'aic': best_aic,
        'order': str(best_order), 'mean_D': mean_D, 'N': N
    }


def fit_all_suppliers(deviation_series, valid_weeks, min_obs=30):
    models = []
    
    for i, (D, N) in enumerate(zip(deviation_series, valid_weeks)):
        model_type, forecast, residuals, params = fit_supplier_model(D, min_obs)
        models.append({
            'model_type': model_type,
            'forecast': forecast,
            'residuals': residuals,
            'params': params
        })
        
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/55 家供应商建模")
    
    return models


def predict_supply(models, Q, x):
    n = len(Q)
    T = 24
    S_pred = np.zeros((n, T))
    
    for i in range(n):
        D_hat = models[i]['forecast']
        S = np.maximum(0, Q[i] + D_hat) * x[i, :]
        S_pred[i, :] = S
    
    return S_pred


def monte_carlo_simulation(models, Q, u, x, M=10000, seed=42):
    np.random.seed(seed)
    
    n = len(Q)
    T = 24
    T_sim = np.zeros((M, T))
    
    print(f"  开始{M}次蒙特卡洛模拟...")
    
    for m in range(M):
        weekly_product = np.zeros(T)
        
        for i in range(n):
            D_hat = models[i]['forecast']
            resid = models[i]['residuals']
            
            if len(resid) > 1:
                noise = np.random.choice(resid, size=T, replace=True)
            else:
                noise = 0
            
            S_sim = np.maximum(0, Q[i] + D_hat + noise) * x[i, :]
            weekly_product += S_sim / u[i]
        
        T_sim[m, :] = weekly_product
    
    P_j = (T_sim >= 28200).mean(axis=0)
    
    return T_sim, P_j


def compute_supply_plan(S_pred):
    S_rounded = np.round(S_pred, 6)
    supply = np.ceil(S_rounded).astype(int)
    return supply


def compute_capacity_stats(supply, u, T_sim, P_j, capacity=28200):
    weekly_total = (supply / u[:, np.newaxis]).sum(axis=0)
    fulfillment_rate = weekly_total / capacity
    weeks_meet = (weekly_total >= capacity).sum()
    
    avg_P = P_j.mean()
    P_all = np.prod(P_j)
    
    return weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all


def compute_confidence_intervals(T_sim, percentiles=[5, 25, 50, 75, 95]):
    ci = np.percentile(T_sim, percentiles, axis=0)
    return ci
