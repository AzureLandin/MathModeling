from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ===== 本地数据加载（与 Q1/Q3/Q4 各自独立，互不读对方 CSV）=====
PROVINCES = ['Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia', 'Liaoning', 'Jilin',
             'Heilongjiang', 'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi',
             'Shandong', 'Henan', 'Hubei', 'Hunan', 'Guangdong', 'Guangxi', 'Hainan', 'Chongqing',
             'Sichuan', 'Guizhou', 'Yunnan', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang']
GDP_2022 = dict(zip(PROVINCES, [41610.9, 16311.3, 42370.4, 25642.6, 23158.6, 28975.1, 13070.2, 15901.0,
                               44652.8, 122875.6, 77715.4, 45045.0, 53109.9, 32074.7, 87435.1, 61345.1,
                               53734.9, 48670.4, 129118.6, 26300.9, 6818.2, 29129.0, 56749.8, 20164.6,
                               28954.2, 32772.7, 11201.6, 3610.1, 5069.6, 17741.3]))
POP_2022 = dict(zip(PROVINCES, [2184, 1363, 7420, 3481, 2401, 4197, 2348, 3099, 2475, 8515, 6577, 6127,
                               4188, 4528, 10163, 9872, 5844, 6604, 12657, 5047, 4027, 3213, 8374, 3856,
                               4693, 3956, 2492, 595, 728, 2587]))
SEED = 20260822


def load_inputs(input_root: Path):
    csv_path = input_root / '附件1-中国2019年-2025年碳排放数据.csv'
    xlsx_path = input_root / '附件2-2022年30个省份排放清单.xlsx'
    d = pd.read_csv(csv_path, parse_dates=['Date'])
    assert len(d) == 17255 and d.Date.nunique() == 2465
    assert set(d.Sector.unique()) == {'Domestic Aviation', 'Ground Transport', 'Industry', 'International Aviation', 'Power', 'Residential', 'Total'}
    d['year'] = d.Date.dt.year
    d['month'] = d.Date.dt.month
    check = d.pivot_table(index='Date', columns='Sector', values='CO2 (Mt)', aggfunc='sum')
    sector_sum = check[['Domestic Aviation', 'Ground Transport', 'Industry', 'Power', 'Residential']].sum(axis=1)
    max_err = float(np.max(np.abs(sector_sum - check['Total'])))
    assert max_err < 1e-5, max_err
    xls = pd.ExcelFile(xlsx_path)
    province_rows = []
    coal_cols = ['Raw_Coal', 'CleanedCoal', 'Other_Washed_Coal', 'Briquettes', 'Coke', 'Coke_Oven_Gas', 'Other_Gas', 'Other_Coking_Products']
    oil_cols = ['Crude_Oil', 'Gasoline', 'Kerosene', 'Diesel_Oil', 'Fuel_Oil', 'LPG', 'Refinery_Gas', 'Other_Petroleum_Products']
    gas_cols = ['Natural_Gas']
    for s in xls.sheet_names:
        if s == 'NOTE':
            continue
        raw = pd.read_excel(xlsx_path, sheet_name=s, header=None)
        cols = raw.iloc[0].tolist()
        row = raw.iloc[3].tolist()
        rec = dict(zip(cols, row))
        assert 'Scope_1_Total' in rec and rec['Scope_1_Total'] is not None
        province_rows.append({'province': s.replace('2022', ''), 'total_mt': float(rec['Scope_1_Total'] or 0),
                              'coal_mt': sum(float(rec.get(c, 0) or 0) for c in coal_cols), 'oil_mt': sum(float(rec.get(c, 0) or 0) for c in oil_cols),
                              'gas_mt': sum(float(rec.get(c, 0) or 0) for c in gas_cols), 'process_mt': float(rec.get('Process', 0) or 0),
                              'other_mt': sum(float(rec.get(c, 0) or 0) for c in ['Scope_2_Heat', 'Scope_2_Electricity', 'Other_Energy'])})
    prov = pd.DataFrame(province_rows)
    name_map = {'Shanghai': 'Shanghai', 'Yunnan': 'Yunnan', 'InnerMongolia': 'Inner Mongolia', 'Beijing': 'Beijing', 'Jilin': 'Jilin', 'Sichuan': 'Sichuan', 'Tianjin': 'Tianjin', 'Ningxia': 'Ningxia', 'Anhui': 'Anhui', 'Shandong': 'Shandong', 'Shanxi': 'Shanxi', 'Guangdong': 'Guangdong', 'Guangxi': 'Guangxi', 'Xinjiang': 'Xinjiang', 'Jiangsu': 'Jiangsu', 'Jiangxi': 'Jiangxi', 'Hebei': 'Hebei', 'Henan': 'Henan', 'Zhejiang': 'Zhejiang', 'Hainan': 'Hainan', 'Hubei': 'Hubei', 'Hunan': 'Hunan', 'Gansu': 'Gansu', 'Fujian': 'Fujian', 'Guizhou': 'Guizhou', 'Liaoning': 'Liaoning', 'Chongqing': 'Chongqing', 'Shaanxi': 'Shaanxi', 'Qinghai': 'Qinghai', 'Heilongjiang': 'Heilongjiang'}
    prov['province'] = prov.province.map(name_map)
    assert prov.province.notna().all() and len(prov) == 30
    prov['gdp_2022_100m'] = prov.province.map(GDP_2022)
    prov['pop_2022_10w'] = prov.province.map(POP_2022)
    assert prov.gdp_2022_100m.notna().all() and prov.pop_2022_10w.notna().all()
    prov['share_pct'] = 100 * prov.total_mt / prov.total_mt.sum()
    prov['per_capita_t'] = 100 * prov.total_mt / prov.pop_2022_10w
    prov['intensity_t_per_10k_yuan'] = 100 * prov.total_mt / prov.gdp_2022_100m
    prov['coal_share_pct'] = 100 * prov.coal_mt / prov.total_mt
    prov['process_share_pct'] = 100 * prov.process_mt / prov.total_mt
    el_cols = ['intensity_t_per_10k_yuan', 'coal_share_pct', 'process_share_pct']
    prov['economic_linkage_index'] = StandardScaler().fit_transform(prov[el_cols]).mean(axis=1)
    return d, prov, max_err


def annualize_2025(df):
    d = df[df.Sector == 'Total'].copy()
    d['year'] = d.Date.dt.year
    d['month'] = d.Date.dt.month
    hist = d[d.year <= 2024].groupby('year')['CO2 (Mt)'].sum()
    frac = d[d.year <= 2024].query('month<=9').groupby('year')['CO2 (Mt)'].sum() / hist
    y25 = d[d.year == 2025].query('month<=9')['CO2 (Mt)'].sum() / frac.mean()
    return float(y25), float(frac.mean()), float(frac.std(ddof=1))


def build_annual(daily, annualized_2025):
    annual = daily.groupby(['year', 'Sector'], as_index=False)['CO2 (Mt)'].sum().rename(columns={'Sector': 'sector', 'CO2 (Mt)': 'total_mt'})
    annual = annual[annual.year <= 2024].copy()
    annual = pd.concat([annual, pd.DataFrame([{'year': 2025, 'sector': 'Total', 'total_mt': annualized_2025}])], ignore_index=True)
    return annual


class SignConstrainedRidge:
    """STIRPAT-ridge with theory-consistent signs: population/GDP/coal nonnegative, clean energy nonpositive."""
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        X1 = np.c_[np.ones(len(X)), X]
        p = X1.shape[1]

        def obj(b):
            return float(np.sum((y - X1 @ b) ** 2) + self.alpha * np.sum(b[1:] ** 2))

        bounds = [(None, None), (0, None), (0, None), (0, None), (None, 0)]
        res = minimize(obj, np.zeros(p), method='L-BFGS-B', bounds=bounds)
        if not res.success:
            raise RuntimeError(res.message)
        self.intercept_ = float(res.x[0])
        self.coef_ = np.asarray(res.x[1:], float)
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, float) @ self.coef_


def drivers_and_model(annual, driver_path):
    """Fit the national model with fold-specific scaling for every validation split."""
    required = ['year', 'gdp_trillion', 'population_million', 'coal_share', 'clean_share', 'source']
    dr = pd.read_csv(driver_path)
    if sorted(dr.columns.tolist()) != sorted(required):
        raise ValueError(f'驱动变量文件字段应为 {required}，实际为 {dr.columns.tolist()}')
    dr = dr[required].sort_values('year').reset_index(drop=True)
    if dr.year.duplicated().any() or dr[required[:-1]].isna().any().any():
        raise ValueError('驱动变量文件存在重复年份或缺失值')
    y = annual[(annual.year <= 2024) & (annual.sector == 'Total')][['year', 'total_mt']].copy()
    m = y.merge(dr, on='year', validate='one_to_one').sort_values('year').reset_index(drop=True)
    feats = ['population_million', 'gdp_trillion', 'coal_share', 'clean_share']
    X = np.log(m[feats])
    yy = np.log(m.total_mt)
    alphas = np.logspace(-4, 4, 80)
    best = None
    # 留一年法：每一折仅以训练年份拟合标准化器，杜绝测试年份特征泄漏。
    for a in alphas:
        errs = []
        for i in range(len(m)):
            train = np.arange(len(m)) != i
            fold_scaler = StandardScaler().fit(X.iloc[train])
            mdl_i = SignConstrainedRidge(alpha=a).fit(fold_scaler.transform(X.iloc[train]), yy.iloc[train])
            errs.append(float(np.exp(mdl_i.predict(fold_scaler.transform(X.iloc[[i]]))[0])))
        mae = mean_absolute_error(m.total_mt, errs)
        if best is None or mae < best[0]:
            best = (mae, a)
    ridge_lambda = float(best[1])
    # 最终模型使用全样本标准化器；该对象仅服务于未来情景预测，不参与回测评分。
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    mdl = SignConstrainedRidge(alpha=ridge_lambda).fit(Xz, yy)
    fitted = np.exp(mdl.predict(Xz))
    coef = pd.DataFrame({'factor': ['population', 'GDP', 'coal_share', 'clean_share'], 'standardized_coefficient': mdl.coef_, 'abs_importance': np.abs(mdl.coef_)})
    coef['importance_pct'] = 100 * coef.abs_importance / coef.abs_importance.sum()
    rolling = []
    baseline = []
    for i in range(3, len(m)):
        train = np.arange(i)
        fold_scaler = StandardScaler().fit(X.iloc[train])
        mdl_i = SignConstrainedRidge(alpha=ridge_lambda).fit(fold_scaler.transform(X.iloc[train]), yy.iloc[train])
        rolling.append(float(np.exp(mdl_i.predict(fold_scaler.transform(X.iloc[[i]]))[0])))
        trend_i = Ridge(alpha=ridge_lambda).fit(m.year.iloc[train].to_numpy().reshape(-1, 1), yy.iloc[train])
        baseline.append(float(np.exp(trend_i.predict([[m.year.iloc[i]]])[0])))
    bt = m[['year', 'total_mt']].copy()
    bt['pred_mt'] = fitted
    bt['residual_mt'] = bt.total_mt - bt.pred_mt
    bt['ape_pct'] = 100 * np.abs(bt.residual_mt) / bt.total_mt
    bt['expanding_pred_mt'] = np.nan
    bt.loc[3:, 'expanding_pred_mt'] = rolling
    bt['trend_baseline_mt'] = np.nan
    bt.loc[3:, 'trend_baseline_mt'] = baseline
    valid = bt.dropna(subset=['expanding_pred_mt'])
    pred_metrics = {'ridge_lambda': ridge_lambda, 'n_years': int(len(m)), 'n_parameters_including_intercept': 5,
                    'fitted_MAE_Mt': float(mean_absolute_error(bt.total_mt, bt.pred_mt)),
                    'fitted_RMSE_Mt': float(mean_squared_error(bt.total_mt, bt.pred_mt) ** 0.5),
                    'fitted_MAPE_pct': float(bt.ape_pct.mean()),
                    'expanding_MAE_Mt': float(mean_absolute_error(valid.total_mt, valid.expanding_pred_mt)),
                    'trend_baseline_MAE_Mt': float(mean_absolute_error(valid.total_mt, valid.trend_baseline_mt)),
                    'residual_sd_Mt': float(bt.residual_mt.std(ddof=1)),
                    'validation_preprocessing': 'StandardScaler fitted on each training fold only'}
    return dr, m, mdl, scaler, coef, bt, pred_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='问题2：约束岭回归驱动因素辨识与回测')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--input-root', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.input_root is None:
        args.input_root = args.project_root / 'data'

    d, prov, max_err = load_inputs(args.input_root)
    annualized_2025, jan_sep_fraction, jan_sep_fraction_sd = annualize_2025(d)
    annual = build_annual(d, annualized_2025)
    driver_path = args.project_root / 'data' / '全国年度驱动变量_来源数据.csv'

    dr, m, mdl, scaler, coef, bt, metrics = drivers_and_model(annual, driver_path)

    results = args.project_root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    coef.to_csv(results / 'q2_驱动因素系数.csv', index=False, encoding='utf-8-sig')
    bt.to_csv(results / 'q2_回测.csv', index=False, encoding='utf-8-sig')
    (args.project_root / 'data').mkdir(parents=True, exist_ok=True)
    dr.to_csv(args.project_root / 'data' / 'external_driver_data.csv', index=False, encoding='utf-8-sig')

    print(json.dumps({'question': 2, **metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
