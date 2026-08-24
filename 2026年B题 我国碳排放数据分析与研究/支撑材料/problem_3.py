from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from problem_2 import drivers_and_model  # 复用 Q2 的拟合函数（代码依赖，非 CSV 依赖）

# ===== 本地数据加载（与 Q1/Q2/Q4 各自独立，互不读对方 CSV）=====
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


def forecast_scenarios(mdl, scaler, anchor_2025, annual, residual_sd, outdir):
    y25 = anchor_2025
    gdp25 = 134.9084 * 1.05
    pop25 = 140.8
    coal25 = .524
    clean25 = .296
    scen_defs = {'基准': (0.6, 0.6, [.045, .040, .035]), '低碳': (1.0, 1.0, [.045, .040, .035]), '强化低碳': (1.4, 1.4, [.043, .038, .032])}
    rows = []
    for name, (coal_step, clean_step, growths) in scen_defs.items():
        gdp = gdp25
        pop = pop25
        coal = coal25
        clean = clean25
        for year in range(2025, 2046):
            if year > 2025:
                gr = growths[0] if year <= 2030 else growths[1] if year <= 2035 else growths[2]
                gdp *= 1 + gr
                pop *= 1 + 0.001
                coal = max(.12, coal - coal_step / 100)
                clean = min(.78, clean + clean_step / 100)
            X = np.log([[pop, gdp, coal, clean]])
            pred = float(np.exp(mdl.predict(scaler.transform(X))[0]))
            rows.append({'scenario': name, 'year': year, 'gdp_trillion': gdp, 'population_million': pop,
                         'coal_share': coal, 'clean_share': clean, 'pred_mt_raw': pred})
    f = pd.DataFrame(rows)
    scale = y25 / f[f.year == 2025].pred_mt_raw.mean()
    f['pred_mt'] = f.pred_mt_raw * scale
    f['intensity_mt_per_trillion_yuan'] = f.pred_mt / f.gdp_trillion
    f['intensity_base_2025'] = f.groupby('scenario')['intensity_mt_per_trillion_yuan'].transform('first')
    f['intensity_reduction_pct'] = 100 * (1 - f.intensity_mt_per_trillion_yuan / f.intensity_base_2025)
    f['total_yoy_pct'] = f.groupby('scenario').pred_mt.pct_change() * 100
    f['row_type'] = np.where(f.year == 2025, 'anchor', 'forecast')
    horizon = np.maximum(1, (f.year - 2025).to_numpy())
    f['lower_mt'] = np.maximum(0, f.pred_mt - 1.96 * residual_sd * np.sqrt(horizon))
    f['upper_mt'] = f.pred_mt + 1.96 * residual_sd * np.sqrt(horizon)
    f['path_valid'] = (f.gdp_trillion > 0) & (f.population_million > 0) & (f.coal_share.between(0, 1)) & (f.clean_share.between(0, 1)) & ((f.coal_share + f.clean_share) <= 1) & (f.pred_mt >= 0)
    return f, scale


def main() -> None:
    parser = argparse.ArgumentParser(description='问题3：三情景碳排放预测')
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
    forecast, calibration = forecast_scenarios(mdl, scaler, annualized_2025, annual, metrics['residual_sd_Mt'], args.project_root / 'results')

    results = args.project_root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(results / 'q3_2026_2045_三情景预测.csv', index=False, encoding='utf-8-sig')
    chk = forecast.groupby('scenario').agg(
        path_valid=('path_valid', 'all'),
        min_coal=('coal_share', 'min'),
        max_clean=('clean_share', 'max'),
        max_energy_sum=('coal_share', lambda x: float((x + forecast.loc[x.index, 'clean_share']).max())),
    ).reset_index()
    chk.to_csv(results / 'q3_情景约束检查.csv', index=False, encoding='utf-8-sig')

    print(json.dumps({'question': 3, 'forecast_rows': int(len(forecast)), 'calibration': calibration,
                      'ridge_lambda': metrics['ridge_lambda'], 'expanding_MAE_Mt': metrics['expanding_MAE_Mt']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
