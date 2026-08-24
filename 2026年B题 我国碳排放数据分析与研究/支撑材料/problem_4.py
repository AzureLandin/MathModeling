from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from problem_1 import classify  # 复用 Q1 的分类函数（代码依赖，非 CSV 依赖）
from problem_2 import drivers_and_model  # 复用 Q2 的拟合函数（代码依赖，非 CSV 依赖）
from problem_3 import forecast_scenarios  # 复用 Q3 的预测函数（代码依赖，非 CSV 依赖）

# ===== 本地数据加载（与 Q1/Q2/Q3 各自独立，互不读对方 CSV）=====
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


def build_q4_outputs(prov, annual, forecast):
    """Build reproducible policy flags and phase indicators from q1--q3 outputs."""
    thresholds = {'intensity_median': float(prov.intensity_t_per_10k_yuan.median()), 'coal_share_median': float(prov.coal_share_pct.median()),
                  'linkage_median': float(prov.economic_linkage_index.median()), 'total_median': float(prov.total_mt.median())}
    policy = prov[['province', 'class_name', 'priority', 'total_mt', 'intensity_t_per_10k_yuan', 'coal_share_pct', 'economic_linkage_index', 'process_share_pct']].copy()
    policy['总量控制'] = policy.priority.eq('重点治理')
    policy['能效改造'] = policy.intensity_t_per_10k_yuan.ge(thresholds['intensity_median'])
    policy['煤炭替代'] = policy.coal_share_pct.ge(thresholds['coal_share_median'])
    policy['市场机制'] = policy.economic_linkage_index.ge(thresholds['linkage_median'])
    policy['需求侧治理'] = policy.total_mt.ge(thresholds['total_median']) & policy.coal_share_pct.lt(thresholds['coal_share_median'])
    policy_cols = ['总量控制', '能效改造', '煤炭替代', '市场机制', '需求侧治理']
    policy['政策工具数'] = policy[policy_cols].sum(axis=1).astype(int)
    policy['政策规则'] = '总量:重点治理；能效:强度≥中位数；煤炭:煤炭占比≥中位数；市场:经济联系≥中位数；需求侧:总量≥中位数且煤炭占比<中位数'
    class_policy = policy.groupby('class_name')[policy_cols].mean().mul(100).reset_index()
    class_policy[policy_cols] = class_policy[policy_cols].round(1)
    class_policy['省份数量'] = policy.groupby('class_name').size().reindex(class_policy.class_name).to_numpy()
    sector_priority = annual[annual.sector != 'Total'].groupby('sector', as_index=False).total_mt.sum().sort_values('total_mt', ascending=False)
    sector_priority['share_pct'] = 100 * sector_priority.total_mt / sector_priority.total_mt.sum()
    sector_priority['rank'] = np.arange(1, len(sector_priority) + 1)
    phase_rows = []
    phase_defs = [('2026--2030', 2030), ('2031--2035', 2035), ('2036--2045', 2045)]
    for scenario, g in forecast.groupby('scenario'):
        base = float(g.loc[g.year == 2025, 'intensity_mt_per_trillion_yuan'].iloc[0])
        for phase, end_year in phase_defs:
            row = g.loc[g.year == end_year].iloc[0]
            phase_rows.append({'scenario': scenario, 'phase': phase, 'end_year': end_year, 'total_mt': float(row.pred_mt),
                               'intensity_mt_per_trillion_yuan': float(row.intensity_mt_per_trillion_yuan),
                               'intensity_index_2025': float(100 * row.intensity_mt_per_trillion_yuan / base),
                               'intensity_reduction_pct': float(row.intensity_reduction_pct),
                               'coal_share_pct': float(100 * row.coal_share), 'clean_share_pct': float(100 * row.clean_share)})
    return policy, class_policy, pd.DataFrame(phase_rows), sector_priority, thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description='问题4：省域—部门—阶段政策映射')
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

    province, pca, _, _, class_names, _, stability, cluster_decision = classify(prov)
    dr, m, mdl, scaler, coef, bt, metrics = drivers_and_model(annual, driver_path)
    forecast, calibration = forecast_scenarios(mdl, scaler, annualized_2025, annual, metrics['residual_sd_Mt'], args.project_root / 'results')
    q4_province, q4_class, q4_phase, q4_sector, q4_thresholds = build_q4_outputs(province, annual, forecast)

    results = args.project_root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    q4_province.to_csv(results / 'q4_省级政策映射.csv', index=False, encoding='utf-8-sig')
    q4_class.to_csv(results / 'q4_类别政策覆盖率.csv', index=False, encoding='utf-8-sig')
    q4_phase.to_csv(results / 'q4_阶段情景指标.csv', index=False, encoding='utf-8-sig')
    q4_sector.to_csv(results / 'q4_部门优先级.csv', index=False, encoding='utf-8-sig')

    print(json.dumps({'question': 4, 'province_rows': int(len(q4_province)), 'class_rows': int(len(q4_class)),
                      'phase_rows': int(len(q4_phase)), 'sector_rows': int(len(q4_sector))}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
