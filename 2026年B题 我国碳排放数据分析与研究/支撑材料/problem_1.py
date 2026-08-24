from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ===== 本地数据加载（与 Q2/Q3/Q4 各自独立，互不读对方 CSV）=====
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
RNG = np.random.default_rng(SEED)
ADJ = {
    'Beijing': ['Tianjin', 'Hebei'], 'Tianjin': ['Beijing', 'Hebei'], 'Hebei': ['Beijing', 'Tianjin', 'Shanxi', 'Inner Mongolia', 'Liaoning', 'Shandong', 'Henan'],
    'Shanxi': ['Hebei', 'Inner Mongolia', 'Shaanxi', 'Henan'], 'Inner Mongolia': ['Hebei', 'Shanxi', 'Liaoning', 'Jilin', 'Heilongjiang', 'Ningxia', 'Shaanxi'],
    'Liaoning': ['Hebei', 'Inner Mongolia', 'Jilin'], 'Jilin': ['Liaoning', 'Inner Mongolia', 'Heilongjiang'], 'Heilongjiang': ['Jilin', 'Inner Mongolia'],
    'Shanghai': ['Jiangsu', 'Zhejiang'], 'Jiangsu': ['Shandong', 'Anhui', 'Zhejiang', 'Shanghai'], 'Zhejiang': ['Shanghai', 'Jiangsu', 'Anhui', 'Jiangxi', 'Fujian'],
    'Anhui': ['Henan', 'Hubei', 'Jiangsu', 'Zhejiang', 'Jiangxi'], 'Fujian': ['Zhejiang', 'Jiangxi', 'Guangdong'], 'Jiangxi': ['Hubei', 'Anhui', 'Zhejiang', 'Fujian', 'Hunan', 'Guangdong'],
    'Shandong': ['Hebei', 'Jiangsu', 'Henan', 'Anhui'], 'Henan': ['Hebei', 'Shanxi', 'Shaanxi', 'Hubei', 'Anhui', 'Shandong'], 'Hubei': ['Henan', 'Shaanxi', 'Chongqing', 'Hunan', 'Jiangxi', 'Anhui'],
    'Hunan': ['Hubei', 'Chongqing', 'Guizhou', 'Guangxi', 'Jiangxi', 'Guangdong'], 'Guangdong': ['Fujian', 'Jiangxi', 'Hunan', 'Guangxi', 'Hainan'], 'Guangxi': ['Yunnan', 'Guizhou', 'Hunan', 'Guangdong', 'Hainan'],
    'Hainan': ['Guangdong', 'Guangxi'], 'Chongqing': ['Shaanxi', 'Hubei', 'Hunan', 'Guizhou', 'Sichuan'], 'Sichuan': ['Qinghai', 'Gansu', 'Shaanxi', 'Chongqing', 'Guizhou', 'Yunnan'],
    'Guizhou': ['Sichuan', 'Chongqing', 'Yunnan', 'Guangxi', 'Hunan'], 'Yunnan': ['Sichuan', 'Guizhou', 'Guangxi'], 'Shaanxi': ['Inner Mongolia', 'Shanxi', 'Henan', 'Hubei', 'Chongqing', 'Sichuan', 'Gansu', 'Ningxia'],
    'Gansu': ['Inner Mongolia', 'Ningxia', 'Shaanxi', 'Sichuan', 'Qinghai', 'Xinjiang'], 'Qinghai': ['Gansu', 'Sichuan', 'Xinjiang'], 'Ningxia': ['Inner Mongolia', 'Shaanxi', 'Gansu'], 'Xinjiang': ['Gansu', 'Qinghai']}


def set_seed(seed: int) -> None:
    """重置问题1使用的置换随机种子。"""
    global RNG
    RNG = np.random.default_rng(seed)


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


def moran(x, W):
    x = np.asarray(x, float)
    z = x - x.mean()
    s0 = W.sum()
    return len(x) / s0 * (z @ W @ z) / (z @ z)


def spatial_tests(prov):
    idx = {p: i for i, p in enumerate(PROVINCES)}
    W = np.zeros((len(PROVINCES), len(PROVINCES)))
    for p, ns in ADJ.items():
        for q in ns:
            if p in idx and q in idx:
                W[idx[p], idx[q]] = 1
    W = np.maximum(W, W.T)
    rows = []
    for col in ['total_mt', 'per_capita_t', 'intensity_t_per_10k_yuan', 'coal_share_pct', 'process_share_pct']:
        x = prov.set_index('province').loc[PROVINCES, col].to_numpy()
        I = moran(x, W)
        null = []
        for _ in range(999):
            null.append(moran(RNG.permutation(x), W))
        p = (1 + sum(abs(v) >= abs(I) for v in null)) / (len(null) + 1)
        rows.append({'indicator': col, 'moran_I': I, 'perm_p': p, 'null_mean': np.mean(null), 'null_sd': np.std(null)})
    prov['scale_quartile'] = pd.qcut(prov.total_mt, 4, labels=False, duplicates='drop') + 1
    kw = []
    for col in ['per_capita_t', 'intensity_t_per_10k_yuan', 'coal_share_pct', 'process_share_pct']:
        groups = [g[col].to_numpy() for _, g in prov.groupby('scale_quartile', observed=True)]
        h, p = stats.kruskal(*groups)
        kw.append({'indicator': col, 'kruskal_H': h, 'kruskal_p': p})
    return pd.DataFrame(rows), pd.DataFrame(kw), W


def classify(prov):
    features = ['total_mt', 'per_capita_t', 'intensity_t_per_10k_yuan', 'coal_share_pct', 'process_share_pct']
    X = np.log1p(prov[features].to_numpy())
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    X2 = pca.fit_transform(Z)
    Zlink = linkage(Z, method='ward')
    silhouette_rows = []
    for k in range(2, 7):
        lab = fcluster(Zlink, t=k, criterion='maxclust')
        silhouette_rows.append({'k': k, 'silhouette': float(silhouette_score(Z, lab))})
    stability = pd.DataFrame(silhouette_rows)
    selected_k = int(stability.sort_values(['silhouette', 'k'], ascending=[False, True]).iloc[0].k)
    stability['selected'] = stability.k.eq(selected_k)
    stability['selection_rule'] = 'max silhouette; ties choose smaller k'
    labels = fcluster(Zlink, t=selected_k, criterion='maxclust')
    cent = pd.DataFrame(Z, columns=features).assign(cluster=labels).groupby('cluster')[features].mean()
    order = cent['total_mt'].sort_values(ascending=False).index.tolist()
    remap = {old: i + 1 for i, old in enumerate(order)}
    prov['cluster'] = pd.Series(labels, index=prov.index).map(remap).astype(int)
    cent2 = prov.groupby('cluster')[features].mean().sort_index()
    med = prov[features].median()
    names = {}
    for k, row in cent2.iterrows():
        high_total = row.total_mt >= med.total_mt
        high_int = row.intensity_t_per_10k_yuan >= med.intensity_t_per_10k_yuan
        high_coal = row.coal_share_pct >= med.coal_share_pct
        if high_total and (high_int or high_coal):
            base = '高规模高压力型'
        elif high_coal:
            base = '煤炭结构压力型'
        elif high_int:
            base = '效率偏弱型'
        else:
            base = '低规模相对低碳型'
        names[k] = base
        if list(names.values()).count(base) > 1:
            names[k] = base + '（结构差异簇）'
    prov['class_name'] = prov.cluster.map(names)
    prov['priority'] = ((prov.total_mt >= prov.total_mt.quantile(.8)) | (prov.intensity_t_per_10k_yuan >= prov.intensity_t_per_10k_yuan.quantile(.8))).map({True: '重点治理', False: '常规提升'})
    decision = {'selected_k': selected_k, 'rule': 'max silhouette; ties choose smaller k', 'best_silhouette': float(stability.loc[stability.selected, 'silhouette'].iloc[0])}
    return prov, pca, X2, Zlink, names, features, stability, decision


def main() -> None:
    parser = argparse.ArgumentParser(description='问题1：省域空间聚类与分类')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent.parent, help='工程根目录（含 data/ 与 results/）')
    parser.add_argument('--input-root', type=Path, default=None, help='原始附件所在目录；默认 <project-root>/data')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.input_root is None:
        args.input_root = args.project_root / 'data'
    set_seed(args.seed)
    project = args.project_root
    results = project / 'results'
    results.mkdir(parents=True, exist_ok=True)

    d, prov, max_err = load_inputs(args.input_root)
    spatial, kruskal, _ = spatial_tests(prov)
    province, pca, _, _, class_names, _, stability, cluster_decision = classify(prov)

    province.to_csv(results / 'q1_省级指标与分类.csv', index=False, encoding='utf-8-sig')
    spatial.to_csv(results / 'q1_Moran检验.csv', index=False, encoding='utf-8-sig')
    kruskal.to_csv(results / 'q1_分组差异检验.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'component': ['PC1', 'PC2'], 'explained_variance_ratio': pca.explained_variance_ratio_}).to_csv(results / 'q1_PCA.csv', index=False)
    stability.to_csv(results / 'q1_聚类稳定性.csv', index=False)
    stability.to_csv(results / 'q1_聚类决策.csv', index=False)
    (project / 'data').mkdir(parents=True, exist_ok=True)
    d.to_csv(project / 'data' / '附件1_清洗后.csv', index=False, encoding='utf-8-sig')

    print(json.dumps({
        'question': 1, 'provinces': int(len(province)), 'clusters': int(province['cluster'].nunique()),
        'class_names': [str(x) for x in province['class_name'].unique()],
        'best_silhouette': float(stability.loc[stability['selected'], 'silhouette'].iloc[0]),
        'moran_indicators': [str(x) for x in spatial['indicator'].tolist()],
        'max_abs_total_consistency_error_Mt': max_err,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
