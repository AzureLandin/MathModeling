"""
问题三主程序：基于偏差序列ARIMA的供应方案预测
"""
import os
import numpy as np
import pandas as pd

from data_loader import load_supplier_data
from problem3_predictor import (build_deviation_series, fit_all_suppliers, predict_supply,
                                  monte_carlo_simulation, compute_supply_plan, compute_capacity_stats,
                                  compute_confidence_intervals)
from problem3_visualization import (plot_model_distribution, plot_gap_distribution, 
                                      plot_fulfillment_rate, plot_acf_examples, plot_weekly_supply_comparison)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "2026D附件", "附件1 近5年402家供应商的相关数据.xlsx")
ORDERING_PATH = os.path.join(BASE_DIR, "results", "problem2_ordering_plan.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def load_ordering_plan():
    """加载问题二的订购方案"""
    df = pd.read_csv(ORDERING_PATH)
    ids = df['供应商ID'].values
    types = df['材料类别'].values
    Q = df['订货量Q'].values
    x = df.iloc[:, 2:26].values
    
    type_map = {'A': 0.6, 'B': 0.66, 'C': 0.72}
    u = np.array([type_map[t] for t in types])
    
    return ids, types, Q, u, x


def get_supplier_indices(supplier_ids, target_ids):
    """获取目标供应商在原始数据中的索引"""
    indices = []
    for tid in target_ids:
        idx = np.where(supplier_ids == tid)[0][0]
        indices.append(idx)
    return indices


def save_model_selection(models, ids, types):
    """保存模型选择结果"""
    rows = []
    for i, m in enumerate(models):
        p = m['params']
        rows.append({
            '供应商ID': ids[i],
            '材料类别': types[i],
            '模型类型': m['model_type'],
            'ADF_p值': round(p['adf_p'], 4) if not np.isnan(p['adf_p']) else 'N/A',
            'LB_p值': round(p['lb_p'], 4) if not np.isnan(p['lb_p']) else 'N/A',
            'AIC': round(p['aic'], 2) if not np.isnan(p['aic']) else 'N/A',
            '偏差均值': round(p['mean_D'], 2),
            '有效周数': p['N']
        })
    
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "problem3E_model_selection.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  模型选择结果已保存: {path}")
    return df


def save_supply_plan(supply, ids, types):
    """保存供应方案"""
    columns = [f'第{j+1}周' for j in range(24)]
    df = pd.DataFrame(supply, columns=columns)
    df.insert(0, '材料类别', types)
    df.insert(0, '供应商ID', ids)
    df['周均供货量'] = supply.mean(axis=1).round(1)
    
    path = os.path.join(RESULTS_DIR, "problem3E_supply_plan.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  供应方案已保存: {path}")
    return df


def save_weekly_summary(weekly_total, fulfillment_rate, P_j):
    """保存每周汇总"""
    df = pd.DataFrame({
        '周次': np.arange(1, 25),
        '总供货量(m³)': weekly_total.round(0).astype(int),
        '满足率(%)': (fulfillment_rate * 100).round(2),
        '达标概率(%)': (P_j * 100).round(2),
        '达标': ['✓' if r >= 1.0 else '✗' for r in fulfillment_rate]
    })
    path = os.path.join(RESULTS_DIR, "problem3E_weekly_summary.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  每周汇总已保存: {path}")
    return df


def save_capacity_stats(weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all, ci):
    """保存产能统计"""
    df = pd.DataFrame({
        '指标': [
            '达标周数(点估计)', '总周数', '达标比例(点估计)',
            '平均满足率', '最小满足率', '最大满足率',
            '平均达标概率', '24周全部达标概率',
            '供货量5%分位均值', '供货量95%分位均值'
        ],
        '数值': [
            weeks_meet, 24, f'{weeks_meet/24:.1%}',
            f'{fulfillment_rate.mean():.2%}', f'{fulfillment_rate.min():.2%}', f'{fulfillment_rate.max():.2%}',
            f'{avg_P:.2%}', f'{P_all:.2%}',
            f'{ci[0].mean():.0f} m³', f'{ci[4].mean():.0f} m³'
        ]
    })
    path = os.path.join(RESULTS_DIR, "problem3E_capacity_stats.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  产能统计已保存: {path}")


def main():
    print("=" * 50)
    print("问题三：基于偏差序列ARIMA的供应方案")
    print("=" * 50)

    # 1. 数据读取
    supplier_ids, material_types, order_data, supply_data = load_supplier_data(DATA_PATH)

    # 2. 加载订购方案
    print("\n正在加载订购方案...")
    plan_ids, plan_types, Q, u, x = load_ordering_plan()
    indices = get_supplier_indices(supplier_ids, plan_ids)
    print(f"  供应商数量: {len(plan_ids)}")

    # 3. 构建偏差序列
    print("\n正在构建偏差序列 D = S - O...")
    deviation_series, valid_weeks = build_deviation_series(order_data, supply_data, indices)
    print(f"  有效周数范围: [{min(valid_weeks)}, {max(valid_weeks)}]")
    print(f"  平均有效周数: {np.mean(valid_weeks):.1f}")

    # 4. 拟合模型
    print("\n正在对55家供应商拟合模型...")
    models = fit_all_suppliers(deviation_series, valid_weeks, min_obs=30)
    
    # 模型类型统计
    arima_count = sum(1 for m in models if 'ARIMA' in m['model_type'])
    mean_count = sum(1 for m in models if '均值' in m['model_type'])
    print(f"\n=== 模型类型分布 ===")
    print(f"  ARIMA模型: {arima_count}家 ({arima_count/55*100:.1f}%)")
    print(f"  均值模型: {mean_count}家 ({mean_count/55*100:.1f}%)")
    
    # 偏差均值统计
    means = [m['params']['mean_D'] for m in models]
    print(f"\n=== 偏差均值统计 ===")
    print(f"  平均偏差: {np.mean(means):.2f} m³")
    print(f"  超额供货(>0): {sum(1 for d in means if d > 0)}家")
    print(f"  供货不足(<0): {sum(1 for d in means if d < 0)}家")
    print(f"  精确交货(=0): {sum(1 for d in means if d == 0)}家")

    # 5. 预测供货量
    print("\n正在预测供货量...")
    S_pred = predict_supply(models, Q, x)
    supply = compute_supply_plan(S_pred)

    # 6. 蒙特卡洛模拟
    print("\n正在进行蒙特卡洛模拟 (M=10000)...")
    T_sim, P_j = monte_carlo_simulation(models, Q, u, x, M=10000)

    # 7. 产能统计
    weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all = compute_capacity_stats(supply, u, T_sim, P_j)
    ci = compute_confidence_intervals(T_sim)
    
    print(f"\n=== 产能满足率统计 ===")
    print(f"  达标周数(点估计): {weeks_meet}/24")
    print(f"  达标比例(点估计): {weeks_meet/24:.1%}")
    print(f"  平均满足率: {fulfillment_rate.mean():.2%}")
    print(f"  最小满足率: {fulfillment_rate.min():.2%}")
    print(f"  最大满足率: {fulfillment_rate.max():.2%}")
    print(f"\n=== 蒙特卡洛概率评估 ===")
    print(f"  平均达标概率: {avg_P:.2%}")
    print(f"  24周全部达标概率: {P_all:.2%}")

    # 8. 保存结果
    print("\n正在保存结果...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    save_model_selection(models, plan_ids, plan_types)
    save_supply_plan(supply, plan_ids, plan_types)
    save_weekly_summary(weekly_total, fulfillment_rate, P_j)
    save_capacity_stats(weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all, ci)

    # 9. 生成图表
    print("\n正在生成图表...")
    plot_model_distribution(models, os.path.join(FIGURES_DIR, "problem3E_model_distribution.png"))
    plot_gap_distribution(models, os.path.join(FIGURES_DIR, "problem3E_gap_distribution.png"))
    plot_fulfillment_rate(fulfillment_rate, P_j, os.path.join(FIGURES_DIR, "problem3E_fulfillment_rate.png"))
    plot_acf_examples(deviation_series, models, plan_ids, os.path.join(FIGURES_DIR, "problem3E_acf_example.png"))
    plot_weekly_supply_comparison(weekly_total, os.path.join(FIGURES_DIR, "problem3E_weekly_supply.png"))

    # 10. 打印每周详情
    print("\n=== 每周供货详情 ===")
    summary_df = pd.DataFrame({
        '周次': np.arange(1, 25),
        '总供货量': weekly_total.round(0).astype(int),
        '满足率': (fulfillment_rate * 100).round(1),
        '达标概率': (P_j * 100).round(1),
        '达标': ['✓' if r >= 1.0 else '✗' for r in fulfillment_rate]
    })
    print(summary_df.to_string(index=False))

    # 11. 打印典型供应商模型
    print("\n=== 典型供应商模型 ===")
    print(f"{'供应商ID':<12} {'材料':<6} {'模型类型':<20} {'偏差均值':<10} {'有效周数':<8}")
    print("-" * 56)
    for i in range(min(15, len(models))):
        m = models[i]
        print(f"{plan_ids[i]:<12} {plan_types[i]:<6} {m['model_type']:<20} {m['params']['mean_D']:<10.2f} {m['params']['N']:<8}")

    print("\n" + "=" * 50)
    print("问题三求解完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
