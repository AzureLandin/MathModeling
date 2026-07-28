"""
问题二主程序：最优原材料订购方案（0-1整数规划）
"""
import os
import math
import numpy as np
import pandas as pd

from data_loader import load_supplier_data
from ordering_solver import load_top55_params, check_feasibility, solve_ordering_problem, compute_weekly_summary
from problem2_visualization import plot_weekly_volume, plot_weekly_suppliers, plot_supplier_heatmap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "2026D附件", "附件1 近5年402家供应商的相关数据.xlsx")
TOP55_PATH = os.path.join(BASE_DIR, "results", "problem1_top55.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def save_ordering_plan(x, ids, types, Q, u, p, F):
    T = 24
    
    columns = [f'第{j+1}周' for j in range(T)]
    df = pd.DataFrame(x, columns=columns)
    df.insert(0, '材料类别', types)
    df.insert(0, '供应商ID', ids)
    
    df['订货次数'] = x.sum(axis=1)
    df['最低次数'] = F
    df['订货量Q'] = Q
    df['可生产产品'] = (Q / u).round(0).astype(int)
    
    path = os.path.join(RESULTS_DIR, "problem2_ordering_plan.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  订购方案已保存: {path}")
    return df


def save_weekly_summary(weekly_volume, weekly_cost, weekly_count, weekly_A, weekly_B, weekly_C):
    df = pd.DataFrame({
        '周次': np.arange(1, 25),
        '总订购量(m³)': weekly_volume.round(0).astype(int),
        '总成本': weekly_cost.round(2),
        '供应商数': weekly_count,
        'A类': weekly_A,
        'B类': weekly_B,
        'C类': weekly_C,
        '达标': ['✓' if v >= 25380 else '✗' for v in weekly_volume]
    })
    
    path = os.path.join(RESULTS_DIR, "problem2_weekly_summary.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  每周汇总已保存: {path}")
    return df


def main():
    print("=" * 50)
    print("问题二：最优原材料订购方案（0-1整数规划）")
    print("=" * 50)

    supplier_ids, material_types, order_data, supply_data = load_supplier_data(DATA_PATH)

    print("\n正在加载55家供应商参数...")
    indices, Q, F, u, p, ids, types = load_top55_params(TOP55_PATH, order_data, supplier_ids)
    
    print(f"\n=== 参数统计 ===")
    print(f"  供应商数量: {len(Q)}")
    print(f"  订货量Q范围: [{Q.min()}, {Q.max()}]")
    print(f"  频次F范围: [{F.min()}, {F.max()}]")
    print(f"  材料类型分布: A={types.count('A')}, B={types.count('B')}, C={types.count('C')}")

    check_feasibility(Q, u, F)

    x, status, obj_val = solve_ordering_problem(Q, u, p, F)
    
    print(f"\n=== 求解结果 ===")
    print(f"  求解状态: {status}")
    print(f"  最优目标值（总成本）: {obj_val:.2f}")

    weekly_volume, weekly_cost, weekly_count, weekly_A, weekly_B, weekly_C = compute_weekly_summary(x, Q, u, p, types)

    print(f"\n=== 每周统计 ===")
    print(f"  平均周订购量: {weekly_volume.mean():.0f} m³")
    print(f"  最小周订购量: {weekly_volume.min():.0f} m³")
    print(f"  达标周数: {(weekly_volume >= 25380).sum()}/24")
    print(f"  平均每周供应商数: {weekly_count.mean():.1f}")

    print("\n正在保存结果...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    plan_df = save_ordering_plan(x, ids, types, Q, u, p, F)
    summary_df = save_weekly_summary(weekly_volume, weekly_cost, weekly_count, weekly_A, weekly_B, weekly_C)

    print("\n正在生成图表...")
    plot_weekly_volume(weekly_volume, os.path.join(FIGURES_DIR, "problem2_weekly_volume.png"))
    plot_weekly_suppliers(weekly_count, weekly_A, weekly_B, weekly_C, os.path.join(FIGURES_DIR, "problem2_weekly_suppliers.png"))
    plot_supplier_heatmap(x, ids, types, os.path.join(FIGURES_DIR, "problem2_heatmap.png"))

    print("\n=== 每周订购量详情 ===")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("问题二求解完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
