"""
问题一主程序：基于CRITIC-TOPSIS的供应商重要性评价
"""
import os
import pandas as pd

from data_loader import load_supplier_data
from indicators import compute_indicators, build_indicator_matrix
from normalize import minmax_normalize
from critic_weight import critic_weight
from topsis import topsis
from problem1_visualization import plot_weight_bar_critic, plot_score_distribution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "2026D附件", "附件1 近5年402家供应商的相关数据.xlsx")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def save_critic_weights(w_critic, sigma, gamma, info):
    df = pd.DataFrame({
        '指标': ['F1_周均供货量', 'F2_周均缺失量', 'F3_供货达标率', 'F4_供货偏差均方'],
        'CRITIC权重': w_critic,
        '对比强度σ': sigma,
        '冲突性γ': gamma,
        '信息承载量I': info
    })
    path = os.path.join(RESULTS_DIR, "problem1_critic_weights.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  CRITIC权重已保存: {path}")


def save_ranking(supplier_ids, material_types, F1, F2, F3, F4, N, weights, C):
    df = pd.DataFrame({
        '供应商ID': supplier_ids,
        '材料类别': material_types,
        '有效周数': N.astype(int),
        'F1_周均供货量': F1,
        'F2_周均缺失量': F2,
        'F3_供货达标率': F3,
        'F4_供货偏差均方': F4,
        '贴近度C': C
    })
    df['排名'] = df['贴近度C'].rank(ascending=False).astype(int)
    df = df.sort_values('排名')

    path_all = os.path.join(RESULTS_DIR, "problem1_supplier_ranking.csv")
    df.to_csv(path_all, index=False, encoding='utf-8-sig')
    print(f"  完整排名已保存: {path_all}")

    top55 = df.head(55)
    path_55 = os.path.join(RESULTS_DIR, "problem1_top55.csv")
    top55.to_csv(path_55, index=False, encoding='utf-8-sig')
    print(f"  前55家供应商已保存: {path_55}")

    return df, top55


def main():
    print("=" * 50)
    print("问题一：基于CRITIC-TOPSIS的供应商重要性评价")
    print("=" * 50)

    supplier_ids, material_types, order_data, supply_data = load_supplier_data(DATA_PATH)

    print("\n正在计算四维评价指标...")
    F1, F2, F3, F4, N = compute_indicators(order_data, supply_data)
    matrix, is_benefit = build_indicator_matrix(F1, F2, F3, F4)

    print("正在进行Min-Max标准化...")
    Y = minmax_normalize(matrix, is_benefit)

    print("正在计算CRITIC权重...")
    w_critic, sigma, R, gamma = critic_weight(Y)
    info = sigma * gamma

    print("\n=== CRITIC权重计算结果 ===")
    indicators = ['F1_周均供货量', 'F2_周均缺失量', 'F3_供货达标率', 'F4_供货偏差均方']
    for name, w, s, g, i in zip(indicators, w_critic, sigma, gamma, info):
        print(f"  {name}: 权重={w:.4f}, σ={s:.4f}, γ={g:.4f}, I={i:.4f}")

    print("\n正在进行TOPSIS排序...")
    C, _, _ = topsis(Y, w_critic)

    print("\n正在保存结果...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    save_critic_weights(w_critic, sigma, gamma, info)
    ranking_df, top55_df = save_ranking(supplier_ids, material_types, F1, F2, F3, F4, N, w_critic, C)

    print("\n正在生成图表...")
    plot_weight_bar_critic(w_critic, os.path.join(FIGURES_DIR, "problem1_weight_comparison.png"))
    threshold = top55_df['贴近度C'].min()
    plot_score_distribution(C, threshold, os.path.join(FIGURES_DIR, "problem1_score_distribution.png"))

    print("\n=== 前10名供应商 ===")
    print(ranking_df.head(10).to_string(index=False))

    print("\n" + "=" * 50)
    print("问题一求解完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
