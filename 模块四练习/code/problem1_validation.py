"""
问题一验证程序：熵权-TOPSIS与CRITIC-TOPSIS对比（稳健性检验）
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from problem_data_loader import load_supplier_data
from problem_indicators import compute_indicators, build_indicator_matrix
from problem_normalize import minmax_normalize
from problem_entropy_weight import entropy_weight
from problem_critic_weight import critic_weight
from problem_topsis import topsis
from problem1_visualization import plot_rank_scatter

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "2026D附件", "附件1 近5年402家供应商的相关数据.xlsx")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def save_entropy_weights(w_entropy, entropy):
    """保存熵权法权重"""
    df = pd.DataFrame({
        '指标': ['F1_周均供货量', 'F2_周均缺失量', 'F3_供货达标率', 'F4_供货偏差均方'],
        '熵权法权重': w_entropy,
        '信息熵': entropy
    })
    path = os.path.join(RESULTS_DIR, "problem1_entropy_weights.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  熵权法权重已保存: {path}")


def save_comparison(supplier_ids, material_types, C_critic, rank_critic, C_entropy, rank_entropy):
    """保存排名对比"""
    df = pd.DataFrame({
        '供应商ID': supplier_ids,
        '材料类别': material_types,
        'CRITIC贴近度': C_critic,
        'CRITIC排名': rank_critic,
        '熵权贴近度': C_entropy,
        '熵权排名': rank_entropy,
        '排名差': rank_critic - rank_entropy
    }).sort_values('CRITIC排名')
    path = os.path.join(RESULTS_DIR, "problem1_robustness_comparison.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  排名对比已保存: {path}")


def compute_robustness(supplier_ids, rank_critic, rank_entropy):
    """计算稳健性指标"""
    top55_critic = set(supplier_ids[rank_critic <= 55])
    top55_entropy = set(supplier_ids[rank_entropy <= 55])
    overlap = top55_critic & top55_entropy

    spearman_rho, p_value = stats.spearmanr(rank_critic, rank_entropy)

    print("\n=== 稳健性验证结果 ===")
    print(f"  前55名重合数: {len(overlap)}/55")
    print(f"  重合率: {len(overlap)/55:.1%}")
    print(f"  Spearman ρ: {spearman_rho:.4f} (p={p_value:.2e})")

    return overlap, spearman_rho


def analyze_boundary(supplier_ids, rank_critic, rank_entropy, overlap):
    """分析边界供应商"""
    boundary_critic = set(supplier_ids[(rank_critic >= 50) & (rank_critic <= 60)])
    boundary_entropy = set(supplier_ids[(rank_entropy >= 50) & (rank_entropy <= 60)])
    boundary_all = boundary_critic | boundary_entropy
    boundary_diff = boundary_all - overlap

    if boundary_diff:
        print("\n=== 边界供应商分析 (排名50-60区间) ===")
        for sid in sorted(boundary_diff):
            rc = rank_critic[supplier_ids == sid][0]
            re = rank_entropy[supplier_ids == sid][0]
            print(f"  {sid}: CRITIC排名={rc}, 熵权排名={re}, 差={rc-re}")


def main():
    print("=" * 50)
    print("问题一验证：熵权-TOPSIS稳健性检验")
    print("=" * 50)

    # 1. 数据读取与指标计算
    supplier_ids, material_types, order_data, supply_data = load_supplier_data(DATA_PATH)
    print("\n正在计算四维评价指标...")
    F1, F2, F3, F4, N = compute_indicators(order_data, supply_data)
    matrix, is_benefit = build_indicator_matrix(F1, F2, F3, F4)

    # 2. 标准化
    Y = minmax_normalize(matrix, is_benefit)

    # 3. CRITIC法 + TOPSIS（主模型，读取已有结果）
    print("正在计算CRITIC法...")
    w_critic, _, _, _ = critic_weight(Y)
    C_critic, _, _ = topsis(Y, w_critic)
    rank_critic = (-C_critic).argsort().argsort() + 1

    # 4. 熵权法 + TOPSIS（验证模型）
    print("正在计算熵权法...")
    w_entropy, entropy_arr = entropy_weight(Y)
    C_entropy, _, _ = topsis(Y, w_entropy)
    rank_entropy = (-C_entropy).argsort().argsort() + 1

    # 5. 保存结果
    print("\n正在保存结果...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    save_entropy_weights(w_entropy, entropy_arr)
    save_comparison(supplier_ids, material_types, C_critic, rank_critic, C_entropy, rank_entropy)

    # 6. 稳健性分析
    overlap, rho = compute_robustness(supplier_ids, rank_critic, rank_entropy)
    analyze_boundary(supplier_ids, rank_critic, rank_entropy, overlap)

    # 7. 生成散点图
    print("\n正在生成散点图...")
    plot_rank_scatter(rank_critic, rank_entropy, rho, os.path.join(FIGURES_DIR, "problem1_rank_scatter.png"))

    print("\n" + "=" * 50)
    print("稳健性验证完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
