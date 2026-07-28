"""
问题一可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_weight_bar_critic(weights, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    indicators = ['F1\n周均供货量', 'F2\n周均缺失量', 'F3\n供货达标率', 'F4\n供货偏差均方']
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800']
    bars = ax.bar(indicators, weights, color=colors, edgecolor='black', linewidth=0.5)

    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{w:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('权重', fontsize=14)
    ax.set_title('CRITIC法指标权重分布（主模型）', fontsize=18, fontweight='bold')
    ax.set_ylim(0, max(weights) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  权重柱状图已保存: {save_path}")


def plot_score_distribution(C, threshold, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(C, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)

    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.9,
            f'Top55分界线\nC={threshold:.4f}',
            color='red', fontsize=10, fontweight='bold')

    ax.set_xlabel('贴近度 C', fontsize=14)
    ax.set_ylabel('供应商数量', fontsize=14)
    ax.set_title('供应商贴近度分布（CRITIC-TOPSIS）', fontsize=18, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  贴近度分布图已保存: {save_path}")


def plot_weight_comparison(w_critic, w_entropy, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4)
    width = 0.35
    indicators = ['F1\n周均供货量', 'F2\n周均缺失量', 'F3\n供货达标率', 'F4\n供货偏差均方']

    ax.bar(x - width/2, w_critic, width, label='CRITIC（主模型）', color='#FF5722')
    ax.bar(x + width/2, w_entropy, width, label='熵权法（验证）', color='#2196F3')

    ax.set_xticks(x)
    ax.set_xticklabels(indicators)
    ax.set_ylabel('权重', fontsize=14)
    ax.set_title('CRITIC vs 熵权法 权重对比', fontsize=18, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  权重对比图已保存: {save_path}")


def plot_rank_scatter(rank_critic, rank_entropy, spearman_rho, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    both = (rank_critic <= 55) & (rank_entropy <= 55)
    only_crit = (rank_critic <= 55) & (rank_entropy > 55)
    only_ent = (rank_critic > 55) & (rank_entropy <= 55)
    other = (rank_critic > 55) & (rank_entropy > 55)

    ax.scatter(rank_critic[other], rank_entropy[other], c='lightgray', s=15, alpha=0.5, label='其他')
    ax.scatter(rank_critic[both], rank_entropy[both], c='#4CAF50', s=25, alpha=0.7, label=f'两者都入选 ({both.sum()})')
    ax.scatter(rank_critic[only_crit], rank_entropy[only_crit], c='#FF5722', s=30, marker='^', label=f'仅CRITIC入选 ({only_crit.sum()})')
    ax.scatter(rank_critic[only_ent], rank_entropy[only_ent], c='#2196F3', s=30, marker='v', label=f'仅熵权入选 ({only_ent.sum()})')

    ax.axvline(x=55.5, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=55.5, color='blue', linestyle='--', alpha=0.5)
    ax.plot([0, 420], [0, 420], 'k-', alpha=0.1)

    ax.set_xlabel('CRITIC-TOPSIS排名（主模型）', fontsize=14)
    ax.set_ylabel('熵权-TOPSIS排名（验证）', fontsize=14)
    ax.set_title(f'两种方法排名对比 (Spearman ρ={spearman_rho:.4f})', fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 420)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  排名散点图已保存: {save_path}")
