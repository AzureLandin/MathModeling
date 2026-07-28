"""
问题三可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_model_distribution(models, save_path):
    """
    绘制模型类型分布饼图
    """
    types = [m['model_type'].split('(')[0].split('{')[0] for m in models]
    unique_types, counts = np.unique(types, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    wedges, texts, autotexts = ax.pie(counts, labels=unique_types, autopct='%1.1f%%',
                                       colors=colors, startangle=90, pctdistance=0.85)
    
    # 中心圆环效果
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax.add_artist(centre_circle)
    
    ax.set_title('供应商模型类型分布', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  模型分布图已保存: {save_path}")


def plot_gap_distribution(models, save_path):
    """
    绘制偏差均值分布直方图
    """
    means = [m['params']['mean_D'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(means, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='D=0 (精确交货)')
    ax.axvline(x=np.mean(means), color='orange', linestyle='-', linewidth=2, 
               label=f'均值={np.mean(means):.1f} m³')
    
    ax.set_xlabel('偏差均值 D̄ (m³)', fontsize=14)
    ax.set_ylabel('供应商数量', fontsize=14)
    ax.set_title('供应商供货偏差均值分布', fontsize=18, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 标注超额/缺货区域
    ax.axvspan(0, max(means) * 1.1, alpha=0.1, color='green', label='超额供货')
    ax.axvspan(min(means) * 1.1, 0, alpha=0.1, color='red', label='供货不足')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  偏差分布图已保存: {save_path}")


def plot_fulfillment_rate(fulfillment_rate, P_j, save_path):
    """
    绘制24周满足率折线图
    """
    weeks = np.arange(1, 25)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color1 = 'steelblue'
    ax1.plot(weeks, fulfillment_rate * 100, 'o-', color=color1, linewidth=2, markersize=6, label='点估计满足率')
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='100%基线')
    ax1.fill_between(weeks, 100, fulfillment_rate * 100, 
                     where=(fulfillment_rate >= 1), alpha=0.2, color='green', label='达标区域')
    ax1.fill_between(weeks, 100, fulfillment_rate * 100, 
                     where=(fulfillment_rate < 1), alpha=0.2, color='red', label='未达标区域')
    
    ax1.set_xlabel('周次', fontsize=14)
    ax1.set_ylabel('满足率 (%)', fontsize=14, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(weeks)
    ax1.set_ylim([85, 145])
    
    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.bar(weeks, P_j * 100, alpha=0.3, color=color2, width=0.4, label='达标概率')
    ax2.set_ylabel('达标概率 (%)', fontsize=14, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 120])
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    ax1.set_title('问题三：24周产能满足率（偏差序列ARIMA）', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  满足率图已保存: {save_path}")


def plot_acf_examples(deviation_series, models, plan_ids, save_path, n_examples=3):
    """
    绘制典型供应商偏差序列ACF图
    """
    from statsmodels.graphics.tsaplots import plot_acf
    
    # 选择使用ARIMA的供应商
    arima_indices = [i for i, m in enumerate(models) if 'ARIMA' in m['model_type']]
    
    if len(arima_indices) < n_examples:
        arima_indices = list(range(min(n_examples, len(models))))
    
    # 选择前n_examples个
    selected = arima_indices[:n_examples]
    
    fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]
    
    for k, idx in enumerate(selected):
        ax = axes[k]
        D = deviation_series[idx]
        
        plot_acf(D, lags=min(30, len(D)//2 - 1), ax=ax, alpha=0.05)
        ax.set_title(f'供应商 {plan_ids[idx]}\n{models[idx]["model_type"]}', fontsize=12)
        ax.set_xlabel('滞后阶数', fontsize=11)
        ax.set_ylabel('自相关系数', fontsize=11)
    
    plt.suptitle('典型供应商偏差序列ACF图', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ACF图已保存: {save_path}")


def plot_weekly_supply_comparison(weekly_total, save_path, capacity=28200):
    """
    绘制每周供货总量柱状图
    """
    weeks = np.arange(1, 25)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if t >= capacity else 'red' for t in weekly_total]
    ax.bar(weeks, weekly_total, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=capacity, color='red', linestyle='--', linewidth=2, label=f'产能要求 ({capacity:,} m³)')
    
    ax.set_xlabel('周次', fontsize=14)
    ax.set_ylabel('总供货量 (产品体积 m³)', fontsize=14)
    ax.set_title('问题三：每周供货总量', fontsize=18, fontweight='bold')
    ax.set_xticks(weeks)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (w, t) in enumerate(zip(weeks, weekly_total)):
        ax.text(w, t + 200, f'{t:,.0f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  周供货量图已保存: {save_path}")
