"""
问题二可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_weekly_volume(weekly_volume, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    weeks = np.arange(1, 25)
    
    colors = ['#4CAF50' if v >= 25380 else '#FF5722' for v in weekly_volume]
    bars = ax.bar(weeks, weekly_volume, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=25380, color='red', linestyle='--', linewidth=2, label='产能底线 (25380 m$^3$)')
    
    ax.set_xlabel('周次', fontsize=14)
    ax.set_ylabel('订购量（产品体积 m$^3$）', fontsize=14)
    ax.set_title('24周每周订购量', fontsize=18, fontweight='bold')
    ax.set_xticks(weeks)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  订购量柱状图已保存: {save_path}")


def plot_weekly_suppliers(weekly_count, weekly_A, weekly_B, weekly_C, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    weeks = np.arange(1, 25)
    
    ax.bar(weeks, weekly_A, color='#2196F3', label='A类', edgecolor='black', linewidth=0.3)
    ax.bar(weeks, weekly_B, bottom=weekly_A, color='#FF9800', label='B类', edgecolor='black', linewidth=0.3)
    ax.bar(weeks, weekly_C, bottom=weekly_A + weekly_B, color='#4CAF50', label='C类', edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('周次', fontsize=14)
    ax.set_ylabel('供应商数量', fontsize=14)
    ax.set_title('每周选中供应商数量（按材料类型）', fontsize=18, fontweight='bold')
    ax.set_xticks(weeks)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  供应商数量图已保存: {save_path}")


def plot_supplier_heatmap(x, ids, types, save_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    type_order = {'A': 0, 'B': 1, 'C': 2}
    sorted_idx = sorted(range(len(types)), key=lambda i: (type_order[types[i]], ids[i]))
    x_sorted = x[sorted_idx, :]
    ids_sorted = [ids[i] for i in sorted_idx]
    types_sorted = [types[i] for i in sorted_idx]
    
    im = ax.imshow(x_sorted, cmap='Reds', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('周次', fontsize=14)
    ax.set_ylabel('供应商', fontsize=14)
    ax.set_title('供应商订购计划热力图', fontsize=18, fontweight='bold')
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(np.arange(1, 25), fontsize=11)
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels([f"{ids_sorted[i]}({types_sorted[i]})" for i in range(len(ids))], fontsize=7)
    
    cbar = plt.colorbar(im, ax=ax, label='是否订购 (0/1)')
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('是否订购 (0/1)', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  热力图已保存: {save_path}")
