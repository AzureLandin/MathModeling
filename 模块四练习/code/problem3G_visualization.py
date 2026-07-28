"""
问题三方案G可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_loss_curve(train_losses, val_losses, save_path):
    """
    绘制训练/验证loss曲线
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', linewidth=1.5, label='训练Loss')
    ax.plot(epochs, val_losses, 'r-', linewidth=1.5, label='验证Loss')
    
    # 标记最优epoch
    best_epoch = np.argmin(val_losses) + 1
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1, 
               label=f'最优Epoch={best_epoch}')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('SimpleRNN训练/验证Loss曲线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Loss曲线已保存: {save_path}")


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
    
    ax1.set_xlabel('周次', fontsize=12)
    ax1.set_ylabel('满足率 (%)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(weeks)
    ax1.set_ylim([85, 145])
    
    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.bar(weeks, P_j * 100, alpha=0.3, color=color2, width=0.4, label='达标概率')
    ax2.set_ylabel('达标概率 (%)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 120])
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    ax1.set_title('问题三方案G：24周产能满足率（SimpleRNN）', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  满足率图已保存: {save_path}")


def plot_weekly_supply_comparison(weekly_total, save_path, capacity=28200):
    """
    绘制每周供货总量柱状图
    """
    weeks = np.arange(1, 25)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if t >= capacity else 'red' for t in weekly_total]
    ax.bar(weeks, weekly_total, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=capacity, color='red', linestyle='--', linewidth=2, label=f'产能要求 ({capacity:,} m³)')
    
    ax.set_xlabel('周次', fontsize=12)
    ax.set_ylabel('总供货量 (产品体积 m³)', fontsize=12)
    ax.set_title('问题三方案G：每周供货总量', fontsize=14)
    ax.set_xticks(weeks)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (w, t) in enumerate(zip(weeks, weekly_total)):
        ax.text(w, t + 200, f'{t:,.0f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  周供货量图已保存: {save_path}")


def plot_pred_vs_actual(y_true, y_pred, metrics, save_path):
    """
    绘制回测期预测vs真实散点图
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测线')
    
    ax.set_xlabel('真实偏差 (m³)', fontsize=12)
    ax.set_ylabel('预测偏差 (m³)', fontsize=12)
    ax.set_title(f'回测期预测 vs 真实\nRMSE={metrics["RMSE"]:.2f}, MAE={metrics["MAE"]:.2f}, R²={metrics["R2"]:.4f}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  预测vs实际图已保存: {save_path}")


def plot_backtest_weekly(weeks, T_true, T_E, T_G, save_path, capacity=28200):
    """
    绘制回测期周总量对比图（E vs G）
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(weeks, T_true, 'ko-', linewidth=2, markersize=6, label='真实值')
    ax.plot(weeks, T_E, 's--', color='steelblue', linewidth=1.5, markersize=5, label='方案E (ARIMA)')
    ax.plot(weeks, T_G, '^--', color='coral', linewidth=1.5, markersize=5, label='方案G (RNN)')
    ax.axhline(y=capacity, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='产能要求')
    
    ax.set_xlabel('周次', fontsize=12)
    ax.set_ylabel('总供货量 (产品体积 m³)', fontsize=12)
    ax.set_title('回测期：方案E vs 方案G 周总量对比', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  周总量对比图已保存: {save_path}")
