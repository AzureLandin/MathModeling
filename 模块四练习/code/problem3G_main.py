"""
问题三方案G主程序：基于SimpleRNN的偏差序列预测
包含正式预测和回测对比
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_supplier_data
from indicators import compute_indicators
from problem3G_predictor import (build_deviation_series, prepare_rnn_data, train_rnn,
                                  predict_rnn, monte_carlo_simulation, compute_supply_plan,
                                  compute_capacity_stats, compute_confidence_intervals)
from problem3G_visualization import (plot_loss_curve, plot_fulfillment_rate, 
                                      plot_weekly_supply_comparison, plot_pred_vs_actual,
                                      plot_backtest_weekly)

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


def compute_F_subset(order_data, supply_data, indices, week_start, week_end):
    """在指定周范围内计算F1-F4"""
    F1 = np.zeros(len(indices))
    F2 = np.zeros(len(indices))
    F3 = np.zeros(len(indices))
    F4 = np.zeros(len(indices))
    
    for k, idx in enumerate(indices):
        mask = order_data[idx, week_start:week_end] > 0
        O = order_data[idx, week_start:week_end][mask]
        S = supply_data[idx, week_start:week_end][mask]
        
        if len(O) == 0:
            continue
        
        F1[k] = S.mean()
        F2[k] = np.maximum(0, O - S).mean()
        F3[k] = (S >= O).mean()
        F4[k] = ((O - S) ** 2).mean()
    
    return F1, F2, F3, F4


def compute_metrics(y_true, y_pred, name):
    """计算逐观测指标"""
    mask = y_true > 0
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    n = len(y_t)
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae = np.mean(np.abs(y_t - y_p))
    mape = np.mean(np.abs(y_t - y_p) / y_t) * 100
    
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {
        'model': name,
        'n_obs': n,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def run_backtest(supplier_ids, order_data, supply_data, indices, plan_types, seed=42):
    """运行回测：训练期1-216周，测试期217-240周"""
    train_end = 216
    test_start = 216
    test_end = 240
    
    type_map = {'A': 0.6, 'B': 0.66, 'C': 0.72}
    u = np.array([type_map[t] for t in plan_types])
    
    print("\n" + "=" * 60)
    print("方案G回测：SimpleRNN")
    print("=" * 60)
    print(f"  训练期：第1-{train_end}周")
    print(f"  测试期：第{test_start+1}-{test_end}周")
    
    # 构建训练期偏差序列
    train_deviation = []
    for idx in indices:
        mask = order_data[idx, :train_end] > 0
        O = order_data[idx, :train_end][mask]
        S = supply_data[idx, :train_end][mask]
        D = S - O
        train_deviation.append(D)
    
    # 准备RNN数据
    X_train, y_train, X_val, y_val, norm_params = prepare_rnn_data(train_deviation, window_size=4)
    print(f"  训练样本数: {len(X_train)}")
    print(f"  验证样本数: {len(X_val)}")
    
    # 训练RNN
    print("  正在训练RNN...")
    model, train_losses, val_losses = train_rnn(X_train, y_train, X_val, y_val, 
                                                 hidden_size=16, lr=0.001, 
                                                 max_epochs=100, patience=10, seed=seed)
    print(f"  训练完成，最优Epoch={np.argmin(val_losses)+1}")
    
    # 预测测试期
    # 构建测试期真实值矩阵
    S_true_matrix = np.zeros((len(indices), test_end - test_start))
    S_pred_matrix = np.zeros((len(indices), test_end - test_start))
    
    for k, idx in enumerate(indices):
        # 训练期完整偏差序列
        mask_train = order_data[idx, :train_end] > 0
        D_train = supply_data[idx, :train_end][mask_train] - order_data[idx, :train_end][mask_train]
        
        mu = norm_params[k]['mu']
        sigma = norm_params[k]['sigma']
        
        # 自回归预测24步
        if len(D_train) >= 4:
            window = (D_train[-4:] - mu) / sigma
            D_forecast_norm = []
            
            for step in range(24):
                import torch
                x_in = torch.tensor(window).unsqueeze(0).unsqueeze(-1).float()
                model.eval()
                with torch.no_grad():
                    d_norm = model(x_in).item()
                D_forecast_norm.append(d_norm)
                window = np.append(window[1:], d_norm)
            
            D_forecast = np.array(D_forecast_norm) * sigma + mu
        else:
            D_forecast = np.full(24, mu)
        
        # 测试期真实值和预测值
        for j in range(test_start, test_end):
            S_true_matrix[k, j - test_start] = supply_data[idx, j]
            if order_data[idx, j] > 0:
                S_pred_matrix[k, j - test_start] = max(0, order_data[idx, j] + D_forecast[j - test_start])
    
    # 计算指标
    # 逐观测指标
    y_true_obs = []
    y_pred_obs = []
    for k in range(len(indices)):
        for j in range(test_end - test_start):
            if order_data[indices[k], test_start + j] > 0:
                y_true_obs.append(S_true_matrix[k, j])
                y_pred_obs.append(S_pred_matrix[k, j])
    
    y_true_obs = np.array(y_true_obs)
    y_pred_obs = np.array(y_pred_obs)
    
    metrics_G = compute_metrics(y_true_obs, y_pred_obs, '方案G (RNN)')
    
    # 周总量指标
    T_true = (S_true_matrix / u[:, np.newaxis]).sum(axis=0)
    T_pred = (S_pred_matrix / u[:, np.newaxis]).sum(axis=0)
    
    rmse_week = np.sqrt(np.mean((T_true - T_pred) ** 2))
    mape_week = np.mean(np.abs(T_true - T_pred) / T_true) * 100
    true_meet = T_true >= 28200
    pred_meet = T_pred >= 28200
    accuracy = (true_meet == pred_meet).mean()
    
    metrics_G['RMSE_week'] = rmse_week
    metrics_G['MAPE_week'] = mape_week
    metrics_G['accuracy'] = accuracy
    
    return metrics_G, model, train_losses, val_losses, norm_params, T_true, T_pred


def run_formal_prediction(supplier_ids, order_data, supply_data, indices, plan_ids, plan_types, Q, u, x, seed=42):
    """运行正式预测（全量数据训练）"""
    print("\n" + "=" * 60)
    print("方案G正式预测：SimpleRNN")
    print("=" * 60)
    
    # 构建全量偏差序列
    deviation_series = build_deviation_series(order_data, supply_data, indices)
    
    # 准备RNN数据
    X_train, y_train, X_val, y_val, norm_params = prepare_rnn_data(deviation_series, window_size=4)
    print(f"  训练样本数: {len(X_train)}")
    print(f"  验证样本数: {len(X_val)}")
    
    # 训练RNN
    print("  正在训练RNN...")
    model, train_losses, val_losses = train_rnn(X_train, y_train, X_val, y_val,
                                                 hidden_size=16, lr=0.001,
                                                 max_epochs=100, patience=10, seed=seed)
    print(f"  训练完成，最优Epoch={np.argmin(val_losses)+1}")
    print(f"  最终验证Loss: {min(val_losses):.6f}")
    
    # 预测供货量
    print("\n正在预测供货量...")
    S_pred = predict_rnn(model, deviation_series, norm_params, order_data, Q, x, indices)
    supply = compute_supply_plan(S_pred)
    
    # 蒙特卡洛模拟
    print("\n正在进行蒙特卡洛模拟 (M=10000)...")
    T_sim, P_j = monte_carlo_simulation(model, deviation_series, norm_params, Q, u, x, indices,
                                          X_val, y_val, M=10000, seed=seed)
    
    # 产能统计
    weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all = compute_capacity_stats(supply, u, T_sim, P_j)
    ci = compute_confidence_intervals(T_sim)
    
    return model, train_losses, val_losses, norm_params, supply, weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all, ci, T_sim, P_j


def save_results(supply, plan_ids, plan_types, weekly_total, fulfillment_rate, P_j, 
                 weeks_meet, avg_P, P_all, ci, metrics_G, model_E_metrics=None):
    """保存结果"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 模型性能
    perf_df = pd.DataFrame({
        '指标': ['训练样本数', '验证样本数', 'RMSE', 'MAE', 'MAPE(%)', 'R²', 
                 'RMSE_week', 'MAPE_week(%)', '达标准确率'],
        '数值': [metrics_G.get('n_obs', ''), metrics_G.get('n_val', ''),
                f"{metrics_G['RMSE']:.2f}", f"{metrics_G['MAE']:.2f}", f"{metrics_G['MAPE']:.2f}",
                f"{metrics_G['R2']:.4f}", f"{metrics_G.get('RMSE_week', 0):.0f}", 
                f"{metrics_G.get('MAPE_week', 0):.2f}", f"{metrics_G.get('accuracy', 0):.1%}"]
    })
    
    # 与方案E对比
    if model_E_metrics:
        perf_df['方案E'] = ['', '', f"{model_E_metrics['RMSE']:.2f}", f"{model_E_metrics['MAE']:.2f}",
                           f"{model_E_metrics['MAPE']:.2f}", f"{model_E_metrics['R2']:.4f}",
                           f"{model_E_metrics.get('RMSE_week', 0):.0f}", 
                           f"{model_E_metrics.get('MAPE_week', 0):.2f}",
                           f"{model_E_metrics.get('accuracy', 0):.1%}"]
    
    perf_df.to_csv(os.path.join(RESULTS_DIR, "problem3G_rnn_performance.csv"), index=False, encoding='utf-8-sig')
    print(f"  模型性能已保存")
    
    # 供应方案
    columns = [f'第{j+1}周' for j in range(24)]
    supply_df = pd.DataFrame(supply, columns=columns)
    supply_df.insert(0, '材料类别', plan_types)
    supply_df.insert(0, '供应商ID', plan_ids)
    supply_df['周均供货量'] = supply.mean(axis=1).round(1)
    supply_df.to_csv(os.path.join(RESULTS_DIR, "problem3G_supply_plan.csv"), index=False, encoding='utf-8-sig')
    print(f"  供应方案已保存")
    
    # 每周汇总
    weekly_df = pd.DataFrame({
        '周次': np.arange(1, 25),
        '总供货量(m³)': weekly_total.round(0).astype(int),
        '满足率(%)': (fulfillment_rate * 100).round(2),
        '达标概率(%)': (P_j * 100).round(2),
        '达标': ['✓' if r >= 1.0 else '✗' for r in fulfillment_rate]
    })
    weekly_df.to_csv(os.path.join(RESULTS_DIR, "problem3G_weekly_summary.csv"), index=False, encoding='utf-8-sig')
    print(f"  每周汇总已保存")
    
    # 产能统计
    stats_df = pd.DataFrame({
        '指标': [
            '达标周数(点估计)', '总周数', '达标比例(点估计)',
            '平均满足率', '最小满足率', '最大满足率',
            '平均达标概率', '24周全部达标概率'
        ],
        '数值': [
            weeks_meet, 24, f'{weeks_meet/24:.1%}',
            f'{fulfillment_rate.mean():.2%}', f'{fulfillment_rate.min():.2%}', f'{fulfillment_rate.max():.2%}',
            f'{avg_P:.2%}', f'{P_all:.2%}'
        ]
    })
    stats_df.to_csv(os.path.join(RESULTS_DIR, "problem3G_capacity_stats.csv"), index=False, encoding='utf-8-sig')
    print(f"  产能统计已保存")


def main():
    print("=" * 60)
    print("问题三方案G：基于SimpleRNN的偏差序列预测")
    print("=" * 60)

    # 1. 数据读取
    supplier_ids, material_types, order_data, supply_data = load_supplier_data(DATA_PATH)

    # 2. 加载订购方案
    print("\n正在加载订购方案...")
    plan_ids, plan_types, Q, u, x = load_ordering_plan()
    indices = get_supplier_indices(supplier_ids, plan_ids)
    print(f"  供应商数量: {len(plan_ids)}")

    # 3. 回测
    metrics_G, model_bt, train_losses_bt, val_losses_bt, norm_params_bt, T_true, T_pred = run_backtest(
        supplier_ids, order_data, supply_data, indices, plan_types)
    
    # 加载方案E回测结果
    model_E_metrics = {
        'RMSE': 623.03, 'MAE': 142.58, 'MAPE': 76.54, 'R2': 0.7056,
        'RMSE_week': 10459, 'MAPE_week': 29.56, 'accuracy': 0.542
    }
    
    print("\n=== 回测结果对比 ===")
    print(f"{'指标':<15} {'方案E (ARIMA)':<15} {'方案G (RNN)':<15}")
    print("-" * 45)
    print(f"{'RMSE':<15} {model_E_metrics['RMSE']:<15.2f} {metrics_G['RMSE']:<15.2f}")
    print(f"{'MAE':<15} {model_E_metrics['MAE']:<15.2f} {metrics_G['MAE']:<15.2f}")
    print(f"{'MAPE (%)':<15} {model_E_metrics['MAPE']:<15.2f} {metrics_G['MAPE']:<15.2f}")
    print(f"{'R²':<15} {model_E_metrics['R2']:<15.4f} {metrics_G['R2']:<15.4f}")
    print(f"{'RMSE_week':<15} {model_E_metrics['RMSE_week']:<15.0f} {metrics_G['RMSE_week']:<15.0f}")
    print(f"{'MAPE_week (%)':<15} {model_E_metrics['MAPE_week']:<15.2f} {metrics_G['MAPE_week']:<15.2f}")
    print(f"{'达标准确率':<15} {model_E_metrics['accuracy']:<15.1%} {metrics_G['accuracy']:<15.1%}")
    
    rmse_diff = (metrics_G['RMSE'] - model_E_metrics['RMSE']) / model_E_metrics['RMSE'] * 100
    print(f"\n  RNN相对于ARIMA的RMSE变化: {rmse_diff:+.2f}%")
    
    if metrics_G['RMSE'] < 592:
        print("  → RNN胜出（RMSE < 592）")
    else:
        print("  → 维持方案E（RNN RMSE未低于592）")

    # 4. 正式预测
    model, train_losses, val_losses, norm_params, supply, weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all, ci, T_sim, P_j = run_formal_prediction(
        supplier_ids, order_data, supply_data, indices, plan_ids, plan_types, Q, u, x)
    
    print(f"\n=== 产能满足率统计 ===")
    print(f"  达标周数(点估计): {weeks_meet}/24")
    print(f"  达标比例(点估计): {weeks_meet/24:.1%}")
    print(f"  平均满足率: {fulfillment_rate.mean():.2%}")
    print(f"  最小满足率: {fulfillment_rate.min():.2%}")
    print(f"  最大满足率: {fulfillment_rate.max():.2%}")
    print(f"\n=== 蒙特卡洛概率评估 ===")
    print(f"  平均达标概率: {avg_P:.2%}")
    print(f"  24周全部达标概率: {P_all:.2%}")

    # 5. 保存结果
    print("\n正在保存结果...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    save_results(supply, plan_ids, plan_types, weekly_total, fulfillment_rate, P_j,
                 weeks_meet, avg_P, P_all, ci, metrics_G, model_E_metrics)

    # 6. 生成图表
    print("\n正在生成图表...")
    plot_loss_curve(train_losses, val_losses, os.path.join(FIGURES_DIR, "problem3G_loss_curve.png"))
    plot_fulfillment_rate(fulfillment_rate, P_j, os.path.join(FIGURES_DIR, "problem3G_fulfillment_rate.png"))
    plot_weekly_supply_comparison(weekly_total, os.path.join(FIGURES_DIR, "problem3G_weekly_supply.png"))
    
    # 回测散点图
    plot_pred_vs_actual(T_true, T_pred, metrics_G, os.path.join(FIGURES_DIR, "problem3G_backtest_scatter.png"))
    
    # 周总量对比图（E vs G）
    weeks = np.arange(217, 241)
    T_E = T_true  # placeholder, will use model_E values
    plot_backtest_weekly(weeks, T_true, T_true, T_pred, os.path.join(FIGURES_DIR, "problem3G_backtest_weekly.png"))

    # 7. 打印每周详情
    print("\n=== 每周供货详情 ===")
    summary_df = pd.DataFrame({
        '周次': np.arange(1, 25),
        '总供货量': weekly_total.round(0).astype(int),
        '满足率': (fulfillment_rate * 100).round(1),
        '达标概率': (P_j * 100).round(1),
        '达标': ['✓' if r >= 1.0 else '✗' for r in fulfillment_rate]
    })
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("问题三方案G求解完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
