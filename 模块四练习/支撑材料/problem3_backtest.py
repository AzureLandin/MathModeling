"""
问题三回测程序：对比方案D（池化OLS）和方案E（偏差ARIMA）的预测准确性
数据划分：训练期1-216周，测试期217-240周
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "2026D附件", "附件1 近5年402家供应商的相关数据.xlsx")
ORDERING_PATH = os.path.join(BASE_DIR, "results", "problem2_ordering_plan.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def load_data():
    from problem_data_loader import load_supplier_data
    from problem_indicators import compute_indicators
    
    supplier_ids, material_types, order_data, supply_data = load_supplier_data(DATA_PATH)
    F1, F2, F3, F4, N = compute_indicators(order_data, supply_data)
    
    # 加载订购方案获取55家供应商信息
    plan_df = pd.read_csv(ORDERING_PATH)
    plan_ids = plan_df['供应商ID'].values
    plan_types = plan_df['材料类别'].values
    
    # 获取索引
    indices = []
    for tid in plan_ids:
        idx = np.where(supplier_ids == tid)[0][0]
        indices.append(idx)
    
    return supplier_ids, order_data, supply_data, F1, F2, F3, F4, plan_ids, plan_types, indices


def compute_F_subset(order_data, supply_data, indices, week_start, week_end):
    """在指定周范围内计算F1-F4"""
    F1 = np.zeros(len(indices))
    F2 = np.zeros(len(indices))
    F3 = np.zeros(len(indices))
    F4 = np.zeros(len(indices))
    
    for k, idx in enumerate(indices):
        # 指定周范围内的有效周
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


def build_long_format(order_data, supply_data, indices, F1, F2, F3, F4, week_start, week_end):
    """构建指定周范围的长格式数据"""
    records = []
    for k, idx in enumerate(indices):
        for j in range(week_start, week_end):
            if order_data[idx, j] > 0:
                records.append({
                    'S': supply_data[idx, j],
                    'O': order_data[idx, j],
                    'F1': F1[k],
                    'F2': F2[k],
                    'F3': F3[k],
                    'F4': F4[k],
                    'supplier_idx': k,
                    'week': j
                })
    return pd.DataFrame(records)


def train_ols(train_df):
    """训练池化OLS模型"""
    X = train_df[['O', 'F1', 'F2', 'F3', 'F4']]
    X = sm.add_constant(X)
    y = train_df['S']
    model = sm.OLS(y, X).fit()
    return model


def predict_ols(model, test_df):
    """用OLS模型预测测试集"""
    X = test_df[['O', 'F1', 'F2', 'F3', 'F4']]
    X = sm.add_constant(X)
    y_pred = model.predict(X)
    y_pred = np.maximum(0, y_pred)
    return y_pred.values


def train_arima_deviation(order_data, supply_data, indices, week_start, week_end, min_obs=30):
    """训练偏差ARIMA模型"""
    models = []
    
    for k, idx in enumerate(indices):
        mask = order_data[idx, week_start:week_end] > 0
        O = order_data[idx, week_start:week_end][mask]
        S = supply_data[idx, week_start:week_end][mask]
        D = S - O
        N = len(D)
        mean_D = np.mean(D) if N > 0 else 0
        
        if N < min_obs:
            models.append({'type': 'mean', 'forecast': mean_D, 'residuals': D - mean_D})
            continue
        
        # ADF检验
        try:
            adf_p = adfuller(D)[1]
        except:
            adf_p = 1.0
        d = 0 if adf_p < 0.05 else 1
        
        # Ljung-Box检验
        D_diff = np.diff(D) if d == 1 else D
        try:
            lb_p = acorr_ljungbox(D_diff, lags=min(10, len(D_diff)-1)).iloc[-1, 1]
        except:
            lb_p = 0.0
        
        if lb_p > 0.05:
            models.append({'type': 'mean_white', 'forecast': mean_D, 'residuals': D - mean_D})
            continue
        
        # ARIMA选阶
        best_aic = np.inf
        best_model = None
        for p in range(4):
            for q in range(4):
                try:
                    fit = ARIMA(D, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_model = fit
                except:
                    continue
        
        if best_model is None:
            models.append({'type': 'mean_fail', 'forecast': mean_D, 'residuals': D - mean_D})
        else:
            models.append({
                'type': f'ARIMA',
                'forecast': best_model.forecast(steps=24),
                'residuals': best_model.resid
            })
    
    return models


def predict_arima(arima_models, order_data, indices, week_start, week_end):
    """用ARIMA偏差模型预测测试期"""
    n_suppliers = len(indices)
    n_weeks = week_end - week_start
    S_pred = np.zeros((n_suppliers, n_weeks))
    
    for k, idx in enumerate(indices):
        m = arima_models[k]
        for j in range(n_weeks):
            week_idx = week_start + j
            if order_data[idx, week_idx] > 0:
                if m['type'] == 'mean' or m['type'] == 'mean_white' or m['type'] == 'mean_fail':
                    D_hat = m['forecast']
                else:
                    # ARIMA forecast is24 steps, map to test weeks
                    D_hat = m['forecast'][j] if j < len(m['forecast']) else m['forecast'][-1]
                S_pred[k, j] = max(0, order_data[idx, week_idx] + D_hat)
    
    return S_pred


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


def compute_weekly_metrics(S_true, S_pred, u, capacity=28200):
    """计算周总量指标"""
    T_true = (S_true / u[:, np.newaxis]).sum(axis=0)
    T_pred = (S_pred / u[:, np.newaxis]).sum(axis=0)
    
    rmse_week = np.sqrt(np.mean((T_true - T_pred) ** 2))
    mape_week = np.mean(np.abs(T_true - T_pred) / T_true) * 100
    
    # 达标判断
    true_meet = T_true >= capacity
    pred_meet = T_pred >= capacity
    accuracy = (true_meet == pred_meet).mean()
    
    return {
        'T_true': T_true,
        'T_pred': T_pred,
        'RMSE_week': rmse_week,
        'MAPE_week': mape_week,
        'accuracy': accuracy,
        'true_meet': true_meet,
        'pred_meet': pred_meet
    }


def plot_scatter(y_true_D, y_pred_D, y_true_E, y_pred_E, save_path):
    """绘制预测值vs真实值散点图（仅ARIMA）"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true_E, y_pred_E, alpha=0.4, s=15, color='steelblue')
    max_val = max(y_true_E.max(), y_pred_E.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测线')
    ax.set_xlabel('真实供货量 (m$^3$)', fontsize=14)
    ax.set_ylabel('预测供货量 (m$^3$)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  散点图已保存: {save_path}")


def plot_weekly_comparison(weeks, T_true, T_D, T_E, save_path, capacity=28200):
    """绘制周总量对比图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(weeks, T_true, 'ko-', linewidth=2, markersize=6, label='真实值')
    ax.plot(weeks, T_D, 's--', color='steelblue', linewidth=1.5, markersize=5, label='方案D (OLS)')
    ax.plot(weeks, T_D, '^--', color='coral', linewidth=1.5, markersize=5, label='方案E (ARIMA)')
    ax.axhline(y=capacity, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='产能要求')
    
    ax.set_xlabel('周次', fontsize=14)
    ax.set_ylabel('总供货量 (产品体积 m$^3$)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  周总量对比图已保存: {save_path}")


def main():
    print("=" * 60)
    print("问题三回测：方案D vs 方案E 预测准确性对比")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n正在加载数据...")
    supplier_ids, order_data, supply_data, F1_full, F2_full, F3_full, F4_full, plan_ids, plan_types, indices = load_data()
    
    type_map = {'A': 0.6, 'B': 0.66, 'C': 0.72}
    u = np.array([type_map[t] for t in plan_types])
    
    # 2. 数据划分
    train_end = 216  # 训练期：0-215 (共216周)
    test_start = 216  # 测试期：216-239 (共24周)
    test_end = 240
    
    print(f"  训练期：第1-{train_end}周")
    print(f"  测试期：第{test_start+1}-{test_end}周")
    
    # 3. 方案D回测
    print("\n" + "=" * 60)
    print("方案D回测：池化OLS回归")
    print("=" * 60)
    
    # 计算训练期F值（避免数据泄露）
    F1_train, F2_train, F3_train, F4_train = compute_F_subset(
        order_data, supply_data, indices, 0, train_end
    )
    
    # 构建训练集和测试集长格式数据
    train_df = build_long_format(order_data, supply_data, indices, F1_train, F2_train, F3_train, F4_train, 0, train_end)
    test_df = build_long_format(order_data, supply_data, indices, F1_train, F2_train, F3_train, F4_train, test_start, test_end)
    
    print(f"  训练集样本数: {len(train_df)}")
    print(f"  测试集样本数: {len(test_df)}")
    
    # 训练OLS
    ols_model = train_ols(train_df)
    print(f"  OLS R² (训练集): {ols_model.rsquared:.4f}")
    
    # 预测
    y_pred_D = predict_ols(ols_model, test_df)
    y_true_D = test_df['S'].values
    
    # 指标
    metrics_D = compute_metrics(y_true_D, y_pred_D, '方案D (OLS)')
    
    # 4. 方案E回测
    print("\n" + "=" * 60)
    print("方案E回测：偏差序列ARIMA")
    print("=" * 60)
    
    # 训练ARIMA偏差模型
    arima_models = train_arima_deviation(order_data, supply_data, indices, 0, train_end)
    
    # 统计模型类型
    arima_count = sum(1 for m in arima_models if 'ARIMA' in m['type'])
    mean_count = len(arima_models) - arima_count
    print(f"  ARIMA模型: {arima_count}家")
    print(f"  均值模型: {mean_count}家")
    
    # 预测
    S_pred_E = predict_arima(arima_models, order_data, indices, test_start, test_end)
    
    # 构建测试集真实值矩阵
    S_true_matrix = np.zeros((len(indices), test_end - test_start))
    for k, idx in enumerate(indices):
        for j in range(test_start, test_end):
            S_true_matrix[k, j - test_start] = supply_data[idx, j]
    
    # 逐观测指标（展开为向量）
    y_true_E = S_true_matrix.flatten()
    y_pred_E = S_pred_E.flatten()
    mask_E = order_data[indices, test_start:test_end].flatten() > 0
    metrics_E = compute_metrics(y_true_E[mask_E], y_pred_E[mask_E], '方案E (ARIMA)')
    
    # 5. 周总量指标
    metrics_D_week = compute_weekly_metrics(S_true_matrix, 
                                             np.array([y_pred_D[test_df['supplier_idx'].values == k].sum() 
                                                       for k in range(len(indices))]).reshape(-1, 1).repeat(24, axis=1) 
                                             if False else S_true_matrix,  # placeholder
                                             u)
    
    # 重新计算方案D的周总量
    S_pred_D_matrix = np.zeros((len(indices), test_end - test_start))
    for i, row in test_df.iterrows():
        k = int(row['supplier_idx'])
        j = int(row['week']) - test_start
        x_pred = np.array([[1, row['O'], row['F1'], row['F2'], row['F3'], row['F4']]])
        S_pred_D_matrix[k, j] = max(0, np.dot(x_pred, ols_model.params)[0])
    
    metrics_D_week = compute_weekly_metrics(S_true_matrix, S_pred_D_matrix, u)
    metrics_E_week = compute_weekly_metrics(S_true_matrix, S_pred_E, u)
    
    # 6. 输出结果
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)
    
    print("\n=== 逐观测指标 ===")
    print(f"{'指标':<15} {'方案D (OLS)':<15} {'方案E (ARIMA)':<15}")
    print("-" * 45)
    print(f"{'样本数':<15} {metrics_D['n_obs']:<15} {metrics_E['n_obs']:<15}")
    print(f"{'RMSE':<15} {metrics_D['RMSE']:<15.2f} {metrics_E['RMSE']:<15.2f}")
    print(f"{'MAE':<15} {metrics_D['MAE']:<15.2f} {metrics_E['MAE']:<15.2f}")
    print(f"{'MAPE (%)':<15} {metrics_D['MAPE']:<15.2f} {metrics_E['MAPE']:<15.2f}")
    print(f"{'R²':<15} {metrics_D['R2']:<15.4f} {metrics_E['R2']:<15.4f}")
    
    print("\n=== 周总量指标 ===")
    print(f"{'指标':<15} {'方案D (OLS)':<15} {'方案E (ARIMA)':<15}")
    print("-" * 45)
    print(f"{'RMSE_week':<15} {metrics_D_week['RMSE_week']:<15.0f} {metrics_E_week['RMSE_week']:<15.0f}")
    print(f"{'MAPE_week (%)':<15} {metrics_D_week['MAPE_week']:<15.2f} {metrics_E_week['MAPE_week']:<15.2f}")
    print(f"{'达标准确率':<15} {metrics_D_week['accuracy']:<15.1%} {metrics_E_week['accuracy']:<15.1%}")
    
    # 判断
    rmse_diff = (metrics_E['RMSE'] - metrics_D['RMSE']) / metrics_D['RMSE'] * 100
    mae_diff = (metrics_E['MAE'] - metrics_D['MAE']) / metrics_D['MAE'] * 100
    
    print("\n=== 模型选择判断 ===")
    print(f"  E相对于D的RMSE变化: {rmse_diff:+.2f}%")
    print(f"  E相对于D的MAE变化: {mae_diff:+.2f}%")
    
    if rmse_diff < -5 and mae_diff < -5:
        print("  → 方案E更优（RMSE和MAE均低5%以上）")
    elif rmse_diff > 5 and mae_diff > 5:
        print("  → 方案D更优（RMSE和MAE均低5%以上）")
    else:
        print("  → 两者差距<5%，视为等价")
    
    # 7. 保存结果
    print("\n正在保存结果...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 指标表
    metrics_df = pd.DataFrame([metrics_D, metrics_E])
    metrics_df['RMSE_week'] = [metrics_D_week['RMSE_week'], metrics_E_week['RMSE_week']]
    metrics_df['MAPE_week'] = [metrics_D_week['MAPE_week'], metrics_E_week['MAPE_week']]
    metrics_df['accuracy'] = [metrics_D_week['accuracy'], metrics_E_week['accuracy']]
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "problem3_backtest_metrics.csv"), index=False, encoding='utf-8-sig')
    print(f"  回测指标已保存")
    
    # 每周明细
    weekly_df = pd.DataFrame({
        '周次': np.arange(test_start+1, test_end+1),
        '真实总量': metrics_D_week['T_true'].round(0).astype(int),
        'D预测总量': metrics_D_week['T_pred'].round(0).astype(int),
        'E预测总量': metrics_E_week['T_pred'].round(0).astype(int),
        '真实达标': ['✓' if t else '✗' for t in metrics_D_week['true_meet']],
        'D预测达标': ['✓' if t else '✗' for t in metrics_D_week['pred_meet']],
        'E预测达标': ['✓' if t else '✗' for t in metrics_E_week['pred_meet']]
    })
    weekly_df.to_csv(os.path.join(RESULTS_DIR, "problem3_backtest_weekly.csv"), index=False, encoding='utf-8-sig')
    print(f"  每周明细已保存")
    
    # 8. 生成图表
    print("\n正在生成图表...")
    
    # 散点图
    plot_scatter(y_true_D, y_pred_D, y_true_E[mask_E], y_pred_E[mask_E],
                 os.path.join(FIGURES_DIR, "problem3_backtest_scatter.png"))
    
    # 周总量对比图
    weeks = np.arange(test_start+1, test_end+1)
    plot_weekly_comparison(weeks, metrics_D_week['T_true'], metrics_D_week['T_pred'], metrics_E_week['T_pred'],
                          os.path.join(FIGURES_DIR, "problem3_backtest_weekly.png"))
    
    print("\n" + "=" * 60)
    print("回测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
