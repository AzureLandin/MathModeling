"""
问题三方案G求解器：基于SimpleRNN的偏差序列预测
沿用方案E的偏差序列，用池化RNN替代逐供应商ARIMA
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GapRNN(nn.Module):
    """单层SimpleRNN模型"""
    def __init__(self, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
                          num_layers=1, nonlinearity='tanh',
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len=4, 1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后时步
        return out.squeeze()


def build_deviation_series(order_data, supply_data, indices):
    """构建55家供应商的偏差序列"""
    deviation_series = []
    for idx in indices:
        mask = order_data[idx, :] > 0
        O = order_data[idx, mask]
        S = supply_data[idx, mask]
        D = S - O
        deviation_series.append(D)
    return deviation_series


def prepare_rnn_data(deviation_series, window_size=4):
    """
    构建RNN训练数据（滑动窗口）
    每家供应商独立标准化，按时间切分训练/验证集
    """
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    # 保存标准化参数和序列信息
    norm_params = []
    
    for i, D in enumerate(deviation_series):
        N = len(D)
        mu = D.mean()
        sigma = D.std()
        if sigma == 0:
            sigma = 1.0  # 防除零
        
        norm_params.append({'mu': mu, 'sigma': sigma, 'N': N})
        
        # Z-Score标准化
        D_norm = (D - mu) / sigma
        
        # 滑动窗口
        n_samples = N - window_size
        if n_samples < 5:
            continue
        
        # 时间切分：前80%训练，后20%验证
        n_train = int(n_samples * 0.8)
        
        for t in range(n_samples):
            X = D_norm[t:t+window_size]
            y = D_norm[t+window_size]
            
            if t < n_train:
                X_train_list.append(X)
                y_train_list.append(y)
            else:
                X_val_list.append(X)
                y_val_list.append(y)
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)
    
    return X_train, y_train, X_val, y_val, norm_params


def train_rnn(X_train, y_train, X_val, y_val, hidden_size=16, lr=0.001, 
              max_epochs=100, patience=10, seed=42):
    """训练SimpleRNN模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 转换为tensor
    X_train_t = torch.tensor(X_train).unsqueeze(-1).float()  # (N, 4, 1)
    y_train_t = torch.tensor(y_train).float()
    X_val_t = torch.tensor(X_val).unsqueeze(-1).float()
    y_val_t = torch.tensor(y_val).float()
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 模型
    model = GapRNN(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 训练
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 加载最优模型
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def predict_rnn(model, deviation_series, norm_params, order_data, Q, x, indices, 
                window_size=4, n_steps=24):
    """
    用RNN自回归预测未来24步偏差
    """
    model.eval()
    
    n = len(Q)
    T = 24
    S_pred = np.zeros((n, T))
    
    for i in range(n):
        D = deviation_series[i]
        mu = norm_params[i]['mu']
        sigma = norm_params[i]['sigma']
        
        # 取最后window_size个值作为初始窗口
        if len(D) < window_size:
            # 序列太短，用均值
            D_forecast = np.full(n_steps, mu)
        else:
            # 标准化最后window_size个值
            window = (D[-window_size:] - mu) / sigma
            
            # 自回归预测24步
            D_forecast_norm = []
            for step in range(n_steps):
                x_in = torch.tensor(window).unsqueeze(0).unsqueeze(-1).float()
                with torch.no_grad():
                    d_norm = model(x_in).item()
                D_forecast_norm.append(d_norm)
                window = np.append(window[1:], d_norm)
            
            # 反标准化
            D_forecast = np.array(D_forecast_norm) * sigma + mu
        
        # 供应方案：S = max(0, Q + D_hat)
        S_pred[i, :] = np.maximum(0, Q[i] + D_forecast) * x[i, :]
    
    return S_pred


def monte_carlo_simulation(model, deviation_series, norm_params, Q, u, x, indices,
                           X_val, y_val, M=10000, seed=42, window_size=4):
    """
    利用验证集残差进行蒙特卡洛模拟
    """
    np.random.seed(seed)
    
    # 计算验证集残差
    model.eval()
    X_val_t = torch.tensor(X_val).unsqueeze(-1).float()
    with torch.no_grad():
        y_pred_val = model(X_val_t).numpy()
    residuals = y_val - y_pred_val
    
    n = len(Q)
    T = 24
    T_sim = np.zeros((M, T))
    
    print(f"  开始{M}次蒙特卡洛模拟...")
    
    for m in range(M):
        weekly_product = np.zeros(T)
        
        for i in range(n):
            D = deviation_series[i]
            mu = norm_params[i]['mu']
            sigma = norm_params[i]['sigma']
            
            if len(D) < window_size:
                D_forecast = np.full(T, mu)
            else:
                window = (D[-window_size:] - mu) / sigma
                D_forecast_norm = []
                
                # 从残差中采样
                noise_samples = np.random.choice(residuals, size=T, replace=True)
                
                for step in range(T):
                    x_in = torch.tensor(window).unsqueeze(0).unsqueeze(-1).float()
                    with torch.no_grad():
                        d_norm = model(x_in).item()
                    # 加入噪声
                    d_norm_noisy = d_norm + noise_samples[step]
                    D_forecast_norm.append(d_norm_noisy)
                    window = np.append(window[1:], d_norm_noisy)
                
                D_forecast = np.array(D_forecast_norm) * sigma + mu
            
            S_sim = np.maximum(0, Q[i] + D_forecast) * x[i, :]
            weekly_product += S_sim / u[i]
        
        T_sim[m, :] = weekly_product
    
    P_j = (T_sim >= 28200).mean(axis=0)
    
    return T_sim, P_j


def compute_supply_plan(S_pred):
    """向上取整"""
    S_rounded = np.round(S_pred, 6)
    supply = np.ceil(S_rounded).astype(int)
    return supply


def compute_capacity_stats(supply, u, T_sim, P_j, capacity=28200):
    """计算产能满足率统计"""
    weekly_total = (supply / u[:, np.newaxis]).sum(axis=0)
    fulfillment_rate = weekly_total / capacity
    weeks_meet = (weekly_total >= capacity).sum()
    
    avg_P = P_j.mean()
    P_all = np.prod(P_j)
    
    return weekly_total, fulfillment_rate, weeks_meet, avg_P, P_all


def compute_confidence_intervals(T_sim, percentiles=[5, 25, 50, 75, 95]):
    """计算置信区间"""
    ci = np.percentile(T_sim, percentiles, axis=0)
    return ci
