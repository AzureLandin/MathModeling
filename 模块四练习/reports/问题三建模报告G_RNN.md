# 问题三建模报告G：基于SimpleRNN的偏差序列预测

## Part A：方法论

### 1. 问题定位

沿用方案E的偏差序列 $D_{ij} = S_{ij} - O_{ij}$，将逐供应商ARIMA替换为池化SimpleRNN，利用所有供应商的偏差序列联合训练，预测未来24周偏差。

### 2. 数据构造

将55家供应商的偏差序列池化，以滑动窗口构造监督学习样本：

- 输入窗口长度：$w = 4$（用前4个有效周的偏差预测下一步）
- 对第 $i$ 家供应商的偏差序列 $\{D_{i,1}, \dots, D_{i,N_i}\}$，生成样本：

$$X_{i,t} = [D_{i,t}, D_{i,t+1}, D_{i,t+2}, D_{i,t+3}], \quad y_{i,t} = D_{i,t+4}$$

其中 $t = 1, 2, \dots, N_i - 4$。

总样本量约为 $\sum_i (N_i - 4) \approx 10000$ 条。

### 3. 模型结构

单层SimpleRNN（vanilla RNN，tanh激活），结构极简：

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

$$\hat{D} = W_o \cdot h_t + b_o$$

| 层 | 参数 |
|------|------|
| SimpleRNN | hidden_size=16, num_layers=1, activation=tanh |
| 全连接输出 | 16 → 1 |
| 激活 | 无（线性输出，回归任务） |
| 损失函数 | MSE |
| 优化器 | Adam, lr=0.001 |
| 训练轮数 | 100（早停，patience=10） |

选取SimpleRNN而非LSTM的理由：偏差序列时间结构仅为AR(2)量级，无需门控机制捕捉长程依赖；每家供应商仅~200步序列，LSTM参数量大（4个门）极易过拟合；SimpleRNN参数量仅为LSTM的1/4，泛化能力更强。

hidden_size取16（而非32），进一步控制模型容量，匹配问题的低复杂度。

### 4. 训练与验证

**数据划分：** 按时间切分（非随机），每家供应商的前80%有效周为训练集，后20%为验证集。保证验证集在时间上晚于训练集，模拟"用过去预测未来"。

**标准化：** 对每家供应商的偏差序列独立做Z-Score标准化（减均值除标准差），预测后反标准化。

### 5. 预测

对每家供应商，取其偏差序列最后4个值作为初始窗口，自回归外推24步：

$$\hat{D}_{i,1} = f_{\text{RNN}}([D_{i,N_i-3}, D_{i,N_i-2}, D_{i,N_i-1}, D_{i,N_i}])$$

$$\hat{D}_{i,2} = f_{\text{RNN}}([D_{i,N_i-2}, D_{i,N_i-1}, D_{i,N_i}, \hat{D}_{i,1}])$$

$$\vdots$$

供应方案：$\hat{S}_{ij} = \max(0, \; Q_i + \hat{D}_{ij}) \cdot x_{ij}$，向上取整。

### 6. 产能满足率与蒙特卡洛

满足率计算同方案E。

蒙特卡洛：对验证集残差经验分布采样，叠加到RNN点预测上，模拟10000次。

### 7. 回测对比

与方案E使用完全相同的回测设定（训练期1-216周，测试期217-240周），计算RMSE、MAE、MAPE、$R^2_{\text{test}}$，直接对比。

### 8. 判断标准

若RNN回测RMSE低于方案E（623.03）超过5%（即<592），选RNN；否则维持方案E。

---

## Part B：编程实现指引

### 依赖

- PyTorch（CPU即可，数据量小）
- 若无PyTorch，可用Keras的SimpleRNN层替代

### 实现流程

```
伪代码（PyTorch）：
import torch
import torch.nn as nn

class GapRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=16,
                          num_layers=1, nonlinearity='tanh',
                          batch_first=True)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        # x: (batch, seq_len=4, 1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后时步
        return out.squeeze()

# 数据准备
samples_X = []  # shape: (N_samples, 4)
samples_y = []  # shape: (N_samples,)

for i in range(55):
    D = gap_series[i]
    mu[i], sigma[i] = D.mean(), D.std()
    if sigma[i] == 0:
        sigma[i] = 1.0  # 防除零
    D_norm = (D - mu[i]) / sigma[i]
    for t in range(len(D_norm) - 4):
        samples_X.append(D_norm[t:t+4])
        samples_y.append(D_norm[t+4])

X = torch.tensor(samples_X).unsqueeze(-1).float()  # (N, 4, 1)
y = torch.tensor(samples_y).float()                 # (N,)

# 时间切分（每家供应商前80%训练，后20%验证）
# 按样本索引标记train/val

# 训练
model = GapRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_loss = inf
patience_counter = 0

for epoch in range(100):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 验证
    model.eval()
    val_pred = model(X_val)
    val_loss = criterion(val_pred, y_val)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最优模型
    else:
        patience_counter += 1
        if patience_counter >= 10:
            break

# 预测（自回归24步）
model.eval()
for i in range(55):
    window = last_4_normalized_gaps[i]  # shape (4,)
    preds = []
    for step in range(24):
        x_in = torch.tensor(window).unsqueeze(0).unsqueeze(-1).float()
        d_norm = model(x_in).item()
        preds.append(d_norm * sigma[i] + mu[i])  # 反标准化
        window = np.append(window[1:], d_norm)
    D_forecast[i] = preds
```

### 关键诊断

1. **回测RMSE vs 623.03**：低于592则RNN胜出，否则维持E。
2. **训练/验证loss曲线**：若验证loss先降后升，过拟合确认，需减小hidden_size或加dropout。
3. **自回归漂移**：检查24步预测中后12步是否明显偏离前12步（误差累积）。

### 边界条件

- 标准化时若 $\sigma_i = 0$（偏差恒定），该供应商直接用均值，不送入RNN。
- 随机种子固定（torch.manual_seed(42)），保证可复现。
- 若PyTorch不可用，用sklearn的MLPRegressor模拟（无循环结构，退化为窗口回归，作为fallback）。

### 输出物

- `results/problem3G_rnn_performance.csv`：训练/验证loss、回测RMSE/MAE/MAPE/R²。
- `results/problem3G_supply_plan.csv`：55×24供应方案。
- `results/problem3G_weekly_summary.csv`：每周总供货量、满足率。
- `results/problem3G_capacity_stats.csv`：达标周数、达标比例。
- `figures/problem3G_loss_curve.png`：训练/验证loss曲线。
- `figures/problem3G_fulfillment_rate.png`：24周满足率折线图。
- `figures/problem3G_pred_vs_actual.png`：回测期预测vs真实散点图。
