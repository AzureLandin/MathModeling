# 问题三建模报告E：基于偏差序列ARIMA的供应方案预测

## Part A：方法论

### 1. 问题定位

对每家供应商定义供货偏差序列（实际供货量与订货量之差），利用时间序列模型预测未来24周的偏差，叠加问题二的固定订货量得到供应方案，统计产能满足率。

### 2. 偏差序列构建

对第 $i$ 家供应商，定义有效周上的供货偏差：

$$D_{ij} = S_{ij} - O_{ij}, \quad j \in W_i$$

$D > 0$ 表示超额供货，$D < 0$ 表示供货不足，$D = 0$ 表示精确交货。

将有效周的偏差按时间顺序排列为序列 $\{D_{i,1}, D_{i,2}, \dots, D_{i,N_i}\}$。

与供货率 $r = S/O$ 的区别：偏差为绝对量（m³），不涉及除法，不存在小分母抬高比值的问题。

### 3. 平稳性检验与模型选择

对每家供应商的偏差序列执行：

**ADF单位根检验：**

$$H_0: \text{序列存在单位根（非平稳）}$$

若 $p < 0.05$，序列平稳，直接拟合ARMA。否则一阶差分 $\Delta D_t = D_t - D_{t-1}$ 后重新检验。

**Ljung-Box自相关检验：**

$$H_0: \text{前} K \text{阶自相关系数均为0（白噪声）}$$

若 $p > 0.05$，偏差序列无时间记忆性，ARIMA退化为常数均值：$\hat{D}_{ij} = \bar{D}_i = \frac{1}{N_i}\sum_{j \in W_i} D_{ij}$。

### 4. ARIMA模型拟合

对存在显著自相关的供应商，拟合ARIMA($p, d, q$)：

$$\phi(B)(1-B)^d D_t = \theta(B) \varepsilon_t$$

定阶：AIC准则，搜索 $p \in \{0,1,2,3\}$，$q \in \{0,1,2,3\}$，$d$ 由ADF确定。

对有效周数 $N_i < 30$ 的供应商，序列过短不做ARIMA，直接用均值 $\bar{D}_i$。

### 5. 预测与供应方案

外推未来24周偏差预测值 $\hat{D}_{i,1}, \dots, \hat{D}_{i,24}$。

供应方案：

$$\hat{S}_{ij} = (Q_i + \hat{D}_{ij}) \cdot x_{ij}$$

截断非负：$\hat{S}_{ij} = \max(0, \; Q_i + \hat{D}_{ij}) \cdot x_{ij}$。

向上取整：$\hat{S}_{ij}^{\text{final}} = \lceil \hat{S}_{ij} \rceil$。

### 6. 产能满足率

$$\hat{T}_j = \sum_{i=1}^{55} \frac{\hat{S}_{ij}^{\text{final}}}{u_i}$$

$$\text{满足率}_j = \frac{\hat{T}_j}{28200}$$

统计24周中 $\hat{T}_j \geq 28200$ 的周数占比。

### 7. 不确定性量化（蒙特卡洛）

利用ARIMA残差（或均值模型的偏差残差）经验分布采样：

$$\hat{S}_{ij}^{(m)} = Q_i + \hat{D}_{ij} + \varepsilon_i^{(m)}$$

模拟 $M = 10000$ 次，计算每周达标概率。

### 8. 模型特点

优势：建模对象为"偏离订货量的幅度"，与题目"实际供货量可能多于或少于订货量"的表述直接对应；不涉及比值运算，无小分母偏差；若偏差存在惯性（连续几周超额/缺货），ARIMA能捕捉并利用。

局限：逐供应商建模，有效周少的供应商参数不稳定；假设偏差的时间结构在未来24周保持不变；不利用供应商间的共性信息（与池化回归互补）。

---

## Part B：编程实现指引

### 实现流程

```
伪代码：
for i in range(55):
    D_series = S[i, W[i]] - O[i, W[i]]  # 偏差序列
    
    if len(D_series) < 30:
        D_forecast[i] = [mean(D_series)] * 24
        model_type[i] = "均值(短序列)"
        continue
    
    # ADF检验
    adf_p = adfuller(D_series)[1]
    d = 0 if adf_p < 0.05 else 1
    
    # Ljung-Box
    series_check = D_series if d == 0 else np.diff(D_series)
    lb_p = acorr_ljungbox(series_check, lags=10).iloc[-1, 1]
    
    if lb_p > 0.05:
        D_forecast[i] = [mean(D_series)] * 24
        model_type[i] = "均值(白噪声)"
        continue
    
    # ARIMA选阶
    best_aic = inf
    for p in range(4):
        for q in range(4):
            try:
                fit = ARIMA(D_series, order=(p,d,q)).fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p,d,q)
            except:
                continue
    
    fit = ARIMA(D_series, order=best_order).fit()
    D_forecast[i] = fit.forecast(steps=24)
    model_type[i] = f"ARIMA{best_order}"

# 供应方案
for i in range(55):
    for j in range(24):
        S_pred[i,j] = max(0, Q[i] + D_forecast[i][j]) * x[i,j]
        S_plan[i,j] = ceil(S_pred[i,j])
```

### 关键输出（用于判断模型是否可用）

编程手跑完后，优先输出以下诊断信息供建模手判断：

1. **模型类型分布**：多少家用ARIMA、多少家退化为均值。若>70%退化为均值，说明偏差无时间结构，ARIMA无优势。
2. **偏差均值 $\bar{D}_i$ 的分布**：若大部分 $\bar{D}_i > 0$（超额供货），预测供货量 = $Q_i + \bar{D}_i > Q_i$，满足率可能偏高；若 $\bar{D}_i$ 接近0或为负，满足率会更保守。
3. **与池化回归（方案D）的满足率对比**：若两者接近（±5%以内），说明结论稳健；若差异大，需分析原因。

### 边界条件

- ARIMA拟合失败（不收敛）时退化为均值。
- 预测偏差截断：$Q_i + \hat{D}_{ij} \geq 0$，防止负供货量。
- 差分序列预测需还原（statsmodels自动处理）。

### 输出物

- `results/problem3E_model_selection.csv`：每家供应商模型类型、ADF p值、LB p值、AIC、偏差均值。
- `results/problem3E_supply_plan.csv`：55×24供应方案。
- `results/problem3E_weekly_summary.csv`：每周总供货量（产品体积）、满足率。
- `results/problem3E_capacity_stats.csv`：达标周数、达标比例。
- `figures/problem3E_model_distribution.png`：模型类型分布饼图。
- `figures/problem3E_gap_distribution.png`：55家供应商偏差均值 $\bar{D}_i$ 的直方图（标注0线）。
- `figures/problem3E_fulfillment_rate.png`：24周满足率折线图。
- `figures/problem3E_acf_example.png`：2-3家典型供应商偏差序列的ACF图。
