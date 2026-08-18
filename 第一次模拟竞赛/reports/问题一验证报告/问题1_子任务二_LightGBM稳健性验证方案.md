# 问题 1（子任务二）：LightGBM 非线性稳健性验证方案

> **目的**：以低复杂度 LightGBM 作为非线性对照模型，检验岭回归对模型形式的稳健性。
>
> **重要说明**：LightGBM 不是对岭回归“正确性”的统计证明，而是不同模型假设下的折外预测对照。由于样本仅包含 10 个区域，本方案严禁采用高复杂度树模型或大范围调参。

---

## 1. 验证目标与预测对象

预测对象与岭回归保持完全一致。对区域 \(i\)，工作日和周末全天充电量分别为：

\[
D_i^{\mathrm{wd}}=\sum_{t=1}^{24}P_{i,t}^{\mathrm{wd}},
\qquad
D_i^{\mathrm{we}}=\sum_{t=1}^{24}P_{i,t}^{\mathrm{we}}.
\]

综合日均充电需求定义为：

\[
y_i=\bar D_i=
\frac{5D_i^{\mathrm{wd}}+2D_i^{\mathrm{we}}}{7},
\qquad i=1,2,\ldots,10.
\]

其中 \(y_i\) 的单位为 \(\mathrm{kWh/day}\)。

本验证检验的问题为：在相同输入信息下，低复杂度非线性模型能否显著优于岭回归的折外预测表现。

---

## 2. 输入特征

第一轮 LightGBM 对照采用与现有岭回归 R1 相同的四项输入特征：

\[
\boldsymbol{x}_i=
\left(
\mathrm{人口密度}_i,
\mathrm{车流量}_i,
\mathrm{商业POI}_i,
\mathrm{现有总充电桩数}_i
\right).
\]

其中：

\[
N_i^{\mathrm{total}}
=N_i^{\mathrm{fast}}+N_i^{\mathrm{slow}}.
\]

注意事项：

1. 不将总桩数、快充桩数、慢充桩数同时输入模型；
2. LightGBM 不需要对输入特征做标准化；
3. 现有总桩数仅作为预测信息输入，不得根据其特征重要性作因果解释；
4. 后续是否将总桩数保留为正式需求模型变量，需结合岭回归消融与本次 LightGBM 对照结果共同判断。

---

## 3. 固定的低复杂度 LightGBM 参数

由于总样本量仅为：

\[
n=10,
\]

不进行大范围网格搜索。采用预先固定的保守参数：

```python
objective = "regression"
n_estimators = 20
learning_rate = 0.05

max_depth = 1
num_leaves = 2
min_child_samples = 4

feature_fraction = 1.0
bagging_fraction = 1.0
bagging_freq = 0

lambda_l1 = 0.0
lambda_l2 = 10.0

random_state = 2026
verbosity = -1
```

参数设计原则：

| 参数 | 设定 | 作用 |
|---|---:|---|
| `max_depth` | 1 | 每棵树仅允许一次二分，避免形成复杂区域划分。 |
| `num_leaves` | 2 | 每棵树只保留两个叶节点。 |
| `min_child_samples` | 4 | 训练折仅有 9 个样本时，限制叶节点样本过少。 |
| `n_estimators` | 20 | 限制迭代轮数。 |
| `learning_rate` | 0.05 | 缓慢学习、抑制过拟合。 |
| `lambda_l2` | 10 | 收缩叶节点输出，提升稳定性。 |
| 采样比例 | 1.0 | 关闭随机行、列采样，保证可复现。 |

严禁在正式验证中使用：

```text
max_depth >= 3
num_leaves > 3
n_estimators >= 100
大范围参数网格搜索
训练集 R2 作为模型选择依据
```

---

## 4. 验证方法：留一交叉验证

采用与岭回归完全一致的留一交叉验证（LOOCV）。对于区域 \(i\)：

1. 将区域 \(i\) 留作测试样本；
2. 其余 9 个区域作为训练集；
3. 使用固定参数训练低复杂度 LightGBM；
4. 得到被留出区域的预测值 \(\hat y_i\)；
5. 重复 10 次，汇总全部折外预测值。

伪代码：

```text
for i in 1,...,10:
    train = all regions except i
    test = region i

    fit LightGBM-L1 on train using fixed parameters
    predict test

collect all 10 out-of-fold predictions
calculate MAE, RMSE, MAPE, and R2_CV
```

岭回归需在训练折中单独进行标准化；LightGBM 直接使用原始输入特征。两者采用相同的外层测试折，具有可比性。

---

## 5. 评价指标

### 平均绝对误差

\[
\mathrm{MAE}
=
\frac{1}{10}
\sum_{i=1}^{10}
|y_i-\hat y_i|.
\]

### 均方根误差

\[
\mathrm{RMSE}
=
\sqrt{
\frac{1}{10}
\sum_{i=1}^{10}
(y_i-\hat y_i)^2
}.
\]

### 平均绝对百分比误差

\[
\mathrm{MAPE}
=
\frac{100\%}{10}
\sum_{i=1}^{10}
\left|
\frac{y_i-\hat y_i}{y_i}
\right|.
\]

### 留一交叉验证决定系数

\[
R^2_{\mathrm{CV}}
=
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}.
\]

此外，必须报告：

- 最大绝对误差；
- 最大相对误差；
- 对应区域编号；
- 预测值是否出现负数；
- 区域 7、8、9 的预测误差变化。

---

## 6. 结果输出格式

### 表 1：统一模型对照

| 模型 | LOOCV-MAE | LOOCV-RMSE | LOOCV-MAPE | \(R^2_{\mathrm{CV}}\) | 最大相对误差 | 负预测数 |
|---|---:|---:|---:|---:|---:|---:|
| 均值基准 |  |  |  |  |  |  |
| 岭回归 R1 | 2967.11 | 3401.18 | 25.97% | 0.2404 | 66.19% | 0 |
| 岭回归 R2 | 3223.30 | 3740.76 | 27.94% | 0.0812 |  | 0 |
| LightGBM-L1 |  |  |  |  |  |  |

说明：岭回归 R1 的最大相对误差为 66.19%，对应区域 7；区域 9 的最大绝对误差为 \(6012.51\ \mathrm{kWh/day}\)，其相对误差为 40.20%。

### 表 2：逐区域折外预测对比

| 区域 | 实际综合日均负荷 | 岭回归 R1 预测 | LightGBM-L1 预测 | 岭回归绝对误差 | LightGBM 绝对误差 | 表现更优模型 |
|---:|---:|---:|---:|---:|---:|---|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| \(\vdots\) |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |

建议补充两张图：

1. 实际综合日均负荷与两类模型 LOOCV 预测值对比图；
2. 两类模型的区域绝对误差对比柱状图。

---

## 7. 模型选择规则

### 情况 A：LightGBM 明显优于岭回归

若同时满足：

\[
\mathrm{RMSE}_{\mathrm{LGB}}
<
0.95\,\mathrm{RMSE}_{\mathrm{Ridge}},
\]

且：

\[
\mathrm{MAE}_{\mathrm{LGB}}
<
\mathrm{MAE}_{\mathrm{Ridge}},
\]

并且未出现明显异常预测值，则说明数据中可能存在一定非线性关系。

此时仍建议将岭回归作为可解释的主模型，把 LightGBM 作为非线性稳健性检验；不能只凭 LightGBM 的微小指标优势完全取代岭回归。

### 情况 B：LightGBM 与岭回归接近

若：

\[
0.95\,\mathrm{RMSE}_{\mathrm{Ridge}}
\leq
\mathrm{RMSE}_{\mathrm{LGB}}
<
\mathrm{RMSE}_{\mathrm{Ridge}},
\]

则不因小于 5% 的 RMSE 改善而改用 LightGBM。此时优先选择岭回归作为主模型，因为其参数更透明、模型复杂度更低。

### 情况 C：LightGBM 更差或不稳定

若：

\[
\mathrm{RMSE}_{\mathrm{LGB}}
\geq
\mathrm{RMSE}_{\mathrm{Ridge}},
\]

或区域 7、8、9 的预测误差显著恶化，则认为当前区域样本不足以支撑非线性树模型，最终论文保留岭回归即可。

---

## 8. 禁止性要求

由于样本量极小，以下内容不得用于正式论文结论：

1. 不使用 LightGBM 训练集 \(R^2\)；
2. 不使用树结构图解释区域规则；
3. 不基于特征重要性排名判断变量重要程度；
4. 不使用 SHAP 值解释特征贡献；
5. 不用 LightGBM 折外预测值替代 2025 年实际综合日均负荷；
6. 不基于单一 LightGBM 结果确定问题 2 的新增桩配置。

现有总桩数可能存在供给约束与历史需求反向影响，故即使模型将其识别为高重要性特征，也不表示“增加充电桩会直接增加需求”。

---

## 9. 结论解释边界

无论对照结果如何，当前模型所对应的是 2025 年区域横截面需求估计，不等同于未来多期时间序列预测。未来需求仍需在明确人口、车流、POI 或新能源汽车保有量情景后再做外推。
