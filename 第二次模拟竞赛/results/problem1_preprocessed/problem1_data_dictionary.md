# 问题1 数据字典  problem1_data_dictionary.md

> 生成时间：2026-08-22 15:16:46
> 口径来源：reports/问题1_数据预处理报告.md  §2-§7

## 1. 输入文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 附件2-排放清单 | `附件/附件2-2022年30个省份排放清单.xlsx` | 30 个省份工作表（`*2022`）+ NOTE；总量取 `TotalEmissions` 行 `Scope_1_Total` 列（Mt CO2） |
| 指标说明2 | `附件/附件2的指标说明.xlsx` | 行业/能源口径核对，不直接参与计算 |
| 人口与GDP | `附件/人口与GDP数据.csv` | 实为 XLSX，含 `人口`/`GDP` 双表，字段 `省份`/`人口（万人）`/`GDP（亿元）`，各 30 行 |
| 附件1 | `附件/附件1-中国2019年-2025年碳排放数据.csv` | 仅作全国总量口径参考，不参与本模块 30 省特征计算 |

## 2. 省份口径

工作表名统一为中文（去 `2022` 后缀映射）：

Shanghai→上海, Yunnan→云南, InnerMongolia→内蒙古, Beijing→北京, Jilin→吉林, Sichuan→四川, Tianjin→天津, Ningxia→宁夏, Anhui→安徽, Shandong→山东, Shanxi→山西, Guangdong→广东, Guangxi→广西, Xinjiang→新疆, Jiangsu→江苏, Jiangxi→江西, Hebei→河北, Henan→河南, Zhejiang→浙江, Hainan→海南, Hubei→湖北, Hunan→湖南, Gansu→甘肃, Fujian→福建, Guizhou→贵州, Liaoning→辽宁, Chongqing→重庆, Shaanxi→陕西, Qinghai→青海, Heilongjiang→黑龙江

合并键：`province`（中文），30 省一一对应，无重复、无缺失。

## 3. 工业口径  C_i^ind

纳入 38 行（采矿业 6 + 制造业 29 + 公用事业 3）：

- 采矿业：Coal Mining and Dressing, Petroleum and Natural Gas Extraction, Ferrous Metals Mining and Dressing, Nonferrous Metals Mining and Dressing, Nonmetal Minerals Mining and Dressing, Other Minerals Mining and Dressing
- 制造业：Food Processing, Food Production, Beverage Production, Tobacco Processing, Textile Industry, Garments and Other Fiber Products, Leather, Furs, Down and Related Products, Timber Processing, Bamboo, Cane, Palm Fiber & Straw Products, Furniture Manufacturing, Papermaking and Paper Products, Printing and Record Medium Reproduction, Cultural, Educational and Sports Articles, Petroleum Processing and Coking, Raw Chemical Materials and Chemical Products, Medical and Pharmaceutical Products, Chemical Fiber, Rubber Products, Plastic Products, Nonmetal Mineral Products, Smelting and Pressing of Ferrous Metals, Smelting and Pressing of Nonferrous Metals, Metal Products, Ordinary Machinery, Equipment for Special Purposes, Transportation Equipment, Electric Equipment and Machinery, Electronic and Telecommunications Equipment, Instruments, Meters, Cultural and Office Machinery, Other Manufacturing Industry
- 公用事业：Production and Supply of Electric Power, Steam and Hot Water; Production and Supply of Gas; Production and Supply of Tap Water

暂不纳入：Farming/Forestry/..., Logging and Transport of Wood and Bamboo, Scrap and waste, Construction（暂不计入，敏感性分析可另设第二产业口径）, Transportation/Storage/Post/Telecom, Wholesale/Retail/Catering, Others, Urban, Rural, TotalEmissions

计算：`C_i^ind = sum(S_ind)`，`Q_i = C_i^ind / C_i`，0—1。

## 4. 主特征表  problem1_province_features_2022.csv

| 字段 | 含义 | 单位 | 来源/公式 |
|------|------|------|-----------|
| province | 省份（中文） | — | 统一映射后 |
| source_sheet | 原始工作表名 | — | 如 Shandong2022 |
| emission_total_mtco2 | 碳排放总量 C_i | Mt CO2 | TotalEmissions 行 Scope_1_Total |
| industrial_emission_mtco2 | 工业碳排放量 | Mt CO2 | 37 行求和 |
| industrial_emission_share | 工业排放占比 Q_i | 0—1 | C_ind / C_total |
| industrial_emission_share_pct | 工业占比百分数 | % | Q_i*100 |
| gdp_2022_100m_yuan | 2022年GDP | 亿元 | 外部 GDP 表 |
| population_2022_10k | 2022年常住人口 | 万人 | 外部人口表 |
| carbon_intensity_tco2_per_10k_yuan | 碳排放强度 I_i | t CO2/万元 | 100*C_Mt / GDP_100M |
| per_capita_emission_tco2_per_person | 人均碳排放 H_i | t CO2/人 | 100*C_Mt / Pop_10k |

> 同一年横截面，GDP/人口直接使用 2022 年值，不做价格平减或外部补充。

## 5. 标准化特征表  problem1_features_standardized_2022.csv

用于 CRITIC-TOPSIS 与 K-means/Ward。

对四特征先取自然对数再做 Min-Max：

```
tilde_x = ln(x),  z = (tilde_x - min(tilde))/(max(tilde)-min(tilde))  ∈[0,1]
```

四列：`z_emission_total`, `z_carbon_intensity`, `z_per_capita_emission`, `z_industrial_emission_share`，均为压力型（越大越高），不做方向反转。后续聚类需检验是否需将两个效率指标合成效率维度以避免权重偏倚（报告 §7.2 提醒）。

## 6. 审计表  industrial_sector_mapping_audit.csv

| 字段 | 说明 |
|------|------|
| emission_total_mtco2 | 总量 C_i |
| industrial_emission_mtco2 | 工业量 |
| nonindustrial_emission_mtco2 | 非工业=总量-工业 |
| industrial_share | Q_i |
| mapped_sector_count | 命中的工业行数（期望 37） |
| unmapped_sector_count | 未预期行数（期望 0） |
| sector_sum_mtco2 | 除 TotalEmissions 外所有行业 Scope_1_Total 之和 |
| total_minus_sector_sum | 总量与行业和的差值（口径/舍入审计） |
| mapping_note | 备注 |

## 7. 外部清洗表  external_province_data_2022_cleaned.csv

由人口与GDP双表合并清洗而来，字段：`province, population_2022_10k, gdp_2022_100m_yuan`，30 行，已做缺失/重复/零负值检查并按省份排序。

## 8. 质量验收（报告 §9）

1. 省份数 30；2. 无重复；3. 每省一条；4. 均有总量/GDP/人口；5. 均为正；6. 工业≤总量；7. 0≤Q≤1；8. 强度/人均为正；9. 无重复计入；10. 审计表可解释；11. 外部与附件2一一合并；12. 无 NaN/Inf/异常字符串。

## 9. 关联中间结果

- descriptive_stats.csv：四特征描述统计（均值/中位/标准差/四分位/CV/极差比等）
- correlation_pearson/spearman.csv：四特征相关系数矩阵（内部验证，不直接入论文正文）
- figures/：四特征分布与相关性可视化（附图例、坐标标签、统一配色）

## 10. 下一步

预处理通过验收后进入：四特征空间差异 Moran 检验 → CRITIC-TOPSIS → K-means 主模型（多初始化/轮廓/CH/DB）→ Ward 对照 → 聚类数与稳定性检验 → 综合得分分级。
