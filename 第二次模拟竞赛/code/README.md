# code/ 使用说明

## 环境
- 解释器：`E:\Anaconda\envs\math_modeling\python.exe`（Python 3.14.7）
- 依赖：`pip install -r requirements.txt`（pandas, numpy, matplotlib, seaborn, openpyxl）

## 运行
```bash
E:\Anaconda\envs\math_modeling\python.exe code/step1_datapreprocess.py
```

## 文件
| 文件 | 说明 |
|------|------|
| `step1_datapreprocess.py` | 问题1数据预处理主程序：总量口径校验、工业38行汇总、四特征构造、ln+Min-Max标准化、描述统计/相关矩阵、12项验收、6张图渲染 |
| `requirements.txt` | 依赖清单 |
| `README.md` | 本文件 |

## 输入
- `附件/附件2-2022年30个省份排放清单.xlsx`（30省 + NOTE）
- `附件/人口与GDP数据.csv`（实为 XLSX，双表：人口/GDP，各30行）

## 输出
- `results/problem1_preprocessed/`：8×CSV + 1×MD + 1×TXT（见数据字典）
- `figures/problem1_*.png`：6张 300dpi 图
- `reports/问题1_预处理结果分析报告.md`：量化分析与异常诊断

## 复现要点
- 省份映射：工作表名去`2022`后按 EN_TO_CN 字典转中文，合并键为 `province`
- 总量取 `TotalEmissions` 行 `Scope_1_Total` 列（Mt），不重加能源列
- 工业汇总 38 行，Construction 暂不纳入；已过滤 Jilin 参考文献行
- 强度 `100*C_Mt/GDP_100M`（t/万元），人均 `100*C_Mt/Pop_10k`（t/人）
- 标准化：ln → Min-Max 到 [0,1]，压力型不反向
