# 建模附件代码：正式提交版

本目录只保留正式交付所需的 `q*.py` 文件，按“一个问题一个模型总入口、一个公共数据处理器”的方式组织。

| 文件 | 对应模型/数据处理 | 输出 |
|---|---|---|
| `q1_milp_model.py` | 问题一：新时间口径确定性 MILP | `正式CSV结果/问题一_正式调度结果.csv` |
| `q2_lightgbm_dispatch_model.py` | 问题二：LightGBM 预测、q80 保护与日前调度 | `问题二_自然日调度结果.csv`、`问题二_模板购电计划.csv` |
| `q3_rolling_dispatch_model.py` | 问题三：F2 日内修正、q75 保护与滚动调度 | `问题三_正式滚动调度结果.csv` |
| `q4_price_dispatch_model.py` | 问题四：Q42、Q43_S2 波动电价调度 | `问题四_4-2正式调度结果.csv`、`问题四_4-3正式调度结果.csv` |
| `q1_q4_csv_processing.py` | 公共数据处理：统一字段、编码、汇总和 SHA-256 清单 | 汇总 CSV、JSON 清单 |

运行位置应为 `C题` 目录。四个问题入口只读取已经核验的冻结账本，不重新训练、不重新求解全年 MILP，也不修改 Excel 原件：

```powershell
python .\建模附件代码\q1_milp_model.py
python .\建模附件代码\q2_lightgbm_dispatch_model.py
python .\建模附件代码\q3_rolling_dispatch_model.py
python .\建模附件代码\q4_price_dispatch_model.py
python .\建模附件代码\q1_q4_csv_processing.py
```

默认命令只导出冻结结果。需要重新计算时，在对应模型命令末尾增加 `--recompute`；问题二至四会重新训练或求解全年模型，耗时明显更长。

完整模型内核保留在项目 `code/`，历史整理版保留在 `archive/建模附件代码_整理前_20260913/`，供结果追溯。比赛附件只需使用本目录和 `正式CSV结果/`。
