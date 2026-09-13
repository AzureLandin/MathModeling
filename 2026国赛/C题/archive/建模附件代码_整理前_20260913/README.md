# 建模附件代码汇总

> 正式提交时优先使用同级目录 `建模附件代码_正式提交/`：该目录只保留四个问题模型总入口和一个公共 CSV 数据处理文件。当前目录保留完整模型内核、冻结兼容依赖和历史审计脚本，用于追溯与重新计算。

## 收尾交付版（正式文件名与 CSV 导出）

本目录新增的正式入口按问题编号命名，适合作为建模附件代码入口：

| 文件 | 用途 | 默认输出 |
|---|---|---|
| `正式入口_问题一.py` | 问题一新时间口径正式调度账本导出 | `正式CSV结果/问题一_正式调度结果.csv` |
| `正式入口_问题二.py` | 问题二 N_free 自然账本和模板计划导出 | `正式CSV结果/问题二_自然日调度结果.csv`、`问题二_模板购电计划.csv` |
| `正式入口_问题三.py` | 问题三 S2 正式滚动调度账本导出 | `正式CSV结果/问题三_正式滚动调度结果.csv` |
| `正式入口_问题四.py` | 问题四 Q42、Q43_S2 正式调度账本导出 | `正式CSV结果/问题四_4-2正式调度结果.csv`、`问题四_4-3正式调度结果.csv` |
| `正式结果CSV导出.py` | 四问统一导出器，可用 `--only q1/q2/q3/q4/all` 选择范围 | 六份正式 CSV、汇总表和导出清单 |

一键导出（只读已核验结果，不重新训练或求解）：

```powershell
python .\建模附件代码\正式结果CSV导出.py
```

导出目录为 `C题/正式CSV结果/`，包含：

- 六份按问题和方案正式命名的 CSV；
- `四问_正式结果汇总.csv`：四问正式评价期的行数和能量/费用汇总；
- `四问_正式结果导出清单.json`：源文件、源 SHA-256、导出 SHA-256、字段清单和行数记录。

统一 CSV 字段采用中文正式名称，数值列保留原账本精度，编码为 `UTF-8-SIG`，可直接用 Excel 打开。问题二至四的公共初始化行仍保留在明细 CSV 中，但 `评价阶段` 标记为 `公共初始化`，汇总只统计 `正式评价期`；问题一标记为 `正式结果`。

正式入口是冻结账本的交付层，不替代下方的模型求解内核。若需要重新计算，必须按“最终主线”表中的原始入口和动态依赖运行，并先在副本上验证。

本目录依据以下当前交付报告整理：

- `reports/总整理/全局索引与一致性检查.md`
- `reports/总整理/问题一二整理报告.md`
- `reports/总整理/问题三四整理报告.md`
- `reports/问题一/问题一_新时间口径正式交付结果.md`
- `reports/问题二/核心文件说明.md`
- `reports/问题三/问题三_最终交付说明.md`
- `reports/问题四/问题四_最终收敛与交付方案.md`

代码文件直接放在本目录根部，而不是再放入 `code/` 子目录。这样保留了原脚本中
`Path(__file__).resolve().parents[1]` 对项目根目录 `C题` 的解析方式。运行时仍应以
`C题` 为项目根目录，并保留原有的 `附件/`、`results/`、`reports/` 和 `figures/`
目录；本目录不是独立脱离数据资产的压缩包。

## 最终主线

| 问题 | 最终方案 | 主计算入口 | 正式填表入口 | 只读报告/图表入口 |
|---|---|---|---|---|
| 一 | 新时间口径 MILP | `q1_start_time_recompute.py` | `q1_finalize_new_time_delivery.py` | 无单独报告渲染入口 |
| 二 | N_free、LightGBM、W28/q80、自由末态 | `29_q2_time_mapping_experiment.py` | `fill_result2_frozen.py` | `30_q2_report_renderer.py` |
| 三 | S2：F2 日内负载修正、Linear 光伏、W28/q75、四节点滚动 | `q3_intraday_load_cost_experiment.py` | `fill_result3_s2.py` | `q3_intraday_load_cost_report.py` |
| 四-2 | Q42：问题二配置迁移到附件4波动电价 | `q4_price_transfer_experiment.py` | `q4_fill_result_tables.py` | `q4_price_transfer_report.py`、`q4_fill_result_tables_report.py` |
| 四-3 | Q43_S2：问题三 S2 配置迁移到附件4波动电价 | `q4_price_transfer_experiment.py` | `q4_fill_result_tables.py` | `q4_price_transfer_report.py`、`q4_fill_result_tables_report.py` |

## 动态依赖

- 问题一的正式重算入口会加载 `02_q1_baseline.py`。
- 问题二主入口会动态加载 `14_q2_ridge_forecast_experiment.py`、
  `19_q2_lightgbm_residual_experiment.py` 和 `21_q2_quantile_level_scan.py`；其冻结兼容
  内核还涉及 `05_q2_baseline.py` 和 `08_q2_quantile_experiment.py`。
- 问题三最终 S2 入口加载 `q3_rolling_baseline_experiment.py`，后者再复用问题二主内核。
- 问题四迁移入口复用 `q3_rolling_baseline_experiment.py`，并读取问题二/三已冻结的
  `results/` 结果档案。

因此，上表列出的基础内核也一并保留，不能只提交每一问的单个主脚本。

## 推荐阅读/复现顺序

1. 先阅读四问最终交付报告，确认当前方案和时间口径。
2. 先检查 `附件/附件1.xlsx` 至 `附件/附件4.xlsx`、附件5模板以及对应的 `results/`
   冻结档案。
3. 按问题一、问题二、问题三、问题四的主计算入口理解依赖关系。
4. 最后使用各问正式填表入口生成或核对 `附件/附件5/` 中的结果表；问题四正式核查
   副本位于 `results/q4_delivery/`，不覆盖附件5原始模板。

原脚本中的绝对路径、运行环境和 `math_modeling` 环境约定保持不变。正式交付前应在
副本上运行并检查输出，不要直接覆盖已核验的结果表。

## 未纳入

未复制 `archive/`、`outputs/`、`node_modules/`、历史参数扫描、维护脚本和问题四旧的
`q4_known_price_experiment.py`。这些属于追溯或诊断资产，不是当前正式答案的复现主线。

`03_q1_milp.py` 因论文旧附录曾明确列出而保留，但它不是当前新时间口径正式结果的主入口；
当前问题一应以 `q1_start_time_recompute.py` 和 `q1_finalize_new_time_delivery.py` 为准。

当前 `论文初稿/论文claude/论文claude.tex` 的附录和问题四正文仍有旧/未完成表述，不能
覆盖上述最终交付报告的方案判断。
