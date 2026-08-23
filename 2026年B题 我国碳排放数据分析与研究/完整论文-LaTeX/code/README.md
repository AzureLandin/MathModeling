# 代码说明

所有可执行代码均按“负责问题 + 功能（英文）”命名：

| 文件 | 职责 |
|---|---|
| `All_common_utilities.py` | 通用绘图导出、颜色配置、哈希工具 |
| `All_data_processing.py` | 附件读取、清洗、2025年年化与省级指标构造 |
| `Q1_spatial_clustering.py` | 问题一：Moran置换检验、PCA与Ward聚类 |
| `Q2_constrained_ridge.py` | 问题二：符号约束STIRPAT--ridge与回测 |
| `Q3_scenario_forecast.py` | 问题三：三情景路径递推与排放预测 |
| `Q4_policy_mapping.py` | 问题四：政策工具映射、部门排序与阶段指标 |
| `All_figure_generation.py` | 全部问题的结果图生成 |
| `All_model_runner.py` | 总运行器：按Q1--Q4顺序生成结果、图形和清单 |
| `All_document_builder.py` | 论文文档构建辅助脚本 |
| `All_document_appendix.py` | 论文附录编辑辅助脚本 |
| `All_latex_writer.py` | LaTeX 初稿写入辅助脚本 |

运行总模型：

```powershell
& 'E:\Anaconda\envs\math_modeling\python.exe' `
  'E:\MathModeling\2026年B题 我国碳排放数据分析与研究\完整论文-LaTeX\code\All_model_runner.py' `
  --project-root 'E:\MathModeling\2026年B题 我国碳排放数据分析与研究' `
  --input-root 'E:\MathModeling\第二次模拟竞赛\附件' `
  --seed 20260822
```
