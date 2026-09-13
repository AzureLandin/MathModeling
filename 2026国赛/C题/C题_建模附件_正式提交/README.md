# C题建模附件正式提交包

本目录是可迁移运行包。将整个 `C题_建模附件_正式提交` 文件夹复制到另一台设备后，包内的附件、代码、冻结结果、报告和重算图均保持相对路径，不再依赖原项目父目录。

| 目录 | 内容 |
|---|---|
| `附件/` | 附件1至附件4原始输入，以及 `附件5/` 中用于重算的正式模板副本 |
| `code/` | 5 个正式入口、统一 CSV 导出器和 57 个关联计算内核；内核文件名保留原名以兼容动态调用 |
| `result/` | 五份正式结果 Excel、统一字段 CSV、汇总 CSV 与导出清单 |
| `results/` | 四问重算所需的冻结中间结果和审计账本，不是临时缓存 |
| `reports/` | 重算脚本引用的任务书、结果说明和口径文档 |
| `figure/` | 按 q1 至 q4 分类的论文主图和求解流程图 |
| `figures/` | 重算内核使用的图形输出目录 |
| `requirements.txt` | 可迁移环境的 Python 第三方依赖清单 |

从包根目录运行：

```powershell
python .\code\q1_q4_csv_processing.py
```

该命令只读取包内 `results/` 的冻结账本并重新导出正式 CSV，不训练、不求解、不覆盖 Excel。四个正式入口分别是 `q1_milp_model.py`、`q2_lightgbm_dispatch_model.py`、`q3_rolling_dispatch_model.py` 和 `q4_price_dispatch_model.py`；默认执行同样的冻结结果导出。显式添加 `--recompute` 才会调用包内完整计算内核，可能需要较长时间和本机 Python 科学计算环境。

问题四的 `result4-2.xlsx`、`result4-3.xlsx` 来自已通过回读核查的正式填报副本，分别对应 Q42 和 Q43_S2，不是空白模板。正式 CSV 的评价区间为 2025-02-01 至 2025-12-31；问题一为附件1确定性单日。完整字段、行数、来源哈希和导出哈希见 `result/四问_正式结果导出清单.json`。在新设备上可先执行 `python -m pip install -r requirements.txt`，再运行入口脚本。
