# Code 文件说明

## 正式入口

| 文件 | 职责 |
|---|---|
| `q1_milp_model.py` | 问题一新时间口径确定性 MILP 总入口 |
| `q2_lightgbm_dispatch_model.py` | 问题二 LightGBM 预测、q80 保护与日前调度总入口 |
| `q3_rolling_dispatch_model.py` | 问题三 F2 日内修正、q75 保护与滚动调度总入口 |
| `q4_price_dispatch_model.py` | 问题四 Q42、Q43_S2 波动电价调度总入口 |
| `q1_q4_csv_processing.py` | 四问统一字段 CSV 导出与结果清单生成 |

## 关联内核

其余带编号或 `q*_...experiment.py`、`q*_...diagnostic.py`、`q*_...report.py` 的脚本是正式入口在 `--recompute` 模式下调用的模型、诊断和报告内核。它们全部随本包复制，文件名保留原名是为了兼容源码中的动态导入；不需要手工逐个运行。

从提交包根目录运行：

```powershell
python .\code\q1_q4_csv_processing.py
```

默认只读取包内 `results/` 的冻结账本，将正式 CSV 写入 `result/`，不重新训练或求解。四个模型入口默认也只导出对应问题的冻结结果。只有显式添加 `--recompute` 才会调用包内完整计算内核；重算可能耗时较长，并需要 `numpy`、`pandas`、`scipy`、`openpyxl`、`matplotlib`、`lightgbm` 等 Python 依赖。
