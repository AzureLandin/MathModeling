# Code 文件说明

| 文件 | 职责 |
|---|---|
| `q1_milp_model.py` | 问题一新时间口径确定性 MILP 总入口 |
| `q2_lightgbm_dispatch_model.py` | 问题二 LightGBM 预测、q80 保护与日前调度总入口 |
| `q3_rolling_dispatch_model.py` | 问题三 F2 日内修正、q75 保护与滚动调度总入口 |
| `q4_price_dispatch_model.py` | 问题四 Q42、Q43_S2 波动电价调度总入口 |
| `q1_q4_csv_processing.py` | 四问统一字段 CSV 导出与结果清单生成 |

从提交包根目录运行：

```powershell
python .\code\q1_q4_csv_processing.py
```

该命令在当前项目中读取已核验账本，并将正式 CSV 写入本包 `result/`，不重新训练或求解。提交包单独拷走后，命令会改为核验包内现成 CSV。四个模型入口默认也只处理对应问题结果。只有显式添加 `--recompute` 才调用原项目完整计算内核；该参数依赖原项目 `code/` 和冻结账本，收尾提交阶段不建议运行。
