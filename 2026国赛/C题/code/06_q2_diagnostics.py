"""Explain saved Q2 baseline failures without changing or refitting policies."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/q2_baseline'


def main():
    assert Path(sys.prefix).name=='math_modeling'
    comparison=pd.read_csv(OUT/'comparison.csv').set_index('policy')
    actual=pd.read_csv(OUT/'lag/dispatch_all_2025.csv')
    actual=actual[actual.date>='2025-02-01']
    emergency=actual[actual.emergency_kWh>1e-6]
    gap=emergency.emergency_cost_yuan.sum()
    result=dict(
        evaluation_days=334,
        mean_mae_improvement_pct=float(100*(1-comparison.loc['mean','net_mae_kW']/comparison.loc['lag','net_mae_kW'])),
        mean_total_cost_increase_pct=float(100*(comparison.loc['mean','total_cost_yuan']/comparison.loc['lag','total_cost_yuan']-1)),
        emergency_slots=len(emergency),
        emergency_slots_at_lower_state_pct=float(100*(emergency.state_end_kWh<=1200+1e-6).mean()),
        emergency_slots_at_discharge_limit_pct=float(100*(emergency.discharge_kWh>=5000/6-1e-6).mean()),
        emergency_cost_20_to_21_pct=float(100*emergency.loc[emergency.slot.between(120,125),'emergency_cost_yuan'].sum()/gap),
        emergency_cost_19_to_21_pct=float(100*emergency.loc[emergency.slot.between(114,125),'emergency_cost_yuan'].sum()/gap),
        caveat='State/power percentages are counts of emergency slots, not shares of emergency energy or cost.')
    # Independent accounting from raw dispatch, including stored-energy change.
    residual=(actual.planned_kWh+actual.pv_kW/6+actual.discharge_kWh+actual.emergency_kWh
              -actual.load_kW/6-actual.charge_kWh-actual.unused_kWh)
    assert residual.abs().max()<1e-6
    losses=.1*actual.charge_kWh.sum()+(1/.9-1)*actual.discharge_kWh.sum()
    global_balance=actual.planned_kWh.sum()+actual.pv_kW.sum()/6+actual.emergency_kWh.sum()-actual.load_kW.sum()/6-actual.unused_kWh.sum()-losses-(actual.state_end_kWh.iloc[-1]-actual.state_start_kWh.iloc[0])
    assert abs(global_balance)<1e-5
    result['independent_global_energy_residual_kWh']=float(global_balance)
    monthly=pd.read_csv(OUT/'lag/monthly_summary.csv')
    monthly['mean_minus_lag_cost_yuan']=pd.read_csv(OUT/'mean/monthly_summary.csv').total_cost_yuan-monthly.total_cost_yuan
    monthly[['month','mean_minus_lag_cost_yuan']].to_csv(OUT/'monthly_candidate_comparison.csv',index=False,encoding='utf-8-sig')
    result['months_mean_cheaper']=int((monthly.mean_minus_lag_cost_yuan<0).sum())
    result['months_mean_dearer']=int((monthly.mean_minus_lag_cost_yuan>0).sum())
    # Generate exactly the paper's requested selected 10-minute and 4-hour tables.
    selected_dates=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
    selected=actual[actual.date.isin(selected_dates)].copy()
    selected[selected.slot.isin([60,72,84,96,108,120])][['date','interval','planned_kWh','emergency_kWh']].to_csv(OUT/'table1_selected_intervals.csv',index=False,encoding='utf-8-sig')
    selected['block']=selected.slot//24
    blocks=selected.groupby(['date','block'])[['charge_kWh','discharge_kWh']].sum().reset_index()
    blocks['interval']=[f'{b*4:02d}:00-{(b+1)*4:02d}:00' for b in blocks.block]
    blocks[['date','interval','charge_kWh','discharge_kWh']].to_csv(OUT/'table2_selected_storage.csv',index=False,encoding='utf-8-sig')
    (OUT/'diagnostic_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第二问首轮诊断与下一轮重点','',
           '以下结论来自已保存的四组固定策略回测，不是新模型的收益声明。评价期为2025年2—12月；完整费用见Q2_Baseline.md。','',
           '## 已验证发现','',
           f"1. 均值方案净负荷MAE比滞后Baseline下降{result['mean_mae_improvement_pct']:.4f}%，但实际总費增加{result['mean_total_cost_increase_pct']:.4f}%。它在{result['months_mean_cheaper']}个月更便宜、{result['months_mean_dearer']}个月更贵，不能认为平滑历史数据普遍提高经济性。",
           f"2. 主Baseline应急费3112931.90元，占总费20.2423%；20:00—21:00占应急费{result['emergency_cost_20_to_21_pct']:.4f}%，19:00—21:00合计占{result['emergency_cost_19_to_21_pct']:.4f}%。",
           f"3. 主Baseline共有{result['emergency_slots']}个10分钟应急时段，其中{result['emergency_slots_at_lower_state_pct']:.4f}%时段末储电量触及1200 kWh下限，{result['emergency_slots_at_discharge_limit_pct']:.4f}%时段放电达到5000 kW限制。这两个比例是时段数比例，不是电量或费用占比。",
           '4. 电池实际日初电量呈持续漂移，候选预测末态6000虽降低均值方案费用，仍未优于滞后Baseline。预测终端约定和执行控制会影响结果，不能只替换预测器。',
           f"5. 独立全期能量核算加入储能损耗和期末电量变化后残差{global_balance:.3e} kWh；这排除了漏记储能损耗或跨日电量重置造成的虚假平衡。",'',
           '## 下一轮建议','',
           '先研究针对高应急成本的计划余量：用当时已有的真实预测残差估计净负荷偏差，增加可校准的日前保护量。所有窗口和保护强度仅在过去日期滚动选择，不能按已看到的全年最优结果回填。与现有主Baseline保持相同控制器和终端约定，先隔离计划余量的作用。', '',
           '随后单独比较晚高峰储能备用或有限时域控制，再研究更长规划窗口和终端价值。当前触底统计说明存量不足常与应急同时发生，但不能单凭该统计判断是买电不足、过早放电、容量不足还是充电机会不足；增加备用也可能把应急提前，必须比较全日总费。', '',
           '暂不需要直接引入深度学习或大规模随机优化。已有结果支持先让5倍应急成本进入决策，而不是仅减少平均预测误差。', '',
           '## 尚未解决','',
           '- 第一版只对点预测场景最优，未对随机需求的期望总费用最优。',
           '- 首日冷启动及评价初末电量差异尚需敏感性/匹配边界对照。',
           '- 下一轮收益必须做因果滚动评价；全年诊断之后不得将同一数据称为从未参与设计的独立测试集。',
           '- 附件5/result2.xlsx仍为原模板；results/q2_baseline/result2_baseline.xlsx仅为首轮计算副本。', '',
           '复现本诊断：conda run -n math_modeling python code/06_q2_diagnostics.py。该脚本不求解或修改计划，读取05生成结果并独立核账。','']
    (ROOT/'reports/第二问首轮诊断.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
