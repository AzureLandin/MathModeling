#!/usr/bin/env python
"""问题二 30 号：时间映射实验报告渲染层（只读 results/q2_time_mapping/）。

分工纪律（本轮起生效）：29 号只跑实验、写数据；30 号只读数据、渲染报告。
编辑本文件永远不会使 29 号的登记签名失效，也不需要重跑任何求解。

产物：reports/问题二_统一新时间映射实验结果报告.md
数据源：results/q2_time_mapping/{summary,contrast,...}.csv、checks.json、
        run_manifest.json、registration.json、figure_integrity.json、tail_interval.csv
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_time_mapping'
REPORT_MD = ROOT / 'reports/问题二_统一新时间映射实验结果报告.md'

assert Path(sys.prefix).name == 'math_modeling', sys.prefix


def load_json(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def money(value):
    return f'{float(value):,.2f}'


def pct(value):
    return f'{float(value):+.4f}'


def main():
    manifest = load_json('run_manifest.json')
    registration = load_json('registration.json')
    checks = load_json('checks.json')
    integrity = load_json('figure_integrity.json')
    tail = pd.read_csv(OUT / 'tail_interval.csv')
    summary = pd.read_csv(OUT / 'summary.csv')
    contrast = pd.read_csv(OUT / 'contrast.csv').iloc[0]
    monthly = pd.read_csv(OUT / 'monthly_contrasts.csv')
    window = pd.read_csv(OUT / 'window_emergency.csv')
    stability = pd.read_csv(OUT / 'stability.csv').iloc[0]
    inventory = pd.read_csv(OUT / 'inventory.csv')
    energy = pd.read_csv(OUT / 'energy_contrasts.csv').iloc[0]
    quantile = load_json('quantile_validation.json')
    mapping = load_json('mapping_checks.json')
    replay = load_json('internal_reproduction.json')
    perturbation = load_json('perturbation_checks.json')
    carry = load_json('carry_check.json')
    halfday = load_json('halfday_check.json')
    forecast_summary = load_json('forecast_summary.json')
    archive_dependency = registration['dependency']
    reference = checks['ledger']
    parameters = registration['parameters']
    dates = pd.date_range('2025-01-01', periods=366)
    terminal_index = 4 * 144 + 143

    def jround(obj, digits=6):
        return json.loads(json.dumps(obj), parse_float=lambda s: round(float(s), digits))

    summary_rows = '\n'.join(
        f'| {row.strategy_id} | {row.mode} | {money(row.planned_cost_yuan)} '
        f'| {money(row.emergency_cost_yuan)} | {money(row.total_cost_yuan)} | '
        f'{int(row.emergency_days)} | {row.emergency_kWh:,.2f} | {row.unused_kWh:,.1f} '
        f'| {row.loss_kWh:,.1f} | {row.final_kWh:,.1f} | {money(row.B_total_cost_yuan)} |'
        for row in summary.itertuples())
    month_rows = '\n'.join(
        f'| {row.month} | {money(row.fixed_total_cost_yuan)} | {money(row.free_total_cost_yuan)} '
        f'| {money(row.free_minus_fixed_yuan)} | {row.delta_emergency_kWh:,.2f} |'
        for row in monthly.itertuples())
    window_rows = '\n'.join(
        f'| {row.scope} | {row.window} | {money(row.emergency_cost_yuan)} | '
        f'{row.emergency_kWh:,.2f} | {money(row.planned_cost_yuan)} |'
        for row in window.itertuples())
    inventory_rows = '\n'.join(
        f'| {row.strategy_id} | {row.nominal_S143_mean_kWh:,.2f} | '
        f'{int(row.nominal_S143_days_at_6000)} | {int(row.nominal_S143_days_at_floor)} '
        f'| {row.actual_initial_kWh:,.2f} | {row.actual_final_kWh:,.2f} '
        f'| {money(row.inventory_adjustment_yuan)} | {money(row.adjusted_cost_yuan)} |'
        for row in inventory.itertuples())
    integrity_rows = '\n'.join(
        f"- `figures/q2_time_mapping/{row['file']}`：{row['width']}×{row['height']} 像素，"
        f"非白像素比例 {row['non_white_fraction']:.4f}，颜色数 {row['distinct_colours']}。"
        for row in integrity)
    boundary_rows = '\n'.join(
        f"| {row['check']} | {str(row['expected'])[:90]} | {str(row['value'])[:90]} | "
        f"{'通过' if row['passed'] else '未通过'} |" for row in checks['boundary']['rounds'])
    forecast_rows = '\n'.join(
        f"| {row['check']} | {str(row['expected'])[:70]} | {str(row['value'])[:70]} | "
        f"{'通过' if row['passed'] else '未通过'} |" for row in checks['forecast']['rounds'])
    mapping_rows = '\n'.join(
        f"| {row['check']} | {row['worst']:.3e} | {row['tolerance']:.0e} | "
        f"{'通过' if row['passed'] else '未通过'} |" for row in mapping['rounds'])
    perturbation_rows = '\n'.join(
        f"| {case['date']} | {case['case']} "
        f"| {case['groups']['N_fixed6000']['prefix_plan_difference_kWh']:.1e} "
        f"| {case['groups']['N_free']['prefix_plan_difference_kWh']:.1e} "
        f"| {case['groups']['N_fixed6000']['perturbation_day_plan_difference_kWh']:.1e} "
        f"| {case['groups']['N_free']['perturbation_day_plan_difference_kWh']:.1e} "
        f"| {case['groups']['N_fixed6000']['perturbation_day_actual_emergency_kWh']:,.2f} |"
        for case in perturbation['cases'])
    tail_rows = '\n'.join(
        f"| {row.strategy_id} | {row.date} | {row.interval} | {money(row.planned_kWh)} | "
        f"{money(row.planned_cost_yuan)} |" for row in tail.itertuples())
    a_fixed = reference['N_fixed6000']['ledger_A']
    a_free = reference['N_free']['ledger_A']

    self_checks = manifest['self_checks_passed']
    assert all(self_checks.values()), self_checks
    assert manifest['protected_unchanged'], 'protected assets changed'

    report = f"""# 问题二：统一新时间映射（start_time_v1）实验结果报告

日期：{datetime.now().date().isoformat()}。状态：**实验自检完成，独立审计待完成**。
本报告由 `code/30_q2_report_renderer.py` 只读渲染 `results/q2_time_mapping/` 的实验产物生成；
实验层（`code/29_q2_time_mapping_experiment.py`，登记签名 {manifest['signature'][:16]}）
与报告层分离，报告文字不参与实验登记签名。未锁定第二问最终模型，未填写正式
`result2.xlsx`，未修改第一问。

## 1. 问题分析

### 1.1 口径与两组

用户确认的区间起点口径是**建模解释，不是题面原文**：源标签 `00:10` 代表 00:10—00:20，
源标签 `0:00+1` 代表**次日 00:00—00:10**。因此一个源行覆盖 `[k日00:10,(k+1)日00:10)`，
是"模板日"；自然日 d 由**前一源行末值接当前源行前 143 值**组成，不能逐行移位或逐日循环。

本轮的显式衔接承诺（属建模假设，必须随结论披露）：

- k 日 0:00 发布覆盖 `[k日00:10,(k+1)日00:10)` 的 144 段模板计划；另有 1 个辅助午夜目标；
- 当前 `[k日00:00,00:10)` 的普通购电已由前一日模板末段承诺，当天不能撤销或重复承诺；
- 0:00 实际 SOC 可测、当前首段真值尚不可用：用**辅助因果预测**估算首段执行后的状态再规划，
  计划仍于 0:00 冻结，不等待 00:10 真值重解；
- **固定 6000 约束落在真正的自然日 24:00（S143）**；模板末端 S144 两组都只受物理界限制；
- 自然日账本与模板账本分开，报告首尾差额。

两组唯一差别是名义末态：`N_fixed6000`（S143=6000 kWh）与 `N_free`（仅 1200—10800 kWh）。
预测（在新信息集下重建的 15 列负载 / 12 列光伏 LightGBM）、W28/q80、公共 1 月、实际反馈、
收费规则与求解器设置完全相同。**旧 27 号的费用、2 月初态与 1 元豁免一律未复用。**

### 1.2 主要结论（账本 A：自然日 2025-02-01—12-31）

| 组 | 末态 | 计划费(元) | 应急费(元) | 现金总费(元) | 应急天数 | 应急电量(kWh) | 未使用(kWh) | 损耗(kWh) | 期末(kWh) | 账本B总费(元) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{summary_rows}

- 主比较 ΔC（自由 − 固定，账本 A）= {money(contrast.A_delta_total_cost_yuan)} 元
  （{pct(contrast.A_delta_total_percent)}%）：计划购电费
  {money(contrast.A_delta_planned_cost_yuan)} 元、应急费
  {money(contrast.A_delta_emergency_cost_yuan)} 元、应急电量
  {contrast.A_delta_emergency_kWh:+,.2f} kWh、应急天数
  {int(contrast.A_delta_emergency_days):+d} 天、损耗 {contrast.A_delta_loss_kWh:+,.1f} kWh、
  期末 {contrast.A_delta_final_kWh:+,.1f} kWh。
- 模板账本 B 的差额为 {money(contrast.B_delta_total_cost_yuan)} 元，
  {"与账本 A 同向" if contrast.B_delta_total_cost_yuan * contrast.A_delta_total_cost_yuan > 0 else "与账本 A 反向"}。
- 逐日 {int(stability.days_improved)} 天更省 / {int(stability.days_worse)} 天更贵；
  逐月 {int(stability.months_improved)}/{int(stability.months_total)} 个月更省。
  最贵单日 {stability.worst_day}（{money(stability.worst_day_yuan)} 元）、
  最省单日 {stability.best_day}（{money(stability.best_day_yuan)} 元）。
- 该结果只在当前 15/12 列预测、q80、公共 1 月与单年数据下成立，**不证明自由末态更优**，
  也不能把新旧口径费用差归因成单纯的 10 分钟移位收益。

## 2. 数据预处理

### 2.1 时间、信息与承诺闭环

只使用附件1 的 144 点电价与附件2 的 365×144 实际功率；不使用附件3/4、未来实测或外部天气。
保留 52560 个原功率值，不平滑、不删点、不插值，只按真实时间戳重新给出两个坐标。

映射与可逆性（`mapping_checks.json`、`artificial_mapping.json`）：

| 检查 | 最大误差 | 阈值 | 结果 |
|---|---:|---:|---|
{mapping_rows}

- 源区间 {mapping['source_interval_count']} 个，覆盖 {mapping['first_interval']} 至
  {mapping['last_interval']}，相邻间隔严格 10 分钟且无重叠无漏段；
- 三处手工独立复算（1月2日、4月11日、12月31日）与主实现逐位一致；
- 2025-01-01 00:00—00:10 缺源观测，保留为**未知**（`{mapping['gap_slot_is_nan']}`），
  不填 0 伪装真值、不进入训练/残差/正式费用；该缺口电池静置，E(00:10)=6000 kWh。

### 2.2 公共 1 月

按方案 2.2 节重建，**不复制旧 2 月初态 8801.462273333342**：

- 1月1日：模板 144 段普通计划置 0，电池静置，其余 143 个有观测区间按真值记应急/未使用；
- 1月2日：模板继续置 0、无有效残差；从 00:00 起恢复实际贪心反馈，首段承诺 qcarry=0；
- 1月3—31日：公共物理预运行，用本方案的朴素预测、r=0、固定名义 S143=6000 与午夜承诺规则；
- 同期另建 LightGBM 因果预测档案，只供训练与校准，**不回写公共 1 月动作**。

2月1日两组继承同一重建初态与 1月31日承诺的午夜 qcarry，之后各自独立状态与承诺。

### 2.3 两个账本

账本 A = 自然日 `[2025-02-01 00:00, 2026-01-01 00:00)`，334 天 48096 段；
账本 B = 模板发布 `[2025-02-01 00:10, 2026-01-01 00:10)`，48096 段但**不是同一集合**。
桥接恒等式 C_B − C_A = cost(h_e) − cost(h_s)：h_s 为 2月1日 00:00—00:10（由 1月31日承诺，
计入 A），h_e 为 2026-01-01 00:00—00:10（由 12月31日发布，只计入 B）。实测桥接残差
{reference['N_fixed6000']['bridge_residual_yuan']:.3e} 元（固定组）与
{reference['N_free']['bridge_residual_yuan']:.3e} 元（自由组）；
尾段（2026-01-01 00:00—00:10）：

| 组 | 日期 | 区间 | 计划购电(kWh) | 计划费(元) |
|---|---|---|---:|---:|
{tail_rows}

## 3. 模型建立

### 3.1 预测目标与信息集

k 日 0:00 发布 145 个目标：h=0 当前 00:00—00:10（辅助）、h=1..143 当前 00:10—23:50、
h=144 次日 00:00—00:10；模板 slot j=h-1。历史按**完整自然日**索引，最早完整日为 1月2日，
发布日 k 的最近完整日为 b=k-1。所有特征只用发布时已知历史。

负载 15 列：b^L、L[b,t]−b^L、L[b,t]−L[b−7,t]、最近至多 4 个已完整可用同星期日同段均值−b^L、
mean(L[b])−mean(L[b−7])、目标日期星期 6 列（周一为参照）与 4 列谐波。
b^L 取目标日期之前最近一个**已完整可用**的同星期自然日，无则回退 b 日同段。

光伏 12 列：b^V=V[b,t]、V[b,t]−V[b−1,t]、最近 7 个完整日同段均值−b^V、
mean(V[b])−mean(V[b−1])、4 列谐波，追加支持窗口的 D/M/τ/v。
窗口由 b−6..b 这 7 个完整自然日的 V>1 kW 钟点确定，最早/最晚各扩 3 段并裁至 [0,143]；
不足 7 日或集合空则 s=0,e=143,v=0（全天不归零）。

**次日午夜不能读取尚未知的 V[k,0]**，h=144 的朴素基准按约定回退到 V[k−1,0]。

训练：每个目标每天一个 LightGBM，只用**历史模板 144 目标**的残差训练；训练窗口为此前
{parameters['train_window_days']} 个发布日中特征有效且 144 个目标标签全部可用的完整发布日，
至少 {parameters['min_train_days']} 日才拟合。前一发布日的 h=144 目标要到当前 00:10 才结束，
因此最新可用训练日通常为 k−2，窗口不向前延伸凑数。

实测首训：负载 {dates[forecast_summary['load_first_trainable_day']].date()}
（日序号 {forecast_summary['load_first_trainable_day']}）、
光伏 {dates[forecast_summary['pv_first_trainable_day']].date()}
（日序号 {forecast_summary['pv_first_trainable_day']}），与方案预估（1月10日/1月9日首个有效
特征日、1月25日/1月24日首训）一致；登记 19 号 `LGBM_PARAMS` 全字段不变
（{len(archive_dependency['lgbm_param_keys'])} 个字段，哈希
{archive_dependency['lgbm_params_sha256'][:16]}）。

### 3.2 q80 与两种样本数

误差 eps[j,h] = (L − V − L_hat + V_hat)/6。k 日取 [k−28,k−1] 中该目标预测有效且
interval_end ≤ k日00:00 的误差，按 h 分别计 m；m<7 则 r=0，否则升序取 1 基整数位
`(80m+99)//100`（索引减 1），不插值、保留负修正与负净需求。

实测：h=144（前一日末段尚不可用）m={quantile['h144_m_values']}、位置
{quantile['position_27']}；其余全部目标 m={quantile['other_m_values']}、位置
{quantile['position_28']}。两组 145 目标预测、残差、保护需求与 m 完全一致，
状态与末态模式不进入预测或校准。

### 3.3 首段估算与正确终端索引

0:00 对辅助保护需求用同一个反馈 F 做**一次名义递推**，得名义 E1（00:10）与名义
c0,d0,e0,w0，不使用当前首段真值；首段名义应急可以非零。随后优化模板 144 段（j=0..143），
S0=名义 E1，约束 1200 ≤ S_j ≤ 10800（j=0..144），固定组 S143=6000，自由组不加等式。

S143 才是真正的自然日 24:00，S144 两组都只受物理界限制；**不复用旧 02/27 的末向量索引，
也不用 terminal=None 隐含日循环**。矩阵证据：两模式目标、integrality、约束矩阵与约束界
完全一致，只有下标 {terminal_index}（=S143）的上下界不同。

实际执行：先以真实当前首段供需与 E(00:00) 执行 qcarry，再执行新计划 q0..q142 到自然日
24:00，把真实状态与 q143 承诺传给下一日；**00:10 不重解计划**。12月31日的 q143 在
2026-01-01 执行一次，用于账本 B 尾段，不发布 2026 年新计划。

## 4. 模型求解与结果

### 4.1 两组总费

（见 1.2 表。）账本 A 与账本 B 各自 48096 段；两组账本 A 的期初状态相同
（{a_fixed['initial_kWh']:,.2f} kWh），期末分别为 {a_fixed['final_kWh']:,.2f} kWh（固定）与
{a_free['final_kWh']:,.2f} kWh（自由）。求解器全年无失败（固定组
{int(summary.iloc[0].formal_solves)} 次、自由组 {int(summary.iloc[1].formal_solves)} 次正式求解，
最大 MIP gap {max(summary.iloc[0].max_mip_gap, summary.iloc[1].max_mip_gap):.2e}）。

### 4.2 分月与分窗口

| 月 | 固定总费(元) | 自由总费(元) | 差(元) | 应急电量差(kWh) |
|---|---:|---:|---:|---:|
{month_rows}

| 口径 | 时段 | 应急费(元) | 应急电量(kWh) | 计划费(元) |
|---|---|---:|---:|---:|
{window_rows}

0—6、6—10、10—24 三段互斥且合计全年；19—21 与 23—24 为诊断口径，**不与主段重复加总**。

### 4.3 库存与估值

| 组 | 名义 S143 均值(kWh) | S143=6000 天数 | S143 触底天数 | 实际期初(kWh) | 实际期末(kWh) | 库存估值调整(元) | 调整后 C*(元) |
|---|---:|---:|---:|---:|---:|---:|---:|
{inventory_rows}

nu = median(p)/0.9 仅作库存诊断，不是售价或目标。能量恒等式残差
{abs(energy.identity_residual_kWh):.3e} kWh。

### 4.4 图表

{integrity_rows}

三图分别为午夜映射示意、两组模板账本费用与自然日分月差额、名义 S143 与实际日末 SOC。
图表已作尺寸/颜色数/非白像素的程序化完整性检查；本环境无法目视图像，未做人工目视核验。

## 5. 验证与适用边界

### 5.1 人工边界与实现检查

| 检查 | 预期 | 实测 | 结果 |
|---|---|---|---|
{boundary_rows}

### 5.2 预测与门控检查

| 检查 | 预期 | 实测 | 结果 |
|---|---|---|---|
{forecast_rows}

### 5.3 因果与未来扰动

| 扰动日 | 扰动 | 固定组前缀计划差 | 自由组前缀计划差 | 固定组当日计划差 | 自由组当日计划差 | 固定组当日实际应急(kWh) |
|---|---|---:|---:|---:|---:|---:|
{perturbation_rows}

6 例未来真值扰动（2月1日、6月21日、12月21日，负载×1.2 或光伏×0.7）只重建扰动日及其后的
训练依赖/预测/校准，并从共同起点重放：此前每日计划、实际状态、特征、标签与保护量必须逐位不变
（最大差 {perturbation['max_prefix_difference']:.3e} kWh），**扰动日自身发布的计划也不变**
（最大差 {perturbation['max_perturbation_day_plan_difference']:.3e} kWh），只有当日实际执行会随
真值改变——这正是午夜承诺的正确行为。

半日前缀（6月21日自然 slot 0—71 与 state[0:73]）最大差
{halfday['max_prefix_difference']:.3e} kWh。qcarry 与前一日模板 slot 143 的最大差
{carry['max_carry_difference_kWh']:.3e} kWh，每个物理区间只执行与结算一次
（账本 A 与 B 各 48096 段）。

### 5.4 内部复现（新运行自身）

**本例不存在事先已登记的新口径费用常数**，因此复现判据是"同一实现从共同状态冷启动再跑一次"：
固定组逐段计划/实际状态/应急/充放电/未使用最大差
{replay['max_segment_difference_kWh']:.3e} kWh，账本 A 现金差
{replay['cost_difference_yuan']:.3e} 元。**未引用旧 27 号的 1 元豁免。**

### 5.5 限制

1. 区间起点口径与"当前午夜执行前日承诺、00:10 不重解"都是**建模假设**，不是题面原文；
   论文与交付说明必须披露。
2. 2025-01-01 00:00—00:10 无源观测，按静置假设处理；2026-01-01 00:00—00:10 只用于账本 B 尾段。
3. 单年多轮方法设计，不是独立盲测；不把 48096 段当独立样本作显著性检验。
4. 新旧口径费用并列只能描述时间/信息/承诺边界的**综合迁移**，不能归因成纯移位收益。
5. 未新增 19 列负载、未调树参数、未扫描分位、未加软终端/二级择优/两日展望。
6. 上游 20/24/26/28 号审计不因本轮自动通过；本轮只覆盖新口径下的预测、账本与终端对照。
7. 三图已程序化检查（尺寸/非白像素/颜色数），**未做人工目视核验**。
8. 登记修订记录：{(len(registration.get('amendments', [])))} 次
   （{'; '.join(a['reason'][:80] for a in registration.get('amendments', [])) or '无'}），
   全部为报告/工程层修复，模型与冻结参数未变。

## 6. 文件与复现

- `results/q2_time_mapping/registration.json`、`run_manifest.json`、`artifact_hashes.json`：
  登记签名、起止时间与 wall time、输入哈希与源码快照。
- `source_interval_mapping.csv`：52560 个功率值与 144 个电价值的完整两坐标映射
  （源文件/工作表/行列、源日期与标签、区间起止、available_at、自然日/槽、模板日/槽）。
- 全部自检：`mapping_checks.json`、`artificial_mapping.json`、`forecast_summary.json`、
  `forecast_fit_log.csv`、`quantile_validation.json`、`boundary_checks.json`、
  `forecast_checks.json`、`internal_reproduction.json`、`carry_check.json`、
  `halfday_check.json`、`perturbation_checks.json`、`checks.json`。
- 无损浮点档案：`archive_float.npz`、`issued_load.npy` 等（CSV 未舍入）。
- 两组目录与校验副本：`{{N_fixed6000,N_free}}/`（自然长表、模板计划、求解日志、账本）、
  `*_validation.xlsx`（不替换正式 result2.xlsx）。
- 汇总表：`summary.csv`、`contrast.csv`、`daily_contrasts.csv`、`monthly_contrasts.csv`、
  `window_emergency.csv`、`stability.csv`、`inventory.csv`、`energy_contrasts.csv`、
  `all_ledgers.json`、`tail_interval.csv`；三图在 `figures/q2_time_mapping/`。
- 复现入口：

```powershell
Set-Location -LiteralPath 'E:/MathModeling/2026国赛/C题'
E:/Anaconda/envs/math_modeling/python.exe code/29_q2_time_mapping_experiment.py --mode full
E:/Anaconda/envs/math_modeling/python.exe code/30_q2_report_renderer.py
```

旧附件、源码与签名结果未改动；共享记忆与进度仅追加"实验自检完成，独立审计待完成"。
"""
    REPORT_MD.write_text(report, encoding='utf-8')
    print(f'report written: {REPORT_MD} ({len(report.splitlines())} lines)')


if __name__ == '__main__':
    main()
