"""Read-only report and charts for Q4; never trains or solves."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/q4_known_price'
REPORT = ROOT/'reports/问题四/问题四_已知波动电价首轮实验结果.md'
FIG = ROOT/'figures/q4_known_price'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    manifest = json.loads((DATA/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    for name,digest in manifest['outputs'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest, name
    s = pd.read_csv(DATA/'summary.csv').set_index('group')
    monthly = pd.read_csv(DATA/'monthly.csv')
    daily = pd.read_csv(DATA/'daily.csv')
    init = json.loads((DATA/'public_initialization.json').read_text(encoding='utf-8'))
    checks = json.loads((DATA/'checks.json').read_text(encoding='utf-8'))
    comparisons = []
    for candidate,base in [('Q42','F42'),('Q43','F43'),('Q43','Q42')]:
        ma = monthly[monthly.group==candidate].set_index('month').total_cost_yuan
        mb = monthly[monthly.group==base].set_index('month').total_cost_yuan
        da = daily[daily.group==candidate].set_index('date').total_cost_yuan
        db = daily[daily.group==base].set_index('date').total_cost_yuan
        saving = s.loc[base,'natural_total_yuan']-s.loc[candidate,'natural_total_yuan']
        comparisons.append([f'{candidate} 对 {base}',f'{saving:,.2f}',
            f'{100*saving/s.loc[base,"natural_total_yuan"]:.4f}%',f'{int((ma<mb-1e-4).sum())}/11',
            f'{int((da<db-1e-4).sum())}/334'])
    versions = pd.read_csv(DATA/'Q43_decisions.csv',parse_dates=['published_at'])
    accepted = versions.groupby(versions.published_at.dt.hour).accepted.agg(['sum','count'])
    chosen = ['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
    selected, charge, events = [], [], []
    for group in ['Q42','Q43']:
        f = pd.read_csv(DATA/f'{group}_dispatch.csv',parse_dates=['interval_start','interval_end'])
        natural = f.iloc[:-1].copy()
        natural['date']=natural.interval_start.dt.strftime('%Y-%m-%d')
        for day, rows in natural.groupby('date'):
            flag=rows.emergency_kWh.to_numpy()>1e-6
            idx=np.where(flag)[0]
            chunks=np.split(idx,np.where(np.diff(idx)>1)[0]+1) if len(idx) else []
            for chunk in chunks:
                a=rows.iloc[chunk]
                events.append(dict(group=group,date=day,start=str(a.interval_start.iloc[0]),
                                   end=str(a.interval_end.iloc[-1]),energy_kWh=a.emergency_kWh.sum()))
            if day not in chosen:
                continue
            selected.append([group,day,f'{rows.total_cost_yuan.sum():,.2f}',
                f'{rows.q0_kWh.sum():,.3f}',f'{rows.q_eff_kWh.sum():,.3f}',
                f'{rows.emergency_kWh.sum():,.3f}',f'{rows.state_start_kWh.iloc[0]:,.3f}',
                f'{rows.state_end_kWh.iloc[-1]:,.3f}'])
            for block,b in rows.groupby(rows.interval_start.dt.hour//4):
                charge.append(dict(group=group,date=day,start_hour=int(block*4),end_hour=int(block*4+4),
                   charge_kWh=b.charge_kWh.sum(),discharge_kWh=b.discharge_kWh.sum()))
    pd.DataFrame(charge).to_csv(DATA/'selected_charge_discharge_4hour.csv',index=False)
    pd.DataFrame(events).to_csv(DATA/'emergency_events.csv',index=False)
    FIG.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].bar(s.index,s.natural_total_yuan/1e6,color=['#adb7c4','#adb7c4','#247ba0','#36a38b'])
    axes[0].set(ylabel='Total cost (million CNY)',title='Feb-Dec 2025; common new January state')
    for candidate,base in [('Q42','F42'),('Q43','F43')]:
        m=monthly.pivot(index='month',columns='group',values='total_cost_yuan')
        axes[1].plot(m.index.str[5:],(m[base]-m[candidate])/1000,marker='o',label=f'{candidate} vs {base}')
    axes[1].axhline(0,color='grey',linewidth=.8)
    axes[1].set(xlabel='Month (2025)',ylabel='Savings (thousand CNY)',title='Reoptimization vs frozen purchases')
    axes[1].legend()
    fig.savefig(FIG/'cost_comparison.png',dpi=170)
    plt.close(fig)
    text = f'''# 问题四已知波动电价首轮实验结果

状态：首轮q80 Baseline已完成计算及有限自检；未做独立全链路审计，未填写正式Excel。计算与报告分层，改报告不触发求解。

## 1. 问题分析

每天0:00已知当天00:10至次日00:10的144段附件4交付电价，包含次日午夜尾段；这是用户选定的信息假设。未来负载、光伏实测和未发布预报仍不可用于计划。4-2每日一次普通计划，4-3在6/12/18点按最新光伏预报修订未来计划。

本轮按本任务此前固定的q80首轮规格运行。问题三其他任务已推进到Linear+q75首选候选；本次Q43明确对应其保留的Linear+q80/B2基准，不冒称已经迁移q75，也不据本次结果替换问题三候选。

## 2. 数据预处理与公共初始化

附件2/4均按365×144读取，日期及时间列逐项核对。自然午夜价格来自前一源行末列，其余来自本行前143列；没有循环错用本日尾价。00:10代表00:10—00:20。

公共一月从6000 kWh出发：1月1日普通计划0、电池静置；缺失的00:00—00:10供需/费用保留未知。1月2日仍为零普通计划、恢复实际反馈。1月3—31日复用已保存朴素预测，不加q80，采用名义自然24:00为6000的MILP，仅将电价替换为附件4，并连续执行实际反馈。共29次求解。

重算后2月1日00:00库存 **{init['feb1_state_kWh']:.9f} kWh**，午夜承诺 **{init['feb1_carry_kWh']:.9f} kWh**。两分支与对照均从此处出发。一月不进入正式2—12月现金费用；6000名义末态只用于公共一月，正式期采用自由末态。

供需预测/保护直接读冻结档案，不重训：Q42来自问题二15/12特征LightGBM与W28/q80；Q43来自问题三15特征负载、Linear光伏转换和同发布组W28/q80。价格改变不改变这些外生预测误差档案。档案来源与哈希见registration.json。

## 3. 模型建立

保持单向效率0.9、1200—10800 kWh库存界及5000 kW充放电功率上限。计划通过MILP求解，实际富余优先充电、缺口优先放电、剩余应急，状态跨日连续。

Q42的名义目标为$\\sum_t p_tq_t^0$；Q43日内目标为$\\sum_t p_t(q_t+0.5|q_t-q_t^0|)$。新旧方案在同一保护需求、价格和真实库存下，以普通费、调整费及反馈预测应急费合计评分，严格降低超过0.0001元才替换。实际费用分别为$\\sum p_t(q_t^0+5e_t)$和$\\sum p_t(q_t^{{eff}}+0.5|q_t^{{eff}}-q_t^0|+5e_t)$；每段相对原始承诺最终一次结算。

F42/F43固定前问保存的原始/有效普通购电计划，从新的公共初态重放同一实际反馈，价格改用附件4，首个午夜承诺替换为公共新承诺。它们是固定计划的可执行回放基准，不是从新初态重新运行固定价优化器的反事实；F43也不重新进行新旧评分。与原动作原封不动重计费相比，此设计匹配了本次初态，因此其应急和库存可因初态变化而略有变化。

## 4. 模型求解与结果

以下统一为自然日账本A：2025-02-01至12-31，334天、48096段，单位元。

{table(['组别','总费用','普通费','调整费','应急费','应急电量/kWh'],[[g]+[f'{s.loc[g,c]:,.2f}' for c in ['natural_total_yuan','ordinary_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','emergency_kWh']] for g in s.index])}

{table(['比较','节省金额/元','节省比例','更省月份','更省天数'],comparisons)}

Q43与Q42的差同时包含光伏预测源和日内调整机制变化，不能解释为纯预报信息价值或纯调整收益。相对各自固定计划基准的差反映该具体重优化策略的收益，不证明全局随机最优。

风险没有全面改善：Q42相对F42应急费增加{s.loc['Q42','emergency_cost_yuan']-s.loc['F42','emergency_cost_yuan']:,.2f}元，应急天数由{int(s.loc['F42','emergency_days'])}增至{int(s.loc['Q42','emergency_days'])}；其节费来自普通购电费下降。Q43相对F43应急电量增加{s.loc['Q43','emergency_kWh']-s.loc['F43','emergency_kWh']:,.2f} kWh，但应急费减少{s.loc['F43','emergency_cost_yuan']-s.loc['Q43','emergency_cost_yuan']:,.2f}元，应急天数由{int(s.loc['F43','emergency_days'])}增至{int(s.loc['Q43','emergency_days'])}。费用、电量与事件数不能互相替代。

Q43日内接受情况：{'; '.join(f'{h}:00接受{int(r["sum"])}/{int(r["count"])}次' for h,r in accepted.iterrows())}。

{table(['组别','自然账本A总费/元','模板账本B总费/元','自然期末库存/kWh','模板期末库存/kWh'],[[g]+[f'{s.loc[g,c]:,.6f}' for c in ['natural_total_yuan','template_total_yuan','final_natural_state_kWh','final_template_state_kWh']] for g in s.index])}

账本B覆盖各模板的00:10起144段；其总费与A之差等于2026-01-01午夜尾段费用减2025-02-01午夜首段费用。所有表格和差额比较必须使用同一账本。

### 指定日期自然日结果

{table(['组别','日期','总费/元','原计划量/kWh','有效普通购电/kWh','应急/kWh','日初库存/kWh','日末库存/kWh'],selected)}

此表购电量按自然日汇总，含前日继承午夜承诺；不是当天0:00发布的模板144段总量。逐段数据含日期、原始/有效计划、供需、储能和费用。指定日六个4小时充放电及连续应急事件见结果目录。

![费用对照](../../figures/q4_known_price/cost_comparison.png)

### 验证与边界

输入日期/时间列、全期价格逐段映射、公共初态、实际能量平衡/状态递推/上下界/功率/互斥、现金结算及双账本桥接均通过。Q42/Q43保存的名义充放电及状态也重新读回核对能量等式。6月21日使用问题二既有独立组装内核按同一新价格及真实初态求解一次，目标差{checks['single_day_q2_kernel']['objective_difference_yuan']:.3g}元；允许等费用多解，不声称计划唯一。

新增主求解1699次（公共一月29＋Q42 334＋Q43 1336），额外单日核对1次，训练0次，整轮{manifest['seconds']:.2f}秒。本轮复用上游供需预测/保护信息边界证据，没有重建训练链或新增全期泄漏扰动测试；已知价格不适用“未来价格扰动后计划不变”检验。未运行DP、联合情景、分位扫描或额外终端实验。

本年数据已用于此前模型设计，结果是因果历史回测，不是独立盲测。自由末态和贪心反馈仍有短视风险，期末库存应与现金费一起阅读。固定计划对照不会证明附加信息的理论最优价值。

### 复现与文件

- 计算：`conda run -n math_modeling python code/q4_known_price_experiment.py`。
- 只读报告：`conda run -n math_modeling python code/q4_known_price_report.py`。
- 数值证据：`results/q4_known_price/`，包含注册哈希、公共一月、各组dispatch、Q42/Q43计划版本及名义完整电量/状态、求解日志、月度/日度、指定日期及检查。
- 本轮保留四组CSV实验结果，不覆盖附件5的正式result4-2.xlsx和result4-3.xlsx。
'''
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(text,encoding='utf-8')
    print(table(['比较','节省金额/元','节省比例','更省月份','更省天数'],comparisons))
    print(s[['natural_total_yuan','final_natural_state_kWh']].to_string())


if __name__=='__main__':
    main()
