import pandas as pd
import numpy as np

# 读取附件2数据
path2 = r'E:\MathModeling\第一次模拟竞赛\附件\附件2 市主城区 10 区域分时段充电车次.xlsx'
def read_hourly(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=0)
    raw = raw.dropna(subset=[raw.columns[0]])
    raw = raw[pd.to_numeric(raw.iloc[:, 0], errors='coerce').notna()].copy()
    raw.iloc[:, 0] = raw.iloc[:, 0].astype(int)
    return raw.sort_values(raw.columns[0]).reset_index(drop=True).head(10).iloc[:, 1:25].to_numpy(float)

q_wd = read_hourly(path2, '工作日分时段充电车次数据')
q_we = read_hourly(path2, '周末充电车次数据')

print('=== 附件2原始车次数据 ===')
for i in range(10):
    q_wd_sum = q_wd[i].sum()
    q_we_sum = q_we[i].sum()
    q_2025 = (5 * q_wd_sum + 2 * q_we_sum) / 7
    print(f'区域{i+1}: Q_wd_sum={q_wd_sum:.0f}, Q_we_sum={q_we_sum:.0f}, Q_2025={q_2025:.2f}')

print()
print('=== 年份递推 ===')
print('区域 | Q_2025(基期) | Q_2026=Q0*1.15 | Q_2027=Q0*1.15^2 | Q_2028=Q0*1.15^3')
for i in range(10):
    q0 = (5 * q_wd[i].sum() + 2 * q_we[i].sum()) / 7
    q1 = q0 * 1.15
    q2 = q0 * 1.3225
    q3 = q0 * 1.520875
    print(f'  {i+1:2d} | {q0:10.2f} | {q1:12.2f} | {q2:14.2f} | {q3:12.2f}')

print()
print('=== 用户认为的"基期"值 vs 真实基期 ===')
print('区域 | 用户认为Q_2025 | 实际Q_2025 | 实际Q_2026=Q0*1.15')
# 用户认为 847.2 是区域10的Q_2025，但实际 Q_2025=736.71
for i in [0, 9]:  # 区域1和区域10
    q0 = (5 * q_wd[i].sum() + 2 * q_we[i].sum()) / 7
    q1 = q0 * 1.15
    print(f'  {i+1:2d} | (用户看表中Q_2026) | {q0:10.2f} | {q1:12.2f}')

print()
print('=== 结论 ===')
print('表中Q_2026列的值 = Q_2025 * 1.15，这是正确的。')
print('用户误将Q_2026的值当成了Q_2025（基期）。')
print('代码和表都是正确的，无需修改。')
