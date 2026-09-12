#!/usr/bin/env python
"""One-off report-layer wording fixes for the four findings in 问题三_分位费用实验接收核查.md.

Findings 1.1 and 1.2 are already applied by earlier edits; this applies 1.3's coverage disclosure
plus the three wording items of 1.4. Report layer only: no recomputation, no rerun.
"""
from pathlib import Path

P = Path("code/q3_bias_quantile_cost_report.py")
t = P.read_text(encoding="utf-8")

subs = [
    # 1.4(b) public initial state
    ("公共 1 月初态（本月不重跑 1 月，公共 1 月重跑 "
     "{int(validation['budget']['public_january_reruns'])} 次）；",
     "公共 1 月预运行产生的 2025-02-01 00:00 实际初态 6075.795025925926 kWh 与 0 kWh 午夜原始/有效承诺"
     "（本月不重跑 1 月，公共 1 月重跑 {int(validation['budget']['public_january_reruns'])} 次；"
     "不是从 2025-01-01 的 6000 kWh 重新开始）；"),
    # 1.4(a) separate the agreed time convention from the transcription history
    ("- 时间与价格口径仍受 `start_time_v1` 与 MinerU 转换的已知限制约束，本轮未复核上游转换。",
     "- 时间与价格口径仍受已约定的区间起点假设 `start_time_v1` 约束（`00:10` 表示 00:10—00:20，"
     "前日末段属于次日 00:00—00:10）；题目转写历史上有过漏字，但本轮数值输入链是 `read_attachments` "
     "直接以 openpyxl 读取附件 1/2，MinerU 转写不是本轮数值依赖，两者不可混为一谈。本轮未复核上游转换。"),
    # 1.4(c) emergency rise is a disclosed risk indicator, not a new constraint
    ("   若采纳，需要先明确应急抬升（+{100.0 * ql.delta_emergency_kWh / rows.loc['L80', 'emergency_kWh']:.1f}% 电量、\n"
     "   +{int(rows.loc['L75', 'emergency_events'] - rows.loc['L80', 'emergency_events'])} 次事件）是否可接受。",
     "   应急抬升（+{100.0 * ql.delta_emergency_kWh / rows.loc['L80', 'emergency_kWh']:.1f}% 电量、\n"
     "   +{int(rows.loc['L75', 'emergency_events'] - rows.loc['L80', 'emergency_events'])} 次事件）是必须并列披露的"
     "风险指标：应急已按 5 倍电价计入总费，模型未另设应急额度或次数约束，因此它不自动构成违约、失供或新的"
     "不可行条件。按本轮既定费用目标，L75 是首选候选；既不为了挑选 L75 而忽略风险，也不为了保留 L80 而临时"
     "增设未定义的风险约束。"),
]

for old, new in subs:
    count = t.count(old)
    assert count == 1, (count, old[:60])
    t = t.replace(old, new)

P.write_text(t, encoding="utf-8")
print("applied", len(subs), "wording fixes")
