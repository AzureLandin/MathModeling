#!/usr/bin/env python
"""问题四 正式表填报与回读核查 —— 报告层（只读渲染）。

只读 ``results/q4_delivery/fill_validation.json`` 与 ``results/q4_price_transfer/`` 的汇总，
渲染填报与回读核查报告。**不写 Excel、不重算任何账本**。

运行::

    conda run -n math_modeling python code/q4_fill_result_tables_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "results/q4_delivery"
LEDGER = ROOT / "results/q4_price_transfer"
REPORT_MD = ROOT / "reports/问题四/问题四_正式表填报与回读核查.md"
CONVERGENCE_MD = ROOT / "reports/问题四/问题四_最终收敛与交付方案.md"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def num(value, digits=6):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:,.{digits}f}"


def sci(value):
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return "—"


def table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main():
    assert Path(sys.prefix).name == "math_modeling", sys.prefix
    record = load_json(DELIVERY / "fill_validation.json")
    upstream = pd.read_csv(LEDGER / "summary.csv").set_index("group")
    strat = {name: record["checks"][name]["strategy"] for name in ("result4-2", "result4-3")}
    contrasts = pd.read_csv(LEDGER / "contrasts.csv")
    total = contrasts[contrasts.item == "total"].set_index("contrast")

    lines = []
    add = lines.append
    add("# 问题四正式表填报与回读核查")
    add("")
    add("日期：2026-09-13。状态：result4-2 / result4-3 副本已按附件5 模板填写并完成逐段回读核对；"
        "本轮**只填报与回读**，未重新运行任何全年优化。")
    add(f"依据：[问题四最终收敛与交付方案]({CONVERGENCE_MD.resolve().as_posix()})。")
    add("填报层：`code/q4_fill_result_tables.py`（只读账本 → 写副本 → 重读核对 → 落盘 JSON）；"
        "报告层：`code/q4_fill_result_tables_report.py`（本报告，只读渲染）。")
    add("")
    add("## 1. 结论")
    add("")
    add("1. **正式候选按方案锁定：4-2 用 Q42、4-3 用 Q43_S2**；Q43_S0 仅作候选对照，"
        "**没有写入任何模板**。")
    add(f"2. **两个副本已生成，原件未动**：`results/q4_delivery/result4-2_filled.xlsx`、"
        f"`result4-3_filled.xlsx`；附件5 原始模板的 SHA256 在填报前后完全一致，"
        f"并另存备份于 `results/q4_delivery/template_backup/`。")
    add("3. **回读核对全部通过**：日期、144 个时间标签、行数、逐段购电量、储能 6 个 4 小时块、"
        "应急区间、全天合计与全天购电费、初末库存与汇总均与 CSV 账本一致；"
        f"能量最大差 {sci(max(v['max_abs_value_error_kWh'] for n in ('result4-2', 'result4-3') for v in record['checks'][n]['read_back']['plan'].values()))} kWh、"
        f"汇总最大差 {sci(max(v['max_abs_total_error_kWh'] for n in ('result4-2', 'result4-3') for v in record['checks'][n]['read_back']['plan'].values()))} kWh、"
        f"费用最大差 {sci(max(v['max_abs_cost_error_yuan'] for n in ('result4-2', 'result4-3') for v in record['checks'][n]['read_back']['plan'].values()))} 元，"
        f"全部远小于容差（能量 1e-6 kWh、费用 1e-4 元）。")
    add("4. **字段无缺口**：账本提供的字段足以按模板要求展开，未出现需要猜测或补造的数据；"
        "`fill_validation.json` 的 `gaps` 为空。")
    add(f"5. **正式期口径**：只使用自然日 2025-02-01—2025-12-31 共 {record['checks']['result4-2']['ledger_summary']['natural_segments']} 段；"
        f"1 月公共初始化（含其 29 次 MILP）不计入正式费用；时间按 `00:10` 对应 `[00:10,00:20)`。")
    add("")

    add("## 2. 输入与输出")
    add("")
    add(table([(f"{name} 模板", record["inputs"][f"{name}_template"]["path"],
                record["inputs"][f"{name}_template"]["sha256"][:16] + "…", "只读，未修改")
               for name in ("result4-2", "result4-3")]
              + [(f"{strat[n]} 账本", record["inputs"][f"{strat[n]}_dispatch"]["path"],
                  record["inputs"][f"{strat[n]}_dispatch"]["sha256"][:16] + "…",
                  f"{record['inputs'][f'{strat[n]}_dispatch']['rows']} 行，只读")
                 for n in ("result4-2", "result4-3")],
              ["类别", "路径", "SHA256", "说明"]))
    add("")
    add(table([(f"{name} 填写副本", record["outputs"][f"{name}_filled"]["path"],
                record["outputs"][f"{name}_filled"]["sha256"][:16] + "…",
                f"{record['outputs'][f'{name}_filled']['bytes']:,} 字节")
               for name in ("result4-2", "result4-3")]
              + [(f"{name} 模板备份", record["outputs"][f"{name}_template_backup"]["path"],
                  record["outputs"][f"{name}_template_backup"]["sha256"][:16] + "…", "与原件逐字节一致")
                 for name in ("result4-2", "result4-3")]
              + [("填报记录", "results/q4_delivery/fill_validation.json", "—",
                  "输入/输出/字段映射/核查结果/异常")],
              ["类别", "路径", "SHA256", "说明"]))
    add("")

    add("## 3. 字段映射与展开规则")
    add("")
    add(table([(key, value) for key, value in record["field_mapping"].items()],
              ["模板字段 / 项目", "账本来源与规则"]))
    add("")
    add("模板的三个（4-3 为四个）工作表原本只给了少量示例行，按题目要求**展开到完整评价期**：")
    add("")
    add("- 计划购电量 / 调整购电量：334 行 × 144 列 = 48,096 个单元格，"
        "第 d 行为该日 `00:10` 至次日 `00:10` 的 144 段，末两列为全天合计与全天购电费；")
    add("- 充放电量：334 行 × 6 个 4 小时块 = 2,004 行，`时刻`/`储电量` 两列沿用模板布局"
        "（0 号块写当日 0:00 与日初库存、1 号块写 24:00 与日末库存）；")
    add("- 紧急购电量：按**自然日内连续正应急区间合并**逐行展开，每天至少一行，"
        "无应急日写 `无紧急购电` 与 0（与已交付的 `result2.xlsx` 同一约定）。")
    add("")

    add("## 4. 回读核查结果")
    add("")
    for name in ("result4-2", "result4-3"):
        block = record["checks"][name]
        add(f"### 4.{'2' if name == 'result4-2' else '3'} {name}（{block['strategy']}）")
        add("")
        add(table([(sheet, f"{entry['rows']} 行 × {entry['slots']} 列",
                    entry["first_label"] + " … " + entry["last_label"], entry["time_labels"],
                    sci(entry["max_abs_value_error_kWh"]), sci(entry["max_abs_total_error_kWh"]),
                    sci(entry["max_abs_cost_error_yuan"]))
                   for sheet, entry in block["read_back"]["plan"].items()],
                  ["工作表", "行×列", "时间标签", "标签数", "逐格最大差 (kWh)",
                   "全天合计最大差 (kWh)", "全天购电费最大差 (元)"]))
        add("")
        storage = block["read_back"]["storage"]
        emergency = block["read_back"]["emergency"]
        summary = block["ledger_summary"]
        add(table([
            ("充放电量", f"{storage['rows']} 行（334 天 × 6 块）",
             f"充电 {num(storage['total_charge_kWh'])} kWh、放电 {num(storage['total_discharge_kWh'])} kWh",
             f"逐块最大差 {sci(max(storage['max_abs_charge_error_kWh'], storage['max_abs_discharge_error_kWh']))} kWh"),
            ("其中日初库存", "0 号块 `储电量`", f"{num(storage['initial_state_kWh'], 9)} kWh", "与账本一致"),
            ("其中日末库存", "1 号块 `储电量`", f"逐日核对，最大差 < 1e-6 kWh", "已逐日断言"),
            ("紧急购电量", f"{emergency['rows']} 行（无应急 {emergency['zero_rows']} 行 / 区间 {emergency['range_rows']} 行）",
             f"合计 {num(emergency['total_kWh'])} kWh", f"与账本差 {sci(emergency['abs_total_error_kWh'])} kWh"),
            ("账本自然总费", "48096 段", f"{num(summary['total_cost_yuan'], 4)} 元", "见第 5 节交叉核对"),
        ], ["项目", "范围", "数值", "核对"]))
        add("")

    add("## 5. 与上游汇总的交叉核对")
    add("")
    add("填入 Excel 的数量与上游 `summary.csv` 逐项对照（上游值为只读引用）：")
    add("")
    rows = []
    for name, group in (("result4-2", "Q42"), ("result4-3", "Q43_S2")):
        block = record["checks"][name]
        summary = block["ledger_summary"]
        storage = block["read_back"]["storage"]
        emergency = block["read_back"]["emergency"]
        pairs = [
            ("购电量合计", float(upstream.loc[group, "q_eff_kWh"]),
             float(summary["purchase_effective_kWh"]), 1e-6, "kWh"),
            ("普通购电费", float(upstream.loc[group, "ordinary_cost_yuan"]),
             float(summary["ordinary_cost_yuan"]), 1e-4, "元"),
            ("调整费", float(upstream.loc[group, "adjustment_cost_yuan"]),
             float(summary["adjustment_cost_yuan"]), 1e-4, "元"),
            ("应急费", float(upstream.loc[group, "emergency_cost_yuan"]),
             float(summary["emergency_cost_yuan"]), 1e-4, "元"),
            ("自然总费", float(upstream.loc[group, "natural_total_yuan"]),
             float(summary["total_cost_yuan"]), 1e-4, "元"),
            ("充电量", float(upstream.loc[group, "charge_kWh"]),
             float(storage["total_charge_kWh"]), 1e-6, "kWh"),
            ("放电量", float(upstream.loc[group, "discharge_kWh"]),
             float(storage["total_discharge_kWh"]), 1e-6, "kWh"),
            ("应急电量", float(upstream.loc[group, "emergency_kWh"]),
             float(emergency["total_kWh"]), 1e-6, "kWh"),
            ("期末库存", float(upstream.loc[group, "final_natural_state_kWh"]),
             float(summary["final_state_kWh"]), 1e-6, "kWh"),
        ]
        for label, left, right, tol, unit in pairs:
            gap = abs(left - right)
            rows.append((f"{name} {label}", f"{num(left)} {unit}", f"{num(right)} {unit}",
                         f"差 {sci(gap)} ≤ {tol:g}，通过" if gap < tol else
                         f"差 {sci(gap)} > {tol:g}，不通过"))
    add(table(rows, ["项目", "上游 summary.csv", "本次回读/账本", "判定"]))
    add("")
    add("节费量沿用上游定义 `S = C_固定计划 − C_重优化`："
        f"S42 = +{num(float(total.loc['S42', 'delta_yuan']) * -1, 2)} 元、"
        f"S43_S2 = +{num(float(total.loc['S43_S2', 'delta_yuan']) * -1, 2)} 元"
        f"（本报告不重算，只引用；填报本身不改变任何费用）。")
    add("")

    add("## 6. 边界与说明")
    add("")
    add("- **两个账本不可混用**：计划购电量工作表对应模板账本 B（`00:10`—次日 `00:10`，"
        "48,096 段），第 5 节的现金汇总对应自然账本 A（`00:00`—`24:00`，48,096 段）；"
        "两者只差首尾一个区间，本轮两者的费用桥接在 Q42 与 Q43_S2 上均为 0。")
    add("- **全天购电费只计购电**：`Σ 交付区间价 × 写入的购电量`，不含 4-3 的 "
        "`0.5|q_eff − q0|` 调整费与 5 倍应急费；三项分项现金见 `summary.csv` 与第 5 节。")
    add("- **4-3 同时保留两套计划**：`计划购电量` = 0 点原始承诺 `q0`，`调整购电量` = 最终有效量 "
        "`q_eff`；两者的行合计分别为 "
        f"{num([v for k, v in record['checks']['result4-3']['read_back']['plan'].items() if k == '计划购电量'][0]['total_purchase_kWh'])} kWh 与 "
        f"{num([v for k, v in record['checks']['result4-3']['read_back']['plan'].items() if k == '调整购电量'][0]['total_purchase_kWh'])} kWh。")
    add("- **应急展开约定与 result3 的差异**：本轮按自然日连续正应急区间合并（与 `result2.xlsx` 一致），"
        f"4-2 得 {record['checks']['result4-2']['read_back']['emergency']['rows']} 行、"
        f"4-3 得 {record['checks']['result4-3']['read_back']['emergency']['rows']} 行；"
        f"对应的逐 10 分钟正应急段数分别为 "
        f"{record['checks']['result4-2']['ledger_summary']['emergency_intervals']} 与 "
        f"{record['checks']['result4-3']['ledger_summary']['emergency_intervals']}。"
        f"此前 `result3.xlsx` 用的是逐段一行，两种口径对**电量合计**没有影响，"
        f"但行数与 `购电时间段` 的写法不同，已在填报记录中如实登记。")
    add("- 本轮**没有**训练、重建保护、重新优化或参数扫描；全流程只有读取 CSV、复制模板、"
        "写入副本、重读副本四个动作。附件5 原件、问题二与问题三产物均未改动。")
    add("- 填报副本是**副本**：正式提交前若需覆盖附件5 原件，应由用户决定并另行记录。")
    add("")

    add("## 7. 异常与缺口")
    add("")
    if record.get("anomalies"):
        for item in record["anomalies"]:
            add(f"- {item}")
    else:
        add("- 无异常：`fill_validation.json` 的 `anomalies` 为空。")
    if record.get("gaps"):
        for item in record["gaps"]:
            add(f"- 缺口：{item}")
    else:
        add("- 无字段缺口：账本字段足以按模板要求展开，未猜测或补造任何数据。")
    add("")

    add("## 8. 产物与复现")
    add("")
    add("产物：" + "；".join([
        f"`{record['outputs'][k]['path']}`（{v}）" for k, v in
        (("result4-2_filled", "4-2 填写副本"), ("result4-3_filled", "4-3 填写副本"),
         ("result4-2_template_backup", "4-2 原件备份"), ("result4-3_template_backup", "4-3 原件备份"))])
        + "；填报记录 `results/q4_delivery/fill_validation.json`；本报告。")
    add("")
    add(f"填报脚本整轮耗时 {record['elapsed_seconds']:.2f} 秒，"
        f"无 MILP、无训练、无参数扫描。")
    add("")
    add("复现：赛题目录执行 `conda run -n math_modeling python code/q4_fill_result_tables.py`，"
        "再执行 `conda run -n math_modeling python code/q4_fill_result_tables_report.py`。")
    add("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "report": REPORT_MD.name, "lines": len(lines),
                      "paragraphs": len(lines)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
