"""问题一至问题四的公共 CSV 数据处理与正式结果导出。

本脚本只读取 ``C题/results`` 中已经冻结的结果，不训练、不求解、不修改
Excel 或原始 CSV。导出的文件使用 UTF-8-SIG，便于 Excel 直接打开；原始
策略标识和来源路径保留在结果列中，公共初始化阶段也不会被静默丢弃。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "正式CSV结果"

CANONICAL_COLUMNS = [
    "问题编号", "方案名称", "账本类型", "评价阶段", "自然日", "时段开始", "时段结束",
    "价格_元每千瓦时", "实际负载_千瓦", "光伏功率_千瓦", "净负荷_千瓦时",
    "计划购电量_千瓦时", "最终有效购电量_千瓦时", "充电量_千瓦时", "放电量_千瓦时",
    "应急购电量_千瓦时", "弃电量_千瓦时", "储能期初_千瓦时", "储能期末_千瓦时",
    "普通购电费_元", "调整购电费_元", "应急购电费_元", "总费用_元",
    "原始策略", "原始时段序号", "原始来源",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.Series(pd.NA, index=frame.index, dtype="object", name=name)


def _col(frame: pd.DataFrame, source: str, target: str) -> pd.Series:
    if source not in frame:
        return _empty(frame, target)
    return frame[source]


def _phase(values: pd.Series) -> pd.Series:
    # 只解析带 ISO 日期前缀的字段；问题一使用纯时刻字符串，不能参与日期解析。
    text = values.astype("string")
    date_text = text.where(text.str.match(r"^\d{4}-\d{2}-\d{2}"))
    dates = pd.to_datetime(date_text, errors="coerce")
    return pd.Series(
        ["公共初始化" if pd.notna(v) and v < pd.Timestamp("2025-02-01") else "正式评价期"
         for v in dates], index=values.index
    )


def _base(frame: pd.DataFrame, problem: str, plan: str, ledger: str,
          source: str, date_col: str, order_col: str | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["问题编号"] = problem
    out["方案名称"] = plan
    out["账本类型"] = ledger
    out["自然日"] = _col(frame, date_col, "自然日")
    out["时段开始"] = _col(frame, "interval_start", "时段开始")
    out["时段结束"] = _col(frame, "interval_end", "时段结束")
    if out["时段开始"].isna().all() and date_col in frame:
        out["时段开始"] = frame[date_col]
    out["评价阶段"] = _phase(out["自然日"])
    out["原始策略"] = _col(frame, "strategy_id", "原始策略")
    if out["原始策略"].isna().all():
        out["原始策略"] = _col(frame, "group", "原始策略")
    out["原始时段序号"] = (_col(frame, order_col, "原始时段序号")
                         if order_col else pd.Series(frame.index, index=frame.index))
    out["原始来源"] = source
    return out


def _finish(out: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "price_yuan_kWh": "价格_元每千瓦时",
        "load_kW": "实际负载_千瓦", "actual_load_kW": "实际负载_千瓦",
        "pv_kW": "光伏功率_千瓦", "actual_pv_kW": "光伏功率_千瓦",
        "net_kWh": "净负荷_千瓦时", "charge_kWh": "充电量_千瓦时",
        "discharge_kWh": "放电量_千瓦时", "emergency_kWh": "应急购电量_千瓦时",
        "unused_kWh": "弃电量_千瓦时", "state_start_kWh": "储能期初_千瓦时",
        "state_end_kWh": "储能期末_千瓦时", "ordinary_cost_yuan": "普通购电费_元",
        "adjustment_cost_yuan": "调整购电费_元", "emergency_cost_yuan": "应急购电费_元",
        "total_cost_yuan": "总费用_元",
    }
    for source, target in mapping.items():
        if source in frame and (target not in out or out[target].isna().all()):
            out[target] = frame[source]
    for col in CANONICAL_COLUMNS:
        if col not in out:
            out[col] = _empty(frame, col)
    return out[CANONICAL_COLUMNS]


def normalize_q1(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = _base(frame, "问题一", "新时间口径确定性MILP", "自然日调度账本",
                source, "interval", "start_minute")
    out["自然日"] = "附件1确定性单日"
    out["评价阶段"] = "正式结果"
    periods = frame["interval"].astype(str).str.split("-", n=1, expand=True)
    out["时段开始"] = periods[0]
    out["时段结束"] = periods[1]
    out["计划购电量_千瓦时"] = _col(frame, "grid_kWh", "计划购电量_千瓦时")
    out["最终有效购电量_千瓦时"] = _col(frame, "grid_kWh", "最终有效购电量_千瓦时")
    out["总费用_元"] = _col(frame, "cost_yuan", "总费用_元")
    out["原始策略"] = "新时间口径确定性MILP"
    return _finish(out, frame)


def normalize_q2_natural(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = _base(frame, "问题二", "N_free：LightGBM负载15列/光伏12列、W28/q80、自由末态",
                "自然日实际调度账本", source, "date", "natural_slot")
    out["计划购电量_千瓦时"] = _col(frame, "plan_kWh", "计划购电量_千瓦时")
    out["最终有效购电量_千瓦时"] = out["计划购电量_千瓦时"]
    out["普通购电费_元"] = _col(frame, "planned_cost_yuan", "普通购电费_元")
    out["应急购电费_元"] = _col(frame, "emergency_cost_yuan", "应急购电费_元")
    out["总费用_元"] = out["普通购电费_元"].fillna(0) + out["应急购电费_元"].fillna(0)
    return _finish(out, frame)


def normalize_q2_template(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = _base(frame, "问题二", "N_free：模板购电计划",
                "模板时段计划账本", source, "publication_date", "template_slot")
    out["计划购电量_千瓦时"] = _col(frame, "planned_kWh", "计划购电量_千瓦时")
    out["最终有效购电量_千瓦时"] = out["计划购电量_千瓦时"]
    out["应急购电量_千瓦时"] = _col(frame, "emergency_kWh", "应急购电量_千瓦时")
    out["普通购电费_元"] = _col(frame, "planned_cost_yuan", "普通购电费_元")
    out["应急购电费_元"] = _col(frame, "emergency_cost_yuan", "应急购电费_元")
    out["总费用_元"] = out["普通购电费_元"].fillna(0) + out["应急购电费_元"].fillna(0)
    out["评价阶段"] = out["自然日"].map(
        lambda value: "正式评价期" if str(value) >= "2025-02-01" else "公共初始化"
    )
    starts = pd.to_datetime(out["时段开始"], errors="coerce")
    out["时段结束"] = (starts + pd.Timedelta(minutes=10)).dt.strftime("%Y-%m-%d %H:%M:%S")
    return _finish(out, frame)


def normalize_q34(frame: pd.DataFrame, source: str, problem: str, plan: str, ledger: str) -> pd.DataFrame:
    out = _base(frame, problem, plan, ledger, source, "owner_date", None)
    starts = pd.to_datetime(frame["interval_start"], errors="raise")
    out["评价阶段"] = pd.Series("正式评价期", index=frame.index)
    out.loc[starts < pd.Timestamp("2025-02-01"), "评价阶段"] = "公共初始化"
    out.loc[starts >= pd.Timestamp("2026-01-01"), "评价阶段"] = "模板跨日尾段"
    out["原始时段序号"] = pd.Series(frame.index, index=frame.index)
    out["计划购电量_千瓦时"] = _col(frame, "q0_kWh", "计划购电量_千瓦时")
    out["最终有效购电量_千瓦时"] = _col(frame, "q_eff_kWh", "最终有效购电量_千瓦时")
    return _finish(out, frame)


EXPORTS: list[tuple[str, Path, str, Callable[[pd.DataFrame, str], pd.DataFrame]]] = [
    ("问题一_正式调度结果.csv", ROOT / "results/q1_start_time_20260912/dispatch_chronological.csv", "问题一", normalize_q1),
    ("问题二_自然日调度结果.csv", ROOT / "results/q2_time_mapping/N_free/natural_dispatch.csv", "问题二", normalize_q2_natural),
    ("问题二_模板购电计划.csv", ROOT / "results/q2_time_mapping/N_free/template_plan.csv", "问题二", normalize_q2_template),
    ("问题三_正式滚动调度结果.csv", ROOT / "results/q3_intraday_load_cost/S2_load_q75/dispatch.csv", "问题三", lambda f, s: normalize_q34(f, s, "问题三", "S2：F2日内负载修正、Linear光伏、F2自身W28/q75、四节点滚动", "滚动实际调度账本")),
    ("问题四_4-2正式调度结果.csv", ROOT / "results/q4_price_transfer/Q42/dispatch.csv", "问题四-4-2", lambda f, s: normalize_q34(f, s, "问题四-4-2", "Q42：问题二配置迁移至附件4波动电价", "波动电价滚动调度账本")),
    ("问题四_4-3正式调度结果.csv", ROOT / "results/q4_price_transfer/Q43_S2/dispatch.csv", "问题四-4-3", lambda f, s: normalize_q34(f, s, "问题四-4-3", "Q43_S2：问题三S2配置迁移至附件4波动电价", "波动电价滚动调度账本")),
]


def export_one(filename: str, source_path: Path, problem: str,
               normalizer: Callable[[pd.DataFrame, str], pd.DataFrame]) -> dict:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = source_path.relative_to(ROOT).as_posix()
    frame = pd.read_csv(source_path, low_memory=False)
    normalized = normalizer(frame, source)
    target = OUT / filename
    normalized.to_csv(target, index=False, encoding="utf-8-sig")
    formal = normalized[normalized["评价阶段"].isin(["正式评价期", "正式结果"])]
    numeric = ["计划购电量_千瓦时", "最终有效购电量_千瓦时", "充电量_千瓦时", "放电量_千瓦时", "应急购电量_千瓦时", "总费用_元"]
    totals = {col: float(pd.to_numeric(formal[col], errors="coerce").fillna(0).sum()) for col in numeric}
    return {
        "文件名": filename, "问题编号": problem, "来源": source,
        "来源SHA256": _sha256(source_path), "导出SHA256": _sha256(target),
        "总行数": int(len(normalized)), "正式评价期行数": int(len(formal)),
        "字段数": int(len(normalized.columns)), "正式评价期合计": totals,
    }


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description="导出四问正式统一字段CSV")
    parser.add_argument("--output", type=Path, default=OUT, help="导出目录，默认 C题/正式CSV结果")
    parser.add_argument("--only", choices=["all", "q1", "q2", "q3", "q4"], default="all",
                        help="只导出指定问题；默认导出四问")
    args = parser.parse_args()
    OUT = args.output if args.output.is_absolute() else ROOT / args.output
    OUT.mkdir(parents=True, exist_ok=True)
    selected = {
        "all": EXPORTS,
        "q1": EXPORTS[:1],
        "q2": EXPORTS[1:3],
        "q3": EXPORTS[3:4],
        "q4": EXPORTS[4:],
    }[args.only]
    records = [export_one(*item) for item in selected]
    summary_rows = []
    for item in records:
        totals = item.pop("正式评价期合计")
        summary_rows.append({**item, **totals})
    if args.only == "all":
        pd.DataFrame(summary_rows).to_csv(
            OUT / "四问_正式结果汇总.csv", index=False, encoding="utf-8-sig"
        )
        manifest = {
            "说明": "只读冻结结果导出；未重新训练、求解或修改原始结果",
            "根目录": str(ROOT), "导出目录": str(OUT), "文件": records,
            "统一字段": CANONICAL_COLUMNS,
        }
        (OUT / "四问_正式结果导出清单.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"status": "ok", "output": str(OUT), "files": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
