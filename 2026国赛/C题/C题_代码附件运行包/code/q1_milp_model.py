"""问题一便携式 MILP 入口。

只读取本包 ``附件/附件1.xlsx``，直接求解 144 个十分钟时段，
并在包根目录自动生成 ``result/`` 和 ``figure/``。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKAGE_ROOT / "code"
INPUT = PACKAGE_ROOT / "附件" / "附件1.xlsx"
RESULT = PACKAGE_ROOT / "result"
FIGURE = PACKAGE_ROOT / "figure"


def load_solver():
    spec = importlib.util.spec_from_file_location("q1_solver_core", CODE_ROOT / "02_q1_baseline.py")
    if spec is None or spec.loader is None:
        raise ImportError("无法加载问题一 MILP 内核")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_attachment():
    workbook = openpyxl.load_workbook(INPUT, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.values)
    finally:
        workbook.close()
    if len(rows) != 145 or any(len(row) < 4 for row in rows[1:]):
        raise ValueError("附件1应包含表头和144个数据时段")
    values = np.asarray([row[1:4] for row in rows[1:]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("附件1含非数值或缺失数据")
    return values[:, 0], values[:, 1], values[:, 2]


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    price, load, pv = read_attachment()
    solver = load_solver()
    (grid, charge, discharge, unused, state), summary = solver.solve(
        load, pv, price, integer=True, initial_kWh=6000.0, terminal_kWh=6000.0
    )
    intervals = [
        f"{i // 6:02d}:{(i % 6) * 10:02d}-{((i + 1) // 6):02d}:{((i + 1) % 6) * 10:02d}"
        for i in range(144)
    ]
    frame = pd.DataFrame(
        {
            "interval": intervals,
            "price_yuan_kWh": price,
            "load_kW": load,
            "pv_kW": pv,
            "grid_kWh": grid,
            "charge_kWh": charge,
            "discharge_kWh": discharge,
            "unused_kWh": unused,
            "state_start_kWh": state[:-1],
            "state_end_kWh": state[1:],
            "cost_yuan": price * grid,
        }
    )
    frame.to_csv(RESULT / "q1_milp_dispatch.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "model": "Q1 deterministic MILP",
                "cost_yuan": summary["cost_yuan"],
                "grid_kWh": summary["grid_kWh"],
                "charge_kWh": summary["charge_kWh"],
                "discharge_kWh": summary["discharge_kWh"],
                "initial_kWh": summary["initial_kWh"],
                "final_kWh": summary["final_kWh"],
                "solver": summary["solver"],
            }
        ]
    ).to_csv(RESULT / "q1_milp_summary.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    ax.plot(np.arange(145) / 6, state, color="#145da0", linewidth=1.8)
    ax.set(xlabel="Time (h)", ylabel="Battery state (kWh)", title="Q1 MILP battery-state trajectory")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE / "q1_milp_soc_trajectory.png")
    plt.close(fig)
    print(
        f"完成：{RESULT / 'q1_milp_dispatch.csv'}；{FIGURE / 'q1_milp_soc_trajectory.png'}；"
        f"购电费 {summary['cost_yuan']:.6f} 元"
    )


if __name__ == "__main__":
    main()
