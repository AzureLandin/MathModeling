# -*- coding: utf-8 -*-
"""
problem1_main.py
问题一主入口：子任务一（分布规律）+ 子任务二（岭回归）+ 情景预测。

用法：
    python problem1_main.py

依赖见同目录 requirements.txt。岭回归为闭式解，不强制 scikit-learn。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problem1_step1_distribution import main as run_distribution
from problem1_step2_prediction import main as run_prediction
from problem1_step2_scenario import main as run_scenario
from problem1_utils import RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, SUMMARY_XLSX, ensure_dirs


def main() -> None:
    ensure_dirs()
    print("问题一主程序启动")
    print(f"结果目录: {RESULTS_DIR}")
    print(f"图件目录: {FIGURES_DIR}")
    print(f"报告目录: {REPORTS_DIR}")
    run_distribution()
    run_prediction()
    run_scenario()
    print("=" * 64)
    print("问题一全部完成")
    print(f"汇总工作簿: {SUMMARY_XLSX}")
    print("=" * 64)


if __name__ == "__main__":
    main()
