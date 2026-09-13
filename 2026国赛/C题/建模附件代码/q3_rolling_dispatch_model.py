"""问题三 S2 日内修正与滚动调度模型总入口。"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="问题三 S2 日内修正与滚动调度模型")
    parser.add_argument("--recompute", action="store_true", help="重新求解全年滚动模型后再导出；默认只导出冻结结果")
    args = parser.parse_args()
    if args.recompute:
        subprocess.run([
            sys.executable, str(ROOT / "code/q3_intraday_load_cost_experiment.py"), "--mode", "full"
        ], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("q1_q4_csv_processing.py")), "--only", "q3"], check=True)


if __name__ == "__main__":
    main()
