"""问题一新时间口径 MILP 模型总入口。"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="问题一新时间口径 MILP 模型")
    parser.add_argument("--recompute", action="store_true", help="重新求解后再导出；默认只导出冻结结果")
    args = parser.parse_args()
    if args.recompute:
        subprocess.run([sys.executable, str(ROOT / "code/q1_start_time_recompute.py")], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("q1_q4_csv_processing.py")), "--only", "q1"], check=True)


if __name__ == "__main__":
    main()
