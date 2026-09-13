"""问题四 Q42/Q43_S2 波动电价调度模型总入口。"""

import argparse
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="问题四波动电价调度模型")
    parser.add_argument("--recompute", action="store_true", help="重新求解波动电价模型后再导出；默认只导出冻结结果")
    args = parser.parse_args()
    if args.recompute:
        subprocess.run([
            sys.executable, str(PROJECT_ROOT / "code/q4_price_transfer_experiment.py"), "--mode", "full"
        ], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("q1_q4_csv_processing.py")), "--only", "q4"], check=True)


if __name__ == "__main__":
    main()
