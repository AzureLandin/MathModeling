"""问题二 LightGBM 预测与日前调度模型总入口。"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="问题二 LightGBM 预测与日前调度模型")
    parser.add_argument("--recompute", action="store_true", help="重新训练和求解后再导出；默认只导出冻结结果")
    args = parser.parse_args()
    if args.recompute:
        subprocess.run([
            sys.executable, str(ROOT / "code/29_q2_time_mapping_experiment.py"), "--mode", "full"
        ], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("q1_q4_csv_processing.py")), "--only", "q2"], check=True)


if __name__ == "__main__":
    main()
