import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = ["prepare_reference_data.py", "q1_analysis.py", "q2_analysis.py", "q3_allocation.py", "q4_matching.py"]


def main():
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        subprocess.run([sys.executable, str(HERE / script)], check=True)


if __name__ == "__main__":
    main()
