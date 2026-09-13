"""问题三正式结果入口：导出 S2 滚动调度账本。"""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    exporter = Path(__file__).with_name("正式结果CSV导出.py")
    sys.argv = [str(exporter), "--only", "q3"] + sys.argv[1:]
    runpy.run_path(str(exporter), run_name="__main__")
