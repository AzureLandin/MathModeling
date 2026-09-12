"""Inspect the price workbook without fitting models or changing inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "\u9644\u4ef6" / "\u9644\u4ef64.xlsx"
OUTPUT = ROOT / "results" / "q4_price_initial_inspection.json"


def main():
    workbook = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    sheets = []
    for sheet in workbook:
        rows = list(sheet.iter_rows(values_only=True))
        values = np.asarray([row[1:] for row in rows[1:]], dtype=float)
        dates = [row[0] for row in rows[1:]]
        sheets.append({
            "name": sheet.title,
            "shape": list(values.shape),
            "first_date": str(dates[0]),
            "last_date": str(dates[-1]),
            "unique_dates": len(set(dates)),
            "first_headers": [str(x) for x in rows[0][:4]],
            "last_headers": [str(x) for x in rows[0][-3:]],
            "nonfinite_count": int((~np.isfinite(values)).sum()),
            "nonpositive_count": int((values <= 0).sum()),
            "minimum_yuan_per_kWh": float(np.nanmin(values)),
            "maximum_yuan_per_kWh": float(np.nanmax(values)),
            "mean_yuan_per_kWh": float(np.nanmean(values)),
            "std_yuan_per_kWh": float(np.nanstd(values)),
            "adjacent_source_rows_same_slot_mae": float(np.nanmean(np.abs(np.diff(values, axis=0)))),
            "unique_daily_curves": int(np.unique(values, axis=0).shape[0]),
        })
    workbook.close()
    result = {
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "openpyxl_version": openpyxl.__version__,
        "scope": "Descriptive inspection only; no forecast backtest or scheduling.",
        "sheets": sheets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
