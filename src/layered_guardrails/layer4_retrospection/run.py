from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from layered_guardrails.layer4_retrospection.accuracy import (
    estimate_accuracy,
    observed_accuracy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])
    input_path = root / "calibration" / "calibrated_confidence.xlsx"
    label_column = config["calibration"]["label_column"]
    rows = []

    workbook = pd.ExcelFile(input_path)
    for measure in workbook.sheet_names:
        frame = pd.read_excel(input_path, sheet_name=measure)
        for method in ("UC", "TS", "VG-GC"):
            estimated = estimate_accuracy(frame[method])
            row = {
                "measure": measure,
                "calibration": method,
                "estimated_accuracy": estimated,
                "num_samples": len(frame),
            }
            if label_column in frame:
                observed = observed_accuracy(frame[label_column])
                row.update(
                    observed_accuracy=observed,
                    absolute_error=abs(estimated - observed),
                )
            rows.append(row)

    output = root / "retrospection"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output / "accuracy_estimation.xlsx", index=False)


if __name__ == "__main__":
    main()
