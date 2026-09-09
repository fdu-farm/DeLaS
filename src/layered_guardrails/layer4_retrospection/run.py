from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import yaml

from layered_guardrails.layer4_retrospection.accuracy import estimate_accuracy, observed_accuracy
from layered_guardrails.scores import CALIBRATION_METHODS


def summarize_groups(frame, measure, group_by, label_column="correctness_label"):
    """Overall estimates and separate summaries for each available metadata field."""
    groups = [("overall", "all", frame)]
    for field in dict.fromkeys(group_by):
        if field not in frame or frame[field].isna().all():
            warnings.warn(f"Subgroup field {field!r} is unavailable; skipping", stacklevel=2)
            continue
        for value, part in frame.groupby(field, dropna=False, sort=False):
            groups.append((field, "(missing)" if pd.isna(value) else value, part))
    rows = []
    for field, value, part in groups:
        for method in CALIBRATION_METHODS:
            estimated = estimate_accuracy(part[method])
            row = {
                "measure": measure, "calibration": method,
                "group_field": field, "group_value": value,
                "estimated_accuracy": estimated, "num_samples": len(part),
            }
            if label_column in part:
                if not part[label_column].isin([0, 1]).all():
                    raise ValueError("Correctness labels must be binary when provided")
                observed = observed_accuracy(part[label_column])
                row.update(observed_accuracy=observed, absolute_error=abs(estimated - observed))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])
    label_column = config["calibration"]["label_column"]
    group_by = config.get("retrospection", {}).get("group_by", [])
    rows = []
    with pd.ExcelFile(root / "calibration" / "calibrated_confidence.xlsx") as workbook:
        for measure in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=measure)
            rows.append(summarize_groups(frame, measure, group_by, label_column))
    output = root / "retrospection"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_excel(output / "accuracy_estimation.xlsx", index=False)


if __name__ == "__main__":
    main()
