from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from layered_guardrails.evaluation.metrics import ace, auroc, ece, safe_rate


UNCERTAINTY_MEASURES = {"AvgEnt", "MaxEnt", "SEnt", "SEne", "VASE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])
    path = root / "calibration" / "calibrated_confidence.xlsx"
    label_column = config["calibration"]["label_column"]
    calibration_rows = []

    workbook = pd.ExcelFile(path)
    for measure in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=measure)
        labels = frame[label_column]
        for method in ("UC", "TS", "VG-GC"):
            confidence = frame[method]
            calibration_rows.append(
                {
                    "measure": measure,
                    "calibration": method,
                    "ECE": ece(labels, confidence),
                    "ACE": ace(labels, confidence),
                    "SafeRate@0.5": safe_rate(labels, confidence, 0.5),
                    "SafeRate@0.6": safe_rate(labels, confidence, 0.6),
                    "SafeRate@0.7": safe_rate(labels, confidence, 0.7),
                }
            )
    screening = pd.read_csv(root / "screening" / "screening_scores.csv")
    hallucination_column = config["screening"]["label_column"]
    screening_rows = []
    for measure in config["calibration"]["measures"]:
        values = screening[measure]
        risk = values if measure in UNCERTAINTY_MEASURES else 1.0 - values
        screening_rows.append(
            {
                "measure": measure,
                "AUROC": auroc(screening[hallucination_column], risk),
            }
        )

    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output / "results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(screening_rows).to_excel(writer, sheet_name="screening", index=False)
        pd.DataFrame(calibration_rows).to_excel(writer, sheet_name="calibration", index=False)


if __name__ == "__main__":
    main()
