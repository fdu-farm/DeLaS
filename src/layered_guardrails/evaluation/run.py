from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from layered_guardrails.evaluation.metrics import ace, auroc, ece, safe_rate
from layered_guardrails.scores import CALIBRATION_METHODS, hallucination_risk


def test_screening_scores(screening, split, measures, label_column):
    frame = screening.loc[screening["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"Screening test split {split!r} is empty")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Screening test sample IDs must be present and unique")
    if not frame[label_column].isin([0, 1]).all():
        raise ValueError("Hallucination labels must be binary")
    return pd.DataFrame([
        {
            "measure": measure, "split": split, "num_samples": len(frame),
            "AUROC": auroc(frame[label_column], hallucination_risk(frame[measure], measure)),
        }
        for measure in measures
    ])


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
        frame = frame.loc[frame["split"] == config["calibration"]["test_split"]]
        if frame.empty:
            raise ValueError(f"Calibration test split is empty for {measure}")
        labels = frame[label_column]
        for method in CALIBRATION_METHODS:
            confidence = frame[method]
            calibration_rows.append(
                {
                    "measure": measure,
                    "calibration": method,
                    "num_samples": len(frame),
                    "ECE": ece(labels, confidence),
                    "ACE": ace(labels, confidence),
                    "SafeRate@0.5": safe_rate(labels, confidence, 0.5),
                    "SafeRate@0.6": safe_rate(labels, confidence, 0.6),
                    "SafeRate@0.7": safe_rate(labels, confidence, 0.7),
                }
            )
    screening = pd.read_csv(root / "screening" / "screening_scores.csv")
    screening_rows = test_screening_scores(
        screening, config["screening"]["test_split"],
        config["calibration"]["measures"], config["screening"]["label_column"],
    )

    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output / "results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(screening_rows).to_excel(writer, sheet_name="screening", index=False)
        pd.DataFrame(calibration_rows).to_excel(writer, sheet_name="calibration", index=False)


if __name__ == "__main__":
    main()
