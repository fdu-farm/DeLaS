from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from layered_guardrails.layer3_calibration.temperature import apply_temperature, fit_temperature
from layered_guardrails.layer3_calibration.delas_cal import (
    apply_delas_cal,
    fit_delas_cal,
    normalize_grounding,
)
from layered_guardrails.layer3_calibration.routing import routing_tables
from layered_guardrails.scores import as_confidence_pair

GROUNDING_COLUMNS = ["VAS", "VAC", "JN"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    parser.add_argument("--labels", default=None, help="CSV with sample_id and correctness_label.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])
    screening = pd.read_csv(root / "screening" / "screening_scores.csv", dtype={"sample_id": str})
    labels_path = Path(args.labels) if args.labels else root / "labeling" / "labels.csv"
    labels = pd.read_csv(labels_path, dtype={"sample_id": str})
    label_column = config["calibration"]["label_column"]
    data = screening.drop(columns=[label_column], errors="ignore").merge(
        labels[["sample_id", label_column]], on="sample_id", validate="one_to_one"
    )
    if len(data) != len(screening):
        raise ValueError("Every screening sample requires a correctness label")
    validation_split = config["calibration"]["validation_split"]
    test_split = config["calibration"]["test_split"]
    if validation_split == test_split:
        raise ValueError("Calibration validation and test splits must differ")
    validation_mask = data["split"] == validation_split
    test_mask = data["split"] == test_split
    if not validation_mask.any() or not test_mask.any():
        raise ValueError("Calibration requires non-empty validation and test splits")
    if not data[label_column].isin([0, 1]).all():
        raise ValueError("Correctness labels must be binary")

    validation_grounding, test_grounding = normalize_grounding(
        data.loc[validation_mask, GROUNDING_COLUMNS].to_numpy(dtype=float),
        data.loc[test_mask, GROUNDING_COLUMNS].to_numpy(dtype=float),
    )
    output = root / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    samples, summaries = [], []
    with pd.ExcelWriter(output / "calibrated_confidence.xlsx", engine="openpyxl") as writer:
        for measure in config["calibration"]["measures"]:
            validation_confidence, test_confidence = as_confidence_pair(
                data.loc[validation_mask, measure], data.loc[test_mask, measure], measure
            )
            validation_labels = data.loc[validation_mask, label_column].to_numpy(dtype=int)
            temperature = fit_temperature(validation_confidence, validation_labels)
            parameters = fit_delas_cal(
                validation_confidence, validation_labels, validation_grounding
            )
            # Keep questions, references and all available subgroup metadata.
            result = data.loc[test_mask].copy()
            result["UC"] = test_confidence
            result["TS"] = apply_temperature(test_confidence, temperature)
            result["DeLaS-Cal"] = apply_delas_cal(test_confidence, test_grounding, parameters)
            result.to_excel(writer, sheet_name=measure, index=False)
            routed, summary = routing_tables(
                result, measure, config.get("routing", {}).get("policies"), label_column
            )
            samples.append(routed)
            summaries.append(summary)
    with pd.ExcelWriter(output / "routing.xlsx", engine="openpyxl") as writer:
        pd.concat(samples, ignore_index=True).to_excel(writer, sheet_name="samples", index=False)
        pd.concat(summaries, ignore_index=True).to_excel(writer, sheet_name="summary", index=False)


if __name__ == "__main__":
    main()
