from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from layered_guardrails.layer3_calibration.temperature import (
    apply_temperature,
    fit_temperature,
)
from layered_guardrails.layer3_calibration.vggc import (
    apply_vggc,
    fit_vggc,
    normalize_grounding,
)


UNCERTAINTY_MEASURES = {"AvgEnt", "MaxEnt", "SEnt", "SEne", "VASE"}
GROUNDING_COLUMNS = ["VAS", "VAC", "JN"]


def as_confidence(values, measure):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if measure in UNCERTAINTY_MEASURES:
        values = np.exp(-np.clip(values, 0.0, 100.0))
    return np.clip(values, 1e-12, 1.0 - 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    parser.add_argument(
        "--labels",
        default=None,
        help="CSV with sample_id and correctness_label; defaults to outputs/labeling/labels.csv.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])
    screening = pd.read_csv(root / "screening" / "screening_scores.csv")
    grounding = pd.read_csv(root / "generation" / "responses_and_features.csv")[
        ["sample_id", *GROUNDING_COLUMNS]
    ]
    labels_path = Path(args.labels) if args.labels else root / "labeling" / "labels.csv"
    labels = pd.read_csv(labels_path)
    for frame in (screening, grounding, labels):
        frame["sample_id"] = frame["sample_id"].astype(str)
    label_column = config["calibration"]["label_column"]
    data = screening.merge(grounding, on="sample_id").merge(
        labels[["sample_id", label_column]], on="sample_id"
    )
    validation_mask = data["split"] == config["calibration"]["validation_split"]
    test_mask = data["split"] == config["calibration"]["test_split"]
    if not validation_mask.any() or not test_mask.any():
        raise ValueError("Calibration requires non-empty validation and test splits.")

    output = root / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    workbook = output / "calibrated_confidence.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for measure in config["calibration"]["measures"]:
            validation_confidence = as_confidence(data.loc[validation_mask, measure], measure)
            test_confidence = as_confidence(data.loc[test_mask, measure], measure)
            validation_labels = data.loc[validation_mask, label_column].to_numpy(dtype=int)
            validation_grounding = data.loc[validation_mask, GROUNDING_COLUMNS].to_numpy(dtype=float)
            test_grounding = data.loc[test_mask, GROUNDING_COLUMNS].to_numpy(dtype=float)
            validation_grounding, test_grounding = normalize_grounding(
                validation_grounding, test_grounding
            )

            temperature = fit_temperature(validation_confidence, validation_labels)
            parameters = fit_vggc(
                validation_confidence, validation_labels, validation_grounding
            )
            result = data.loc[
                test_mask, ["sample_id", "split", label_column, *GROUNDING_COLUMNS]
            ].copy()
            result["UC"] = test_confidence
            result["TS"] = apply_temperature(test_confidence, temperature)
            result["VG-GC"] = apply_vggc(test_confidence, test_grounding, parameters)
            result.to_excel(writer, sheet_name=measure[:31], index=False)


if __name__ == "__main__":
    main()
