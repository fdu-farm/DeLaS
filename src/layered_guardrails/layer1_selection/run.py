from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from layered_guardrails.layer1_selection.model_selection import (
    select_best_model,
    summarize_model,
)


def parse_model_workbook(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use MODEL=PATH for --model-workbook.")
    model, path = value.split("=", 1)
    if not model or not path:
        raise argparse.ArgumentTypeError("Use MODEL=PATH for --model-workbook.")
    return model, path


def main():
    parser = argparse.ArgumentParser(
        description="Estimate candidate-model accuracy and select the best model."
    )
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    parser.add_argument(
        "--model-workbook",
        action="append",
        type=parse_model_workbook,
        help="Candidate in MODEL=PATH format. Repeat for multiple models.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(config["paths"]["output_root"])
    candidates = args.model_workbook or [
        (
            "medgemma",
            str(output_root / "calibration" / "calibrated_confidence.xlsx"),
        )
    ]

    estimates = pd.concat(
        [
            summarize_model(workbook=path, model_name=model)
            for model, path in candidates
        ],
        ignore_index=True,
    )
    selected = select_best_model(estimates)
    output = output_root / "layer1_selection"
    output.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output / "model_selection.xlsx", engine="openpyxl") as writer:
        estimates.to_excel(writer, sheet_name="estimates", index=False)
        selected.to_excel(writer, sheet_name="selected_models", index=False)


if __name__ == "__main__":
    main()
