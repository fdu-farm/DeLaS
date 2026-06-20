from __future__ import annotations

import pandas as pd

from layered_guardrails.layer1_selection.accuracy import estimate_accuracy


def summarize_model(
    workbook: str,
    model_name: str,
    methods: tuple[str, ...] = ("UC", "TS", "VG-GC"),
) -> pd.DataFrame:
    """Estimate model accuracy for every measure and calibration method."""
    rows = []
    excel = pd.ExcelFile(workbook)
    for measure in excel.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=measure)
        for method in methods:
            if method not in frame:
                continue
            rows.append(
                {
                    "model": model_name,
                    "measure": measure,
                    "calibration": method,
                    "estimated_accuracy": estimate_accuracy(frame[method]),
                    "num_samples": len(frame),
                }
            )
    return pd.DataFrame(rows)


def select_best_model(estimates: pd.DataFrame) -> pd.DataFrame:
    """Select the model with highest estimated accuracy for each strategy."""
    required = {"model", "measure", "calibration", "estimated_accuracy"}
    missing = required - set(estimates.columns)
    if missing:
        raise ValueError(f"Missing model-selection columns: {sorted(missing)}")
    indices = estimates.groupby(["measure", "calibration"])[
        "estimated_accuracy"
    ].idxmax()
    return estimates.loc[indices].sort_values(
        ["measure", "calibration"]
    ).reset_index(drop=True)
