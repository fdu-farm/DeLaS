from __future__ import annotations

import numpy as np
import pandas as pd

from layered_guardrails.scores import CALIBRATION_METHODS


def load_candidates(candidates, label_column="correctness_label"):
    """Align identical target questions; candidate order deterministically breaks ties."""
    if not candidates or len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Provide at least one candidate with a unique model name")
    rows, reference, expected_sheets = [], None, None
    for model, path in candidates:
        with pd.ExcelFile(path) as workbook:
            if expected_sheets is None:
                expected_sheets = workbook.sheet_names
            elif set(workbook.sheet_names) != set(expected_sheets):
                raise ValueError("Candidate workbooks must contain the same confidence measures")
            for measure in expected_sheets:
                frame = pd.read_excel(workbook, sheet_name=measure, dtype={"sample_id": str})
                required = {"sample_id", "question", "answer", "split", *CALIBRATION_METHODS}
                if required - set(frame):
                    raise ValueError(f"{model}/{measure} missing columns: {sorted(required - set(frame))}")
                if frame.empty or frame["sample_id"].isna().any():
                    raise ValueError("Candidate sample IDs must be present and non-empty")
                if frame["sample_id"].duplicated().any():
                    raise ValueError(f"Duplicate sample IDs in {model}/{measure}")
                frame = frame.set_index("sample_id").sort_index()
                identity = frame[["question", "answer", "split"]].fillna("").astype(str)
                if reference is None:
                    reference = identity
                elif not reference.index.equals(identity.index) or not reference.equals(identity):
                    raise ValueError("Candidate sample sets, questions, references or splits differ")
                for method in CALIBRATION_METHODS:
                    confidence = frame[method].to_numpy(dtype=float)
                    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
                        raise ValueError(f"Invalid confidence in {model}/{measure}/{method}")
                    part = frame[["question", "answer", "split"]].reset_index()
                    part["model"], part["measure"], part["calibration"] = model, measure, method
                    part["confidence"] = confidence
                    if label_column in frame:
                        if not frame[label_column].isin([0, 1]).all():
                            raise ValueError("Correctness labels must be binary when provided")
                        part[label_column] = frame[label_column].to_numpy()
                    rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_predictions(predictions):
    return predictions.groupby(["model", "measure", "calibration"], sort=False).agg(
        estimated_accuracy=("confidence", "mean"), num_samples=("sample_id", "size")
    ).reset_index()


def summarize_model(workbook, model_name, methods=CALIBRATION_METHODS):
    predictions = load_candidates([(model_name, workbook)])
    return summarize_predictions(predictions[predictions["calibration"].isin(methods)])


def select_best_model(estimates: pd.DataFrame) -> pd.DataFrame:
    """Select highest mean correctness confidence; first candidate wins ties."""
    required = {"model", "measure", "calibration", "estimated_accuracy"}
    if required - set(estimates):
        raise ValueError(f"Missing model-selection columns: {sorted(required - set(estimates))}")
    indices = estimates.groupby(["measure", "calibration"], sort=False)[
        "estimated_accuracy"
    ].idxmax()
    return estimates.loc[indices].reset_index(drop=True)


def select_per_question(predictions):
    """Select by confidence only; correctness labels are never selection inputs."""
    indices = predictions.groupby(["measure", "calibration", "sample_id"], sort=False)[
        "confidence"
    ].idxmax()
    return predictions.loc[indices].rename(columns={"model": "selected_model"}).reset_index(drop=True)


def question_selection_summary(selected, label_column="correctness_label"):
    rows = []
    for (measure, method), frame in selected.groupby(["measure", "calibration"], sort=False):
        row = {
            "measure": measure, "calibration": method, "num_questions": len(frame),
            "mean_selected_confidence": float(frame["confidence"].mean()),
        }
        if label_column in frame and frame[label_column].notna().all():
            row["selected_accuracy"] = float(frame[label_column].mean())
        rows.append(row)
    return pd.DataFrame(rows)
