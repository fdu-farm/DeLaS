from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from layered_guardrails.layer2_screening.delas_scr import DeLaSScr
from layered_guardrails.layer2_screening.baselines.linear_probe import (
    confidence as linear_confidence,
    fit_linear_probe,
)


def _load_hidden(hidden_dir: Path, sample_id: str) -> np.ndarray:
    import torch

    obj = torch.load(hidden_dir / f"{sample_id}.pt", map_location="cpu", weights_only=True)
    hs = obj["clean"]
    if isinstance(hs, torch.Tensor):
        hs = hs.detach().float().cpu().numpy()
    return np.asarray(hs, dtype=np.float32).ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    parser.add_argument(
        "--labels",
        default=None,
        help="CSV with sample_id and hallucination_label; defaults to labeling/labels.csv under the configured output root.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])

    generation = pd.read_csv(root / "generation" / "responses_and_features.csv", dtype={"sample_id": str})
    baselines = pd.read_csv(root / "screening" / "baseline_scores.csv", dtype={"sample_id": str})
    labels_path = Path(args.labels) if args.labels else root / "labeling" / "labels.csv"
    labels = pd.read_csv(labels_path, dtype={"sample_id": str})
    for frame in (generation, baselines, labels):
        frame["sample_id"] = frame["sample_id"].astype(str)
    label_column = config["screening"]["label_column"]
    data = generation.merge(
        baselines, on=["sample_id", "split"], validate="one_to_one"
    ).merge(
        labels[["sample_id", label_column]], on="sample_id", validate="one_to_one"
    )

    hidden_dir = root / "generation" / "hidden_states"
    data["_hs"] = data["sample_id"].apply(lambda sid: _load_hidden(hidden_dir, sid))

    train_mask = data["split"] == config["screening"]["train_split"]
    val_mask = data["split"] == config["screening"]["val_split"]

    if len(data) != len(generation):
        raise ValueError("Every generated sample requires matching baseline scores and labels")
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Screening requires non-empty training and validation splits")
    split_names = [config["screening"][key] for key in ("train_split", "val_split", "test_split")]
    if len(set(split_names)) != 3:
        raise ValueError("Training, validation and test splits must differ")
    if not data[label_column].isin([0, 1]).all():
        raise ValueError("Hallucination labels must be binary")

    def stack_hs(mask):
        return np.stack(data.loc[mask, "_hs"].to_list())

    model = DeLaSScr().fit(
        data.loc[train_mask, "Delta"], stack_hs(train_mask), data.loc[train_mask, label_column],
        data.loc[val_mask, "Delta"], stack_hs(val_mask), data.loc[val_mask, label_column],
    )
    all_hs = np.stack(data["_hs"].to_list())
    data["DeLaS-Scr"] = model.confidence(data["Delta"], all_hs)

    linear_probe = fit_linear_probe(stack_hs(train_mask), data.loc[train_mask, label_column])
    data["LinearProbe"] = linear_confidence(linear_probe, all_hs)

    output = root / "screening"
    model.save(output / "delas_scr.joblib")
    joblib.dump(linear_probe, output / "linear_probe.joblib")
    # Preserve question and subgroup metadata for downstream selection and review.
    data.drop(columns="_hs").to_csv(output / "screening_scores.csv", index=False)


if __name__ == "__main__":
    main()
