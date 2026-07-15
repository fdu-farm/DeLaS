from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from layered_guardrails.layer2_screening.vip import ViP


def _load_hidden(hidden_dir: Path, sample_id: str) -> np.ndarray:
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
        help="CSV with sample_id and hallucination_label; defaults to outputs/labeling/labels.csv.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])

    generation = pd.read_csv(root / "generation" / "responses_and_features.csv")
    baselines = pd.read_csv(root / "screening" / "baseline_scores.csv")
    labels_path = Path(args.labels) if args.labels else root / "labeling" / "labels.csv"
    labels = pd.read_csv(labels_path)
    for frame in (generation, baselines, labels):
        frame["sample_id"] = frame["sample_id"].astype(str)
    label_column = config["screening"]["label_column"]
    data = generation.merge(baselines, on=["sample_id", "split"]).merge(
        labels[["sample_id", label_column]], on="sample_id"
    )

    hidden_dir = root / "generation" / "hidden_states"
    data["_hs"] = data["sample_id"].apply(lambda sid: _load_hidden(hidden_dir, sid))

    train_mask = data["split"] == config["screening"]["train_split"]
    val_mask = data["split"] == config["screening"]["val_split"]

    def stack_hs(mask):
        return np.stack(data.loc[mask, "_hs"].to_list())

    model = ViP().fit(
        data.loc[train_mask, "Delta"], stack_hs(train_mask), data.loc[train_mask, label_column],
        data.loc[val_mask, "Delta"], stack_hs(val_mask), data.loc[val_mask, label_column],
    )
    all_hs = np.stack(data["_hs"].to_list())
    data["ViP"] = model.confidence(data["Delta"], all_hs)

    output = root / "screening"
    model.save(output / "vip.joblib")
    columns = [
        "sample_id", "split", label_column, "ViP", "AvgProb", "MaxProb",
        "AvgEnt", "MaxEnt", "SEnt", "SEne", "VASE", "RadFlag",
    ]
    data[columns].to_csv(output / "screening_scores.csv", index=False)
    (output / "baseline_scores.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
