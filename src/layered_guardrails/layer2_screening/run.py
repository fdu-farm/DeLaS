from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from layered_guardrails.layer2_screening.vip import ViP


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

    train = data["split"] == config["screening"]["train_split"]
    model = ViP().fit(data.loc[train, "Delta"], data.loc[train, label_column])
    data["ViP"] = model.confidence(data["Delta"])
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
