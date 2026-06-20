from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from layered_guardrails.layer2_screening.baselines.consistency import (
    EntailmentClusterer,
    consistency_baselines,
)
from layered_guardrails.layer2_screening.baselines.logits import (
    logits_baselines,
    sequence_statistics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["paths"]["output_root"])
    generation = root / "generation"
    output = root / "screening"
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(generation / "responses_and_features.csv")
    frame["sample_id"] = frame["sample_id"].astype(str)
    sampled = pd.read_json(generation / "sampled_responses.jsonl", lines=True)
    sampled["sample_id"] = sampled["sample_id"].astype(str)
    clusterer = EntailmentClusterer(config["screening"]["entailment_model"], config["device"])
    rows = []

    for row in tqdm(frame.to_dict("records"), desc="Screening baselines"):
        sample_id = str(row["sample_id"])
        greedy = np.load(generation / "logits" / f"{sample_id}.npz")
        scores = logits_baselines(greedy["logits"], greedy["token_ids"])
        sample_arrays = np.load(
            generation / "sampled_logits" / f"{sample_id}.npz", allow_pickle=True
        )
        sample_rows = sampled[sampled["sample_id"].astype(str) == sample_id]
        clean_responses = sample_rows[sample_rows["condition"] == "clean"].sort_values("sample_index")["response"].tolist()
        noisy_responses = sample_rows[sample_rows["condition"] == "intervened"].sort_values("sample_index")["response"].tolist()
        scores.update(
            consistency_baselines(
                list(sample_arrays["clean"]),
                list(sample_arrays["intervened"]),
                row["response"],
                clean_responses,
                noisy_responses,
                row["question"],
                clusterer,
                sequence_statistics,
            )
        )
        rows.append({"sample_id": sample_id, "split": row["split"], **scores})
        pd.DataFrame(rows).to_csv(output / "baseline_scores.csv", index=False)


if __name__ == "__main__":
    main()
