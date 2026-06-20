from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from layered_guardrails.layer0_input.generation.data import load_vqa_rad
from layered_guardrails.layer0_input.generation.features import (
    compute_delta,
    vision_intervention,
)
from layered_guardrails.layer0_input.generation.medgemma import MedGemmaRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output = Path(config["paths"]["output_root"]) / "generation"
    logits_dir, hidden_dir, sampled_dir = output / "logits", output / "hidden_states", output / "sampled_logits"
    for directory in (output, logits_dir, hidden_dir, sampled_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runner = MedGemmaRunner(
        config["model"]["path"], config["device"], config["model"].get("torch_dtype", "bfloat16")
    )
    rows, sampled_rows = [], []
    data = load_vqa_rad(
        config["dataset"]["huggingface_id"],
        config["dataset"]["split"],
        config["dataset"].get("max_samples"),
        config["dataset"].get("validation_fraction", 0.2),
        seed,
    )
    gen_cfg = config["generation"]
    intervention_cfg = gen_cfg["intervention"]

    for item in tqdm(data, desc="MedGemma | VQA-RAD"):
        sample_id = item["sample_id"]
        clean = runner.generate(
            item["question"], item["image"], config["model"]["max_new_tokens"], grounding=True
        )
        stable_id = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed + stable_id)
        changed_image = vision_intervention(item["image"], generator=generator, **intervention_cfg)
        intervened = runner.generate(
            item["question"], changed_image, config["model"]["max_new_tokens"], grounding=False
        )
        delta = compute_delta(clean.hidden_state, intervened.hidden_state)

        np.savez_compressed(logits_dir / f"{sample_id}.npz", logits=clean.logits, token_ids=clean.token_ids)
        torch.save(
            {"clean": clean.hidden_state, "intervened": intervened.hidden_state},
            hidden_dir / f"{sample_id}.pt",
        )
        rows.append(
            {
                "sample_id": sample_id,
                "split": item["split"],
                "question": item["question"],
                "answer": item["answer"],
                "response": clean.response,
                "intervened_response": intervened.response,
                "VAS": clean.VAS,
                "VAC": clean.VAC,
                "JN": clean.JN,
                "Delta": delta,
            }
        )

        sample_logits, noisy_sample_logits = [], []
        for sample_index in range(int(gen_cfg["num_consistency_samples"])):
            sampled_changed_image = vision_intervention(
                item["image"], generator=generator, **intervention_cfg
            )
            sampled = runner.generate(
                item["question"],
                item["image"],
                config["model"]["max_new_tokens"],
                sample=True,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
            )
            noisy_sampled = runner.generate(
                item["question"],
                sampled_changed_image,
                config["model"]["max_new_tokens"],
                sample=True,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
            )
            sample_logits.append(sampled.logits)
            noisy_sample_logits.append(noisy_sampled.logits)
            sampled_rows.extend(
                [
                    {"sample_id": sample_id, "sample_index": sample_index, "condition": "clean", "response": sampled.response},
                    {"sample_id": sample_id, "sample_index": sample_index, "condition": "intervened", "response": noisy_sampled.response},
                ]
            )
        np.savez_compressed(
            sampled_dir / f"{sample_id}.npz",
            clean=np.array(sample_logits, dtype=object),
            intervened=np.array(noisy_sample_logits, dtype=object),
        )

        pd.DataFrame(rows).to_csv(output / "responses_and_features.csv", index=False)
        with (output / "sampled_responses.jsonl").open("w", encoding="utf-8") as handle:
            for row in sampled_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
