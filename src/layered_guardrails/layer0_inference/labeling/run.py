from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from layered_guardrails.layer0_inference.labeling.prompt import (
    build_evaluation_text,
    parse_correctness_label,
)


PENDING = "pending"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label generated VQA responses with a user-selected model through an OpenAI-compatible endpoint."
    )
    parser.add_argument("--config", default="configs/medgemma_vqa_rad.yaml")
    parser.add_argument("--input", default=None, help="Generation CSV; defaults to config output path.")
    parser.add_argument("--output", default=None, help="Label CSV; defaults to labeling/labels.csv under the configured output root.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None, help="Labeling model; overrides labeling.model.")
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def completed(value) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "nan", PENDING}


def build_client(api_key: str, base_url: str):
    if not api_key:
        raise ValueError(
            "Missing API key. Set DASHSCOPE_API_KEY/OPENAI_API_KEY or pass --api-key."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def initialize_output(input_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        frame = pd.read_csv(output_path, dtype={"sample_id": str})
    else:
        source = pd.read_csv(input_path, dtype={"sample_id": str})
        required = {"sample_id", "question", "answer", "response"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
        frame = source[["sample_id", "split", "question", "answer", "response"]].copy()
        frame["correctness_label"] = PENDING
        frame["hallucination_label"] = PENDING
        frame["evaluator_reason"] = PENDING
    frame["sample_id"] = frame["sample_id"].astype(str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def request_label(client, model: str, row, max_retries: int) -> tuple[int, str]:
    content = build_evaluation_text(row["question"], row["answer"], row["response"])
    error = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You precisely evaluate medical VQA response correctness.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=0,
            )
            text = completion.choices[0].message.content or ""
            label = parse_correctness_label(text)
            if label is not None:
                return label, text
            error = ValueError(f"Could not parse evaluator response: {text!r}")
        except Exception as exc:
            error = exc
        if attempt + 1 < max_retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"Evaluator failed after {max_retries} attempts") from error


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_root = Path(config["paths"]["output_root"])
    label_config = config.get("labeling", {})
    input_path = Path(args.input) if args.input else output_root / "generation" / "responses_and_features.csv"
    output_path = Path(args.output) if args.output else output_root / "labeling" / "labels.csv"
    api_key = (
        args.api_key
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = args.base_url or label_config.get(
        "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = args.model if args.model is not None else label_config.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Specify a labeling model in labeling.model or with --model.")
    model = model.strip()
    client = build_client(api_key, base_url)
    frame = initialize_output(input_path, output_path)
    # Pending strings and completed integer labels share these columns.
    for column in ("correctness_label", "hallucination_label"):
        frame[column] = frame[column].astype(object)

    for index, row in tqdm(frame.iterrows(), total=len(frame), desc=f"Labeling with {model}"):
        if completed(row.get("correctness_label")):
            continue
        correctness, reason = request_label(client, model, row, args.max_retries)
        frame.at[index, "correctness_label"] = correctness
        frame.at[index, "hallucination_label"] = 1 - correctness
        frame.at[index, "evaluator_reason"] = reason
        frame.to_csv(output_path, index=False)

    print(f"Saved labels to {output_path}")


if __name__ == "__main__":
    main()
