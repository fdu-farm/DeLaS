from __future__ import annotations

from typing import Any



def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"None of the fields {names} exist in dataset row: {list(row)}")


def load_vqa_rad(
    dataset_id: str,
    split: str | None,
    max_samples: int | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
    metadata_fields=("question_type", "answer_type", "modality", "organ"),
):
    """Load VQA-RAD from Hugging Face and normalize its common field names."""
    from datasets import load_dataset

    loaded = load_dataset(dataset_id, split=split)
    datasets = {split: loaded} if split is not None else dict(loaded)
    if split is None and "validation" not in datasets and "train" in datasets:
        divided = datasets["train"].train_test_split(
            test_size=validation_fraction, seed=seed, shuffle=True
        )
        datasets["train"] = divided["train"]
        datasets["validation"] = divided["test"]
    for split_name, dataset in datasets.items():
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        for index, row in enumerate(dataset):
            metadata = {field: row[field] for field in metadata_fields if field in row}
            yield {
                **metadata,
                "sample_id": str(row.get("q_idx", row.get("id", f"{split_name}-{index}"))),
                "image": _first(row, ("image", "img")),
                "question": str(_first(row, ("question", "question_text"))),
                "answer": str(_first(row, ("answer", "answers"))),
                "split": str(row.get("split", split_name)),
            }


def build_question_prompt(question: str) -> str:
    return (
        "Given this medical image, provide a very short, definitive, and concise "
        "answer. If possible, answer with a single word or short phrase.\n"
        f"Question: {question}\n"
    )
