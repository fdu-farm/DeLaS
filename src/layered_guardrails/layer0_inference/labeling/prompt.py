from __future__ import annotations

import re


EVALUATION_PROMPT = """ROLE: You are an impartial medical Question-Answer evaluator.

INPUTS:
You will receive a medical-imaging question, its reference answer, and a model response.

TASK:
Decide whether the model response is semantically correct according to the reference answer.
Judge the final conclusion rather than intermediate reasoning.

SCORING:
Correct response: 1
Incorrect, contradictory, unsupported, incomplete, or uncertain response: 0

GUIDELINES:
1. Judge only against the supplied reference answer.
2. Accept semantic equivalents, standard medical synonyms, abbreviations, and spelling variants.
3. Laterality, location, stage, modality, and other categorical attributes must match.
4. Numeric answers must match after reasonable unit normalization.
5. Multi-label responses must contain all required findings and no contradictory finding.
6. A hedged response is incorrect when the reference answer is definite.
7. Ignore formatting, tone, and non-conflicting disclaimers.

OUTPUT FORMAT:
Line 1: only 1 or 0.
Line 2: one brief sentence explaining the decision.

Do not use Markdown or code fences."""


def build_evaluation_text(question: str, reference_answer: str, response: str) -> str:
    return (
        f"{EVALUATION_PROMPT}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Reference Answer]\n{reference_answer}\n\n"
        f"[Model Response]\n{response}\n"
    )


def parse_correctness_label(text: str) -> int | None:
    """Parse the first standalone 0/1 decision from an evaluator response."""
    if not isinstance(text, str):
        return None
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    for line in lines:
        if re.fullmatch(r"[01]", line):
            return int(line)
    if lines:
        match = re.match(r"([01])\b", lines[0])
        if match:
            return int(match.group(1))
    return None

