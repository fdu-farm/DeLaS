from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from layered_guardrails.layer0_inference.generation.data import build_question_prompt
from layered_guardrails.layer0_inference.generation.features import (
    attention_features,
    extract_last_generation_hidden,
    jacobian_norm,
)


@dataclass
class GenerationOutput:
    response: str
    token_ids: np.ndarray
    logits: np.ndarray
    hidden_state: torch.Tensor
    VAS: float | None = None
    VAC: float | None = None
    JN: float | None = None


class MedGemmaRunner:
    def __init__(self, model_path: str, device: str, dtype: str = "bfloat16"):
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        ).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.device = device

    def prepare(self, question, image):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a medical image analysis expert."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_question_prompt(question)},
                    {"type": "image", "image": image},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.model.dtype)

    def generate(
        self,
        question,
        image,
        max_new_tokens=200,
        sample=False,
        temperature=1.0,
        top_p=0.9,
        grounding=False,
    ) -> GenerationOutput:
        inputs = self.prepare(question, image)
        input_length = inputs["input_ids"].shape[-1]
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_attentions": grounding,
            "output_hidden_states": True,
        }
        if sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        token_ids = outputs.sequences[0, input_length:]
        result = GenerationOutput(
            response=self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            token_ids=token_ids.detach().cpu().numpy(),
            logits=torch.cat(outputs.scores, dim=0).detach().float().cpu().numpy().astype(np.float16),
            hidden_state=extract_last_generation_hidden(outputs, input_length, self.processor.tokenizer),
        )
        if grounding:
            result.VAS, result.VAC = attention_features(inputs, outputs, self.processor.tokenizer)
            result.JN = jacobian_norm(self.model, inputs, outputs)
        return result
