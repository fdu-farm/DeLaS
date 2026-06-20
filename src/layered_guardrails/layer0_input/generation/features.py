from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


TRAILING_TOKENS = {"\n", "<end_of_turn>"}


@dataclass
class GroundingFeatures:
    VAS: float
    VAC: float
    JN: float
    Delta: float


def last_valid_step(outputs, input_length: int, tokenizer) -> int:
    ids = outputs.sequences[0, input_length:].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    for step in range(len(tokens) - 1, -1, -1):
        if tokens[step] not in TRAILING_TOKENS:
            return step
    raise ValueError("The generated response contains no valid token.")


def extract_last_generation_hidden(outputs, input_length: int, tokenizer) -> torch.Tensor:
    """Match the original MedGemma chain's final non-junk response representation."""
    step = last_valid_step(outputs, input_length, tokenizer)
    hidden_step = min(step, len(outputs.hidden_states) - 1)
    return outputs.hidden_states[hidden_step][-1][0, -1].detach().float().cpu()


def compute_delta(clean_hidden: torch.Tensor, intervened_hidden: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(clean_hidden - intervened_hidden, ord=2).item())


def vision_intervention(
    image: Image.Image,
    rotation_degrees: float,
    translate: float,
    gaussian_std: float,
    generator: torch.Generator | None = None,
) -> Image.Image:
    """Mild rotation/translation plus additive Gaussian noise."""
    angle = float(torch.empty(1).uniform_(-rotation_degrees, rotation_degrees, generator=generator))
    width, height = image.size
    max_dx, max_dy = translate * width, translate * height
    dx = int(float(torch.empty(1).uniform_(-max_dx, max_dx, generator=generator)))
    dy = int(float(torch.empty(1).uniform_(-max_dy, max_dy, generator=generator)))
    transformed = TF.affine(image, angle=angle, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])
    tensor = TF.to_tensor(transformed)
    noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * gaussian_std
    return TF.to_pil_image((tensor + noise).clamp(0.0, 1.0))


def _vision_mask(inputs, tokenizer) -> torch.Tensor:
    image_token_id = tokenizer.convert_tokens_to_ids("<image_soft_token>")
    return inputs["input_ids"][0].detach().cpu().eq(image_token_id)


def attention_features(inputs, outputs, tokenizer) -> tuple[float, float]:
    """Compute answer-averaged VAS and normalized visual-attention entropy (VAC)."""
    prompt_length = inputs["input_ids"].shape[-1]
    vision_mask = _vision_mask(inputs, tokenizer)
    final_step = last_valid_step(outputs, prompt_length, tokenizer)
    vas_values, vac_values = [], []

    for step in range(final_step + 1):
        attention = outputs.attentions[step][-1].float()[0, :, -1, :].mean(dim=0).cpu()
        key_mask = torch.zeros(attention.numel(), dtype=torch.bool)
        key_mask[:prompt_length] = vision_mask
        visual = attention[key_mask]
        mass = visual.sum()
        if visual.numel() == 0 or mass <= 0:
            continue
        distribution = (visual / mass).clamp_min(1e-12)
        entropy = -(distribution * distribution.log()).sum()
        vas_values.append(float(mass))
        vac_values.append(float(entropy / math.log(max(visual.numel(), 2))))

    if not vas_values:
        return float("nan"), float("nan")
    return float(np.mean(vas_values)), float(np.mean(vac_values))


def jacobian_norm(model, prompt_inputs, outputs) -> float:
    """Frobenius norm of response log-likelihood gradient w.r.t. image pixels."""
    device = next(model.parameters()).device
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])
    full_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value) and key not in {"input_ids", "attention_mask", "position_ids", "cache_position"}:
            full_inputs[key] = value.detach().clone().to(device)
    full_inputs["input_ids"] = outputs.sequences.detach().clone().to(device)
    full_inputs["attention_mask"] = torch.ones_like(full_inputs["input_ids"])

    pixel_key = next(
        (key for key in ("pixel_values", "images", "image", "vision_pixel_values") if key in full_inputs),
        None,
    )
    if pixel_key is None:
        raise KeyError("No image tensor was found in the processor output.")
    pixels = full_inputs[pixel_key].detach().float().clone().requires_grad_(True)
    full_inputs[pixel_key] = pixels

    generated = full_inputs["input_ids"][:, prompt_length:]
    tokens = model(**full_inputs).logits[:, :-1]
    targets = full_inputs["input_ids"][:, 1:]
    token_log_probs = torch.log_softmax(tokens, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    response_log_prob = token_log_probs[:, prompt_length - 1 : prompt_length - 1 + generated.shape[1]].mean()
    gradient = torch.autograd.grad(-response_log_prob, pixels, retain_graph=False)[0]
    return float(torch.linalg.vector_norm(gradient).item())
