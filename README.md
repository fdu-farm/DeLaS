<div align="center">

# Deployment-Centered Multi-Layered Safeguards for Medical Vision–Language Models

Official reference implementation

MedGemma · VQA-RAD · Visual Grounding · Hallucination Screening · Confidence Calibration

[Configuration](configs/medgemma_vqa_rad.yaml) · [Citation](CITATION.cff)

</div>

<p align="center">
  <img src="docs/assets/deployment-centered-safeguards-overview.png" alt="Overview of the deployment-centered multi-layered safeguards framework" width="92%">
</p>

<p align="center"><em>
The framework supports model selection, hallucination-risk screening,
confidence-guided decisions, and retrospective reliability review.
</em></p>

## 🧭 Overview

This repository is organized around **Layer 0–4**, using **MedGemma** and
**VQA-RAD** as the public example:

| Layer | Stage | Purpose |
|---|---|---|
| 0 | Input preparation | Generate outputs, extract grounding features, and produce evaluation labels |
| 1 | Model selection | Estimate candidate-model accuracy and support model selection |
| 2 | Risk screening | Screen hallucination risk using ViP and eight baselines |
| 3 | Confidence calibration | Compare UC, temperature scaling (TS), and VG-GC |
| 4 | Retrospection | Perform retrospective accuracy and reliability review |

Evaluation utilities for AUROC, ECE, ACE, and safe rate are provided in
the cross-layer evaluation module.

Layer 0 is the reproducibility and data-preparation stage. The safeguard
framework described in the paper comprises Layers 1–4.

### 🧠 Core methods

**Visual Intervention Probing (ViP).** Layer 2 screens hallucination-prone
responses by measuring their sensitivity to a mild visual intervention. The
intervention-induced representation shift is defined as

$$
\Delta = \left\|h_{\mathrm{ori}}-h_{\mathrm{int}}\right\|_2,
$$

where $h_{\mathrm{ori}}$ and $h_{\mathrm{int}}$ are the last-layer hidden
states at the final non-special response token for the original and intervened
images. ViP then uses a stacked meta-classifier to estimate hallucination risk:

$$
p_{\mathrm{hall}}=
\sigma\left(w_m^\top[z_h;z_\Delta]+b_m\right).
$$

Here, $z_h$ is the decision score of a representation probe on the original
response hidden state, and $z_\Delta$ is the decision score of an intervention
probe on the representation shift:

$$
z_h=f_h(h_{\mathrm{ori}}),\qquad
z_\Delta=f_\Delta(\Delta).
$$

**Vision-Grounded Gated Calibration (VG-GC).** Layer 3 converts an uncalibrated
reliability score $s$ into a calibrated probability of correctness. It uses
three visual-grounding cues: **Vision Attention Share (VAS)**, measuring how
much the response attends to the image; **Vision Attention Concentration
(VAC)**, measuring how focused that attention is; and **Jacobian Norm (JN)**,
measuring response sensitivity to the visual input. These cues determine a
sample-specific gate that adjusts the original score:

$$
g=\sigma\left(w_c^\top[\mathrm{VAS},\mathrm{VAC},\mathrm{JN}]+b_c\right),
\qquad
p_{\mathrm{correct}}=s\cdot g^{1/T}+\beta.
$$

VG-GC therefore preserves the original reliability signal while correcting it
according to how strongly each response is grounded in the image.

Generation saves model outputs and grounding features only. Confidence
baselines—AvgProb, MaxProb, AvgEnt, MaxEnt, SEnt, SEne, VASE, and RadFlag—are
computed separately during Layer 2 risk screening.

## 📁 Repository structure

```text
repository/
├── configs/                    Experiment configurations
├── docs/                       README assets
├── scripts/                    End-to-end helper scripts
├── src/                        Installable Python source
│   └── <package>/
│       ├── layer0_input/           Generation, features, and labeling
│       ├── layer1_selection/       Accuracy estimation and model selection
│       ├── layer2_screening/       ViP and screening baselines
│       ├── layer3_calibration/     TS and VG-GC calibration
│       ├── layer4_retrospection/   Retrospective reliability review
│       └── evaluation/
├── CITATION.cff
└── pyproject.toml
```

## ⚙️ Setup

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
ruff check .
```

Edit `configs/medgemma_vqa_rad.yaml` to set the MedGemma model path,
Hugging Face dataset ID, device, and output directory. Authentication should
be supplied through the normal Hugging Face login/environment mechanism; no
token is stored in code.

When VQA-RAD has no validation split, the code deterministically reserves 20%
of its training split for calibration.

## 🏷️ Hallucination labeling

The labeling stage produces a CSV containing:

```text
sample_id,split,question,answer,response,correctness_label,
hallucination_label,evaluator_reason
```

A `labels.example.csv` template is included with the Layer 0 labeling code.

`hallucination_label=1` means hallucinated. `correctness_label=1` means
correct. The current protocol treats an incorrect response as hallucinated, so
`hallucination_label = 1 - correctness_label`. Target-domain accuracy
estimation itself does not use labels.

Set `DASHSCOPE_API_KEY` or `OPENAI_API_KEY` before running labeling. API keys
are never stored in the repository.

## 🚀 Quick start

Install the package, then run the full workflow from the repository root:

```bash
export DASHSCOPE_API_KEY="your-api-key"
scripts/run_pipeline.sh configs/medgemma_vqa_rad.yaml
```

The script executes input preparation and labeling, Layer 2 screening, Layer 3
calibration, Layer 1 model selection, Layer 4 retrospection, and cross-layer
evaluation.

Both screening and calibration default to
`outputs/labeling/labels.csv`. A different label file can still be supplied
with `--labels`.

The generation stage performs one greedy clean-image pass, one greedy
intervened-image pass, and the configured clean/intervened stochastic passes
for consistency baselines.

## 📦 Outputs

```text
outputs/
├── generation/
│   ├── responses_and_features.csv
│   ├── logits/{sample_id}.npz
│   └── hidden_states/{sample_id}.pt
├── labeling/labels.csv
├── screening/screening_scores.csv
├── calibration/calibrated_confidence.xlsx
├── layer1_selection/model_selection.xlsx
├── retrospection/accuracy_estimation.xlsx
└── evaluation/results.xlsx
```

`calibrated_confidence.xlsx` contains one sheet per measure. Every sheet has
the sample-level `UC`, `TS`, and `VG-GC` confidence values.

## 📝 Implementation notes

- VAS and VAC use answer-token attention from the final decoder layer.
- JN is the norm of the response log-likelihood gradient with respect to image
  pixels.
- Large model weights, datasets, generated logits, and hidden states are not
  included in the repository.

## 📚 Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff). On GitHub,
the repository's **Cite this repository** button will use this file.

> [!CAUTION]
> This research code is not a medical device and is not intended for direct
> clinical use.
