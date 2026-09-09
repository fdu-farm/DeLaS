<div align="center">

# Deployment-Oriented Multi-Layered Safeguards for Medical Vision–Language Models

DeLaS — public reference implementation for MedGemma and VQA-RAD

MedGemma · VQA-RAD · Visual Grounding · Hallucination Screening · Confidence Calibration

[Configuration](configs/medgemma_vqa_rad.yaml) · [Citation](CITATION.cff)

</div>

<p align="center">
  <img src="docs/assets/DeLaS_overview.png" alt="Overview of the deployment-oriented multi-layered safeguards framework" width="92%">
</p>

<p align="center"><em>
The framework supports model selection, hallucination-risk screening,
confidence-guided decisions, and retrospective reliability review.
</em></p>

## 🧭 Overview

This repository is organized around **Layer 0 and Layers I–IV**, using **MedGemma** and
**VQA-RAD** as the public example. It implements the framework workflow,
not the full collection of experiments and reader studies reported in the paper:

| Layer | Stage | Purpose |
|---|---|---|
| 0 | Inference and preparation | Generate outputs, extract grounding features, and produce evaluation labels |
| I | Model selection | Estimate candidate-model accuracy and support model selection |
| II | Risk screening | Screen hallucination risk using DeLaS-Scr and nine baselines |
| III | Confidence calibration | Compare UC, temperature scaling (TS), and DeLaS-Cal |
| IV | Retrospection | Perform retrospective accuracy and reliability review |

Evaluation utilities for AUROC, ECE, ACE, and safe rate are provided in
the cross-layer evaluation module.

Layer 0 is the reproducibility and data-preparation stage. The safeguard
framework described in the paper comprises Layers I–IV.

### 🧠 Core methods

**DeLaS-Scr.** Layer II uses DeLaS-Scr to screen for
hallucination-prone responses, particularly those insufficiently supported by
the medical image. The underlying intuition is that image-grounded responses
should maintain relatively stable internal representations under a mild,
controlled visual intervention, whereas weakly grounded responses are more
likely to exhibit larger representation shifts. The intervention-induced
representation shift is defined as

$$
\Delta = \left\|h_{\mathrm{ori}}-h_{\mathrm{int}}\right\|_2,
$$

where $h_{\mathrm{ori}}$ and $h_{\mathrm{int}}$ are the last-layer hidden
states at the final non-special response token for the original and intervened
images. DeLaS-Scr then uses a stacked meta-classifier to estimate hallucination risk:

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

The two probes use `saga` logistic regression on the training split; an `lbfgs`
meta-classifier is fitted on the two validation decision scores. All three
classifiers use `C=1` and `max_iter=1500`. The intervention shift is standardized
using training statistics. The reported DeLaS-Scr confidence is $1-p_{\mathrm{hall}}$.

**DeLaS-Cal.** Layer III converts an uncalibrated
reliability score $s$ into a calibrated probability of correctness. It uses
three visual-grounding cues: **Vision Attention Share (VAS)**, measuring how
much the response attends to the image; **Vision Attention Concentration
(VAC)**, computed as normalized attention entropy (larger values indicate
more dispersed attention); and **Jacobian Norm (JN)**,
measuring response sensitivity to the visual input. These cues determine a
sample-specific gate that adjusts the original score:

$$
g=\sigma\left(w_c^\top[\mathrm{VAS},\mathrm{VAC},\mathrm{JN}]+b_c\right),
\qquad
p_{\mathrm{correct}}=s\cdot g^{1/T}+\beta.
$$

The three cues are scaled using validation medians and interquartile ranges.
The six parameters are fitted by validation BCE with analytic gradients and five
deterministic starts. SEne is converted to confidence as
$s=1-\exp(-\max(E,0)/\max(\operatorname{median}_{\mathrm{val}}(E),\epsilon))$,
using the same validation scale for test samples.

DeLaS-Cal therefore preserves the original reliability signal while correcting it
according to how strongly each response is grounded in the image.

Generation saves model outputs and grounding features only. Confidence
baselines—LinearProbe, AvgProb, MaxProb, AvgEnt, MaxEnt, SEnt, SEne, VASE, and RadFlag—are
computed separately during Layer II risk screening.

## 📁 Repository structure

```text
repository/
├── configs/                    Experiment configurations
├── docs/                       README assets
├── scripts/                    End-to-end helper scripts
├── src/                        Installable Python source
│   └── <package>/
│       ├── scores.py              Shared score direction and confidence conversion
│       ├── layer0_inference/       Generation, features, and labeling
│       ├── layer1_selection/       Accuracy estimation and model selection
│       ├── layer2_screening/       DeLaS-Scr and screening baselines
│       ├── layer3_calibration/     TS and DeLaS-Cal calibration
│       ├── layer4_retrospection/   Retrospective reliability review
│       └── evaluation/
├── tests/test_workflow.py       CPU workflow and boundary checks
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
python -m unittest discover -s tests -v
```

Edit `configs/medgemma_vqa_rad.yaml` to set the MedGemma model path,
Hugging Face dataset ID, device, and output directory. Authentication should
be supplied through the normal Hugging Face login/environment mechanism; no
token is stored in code.

When VQA-RAD has no validation split, the code deterministically reserves 20%
of its training split for calibration.

## 🏷️ Hallucination labeling

The labeling stage uses a single user-selected model to compare each response
against its reference answer. `labeling.model` is intentionally empty: set it
to a model available at the configured API endpoint before running the pipeline.
For the labeling command, `--model` overrides the configured model. An empty
model selection raises an error before creating the API client or label output.
It produces a CSV containing:

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

The script executes input preparation and labeling, Layer II screening, Layer III
calibration, Layer I model selection, Layer IV retrospection, and cross-layer
evaluation.

Both screening and calibration default to
`outputs/delas/labeling/labels.csv`. A different label file can still be supplied
with `--labels`.

The generation stage performs one greedy clean-image pass, one greedy
intervened-image pass, and the configured clean/intervened stochastic passes
for consistency baselines.

## 📦 Outputs

```text
outputs/delas/
├── generation/
│   ├── responses_and_features.csv
│   ├── logits/{sample_id}.npz
│   └── hidden_states/{sample_id}.pt
├── labeling/labels.csv
├── screening/
│   ├── screening_scores.csv
│   ├── delas_scr.joblib
│   └── linear_probe.joblib
├── calibration/calibrated_confidence.xlsx
├── calibration/routing.xlsx
├── layer1_selection/model_selection.xlsx
├── retrospection/accuracy_estimation.xlsx
└── evaluation/results.xlsx
```

`calibrated_confidence.xlsx` contains one sheet per measure. Every sheet has
the sample-level `UC`, `TS`, and `DeLaS-Cal` confidence values.

## Workflow outputs and policies

DeLaS is the complete framework. The `DeLaS-Cal` column in the `DeLaS-Scr`
worksheet represents calibrated screening confidence. `DeLaS-Scr` and
`LinearProbe` columns store correctness confidence; their hallucination risk is
one minus that confidence. Entropy measures use their raw values as risk. SEne
uses negative raw energy as risk and the validation-scaled positive conversion
above for calibration. AUROC is computed only on `screening.test_split`.

The independent `LinearProbe` baseline uses only clean hidden states, with
`lbfgs`, `C=1`, and `max_iter=1500`, trained on the training split. Its fitted
model is separate from both probes inside DeLaS-Scr.

`calibration/routing.xlsx` contains `samples` and `summary` sheets for all
measures and calibration methods. Routing uses the correctness probability:
`p <= low` rejects, `low < p <= high` refers for clinician review, and
`p > high` accepts. SafeRate is correctness among accepted samples; it is NaN
when no sample is accepted. The summary includes counts and fractions for all
three routes. Policies can be edited in the configuration:

| Policy | Low | High |
|---|---|---|
| High coverage | 0.30 | 0.50 |
| Balanced | 0.40 | 0.60 |
| High precision | 0.50 | 0.70 |

For Layer IV, `retrospection.group_by` names dataset metadata fields to retain
through generation, screening and calibration. Each available field is grouped
separately, alongside the overall summary. Missing fields produce a warning
and are skipped; partially missing values form a `(missing)` subgroup. No
question types or clinical attributes are inferred. Observed accuracy and
absolute error are reported only when correctness labels are provided.

For Layer I, compare multiple models on the same target samples:

```bash
layered-guardrails-select --config configs/medgemma_vqa_rad.yaml \
  --model-workbook model_a=/path/to/model_a/calibrated_confidence.xlsx \
  --model-workbook model_b=/path/to/model_b/calibrated_confidence.xlsx
```

Candidate workbooks must have the same measures, sample IDs, questions,
reference answers and splits, with unique IDs. Rows may appear in a different
order. `selected_models` selects by mean confidence; `selected_questions`
selects by per-question confidence. Ties use the supplied candidate order.
`question_summary` reports selected-answer accuracy when labels are available.
Labels never determine which candidate is selected.

The default output root is `outputs/delas` so earlier outputs remain separate.
Reinstall the package after updating module names and rerun the workflow to
produce the new model files and score schema. Reusing the same output root
updates its generated results. The overview image is retained unchanged under
its new filename and will be updated separately.

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
