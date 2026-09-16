<div align="center">

# DeLaS: Deployment-Oriented Multi-Layered Safeguards for Medical Vision–Language Models

Medical VLM, Reliability-aware deployment, Hallucination

[Configuration](configs/medgemma_vqa_rad.yaml) · [Citation](#citation)

</div>

<p align="center">
  <img src="docs/assets/DeLaS_overview.png" alt="Overview of the deployment-oriented multi-layered safeguards framework" width="92%">
</p>

<p align="center"><em>
Selection → Screening → Decision → Retrospection
</em></p>

## 🧭 Overview

Medical vision–language models can produce plausible answers that are unsupported
by the image. **DeLaS** incorporates visual grounding into reliability assessment
across the deployment pathway, connecting model selection, hallucination screening,
confidence-guided decisions, and retrospective monitoring.

| Layer | Stage | Purpose |
|---|---|---|
| I | Selection | Estimate target-setting accuracy and select the most reliable candidate VLM |
| II | Screening | Identify hallucination-prone responses with DeLaS-Scr |
| III | Decision | Calibrate confidence with DeLaS-Cal and guide reject/review/accept routing |
| IV | Retrospection | Estimate reliability across datasets and clinically meaningful subgroups |

Calibrated response confidence provides a shared signal for model selection,
routing, and subgroup monitoring. Selection and retrospective accuracy estimation
operate without ground-truth answers in the target setting; the screening and
calibration components are fitted using labeled development data.

The paper evaluates DeLaS across three medical VLMs, eight public benchmarks,
and an independent breast-ultrasound cohort. DeLaS identified the best-performing
VLM on all nine benchmarks and improved screening, calibration, and subgroup
reliability estimation. In a multi-clinician reader study, DeLaS-guided assistance
reduced incorrect-response propagation to **8.25%**, compared with **15.50–32.50%**
under the comparator strategies.

## 🧠 Core methods

### DeLaS-Scr: visual-intervention-based screening

DeLaS-Scr examines how a response representation changes under a mild, controlled
visual intervention. Weakly grounded responses tend to exhibit larger shifts:

$$
\Delta = \left\|h_{\mathrm{ori}}-h_{\mathrm{int}}\right\|_2,
$$

where $h_{\mathrm{ori}}$ and $h_{\mathrm{int}}$ are response representations from
the original and intervened images. Combining the original representation with
this intervention-induced shift produces a grounding-aware estimate of
hallucination risk.

### DeLaS-Cal: grounding-aware confidence calibration

DeLaS-Cal adjusts a response-level reliability score using three complementary
visual-grounding cues: **Vision Attention Share (VAS)**, **Vision Attention
Concentration (VAC)**, and **Jacobian Norm (JN)**. Together, they characterize
the amount, distribution, and sensitivity of visual reliance.

These cues form a sample-specific gate that modulates the original score $s$:

$$
g=\sigma\left(w_c^\top[\mathrm{VAS},\mathrm{VAC},\mathrm{JN}]+b_c\right),
\qquad
p_{\mathrm{correct}}=s\cdot g^{1/T}+\beta.
$$

Here, $\sigma$ is the sigmoid function, $T$ controls the gating effect, and
$\beta$ provides an offset correction. The resulting confidence estimates the
probability of response correctness, supporting both individual decisions and
aggregate reliability assessment.

## 📁 Repository structure

```text
repository/
├── configs/                         Workflow configuration
├── docs/                            Framework overview and assets
├── scripts/                         End-to-end workflow
├── src/layered_guardrails/
│   ├── layer0_inference/            Inference and feature preparation
│   ├── layer1_selection/            Model selection
│   ├── layer2_screening/            DeLaS-Scr and screening baselines
│   ├── layer3_calibration/          DeLaS-Cal and confidence routing
│   ├── layer4_retrospection/        Subgroup reliability assessment
│   └── evaluation/                  Cross-layer evaluation
├── tests/                           Workflow checks
├── CITATION.cff
└── pyproject.toml
```

Layer 0 prepares the inputs and features used by the four safeguard layers.

## 🚀 Getting started

Install the package from the repository root:

```bash
python -m pip install -e .
```

Configure the runtime settings and required credentials for your environment,
then run the workflow:

```bash
scripts/run_pipeline.sh configs/medgemma_vqa_rad.yaml
```

For development and verification:

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m unittest discover -s tests -v
```

## 📦 Workflow outputs and policies

Results are saved under `outputs/delas/`, including screening scores, calibrated
confidence, routing decisions, model comparisons, and retrospective summaries.
Evaluation covers hallucination discrimination (AUROC), calibration (ECE and ACE),
and correctness among accepted responses (safe rate).

Two configurable confidence thresholds assign responses to **reject**, **clinician
review**, or **accept** pathways. Coverage-oriented and precision-oriented policies
allow different trade-offs between response retention and reliability.

For model selection, candidate VLMs are compared on the same target samples using
calibrated confidence. For retrospection, confidence is aggregated within
prespecified subgroups to reveal reliability differences that overall averages
may obscure.

<a id="citation"></a>

## 📚 Citation

The manuscript describing DeLaS is currently under review. If you use this code,
please cite the software repository below. The paper citation will be added when
publicly available.

> Lei L, et al. (2026). *DeLaS: Deployment-Oriented Multi-Layered Safeguards for Medical
> Vision–Language Models* [Software]. GitHub. https://github.com/fdu-farm/DeLaS
