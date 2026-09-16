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

Medical vision–language models (VLMs) are approaching clinical use, but
hallucinations in the form of plausible yet visually unsupported answers remain
a major barrier to safe deployment. **DeLaS** incorporates visual grounding into
reliability assessment across the deployment pathway, connecting model selection,
hallucination screening, confidence-guided decisions, and retrospective monitoring.

| Layer | Stage | Purpose |
|---|---|---|
| I | Selection | Identify the best-performing VLM at both benchmark and question levels |
| II | Screening | Identify hallucination-prone responses with DeLaS-Scr |
| III | Decision | Calibrate confidence with DeLaS-Cal and guide reject/review/accept routing |
| IV | Retrospection | Estimate reliability across datasets and clinically meaningful subgroups |

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
\Delta = \left\|\mathbf{h}_{\mathrm{ori}}-\mathbf{h}_{\mathrm{int}}\right\|_2,
$$

where $\mathbf{h}_{\mathrm{ori}}$ and $\mathbf{h}_{\mathrm{int}}$ are response representations from
the original and intervened images.

DeLaS-Scr uses two separate probes to map the original response representation
and the intervention-induced shift to decision scores, then combines them to
estimate hallucination risk:

$$
p_{\mathrm{hall}}
=\sigma\left(\mathbf{w}_m^\top[z_h;z_{\Delta}]+b_m\right).
$$

Here, $z_h$ and $z_{\Delta}$ are the decision scores from two
$\ell_2$-regularized logistic regression probes applied to the original
representation and the shift, respectively. The scores are concatenated and
combined using learned weights $\mathbf{w}_m$ and bias $b_m$, followed by the
sigmoid function $\sigma$. Higher $p_{\mathrm{hall}}$ indicates greater
hallucination risk.

### DeLaS-Cal: grounding-aware confidence calibration

DeLaS-Cal adjusts a response-level reliability score using three complementary
visual-grounding cues: **Vision Attention Share (VAS)**, **Vision Attention
Concentration (VAC)**, and **Jacobian Norm (JN)**. Together, they characterize
the amount, distribution, and sensitivity of visual reliance.

These cues form a sample-specific gate that modulates the original score $s$:

$$
g=\sigma\left(\mathbf{w}_c^\top[\mathrm{VAS},\mathrm{VAC},\mathrm{JN}]+b_c\right),
\qquad
p_{\mathrm{correct}}=s\cdot g^{1/T}+\beta.
$$

Here, $\sigma$ is the sigmoid function, $T$ controls the gating effect, and
$\beta$ provides an offset correction.

The resulting probability of response correctness serves as a shared reliability
signal for model selection, response routing, and subgroup monitoring. Model
selection and retrospective accuracy estimation use this signal without requiring
ground-truth answers in the target setting. Likewise, DeLaS-Scr and DeLaS-Cal
require no ground-truth labels at deployment once fitted on labeled development
data.

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

For a quick start, run DeLaS with MedGemma on the VQA-RAD dataset.
The example workflow includes response generation, labeling, fitting, and evaluation.
Configure the runtime settings and required credentials in your environment,
then execute:

```bash
scripts/run_pipeline.sh configs/medgemma_vqa_rad.yaml
```

To evaluate another VLM–VQA pair, create a configuration for your model and dataset,
adapting the model interface and data loading as needed. Then run the same
workflow with your configuration file:

```bash
scripts/run_pipeline.sh path/to/your_config.yaml
```

## 📦 Workflow outputs

The workflow saves its outputs under `outputs/delas/`, including
hallucination scores, calibrated confidence, routing decisions,
model comparisons, and subgroup reliability summaries.

With reference labels, evaluation reports include hallucination
discrimination (AUROC), calibration (ECE and ACE), and correctness
among accepted responses (safe rate).

<a id="citation"></a>

## 📚 Citation

The manuscript describing DeLaS is currently under review. If you use this code,
please cite the software repository below. The paper citation will be added when
publicly available.

> Liu L, et al. (2026). *DeLaS: Deployment-Oriented Multi-Layered Safeguards for Medical
> Vision–Language Models* [Software]. GitHub. https://github.com/fdu-farm/DeLaS

```bibtex
@misc{liu2026delas,
  author = {Liu, Lei and others},
  title  = {{DeLaS}: Deployment-Oriented Multi-Layered Safeguards
            for Medical Vision-Language Models},
  year   = {2026},
  url    = {https://github.com/fdu-farm/DeLaS}
}
```
