#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/medgemma_vqa_rad.yaml}"

layered-guardrails-generate --config "$CONFIG"
layered-guardrails-label --config "$CONFIG"
layered-guardrails-baselines --config "$CONFIG"
layered-guardrails-screen --config "$CONFIG"
layered-guardrails-calibrate --config "$CONFIG"
layered-guardrails-select --config "$CONFIG"
layered-guardrails-retrospect --config "$CONFIG"
layered-guardrails-evaluate --config "$CONFIG"
