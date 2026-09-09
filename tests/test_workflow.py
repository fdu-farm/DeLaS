from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import yaml

from layered_guardrails.evaluation.metrics import safe_rate
from layered_guardrails.evaluation.run import test_screening_scores as screening_metrics
from layered_guardrails.layer0_inference.generation.data import load_vqa_rad
from layered_guardrails.layer1_selection.model_selection import (
    load_candidates, select_best_model, select_per_question, summarize_predictions,
)
from layered_guardrails.layer2_screening.baselines.linear_probe import confidence, fit_linear_probe
from layered_guardrails.layer2_screening.delas_scr import DeLaSScr
from layered_guardrails.layer3_calibration.routing import route, routing_tables
from layered_guardrails.layer4_retrospection.run import summarize_groups
from layered_guardrails.scores import CALIBRATION_METHODS, as_confidence_pair, hallucination_risk


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def frame(self):
        return pd.DataFrame({
            "sample_id": ["001", "002", "003"], "question": ["q1", "q2", "q3"],
            "answer": ["a1", "a2", "a3"], "split": ["heldout"] * 3,
            "UC": [0.4, 0.6, 0.8], "TS": [0.4, 0.6, 0.8], "DeLaS-Cal": [0.4, 0.6, 0.8],
            "correctness_label": [0, 1, 1], "question_type": ["organ", "organ", "modality"],
        })

    def workbook(self, name, frame):
        path = self.root / f"{name}.xlsx"
        frame.to_excel(path, sheet_name="LinearProbe", index=False)
        return name, str(path)

    def test_score_directions_and_validation_only_energy_scale(self):
        val = np.array([1., 2., 3.])
        target = np.array([0., 2., 20.])
        _, actual = as_confidence_pair(val, target, "SEne")
        np.testing.assert_allclose(actual, np.clip(1 - np.exp(-target / 2), 1e-10, 1 - 1e-10))
        np.testing.assert_array_equal(hallucination_risk(target, "SEne"), -target)
        for measure in ["AvgEnt", "MaxEnt", "SEnt", "VASE"]:
            np.testing.assert_array_equal(hallucination_risk(val, measure), val)
            np.testing.assert_allclose(as_confidence_pair(val, val, measure)[0], np.exp(-val))
        for measure in ["DeLaS-Scr", "LinearProbe", "AvgProb", "MaxProb", "RadFlag"]:
            np.testing.assert_allclose(hallucination_risk([.1, .9], measure), [.9, .1])
        self.assertTrue(np.isfinite(as_confidence_pair([0, 0], [0, 1], "SEne")[1]).all())
        with self.assertRaises(ValueError):
            hallucination_risk([1], "unknown")

    def test_routing_equal_thresholds_and_empty_acceptance(self):
        values = [0, .4, .5, .6, np.nextafter(.6, 1), 1]
        self.assertEqual(route(values, .4, .6).tolist(),
                         ["reject", "reject", "review", "review", "accept", "accept"])
        self.assertEqual(safe_rate([0, 1], [.6, .7], .6), 1.)
        self.assertTrue(np.isnan(safe_rate([1], [.6], .6)))
        with self.assertRaises(ValueError):
            route([.5], .6, .4)
        with self.assertRaises(ValueError):
            route([np.nan], .4, .6)
        samples, summary = routing_tables(self.frame(), "LinearProbe", {"balanced": [.4, .6]})
        self.assertEqual(len(samples), 9)
        np.testing.assert_array_equal(summary[["reject_count", "review_count", "accept_count"]],
                                      np.ones((3, 3)))
        self.assertTrue((summary["safe_rate"] == 1).all())

    def test_screening_metrics_use_only_configured_test_split(self):
        frame = pd.DataFrame({
            "sample_id": ["a", "b", "c", "d"], "split": ["train", "validation", "heldout", "heldout"],
            "hallucination_label": [0, 1, 0, 1], "SEne": [0., 5., 5., 0.],
        })
        result = screening_metrics(frame, "heldout", ["SEne"], "hallucination_label")
        self.assertEqual(result.iloc[0]["num_samples"], 2)
        self.assertEqual(result.iloc[0]["AUROC"], 1.)
        with self.assertRaises(ValueError):
            screening_metrics(frame, "test", ["SEne"], "hallucination_label")

    def test_metadata_is_preserved_without_inference(self):
        source = {"id": "001", "image": object(), "question": "q", "answer": "a", "question_type": "organ"}
        dataset_module = SimpleNamespace(load_dataset=lambda *a, **k: {"heldout": [source]})
        with patch.dict(sys.modules, {"datasets": dataset_module}):
            item = next(load_vqa_rad("unused", None))
        self.assertEqual(item["question_type"], "organ")
        self.assertNotIn("modality", item)
        self.assertEqual(item["sample_id"], "001")

    def test_subgroups_and_optional_labels(self):
        result = summarize_groups(self.frame(), "LinearProbe", ["question_type"])
        organ = result[(result.group_value == "organ") & (result.calibration == "UC")].iloc[0]
        self.assertEqual(organ.num_samples, 2)
        self.assertEqual(organ.estimated_accuracy, .5)
        self.assertEqual(organ.observed_accuracy, .5)
        unlabeled = self.frame().drop(columns="correctness_label")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = summarize_groups(unlabeled, "LinearProbe", ["organ"])
        self.assertEqual(len(caught), 1)
        self.assertNotIn("observed_accuracy", result)

    def test_model_selection_alignment_ties_and_label_independence(self):
        a, b = self.frame(), self.frame().iloc[::-1].copy()
        for method in CALIBRATION_METHODS:
            b[method] = [.8, .9, .2]
        b["correctness_label"] = 0
        candidates = [self.workbook("a", a), self.workbook("b", b)]
        predictions = load_candidates(candidates)
        selected = select_per_question(predictions)
        rows = selected[selected.calibration == "UC"].set_index("sample_id")
        self.assertEqual(rows.loc["001", "selected_model"], "a")
        self.assertEqual(rows.loc["002", "selected_model"], "b")
        self.assertEqual(rows.loc["003", "selected_model"], "a")
        self.assertEqual(rows.loc["002", "correctness_label"], 0)
        winners = select_best_model(summarize_predictions(predictions))
        self.assertTrue((winners["model"] == "b").all())
        a = a.drop(columns="correctness_label")
        b = b.drop(columns="correctness_label")
        self.assertEqual(len(select_per_question(load_candidates([
            self.workbook("a", a), self.workbook("b", b)
        ]))), 9)

    def test_model_selection_rejects_mismatched_or_duplicate_samples(self):
        a = self.frame()
        for change in ["missing", "question", "duplicate", "answer", "split"]:
            b = a.copy()
            if change == "missing":
                b = b.iloc[:2]
            elif change == "duplicate":
                b.loc[1, "sample_id"] = "001"
            else:
                b.loc[0, change] = "different"
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_candidates([self.workbook("a", a), self.workbook("b", b)])

    def test_probe_is_independent_and_serializable(self):
        rng = np.random.default_rng(42)
        hidden = rng.normal(size=(40, 4)).astype(np.float32)
        labels = (hidden[:, 0] > 0).astype(int)
        probe = fit_linear_probe(hidden[:24], labels[:24])
        scr = DeLaSScr().fit(np.arange(24), hidden[:24], labels[:24],
                            np.arange(24, 40), hidden[24:], labels[24:])
        self.assertEqual([scr.linear_probe.solver, scr.causal.solver, scr.meta.solver],
                         ["saga", "saga", "lbfgs"])
        self.assertEqual([scr.linear_probe.C, scr.causal.C, scr.meta.C], [1., 1., 1.])
        self.assertIsNot(probe, scr.linear_probe)
        self.assertEqual(probe.solver, "lbfgs")
        self.assertEqual(scr.meta.n_features_in_, 2)
        path = self.root / "linear_probe.joblib"
        joblib.dump(probe, path)
        np.testing.assert_array_equal(confidence(probe, hidden), confidence(joblib.load(path), hidden))

    def test_labeling_requires_model_before_client_or_output(self):
        labeling = importlib.import_module("layered_guardrails.layer0_inference.labeling.run")
        for settings in [{}, {"model": ""}, {"model": "  "}, {"model": None}]:
            config_path = self.root / "labeling.yaml"
            config_path.write_text(yaml.safe_dump({
                "paths": {"output_root": str(self.root)}, "labeling": settings,
            }))
            with self.subTest(settings=settings):
                with patch.object(sys, "argv", ["label", "--config", str(config_path)]):
                    with patch.object(labeling, "build_client") as client:
                        with patch.object(labeling, "initialize_output") as initialize:
                            with self.assertRaisesRegex(ValueError, "labeling.model.*--model"):
                                labeling.main()
                            client.assert_not_called()
                            initialize.assert_not_called()
        self.assertFalse((self.root / "labeling").exists())

    def test_labeling_model_configuration_and_cli_override(self):
        labeling = importlib.import_module("layered_guardrails.layer0_inference.labeling.run")
        output = self.root / "labeling"
        output.mkdir()
        config_path = self.root / "labeling.yaml"
        config_path.write_text(yaml.safe_dump({
            "paths": {"output_root": str(self.root)},
            "labeling": {"model": "configured-model"},
        }))
        for extra, expected in [([], "configured-model"), (["--model", " cli-model "], "cli-model")]:
            frame = pd.DataFrame([{
                "sample_id": "001", "question": "q", "answer": "a", "response": "r",
                "correctness_label": "pending", "hallucination_label": "pending",
                "evaluator_reason": "pending",
            }])
            with self.subTest(extra=extra):
                with patch.object(sys, "argv", ["label", "--config", str(config_path), *extra]):
                    with patch.object(labeling, "build_client"):
                        with patch.object(labeling, "initialize_output", return_value=frame):
                            with patch.object(labeling, "request_label", return_value=(1, "correct")) as request:
                                labeling.main()
                                self.assertEqual(request.call_args.args[1], expected)
            self.assertEqual(frame.loc[0, "correctness_label"], 1)

    def test_pipeline_from_precomputed_inference(self):
        """Exercise actual CLI functions; only the tensor file reader is substituted."""
        root = self.root
        for name in ["generation", "screening", "labeling"]:
            (root / name).mkdir()
        config = yaml.safe_load(Path("configs/medgemma_vqa_rad.yaml").read_text())
        config["paths"]["output_root"] = str(root)
        config["screening"]["test_split"] = "heldout"
        config["calibration"]["test_split"] = "heldout"
        config["retrospection"]["group_by"] = ["question_type"]
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        rng = np.random.default_rng(10)
        n = 72
        hidden = rng.normal(size=(n, 5)).astype(np.float32)
        hall = np.tile([0, 1], n // 2)
        hidden[:, 0] += hall
        frame = pd.DataFrame({
            "sample_id": [f"{i:03}" for i in range(n)],
            "split": ["train"] * 32 + ["validation"] * 24 + ["heldout"] * 16,
            "question": [f"question {i}" for i in range(n)], "answer": ["reference"] * n,
            "response": ["response"] * n, "question_type": np.tile(["organ", "modality"], n // 2),
            "VAS": rng.uniform(.1, .8, n), "VAC": rng.uniform(.1, .8, n), "JN": rng.uniform(0, 2, n),
            "Delta": rng.uniform(0, 3, n),
        })
        frame.to_csv(root / "generation/responses_and_features.csv", index=False)
        scores = frame[["sample_id", "split"]].copy()
        for measure in config["calibration"]["measures"]:
            if measure not in {"LinearProbe", "DeLaS-Scr"}:
                scores[measure] = rng.uniform(.1, .9, n)
        scores.to_csv(root / "screening/baseline_scores.csv", index=False)
        labels = frame[["sample_id"]].copy()
        labels["hallucination_label"] = hall
        labels["correctness_label"] = 1 - hall
        labels.to_csv(root / "labeling/labels.csv", index=False)
        screen = importlib.import_module("layered_guardrails.layer2_screening.run")
        with patch.object(screen, "_load_hidden", side_effect=lambda directory, sid: hidden[int(sid)]):
            with patch.object(sys, "argv", ["screen", "--config", str(config_path)]):
                screen.main()
        for module in ["layer3_calibration", "layer1_selection", "layer4_retrospection", "evaluation"]:
            with patch.object(sys, "argv", [module, "--config", str(config_path)]):
                importlib.import_module(f"layered_guardrails.{module}.run").main()
        calibration = pd.read_excel(root / "calibration/calibrated_confidence.xlsx", sheet_name=None)
        self.assertEqual(set(calibration), set(config["calibration"]["measures"]))
        for result in calibration.values():
            self.assertEqual(len(result), 16)
            self.assertEqual(set(result["split"]), {"heldout"})
            self.assertIn("question_type", result)
            self.assertTrue(np.isfinite(result[list(CALIBRATION_METHODS)].to_numpy()).all())
        screening = pd.read_excel(root / "evaluation/results.xlsx", sheet_name="screening")
        self.assertTrue((screening.num_samples == 16).all())
        routing = pd.read_excel(root / "calibration/routing.xlsx", sheet_name="summary")
        self.assertEqual(len(routing), len(calibration) * 3 * 3)
        self.assertTrue((routing[["reject_count", "review_count", "accept_count"]].sum(axis=1) == 16).all())
        groups = pd.read_excel(root / "retrospection/accuracy_estimation.xlsx")
        self.assertEqual(set(groups.group_field), {"overall", "question_type"})
        selected = pd.read_excel(root / "layer1_selection/model_selection.xlsx", sheet_name="selected_questions")
        self.assertEqual(len(selected), len(calibration) * 3 * 16)
        self.assertTrue((root / "screening/delas_scr.joblib").is_file())
        self.assertTrue((root / "screening/linear_probe.joblib").is_file())


if __name__ == "__main__":
    unittest.main()
