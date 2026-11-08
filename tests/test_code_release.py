"""Behavioral checks using generated inputs only; no research data are read."""

import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
from target_registration.core import base, global_local_base, proposed
from target_registration.data import GT_FIELDS, load_training_protocol, validate_keys
from target_registration.inference import load_model, load_visible_bbox, register_one
from target_registration.point_metrics import (
    bilinear, make_local_mask, replay_global_point, replay_local_forward_point,
)

torch.set_num_threads(2)


class SyntheticFixedModel(torch.nn.Module):
    """Return fixed geometry to exercise renderer selection independently."""

    def __init__(self, confidence, gate):
        super().__init__()
        self.confidence = confidence
        self.gate = gate

    def forward(self, visible, infrared, bbox, has_bbox, enable_local=True):
        def scalar(value):
            return visible.new_tensor([value])
        return {
            "geo": {"tx": scalar(1), "ty": scalar(0), "angle": scalar(0),
                    "scale_x": scalar(1), "scale_y": scalar(1)},
            "confidence_error_px": scalar(self.confidence),
            "local": {"gate": scalar(self.gate), "raw_dx": scalar(1), "raw_dy": scalar(-1)},
        }


class CodeReleaseTest(unittest.TestCase):
    def test_empty_and_subpixel_visible_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic_box.txt"
            path.write_text("", encoding="utf-8")
            bbox, present = load_visible_bbox(path, 960, 768)
            self.assertFalse(present)
            np.testing.assert_array_equal(bbox, np.zeros(4, dtype=np.float32))
            # Pixel coordinates below 1.5 must still be normalized.
            path.write_text("0 0.0001 0.0001 0.0001 0.0001\n", encoding="utf-8")
            bbox, present = load_visible_bbox(path, 960, 768)
            self.assertTrue(present)
            expected = np.asarray(base.load_txt_bbox(path, (768, 960)), dtype=np.float32)
            expected /= np.asarray([960, 768, 960, 768], dtype=np.float32)
            np.testing.assert_array_equal(bbox, expected)
            self.assertLess(float(bbox.max()), 0.01)

    def test_local_stage_freezes_backbone_and_propagates_gradients(self):
        torch.manual_seed(17)
        prior = {"base_angle_deg": 0.0, "base_scale_x": 1.0, "base_scale_y": 1.0}
        initializer = global_local_base.BaseGlobalLocalRegistrationNet(0.0, 1.0, 1.0)
        model, _, _ = proposed.build_model(
            {"model": initializer.state_dict(), "prior_stats": prior}, torch.device("cpu")
        )
        model.eval()
        model.target_local_corr.train()
        visible = torch.rand(1, 3, 384, 480)
        infrared = torch.rand_like(visible)
        bbox = torch.tensor([[0.45, 0.43, 0.55, 0.57]])
        has_bbox = torch.ones(1)
        outputs = model(visible, infrared, bbox, has_bbox, True)
        raw_center = torch.tensor([[240.0, 192.0]])
        geo = outputs["geo"]
        transformed = global_local_base.transform_points_anisotropic(
            raw_center, geo["tx"], geo["ty"], geo["angle"], geo["scale_x"], geo["scale_y"]
        )
        batch = {
            "ir_uav_raw": raw_center,
            "vis_uav_center": transformed.detach() + torch.tensor([[2.0, -1.0]]),
            "raw_to_input_scale": torch.tensor([[0.5, 0.5]]),
            "has_bbox": has_bbox,
        }
        loss, detail = proposed.target_training_loss(outputs, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(detail["valid"]), 1.0)
        loss.backward()
        trainable = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("target_local_corr.") for name, _ in trainable))
        self.assertTrue(any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
                            for _, parameter in trainable))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()
                            if not parameter.requires_grad))

    def test_trained_model_loader_does_not_reinitialize_local_branch(self):
        torch.manual_seed(23)
        model = proposed.ProposedRegistrationNet(0.0, 1.0, 1.0)
        with torch.no_grad():
            for parameter in model.target_local_corr.parameters():
                parameter.fill_(0.123)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.pt"
            torch.save({"model": model.state_dict(), "prior_stats": {
                "base_angle_deg": 0.0, "base_scale_x": 1.0, "base_scale_y": 1.0,
            }}, path)
            loaded, _ = load_model(path, torch.device("cpu"))
            self.assertFalse(loaded.edge_enabled)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)

    def test_renderer_selection_without_manual_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("visible", "infrared", "labels"):
                (root / name).mkdir()
            # Deterministic synthetic image, not an image from the research set.
            generator = np.random.default_rng(19)
            image = generator.integers(0, 256, (768, 960, 3), dtype=np.uint8)
            for name in ("visible", "infrared"):
                ok, encoded = cv2.imencode(".png", image)
                self.assertTrue(ok)
                encoded.tofile(str(root / name / "synthetic_pair.png"))
            (root / "labels" / "synthetic_pair.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )
            calibration = {"confidence_threshold_error_px": 1.0, "gate_threshold": 0.5}
            cases = [
                (0.0, 0.0, 0.0, "global"),
                (0.0, 1.0, 0.0, "highres_target"),
                (2.0, 0.0, 0.0, "fallback_prior"),
                (2.0, 0.0, 90.0, "fallback_identity"),
            ]
            for confidence, gate, prior_angle, expected in cases:
                with self.subTest(output=expected):
                    result, metadata = register_one(
                        "synthetic_pair", SyntheticFixedModel(confidence, gate),
                        {"base_angle_deg": prior_angle, "base_scale_x": 1.0, "base_scale_y": 1.0},
                        calibration, torch.device("cpu"),
                        root / "visible", root / "infrared", root / "labels",
                    )
                    self.assertEqual(metadata["final_source"], expected)
                    self.assertEqual(result.shape, (768, 960))
                    self.assertEqual(result.dtype, np.uint8)
                    self.assertEqual(metadata["global_tx_full_px"], 2.0)
            # No annotation CSV or manually specified infrared center exists.
            self.assertFalse(list(root.rglob("*.csv")))
            (root / "labels/synthetic_pair.txt").write_text("", encoding="utf-8")
            result, metadata = register_one(
                "synthetic_pair", SyntheticFixedModel(0.0, 1.0),
                {"base_angle_deg": 0.0, "base_scale_x": 1.0, "base_scale_y": 1.0},
                calibration, torch.device("cpu"),
                root / "visible", root / "infrared", root / "labels",
            )
            self.assertFalse(metadata["has_bbox"])
            self.assertEqual(metadata["final_source"], "global")

    def test_point_evaluation_uses_gaussian_field_inversion(self):
        meta = {
            "bbox_normalized": [0.45, 0.45, 0.55, 0.55], "has_bbox": True,
            "local_dx_full_px": 6.0, "local_dy_full_px": -3.0,
        }
        px, py, converged, _, residual, _ = replay_local_forward_point(base, 480.0, 384.0, meta)
        self.assertTrue(converged)
        self.assertLess(residual, 1e-6)
        mask = make_local_mask(base, meta)
        weight = bilinear(mask, px, py)
        self.assertAlmostEqual(px - weight * 6.0, 480.0, places=6)
        self.assertAlmostEqual(py + weight * 3.0, 384.0, places=6)
        global_meta = {"global_tx_full_px": 3.0, "global_ty_full_px": -4.0,
                       "angle_deg": 0.0, "scale_x": 1.0, "scale_y": 1.0}
        gx, gy = replay_global_point(global_local_base, 100.0, 200.0, global_meta)
        self.assertAlmostEqual(math.hypot(gx - 100.0, gy - 200.0), 5.0, places=4)

    def test_training_binding_rejects_leakage_and_never_requires_test_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("visible", "infrared", "labels"):
                (root / name).mkdir()
            splits = {"train": ["train_sample"], "val": ["validation_sample"], "test": ["test_sample"]}
            (root / "split.json").write_text(json.dumps(splits), encoding="utf-8")
            fields = ["key", *GT_FIELDS.values()]
            with (root / "manual.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for name in ("train_sample", "validation_sample"):
                    row = {column: 0.0 for column in GT_FIELDS.values()}
                    row.update(key=name, image_width=960, image_height=768, scale_x=1, scale_y=1)
                    writer.writerow(row)
                    # The binder checks existence; image decoding is tested separately.
                    for folder in ("visible", "infrared"):
                        (root / folder / (name + ".jpg")).touch()
                    (root / "labels" / (name + ".txt")).write_text("", encoding="utf-8")
            config = {
                "dataset_root": ".", "visible_dir": "visible", "infrared_dir": "infrared",
                "label_dir": "labels", "manual_gt_csv": "manual.csv", "split_json": "split.json",
                "expected_counts": {"train": 1, "val": 1, "test": 1},
            }
            config_path = root / "training.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            pairs, counts = load_training_protocol(config_path)
            self.assertEqual(set(pairs), {"train", "val"})
            self.assertEqual(counts["test"], 1)
            self.assertFalse((root / "visible" / "test_sample.jpg").exists())
            splits["test"] = ["train_sample"]
            (root / "split.json").write_text(json.dumps(splits), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                load_training_protocol(config_path)
            with self.assertRaises(ValueError):
                validate_keys(["../outside"], "bad ID")

    def test_release_check_rejects_accidental_data_files(self):
        spec = importlib.util.spec_from_file_location("release_check", ROOT / "scripts/verify_code_only.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forbidden = root / "sample.png"
            forbidden.write_bytes(b"not-an-image")
            self.assertTrue(module.inspect_file(forbidden, root))
            config = root / "private.local.json"
            config.write_text("{}", encoding="utf-8")
            self.assertTrue(module.inspect_file(config, root))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_code_only.py")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_command_line_registration_and_known_five_pixel_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("visible", "infrared", "labels"):
                (root / name).mkdir()
            image = np.random.default_rng(31).integers(0, 256, (768, 960, 3), dtype=np.uint8)
            for name in ("visible", "infrared"):
                ok, encoded = cv2.imencode(".png", image)
                self.assertTrue(ok)
                encoded.tofile(str(root / name / "synthetic_cli.png"))
            (root / "labels/synthetic_cli.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (root / "ids.txt").write_text("synthetic_cli\n", encoding="utf-8")
            prior = {"base_angle_deg": 0.0, "base_scale_x": 1.0, "base_scale_y": 1.0}
            model = proposed.ProposedRegistrationNet(0.0, 1.0, 1.0)
            torch.save({"model": model.state_dict(), "prior_stats": prior}, root / "model.pt")
            (root / "calibration.json").write_text(json.dumps({
                "confidence_threshold_error_px": 1e12, "gate_threshold": 1.0,
            }), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/register.py"),
                       "--visible", str(root / "visible"), "--infrared", str(root / "infrared"),
                       "--labels", str(root / "labels"), "--ids", str(root / "ids.txt"),
                       "--checkpoint", str(root / "model.pt"),
                       "--calibration", str(root / "calibration.json"),
                       "--output", str(root / "registered"), "--device", "cpu"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            meta = json.loads((root / "registered/transforms/synthetic_cli.json").read_text())
            self.assertEqual(meta["final_source"], "global")
            for name in ("visible_path", "infrared_path", "label_path"):
                self.assertEqual(meta[name], Path(meta[name]).name)
            px, py = replay_global_point(global_local_base, 400.0, 300.0, meta)
            with (root / "centers.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "key", "visible_uav_cx", "visible_uav_cy", "ir_uav_cx_gt_raw", "ir_uav_cy_gt_raw",
                ])
                writer.writeheader()
                writer.writerow({"key": "synthetic_cli", "visible_uav_cx": px + 3.0,
                                 "visible_uav_cy": py + 4.0,
                                 "ir_uav_cx_gt_raw": 400.0, "ir_uav_cy_gt_raw": 300.0})
            evaluation = subprocess.run([
                sys.executable, str(ROOT / "scripts/evaluate.py"),
                "--gt", str(root / "centers.csv"), "--results", str(root / "registered"),
                "--output", str(root / "metrics"),
            ], capture_output=True, text=True)
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
            metrics = json.loads((root / "metrics/metrics.json").read_text())
            self.assertEqual(metrics["pairs"], 1)
            self.assertAlmostEqual(metrics["mean_tre_px"], 5.0, places=5)
            self.assertAlmostEqual(metrics["rmse_tre_px"], 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
