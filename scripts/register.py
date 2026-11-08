"""Register user-supplied infrared images; manual registration GT is not read."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from target_registration.data import validate_keys
from target_registration.inference import imwrite_png, load_model, register_one
from target_registration.training import fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible", required=True, type=Path)
    parser.add_argument("--infrared", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="Visible YOLO target boxes")
    parser.add_argument("--ids", required=True, type=Path, help="One pair ID per line")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit")
    args = parser.parse_args()
    if args.threads < 1 or args.limit < 0:
        parser.error("threads must be positive and limit must be nonnegative")
    keys = validate_keys(args.ids.read_text(encoding="utf-8-sig").split(), "inference IDs")
    if args.limit:
        keys = keys[:args.limit]
    torch.set_num_threads(args.threads)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
    for name in ("confidence_threshold_error_px", "gate_threshold"):
        value = calibration.get(name)
        if not isinstance(value, (int, float)) or not float("-inf") < value < float("inf"):
            raise ValueError(f"Missing or invalid calibrated value: {name}")
    if not 0 <= calibration["gate_threshold"] <= 1:
        raise ValueError("gate_threshold must be between 0 and 1")
    model, prior = load_model(args.checkpoint, device)
    output = fresh_output(args.output)
    (output / "registered_ir").mkdir()
    (output / "transforms").mkdir()
    rows = []
    started = time.perf_counter()
    for index, key in enumerate(keys, 1):
        image, metadata = register_one(
            key, model, prior, calibration, device,
            args.visible, args.infrared, args.labels,
        )
        imwrite_png(output / "registered_ir" / f"{key}.png", image)
        # Do not store workstation paths in generated metadata.
        for name in ("visible_path", "infrared_path", "label_path"):
            metadata[name] = Path(metadata[name]).name
        metadata.update(prior_stats=prior, evaluation_gt_used=False)
        (output / "transforms" / f"{key}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        rows.append({name: metadata[name] for name in (
            "key", "final_source", "confidence_error_px", "local_gate",
            "model_runtime_ms", "full_warp_runtime_ms",
        )})
        print(f"[{index}/{len(keys)}] {key}: {metadata['final_source']}", flush=True)
    with (output / "registration.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "pairs": len(rows), "device": str(device),
        "seconds": time.perf_counter() - started, "manual_gt_read": False,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
