"""Evaluate predicted registration fields against user-supplied target centers."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from target_registration.core import base, global_local_base
from target_registration.point_metrics import replay_global_point, replay_local_forward_point
from target_registration.training import fresh_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    gt = {}
    with args.gt.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["key"].strip()
            if key in gt:
                raise ValueError(f"Duplicate target annotation: {key}")
            gt[key] = row
    files = sorted((args.results / "transforms").glob("*.json"))
    if not files:
        raise ValueError("No registration transforms found; run register.py first")
    if not args.allow_subset and {path.stem for path in files} != set(gt):
        raise ValueError("GT and result IDs differ; use a matching GT file or --allow-subset")
    rows = []
    for path in files:
        meta = json.loads(path.read_text(encoding="utf-8-sig"))
        key = meta["key"]
        if key != path.stem:
            raise ValueError("Transform filename and key do not match")
        target = gt[key]
        if not (args.results / "registered_ir" / f"{key}.png").is_file():
            raise FileNotFoundError(f"Missing registered output: {key}")
        ix, iy = float(target["ir_uav_cx_gt_raw"]), float(target["ir_uav_cy_gt_raw"])
        vx, vy = float(target["visible_uav_cx"]), float(target["visible_uav_cy"])
        if not all(math.isfinite(value) for value in (ix, iy, vx, vy)):
            raise ValueError(f"Non-finite target point: {key}")
        source = meta["final_source"]
        converged, residual = True, 0.0
        if source == "fallback_identity":
            px, py = ix, iy
        elif source == "fallback_prior":
            # Require the actual checkpoint prior; never invent a fallback prior.
            prior = meta["prior_stats"]
            prior_meta = dict(
                meta, global_tx_full_px=0.0, global_ty_full_px=0.0,
                angle_deg=prior["base_angle_deg"],
                scale_x=prior["base_scale_x"], scale_y=prior["base_scale_y"],
            )
            px, py = replay_global_point(global_local_base, ix, iy, prior_meta)
        elif source in ("global", "highres_target"):
            px, py = replay_global_point(global_local_base, ix, iy, meta)
            if source == "highres_target":
                px, py, converged, _, residual, _ = replay_local_forward_point(base, px, py, meta)
        else:
            raise ValueError(f"Unknown registration output: {source}")
        rows.append({
            "key": key, "final_source": source, "tre_px": math.hypot(px - vx, py - vy),
            "pred_x": px, "pred_y": py, "visible_gt_x": vx, "visible_gt_y": vy,
            "solver_converged": converged, "solver_residual_px": residual,
        })
    values = np.asarray([row["tre_px"] for row in rows], dtype=np.float64)
    summary = {
        "pairs": len(rows), "mean_tre_px": float(values.mean()),
        "median_tre_px": float(np.median(values)),
        "rmse_tre_px": float(np.sqrt(np.mean(values ** 2))),
        "p95_tre_px": float(np.percentile(values, 95)),
        "success_le_5px": float(np.mean(values <= 5)),
        "success_le_10px": float(np.mean(values <= 10)),
        "solver_nonconverged_count": sum(not row["solver_converged"] for row in rows),
        "coordinate_system": "original 960 x 768 image pixels",
        "manual_gt_used_only_after_inference": True,
    }
    output = fresh_output(args.output)
    with (output / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
