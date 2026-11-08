"""Proposed target-aware registration model and target-local optimization."""

from __future__ import annotations

import csv

import hashlib

import json

import math

import os

import random

import shutil

import sys

import time

from datetime import datetime

from pathlib import Path

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from . import global_local_base

from . import base

TRAINING_OUTPUT_ROOT = Path("outputs/proposed")

BASE_CHECKPOINT = Path("weights/base_initialization.pt")

EPOCHS = 15               # Final-stage epoch budget.

LR = 1e-4

WEIGHT_DECAY = 1e-4

BATCH_SIZE = 2            # Training batch size.

LOCAL_SUPERVISION_MAX_FULL_PX = 45.0

TARGET_NATIVE_WINDOW_F4 = 24

TARGET_LOCAL_RADIUS_F4 = 4

TARGET_F4_STRIDE = 4.0

ENTROPY_WEIGHT = 0.001

EDGE_RADIUS_NET_PX = 2.0

EDGE_STEP_NET_PX = 0.5

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def native_f4_window(feat, bbox, has_bbox, size=TARGET_NATIVE_WINDOW_F4):
    """Extract fixed-size windows directly from the 1/4-resolution feature maps."""
    b, _, h, w = feat.shape
    cx = ((bbox[:, 0] + bbox[:, 2]) * 0.5 * w - 0.5).round()
    cy = ((bbox[:, 1] + bbox[:, 3]) * 0.5 * h - 0.5).round()
    cx = torch.where(has_bbox > 0, cx, torch.full_like(cx, (w - 1) * 0.5))
    cy = torch.where(has_bbox > 0, cy, torch.full_like(cy, (h - 1) * 0.5))
    ox = torch.arange(size, device=feat.device, dtype=feat.dtype) - (size // 2)
    oy = torch.arange(size, device=feat.device, dtype=feat.dtype) - (size // 2)
    yy, xx = torch.meshgrid(oy, ox, indexing="ij")
    gx = cx[:, None, None] + xx[None]
    gy = cy[:, None, None] + yy[None]
    nx = (gx + 0.5) * (2.0 / w) - 1.0
    ny = (gy + 0.5) * (2.0 / h) - 1.0
    grid = torch.stack([nx, ny], dim=-1)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

class TargetLocalCorrelationF4(nn.Module):
    def __init__(self):
        super().__init__()
        self.corr = base.CorrelationSoftArgmax(
            in_dim=base.F4_DIM,
            proj_dim=base.CORR_DIM_FINE,
            radius=TARGET_LOCAL_RADIUS_F4,
            temperature=global_local_base.FINE_TEMPERATURE,
            shift_prior_sigma=float(TARGET_LOCAL_RADIUS_F4),
        )

    def initialize_from_fine(self, fine_corr):
        source = {k: v for k, v in fine_corr.state_dict().items() if k != "shift_xy"}
        missing, unexpected = self.corr.load_state_dict(source, strict=False)
        if unexpected or any(k != "shift_xy" for k in missing):
            raise RuntimeError(f"Fine-correlation init mismatch: missing={missing}, unexpected={unexpected}")
        return {"source": "E1 fine_corr projection/confidence weights", "missing": list(missing), "unexpected": list(unexpected)}

    def forward(self, visible_f4, global_ir_f4, bbox, has_bbox):
        rv = native_f4_window(visible_f4, bbox, has_bbox)
        ri = native_f4_window(global_ir_f4, bbox, has_bbox)
        out = self.corr(rv, ri)
        dx = out["dx_feature"] * TARGET_F4_STRIDE * has_bbox
        dy = out["dy_feature"] * TARGET_F4_STRIDE * has_bbox
        return {**out, "raw_dx": dx, "raw_dy": dy, "candidate_dx": dx, "candidate_dy": dy}

def minimum_edge_bbox(bbox, h, w):
    x1, y1, x2, y2 = bbox.unbind(1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1).clamp(min=16.0 / w)
    bh = (y2 - y1).clamp(min=16.0 / h)
    return torch.stack([
        (cx - bw*0.5).clamp(0, 1),
        (cy - bh*0.5).clamp(0, 1),
        (cx + bw*0.5).clamp(0, 1),
        (cy + bh*0.5).clamp(0, 1)
    ], dim=1)

def vectorized_edge_micro_refine(visible, global_ir, bbox, has_bbox, initial_dx, initial_dy):
    offsets = torch.arange(-EDGE_RADIUS_NET_PX, EDGE_RADIUS_NET_PX + 1e-6, EDGE_STEP_NET_PX, device=visible.device, dtype=visible.dtype)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    candidates = torch.stack([ox.flatten(), oy.flatten()], dim=1)
    b, c, h, w = global_ir.shape
    k = candidates.shape[0]
    total_dx = initial_dx[:, None] + candidates[None, :, 0]
    total_dy = initial_dy[:, None] + candidates[None, :, 1]
    images = global_ir[:, None].expand(-1, k, -1, -1, -1).reshape(b*k, c, h, w)
    boxes = bbox[:, None].expand(-1, k, -1).reshape(b*k, 4)
    flags = has_bbox[:, None].expand(-1, k).reshape(b*k)
    warped = base.gaussian_local_warp(images, boxes, flags, total_dx.reshape(-1), total_dy.reshape(-1), radius_expand=3.0)
    vis_edge = base.sobel_mag(base.rgb_to_gray(visible))
    ref = vis_edge[:, None].expand(-1, k, -1, -1, -1).reshape(b*k, 1, h, w)
    mov = base.sobel_mag(base.rgb_to_gray(warped))
    edge_boxes = minimum_edge_bbox(bbox, h, w)
    edge_boxes = edge_boxes[:, None].expand(-1, k, -1).reshape(b*k, 4)
    mask = base.gaussian_target_mask(edge_boxes, flags, h, w, expand=3.0)
    score = base.weighted_ncc_torch(ref, mov, mask).reshape(b, k)
    best = score.argmax(dim=1)
    chosen = candidates[best]
    final_dx = (initial_dx + chosen[:, 0]) * has_bbox
    final_dy = (initial_dy + chosen[:, 1]) * has_bbox
    return final_dx, final_dy, score.max(dim=1).values

class ProposedRegistrationNet(global_local_base.BaseGlobalLocalRegistrationNet):
    def __init__(self, base_angle_deg, base_scale_x, base_scale_y):
        super().__init__(base_angle_deg, base_scale_x, base_scale_y)
        self.target_local_corr = TargetLocalCorrelationF4()
        self.edge_enabled = False

    def forward(self, visible, infrared, bbox, has_bbox, enable_local=True):
        fv = self.visible_encoder(base.model_input(visible))
        fi0 = self.thermal_encoder(base.model_input(infrared))
        as_out = self.angle_scale_head(fv["f8"], fi0["f8"])
        z = torch.zeros_like(as_out["angle"])
        ir_rs = global_local_base.warp_anisotropic_px(infrared, z, z, as_out["angle"], as_out["scale_x"], as_out["scale_y"])
        fi_rs = self.thermal_encoder(base.model_input(ir_rs))
        coarse = self.coarse_corr(fv["f8"], fi_rs["f8"])
        coarse_tx = coarse["dx_feature"] * global_local_base.COARSE_STRIDE
        coarse_ty = coarse["dy_feature"] * global_local_base.COARSE_STRIDE
        ir_coarse = global_local_base.warp_anisotropic_px(infrared, coarse_tx, coarse_ty, as_out["angle"], as_out["scale_x"], as_out["scale_y"])
        fi_coarse = self.thermal_encoder(base.model_input(ir_coarse))
        fine = self.fine_corr(fv["f4"], fi_coarse["f4"])
        fine_tx = fine["dx_feature"] * global_local_base.FINE_STRIDE
        fine_ty = fine["dy_feature"] * global_local_base.FINE_STRIDE
        tx = coarse_tx + fine_tx
        ty = coarse_ty + fine_ty
        global_ir = global_local_base.warp_anisotropic_px(infrared, tx, ty, as_out["angle"], as_out["scale_x"], as_out["scale_y"])
        vis_edge = base.sobel_mag(base.rgb_to_gray(visible))
        ir_edge = base.sobel_mag(base.rgb_to_gray(infrared))
        global_edge = base.sobel_mag(base.rgb_to_gray(global_ir))
        ncc_before = base.weighted_ncc_torch(vis_edge, ir_edge)
        ncc_global = base.weighted_ncc_torch(vis_edge, global_edge)
        conf_desc = torch.stack([coarse["peak_prob"], coarse["entropy"], fine["peak_prob"],
                                 fine["entropy"], ncc_before, ncc_global], dim=1)
        confidence_error_px = self.confidence_head(conf_desc)
        geo = {
            "tx": tx, "ty": ty,
            "coarse_tx": coarse_tx, "coarse_ty": coarse_ty,
            "fine_tx": fine_tx, "fine_ty": fine_ty,
            "angle": as_out["angle"],
            "scale_x": as_out["scale_x"],
            "scale_y": as_out["scale_y"],
            "angle_delta": as_out["angle_delta"],
            "log_scale_x_delta": as_out["log_scale_x_delta"],
            "log_scale_y_delta": as_out["log_scale_y_delta"]
        }
        if enable_local:
            fi_global = self.thermal_encoder(base.model_input(global_ir))
            old = self.target_head(fv["f8"], fi_global["f8"], bbox, has_bbox)
            local = self.target_local_corr(fv["f4"], fi_global["f4"], bbox, has_bbox)
            local["gate"] = old["gate"]
            local["gate_logit"] = old["gate_logit"]
            local["old_raw_dx"] = old["raw_dx"]
            local["old_raw_dy"] = old["raw_dy"]
            if self.edge_enabled:
                dx, dy, edge_score = vectorized_edge_micro_refine(
                    visible, global_ir, bbox, has_bbox, local["raw_dx"], local["raw_dy"])
                local["pre_edge_dx"] = local["raw_dx"]
                local["pre_edge_dy"] = local["raw_dy"]
                local["raw_dx"] = dx
                local["raw_dy"] = dy
                local["candidate_dx"] = dx
                local["candidate_dy"] = dy
                local["edge_search_score"] = edge_score
            target_candidate_ir = base.gaussian_local_warp(global_ir, bbox, has_bbox,
                                                            local["candidate_dx"], local["candidate_dy"], radius_expand=3.0)
        else:
            zz = torch.zeros(visible.shape[0], device=visible.device, dtype=visible.dtype)
            local = {
                "raw_dx": zz, "raw_dy": zz,
                "candidate_dx": zz, "candidate_dy": zz,
                "gate": zz, "gate_logit": zz,
                "entropy": zz, "peak_prob": zz
            }
            target_candidate_ir = global_ir
        return {
            "global_ir": global_ir,
            "target_candidate_ir": target_candidate_ir,
            "geo": geo,
            "local": local,
            "target_local_corr": local,
            "coarse_corr": coarse,
            "fine_corr": fine,
            "ncc_before": ncc_before,
            "ncc_global": ncc_global,
            "confidence_error_px": confidence_error_px
        }

def build_model(checkpoint, device):
    prior = checkpoint["prior_stats"]
    model = ProposedRegistrationNet(prior["base_angle_deg"], prior["base_scale_x"], prior["base_scale_y"]).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    # Only the new target-local branch may be absent from the base checkpoint.
    expected_missing = {f"target_local_corr.corr.{k}" for k in model.target_local_corr.corr.state_dict()}
    missing_set = set(missing)
    if unexpected or (missing_set != expected_missing and missing_set != set()):
        # Reject incompatible checkpoint keys.
        if not missing_set.issubset(expected_missing) or unexpected:
            raise RuntimeError(f"E1 load mismatch missing={missing} unexpected={unexpected}")
    # Initialize the target-local branch from fine-correlation parameters.
    init = model.target_local_corr.initialize_from_fine(model.fine_corr)
    # Update the target-local correlation branch only.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.target_local_corr.parameters():
        p.requires_grad = True
    return model, prior, init

def metric_summary(values):
    """Summarize target errors and tolerance rates."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "rmse": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "le_2px": 0,
            "le_5px": 0,
            "le_10px": 0,
            "le_20px": 0,
            "le_2px_ratio": 0.0,
            "le_5px_ratio": 0.0,
            "le_10px_ratio": 0.0,
            "le_20px_ratio": 0.0,
            "gt_50px": 0,
            "gt_100px": 0,
        }
    n = int(a.size)
    le_2 = int(np.sum(a <= 2.0))
    le_5 = int(np.sum(a <= 5.0))
    le_10 = int(np.sum(a <= 10.0))
    le_20 = int(np.sum(a <= 20.0))
    return {
        "n": n,
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "rmse": float(np.sqrt(np.mean(a * a))),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "le_2px": le_2,
        "le_2px_ratio": le_2 / n,
        "le_5px": le_5,
        "le_5px_ratio": le_5 / n,
        "le_10px": le_10,
        "le_10px_ratio": le_10 / n,
        "le_20px": le_20,
        "le_20px_ratio": le_20 / n,
        "gt_50px": int(np.sum(a > 50.0)),
        "gt_100px": int(np.sum(a > 100.0)),
    }

def fixed_conf_calibration(records, confidence_threshold):
    best = None
    for gate in np.linspace(global_local_base.GATE_SCAN_MIN, global_local_base.GATE_SCAN_MAX, global_local_base.GATE_SCAN_POINTS):
        vals = []
        for r in records:
            if r["confidence_error_px"] > confidence_threshold:
                vals.append(r["fallback_tre"])
            elif r["gate"] >= gate:
                vals.append(r["target_candidate_tre"])
            else:
                vals.append(r["full_global_tre"])
        item = {
            "gate_threshold": float(gate),
            "mean": float(np.mean(vals)),
            "metrics": metric_summary(vals),
            "target_count": int(sum(r["confidence_error_px"] <= confidence_threshold and r["gate"] >= gate for r in records)),
            "fallback_count": int(sum(r["confidence_error_px"] > confidence_threshold for r in records))
        }
        if best is None or item["mean"] < best["mean"]:
            best = item
    return {
        "source": "VALIDATION_ONLY",
        "confidence_threshold_error_px": float(confidence_threshold),
        **best
    }

def apply_calibration(records, calibration):
    out = []
    for r in records:
        row = dict(r)
        if r["confidence_error_px"] > calibration["confidence_threshold_error_px"]:
            row["final_tre"] = r["fallback_tre"]
            row["final_source"] = r["fallback_kind"]
        elif r["gate"] >= calibration["gate_threshold"]:
            row["final_tre"] = r["target_candidate_tre"]
            row["final_source"] = "highres_target"
        else:
            row["final_tre"] = r["full_global_tre"]
            row["final_source"] = "global"
        row["target_applied"] = int(row["final_source"] == "highres_target")
        out.append(row)
    return out

def target_training_loss(outputs, batch):
    geo = outputs["geo"]
    local = outputs["local"]
    pred = global_local_base.transform_points_anisotropic(
        batch["ir_uav_raw"],
        geo["tx"], geo["ty"],
        geo["angle"], geo["scale_x"], geo["scale_y"]
    )
    target = batch["vis_uav_center"] - pred.detach()
    full = target / batch["raw_to_input_scale"].clamp(min=1e-6)
    norm = torch.sqrt((full*full).sum(dim=1) + 1e-6)
    valid = batch["has_bbox"] * (norm <= LOCAL_SUPERVISION_MAX_FULL_PX).to(target.dtype)
    dx = F.smooth_l1_loss(local["raw_dx"], target[:, 0], reduction="none")
    dy = F.smooth_l1_loss(local["raw_dy"], target[:, 1], reduction="none")
    regression = global_local_base.masked_mean(dx + dy, valid)
    entropy = global_local_base.masked_mean(local["entropy"], valid)
    return regression + ENTROPY_WEIGHT * entropy, {
        "regression": regression,
        "entropy": entropy,
        "valid": valid.sum(),
        "target_norm_full": norm.mean()
    }

def collect_records(model, loader, device, prior):
    model.eval()
    with torch.inference_mode():
        return global_local_base.collect_eval_records(model, loader, device, prior)

def compute_edge_alignment(model, loader, device, calibration):
    scores = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            keys = list(batch["key"])
            batch = global_local_base.move_batch_to_device(batch, device)
            out = model(batch["visible"], batch["infrared"], batch["bbox"], batch["has_bbox"], True)
            vis_edge = base.sobel_mag(base.rgb_to_gray(batch["visible"]))
            mask = base.gaussian_target_mask(
                minimum_edge_bbox(batch["bbox"], global_local_base.INPUT_H, global_local_base.INPUT_W),
                batch["has_bbox"], global_local_base.INPUT_H, global_local_base.INPUT_W, expand=3.0
            )
            for i, key in enumerate(keys):
                if float(out["confidence_error_px"][i]) > calibration["confidence_threshold_error_px"]:
                    final = batch["infrared"][i:i+1]
                elif float(out["local"]["gate"][i]) >= calibration["gate_threshold"]:
                    final = out["target_candidate_ir"][i:i+1]
                else:
                    final = out["global_ir"][i:i+1]
                e = base.sobel_mag(base.rgb_to_gray(final))
                scores[key] = float(base.weighted_ncc_torch(vis_edge[i:i+1], e, mask[i:i+1]).cpu())
    return scores

def measure_runtime(model, pairs, device):
    loader = global_local_base.make_loader(pairs, augment=False, shuffle=False, batch_size=1)
    batches = list(loader)
    model.eval()
    with torch.inference_mode():
        for batch in batches[:5]:
            batch = global_local_base.move_batch_to_device(batch, device)
            model(batch["visible"], batch["infrared"], batch["bbox"], batch["has_bbox"], True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        values = []
        for batch in batches:
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()
            batch = global_local_base.move_batch_to_device(batch, device)
            model(batch["visible"], batch["infrared"], batch["bbox"], batch["has_bbox"], True)
            torch.cuda.synchronize() if device.type == "cuda" else None
            values.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples": len(values),
        "mean_runtime_ms": float(np.mean(values)),
        "min_runtime_ms": float(np.min(values)),
        "max_runtime_ms": float(np.max(values)),
        "per_pair_ms": values
    }

def train_seed(seed, train_pairs, val_pairs, checkpoint, device, confidence_threshold, edge=False):
    """Train the local branch using the fixed global reliability threshold."""
    global_local_base.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model, prior, init = build_model(checkpoint, device)
    model.edge_enabled = False
    train_loader = global_local_base.make_loader(train_pairs, augment=True, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = global_local_base.make_loader(val_pairs, augment=False, shuffle=False, batch_size=BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.target_local_corr.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # Retain the global reliability threshold from the base model.
    global_conf_threshold = confidence_threshold
    seed_dir = TRAINING_OUTPUT_ROOT / "TARGET_LOCAL_CORRELATION_STAGE" / f"seed_{seed}"
    (seed_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    for epoch in range(1, EPOCHS + 1):
        model.eval()
        model.target_local_corr.train()
        total = reg = ent = seen = 0.0
        for batch in train_loader:
            batch = global_local_base.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                outputs = model(batch["visible"], batch["infrared"], batch["bbox"], batch["has_bbox"], True)
                loss, detail = target_training_loss(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.target_local_corr.parameters(), global_local_base.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            bs = batch["visible"].shape[0]
            seen += bs
            total += float(loss.detach()) * bs
            reg += float(detail["regression"].detach()) * bs
            ent += float(detail["entropy"].detach()) * bs
        scheduler.step()
        records = collect_records(model, val_loader, device, prior)
        cal = fixed_conf_calibration(records, global_conf_threshold)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": total / seen,
            "train_regression": reg / seen,
            "train_entropy": ent / seen,
            "mean_tre": cal["metrics"]["mean"],
            "rmse": cal["metrics"]["rmse"],
            "median": cal["metrics"]["median"],
            "p95": cal["metrics"]["p95"],
            "max": cal["metrics"]["max"],
            "confidence_threshold": global_conf_threshold,
            "gate_threshold": cal["gate_threshold"],
            "target_count": cal["target_count"],
            "fallback_count": cal["fallback_count"]
        }
        history.append(row)
        payload = {
            "model": model.state_dict(),
            "prior_stats": prior,
            "epoch": epoch,
            "seed": seed,
            "experiment": "TARGET_LOCAL_CORRELATION_STAGE",
            "selection_metrics": row,
            "calibration": cal,
            "initialization": init,
            "base_checkpoint_sha256": sha256(BASE_CHECKPOINT),
            "test_data_accessed": False,
            "fresh_test_accessed": False,
            "created": datetime.now().isoformat(timespec="seconds")
        }
        torch.save(payload, seed_dir / "checkpoints" / f"epoch_{epoch:02d}.pt")
        if best is None or row["mean_tre"] < best["metrics"]["mean"]:
            torch.save(payload, seed_dir / "best.pt")
            best = {
                "metrics": cal["metrics"],
                "calibration": cal,
                "epoch": epoch,
                "records": apply_calibration(records, cal),
                "history": list(history)
            }
        print(f"[Proposed seed={seed} epoch={epoch:02d}/15] mean={row['mean_tre']:.6f} "
              f"rmse={row['rmse']:.6f} gate={row['gate_threshold']:.2f} best={best['epoch']==epoch}", flush=True)
    best_ck = torch.load(seed_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ck["model"], strict=True)
    model.eval()
    best["runtime"] = measure_runtime(model, val_pairs, device)
    edge_scores = compute_edge_alignment(model, val_loader, device, best["calibration"])
    for r in best["records"]:
        r["local_edge_alignment_score"] = edge_scores[r["key"]]
    return model, prior, best, seed_dir
