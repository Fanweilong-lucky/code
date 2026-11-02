"""Dependency-closed V32 research core; numerical definitions retained from the frozen experiment."""

from __future__ import annotations

import csv

import json

import math

import os

import random

from datetime import datetime

from pathlib import Path

import cv2

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from . import base

INPUT_W = 480

INPUT_H = 384

BATCH_SIZE = 2

NUM_WORKERS = 0

FALLBACK_ANGLE_DEG = 0.0

FALLBACK_SCALE_X = 0.955

FALLBACK_SCALE_Y = 0.970

MAX_ANGLE_DELTA_DEG = 1.50

MAX_LOG_SCALE_X_DELTA = 0.085

MAX_LOG_SCALE_Y_DELTA = 0.085

COARSE_STRIDE = 8

FINE_STRIDE = 4

COARSE_RADIUS = 6

FINE_RADIUS = 2

COARSE_TEMPERATURE = 0.055

FINE_TEMPERATURE = 0.045

SHIFT_PRIOR_SIGMA_COARSE = 4.5

SHIFT_PRIOR_SIGMA_FINE = 2.0

ROI_EXPAND = 4.0

ROI_FEATURE_SIZE = 12

MAX_LOCAL_DX_PX = 20.0

MAX_LOCAL_DY_PX = 20.0

TARGET_REQUIRED_FULL_PX = 5.0

LOCAL_SUPERVISION_MAX_FULL_PX = 45.0

USE_TRANSLATION_JITTER = True

JITTER_TX_PX = 6.0

JITTER_TY_PX = 6.0

GRAD_CLIP_NORM = 5.0

W_GLOBAL_STRUCTURE = 0.60

W_HUMAN_PARAM = 2.50

W_HUMAN_GRID = 2.20

W_COARSE_TRANSLATION = 0.45

W_PRIOR_REG = 0.010

W_CORR_ENTROPY = 0.003

W_CONFIDENCE = 0.65

W_TARGET_STRUCTURE = 0.25

W_HUMAN_LOCAL = 2.50

W_HUMAN_GATE = 1.40

GATE_SCAN_MIN = 0.05

GATE_SCAN_MAX = 0.95

GATE_SCAN_POINTS = 91

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def train_prior_statistics(train_pairs):
    if not train_pairs:
        return {
            "base_angle_deg": FALLBACK_ANGLE_DEG,
            "base_scale_x": FALLBACK_SCALE_X,
            "base_scale_y": FALLBACK_SCALE_Y,
        }

    angles = np.array([p["human_gt"]["angle_deg"] for p in train_pairs], dtype=np.float64)
    sx = np.array([p["human_gt"]["scale_x"] for p in train_pairs], dtype=np.float64)
    sy = np.array([p["human_gt"]["scale_y"] for p in train_pairs], dtype=np.float64)

    return {
        "base_angle_deg": float(np.median(angles)),
        "base_scale_x": float(np.median(sx)),
        "base_scale_y": float(np.median(sy)),
    }

def fallback_bbox_around_center(cx, cy, raw_w, raw_h, size_full_px=32.0):
    half = size_full_px / 2.0
    return [
        max(0.0, cx - half),
        max(0.0, cy - half),
        min(raw_w, cx + half),
        min(raw_h, cy + half),
    ]

def linear_part_np(angle_deg, scale_x, scale_y):
    th = math.radians(float(angle_deg))
    c = math.cos(th)
    s = math.sin(th)
    return np.array(
        [[c * float(scale_x), s * float(scale_y)],
         [-s * float(scale_x), c * float(scale_y)]],
        dtype=np.float64,
    )

class H30THumanDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        item = self.pairs[index]
        gt = item["human_gt"]

        vis_bgr = imread_u(item["visible"])
        ir_bgr = imread_u(item["infrared"])

        if ir_bgr.shape[:2] != vis_bgr.shape[:2]:
            ir_bgr = cv2.resize(
                ir_bgr,
                (vis_bgr.shape[1], vis_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        raw_h, raw_w = vis_bgr.shape[:2]
        sx_img = INPUT_W / raw_w
        sy_img = INPUT_H / raw_h

        # H30T data should preserve aspect ratio when resized.
        if abs(sx_img - sy_img) > 1e-4:
            raise RuntimeError(
                f"Non-uniform image resize is not supported in V3.1: "
                f"{item['key']} sx={sx_img}, sy={sy_img}"
            )

        # 全局参数（直接从 CSV 读取，作为 Global 监督标签）
        tx = gt["tx_px"] * sx_img
        ty = gt["ty_px"] * sy_img
        angle = gt["angle_deg"]
        scale_x = gt["scale_x"]
        scale_y = gt["scale_y"]

        # 红外中心点（用于后续可能的计算，但局部监督改用 local_residual）
        ir_center = np.array(
            [gt["ir_uav_cx_gt_raw"] * sx_img,
             gt["ir_uav_cy_gt_raw"] * sy_img],
            dtype=np.float32,
        )

        vis_center = np.array(
            [gt["visible_uav_cx"] * sx_img,
             gt["visible_uav_cy"] * sy_img],
            dtype=np.float32,
        )

        # 局部残差（直接从 CSV 读取，作为 Local 监督标签）
        local_residual = np.array(
            [gt["local_residual_dx_px"] * sx_img,
             gt["local_residual_dy_px"] * sy_img],
            dtype=np.float32,
        )

        bbox = base.load_txt_bbox(item["label"], vis_bgr.shape)
        if bbox is None:
            bbox = fallback_bbox_around_center(
                gt["visible_uav_cx"],
                gt["visible_uav_cy"],
                raw_w,
                raw_h,
            )

        x1, y1, x2, y2 = bbox
        bbox_norm = np.array(
            [x1 / raw_w, y1 / raw_h, x2 / raw_w, y2 / raw_h],
            dtype=np.float32,
        )

        vis_bgr = cv2.resize(vis_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        ir_bgr = cv2.resize(ir_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)

        # 平移抖动增强（保持不变）
        if self.augment and USE_TRANSLATION_JITTER:
            jx = random.uniform(-JITTER_TX_PX, JITTER_TX_PX)
            jy = random.uniform(-JITTER_TY_PX, JITTER_TY_PX)
            J = np.array([[1.0, 0.0, jx], [0.0, 1.0, jy]], dtype=np.float32)

            ir_bgr = cv2.warpAffine(
                ir_bgr,
                J,
                (INPUT_W, INPUT_H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            A = linear_part_np(angle, scale_x, scale_y)
            dt = A @ np.array([jx, jy], dtype=np.float64)
            tx -= float(dt[0])
            ty -= float(dt[1])
            ir_center[0] += jx
            ir_center[1] += jy

        vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ir = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.augment:
            if random.random() < 0.60:
                gamma = random.uniform(0.80, 1.20)
                vis = np.clip(vis ** gamma, 0, 1)
            if random.random() < 0.60:
                gamma = random.uniform(0.80, 1.20)
                ir = np.clip(ir ** gamma, 0, 1)
            if random.random() < 0.40:
                noise = np.random.normal(0.0, 0.008, ir.shape).astype(np.float32)
                ir = np.clip(ir + noise, 0, 1)

        return {
            "key": item["key"],
            "visible": torch.from_numpy(vis.transpose(2, 0, 1)),
            "infrared": torch.from_numpy(ir.transpose(2, 0, 1)),
            "bbox": torch.from_numpy(bbox_norm),
            "has_bbox": torch.tensor(1.0, dtype=torch.float32),
            "human_geo": torch.tensor(
                [tx, ty, angle, scale_x, scale_y], dtype=torch.float32
            ),
            "ir_uav_raw": torch.from_numpy(ir_center),
            "vis_uav_center": torch.from_numpy(vis_center),
            "raw_to_input_scale": torch.tensor([sx_img, sy_img], dtype=torch.float32),
            "raw_size": torch.tensor([raw_w, raw_h], dtype=torch.float32),
            # 新增：人工局部残差标签
            "local_residual_gt": torch.from_numpy(local_residual),
        }

def warp_anisotropic_px(image, tx_px, ty_px, angle_deg, scale_x, scale_y):
    """
    Human annotation convention:
        p_out = R(angle) @ S(scale_x, scale_y) @ (p_in - center)
                + center + [tx, ty]
    center is exactly (W/2, H/2), matching the manual GT tool.
    """
    B, C, H, W = image.shape
    dtype = image.dtype
    device = image.device

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    cx = W / 2.0
    cy = H / 2.0

    a = angle_deg * math.pi / 180.0
    c = torch.cos(a).view(B, 1, 1)
    s = torch.sin(a).view(B, 1, 1)
    sx = scale_x.view(B, 1, 1).clamp(0.75, 1.25)
    sy = scale_y.view(B, 1, 1).clamp(0.75, 1.25)
    tx = tx_px.view(B, 1, 1)
    ty = ty_px.view(B, 1, 1)

    xo = xx - cx - tx
    yo = yy - cy - ty

    # inverse R, then inverse S
    xr = c * xo - s * yo
    yr = s * xo + c * yo
    xi = xr / sx + cx
    yi = yr / sy + cy

    xn = 2.0 * (xi + 0.5) / W - 1.0
    yn = 2.0 * (yi + 0.5) / H - 1.0
    grid = torch.stack([xn, yn], dim=-1)

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

def transform_points_anisotropic(
    points_xy, tx_px, ty_px, angle_deg, scale_x, scale_y,
    width=INPUT_W, height=INPUT_H
):
    """points_xy: [B,N,2] or [B,2]."""
    squeeze = False
    if points_xy.ndim == 2:
        points_xy = points_xy.unsqueeze(1)
        squeeze = True

    cx = width / 2.0
    cy = height / 2.0
    center = points_xy.new_tensor([cx, cy])
    p = points_xy - center

    x = p[..., 0] * scale_x.view(-1, 1)
    y = p[..., 1] * scale_y.view(-1, 1)

    a = angle_deg * math.pi / 180.0
    c = torch.cos(a).view(-1, 1)
    s = torch.sin(a).view(-1, 1)

    xo = c * x + s * y + cx + tx_px.view(-1, 1)
    yo = -s * x + c * y + cy + ty_px.view(-1, 1)
    out = torch.stack([xo, yo], dim=-1)

    return out[:, 0] if squeeze else out

def standard_grid_points(batch_size, device, dtype):
    xs = torch.tensor([0.10, 0.50, 0.90], device=device, dtype=dtype) * INPUT_W
    ys = torch.tensor([0.10, 0.50, 0.90], device=device, dtype=dtype) * INPUT_H
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return pts.unsqueeze(0).expand(batch_size, -1, -1)

class AngleAnisoScaleHead(nn.Module):
    def __init__(self, base_angle_deg, base_scale_x, base_scale_y):
        super().__init__()
        self.register_buffer("base_angle_deg", torch.tensor(float(base_angle_deg)))
        self.register_buffer("base_scale_x", torch.tensor(float(base_scale_x)))
        self.register_buffer("base_scale_y", torch.tensor(float(base_scale_y)))

        self.mlp = nn.Sequential(
            nn.Linear(base.F8_DIM * 3, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, 3),
        )

    def forward(self, fv8, fi8):
        gv = fv8.mean(dim=(2, 3))
        gi = fi8.mean(dim=(2, 3))
        z = torch.cat([gv, gi, torch.abs(gv - gi)], dim=1)
        raw = self.mlp(z)

        angle_delta = torch.tanh(raw[:, 0]) * MAX_ANGLE_DELTA_DEG
        log_sx_delta = torch.tanh(raw[:, 1]) * MAX_LOG_SCALE_X_DELTA
        log_sy_delta = torch.tanh(raw[:, 2]) * MAX_LOG_SCALE_Y_DELTA

        return {
            "angle": self.base_angle_deg + angle_delta,
            "scale_x": self.base_scale_x * torch.exp(log_sx_delta),
            "scale_y": self.base_scale_y * torch.exp(log_sy_delta),
            "angle_delta": angle_delta,
            "log_scale_x_delta": log_sx_delta,
            "log_scale_y_delta": log_sy_delta,
        }

class TargetCrossAttentionV31(nn.Module):
    def __init__(self, dim=base.F8_DIM, heads=4):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, ROI_FEATURE_SIZE * ROI_FEATURE_SIZE, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.05, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 3),
        )

    def forward(self, fv, fi, bbox, has_bbox):
        rv = base.crop_feature_roi(fv, bbox, has_bbox, ROI_FEATURE_SIZE, ROI_EXPAND)
        ri = base.crop_feature_roi(fi, bbox, has_bbox, ROI_FEATURE_SIZE, ROI_EXPAND)

        q = rv.flatten(2).transpose(1, 2) + self.pos
        kv = ri.flatten(2).transpose(1, 2) + self.pos
        a, _ = self.attn(q, kv, kv, need_weights=False)
        x = self.norm1(q + a)
        x = self.norm2(x + self.ffn(x))

        cross_pool = x.mean(dim=1)
        v_pool = q.mean(dim=1)
        i_pool = kv.mean(dim=1)
        z = torch.cat([cross_pool, v_pool - i_pool, torch.abs(v_pool - i_pool)], dim=1)
        raw = self.head(z)

        raw_dx = torch.tanh(raw[:, 0]) * MAX_LOCAL_DX_PX
        raw_dy = torch.tanh(raw[:, 1]) * MAX_LOCAL_DY_PX
        gate_logit = raw[:, 2]
        gate = torch.sigmoid(gate_logit) * has_bbox

        return {
            "raw_dx": raw_dx,
            "raw_dy": raw_dy,
            "candidate_dx": raw_dx,
            "candidate_dy": raw_dy,
            "gate_logit": gate_logit,
            "gate": gate,
        }

class GlobalConfidenceHead(nn.Module):
    """Predict expected transform-grid error in network pixels."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, descriptor):
        return F.softplus(self.net(descriptor.detach())[:, 0])

class H30TRegistrationNetV31(nn.Module):
    def __init__(self, base_angle_deg, base_scale_x, base_scale_y):
        super().__init__()
        self.visible_encoder = base.PyramidStructuralEncoder()
        self.thermal_encoder = base.PyramidStructuralEncoder()
        self.angle_scale_head = AngleAnisoScaleHead(
            base_angle_deg, base_scale_x, base_scale_y
        )
        self.coarse_corr = base.CorrelationSoftArgmax(
            in_dim=base.F8_DIM,
            proj_dim=base.CORR_DIM_COARSE,
            radius=COARSE_RADIUS,
            temperature=COARSE_TEMPERATURE,
            shift_prior_sigma=SHIFT_PRIOR_SIGMA_COARSE,
        )
        self.fine_corr = base.CorrelationSoftArgmax(
            in_dim=base.F4_DIM,
            proj_dim=base.CORR_DIM_FINE,
            radius=FINE_RADIUS,
            temperature=FINE_TEMPERATURE,
            shift_prior_sigma=SHIFT_PRIOR_SIGMA_FINE,
        )
        self.target_head = TargetCrossAttentionV31(dim=base.F8_DIM)
        self.confidence_head = GlobalConfidenceHead()

    def forward(self, visible, infrared, bbox, has_bbox, enable_local=True):
        fv = self.visible_encoder(base.model_input(visible))
        fi0 = self.thermal_encoder(base.model_input(infrared))

        as_out = self.angle_scale_head(fv["f8"], fi0["f8"])
        z = torch.zeros_like(as_out["angle"])

        ir_rs = warp_anisotropic_px(
            infrared, z, z,
            as_out["angle"], as_out["scale_x"], as_out["scale_y"]
        )
        fi_rs = self.thermal_encoder(base.model_input(ir_rs))

        coarse = self.coarse_corr(fv["f8"], fi_rs["f8"])
        coarse_tx = coarse["dx_feature"] * COARSE_STRIDE
        coarse_ty = coarse["dy_feature"] * COARSE_STRIDE

        ir_coarse = warp_anisotropic_px(
            infrared, coarse_tx, coarse_ty,
            as_out["angle"], as_out["scale_x"], as_out["scale_y"]
        )
        fi_coarse = self.thermal_encoder(base.model_input(ir_coarse))

        fine = self.fine_corr(fv["f4"], fi_coarse["f4"])
        fine_tx = fine["dx_feature"] * FINE_STRIDE
        fine_ty = fine["dy_feature"] * FINE_STRIDE

        tx = coarse_tx + fine_tx
        ty = coarse_ty + fine_ty

        global_ir = warp_anisotropic_px(
            infrared, tx, ty,
            as_out["angle"], as_out["scale_x"], as_out["scale_y"]
        )

        vis_edge = base.sobel_mag(base.rgb_to_gray(visible))
        ir_edge = base.sobel_mag(base.rgb_to_gray(infrared))
        global_edge = base.sobel_mag(base.rgb_to_gray(global_ir))
        ncc_before = base.weighted_ncc_torch(vis_edge, ir_edge)
        ncc_global = base.weighted_ncc_torch(vis_edge, global_edge)

        conf_desc = torch.stack(
            [coarse["peak_prob"], coarse["entropy"],
             fine["peak_prob"], fine["entropy"],
             ncc_before, ncc_global],
            dim=1,
        )
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
            "log_scale_y_delta": as_out["log_scale_y_delta"],
        }

        if enable_local:
            fi_global = self.thermal_encoder(base.model_input(global_ir))
            local = self.target_head(fv["f8"], fi_global["f8"], bbox, has_bbox)
            target_candidate_ir = base.gaussian_local_warp(
                global_ir, bbox, has_bbox,
                local["candidate_dx"], local["candidate_dy"], radius_expand=3.0
            )
        else:
            zz = torch.zeros(visible.shape[0], dtype=visible.dtype, device=visible.device)
            local = {
                "raw_dx": zz, "raw_dy": zz,
                "candidate_dx": zz, "candidate_dy": zz,
                "gate_logit": zz, "gate": zz,
            }
            target_candidate_ir = global_ir

        return {
            "global_ir": global_ir,
            "target_candidate_ir": target_candidate_ir,
            "geo": geo,
            "local": local,
            "coarse_corr": coarse,
            "fine_corr": fine,
            "ncc_before": ncc_before,
            "ncc_global": ncc_global,
            "confidence_error_px": confidence_error_px,
        }

def masked_mean(value, weight):
    return (value * weight).sum() / (weight.sum() + 1e-6)

def smooth_norm(pred, target, scale):
    return F.smooth_l1_loss(pred / scale, target / scale, reduction="none")

def gt_geo_dict(human_geo):
    return {
        "tx": human_geo[:, 0],
        "ty": human_geo[:, 1],
        "angle": human_geo[:, 2],
        "scale_x": human_geo[:, 3],
        "scale_y": human_geo[:, 4],
    }

def geometry_grid(geo, batch_size, device, dtype):
    pts = standard_grid_points(batch_size, device, dtype)
    return transform_points_anisotropic(
        pts,
        geo["tx"], geo["ty"], geo["angle"], geo["scale_x"], geo["scale_y"]
    )

def compute_gate_pos_weight(train_pairs):
    """
    根据人工 GT 中的局部残差计算 gate 正样本权重。
    由于现在使用 local_residual，可直接从 CSV 读取。
    """
    pos = 0
    neg = 0
    for p in train_pairs:
        gt = p["human_gt"]
        dx = gt["local_residual_dx_px"]
        dy = gt["local_residual_dy_px"]
        norm = math.hypot(dx, dy)
        if norm > LOCAL_SUPERVISION_MAX_FULL_PX:
            continue
        if norm > TARGET_REQUIRED_FULL_PX:
            pos += 1
        else:
            neg += 1

    if pos <= 0:
        return 1.0
    return float(np.clip(neg / pos, 1.0, 4.0))

def registration_losses(visible, outputs, batch, stage, gate_pos_weight):
    geo = outputs["geo"]
    local = outputs["local"]
    human_geo = batch["human_geo"]
    gt_geo = gt_geo_dict(human_geo)
    B = visible.shape[0]

    # Global structure
    loss_global_structure = (1.0 - outputs["ncc_global"]).mean()

    # Parameter supervision
    l_tx = smooth_norm(geo["tx"], gt_geo["tx"], 40.0)
    l_ty = smooth_norm(geo["ty"], gt_geo["ty"], 36.0)
    l_angle = smooth_norm(geo["angle"], gt_geo["angle"], 1.0)
    l_sx = F.smooth_l1_loss(
        torch.log(geo["scale_x"].clamp(min=0.75)) / 0.06,
        torch.log(gt_geo["scale_x"].clamp(min=0.75)) / 0.06,
        reduction="none",
    )
    l_sy = F.smooth_l1_loss(
        torch.log(geo["scale_y"].clamp(min=0.75)) / 0.06,
        torch.log(gt_geo["scale_y"].clamp(min=0.75)) / 0.06,
        reduction="none",
    )
    loss_human_param = (l_tx + l_ty + l_angle + l_sx + l_sy).mean()

    # Grid supervision
    pred_grid = geometry_grid(geo, B, visible.device, visible.dtype)
    gt_grid = geometry_grid(gt_geo, B, visible.device, visible.dtype)
    grid_delta = pred_grid - gt_grid
    loss_human_grid = F.smooth_l1_loss(
        grid_delta / 10.0,
        torch.zeros_like(grid_delta),
        reduction="mean",
    )
    grid_error_per = torch.sqrt((grid_delta ** 2).sum(dim=2) + 1e-6).mean(dim=1)

    # Coarse translation auxiliary
    loss_coarse = (
        smooth_norm(geo["coarse_tx"], gt_geo["tx"], 48.0)
        + smooth_norm(geo["coarse_ty"], gt_geo["ty"], 48.0)
    ).mean()

    # Prior regularization
    loss_prior_reg = (
        (geo["angle_delta"] / MAX_ANGLE_DELTA_DEG) ** 2
        + (geo["log_scale_x_delta"] / MAX_LOG_SCALE_X_DELTA) ** 2
        + (geo["log_scale_y_delta"] / MAX_LOG_SCALE_Y_DELTA) ** 2
    ).mean()

    corr_entropy = (
        outputs["coarse_corr"]["entropy"].mean()
        + outputs["fine_corr"]["entropy"].mean()
    ) / 2.0

    # Confidence target is the actual Human transform-grid error; detached.
    loss_confidence = F.smooth_l1_loss(
        outputs["confidence_error_px"] / 20.0,
        grid_error_per.detach() / 20.0,
        reduction="mean",
    )

    # ========== 修改部分：直接使用 local_residual_gt 作为局部监督 ==========
    # 原动态计算：pred_global_point = transform_points_anisotropic(...)
    #             dynamic_target = batch["vis_uav_center"] - pred_global_point.detach()
    # 现在直接读取 CSV 中的 local_residual
    dynamic_target = batch["local_residual_gt"]   # shape [B, 2]

    scale_to_input = batch["raw_to_input_scale"].clamp(min=1e-6)
    dynamic_full = dynamic_target / scale_to_input
    dynamic_full_norm = torch.sqrt((dynamic_full ** 2).sum(dim=1) + 1e-6)

    local_valid = (
        batch["has_bbox"]
        * (dynamic_full_norm <= LOCAL_SUPERVISION_MAX_FULL_PX).to(visible.dtype)
    )
    gate_target = (dynamic_full_norm > TARGET_REQUIRED_FULL_PX).to(visible.dtype)

    gate_bce = F.binary_cross_entropy_with_logits(
        local["gate_logit"],
        gate_target,
        reduction="none",
        pos_weight=torch.tensor(
            float(gate_pos_weight), dtype=visible.dtype, device=visible.device
        ),
    )
    loss_human_gate = masked_mean(gate_bce, local_valid)

    local_dx_loss = smooth_norm(
        local["raw_dx"], dynamic_target[:, 0], MAX_LOCAL_DX_PX
    )
    local_dy_loss = smooth_norm(
        local["raw_dy"], dynamic_target[:, 1], MAX_LOCAL_DY_PX
    )
    local_weight = local_valid * (0.35 + 0.65 * gate_target)
    loss_human_local = masked_mean(local_dx_loss + local_dy_loss, local_weight)

    # Weak target structure loss on full residual candidate.
    vis_edge = base.sobel_mag(base.rgb_to_gray(visible))
    target_edge = base.sobel_mag(base.rgb_to_gray(outputs["target_candidate_ir"]))
    target_mask = base.gaussian_target_mask(
        batch["bbox"], batch["has_bbox"], visible.shape[2], visible.shape[3], expand=4.0
    )
    target_ncc = base.weighted_ncc_torch(vis_edge, target_edge, target_mask)
    loss_target_structure = masked_mean(1.0 - target_ncc, batch["has_bbox"])

    global_total = (
        W_GLOBAL_STRUCTURE * loss_global_structure
        + W_HUMAN_PARAM * loss_human_param
        + W_HUMAN_GRID * loss_human_grid
        + W_COARSE_TRANSLATION * loss_coarse
        + W_PRIOR_REG * loss_prior_reg
        + W_CORR_ENTROPY * corr_entropy
        + W_CONFIDENCE * loss_confidence
    )

    target_total = (
        W_TARGET_STRUCTURE * loss_target_structure
        + W_HUMAN_LOCAL * loss_human_local
        + W_HUMAN_GATE * loss_human_gate
    )

    if stage == "A":
        total = global_total
    elif stage == "B":
        total = target_total
    elif stage == "C":
        total = global_total + target_total
    else:
        raise ValueError("Bad stage")

    return {
        "total": total,
        "global_structure": loss_global_structure,
        "human_param": loss_human_param,
        "human_grid": loss_human_grid,
        "coarse_translation": loss_coarse,
        "prior_reg": loss_prior_reg,
        "corr_entropy": corr_entropy,
        "confidence": loss_confidence,
        "target_structure": loss_target_structure,
        "human_local": loss_human_local,
        "human_gate": loss_human_gate,
        "grid_error_px": grid_error_per.mean(),
        "dynamic_target_full_px": dynamic_full_norm.mean(),
        "mean_gate": local["gate"].mean(),
        "confidence_error_px": outputs["confidence_error_px"].mean(),
        "coarse_peak": outputs["coarse_corr"]["peak_prob"].mean(),
        "coarse_entropy": outputs["coarse_corr"]["entropy"].mean(),
        "fine_peak": outputs["fine_corr"]["peak_prob"].mean(),
        "fine_entropy": outputs["fine_corr"]["entropy"].mean(),
    }

def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag

def configure_stage(model, stage):
    if stage == "A":
        set_requires_grad(model.visible_encoder, True)
        set_requires_grad(model.thermal_encoder, True)
        set_requires_grad(model.angle_scale_head, True)
        set_requires_grad(model.coarse_corr, True)
        set_requires_grad(model.fine_corr, True)
        set_requires_grad(model.confidence_head, True)
        set_requires_grad(model.target_head, False)
    elif stage == "B":
        set_requires_grad(model.visible_encoder, False)
        set_requires_grad(model.thermal_encoder, False)
        set_requires_grad(model.angle_scale_head, False)
        set_requires_grad(model.coarse_corr, False)
        set_requires_grad(model.fine_corr, False)
        set_requires_grad(model.confidence_head, False)
        set_requires_grad(model.target_head, True)
    elif stage == "C":
        set_requires_grad(model, True)
    else:
        raise ValueError("Bad stage")

def set_stage_train_mode(model, stage):
    """Fix V3 Stage-B frozen-Dropout issue explicitly."""
    model.train()
    if stage == "B":
        model.visible_encoder.eval()
        model.thermal_encoder.eval()
        model.angle_scale_head.eval()
        model.coarse_corr.eval()
        model.fine_corr.eval()
        model.confidence_head.eval()
        model.target_head.train()

def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]

DEVICE_KEYS = [
    "visible", "infrared", "bbox", "has_bbox", "human_geo",
    "ir_uav_raw", "vis_uav_center", "raw_to_input_scale", "raw_size",
    "local_residual_gt",   # 新增
]

def move_batch_to_device(batch, device):
    for key in DEVICE_KEYS:
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch

def make_loader(pairs, augment, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(
        H30THumanDataset(pairs, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

@torch.no_grad()
def validate(model, loader, device, stage, gate_pos_weight):
    model.eval()
    sums = {}
    seen = 0
    enable_local = stage != "A"

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["visible"], batch["infrared"],
            batch["bbox"], batch["has_bbox"],
            enable_local=enable_local,
        )
        losses = registration_losses(
            batch["visible"], outputs, batch, stage, gate_pos_weight
        )
        bs = batch["visible"].shape[0]
        seen += bs
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach()) * bs

    return {k: v / max(1, seen) for k, v in sums.items()}

def checkpoint_payload(model, prior_stats, split_counts, epoch, stage, val_metrics):
    return {
        "model": model.state_dict(),
        "prior_stats": prior_stats,
        "split_counts": split_counts,
        "epoch": epoch,
        "stage": stage,
        "val_metrics": val_metrics,
        "config": {
            "input_w": INPUT_W,
            "input_h": INPUT_H,
            "target_required_full_px": TARGET_REQUIRED_FULL_PX,
            "local_supervision_max_full_px": LOCAL_SUPERVISION_MAX_FULL_PX,
        },
    }

def metric_summary(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    n = int(a.size)
    le_2 = int(np.sum(a <= 2.0))
    le_5 = int(np.sum(a <= 5.0))
    le_10 = int(np.sum(a <= 10.0))
    le_20 = int(np.sum(a <= 20.0))
    return {
        "n": n,
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "rmse": float(np.sqrt(np.mean(a * a))),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
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

def numpy_transform_point(x, y, tx, ty, angle_deg, scale_x, scale_y, width, height):
    th = math.radians(float(angle_deg))
    c = math.cos(th)
    s = math.sin(th)
    cx = width / 2.0
    cy = height / 2.0
    x0 = (x - cx) * scale_x
    y0 = (y - cy) * scale_y
    return (
        c * x0 + s * y0 + cx + tx,
        -s * x0 + c * y0 + cy + ty,
    )

def ncc_per_sample(visible, moving):
    ve = base.sobel_mag(base.rgb_to_gray(visible))
    me = base.sobel_mag(base.rgb_to_gray(moving))
    return base.weighted_ncc_torch(ve, me)

@torch.no_grad()
def collect_eval_records(model, loader, device, prior):
    records = []
    for batch in loader:
        keys = list(batch["key"])
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["visible"], batch["infrared"],
            batch["bbox"], batch["has_bbox"],
            enable_local=True,
        )
        B = batch["visible"].shape[0]
        z = torch.zeros(B, device=device, dtype=batch["visible"].dtype)
        prior_angle = torch.full_like(z, float(prior["base_angle_deg"]))
        prior_sx = torch.full_like(z, float(prior["base_scale_x"]))
        prior_sy = torch.full_like(z, float(prior["base_scale_y"]))
        prior_ir = warp_anisotropic_px(
            batch["infrared"], z, z, prior_angle, prior_sx, prior_sy
        )
        identity_score = ncc_per_sample(batch["visible"], batch["infrared"])
        prior_score = ncc_per_sample(batch["visible"], prior_ir)

        for i in range(B):
            scale_xy = batch["raw_to_input_scale"][i].detach().cpu().numpy()
            sx_img = float(scale_xy[0])
            sy_img = float(scale_xy[1])
            irp = batch["ir_uav_raw"][i].detach().cpu().numpy()
            vcp = batch["vis_uav_center"][i].detach().cpu().numpy()
            geo = outputs["geo"]

            tx = float(geo["tx"][i].detach().cpu())
            ty = float(geo["ty"][i].detach().cpu())
            angle = float(geo["angle"][i].detach().cpu())
            psx = float(geo["scale_x"][i].detach().cpu())
            psy = float(geo["scale_y"][i].detach().cpu())

            gx, gy = numpy_transform_point(
                float(irp[0]), float(irp[1]),
                tx, ty, angle, psx, psy, INPUT_W, INPUT_H
            )
            full_global_tre = math.hypot(
                (gx - float(vcp[0])) / sx_img,
                (gy - float(vcp[1])) / sy_img,
            )

            pgx, pgy = numpy_transform_point(
                float(irp[0]), float(irp[1]),
                0.0, 0.0,
                float(prior["base_angle_deg"]),
                float(prior["base_scale_x"]),
                float(prior["base_scale_y"]),
                INPUT_W, INPUT_H,
            )
            prior_tre = math.hypot(
                (pgx - float(vcp[0])) / sx_img,
                (pgy - float(vcp[1])) / sy_img,
            )

            identity_tre = math.hypot(
                (float(irp[0]) - float(vcp[0])) / sx_img,
                (float(irp[1]) - float(vcp[1])) / sy_img,
            )

            if float(prior_score[i].detach().cpu()) >= float(identity_score[i].detach().cpu()):
                fallback_kind = "prior"
                fallback_tre = prior_tre
            else:
                fallback_kind = "identity"
                fallback_tre = identity_tre

            # 注意：这里读取的是模型预测的局部偏移，用于评估
            local_dx = float(outputs["local"]["raw_dx"][i].detach().cpu())
            local_dy = float(outputs["local"]["raw_dy"][i].detach().cpu())
            # 计算应用局部偏移后的 TRE（使用预测值）
            target_tre = math.hypot(
                (gx + local_dx - float(vcp[0])) / sx_img,
                (gy + local_dy - float(vcp[1])) / sy_img,
            )

            # 也可以计算基于 CSV 中 local_residual 的 TRE（真值）
            local_gt = batch["local_residual_gt"][i].detach().cpu().numpy()
            gt_tre = math.hypot(
                (gx + local_gt[0] - float(vcp[0])) / sx_img,
                (gy + local_gt[1] - float(vcp[1])) / sy_img,
            )

            records.append({
                "key": keys[i],
                "confidence_error_px": float(outputs["confidence_error_px"][i].detach().cpu()),
                "gate": float(outputs["local"]["gate"][i].detach().cpu()),
                "full_global_tre": full_global_tre,
                "target_candidate_tre": target_tre,
                "gt_local_tre": gt_tre,
                "fallback_kind": fallback_kind,
                "fallback_tre": fallback_tre,
                "identity_tre": identity_tre,
                "prior_tre": prior_tre,
                "tx": tx, "ty": ty, "angle": angle,
                "scale_x": psx, "scale_y": psy,
                "local_dx": local_dx, "local_dy": local_dy,
                "coarse_peak": float(outputs["coarse_corr"]["peak_prob"][i].detach().cpu()),
                "coarse_entropy": float(outputs["coarse_corr"]["entropy"][i].detach().cpu()),
                "fine_peak": float(outputs["fine_corr"]["peak_prob"][i].detach().cpu()),
                "fine_entropy": float(outputs["fine_corr"]["entropy"][i].detach().cpu()),
                "ncc_before": float(identity_score[i].detach().cpu()),
                "ncc_global": float(outputs["ncc_global"][i].detach().cpu()),
                "ncc_prior": float(prior_score[i].detach().cpu()),
            })
    return records

def objective_value(values, metric):
    a = np.asarray(values, dtype=np.float64)
    if metric == "rmse":
        return float(np.sqrt(np.mean(a * a)))
    return float(np.mean(a))
