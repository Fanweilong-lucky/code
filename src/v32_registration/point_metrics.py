from __future__ import annotations

import math

import numpy as np

import torch

W=960
H=768

def bilinear(mask: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample a HxW mask at pixel-coordinate (x,y)."""
    h, w = mask.shape

    # The relevant local points should stay inside the native image.
    # Clamp only for numerical safety at the boundary.
    x = min(max(float(x), 0.0), w - 1.0)
    y = min(max(float(y), 0.0), h - 1.0)

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    wx = x - x0
    wy = y - y0

    v00 = float(mask[y0, x0])
    v10 = float(mask[y0, x1])
    v01 = float(mask[y1, x0])
    v11 = float(mask[y1, x1])

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )

def replay_global_point(v31, x, y, meta):
    """Use the frozen implementation's point-transform helper."""
    pts = torch.tensor([[[float(x), float(y)]]], dtype=torch.float32)

    tx = torch.tensor([float(meta["global_tx_full_px"])], dtype=torch.float32)
    ty = torch.tensor([float(meta["global_ty_full_px"])], dtype=torch.float32)
    ang = torch.tensor([float(meta["angle_deg"])], dtype=torch.float32)
    sx = torch.tensor([float(meta["scale_x"])], dtype=torch.float32)
    sy = torch.tensor([float(meta["scale_y"])], dtype=torch.float32)

    with torch.no_grad():
        out = v31.transform_points_anisotropic(
            pts,
            tx,
            ty,
            ang,
            sx,
            sy,
            width=W,
            height=H,
        )

    q = out[0, 0].detach().cpu().numpy().astype(np.float64)
    return float(q[0]), float(q[1])

def make_local_mask(base, meta):
    bbox = torch.tensor(
        [meta["bbox_normalized"]],
        dtype=torch.float32,
    )
    has_bbox = torch.tensor(
        [1.0 if bool(meta["has_bbox"]) else 0.0],
        dtype=torch.float32,
    )

    with torch.no_grad():
        mask = base.gaussian_target_mask(
            bbox,
            has_bbox,
            H,
            W,
            3.0,
        )

    # gaussian_target_mask returns [B,1,H,W] in the frozen implementation.
    arr = mask[0, 0].detach().cpu().numpy().astype(np.float64)
    if arr.shape != (H, W):
        raise RuntimeError(f"Unexpected local mask shape: {arr.shape}")
    return arr

def replay_local_forward_point(base, x_src, y_src, meta):
    """
    Invert the exact inverse-sampling field used by gaussian_local_warp.

    gaussian_local_warp defines:
        source = output - weight(output) * d

    so forward source->output satisfies:
        output = source + weight(output) * d
    """
    dx = float(meta["local_dx_full_px"])
    dy = float(meta["local_dy_full_px"])

    mask = make_local_mask(base, meta)

    # Good initialization because at the UAV target region weight is usually
    # near its maximum; fixed-point iteration then solves the exact field.
    qx = float(x_src + dx)
    qy = float(y_src + dy)

    converged = False
    for it in range(100):
        w = bilinear(mask, qx, qy)
        nx = float(x_src + w * dx)
        ny = float(y_src + w * dy)

        if math.hypot(nx - qx, ny - qy) < 1e-7:
            qx, qy = nx, ny
            converged = True
            break

        qx, qy = nx, ny

    # Residual of source = output - weight(output)*d.
    w_final = bilinear(mask, qx, qy)
    rx = qx - w_final * dx - x_src
    ry = qy - w_final * dy - y_src
    residual = math.hypot(rx, ry)

    return qx, qy, converged, it + 1, residual, w_final
