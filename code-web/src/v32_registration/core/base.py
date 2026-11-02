"""Dependency-closed V32 research core; numerical definitions retained from the frozen experiment."""

from __future__ import annotations

import csv

import json

import math

import random

from pathlib import Path

import cv2

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

LABEL_FORMAT = "yolo"

UAV_CLASS_ID = 0

TARGET_SELECT = "largest"

SHIFT_PRIOR_WEIGHT = 0.08

F4_DIM = 64

F8_DIM = 128

CORR_DIM_COARSE = 96

CORR_DIM_FINE = 64

def load_txt_bbox(
    txt_path,
    image_shape
):
    if txt_path is None:
        return None

    txt_path = Path(
        txt_path
    )

    if not txt_path.exists():
        return None

    h, w = image_shape[:2]

    candidates = []

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                vals = [
                    float(v)
                    for v in
                    line.replace(
                        ",", " "
                    ).split()
                ]

            except ValueError:

                continue

            if LABEL_FORMAT == "yolo":

                if len(vals) < 5:
                    continue

                class_id = int(
                    vals[0]
                )

                xc, yc, bw, bh = \
                    vals[1:5]

                x1 = (
                    xc -
                    bw / 2
                ) * w

                y1 = (
                    yc -
                    bh / 2
                ) * h

                x2 = (
                    xc +
                    bw / 2
                ) * w

                y2 = (
                    yc +
                    bh / 2
                ) * h

            elif LABEL_FORMAT == "xyxy":

                if len(vals) == 4:

                    class_id = 0
                    x1, y1, x2, y2 = vals

                elif len(vals) >= 5:

                    class_id = int(
                        vals[0]
                    )

                    x1, y1, x2, y2 = \
                        vals[1:5]

                else:
                    continue

            elif LABEL_FORMAT == "xywh":

                if len(vals) == 4:

                    class_id = 0
                    x, y, bw, bh = vals

                elif len(vals) >= 5:

                    class_id = int(
                        vals[0]
                    )

                    x, y, bw, bh = \
                        vals[1:5]

                else:
                    continue

                x1 = x
                y1 = y
                x2 = x + bw
                y2 = y + bh

            else:

                raise ValueError(
                    "LABEL_FORMAT invalid"
                )

            if (
                UAV_CLASS_ID is not None
                and
                class_id != UAV_CLASS_ID
            ):
                continue

            x1 = float(
                np.clip(
                    x1, 0, w - 1
                )
            )

            y1 = float(
                np.clip(
                    y1, 0, h - 1
                )
            )

            x2 = float(
                np.clip(
                    x2, 1, w
                )
            )

            y2 = float(
                np.clip(
                    y2, 1, h
                )
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):
                continue

            area = (
                (x2 - x1) *
                (y2 - y1)
            )

            candidates.append(
                (
                    area,
                    [
                        x1, y1,
                        x2, y2
                    ]
                )
            )

    if not candidates:
        return None

    if TARGET_SELECT == "largest":

        candidates.sort(
            key=lambda x:
                x[0],
            reverse=True
        )

    return candidates[0][1]

def rgb_to_gray(
    x
):
    return (
        0.299 *
        x[:, 0:1]
        +
        0.587 *
        x[:, 1:2]
        +
        0.114 *
        x[:, 2:3]
    )

def sobel_mag(
    gray
):
    dtype = gray.dtype
    device = gray.device

    kx = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=dtype,
        device=device
    ).view(
        1, 1, 3, 3
    )

    ky = torch.tensor(
        [
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ],
        dtype=dtype,
        device=device
    ).view(
        1, 1, 3, 3
    )

    gx = F.conv2d(
        gray,
        kx,
        padding=1
    )

    gy = F.conv2d(
        gray,
        ky,
        padding=1
    )

    mag = torch.sqrt(
        gx * gx +
        gy * gy +
        1e-6
    )

    scale = mag.flatten(
        2
    ).mean(
        dim=2,
        keepdim=True
    ).view(
        -1, 1, 1, 1
    )

    mag = (
        mag /
        (
            scale +
            1e-6
        )
    )

    return torch.clamp(
        mag,
        0,
        6
    ) / 6.0

def model_input(
    rgb
):
    gray = rgb_to_gray(
        rgb
    )

    edge = sobel_mag(
        gray
    )

    return torch.cat(
        [
            gray,
            edge
        ],
        dim=1
    )

def gaussian_target_mask(
    bbox_norm,
    has_bbox,
    H,
    W,
    expand
):
    B = bbox_norm.shape[0]

    dtype = bbox_norm.dtype
    device = bbox_norm.device

    x1, y1, x2, y2 = \
        bbox_norm.unbind(
            dim=1
        )

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    bw = (
        x2 - x1
    ).clamp(
        min=1.0 / W
    ) * expand

    bh = (
        y2 - y1
    ).clamp(
        min=1.0 / H
    ) * expand

    yy, xx = torch.meshgrid(
        torch.linspace(
            0,
            1,
            H,
            dtype=dtype,
            device=device
        ),
        torch.linspace(
            0,
            1,
            W,
            dtype=dtype,
            device=device
        ),
        indexing="ij"
    )

    xx = xx.unsqueeze(
        0
    )

    yy = yy.unsqueeze(
        0
    )

    sx = (
        bw /
        3.0
    ).view(
        B, 1, 1
    ).clamp(
        min=2.0 / W
    )

    sy = (
        bh /
        3.0
    ).view(
        B, 1, 1
    ).clamp(
        min=2.0 / H
    )

    mask = torch.exp(
        -0.5 *
        (
            (
                (
                    xx -
                    cx.view(
                        B, 1, 1
                    )
                ) /
                sx
            ) ** 2
            +
            (
                (
                    yy -
                    cy.view(
                        B, 1, 1
                    )
                ) /
                sy
            ) ** 2
        )
    )

    return (
        mask.unsqueeze(
            1
        )
        *
        has_bbox.view(
            B, 1, 1, 1
        )
    )

def gaussian_local_warp(
    image,
    bbox_norm,
    has_bbox,
    dx_px,
    dy_px,
    radius_expand=3.0
):
    B, C, H, W = \
        image.shape

    dtype = image.dtype
    device = image.device

    weight = gaussian_target_mask(
        bbox_norm,
        has_bbox,
        H,
        W,
        radius_expand
    )[:, 0]

    yy, xx = torch.meshgrid(
        torch.arange(
            H,
            dtype=dtype,
            device=device
        ),
        torch.arange(
            W,
            dtype=dtype,
            device=device
        ),
        indexing="ij"
    )

    xx = xx.unsqueeze(
        0
    ).expand(
        B, -1, -1
    )

    yy = yy.unsqueeze(
        0
    ).expand(
        B, -1, -1
    )

    src_x = (
        xx -
        weight *
        dx_px.view(
            B, 1, 1
        )
    )

    src_y = (
        yy -
        weight *
        dy_px.view(
            B, 1, 1
        )
    )

    xn = (
        2.0 *
        (src_x + 0.5) /
        W -
        1.0
    )

    yn = (
        2.0 *
        (src_y + 0.5) /
        H -
        1.0
    )

    grid = torch.stack(
        [
            xn,
            yn
        ],
        dim=-1
    )

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )

class ConvGNAct(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1
    ):
        super().__init__()

        groups = min(
            8,
            out_ch
        )

        while (
            out_ch %
            groups != 0
        ):
            groups -= 1

        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(
                groups,
                out_ch
            ),
            nn.GELU()
        )

    def forward(
        self,
        x
    ):
        return self.net(
            x
        )

class PyramidStructuralEncoder(nn.Module):
    """
    Input 480x384:
        stride-2 : 240x192
        stride-4 : 120x96
        stride-8 : 60x48
    """

    def __init__(
        self
    ):
        super().__init__()

        self.s2 = nn.Sequential(
            ConvGNAct(
                2, 32, 2
            ),
            ConvGNAct(
                32, 32, 1
            )
        )

        self.s4 = nn.Sequential(
            ConvGNAct(
                32, F4_DIM, 2
            ),
            ConvGNAct(
                F4_DIM, F4_DIM, 1
            )
        )

        self.s8 = nn.Sequential(
            ConvGNAct(
                F4_DIM, 96, 2
            ),
            ConvGNAct(
                96, F8_DIM, 1
            ),
            ConvGNAct(
                F8_DIM, F8_DIM, 1
            )
        )

    def forward(
        self,
        x
    ):
        f2 = self.s2(
            x
        )

        f4 = self.s4(
            f2
        )

        f8 = self.s8(
            f4
        )

        return {
            "f4":
                f4,

            "f8":
                f8
        }

def integer_shift_feature(
    feat,
    dx,
    dy
):
    """
    Translation convention:
        candidate (dx,dy) means moving feature content is translated
        by +dx,+dy toward the reference.

    Output uses zero padding and no wraparound.
    """

    B, C, H, W = \
        feat.shape

    out = torch.zeros_like(
        feat
    )

    x_dst1 = max(
        0,
        dx
    )

    x_dst2 = min(
        W,
        W + dx
    )

    y_dst1 = max(
        0,
        dy
    )

    y_dst2 = min(
        H,
        H + dy
    )

    x_src1 = max(
        0,
        -dx
    )

    x_src2 = min(
        W,
        W - dx
    )

    y_src1 = max(
        0,
        -dy
    )

    y_src2 = min(
        H,
        H - dy
    )

    if (
        x_dst2 <= x_dst1
        or
        y_dst2 <= y_dst1
    ):
        return out

    out[
        :,
        :,
        y_dst1:y_dst2,
        x_dst1:x_dst2
    ] = feat[
        :,
        :,
        y_src1:y_src2,
        x_src1:x_src2
    ]

    return out

def integer_shift_mask(
    B,
    H,
    W,
    dx,
    dy,
    device,
    dtype
):
    base = torch.ones(
        (
            B, 1, H, W
        ),
        device=device,
        dtype=dtype
    )

    return integer_shift_feature(
        base,
        dx,
        dy
    )

class CorrelationSoftArgmax(nn.Module):
    """
    Explicit local translation search.

    For every candidate integer shift:
      - shift thermal feature
      - calculate normalized feature correlation
      - spatially aggregate correlation with a learned visible confidence map

    Then softmax over candidate shifts and soft-argmax:
        dx = sum p_i * shift_x_i
        dy = sum p_i * shift_y_i
    """

    def __init__(
        self,
        in_dim,
        proj_dim,
        radius,
        temperature,
        shift_prior_sigma
    ):
        super().__init__()

        self.radius = int(
            radius
        )

        self.temperature = float(
            temperature
        )

        self.shift_prior_sigma = float(
            shift_prior_sigma
        )

        self.ref_proj = nn.Sequential(
            nn.Conv2d(
                in_dim,
                proj_dim,
                1,
                bias=False
            ),
            nn.GroupNorm(
                8,
                proj_dim
            )
        )

        self.mov_proj = nn.Sequential(
            nn.Conv2d(
                in_dim,
                proj_dim,
                1,
                bias=False
            ),
            nn.GroupNorm(
                8,
                proj_dim
            )
        )

        self.confidence = nn.Sequential(
            nn.Conv2d(
                in_dim,
                32,
                3,
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                32,
                1,
                1
            ),
            nn.Sigmoid()
        )

        shifts = []

        for dy in range(
            -self.radius,
            self.radius + 1
        ):
            for dx in range(
                -self.radius,
                self.radius + 1
            ):
                shifts.append(
                    (
                        dx,
                        dy
                    )
                )

        shift_tensor = torch.tensor(
            shifts,
            dtype=torch.float32
        )

        self.register_buffer(
            "shift_xy",
            shift_tensor
        )

    def forward(
        self,
        reference_feat,
        moving_feat
    ):
        B, C, H, W = \
            reference_feat.shape

        ref = self.ref_proj(
            reference_feat
        )

        mov = self.mov_proj(
            moving_feat
        )

        ref = F.normalize(
            ref,
            dim=1
        )

        mov = F.normalize(
            mov,
            dim=1
        )

        confidence = (
            0.15 +
            0.85 *
            self.confidence(
                reference_feat
            )
        )

        scores = []

        for shift_index in range(
            self.shift_xy.shape[0]
        ):

            dx = int(
                self.shift_xy[
                    shift_index,
                    0
                ].item()
            )

            dy = int(
                self.shift_xy[
                    shift_index,
                    1
                ].item()
            )

            shifted = integer_shift_feature(
                mov,
                dx,
                dy
            )

            valid = integer_shift_mask(
                B,
                H,
                W,
                dx,
                dy,
                ref.device,
                ref.dtype
            )

            corr_map = (
                ref *
                shifted
            ).sum(
                dim=1,
                keepdim=True
            )

            weight = (
                confidence *
                valid
            )

            score = (
                corr_map *
                weight
            ).sum(
                dim=(1, 2, 3)
            ) / (
                weight.sum(
                    dim=(1, 2, 3)
                )
                +
                1e-6
            )

            # weak zero-centered H30T residual prior
            if (
                SHIFT_PRIOR_WEIGHT
                > 0
            ):

                shift_radius2 = (
                    dx * dx +
                    dy * dy
                )

                prior_penalty = (
                    shift_radius2 /
                    (
                        2.0 *
                        self.shift_prior_sigma ** 2
                    )
                )

                score = (
                    score -
                    SHIFT_PRIOR_WEIGHT *
                    prior_penalty
                )

            scores.append(
                score
            )

        scores = torch.stack(
            scores,
            dim=1
        )  # B,S

        probs = torch.softmax(
            scores /
            self.temperature,
            dim=1
        )

        shift_xy = self.shift_xy.to(
            dtype=probs.dtype
        )

        expected = torch.matmul(
            probs,
            shift_xy
        )  # B,2

        entropy = -(
            probs *
            torch.log(
                probs +
                1e-8
            )
        ).sum(
            dim=1
        )

        entropy = (
            entropy /
            math.log(
                probs.shape[1]
            )
        )

        peak_prob = probs.max(
            dim=1
        ).values

        return {
            "dx_feature":
                expected[:, 0],

            "dy_feature":
                expected[:, 1],

            "scores":
                scores,

            "probs":
                probs,

            "entropy":
                entropy,

            "peak_prob":
                peak_prob
        }

def crop_feature_roi(
    feat,
    bbox_norm,
    has_bbox,
    out_size,
    expand
):
    B, C, H, W = \
        feat.shape

    x1, y1, x2, y2 = \
        bbox_norm.unbind(
            dim=1
        )

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    bw = (
        x2 - x1
    ).clamp(
        min=0.01
    ) * expand

    bh = (
        y2 - y1
    ).clamp(
        min=0.01
    ) * expand

    cx = torch.where(
        has_bbox > 0,
        cx,
        torch.full_like(
            cx,
            0.5
        )
    )

    cy = torch.where(
        has_bbox > 0,
        cy,
        torch.full_like(
            cy,
            0.5
        )
    )

    bw = torch.where(
        has_bbox > 0,
        bw,
        torch.full_like(
            bw,
            0.5
        )
    )

    bh = torch.where(
        has_bbox > 0,
        bh,
        torch.full_like(
            bh,
            0.5
        )
    )

    x1e = (
        cx -
        bw / 2
    ).clamp(
        0, 1
    )

    x2e = (
        cx +
        bw / 2
    ).clamp(
        0, 1
    )

    y1e = (
        cy -
        bh / 2
    ).clamp(
        0, 1
    )

    y2e = (
        cy +
        bh / 2
    ).clamp(
        0, 1
    )

    lin = torch.linspace(
        0,
        1,
        out_size,
        dtype=feat.dtype,
        device=feat.device
    )

    gy, gx = torch.meshgrid(
        lin,
        lin,
        indexing="ij"
    )

    gx = gx.unsqueeze(
        0
    )

    gy = gy.unsqueeze(
        0
    )

    x = (
        x1e.view(
            B, 1, 1
        ) *
        (
            1 -
            gx
        )
        +
        x2e.view(
            B, 1, 1
        ) *
        gx
    )

    y = (
        y1e.view(
            B, 1, 1
        ) *
        (
            1 -
            gy
        )
        +
        y2e.view(
            B, 1, 1
        ) *
        gy
    )

    grid = torch.stack(
        [
            x * 2 - 1,
            y * 2 - 1
        ],
        dim=-1
    )

    return F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False
    )

def weighted_ncc_torch(
    a,
    b,
    weight=None
):
    if weight is None:

        weight = torch.ones_like(
            a
        )

    dims = (
        1, 2, 3
    )

    sw = weight.sum(
        dim=dims,
        keepdim=True
    ).clamp(
        min=1e-6
    )

    ma = (
        a *
        weight
    ).sum(
        dim=dims,
        keepdim=True
    ) / sw

    mb = (
        b *
        weight
    ).sum(
        dim=dims,
        keepdim=True
    ) / sw

    da = a - ma
    db = b - mb

    num = (
        weight *
        da *
        db
    ).sum(
        dim=dims
    )

    den = torch.sqrt(
        (
            weight *
            da *
            da
        ).sum(
            dim=dims
        )
        *
        (
            weight *
            db *
            db
        ).sum(
            dim=dims
        )
        +
        1e-6
    )

    return (
        num /
        den
    )
