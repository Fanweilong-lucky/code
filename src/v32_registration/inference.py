from __future__ import annotations

import inspect, time

from pathlib import Path

import cv2

import numpy as np

import torch

from .core import base, v31, v32

def load_model(checkpoint, device):
    """Load the trained V32 local head directly; do not initialize it from V31 again."""
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    prior = ckpt['prior_stats']
    model = v32.H30TRegistrationNetV32(prior['base_angle_deg'], prior['base_scale_x'], prior['base_scale_y']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.edge_enabled = False
    model.eval()
    return model, prior

EXPECTED_RAW_W=960
EXPECTED_RAW_H=768
EXPECTED_INPUT_W=480
EXPECTED_INPUT_H=384

IMAGE_EXTS=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def find_image(folder: Path, key: str) -> Path:
    for ext in IMAGE_EXTS:
        p = folder / f"{key}{ext}"

        if p.is_file():
            return p

    raise FileNotFoundError(
        f"Image not found: folder={folder}, key={key}"
    )

def imread_unicode(path: Path, flags: int):
    buf = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        buf,
        flags,
    )

    if image is None:
        raise RuntimeError(
            f"Failed to read image: {path}"
        )

    return image

def imwrite_png(path: Path, image: np.ndarray):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to encode PNG: {path}"
        )

    encoded.tofile(
        str(path)
    )

def image_to_rgb_tensor(
    image: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:

    if image.ndim == 2:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB,
        )
    else:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    rgb = cv2.resize(
        rgb,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    x = torch.from_numpy(
        np.ascontiguousarray(rgb)
    ).permute(
        2, 0, 1
    ).float() / 255.0

    return x.unsqueeze(0).to(device)

def gray_to_tensor(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image

    x = torch.from_numpy(
        np.ascontiguousarray(gray)
    ).float() / 255.0

    return (
        x
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

def tensor_gray_to_u8(
    x: torch.Tensor,
) -> np.ndarray:

    a = (
        x.detach()
        .float()
        .cpu()
        .squeeze()
        .clamp(0.0, 1.0)
        .numpy()
    )

    return np.rint(
        a * 255.0
    ).astype(np.uint8)

def call_frozen_bbox_loader(
    label_path: Path,
    width: int,
    height: int,
):
    """
    Use the frozen Project11 source function load_txt_bbox()
    instead of re-implementing ROI selection.

    Several positional signatures are supported so this wrapper
    does not alter the frozen function itself.
    """

    fn = base.load_txt_bbox

    attempts = [
        (str(label_path), (height, width)),
        (label_path, (height, width)),
        (str(label_path), (height, width, 3)),
        (label_path, (height, width, 3)),
        (str(label_path),),
        (label_path,),
        (str(label_path), width, height),
        (label_path, width, height),
        (str(label_path), height, width),
        (label_path, height, width),
    ]

    errors = []

    result = None

    for args in attempts:
        try:
            result = fn(*args)
            break

        except TypeError as exc:
            errors.append(
                f"{args}: {exc}"
            )

    if result is None:
        raise RuntimeError(
            "Unable to call frozen load_txt_bbox(). "
            f"signature={inspect.signature(fn)}; "
            f"errors={errors}"
        )

    has_bbox = True
    bbox = result

    if isinstance(
        result,
        (tuple, list),
    ) and len(result) == 2:

        first, second = result

        # Common frozen API:
        # (bbox, has_bbox)
        if np.isscalar(second):
            bbox = first
            has_bbox = bool(second)

    if bbox is None:
        return (
            np.zeros(4, dtype=np.float32),
            False,
        )

    if isinstance(bbox, torch.Tensor):
        bbox = (
            bbox.detach()
            .cpu()
            .numpy()
        )

    bbox = np.asarray(
        bbox,
        dtype=np.float32,
    ).reshape(-1)

    if bbox.size != 4:
        raise RuntimeError(
            f"Unexpected bbox shape for {label_path}: "
            f"{bbox.shape}, value={bbox}"
        )

    if not np.isfinite(bbox).all():
        raise RuntimeError(
            f"Non-finite bbox: {label_path}"
        )

    # Frozen model expects normalized bbox coordinates.
    # If frozen helper returned pixel coordinates,
    # preserve its coordinate convention while normalizing axes.
    if np.max(np.abs(bbox)) > 1.5:
        bbox = bbox.copy()

        bbox[0] /= float(width)
        bbox[2] /= float(width)

        bbox[1] /= float(height)
        bbox[3] /= float(height)

    if np.max(np.abs(bbox)) > 2.0:
        raise RuntimeError(
            f"Unexpected normalized bbox values: "
            f"{label_path}: {bbox}"
        )

    return bbox.astype(
        np.float32
    ), bool(has_bbox)

@torch.inference_mode()
def register_one(
    key: str,
    model,
    prior: dict,
    calibration: dict,
    device: torch.device,
    visible_root: Path,
    infrared_root: Path,
    label_root: Path,
):

    visible_path = find_image(
        visible_root,
        key,
    )

    infrared_path = find_image(
        infrared_root,
        key,
    )

    label_path = (
        label_root
        / f"{key}.txt"
    )

    if not label_path.is_file():
        raise FileNotFoundError(
            label_path
        )

    visible_native = imread_unicode(
        visible_path,
        cv2.IMREAD_COLOR,
    )

    infrared_native = imread_unicode(
        infrared_path,
        cv2.IMREAD_UNCHANGED,
    )

    vh, vw = visible_native.shape[:2]
    ih, iw = infrared_native.shape[:2]

    if (
        vw != EXPECTED_RAW_W
        or vh != EXPECTED_RAW_H
    ):
        raise RuntimeError(
            f"{key}: visible size "
            f"{vw}x{vh}, expected 960x768"
        )

    if (
        iw != EXPECTED_RAW_W
        or ih != EXPECTED_RAW_H
    ):
        raise RuntimeError(
            f"{key}: infrared size "
            f"{iw}x{ih}, expected 960x768"
        )

    bbox_np, has_bbox_bool = (
        call_frozen_bbox_loader(
            label_path,
            EXPECTED_RAW_W,
            EXPECTED_RAW_H,
        )
    )

    visible = image_to_rgb_tensor(
        visible_native,
        EXPECTED_INPUT_W,
        EXPECTED_INPUT_H,
        device,
    )

    infrared = image_to_rgb_tensor(
        infrared_native,
        EXPECTED_INPUT_W,
        EXPECTED_INPUT_H,
        device,
    )

    bbox = torch.from_numpy(
        bbox_np
    ).view(
        1, 4
    ).to(
        device=device,
        dtype=visible.dtype,
    )

    has_bbox = torch.tensor(
        [1.0 if has_bbox_bool else 0.0],
        device=device,
        dtype=visible.dtype,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    t_model0 = time.perf_counter()

    outputs = model(
        visible,
        infrared,
        bbox,
        has_bbox,
        True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    model_runtime_ms = (
        time.perf_counter()
        - t_model0
    ) * 1000.0

    geo = outputs["geo"]

    confidence = float(
        outputs[
            "confidence_error_px"
        ][0].detach().cpu()
    )

    gate = float(
        outputs[
            "local"
        ][
            "gate"
        ][0].detach().cpu()
    )

    tx_model = float(
        geo["tx"][0]
        .detach()
        .cpu()
    )

    ty_model = float(
        geo["ty"][0]
        .detach()
        .cpu()
    )

    angle_deg = float(
        geo["angle"][0]
        .detach()
        .cpu()
    )

    scale_x = float(
        geo["scale_x"][0]
        .detach()
        .cpu()
    )

    scale_y = float(
        geo["scale_y"][0]
        .detach()
        .cpu()
    )

    local_dx_model = float(
        outputs[
            "local"
        ][
            "raw_dx"
        ][0].detach().cpu()
    )

    local_dy_model = float(
        outputs[
            "local"
        ][
            "raw_dy"
        ][0].detach().cpu()
    )

    # --------------------------------------------------------
    # Frozen V31 confidence fallback decision
    # --------------------------------------------------------

    z = torch.zeros(
        1,
        device=device,
        dtype=visible.dtype,
    )

    prior_angle = torch.full_like(
        z,
        float(
            prior["base_angle_deg"]
        ),
    )

    prior_sx = torch.full_like(
        z,
        float(
            prior["base_scale_x"]
        ),
    )

    prior_sy = torch.full_like(
        z,
        float(
            prior["base_scale_y"]
        ),
    )

    prior_ir = v31.warp_anisotropic_px(
        infrared,
        z,
        z,
        prior_angle,
        prior_sx,
        prior_sy,
    )

    identity_score = float(
        v31.ncc_per_sample(
            visible,
            infrared,
        )[0].detach().cpu()
    )

    prior_score = float(
        v31.ncc_per_sample(
            visible,
            prior_ir,
        )[0].detach().cpu()
    )

    conf_threshold = float(
        calibration[
            "confidence_threshold_error_px"
        ]
    )

    gate_threshold = float(
        calibration[
            "gate_threshold"
        ]
    )

    use_fallback = (
        confidence
        > conf_threshold
    )

    # --------------------------------------------------------
    # Full-resolution replay
    # --------------------------------------------------------

    t_full0 = time.perf_counter()

    ir_full = gray_to_tensor(
        infrared_native,
        device,
    )

    full_scale_x = (
        EXPECTED_RAW_W
        / EXPECTED_INPUT_W
    )

    full_scale_y = (
        EXPECTED_RAW_H
        / EXPECTED_INPUT_H
    )

    tx_full = (
        tx_model
        * full_scale_x
    )

    ty_full = (
        ty_model
        * full_scale_y
    )

    local_dx_full = (
        local_dx_model
        * full_scale_x
    )

    local_dy_full = (
        local_dy_model
        * full_scale_y
    )

    dtype = ir_full.dtype

    def scalar(value):
        return torch.tensor(
            [float(value)],
            device=device,
            dtype=dtype,
        )

    if use_fallback:

        if prior_score >= identity_score:

            final_full = (
                v31.warp_anisotropic_px(
                    ir_full,
                    scalar(0.0),
                    scalar(0.0),
                    scalar(
                        prior[
                            "base_angle_deg"
                        ]
                    ),
                    scalar(
                        prior[
                            "base_scale_x"
                        ]
                    ),
                    scalar(
                        prior[
                            "base_scale_y"
                        ]
                    ),
                )
            )

            final_source = (
                "fallback_prior"
            )

        else:

            final_full = ir_full

            final_source = (
                "fallback_identity"
            )

    else:

        global_full = (
            v31.warp_anisotropic_px(
                ir_full,
                scalar(tx_full),
                scalar(ty_full),
                scalar(angle_deg),
                scalar(scale_x),
                scalar(scale_y),
            )
        )

        if (
            gate
            >= gate_threshold
            and has_bbox_bool
        ):

            final_full = (
                base.gaussian_local_warp(
                    global_full,
                    bbox,
                    has_bbox,
                    scalar(
                        local_dx_full
                    ),
                    scalar(
                        local_dy_full
                    ),
                    radius_expand=3.0,
                )
            )

            final_source = (
                "highres_target"
            )

        else:

            final_full = global_full

            final_source = "global"

    if device.type == "cuda":
        torch.cuda.synchronize()

    full_warp_runtime_ms = (
        time.perf_counter()
        - t_full0
    ) * 1000.0

    registered = (
        tensor_gray_to_u8(
            final_full[0, 0]
        )
    )

    if registered.shape != (
        EXPECTED_RAW_H,
        EXPECTED_RAW_W,
    ):
        raise RuntimeError(
            f"{key}: output shape "
            f"{registered.shape}"
        )

    metadata = {
        "key": key,

        "visible_path":
            str(visible_path),

        "infrared_path":
            str(infrared_path),

        "label_path":
            str(label_path),

        "output_width":
            EXPECTED_RAW_W,

        "output_height":
            EXPECTED_RAW_H,

        "model_input_width":
            EXPECTED_INPUT_W,

        "model_input_height":
            EXPECTED_INPUT_H,

        "bbox_normalized":
            [
                float(x)
                for x in bbox_np
            ],

        "has_bbox":
            bool(has_bbox_bool),

        "confidence_error_px":
            confidence,

        "confidence_threshold_px":
            conf_threshold,

        "local_gate":
            gate,

        "gate_threshold":
            gate_threshold,

        "final_source":
            final_source,

        "global_tx_model_px":
            tx_model,

        "global_ty_model_px":
            ty_model,

        "global_tx_full_px":
            tx_full,

        "global_ty_full_px":
            ty_full,

        "angle_deg":
            angle_deg,

        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "local_dx_model_px":
            local_dx_model,

        "local_dy_model_px":
            local_dy_model,

        "local_dx_full_px":
            local_dx_full,

        "local_dy_full_px":
            local_dy_full,

        "identity_ncc":
            identity_score,

        "prior_ncc":
            prior_score,

        "model_runtime_ms":
            model_runtime_ms,

        "full_warp_runtime_ms":
            full_warp_runtime_ms,
    }

    return (
        registered,
        metadata,
    )
