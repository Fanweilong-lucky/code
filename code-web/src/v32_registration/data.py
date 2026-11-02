"""Bind user-supplied private data without distributing images or annotations."""

import csv
import json
import math
import re
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
GT_FIELDS = {
    "raw_w": "image_width",
    "raw_h": "image_height",
    "visible_uav_cx": "visible_uav_cx",
    "visible_uav_cy": "visible_uav_cy",
    "ir_uav_cx_gt_raw": "ir_uav_cx_gt_raw",
    "ir_uav_cy_gt_raw": "ir_uav_cy_gt_raw",
    "tx_px": "tx_px",
    "ty_px": "ty_px",
    "angle_deg": "angle_deg",
    "scale_x": "scale_x",
    "scale_y": "scale_y",
    "local_residual_dx_px": "local_residual_dx_px",
    "local_residual_dy_px": "local_residual_dy_px",
}


def validate_keys(keys, description):
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"{description}: expected a nonempty list of string IDs")
    if any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", k) for k in keys):
        raise ValueError(f"{description}: IDs must contain only letters, digits, '_' or '-'")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{description}: duplicate IDs")
    return keys


def find_private_image(directory, key):
    for extension in IMAGE_EXTENSIONS:
        path = directory / (key + extension)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image for {key!r} in {directory}")


def load_training_protocol(config, repository_root=None):
    """Read train/validation records only; test IDs check split separation.

    The config and all data are supplied by the user. Relative dataset_root
    values are resolved against the config file, not a machine-specific path.
    The optional repository_root argument preserves the training wrapper API.
    """
    config = Path(config).resolve()
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    if not cfg.get("dataset_root"):
        raise ValueError("Set dataset_root in a private copy of the example config; no data are bundled.")
    root = Path(cfg["dataset_root"]).expanduser()
    if not root.is_absolute():
        root = config.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Private dataset not found: {root}. See docs/DATA_INTERFACE.md.")

    splits = json.loads((root / cfg["split_json"]).read_text(encoding="utf-8-sig"))
    for name in ("train", "val", "test"):
        validate_keys(splits.get(name), name)
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(splits[first]) & set(splits[second]):
            raise ValueError(f"Overlapping IDs between {first} and {second}")
    counts = {name: len(splits[name]) for name in ("train", "val", "test")}
    expected = cfg.get("expected_counts", {"train": 1147, "val": 164, "test": 328})
    if counts != expected:
        raise ValueError(f"Split counts {counts} do not match configured counts {expected}")

    wanted = set(splits["train"]) | set(splits["val"])
    rows = {}
    with (root / cfg["manual_gt_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"key", *GT_FIELDS.values()} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing annotation columns: {sorted(missing)}")
        for row in reader:
            key = row["key"].strip()
            if key not in wanted:
                continue
            if key in rows:
                raise ValueError(f"Duplicate annotation for {key}")
            rows[key] = row
    if set(rows) != wanted:
        raise ValueError("Missing train/validation annotations")

    def make_pair(key):
        gt = {name: float(rows[key][column]) for name, column in GT_FIELDS.items()}
        if not all(math.isfinite(value) for value in gt.values()):
            raise ValueError(f"Non-finite annotation for {key}")
        if (gt["raw_w"], gt["raw_h"]) != (960, 768):
            raise ValueError(f"{key}: the paper protocol expects 960 x 768 images")
        if gt["scale_x"] <= 0 or gt["scale_y"] <= 0:
            raise ValueError(f"{key}: scales must be positive")
        label = root / cfg["label_dir"] / (key + ".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
        return {
            "key": key,
            "human_gt": gt,
            "visible": find_private_image(root / cfg["visible_dir"], key),
            "infrared": find_private_image(root / cfg["infrared_dir"], key),
            "label": label,
        }

    pairs = {name: [make_pair(key) for key in splits[name]] for name in ("train", "val")}
    return pairs, counts
