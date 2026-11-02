# Private data interface

No images, annotations, split membership lists or fitted artifacts are supplied.
All paths below describe a user-created dataset outside the repository.

## Image and visible-box inputs

- Original dimensions: 960 × 768 for both modalities.
- Matching filenames: the same string ID in each modality and label folder.
- Image extensions supported by wrappers: `.jpg`, `.jpeg`, `.png`, `.bmp`,
  `.tif`, `.tiff` (lowercase extensions).
- Original experiment inputs are stored visible RGB and pseudocolor infrared
  images, decoded by OpenCV. Do not silently replace pseudocolor inputs with
  a different infrared representation when attempting exact reproduction.
- Labels use YOLO rows: `class_id center_x center_y width height`, with normalized
  coordinates. The original loader selects the largest class-0 box and converts
  it to pixel-coordinate `x1,y1,x2,y2` before network normalization.
- Inference requires a label file per ID. An empty file produces `has_bbox=False`
  and disables target-local correction. The CLI requires an explicit ID list.
- Image inputs are supplied by users; no acquisition metadata or example image
  from the confidential dataset is included.

## Training annotations

The privately supplied annotation CSV must contain these column names:

```text
key
image_width, image_height
visible_uav_cx, visible_uav_cy
ir_uav_cx_gt_raw, ir_uav_cy_gt_raw
tx_px, ty_px, angle_deg, scale_x, scale_y
local_residual_dx_px, local_residual_dy_px
```

Coordinates and displacements are in original-image pixels. Positive image axes
point right and down. Transformation conventions follow
`core/v31.py::transform_points_anisotropic` and its centered inverse-sampling
counterpart `warp_anisotropic_px`; preserve those conventions when constructing
new annotations. The visible image is fixed and infrared image is moving.
The two manual local-correction fields are distinct from target-center-derived
supervision used by the final V32 optimization stage.

The split JSON has three keys: `train`, `val`, `test`, each mapping to a list of
string IDs. IDs may contain letters, digits, underscores and hyphens; lists must
be nonempty, unique and mutually disjoint. Historical split tags in an annotation
CSV are ignored. The paper protocol uses 1147/164/328 pairs. For another dataset,
explicitly change `expected_counts` in your private config; this defines a new
experiment and does not reproduce the paper's split.

Training binds train/validation images and annotations only. Test IDs are used to
check split separation; test images are not opened by the training data binder.
Inference is independent of the manual registration GT CSV.

## Calibration and checkpoints

No trained checkpoint or fitted threshold is included. Train V31 and then V32
to generate `best_joint.pt`, `v32_final.pt` and matching calibration files.
Alternatively use separately authorized checkpoints from the same architecture.
V32 checkpoint dictionaries contain `model` and `prior_stats`; the latter contains
`base_angle_deg`, `base_scale_x`, `base_scale_y`.

Inference calibration must contain numeric `confidence_threshold_error_px` and
`gate_threshold`. The example JSON intentionally contains `null` so it cannot
be mistaken for the actual validation-calibrated parameters. Use a checkpoint
and calibration produced by the same training protocol.

## Evaluation annotations

The private target-center CSV needs:

```text
key, visible_uav_cx, visible_uav_cy, ir_uav_cx_gt_raw, ir_uav_cy_gt_raw
```

Coordinates refer to the original 960 × 768 images. By default the evaluation
expects exact correspondence between GT IDs and saved transform IDs. For a
deliberate inference subset use `--allow-subset`; the reported sample count is
always the actual count processed. Do not interpret subset results as the full
paper benchmark. Evaluation outputs remain private.
