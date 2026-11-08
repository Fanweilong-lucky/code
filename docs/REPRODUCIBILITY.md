# Implementation and reproduction scope

## Implementation

`core/base.py` implements grayscale–gradient preprocessing, the pyramid encoder,
correlation, ROI sampling, Gaussian masks and Gaussian Local Warp.
`core/global_local_base.py` implements global geometry, target gating, reliability estimation,
training losses and the first three stages. `core/proposed.py` adds the 1/4-resolution
target-local correlation branch and its final optimization stage. The base
model and target-local module form one staged registration pipeline.

## Training protocol

The seed is 20260812. The first three stages use AdamW, batch size 2, weight decay
1e-4, and cosine scheduling within each stage:

| Stage | Epochs | Initial learning rate | Scope |
|---|---:|---:|---|
| A | 45 | 2e-4 | Global registration |
| B | 25 | 2e-4 | Target-region optimization |
| C | 15 | 4e-5 | Joint global–local optimization |
| D | 15 | 1e-4 | Target-local correlation branch only |

Stages A–C train the Base Global–Local Model from random initialization.
Stage D completes the Proposed Method by loading the trained base model and
initializing the target-local correlation branch from its fine-correlation
projection. Only the new branch is updated in stage D.
Selection in stages A–C uses validation loss. Stage D uses validation mean
target error and its validation-calibrated gate, with the base model's reliability
threshold held fixed. Selection and calibration use validation data only.
During training, the validation target-center criterion applies the predicted
local displacement to the globally transformed center. Final point-TRE
evaluation instead inverts the Gaussian sampling field described below.

The target-local feature window is 24 × 24 on 1/4-resolution features, with
search radius 4. Valid center-derived local supervision is limited to 45
original-image pixels, and the entropy regularization coefficient is 0.001.
Edge micro-refinement is disabled in the released inference path.

CUDA training uses automatic mixed precision. CPU software tests do not
establish bitwise GPU reproducibility or replace complete dataset evaluation.

## Coordinate and output conventions

Model input: 480 × 384; original images and reported TRE: 960 × 768.
Model-space translations and local displacement are multiplied by two during
full-resolution rendering; angles and scales are unchanged.

The global forward point transform and backward image sampler share the
original image-center convention. Local image resampling uses
`source = output - gaussian_weight(output) * displacement`. Point evaluation
therefore inverts that field numerically, with at most 100 iterations and a
1e-7-pixel convergence tolerance. Evaluation reports nonconvergence rather than
silently discarding such samples.

## Review materials and evaluation scope

Only source code is provided for confidential peer-review evaluation.
Data confidentiality prevents this package from independently reproducing the
original experimental numbers. Trained
weights, fitted calibration parameters, private annotations and split lists are
not included. A user can inspect the algorithm, run synthetic software tests,
and train/evaluate it on separately authorized compatible data.

SuperYOLO detection and external registration comparisons require their
separate implementations.
