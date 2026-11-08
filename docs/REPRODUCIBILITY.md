# Implementation and reproduction scope

## Preserved numerical core

`core/base.py` implements grayscale–gradient preprocessing, the pyramid encoder,
correlation, ROI sampling, Gaussian masks and Gaussian Local Warp.
`core/v31.py` implements global geometry, target gating, reliability estimation,
training losses and the first three stages. `core/v32.py` adds the 1/4-resolution
target-local correlation branch and its final optimization stage. These modules
are necessary dependencies of one method, not three independent submissions.

The release retains the numerical model/loss/warp definitions from the existing
V32 implementation. Data binding and command-line paths are portable wrappers.
No private-data converter, dataset sampler, historical launcher or detector
repository is included.

## Training protocol

The seed is 20260812. The first three stages use AdamW, batch size 2, weight decay
1e-4, and cosine scheduling within each stage:

| Stage | Epochs | Initial learning rate | Scope |
|---|---:|---:|---|
| A | 45 | 2e-4 | Global registration |
| B | 25 | 2e-4 | Target-region optimization |
| C | 15 | 4e-5 | Joint global–local optimization |
| V32 | 15 | 1e-4 | Target-local correlation branch only |

The V31 stages begin from random initialization; the V32 stage loads the trained
V31 model and initializes the new branch from its fine-correlation projection.
It must not be described as a separately random-initialized complete network.
V31 stage selection uses validation loss. The V32 stage uses validation mean
target error and its validation-calibrated gate, with the V31 reliability
threshold held fixed. The supplied core performs the original per-epoch
selection/calibration, rather than using test images for either purpose.

The target-local feature window is 24 × 24 on 1/4-resolution features, with
search radius 4. Valid center-derived local supervision is limited to 45
original-image pixels, and the entropy regularization coefficient is 0.001.
Edge micro-refinement is disabled in the released inference path.

CUDA training uses automatic mixed precision. CPU synthetic tests do not
establish bitwise GPU reproducibility. No complete retraining was performed
during this code-only packaging task.

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

## Limitations of this release

Only code is public in this package. Data confidentiality prevents this package
from independently reproducing the original experimental numbers. Trained
weights, fitted calibration parameters, private annotations and split lists are
not included. A user can inspect the algorithm, run synthetic software tests,
and train/evaluate it on separately authorized compatible data.

This repository implements registration. SuperYOLO detection and external
registration comparisons require their separate implementations; they are not
silently replaced by the synthetic tests provided here.
