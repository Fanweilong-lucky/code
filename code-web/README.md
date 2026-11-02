# Target-aware Global–Local Infrared–Visible Image Registration

Code for **Target-aware global–local infrared–visible image registration for small unmanned aerial vehicle targets**.

[中文说明](README.zh-CN.md)

## Release scope

This repository contains the V32 registration implementation, staged training,
inference, point-TRE evaluation, configuration templates, and synthetic tests.
**No dataset is distributed.** Images, target annotations, split memberships,
per-image transforms/results, trained checkpoints, and fitted calibration files
are not included. The dataset is subject to confidentiality restrictions.
No dataset download or external weight download is performed by the code.

This is a code-only implementation release. Reproducing the paper's numerical
results requires separately authorized access to the original data and the
matching trained model/calibration, or retraining on those data. The synthetic
tests check software behavior, not the paper's experimental accuracy.

## Method

The fixed visible image and moving infrared image are processed at 480 × 384.
Separate grayscale–gradient encoders support rotation–anisotropic-scale
estimation and coarse-to-fine translation correlation. Target-local correlation
uses fixed-size windows on 1/4-resolution features. Gaussian Local Warp applies
a smooth target-centered correction, with global reliability estimation and
target gating selecting the output. Inference replays the selected transform at
960 × 768 and saves a grayscale infrared image.

Visible target boxes are required spatial priors; they are not produced by a
detector in this repository. The largest class-0 visible box is selected.
Do not substitute registered infrared images for the original moving inputs.
The stored original pseudocolor inputs follow the original preprocessing;
grayscale conversion for full-resolution output uses OpenCV BGR-to-gray.

## Installation

Use a separate environment; Python 3.11 is recommended (Python 3.9 is also
supported). The local CPU checks used PyTorch 2.8.0+cu126, NumPy 2.0.2 and
OpenCV 4.13.0. Install the hardware-appropriate PyTorch 2.8.0 build first.
For CPU-only checks:

```bash
python -m venv .venv
# Activate the environment using your platform's command.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run commands from this repository root. The scripts automatically import `src/`;
an editable package install (`python -m pip install -e .`) is optional.

## Code checks without any real dataset

```bash
python scripts/verify_code_only.py
python -m unittest discover -s tests -v
```

Tests generate temporary synthetic inputs at runtime and remove them afterward.
They do not read the authors' images, labels, weights or experimental results.

## Training

Prepare data outside this repository using [the data interface](docs/DATA_INTERFACE.md).
Copy `configs/training_data.example.json` to a private config outside the
repository and set `dataset_root` and the relative input paths. Relative
`dataset_root` values are resolved against that private config file.
Keep training outputs outside the repository as well.

```bash
python scripts/train_v31.py --config ../private_config/training.json --output ../private_runs/v31 --device cuda
python scripts/train_v32.py --config ../private_config/training.json --initialization ../private_runs/v31/best_joint.pt --calibration ../private_runs/v31/calibration_v31.json --output ../private_runs/v32 --device cuda
```

`v31.py` is a required implementation dependency, not a competing old release
that can be deleted. It implements the first three optimization stages; V32
implements the final target-local correlation stage. V31 starts from random
initialization. V32 starts from the trained V31 model and updates only the
target-local correlation branch. See [training details](docs/REPRODUCIBILITY.md).

## Registration inference

Supply a trained V32 checkpoint and its matching validation calibration file.
`configs/calibration.example.json` documents the required fields; its `null`
values are deliberate placeholders, not usable thresholds. Training generates
the actual calibration. The inference IDs file contains one pair ID per line.
See the data interface for visible YOLO box conventions.

```bash
python scripts/register.py --visible ../private_data/visible --infrared ../private_data/infrared_raw --labels ../private_data/labels --ids ../private_data/inference_ids.txt --checkpoint ../private_runs/v32/v32_final.pt --calibration ../private_runs/v32/calibration_v32.json --output ../private_runs/registered --device cuda
```

The network does not read manual infrared centers or registration GT at inference.
Outputs comprise grayscale PNGs and predicted transform metadata. These are
private run products and must not be added to the public repository.
Use `--device cpu` for CPU execution. Existing nonempty outputs are preserved.

## Target registration error evaluation

```bash
python scripts/evaluate.py --gt ../private_data/test_centers.csv --results ../private_runs/registered --output ../private_runs/evaluation
```

The GT CSV is supplied privately; its minimal columns are documented in
`docs/DATA_INTERFACE.md`. Evaluation uses original 960 × 768 coordinates and
reports mean/median TRE, RMSE, P95, and tolerance rates. The local point transform
is obtained by numerically inverting the actual Gaussian backward-sampling
field, rather than treating local image warping as uniform translation.

## Layout

```text
src/v32_registration/      Numerical core, data binding, training and inference
scripts/                  Training, registration, evaluation and release checks
configs/                  Empty configuration templates only
tests/                    Synthetic software tests
docs/                     Algorithm, data interface and reproduction notes
.githooks/                Optional local pre-commit check
.github/workflows/        Code-only check and CPU synthetic tests
```

## Before uploading to GitHub

Create a new repository from this folder, not from the original experiment or
sample-release directories. Run the code-only check before upload. For command
line Git, the optional hook adds an index check:

```bash
git init
git config core.hooksPath .githooks
git add .
python scripts/verify_code_only.py --tracked
```

Review the staged file list before committing. `.gitignore` and the checker
provide safeguards; neither replaces a review of new source text. No historical
Git repository or experimental artifacts are included in this package.

## Publication metadata and license

The paper title above is supplied by the authors. No DOI, acceptance status,
repository URL, or publication year is asserted. Authors should add verified
citation metadata when available and choose a code license before granting
redistribution rights. No license choice for the authors' code has been made
on their behalf. Third-party dependencies retain their respective licenses.

SuperYOLO and external registration baselines are separate projects; their code,
weights and datasets are not vendored into this V32 registration release.
