# Target-aware Global–Local Infrared–Visible Image Registration

Companion code for **Target-aware global–local infrared–visible image registration for small unmanned aerial vehicle targets**.

## Usage notice

This code is authorized only for journal editors and peer reviewers to inspect
and run for manuscript evaluation; commercial use without permission is prohibited.

## Method

The proposed method combines a five-parameter global transformation with
target-level residual displacement correction. Grayscale–gradient encoders
support rotation and anisotropic-scale estimation followed by coarse-to-fine
translation correlation. Target-local correlation operates on fixed-size
windows from 1/4-resolution feature maps. Gaussian Local Warp applies a smooth
local correction, while global reliability estimation and target gating select
the final registration result.

- Fixed image: visible RGB; moving image: original stored infrared image.
- Network input: 480 × 384; output and TRE coordinates: 960 × 768.
- Spatial prior: visible target bounding box in YOLO format.
- Output: registered grayscale infrared image.

The original infrared inputs are stored in pseudocolor. Input preprocessing
follows the supplied implementation; full-resolution output rendering uses
OpenCV BGR-to-gray conversion. The largest class-0 visible target box is selected.
These boxes are supplied annotations, not detector predictions generated here.

## Included materials

The repository contains model definitions, staged training, registration,
target registration error (TRE) evaluation, configuration templates, and
synthetic software tests. It does not contain images, target annotations,
split membership lists, trained checkpoints, fitted calibration files, or
experimental outputs. The dataset is subject to confidentiality restrictions.
The code does not download datasets or model weights.

Reproducing the paper's numerical results requires separately authorized access
to the original data and matching checkpoints/calibration, or retraining on
those data. The supplied software tests do not reproduce those results.

## Installation

Use a separate Python environment. Python 3.11 is recommended; Python 3.9 is
also supported. Install PyTorch 2.8.0 for the intended hardware, followed by
the remaining dependencies. For CPU execution:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU execution, install the hardware-appropriate PyTorch 2.8.0 build instead
of the CPU build. Run commands from the repository root. The scripts import
`src/` automatically; `python -m pip install -e .` is optional.

## Data interface

Prepare separately authorized data outside this repository according to
[DATA_INTERFACE.md](docs/DATA_INTERFACE.md). Copy
`configs/training_data.example.json` to a private configuration file outside
the repository and set `dataset_root` and the relative input paths. Relative
`dataset_root` values are resolved against that configuration file.

The configuration examples intentionally contain no private paths or fitted
thresholds. Actual checkpoints and calibration files must be supplied separately
or generated through training. Keep all inputs and run products outside the
repository.

## Training

The **Base Global–Local Model** implements global registration, target gating,
and local warping. The **Proposed Method** adds target-local correlation.
The first three stages train the base model from random initialization.
The final stage loads that trained model and updates only the target-local
correlation branch.

```bash
python scripts/train_base.py --config ../private_config/training.json --output ../private_runs/global_local_base --device cuda
python scripts/train_local.py --config ../private_config/training.json --initialization ../private_runs/global_local_base/best_joint.pt --calibration ../private_runs/global_local_base/calibration_base.json --output ../private_runs/proposed --device cuda
```

Both `core/global_local_base.py` and `core/proposed.py` are required.
See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the training stages,
hyperparameters, coordinate conventions, and evaluation scope.

## Registration

Provide the trained checkpoint and its matching validation calibration file.
The IDs file contains one image-pair ID per line. Registration uses visible
target boxes but does not read manual infrared target centers or registration GT.

```bash
python scripts/register.py --visible ../private_data/visible --infrared ../private_data/infrared_raw --labels ../private_data/labels --ids ../private_data/inference_ids.txt --checkpoint ../private_runs/proposed/proposed_final.pt --calibration ../private_runs/proposed/calibration_proposed.json --output ../private_runs/registered --device cuda
```

Use `--device cpu` for CPU execution. Results include grayscale PNGs and
predicted transform metadata. Existing nonempty outputs are not overwritten.
An empty visible-label file disables target-local correction for that sample.
The `null` values in `configs/calibration.example.json` are not usable thresholds.

## Evaluation

```bash
python scripts/evaluate.py --gt ../private_data/test_centers.csv --results ../private_runs/registered --output ../private_runs/evaluation
```

The GT CSV is supplied separately. Evaluation reports mean and median TRE,
RMSE, P95, and tolerance rates in original 960 × 768 image coordinates. Local
target positions are obtained by numerically inverting the Gaussian
backward-sampling field. Required fields are listed in the data interface.

## Software checks

```bash
python scripts/verify_code_only.py
python -m unittest discover -s tests -v
```

Tests generate temporary synthetic inputs and remove them afterward. They
check inference, evaluation, result selection, local-branch gradients,
checkpoint loading, split separation, and the absence of disallowed artifacts.
They do not read the confidential dataset or establish the paper's accuracy.

The software checks use CPU execution with PyTorch 2.8.0+cu126,
NumPy 2.0.2, and OpenCV 4.13.0. Complete retraining and original-dataset
evaluation are outside the scope of these checks.

## Structure

```text
src/target_registration/   Network, losses, warping, data binding and training
scripts/                   Training, registration, evaluation and file checks
configs/                   Configuration templates without private values
tests/                     Synthetic software tests
docs/                      Data interface and reproducibility notes
.githooks/                 Optional local pre-commit check
.github/workflows/         Code-only checks and CPU tests
LICENSE.md                 Peer-review-only use restrictions
```

SuperYOLO and external registration baselines are separate implementations;
their code, weights, and datasets are not included. Third-party dependencies
retain their respective licenses.
