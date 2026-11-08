"""Train the final Proposed Method target-local branch from a trained Base Global–Local Model checkpoint."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from target_registration.training import train_local


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initialization", required=True, type=Path, help="Trained Base Global–Local Model best_joint.pt")
    parser.add_argument("--calibration", required=True, type=Path, help="Base Global–Local Model validation calibration")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    train_local(args.config, args.output, ROOT, args.device, args.initialization, args.calibration)
