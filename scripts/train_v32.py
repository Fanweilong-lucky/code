"""Train the final V32 target-local branch from a trained V31 checkpoint."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v32_registration.training import train_v32


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initialization", required=True, type=Path, help="Trained V31 best_joint.pt")
    parser.add_argument("--calibration", required=True, type=Path, help="V31 validation calibration")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    train_v32(args.config, args.output, ROOT, args.device, args.initialization, args.calibration)
