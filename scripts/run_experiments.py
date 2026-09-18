"""Run the full experiment suite and regenerate every figure/table.

    python scripts/run_experiments.py            # matrix + robustness + figures
    python scripts/run_experiments.py --quick    # 5 images each
    python scripts/run_experiments.py --skip-robustness

Steps (each is its own script under experiments/ and can be run alone):
  1. experiments/run_matrix.py       -> results/matrix/
  2. experiments/run_robustness.py   -> results/robustness/
  3. experiments/generate_figures.py -> results/figures/, results/tables/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--include-shipped-64-002", action="store_true")
    args = parser.parse_args(argv)

    quick = ["--quick"] if args.quick else []
    steps: list[list[str]] = []
    if not args.skip_matrix:
        extra = ["--include-shipped-64-002"] if args.include_shipped_64_002 else []
        steps.append([sys.executable, str(PROJECT_ROOT / "experiments" / "run_matrix.py"), *quick, *extra])
    if not args.skip_robustness:
        steps.append([sys.executable, str(PROJECT_ROOT / "experiments" / "run_robustness.py"), *quick])
    steps.append([sys.executable, str(PROJECT_ROOT / "experiments" / "generate_figures.py")])
    for cmd in steps:
        print(f"\n$ {' '.join(Path(c).name if i < 2 else c for i, c in enumerate(cmd))}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"step failed with exit code {rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
