"""Train the windowed 1D-CNN blind decoder(s).

    python scripts/train.py                          # one decoder from configs/windowed_cnn.yaml
    python scripts/train.py --alpha 0.01 --bit-length 32 --payload-mode random
    python scripts/train.py --grid                   # every (bits, alpha) point (experiments/train_blind_grid.py)
    python scripts/train.py --grid --quick

Thin wrapper: all arguments after the flags are forwarded unchanged to
``training/colab_train_decoder.py --processed`` (single run) or
``experiments/train_blind_grid.py`` (``--grid``). Requires
``data/processed/div2k_256`` (``python scripts/ingest_div2k_stream.py``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--grid" in argv:
        argv.remove("--grid")
        cmd = [sys.executable, str(PROJECT_ROOT / "experiments" / "train_blind_grid.py"), *argv]
    else:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "training" / "colab_train_decoder.py"),
            "--project-root",
            str(PROJECT_ROOT),
            "--processed",
            *argv,
        ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
