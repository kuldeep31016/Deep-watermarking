"""Train one windowed 1D-CNN blind decoder per (bit_length, alpha) grid point.

The research paper reports blind bit accuracy for 6 watermark lengths x 3
embedding strengths, but the repository shipped a single decoder (64-bit,
alpha = 0.02). This script fills the grid by calling the real trainer
(``training/colab_train_decoder.py``) once per point with **random payloads**
(``--payload-mode random``) so every decoder is evaluated on payloads it never
saw during training.

Each point writes a self-contained run directory::

    models/windowed_cnn_grid/<bits>bit_a<alpha>/
        windowed_cnn_best.pt   windowed_cnn_last.pt   metrics.json
        training_config.json   training_history.csv   dataset_split.json ...

plus ``models/windowed_cnn_grid/index.json`` listing every finished point. The
experiment matrix (``experiments/run_matrix.py``) reads that index to pick the
blind decoder for each configuration.

A 256x256 cover's LL sub-band has 128 singular values; wider payloads overflow
into HL (then LH, HH) exactly as the frozen embedder does, and the decoder reads
the same sub-bands (``WindowedCNNConfig.extra_subbands``). 256-bit points use
half as many payloads per image to keep training time comparable.

Usage::

    python experiments/train_blind_grid.py                      # full grid, 50 epochs
    python experiments/train_blind_grid.py --quick              # 2 epochs, 40 images
    python experiments/train_blind_grid.py --bits 8 64 --alphas 0.005 0.015
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRAINER = PROJECT_ROOT / "training" / "colab_train_decoder.py"
GRID_DIR = PROJECT_ROOT / "models" / "windowed_cnn_grid"
CACHE_DIR = PROJECT_ROOT / "content" / "windows_grid"

DEFAULT_BITS = (8, 16, 32, 64, 128, 256)
DEFAULT_ALPHAS = (0.005, 0.010, 0.015, 0.020)
MAX_BITS = 512  # LL + HL + LH + HH of a 256x256 cover


def run_dir(bits: int, alpha: float) -> Path:
    return GRID_DIR / f"{bits}bit_a{alpha:.3f}"


def checkpoint_for(bits: int, alpha: float) -> Path | None:
    """The trained grid checkpoint for a configuration (unpacked from its
    committed ``windowed_cnn_model_package.zip`` if needed), or ``None``."""
    from src.evaluation.decoder_loader import ensure_windowed_checkpoint

    return ensure_windowed_checkpoint(run_dir(bits, alpha) / "windowed_cnn_best.pt")


def _train_point(
    bits: int,
    alpha: float,
    *,
    epochs: int | None,
    payloads_per_image: int,
    limits: dict[str, int | None],
    python: str,
) -> dict:
    out = run_dir(bits, alpha)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-u",  # unbuffered so train.log is readable while the point trains
        str(TRAINER),
        "--project-root",
        str(PROJECT_ROOT),
        "--processed",
        "--payload-mode",
        "random",
        "--bit-length",
        str(bits),
        "--alpha",
        f"{alpha:.3f}",
        "--payloads-per-image",
        str(payloads_per_image),
        "--out-dir",
        str(out),
        "--cache-dir",
        str(CACHE_DIR / out.name),
        "--no-copy-to-repo",
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    for key, value in limits.items():
        if value is not None:
            cmd += [f"--{key.replace('_', '-')}", str(value)]
    t0 = time.time()
    log = out / "train.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    return {
        "bits": bits,
        "alpha": alpha,
        "run_dir": str(out.relative_to(PROJECT_ROOT)),
        "returncode": proc.returncode,
        "train_seconds": round(elapsed, 1),
        "checkpoint": (
            str((out / "windowed_cnn_best.pt").relative_to(PROJECT_ROOT))
            if (out / "windowed_cnn_best.pt").is_file()
            else None
        ),
        "test_bit_accuracy": (metrics or {}).get("test_bit_level", {}).get("bit_accuracy"),
        "test_ber": (metrics or {}).get("test_bit_level", {}).get("ber"),
        "best_epoch": (metrics or {}).get("best_epoch"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--epochs", type=int, default=None, help="default: configs/windowed_cnn.yaml (50)")
    parser.add_argument("--payloads-per-image", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="2 epochs on 40/10/10 images")
    parser.add_argument("--force", action="store_true", help="retrain points that already have a checkpoint")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    limits: dict[str, int | None] = {"train_images": None, "val_images": None, "test_images": None}
    epochs = args.epochs
    if args.quick:
        limits = {"train_images": 40, "val_images": 10, "test_images": 10}
        epochs = epochs or 2

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    index_path = GRID_DIR / "index.json"
    index: dict = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"points": {}}

    for bits in args.bits:
        if bits > MAX_BITS:
            print(f"[grid] skip {bits} bits: exceeds the {MAX_BITS}-bit single-level capacity of a 256x256 cover")
            continue
        for alpha in args.alphas:
            key = run_dir(bits, alpha).name
            if not args.force and checkpoint_for(bits, alpha) is not None and key in index["points"]:
                print(f"[grid] {key}: already trained (use --force to retrain)")
                continue
            print(f"[grid] training {key} ...")
            entry = _train_point(
                bits,
                alpha,
                epochs=epochs,
                payloads_per_image=max(1, args.payloads_per_image // 2) if bits >= 256 else args.payloads_per_image,
                limits=limits,
                python=args.python,
            )
            index["points"][key] = entry
            index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
            status = "ok" if entry["returncode"] == 0 else f"FAILED rc={entry['returncode']} (see train.log)"
            print(
                f"[grid] {key}: {status} test_bit_acc={entry['test_bit_accuracy']} "
                f"({entry['train_seconds']}s)"
            )
    print(f"[grid] index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
