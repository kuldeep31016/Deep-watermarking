"""Blind accuracy per bit position (= per singular-value index).

Embeds random payloads into held-out test images and decodes them with a
windowed decoder, then reports accuracy for every bit index separately. This
answers *where in the singular-value spectrum* the blind decoder works: the
leading singular values decay steeply (sigma_0 >> sigma_1 >> ...), so a
"local trend" window has little to compare against there, while the smooth
mid-spectrum is decodable.

Usage::

    python experiments/per_bit_accuracy.py --bits 64 --alpha 0.015
    python experiments/per_bit_accuracy.py --checkpoint path/to/windowed_cnn_best.pt --alpha 0.02

Writes ``results/per_bit/<bits>bit_a<alpha>.csv`` and ``.png``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_matrix import load_images  # noqa: E402
from experiments.train_blind_grid import checkpoint_for  # noqa: E402
from src.evaluation.windowed_extract import WindowedExtractor  # noqa: E402
from src.watermark.embed import EmbedConfig, embed  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.015)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "processed" / "div2k_256" / "test")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--payloads-per-image", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "per_bit")
    args = parser.parse_args(argv)

    ckpt = args.checkpoint or checkpoint_for(args.bits, args.alpha)
    if ckpt is None:
        print(f"no decoder for {args.bits} bits / alpha {args.alpha}", file=sys.stderr)
        return 1
    decoder = WindowedExtractor.from_checkpoint(ckpt)
    bits = decoder.config.bit_length
    images = load_images(args.images, args.n_images, 256)
    cfg = EmbedConfig(alpha=args.alpha, bit_length=bits)

    correct = np.zeros(bits)
    total = 0
    for idx, (_, image) in enumerate(images):
        for k in range(args.payloads_per_image):
            payload = np.random.default_rng([7, idx, k]).integers(0, 2, bits)
            wm = embed(image, payload.tolist(), cfg).watermarked_image
            pred = np.asarray(decoder.extract_bits(wm))
            correct += pred == payload
            total += 1
    acc = correct / total

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{bits}bit_a{args.alpha:.3f}"
    csv_path, png_path = stem.parent / (stem.name + ".csv"), stem.parent / (stem.name + ".png")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bit_index", "accuracy", "n"])
        for i, a in enumerate(acc):
            w.writerow([i, round(float(a), 4), total])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(range(bits), acc, color="#3b82f6")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("bit index (= singular-value index)")
        ax.set_ylabel("blind accuracy")
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"{bits}-bit, alpha={args.alpha}, {total} (image, payload) pairs, {Path(ckpt).parent.name}")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc!r}")

    groups = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
    print(f"decoder {ckpt}  overall accuracy {acc.mean():.4f}  n={total}")
    for lo, hi in groups:
        if lo < bits:
            print(f"  bits {lo:>3}-{min(hi, bits) - 1:<3}: {acc[lo:min(hi, bits)].mean():.4f}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
