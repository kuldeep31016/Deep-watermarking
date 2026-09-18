"""Extract a watermark from a (watermarked) image from the command line.

Blind (default - the original image is NOT used)::

    python scripts/extract.py --image watermarked.png --watermark-length 64 --alpha 0.01
    python scripts/extract.py --image watermarked.png --payload watermarked.png.payload.json

Non-blind reference (needs the original)::

    python scripts/extract.py --image watermarked.png --original input.jpg --payload ...

The blind decoder is the windowed 1D-CNN trained for the requested
``(watermark-length, alpha)`` (``models/windowed_cnn_grid/``, produced by
``experiments/train_blind_grid.py``); for 64 bits / alpha 0.02 the shipped
checkpoint is used. Passing ``--payload`` (written by ``scripts/embed.py``) or
``--reference-bits`` scores the result with BER / bit accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import decoder_loader  # noqa: E402
from src.evaluation.metrics import recovery_report  # noqa: E402
from src.evaluation.windowed_extract import WindowedExtractor  # noqa: E402
from src.watermark.embed import EmbedConfig, extract_traditional  # noqa: E402


def _resolve_checkpoint(bits: int, alpha: float, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    return decoder_loader._resolve_windowed_checkpoint(bits, alpha)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="the watermarked image")
    parser.add_argument("--payload", type=Path, default=None, help="payload json from scripts/embed.py")
    parser.add_argument("--watermark-length", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--reference-bits", default=None, help="0/1 string to score against")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit windowed-CNN checkpoint")
    parser.add_argument("--original", type=Path, default=None, help="run the NON-blind reference decoder instead")
    args = parser.parse_args(argv)

    payload = json.loads(args.payload.read_text(encoding="utf-8")) if args.payload else {}
    bits = args.watermark_length or payload.get("bit_length")
    alpha = args.alpha if args.alpha is not None else payload.get("alpha")
    reference = args.reference_bits or payload.get("bits")
    if bits is None:
        print("--watermark-length (or --payload) is required", file=sys.stderr)
        return 2

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"could not read {args.image}", file=sys.stderr)
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.original is not None:
        if alpha is None:
            print("--alpha (or --payload) is required for the non-blind decoder", file=sys.stderr)
            return 2
        obgr = cv2.imread(str(args.original), cv2.IMREAD_COLOR)
        if obgr is None:
            print(f"could not read {args.original}", file=sys.stderr)
            return 1
        original = cv2.cvtColor(obgr, cv2.COLOR_BGR2RGB)
        extra = tuple(payload.get("subband_order", ["LL"])[1:])
        config = EmbedConfig(wavelet=payload.get("wavelet", "haar"), alpha=float(alpha), bit_length=int(bits), extra_subbands=extra)
        recovered = extract_traditional(rgb, original, config)
        mode = "non-blind (original image used)"
        probs = None
    else:
        if alpha is None:
            print("--alpha (or --payload) is required to pick the blind decoder trained for that strength", file=sys.stderr)
            return 2
        ckpt = _resolve_checkpoint(int(bits), float(alpha), args.checkpoint)
        if ckpt is None:
            print(
                f"no blind decoder for {bits} bits / alpha {alpha}: train one with "
                f"`python experiments/train_blind_grid.py --bits {bits} --alphas {alpha}` or pass --checkpoint",
                file=sys.stderr,
            )
            return 3
        extractor = WindowedExtractor.from_checkpoint(ckpt)
        if extractor.config.bit_length != int(bits):
            print(f"checkpoint {ckpt} declares {extractor.config.bit_length} bits, not {bits}", file=sys.stderr)
            return 3
        probs = extractor.extract_proba(rgb)
        recovered = (probs > 0.5).astype(int).tolist()
        mode = f"blind (windowed 1D-CNN {ckpt.relative_to(PROJECT_ROOT) if ckpt.is_relative_to(PROJECT_ROOT) else ckpt}; original NOT used)"

    print(f"mode           : {mode}")
    print(f"recovered bits : {''.join(map(str, recovered))}")
    if probs is not None:
        print(f"mean confidence: {float(np.mean(np.abs(probs - 0.5)) * 2):.4f}  (per-bit |p-0.5|*2)")
    if reference:
        ref = [int(c) for c in reference.strip()]
        if len(ref) != len(recovered):
            print(f"reference has {len(ref)} bits, recovered {len(recovered)}", file=sys.stderr)
            return 2
        rep = recovery_report(ref, recovered)
        print(f"reference bits : {reference}")
        print(f"bit accuracy   : {rep['bit_accuracy']:.4f}   BER: {rep['ber']:.4f}   NC: {rep['nc']:.4f}")
    else:
        print("(no reference bits supplied - accuracy not computed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
