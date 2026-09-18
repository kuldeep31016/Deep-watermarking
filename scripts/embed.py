"""Embed a watermark into one image from the command line.

    python scripts/embed.py --image input.jpg --watermark-length 64 --alpha 0.01 \
        [--uuid 123e4567-...] [--out watermarked.png] [--residual residual.png]

Writes the watermarked PNG plus ``<out>.payload.json`` (UUID, bits, config) so
``scripts/extract.py`` can score the recovered bits later. Prints PSNR / SSIM
and the non-blind reference recovery. Uses the frozen embedder
(``src.watermark.embed``) and the UUID -> SHA-256 payload generator.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid as _uuid
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import quality_report, recovery_report  # noqa: E402
from src.watermark.embed import EmbedConfig, compute_residual, embed, extract_traditional  # noqa: E402
from src.watermark.watermark_generator import SUPPORTED_BIT_LENGTHS, generate_from_uuid  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--watermark-length", type=int, default=64, choices=sorted(SUPPORTED_BIT_LENGTHS))
    parser.add_argument("--alpha", type=float, default=0.010)
    parser.add_argument("--uuid", default=None, help="UUID to hash (default: a fresh uuid4)")
    parser.add_argument("--wavelet", default="haar")
    parser.add_argument("--out", type=Path, default=None, help="default: <image>_wm.png")
    parser.add_argument("--residual", type=Path, default=None, help="optional amplified |orig - wm| PNG")
    args = parser.parse_args(argv)

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"could not read {args.image}", file=sys.stderr)
        return 1
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    identifier = str(_uuid.UUID(args.uuid)) if args.uuid else str(_uuid.uuid4())
    bits = generate_from_uuid(identifier, args.watermark_length)
    h, w = rgb.shape[:2]
    per_band = min((h + 1) // 2, (w + 1) // 2)
    extra = ("HL",) if args.watermark_length > per_band else ()
    config = EmbedConfig(wavelet=args.wavelet, alpha=args.alpha, bit_length=args.watermark_length, extra_subbands=extra)

    result = embed(rgb, bits, config)
    out = args.out or args.image.with_name(args.image.stem + "_wm.png")
    cv2.imwrite(str(out), cv2.cvtColor(result.watermarked_image, cv2.COLOR_RGB2BGR))
    if args.residual:
        cv2.imwrite(str(args.residual), cv2.cvtColor(compute_residual(rgb, result.watermarked_image), cv2.COLOR_RGB2BGR))

    quality = quality_report(rgb, result.watermarked_image)
    recovery = recovery_report(bits, extract_traditional(result.watermarked_image, rgb, config))
    payload = {
        "uuid": identifier,
        "bit_length": args.watermark_length,
        "bits": "".join(map(str, bits)),
        "alpha": args.alpha,
        "wavelet": args.wavelet,
        "subband_order": list(config.subband_order),
        "image_size": [int(w), int(h)],
        "quality": {k: round(v, 6) for k, v in quality.items()},
        "nonblind_reference": {k: round(v, 6) for k, v in recovery.items()},
    }
    payload_path = out.with_suffix(out.suffix + ".payload.json")
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"watermarked : {out}")
    print(f"payload     : {payload_path}")
    print(f"uuid        : {identifier}")
    print(f"bits ({args.watermark_length:>3})  : {payload['bits']}")
    print(f"PSNR / SSIM : {quality['psnr']:.2f} dB / {quality['ssim']:.4f}")
    print(f"non-blind reference BER: {recovery['ber']:.4f} (needs the original; blind decode: scripts/extract.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
