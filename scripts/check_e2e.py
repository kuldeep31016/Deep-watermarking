"""End-to-end check: every watermark length x alpha, through real files and the API.

For each (bits, alpha):
  generate UUID -> SHA-256 payload -> embed -> save PNG -> reload PNG ->
  blind extract (decoder trained for that point; original NOT used) -> score,
  plus the non-blind reference and, at the app's alpha (0.02), the HTTP route
  ``/api/final-model/embed`` -> ``/api/final-model/extract/blind``.

Exit code 0 means every step *ran*; accuracy is printed, not asserted (blind
accuracy is what it is - see results/). Missing decoders are reported as such.

    python scripts/check_e2e.py                 # 3 DIV2K test images (or a synthetic cover)
    python scripts/check_e2e.py --image my.jpg  # your own cover
"""

from __future__ import annotations

import argparse
import base64
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import decoder_loader  # noqa: E402
from src.evaluation.metrics import quality_report, recovery_report  # noqa: E402
from src.watermark.embed import EmbedConfig, embed, extract_traditional  # noqa: E402
from src.watermark.watermark_generator import generate_from_uuid  # noqa: E402

BITS = (8, 16, 32, 64, 128, 256)
ALPHAS = (0.005, 0.010, 0.015, 0.020)


def _covers(explicit: Path | None, n: int) -> list[tuple[str, np.ndarray]]:
    if explicit is not None:
        bgr = cv2.imread(str(explicit), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"could not read {explicit}")
        return [(explicit.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]
    test_dir = PROJECT_ROOT / "data" / "processed" / "div2k_256" / "test"
    paths = sorted(test_dir.glob("*.png"))[:n]
    if paths:
        return [(p.stem, cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) for p in paths]
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:256]
    base = 128 + 90 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    img = np.stack([base, np.roll(base, 5, 0), np.roll(base, 9, 1)], axis=2) + rng.normal(0, 5, (256, 256, 3))
    return [("synthetic", np.clip(img, 0, 255).astype(np.uint8))]


def _overflow(bits: int, shape: tuple[int, int]) -> tuple[str, ...]:
    per = min((shape[0] + 1) // 2, (shape[1] + 1) // 2)
    order = ("HL", "LH", "HH")
    extra: list[str] = []
    while bits > per * (1 + len(extra)) and len(extra) < 3:
        extra.append(order[len(extra)])
    return tuple(extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args(argv)

    covers = _covers(args.image, args.n_images)
    tmp = Path(tempfile.mkdtemp(prefix="e2e_"))
    failures: list[str] = []
    print(f"covers: {[n for n, _ in covers]}  (blind decoders: original image is never passed)\n")
    print(f"{'bits':>4} {'alpha':>6} {'PSNR':>6} {'SSIM':>6} {'non-blind':>9} {'blind':>6}  decoder")
    for bits in BITS:
        for alpha in ALPHAS:
            psnrs, ssims, nb_acc, bl_acc = [], [], [], []
            decoder_name = "—"
            for name, cover in covers:
                cfg = EmbedConfig(alpha=alpha, bit_length=bits, extra_subbands=_overflow(bits, cover.shape[:2]))
                payload = generate_from_uuid(uuid.uuid4(), bits)
                try:
                    res = embed(cover, payload, cfg)
                    path = tmp / f"{name}_{bits}_{alpha}.png"
                    assert cv2.imwrite(str(path), cv2.cvtColor(res.watermarked_image, cv2.COLOR_RGB2BGR))
                    reloaded = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    q = quality_report(cover, res.watermarked_image)
                    psnrs.append(q["psnr"]); ssims.append(q["ssim"])
                    nb_acc.append(recovery_report(payload, extract_traditional(reloaded, cover, cfg))["bit_accuracy"])
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{bits}/{alpha} embed/non-blind on {name}: {exc!r}")
                    continue
                try:
                    dec = decoder_loader.windowed_decoder(bit_length=bits, alpha=alpha)
                    decoder_name = f"{bits}bit_a{alpha:.3f}"
                    bl_acc.append(recovery_report(payload, dec.extract_bits(str(path)))["bit_accuracy"])
                except decoder_loader.DecoderUnavailableError:
                    decoder_name = "NO DECODER"
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{bits}/{alpha} blind on {name}: {exc!r}")
            fmt = lambda v: f"{np.mean(v):.3f}" if v else "  n/a"  # noqa: E731
            print(f"{bits:>4} {alpha:>6.3f} {np.mean(psnrs) if psnrs else 0:>6.2f} {fmt(ssims):>6} {fmt(nb_acc):>9} {fmt(bl_acc):>6}  {decoder_name}")

    if not args.skip_api:
        print("\nHTTP API at the app's fixed alpha=0.02 (payload_source=uuid):")
        from fastapi.testclient import TestClient

        from src.app.main import app

        client = TestClient(app)
        info = client.get("/api/final-model/info").json()
        print(f"  decoder_mode={info['decoder_mode']} windowed sizes={info['windowed_cnn'].get('available_sizes')}")
        name, cover = covers[0]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(cover, cv2.COLOR_RGB2BGR))
        for bits in BITS:
            r = client.post("/api/final-model/embed", files={"image": ("c.png", buf.tobytes(), "image/png")},
                            data={"payload_source": "uuid", "payload_uuid": str(uuid.uuid4()), "bit_length": bits})
            if r.status_code != 200:
                failures.append(f"API embed {bits}: {r.status_code} {r.text[:120]}")
                print(f"  {bits:>4} bits: embed FAILED {r.status_code} {r.json().get('detail', '')[:100]}")
                continue
            d = r.json()
            wm = base64.b64decode(d["download"]["data_uri"].split(",", 1)[1])
            r2 = client.post("/api/final-model/extract/blind", files={"image": ("wm.png", wm, "image/png")},
                             data={"payload_bit_length": bits, "expected_bits": d["payload"]["bit_string"]})
            if r2.status_code != 200:
                failures.append(f"API blind {bits}: {r2.status_code} {r2.text[:120]}")
                print(f"  {bits:>4} bits: blind FAILED {r2.status_code} {r2.json().get('detail', '')[:100]}")
                continue
            b = r2.json()
            if not b.get("blind_supported", True):
                print(f"  {bits:>4} bits: embed ok (PSNR {d['quality_metrics']['psnr_db']}) | blind unsupported: {b['reason'][:80]}")
                continue
            print(f"  {bits:>4} bits: embed ok (PSNR {d['quality_metrics']['psnr_db']}, non-blind acc "
                  f"{d['recovery_metrics']['bit_accuracy']}) | blind {b['decoder']} {b['model']} acc "
                  f"{b['reference_scoring']['bit_accuracy']}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("all steps ran (accuracy values above are measured, not asserted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
