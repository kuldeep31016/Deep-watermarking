"""The 6 x 3 experiment matrix: watermark length x embedding strength.

For every ``(bit_length, alpha)`` in ``configs/baseline.yaml`` and every held-out
DIV2K test image, this script embeds a **per-image random payload** with the
frozen DWT-SVD embedder and records, from real computation only:

* image quality   - PSNR, SSIM, MSE (original vs watermarked)
* non-blind recovery - BER / bit accuracy with ``extract_traditional`` (needs the original)
* blind recovery  - BER / bit accuracy with the windowed 1D-CNN trained for that
                    exact ``(bit_length, alpha)`` by ``experiments/train_blind_grid.py``
                    (columns are empty when no such decoder exists - never
                    substituted by a decoder trained for another point)
* runtime         - embed / non-blind / blind wall-clock per image
* provenance      - image id, decoder checkpoint + sha256, experiment id, dataset source

Outputs (``results/matrix/``)::

    per_image.csv      one row per (alpha, bits, image)
    summary.csv        mean/std per (alpha, bits)
    summary.json       summary + run metadata
    tables/matrix.md   the paper-style table (PSNR / SSIM / blind accuracy)

Bit accuracy and BER are reported in the 0-1 range everywhere.

Usage::

    python experiments/run_matrix.py                 # full test split (100 images)
    python experiments/run_matrix.py --quick         # 5 images
    python experiments/run_matrix.py --images DIR    # any folder of PNG/JPG covers
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_blind_grid import checkpoint_for  # noqa: E402
from src.evaluation.metrics import quality_report, recovery_report  # noqa: E402
from src.watermark.embed import EmbedConfig, embed, extract_traditional  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "matrix"
SHIPPED_CHECKPOINT = PROJECT_ROOT / "models" / "experimental" / "windowed_cnn" / "windowed_cnn_best.pt"


def load_matrix_config() -> dict:
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "baseline.yaml").read_text(encoding="utf-8"))
    return raw["baseline"]


def load_images(image_dir: Path, limit: int | None, size: int | None) -> list[tuple[str, np.ndarray]]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no images in {image_dir}")
    out = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable image: {path}")
        if size is not None and bgr.shape[:2] != (size, size):
            bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        out.append((path.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return out


def embed_config(cfg: dict, alpha: float, n_bits: int, image_size: int) -> EmbedConfig:
    """LL only; overflow into HL/LH/HH only when the payload exceeds one sub-band."""
    per_band = image_size // 2
    order = ["HL", "LH", "HH"]
    extra: list[str] = []
    while n_bits > per_band * (1 + len(extra)):
        extra.append(order[len(extra)])
    return EmbedConfig(
        wavelet=cfg["wavelet"],
        mode=cfg["mode"],
        subband=cfg["subband"],
        extra_subbands=tuple(extra),
        alpha=alpha,
        bit_length=n_bits,
        start_sv_index=cfg["start_sv_index"],
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo, else the absolute path."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _blind_decoder(n_bits: int, alpha: float, allow_shipped: bool):
    """The windowed decoder trained for exactly this point, else ``None``."""
    from src.evaluation.windowed_extract import WindowedExtractor

    path = checkpoint_for(n_bits, alpha)
    if path is None and allow_shipped and n_bits == 64 and abs(alpha - 0.02) < 1e-9 and SHIPPED_CHECKPOINT.is_file():
        path = SHIPPED_CHECKPOINT
    if path is None:
        return None, None
    ext = WindowedExtractor.from_checkpoint(path)
    if ext.config.bit_length != n_bits:
        return None, None
    return ext, path


def _dataset_source(image_dir: Path) -> dict | str:
    for candidate in (image_dir / "SOURCE.json", image_dir.parent / "SOURCE.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return "unknown"


def _agg(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    # finite values only (PSNR is +inf when an attack is a no-op)
    vals = np.asarray([r[key] for r in rows if r.get(key) not in (None, "")], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    return round(float(vals.mean()), 4), round(float(vals.std()), 4)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "processed" / "div2k_256" / "test")
    parser.add_argument("--eval-num-images", type=int, default=None, help="default: all images in the folder")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--quick", action="store_true", help="5 images")
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--bits", type=int, nargs="+", default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--include-shipped-64-002",
        action="store_true",
        help="also evaluate the shipped 64-bit alpha=0.02 decoder as an extra matrix row",
    )
    args = parser.parse_args(argv)

    cfg = load_matrix_config()
    alphas = args.alphas or [float(a) for a in cfg["alphas"]]
    bits_list = args.bits or [int(b) for b in cfg["payload_bits"]]
    if args.include_shipped_64_002 and 0.02 not in alphas:
        alphas = alphas + [0.02]
    limit = 5 if args.quick else args.eval_num_images
    images = load_images(args.images, limit, args.image_size)
    experiment_id = f"matrix-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    seed = int(cfg["seed"])

    results_dir = args.results_dir
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    decoders_used: dict[str, dict] = {}
    for alpha in alphas:
        for n_bits in bits_list:
            econf = embed_config(cfg, alpha, n_bits, args.image_size)
            decoder, ckpt = _blind_decoder(n_bits, alpha, args.include_shipped_64_002)
            model_version = None
            if ckpt is not None:
                model_version = f"{_rel(ckpt)}@{_sha256(ckpt)}"
                decoders_used[f"{n_bits}bit_a{alpha:.3f}"] = {"checkpoint": _rel(ckpt), "sha256_12": _sha256(ckpt)}
            for idx, (name, image) in enumerate(images):
                # one fixed random payload per (bits, image): reproducible, and
                # different across images so the matrix is not one payload's luck
                payload = np.random.default_rng([seed, n_bits, idx]).integers(0, 2, n_bits).tolist()

                t0 = time.perf_counter()
                result = embed(image, payload, econf)
                t_embed = time.perf_counter() - t0
                wm = result.watermarked_image

                t0 = time.perf_counter()
                nonblind = extract_traditional(wm, image, econf)
                t_nonblind = time.perf_counter() - t0

                q = quality_report(image, wm)
                nb = recovery_report(payload, nonblind)
                row = {
                    "experiment_id": experiment_id,
                    "alpha": alpha,
                    "bits": n_bits,
                    "subband_order": "+".join(econf.subband_order),
                    "image": name,
                    "psnr": round(q["psnr"], 4),
                    "ssim": round(q["ssim"], 6),
                    "mse": round(q["mse"], 6),
                    "nonblind_ber": round(nb["ber"], 6),
                    "nonblind_bit_accuracy": round(nb["bit_accuracy"], 6),
                    "blind_ber": "",
                    "blind_bit_accuracy": "",
                    "blind_exact": "",
                    "t_embed_ms": round(t_embed * 1e3, 2),
                    "t_nonblind_ms": round(t_nonblind * 1e3, 2),
                    "t_blind_ms": "",
                    "model_version": model_version or "",
                }
                if decoder is not None:
                    # blind: the watermarked pixels only (round-tripped through PNG
                    # bytes so the decoder sees exactly what a saved file holds)
                    ok, buf = cv2.imencode(".png", cv2.cvtColor(wm, cv2.COLOR_RGB2BGR))
                    assert ok
                    reloaded = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    t0 = time.perf_counter()
                    blind_bits = decoder.extract_bits(reloaded)
                    t_blind = time.perf_counter() - t0
                    bl = recovery_report(payload, blind_bits)
                    row.update(
                        {
                            "blind_ber": round(bl["ber"], 6),
                            "blind_bit_accuracy": round(bl["bit_accuracy"], 6),
                            "blind_exact": int(bl["ber"] == 0.0),
                            "t_blind_ms": round(t_blind * 1e3, 2),
                        }
                    )
                rows.append(row)
            print(
                f"[matrix] alpha={alpha:.3f} bits={n_bits:>3} n={len(images)} "
                f"psnr={_agg([r for r in rows if r['alpha']==alpha and r['bits']==n_bits],'psnr')[0]} "
                f"blind_acc={_agg([r for r in rows if r['alpha']==alpha and r['bits']==n_bits],'blind_bit_accuracy')[0]}"
            )

    per_image_csv = results_dir / "per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict] = []
    for alpha in alphas:
        for n_bits in bits_list:
            subset = [r for r in rows if r["alpha"] == alpha and r["bits"] == n_bits]
            if not subset:
                continue
            entry = {"alpha": alpha, "bits": n_bits, "subband_order": subset[0]["subband_order"], "n_images": len(subset)}
            for key in ("psnr", "ssim", "mse", "nonblind_ber", "nonblind_bit_accuracy", "blind_ber", "blind_bit_accuracy", "t_embed_ms", "t_nonblind_ms", "t_blind_ms"):
                m, s = _agg(subset, key)
                entry[f"{key}_mean"] = m
                entry[f"{key}_std"] = s
            blind_rows = [r for r in subset if r["blind_exact"] != ""]
            entry["blind_exact_rate"] = round(sum(r["blind_exact"] for r in blind_rows) / len(blind_rows), 4) if blind_rows else None
            entry["blind_decoder"] = subset[0]["model_version"] or None
            summary.append(entry)

    summary_csv = results_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    meta = {
        "experiment_id": experiment_id,
        "description": "6x3 matrix: watermark length x alpha; frozen DWT-SVD embedder; non-blind + windowed blind decoders",
        "embedding_rule": "S'[i] = S[i] * (1 + alpha * (2*bit - 1)) on YCrCb-Y, haar, LL (+HL overflow > 128 bits)",
        "payloads": "per-(bits, image) random bits, rng([seed, bits, image_index])",
        "seed": seed,
        "image_dir": str(args.images),
        "image_size": args.image_size,
        "n_images": len(images),
        "image_ids": [n for n, _ in images],
        "dataset_source": _dataset_source(args.images),
        "blind_decoders": decoders_used,
        "accuracy_format": "fraction in [0, 1]",
        "summary": summary,
    }
    (results_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # paper-style table
    lines = [
        f"# Matrix results ({experiment_id})",
        "",
        f"n = {len(images)} images from `{args.images}`; blind columns come from the decoder trained for that exact point (blank = no such decoder).",
        "",
        "| Bits | alpha | PSNR (dB) | SSIM | Non-blind acc | Blind acc | Blind exact | Blind decoder |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in summary:
        lines.append(
            f"| {e['bits']} | {e['alpha']:.3f} | {e['psnr_mean']:.2f} | {e['ssim_mean']:.4f} | "
            f"{e['nonblind_bit_accuracy_mean']:.4f} | "
            f"{'' if e['blind_bit_accuracy_mean'] is None else f'{e['blind_bit_accuracy_mean']:.4f}'} | "
            f"{'' if e['blind_exact_rate'] is None else f'{e['blind_exact_rate']:.2f}'} | "
            f"{e['blind_decoder'] or ''} |"
        )
    (results_dir / "tables" / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[matrix] wrote {per_image_csv}, {summary_csv}, summary.json, tables/matrix.md")
    print("\n".join(lines[4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
