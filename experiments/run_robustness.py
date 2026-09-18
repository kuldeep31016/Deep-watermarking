"""Attack / robustness experiment for the windowed 1D-CNN blind decoder.

For every held-out test image: embed a fixed random payload with the frozen
DWT-SVD embedder, apply every attack x severity from ``configs/robustness.yaml``
(``src/evaluation/attacks.py``), then decode the attacked image with

* the **blind** windowed 1D-CNN trained for that embed point (watermarked pixels only), and
* the **non-blind** reference decoder (needs the original), for comparison.

Nothing is simulated; every row is a real embed -> attack -> decode. Geometric
attacks are not re-synchronised before decoding.

Outputs (``results/robustness/``)::

    per_image.csv   one row per (attack, severity, image):
                    attack, severity, params, psnr_attacked, ssim_attacked,
                    blind_ber, blind_bit_accuracy, nonblind_ber, nonblind_bit_accuracy
    summary.csv     mean/std per (attack, severity)
    summary.json    summary + metadata (decoder checkpoint, dataset source ...)

Usage::

    python experiments/run_robustness.py                 # config defaults (100 images)
    python experiments/run_robustness.py --quick         # 5 images
    python experiments/run_robustness.py --decoder shipped --alpha 0.02
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

from experiments.run_matrix import _dataset_source, _rel, load_images  # noqa: E402
from experiments.train_blind_grid import checkpoint_for  # noqa: E402
from src.evaluation.attacks import ATTACKS, apply_attack, severity_of  # noqa: E402
from src.evaluation.metrics import psnr, recovery_report, ssim  # noqa: E402
from src.evaluation.windowed_extract import WindowedExtractor  # noqa: E402
from src.watermark.embed import EmbedConfig, embed, extract_traditional  # noqa: E402

SHIPPED_CHECKPOINT = PROJECT_ROOT / "models" / "experimental" / "windowed_cnn" / "windowed_cnn_best.pt"


def load_config() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "robustness.yaml").read_text(encoding="utf-8"))["robustness"]


def _resolve_checkpoint(kind: str, bits: int, alpha: float) -> Path:
    if kind == "shipped":
        if not SHIPPED_CHECKPOINT.is_file():
            raise FileNotFoundError(f"shipped decoder missing: {SHIPPED_CHECKPOINT}")
        return SHIPPED_CHECKPOINT
    path = checkpoint_for(bits, alpha)
    if path is None:
        raise FileNotFoundError(
            f"no grid decoder for {bits} bits / alpha {alpha:.3f}; run "
            f"experiments/train_blind_grid.py --bits {bits} --alphas {alpha}"
        )
    return path


def _agg(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    # finite values only (PSNR is +inf when an attack is a no-op)
    vals = np.asarray([r[key] for r in rows if r.get(key) not in (None, "")], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    return round(float(vals.mean()), 4), round(float(vals.std()), 4)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="5 images")
    parser.add_argument("--eval-num-images", type=int, default=None)
    parser.add_argument("--decoder", choices=["grid", "shipped"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--bit-length", type=int, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    e = cfg["embed"]
    alpha = args.alpha if args.alpha is not None else float(e["alpha"])
    bits = args.bit_length if args.bit_length is not None else int(e["bit_length"])
    decoder_kind = args.decoder or cfg.get("decoder", "grid")
    econf = EmbedConfig(
        wavelet=e["wavelet"], mode=e["mode"], subband=e["subband"], start_sv_index=int(e["start_sv_index"]),
        alpha=alpha, bit_length=bits,
    )
    ckpt = _resolve_checkpoint(decoder_kind, bits, alpha)
    decoder = WindowedExtractor.from_checkpoint(ckpt)
    if decoder.config.bit_length != bits:
        raise ValueError(f"decoder {ckpt} declares {decoder.config.bit_length} bits, embed point is {bits}")

    image_dir = args.images or PROJECT_ROOT / cfg["data"]["processed_root"] / cfg["data"]["eval_split"]
    limit = 5 if args.quick else (args.eval_num_images or int(cfg["data"]["eval_num_images"]))
    images = load_images(image_dir, limit, int(cfg["data"]["image_size"]))
    results_dir = args.results_dir or PROJECT_ROOT / cfg.get("results_dir", "results/robustness")
    results_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"robustness-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # attack points: identity first (severity 0 reference), then the config
    points: list[tuple[str, dict]] = [("identity", {})]
    for name, severities in cfg["attacks"].items():
        if name not in ATTACKS:
            raise ValueError(f"unknown attack {name!r} in configs/robustness.yaml")
        points += [(name, dict(p)) for p in severities]

    # embed once per image
    prepared = []
    for idx, (name, image) in enumerate(images):
        payload = np.random.default_rng(int(cfg["payload_seed"]) + idx).integers(0, 2, bits).tolist()
        wm = embed(image, payload, econf).watermarked_image
        prepared.append((name, image, wm, payload))

    rows: list[dict] = []
    for attack, params in points:
        for idx, (name, image, wm, payload) in enumerate(prepared):
            attacked = apply_attack(attack, wm, params, seed=int(cfg["noise_seed"]) + idx)
            blind_bits = decoder.extract_bits(attacked)
            nonblind_bits = extract_traditional(attacked, image, econf)
            bl = recovery_report(payload, blind_bits)
            nb = recovery_report(payload, nonblind_bits)
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "attack": attack,
                    "severity": severity_of(attack, params) if params else 0.0,
                    "params": json.dumps(params, sort_keys=True),
                    "image": name,
                    "psnr_attacked": round(psnr(wm, attacked), 4),
                    "ssim_attacked": round(ssim(wm, attacked), 6),
                    "blind_ber": round(bl["ber"], 6),
                    "blind_bit_accuracy": round(bl["bit_accuracy"], 6),
                    "nonblind_ber": round(nb["ber"], 6),
                    "nonblind_bit_accuracy": round(nb["bit_accuracy"], 6),
                }
            )
        sub = rows[-len(prepared):]
        print(
            f"[robustness] {attack:16s} {json.dumps(params):22s} blind BER {_agg(sub, 'blind_ber')[0]:.4f} "
            f"non-blind BER {_agg(sub, 'nonblind_ber')[0]:.4f} PSNR {_agg(sub, 'psnr_attacked')[0]}"
        )

    with (results_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for attack, params in points:
        key = json.dumps(params, sort_keys=True)
        sub = [r for r in rows if r["attack"] == attack and r["params"] == key]
        entry = {"attack": attack, "severity": sub[0]["severity"], "params": key, "n_images": len(sub)}
        for col in ("psnr_attacked", "ssim_attacked", "blind_ber", "blind_bit_accuracy", "nonblind_ber", "nonblind_bit_accuracy"):
            entry[f"{col}_mean"], entry[f"{col}_std"] = _agg(sub, col)
        summary.append(entry)
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (results_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "embed": {"alpha": alpha, "bit_length": bits, "wavelet": econf.wavelet, "subband": econf.subband},
                "decoder": {"kind": decoder_kind, "checkpoint": _rel(ckpt), "sha256_12": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]},
                "n_images": len(images),
                "image_dir": str(image_dir),
                "dataset_source": _dataset_source(image_dir),
                "payload_seed": cfg["payload_seed"],
                "noise_seed": cfg["noise_seed"],
                "accuracy_format": "fraction in [0, 1]",
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[robustness] wrote {results_dir}/per_image.csv, summary.csv, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
