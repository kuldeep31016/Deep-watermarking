"""Stream-ingest DIV2K official archives into data/processed/div2k_256.

Disk-friendly alternative to scripts/setup_div2k.py: reads the PNGs straight
from the .zip archives one at a time, resizes to 256x256 (INTER_AREA, matching
src/utils/dataset_pipeline.process_dataset), and writes only the small processed
squares - no full-size PNG files ever touch disk. Uses the documented
700/100/100 split:

    train      = DIV2K IDs 0001-0700
    validation = DIV2K IDs 0701-0800
    test       = DIV2K IDs 0801-0900

Source variants (``--variant``), all from https://data.vision.ee.ethz.ch/cvl/DIV2K/:

    HR  (default)  DIV2K_train_HR.zip + DIV2K_valid_HR.zip            (~3.5 GB + 0.45 GB)
    X4             DIV2K_train_LR_bicubic_X4.zip + DIV2K_valid_LR_bicubic_X4.zip
                   (~247 MB + 32 MB; the same 900 photos bicubic-downsampled 4x
                   by the DIV2K authors, ~510x340). Because every image is
                   area-resized to 256x256 anyway, X4 is a practical substitute
                   when disk is scarce; the provenance is recorded in
                   data/processed/div2k_256/SOURCE.json so results can say which
                   variant they were produced from.

Run: python scripts/ingest_div2k_stream.py [--variant HR|X4] [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "data" / "raw" / "archives"
IMAGE_SIZE = 256
# "DIV2K_train_HR/0001.png" (HR) or "DIV2K_train_LR_bicubic/X4/0001x4.png" (X4)
ID_PNG = re.compile(r"^(?:.*/)?(\d{4})(?:x\d)?\.png$")

VARIANTS = {
    "HR": ("DIV2K_train_HR.zip", "DIV2K_valid_HR.zip"),
    "X4": ("DIV2K_train_LR_bicubic_X4.zip", "DIV2K_valid_LR_bicubic_X4.zip"),
}


def _split_for(image_id: int) -> str:
    if 1 <= image_id <= 700:
        return "train"
    if 701 <= image_id <= 800:
        return "validation"
    if 801 <= image_id <= 900:
        return "test"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="HR")
    parser.add_argument("--check", action="store_true", help="only report split counts and exit")
    args = parser.parse_args(argv)

    out_root = ROOT / "data" / "processed" / "div2k_256"
    expected = {"train": 700, "validation": 100, "test": 100}
    if args.check:
        counts = {k: len(list((out_root / k).glob("*.png"))) for k in expected}
        source = out_root / "SOURCE.json"
        print(f"[check] {counts} source={source.read_text() if source.is_file() else 'unknown'}")
        return 0 if counts == expected else 1

    train_zip, valid_zip = (ARCHIVES / name for name in VARIANTS[args.variant])
    for z in (train_zip, valid_zip):
        if not z.is_file():
            print(
                f"missing archive: {z}\n"
                f"  curl -o {z} https://data.vision.ee.ethz.ch/cvl/DIV2K/{z.name}"
            )
            return 1

    written: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for z in (train_zip, valid_zip):
        print(f"[ingest] streaming {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            for member in zf.infolist():
                match = ID_PNG.match(member.filename)
                if not match:
                    continue
                image_id = int(match.group(1))
                split = _split_for(image_id)
                if not split:
                    continue
                target = out_root / split / f"{image_id:04d}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file():
                    continue
                data = zf.read(member)
                arr = np.frombuffer(data, dtype="uint8")
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"  unreadable {member.filename}")
                    continue
                resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(target), resized):
                    print(f"  write failure {target}")
                    return 1
                written[split] += 1
                if written[split] % 100 == 0:
                    print(f"  ... {member.filename} -> {split} (total {written[split]})")

    print(f"[ingest] done: {written}")
    counts = {k: len(list((out_root / k).glob("*.png"))) for k in expected}
    if counts != expected:
        print(f"[ingest] WARNING: expected {expected}, got {counts}")
        return 1
    (out_root / "SOURCE.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "archives": [train_zip.name, valid_zip.name],
                "image_size": IMAGE_SIZE,
                "interpolation": "INTER_AREA",
                "split": {"train": "0001-0700", "validation": "0701-0800", "test": "0801-0900"},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[ingest] data/processed/div2k_256 ready (image_size={IMAGE_SIZE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())