"""The experiment scripts run end-to-end on a tiny synthetic cover set and
write every artifact they promise (no DIV2K needed)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments import generate_figures, run_matrix, run_robustness, train_blind_grid
from src.evaluation import attacks
from src.models.windowed_cnn import WindowedCNNConfig, WindowedCNNExtractor, save_windowed_checkpoint


def _write_covers(root: Path, n: int = 2, size: int = 128) -> Path:
    root.mkdir(parents=True)
    for i in range(n):
        rng = np.random.default_rng(i)
        yy, xx = np.mgrid[0:size, 0:size]
        base = 128 + 90 * np.sin(xx / (11.0 + i)) * np.cos(yy / 19.0)
        img = np.stack([base, np.roll(base, 3, 0), np.roll(base, 7, 1)], axis=2) + rng.normal(0, 5, (size, size, 3))
        cv2.imwrite(str(root / f"{i:04d}.png"), np.clip(img, 0, 255).astype(np.uint8))
    (root / "SOURCE.json").write_text(json.dumps({"variant": "synthetic-test"}))
    return root


def test_matrix_runner_writes_all_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    covers = _write_covers(tmp_path / "covers")
    monkeypatch.setattr(train_blind_grid, "GRID_DIR", tmp_path / "empty_grid")  # no trained decoders
    out = tmp_path / "results" / "matrix"
    rc = run_matrix.run(
        ["--images", str(covers), "--results-dir", str(out), "--image-size", "128", "--bits", "8", "32", "--alphas", "0.005", "0.015"]
    )
    assert rc == 0
    for name in ("per_image.csv", "summary.csv", "summary.json", "tables/matrix.md"):
        assert (out / name).is_file(), name
    rows = list(csv.DictReader((out / "per_image.csv").open()))
    assert len(rows) == 2 * 2 * 2  # bits x alphas x images
    assert {r["bits"] for r in rows} == {"8", "32"}
    assert all(0.0 <= float(r["nonblind_bit_accuracy"]) <= 1.0 for r in rows)
    assert all(r["blind_bit_accuracy"] == "" for r in rows), "no decoder exists for these points -> blank, never substituted"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["dataset_source"] == {"variant": "synthetic-test"}
    assert summary["accuracy_format"] == "fraction in [0, 1]"


def test_matrix_uses_grid_decoder_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    covers = _write_covers(tmp_path / "covers")
    grid = tmp_path / "grid"
    monkeypatch.setattr(train_blind_grid, "GRID_DIR", grid)
    run = grid / "16bit_a0.010"
    run.mkdir(parents=True)
    save_windowed_checkpoint(str(run / "windowed_cnn_best.pt"), WindowedCNNExtractor(WindowedCNNConfig(bit_length=16, image_size=128)))
    out = tmp_path / "results" / "matrix"
    assert run_matrix.run(["--images", str(covers), "--results-dir", str(out), "--image-size", "128", "--bits", "16", "--alphas", "0.010"]) == 0
    rows = list(csv.DictReader((out / "per_image.csv").open()))
    assert all(r["blind_bit_accuracy"] != "" for r in rows)
    assert all("16bit_a0.010" in r["model_version"] for r in rows)


def test_robustness_runner_and_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    covers = _write_covers(tmp_path / "covers")
    ckpt = tmp_path / "shipped.pt"
    save_windowed_checkpoint(str(ckpt), WindowedCNNExtractor(WindowedCNNConfig(bit_length=64, image_size=128)))
    monkeypatch.setattr(run_robustness, "SHIPPED_CHECKPOINT", ckpt)
    # a small attack set so the test stays fast
    monkeypatch.setattr(
        run_robustness,
        "load_config",
        lambda: {
            "decoder": "shipped",
            "embed": {"wavelet": "haar", "mode": "symmetric", "subband": "LL", "start_sv_index": 0, "alpha": 0.02, "bit_length": 64},
            "data": {"processed_root": "unused", "eval_split": "unused", "eval_num_images": 2, "image_size": 128},
            "payload_seed": 1,
            "noise_seed": 2,
            "attacks": {"jpeg_compress": [{"quality": 90}], "salt_pepper": [{"density": 0.01}]},
        },
    )
    results = tmp_path / "results"
    assert run_robustness.run(["--images", str(covers), "--results-dir", str(results / "robustness")]) == 0
    rows = list(csv.DictReader((results / "robustness" / "per_image.csv").open()))
    assert {r["attack"] for r in rows} == {"identity", "jpeg_compress", "salt_pepper"}
    assert len(rows) == 3 * 2

    # figures need a matrix csv too
    _ = run_matrix.run(["--images", str(covers), "--results-dir", str(results / "matrix"), "--image-size", "128", "--bits", "8", "--alphas", "0.01"])
    assert generate_figures.main(["--results", str(results), "--per-image-point", "8", "0.01"]) == 0
    for name in ("psnr_vs_length.png", "ssim_vs_length.png", "bit_accuracy_vs_length.png", "ber_vs_length.png", "alpha_comparison.png", "per_image_8bit_a0.010.png", "robustness_ber.png"):
        assert (results / "figures" / name).is_file(), name
    assert (results / "tables" / "matrix_summary.md").is_file()
    assert (results / "tables" / "robustness_summary.md").is_file()


def test_every_configured_attack_is_registered() -> None:
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "robustness.yaml").read_text())["robustness"]
    assert set(cfg["attacks"]) <= set(attacks.ATTACKS)
    assert "salt_pepper" in cfg["attacks"]
