"""Proof that blind extraction never touches the original image.

The test embeds, writes the watermarked PNG to disk, drops every in-memory
array, makes the non-blind reference decoder raise if anything calls it, and
then decodes from the file alone. It also pins the extractor signature: the
only positional argument is the image.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.evaluation import decoder_loader
from src.evaluation import windowed_extract
from src.evaluation.windowed_extract import WindowedExtractor
from src.models.windowed_cnn import WindowedCNNConfig, WindowedCNNExtractor, save_windowed_checkpoint
from src.watermark import embed as embed_module
from src.watermark.embed import EmbedConfig, embed


def _cover(seed: int = 21, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 128 + 80 * np.sin(xx / 17.0) * np.cos(yy / 23.0)
    img = np.stack([base, np.roll(base, 5, 0), np.roll(base, 9, 1)], axis=2) + rng.normal(0, 4, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_extractor_api_takes_only_the_image() -> None:
    for fn in (WindowedExtractor.extract_bits, WindowedExtractor.extract_proba):
        params = [p for p in inspect.signature(fn).parameters if p != "self"]
        assert params == ["image"], f"{fn.__name__} must take only the image; got {params}"


@pytest.mark.parametrize("bits", [16, 64])
def test_blind_extraction_from_saved_file_without_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bits: int
) -> None:
    cfg = WindowedCNNConfig(bit_length=bits, image_size=128)
    ckpt = tmp_path / "tiny.pt"
    save_windowed_checkpoint(str(ckpt), WindowedCNNExtractor(cfg))

    payload = np.random.default_rng(bits).integers(0, 2, bits).tolist()
    result = embed(_cover(), payload, EmbedConfig(alpha=0.02, bit_length=bits))
    wm_path = tmp_path / "watermarked.png"
    assert cv2.imwrite(str(wm_path), cv2.cvtColor(result.watermarked_image, cv2.COLOR_RGB2BGR))
    del result  # the original image, singular values and watermarked array are gone

    # anything that needs the original must not be reachable from the blind path
    def _forbidden(*_a, **_k):
        raise AssertionError("non-blind reference decoder was called during blind extraction")

    monkeypatch.setattr(embed_module, "extract_traditional", _forbidden)
    monkeypatch.setattr(windowed_extract, "embed", None, raising=False)

    extractor = WindowedExtractor.from_checkpoint(ckpt)
    recovered = extractor.extract_bits(str(wm_path))  # the file is the ONLY input
    assert len(recovered) == bits
    assert set(recovered) <= {0, 1}
    probs = extractor.extract_proba(str(wm_path))
    assert probs.shape == (bits,) and np.all((probs >= 0) & (probs <= 1))


def test_shipped_decoder_recovers_more_than_chance_from_file_only(tmp_path: Path) -> None:
    """The committed 64-bit / alpha 0.02 decoder, on an unseen synthetic cover,
    from the PNG alone. Chance is BER 0.5; a 5-trial mean well below it shows the
    decoder reads the watermark and not the original."""
    ckpt = decoder_loader.ensure_windowed_checkpoint()
    if ckpt is None:
        pytest.skip("shipped windowed checkpoint not available")
    extractor = WindowedExtractor.from_checkpoint(ckpt)
    bers = []
    for k in range(5):
        payload = np.random.default_rng(300 + k).integers(0, 2, 64).tolist()
        wm = embed(_cover(seed=40 + k, size=256), payload, EmbedConfig(alpha=0.02, bit_length=64)).watermarked_image
        path = tmp_path / f"wm{k}.png"
        cv2.imwrite(str(path), cv2.cvtColor(wm, cv2.COLOR_RGB2BGR))
        del wm
        recovered = extractor.extract_bits(str(path))
        bers.append(float(np.mean(np.asarray(recovered) != np.asarray(payload))))
    assert np.mean(bers) < 0.45, bers
