"""Decoder selection for the app's blind-extraction path.

The app has two CNN blind decoders:

* ``"current"`` - the Phase 8 ``BlindExtractor`` (one CNN pass over the whole
  leading spectrum). Its checkpoints (``models/phase8_cnn/*.pt``) are **not**
  shipped with the repository.
* ``"windowed_cnn"`` - the per-bit windowed 1D-CNN (trained on DIV2K with the
  actual project embedder; see ``docs/windowed_cnn.md``). One decoder exists
  per ``(bit_length, alpha)`` point under ``models/windowed_cnn_grid/`` (from
  ``experiments/train_blind_grid.py``; each ships as a small
  ``windowed_cnn_model_package.zip`` that is unpacked on first use), plus the
  original 64-bit / alpha 0.02 demo decoder in ``content/out/``.

Selection is done by the ``DECODER_MODE`` environment variable::

    DECODER_MODE=windowed_cnn python scripts/run_app.py   # force windowed
    DECODER_MODE=current      python scripts/run_app.py   # force Phase 8

When the variable is **unset** the mode is chosen automatically: the windowed
decoder is used whenever its checkpoint is available and no Phase 8 checkpoint
is present; otherwise ``"current"``. That is what makes a fresh clone serve
blind extraction out of the box instead of answering HTTP 503.

A requested decoder that is not usable (no checkpoint / width mismatch) falls
back to ``"current"``. The resulting ``(decoder, mode, reason)`` is surfaced in
the extraction response so results are never silently produced by an
unexpected network.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from src.evaluation.windowed_extract import WindowedExtractor

__all__ = [
    "DECODER_MODES",
    "DECODER_MODE_ENV",
    "PHASE8_MODEL_DIR",
    "WINDOWED_CHECKPOINT",
    "WINDOWED_MODEL_DIR",
    "WINDOWED_PACKAGE_ZIP",
    "GRID_DIR",
    "DecoderUnavailableError",
    "DefaultDecoder",
    "default_decoder_mode",
    "ensure_windowed_checkpoint",
    "grid_checkpoint",
    "phase8_checkpoint_available",
    "windowed_available_sizes",
    "windowed_checkpoint_available",
    "windowed_decoder",
]

DECODER_MODE_ENV = "DECODER_MODE"
DECODER_MODES = ("current", "windowed_cnn")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Phase 8 checkpoints (not shipped; ``models/**`` is git-ignored).
PHASE8_MODEL_DIR = _PROJECT_ROOT / "models" / "phase8_cnn"

# Windowed-CNN artifacts. ``models/**`` is git-ignored, so the trained package
# is committed as a zip under ``content/out`` and unpacked here on first use.
WINDOWED_MODEL_DIR = _PROJECT_ROOT / "models" / "experimental" / "windowed_cnn"
WINDOWED_CHECKPOINT = WINDOWED_MODEL_DIR / "windowed_cnn_best.pt"
WINDOWED_PACKAGE_ZIP = _PROJECT_ROOT / "content" / "out" / "windowed_cnn_model_package.zip"

# Per-(bit_length, alpha) decoders trained by experiments/train_blind_grid.py.
GRID_DIR = _PROJECT_ROOT / "models" / "windowed_cnn_grid"
GRID_ALPHA_DEFAULT = 0.02  # the app's frozen embedding strength


class DefaultDecoder:
    """Sentinel for the production phase-8 CNN (selected through ``blind_extractor``)."""


class DecoderUnavailableError(RuntimeError):
    """The requested decoder exists as a concept but cannot be loaded here."""


def ensure_windowed_checkpoint(
    checkpoint: str | Path | None = None, package: str | Path | None = None
) -> Path | None:
    """Return the windowed checkpoint path, unpacking its committed zip package
    next to it if the file is missing. ``None`` when neither exists.

    ``package`` defaults to the shipped demo package for the default checkpoint
    and to ``<checkpoint dir>/windowed_cnn_model_package.zip`` otherwise.
    """
    path = Path(checkpoint) if checkpoint else WINDOWED_CHECKPOINT
    if path.is_file():
        return path
    if package is not None:
        zip_path = Path(package)
    elif checkpoint is None:
        zip_path = WINDOWED_PACKAGE_ZIP
    else:
        zip_path = path.parent / "windowed_cnn_model_package.zip"
    if zip_path.is_file():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(path.parent)
        except (OSError, zipfile.BadZipFile) as exc:  # pragma: no cover - disk problems
            print(f"[decoder_loader] warning: could not unpack {zip_path}: {exc!r}")
        if path.is_file():
            return path
    return None


def grid_checkpoint(bit_length: int, alpha: float = GRID_ALPHA_DEFAULT) -> Path | None:
    """Checkpoint of the grid decoder trained for exactly ``(bit_length, alpha)``,
    unpacked from its package if needed; ``None`` when that point was never trained."""
    run_dir = GRID_DIR / f"{bit_length}bit_a{alpha:.3f}"
    return ensure_windowed_checkpoint(run_dir / "windowed_cnn_best.pt")


def _resolve_windowed_checkpoint(bit_length: int | None, alpha: float) -> Path | None:
    if bit_length is None:
        return ensure_windowed_checkpoint()
    path = grid_checkpoint(bit_length, alpha)
    if path is None and bit_length == 64 and abs(alpha - GRID_ALPHA_DEFAULT) < 1e-9:
        path = ensure_windowed_checkpoint()  # the shipped demo decoder
    return path


def windowed_available_sizes(
    alpha: float = GRID_ALPHA_DEFAULT, candidates: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
) -> list[int]:
    """Payload widths that have a windowed decoder for ``alpha``."""
    return [n for n in candidates if _resolve_windowed_checkpoint(n, alpha) is not None]


def windowed_checkpoint_available() -> bool:
    return ensure_windowed_checkpoint() is not None


def phase8_checkpoint_available() -> bool:
    return PHASE8_MODEL_DIR.is_dir() and any(PHASE8_MODEL_DIR.glob("*.pt"))


def default_decoder_mode() -> str:
    """The decoder the blind-extraction path should use.

    ``DECODER_MODE`` set -> that mode (an unknown value is rejected with a
    warning and falls back to ``"current"``). Unset -> automatic: the windowed
    decoder when its checkpoint is available and no Phase 8 checkpoint exists,
    otherwise ``"current"``.
    """
    raw = os.environ.get(DECODER_MODE_ENV, "").strip().lower()
    if not raw:
        if windowed_checkpoint_available() and not phase8_checkpoint_available():
            return "windowed_cnn"
        return "current"
    if raw in DECODER_MODES:
        return raw
    print(
        f"[decoder_loader] warning: {DECODER_MODE_ENV}={raw!r} is not a known decoder "
        f"mode (known: {', '.join(DECODER_MODES)}); falling back to 'current'."
    )
    return "current"


def windowed_decoder(
    *,
    device: str = "cpu",
    checkpoint: str | Path | None = None,
    bit_length: int | None = None,
    alpha: float = GRID_ALPHA_DEFAULT,
) -> WindowedExtractor:
    """Load a windowed 1D-CNN decoder.

    ``checkpoint`` loads that file. Otherwise ``bit_length`` selects the grid
    decoder trained for ``(bit_length, alpha)`` (falling back to the shipped
    demo decoder for 64 bits at alpha 0.02); with neither, the shipped decoder.
    Raises :class:`DecoderUnavailableError` when no trained artifact exists.
    """
    if checkpoint is not None:
        path = ensure_windowed_checkpoint(checkpoint)
        if path is None:
            raise DecoderUnavailableError(
                f"windowed-cnn decoder checkpoint not found at {checkpoint} (and no package zip "
                f"next to it). Train one with `python scripts/train.py` as documented in "
                f"docs/windowed_cnn.md."
            )
    else:
        path = _resolve_windowed_checkpoint(bit_length, alpha)
        if path is None:
            want = f"{bit_length} bits at alpha {alpha:.3f}" if bit_length else "the default width"
            raise DecoderUnavailableError(
                f"no windowed-cnn decoder for {want}. Train it with "
                f"`python experiments/train_blind_grid.py --bits {bit_length or 64} --alphas {alpha}` "
                f"(DIV2K + the project embedder) as documented in docs/windowed_cnn.md."
            )
    try:
        return WindowedExtractor.from_checkpoint(str(path), device=device)
    except Exception as exc:
        raise DecoderUnavailableError(
            f"could not load windowed-cnn decoder {path.name}: {exc!r}"
        ) from exc
