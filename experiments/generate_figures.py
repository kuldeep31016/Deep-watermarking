"""Generate figures and tables from experiment output - never from typed-in numbers.

Reads::

    results/matrix/per_image.csv       (experiments/run_matrix.py)
    results/robustness/per_image.csv   (experiments/run_robustness.py, optional)

Writes ``results/figures/``::

    psnr_vs_length.png            PSNR vs watermark length, one line per alpha   (paper Fig. 3)
    ssim_vs_length.png            SSIM vs watermark length, one line per alpha   (paper Fig. 3)
    bit_accuracy_vs_length.png    blind + non-blind accuracy vs length per alpha (paper Fig. 4)
    ber_vs_length.png             blind + non-blind BER vs length per alpha
    alpha_comparison.png          PSNR / SSIM / blind accuracy grouped by alpha
    per_image_<bits>bit_a<alpha>.png   PSNR / SSIM / accuracy per test image      (paper Fig. 5)
    robustness_<metric>.png       BER vs attack severity per attack family (if available)

and ``results/tables/`` with Markdown versions of the aggregated tables.

Usage::

    python experiments/generate_figures.py
    python experiments/generate_figures.py --per-image-point 128 0.01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "results"


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _line_vs_length(df: pd.DataFrame, metric: str, ylabel: str, out: Path, *, second: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for alpha, sub in sorted(df.groupby("alpha")):
        agg = sub.groupby("bits")[metric].mean().dropna()
        if agg.empty:
            continue
        ax.plot(agg.index, agg.values, marker="o", label=f"{ylabel.split(' ')[0]} α={alpha:.3f}")
        if second and second in sub.columns:
            agg2 = sub.groupby("bits")[second].mean().dropna()
            if not agg2.empty:
                ax.plot(agg2.index, agg2.values, marker="s", linestyle="--", label=f"{second.replace('_', ' ')} α={alpha:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["bits"].unique()))
    ax.set_xticklabels([str(b) for b in sorted(df["bits"].unique())])
    ax.set_xlabel("watermark length (bits)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def matrix_figures(per_image: Path, fig_dir: Path, tab_dir: Path, per_image_point: tuple[int, float]) -> None:
    df = pd.read_csv(per_image)
    df = _numeric(df, ["psnr", "ssim", "mse", "nonblind_ber", "nonblind_bit_accuracy", "blind_ber", "blind_bit_accuracy"])

    _line_vs_length(df, "psnr", "PSNR (dB)", fig_dir / "psnr_vs_length.png")
    _line_vs_length(df, "ssim", "SSIM", fig_dir / "ssim_vs_length.png")
    _line_vs_length(df, "blind_bit_accuracy", "blind accuracy", fig_dir / "bit_accuracy_vs_length.png", second="nonblind_bit_accuracy")
    _line_vs_length(df, "blind_ber", "blind BER", fig_dir / "ber_vs_length.png", second="nonblind_ber")

    # alpha comparison: grouped bars, mean over lengths and images
    agg = df.groupby("alpha")[["psnr", "ssim", "blind_bit_accuracy", "nonblind_bit_accuracy"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (col, label) in zip(axes, [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("blind_bit_accuracy", "blind accuracy")]):
        ax.bar([f"{a:.3f}" for a in agg.index], agg[col].values, color="#3b82f6")
        ax.set_title(label)
        ax.set_xlabel("alpha")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "alpha_comparison.png", dpi=150)
    plt.close(fig)

    # per-image plots at one point (paper Fig. 5)
    bits, alpha = per_image_point
    sub = df[(df["bits"] == bits) & (df["alpha"].sub(alpha).abs() < 1e-9)].reset_index(drop=True)
    if not sub.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        x = range(len(sub))
        axes[0].plot(x, sub["psnr"], color="#3b82f6"); axes[0].set_title(f"PSNR per image ({bits} bits, α={alpha})")
        axes[1].plot(x, sub["ssim"], color="#ef4444"); axes[1].set_title("SSIM per image")
        acc_col = "blind_bit_accuracy" if sub["blind_bit_accuracy"].notna().any() else "nonblind_bit_accuracy"
        axes[2].plot(x, sub[acc_col], color="#22c55e"); axes[2].set_title(f"{acc_col.replace('_', ' ')} per image")
        for ax in axes:
            ax.set_xlabel("image index")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"per_image_{bits}bit_a{alpha:.3f}.png", dpi=150)
        plt.close(fig)

    table = (
        df.groupby(["bits", "alpha"])[["psnr", "ssim", "nonblind_bit_accuracy", "blind_bit_accuracy", "blind_ber"]]
        .mean()
        .reset_index()
    )
    table.to_markdown(tab_dir / "matrix_summary.md", index=False, floatfmt=".4f")
    print(f"[figures] matrix: {len(df)} rows -> {fig_dir}")


def robustness_figures(per_image: Path, fig_dir: Path, tab_dir: Path) -> None:
    df = pd.read_csv(per_image)
    df = _numeric(df, ["severity", "psnr_attacked", "blind_ber", "nonblind_ber", "blind_bit_accuracy"])
    attacks = [a for a in df["attack"].unique() if a != "identity"]
    fig, axes = plt.subplots(2, (len(attacks) + 1) // 2, figsize=(4 * ((len(attacks) + 1) // 2), 6.5))
    axes = list(axes.flat)
    for ax, attack in zip(axes, attacks):
        sub = df[df["attack"] == attack]
        agg = sub.groupby("severity")[["blind_ber", "nonblind_ber"]].mean()
        ax.plot(agg.index, agg["blind_ber"], marker="o", label="blind (windowed CNN)")
        ax.plot(agg.index, agg["nonblind_ber"], marker="s", linestyle="--", label="non-blind")
        ax.axhline(0.5, color="grey", linewidth=0.8, linestyle=":")
        ax.set_title(attack)
        ax.set_ylabel("BER")
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(attacks):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "robustness_ber.png", dpi=150)
    plt.close(fig)
    table = df.groupby(["attack", "severity"])[["psnr_attacked", "blind_ber", "blind_bit_accuracy", "nonblind_ber"]].mean().reset_index()
    table.to_markdown(tab_dir / "robustness_summary.md", index=False, floatfmt=".4f")
    print(f"[figures] robustness: {len(df)} rows -> {fig_dir / 'robustness_ber.png'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--per-image-point", nargs=2, type=float, default=(128, 0.01), metavar=("BITS", "ALPHA"))
    args = parser.parse_args(argv)

    fig_dir = args.results / "figures"
    tab_dir = args.results / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = args.results / "matrix" / "per_image.csv"
    if matrix_csv.is_file():
        matrix_figures(matrix_csv, fig_dir, tab_dir, (int(args.per_image_point[0]), float(args.per_image_point[1])))
    else:
        print(f"[figures] no matrix results at {matrix_csv} (run experiments/run_matrix.py)")

    rob_csv = args.results / "robustness" / "per_image.csv"
    if rob_csv.is_file():
        robustness_figures(rob_csv, fig_dir, tab_dir)
    else:
        print(f"[figures] no robustness results at {rob_csv} (run experiments/run_robustness.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
