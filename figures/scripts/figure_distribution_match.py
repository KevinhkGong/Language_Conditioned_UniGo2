#!/usr/bin/env python3
"""
Generate the Stage D distribution-match validation figure.

For each of the 12 joints of the Stage D residual policy, plot the
predicted standard deviation alongside the ground-truth standard deviation
on the held-out validation set. Matched bar heights at every joint
demonstrate that the policy produces state-dependent corrections that
capture the per-joint variance of the demonstration distribution, rather
than collapsing to per-joint means (the "no mode collapse" claim from
the report's Section IV-D1).

Inputs are the eval JSON files written by the Stage D evaluation pipeline.

Usage:
    python figure_distribution_match.py \
        --v5 eval_d_v5.json \
        --out figures/stage_d_distribution_match.pdf

    # Overlay v6 chunked as a third bar per joint:
    python figure_distribution_match.py \
        --v5 eval_d_v5.json \
        --v6 eval_d_v6_chunked.json \
        --include-v6 \
        --out figures/stage_d_distribution_match.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Paper-style matplotlib config. Falls back to system serif if Computer
# Modern is unavailable; enable text.usetex once your build environment
# has LaTeX installed for full font matching with the report body.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Neutral academic palette. Ground truth in light gray (visually recedes),
# predictions in deep navy / teal so the eye reads "does pred match truth?"
COLOR_TRUE = "#bdc3c7"
COLOR_V5 = "#2c3e50"
COLOR_V6 = "#16a085"


def load_eval(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_figure(eval_v5: dict, eval_v6: dict | None = None) -> plt.Figure:
    joint_names = eval_v5["joint_names"]
    pred_v5 = np.array(eval_v5["pred_std"])
    true_v5 = np.array(eval_v5["true_std"])
    n = len(joint_names)
    has_v6 = eval_v6 is not None

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    x = np.arange(n)

    if has_v6:
        pred_v6 = np.array(eval_v6["pred_std"])
        width = 0.27
        ax.bar(x - width, true_v5, width,
               color=COLOR_TRUE, edgecolor="black", linewidth=0.4,
               label="Ground truth", zorder=3)
        ax.bar(x, pred_v5, width,
               color=COLOR_V5, edgecolor="black", linewidth=0.4,
               label="Predicted (v5)", zorder=3)
        ax.bar(x + width, pred_v6, width,
               color=COLOR_V6, edgecolor="black", linewidth=0.4,
               label="Predicted (v6 chunked)", zorder=3)
    else:
        width = 0.4
        ax.bar(x - width / 2, true_v5, width,
               color=COLOR_TRUE, edgecolor="black", linewidth=0.4,
               label="Ground truth", zorder=3)
        ax.bar(x + width / 2, pred_v5, width,
               color=COLOR_V5, edgecolor="black", linewidth=0.4,
               label="Predicted (v5)", zorder=3)

    # Subtle vertical separators between legs to make the four leg groups
    # readable even at column width.
    for leg_boundary in [2.5, 5.5, 8.5]:
        ax.axvline(leg_boundary, color="lightgray", linewidth=0.5,
                   linestyle=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=45, ha="right")
    ax.set_ylabel("Joint residual std (rad)")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.5,
              borderpad=0.3)

    return fig


def print_match_summary(eval_v5: dict) -> None:
    joint_names = eval_v5["joint_names"]
    pred = np.array(eval_v5["pred_std"])
    true = np.array(eval_v5["true_std"])
    ratios = pred / true
    is_fr = np.array([n.startswith("FR_") for n in joint_names])

    print("Distribution match summary (Stage D v5):")
    print(f"  All joints       ratio range = "
          f"[{ratios.min():.3f}, {ratios.max():.3f}]   "
          f"mean = {ratios.mean():.3f}")
    print(f"  FR press leg     ratio range = "
          f"[{ratios[is_fr].min():.3f}, {ratios[is_fr].max():.3f}]")
    print(f"  Support legs     ratio range = "
          f"[{ratios[~is_fr].min():.3f}, {ratios[~is_fr].max():.3f}]")
    print()
    print("Per-joint detail:")
    for name, p, t in zip(joint_names, pred, true):
        print(f"  {name:<10}  pred_std = {p:.4f}  true_std = {t:.4f}  "
              f"ratio = {p / t:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Stage D distribution-match validation figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--v5", type=Path, default=Path("eval_d_v5.json"),
        help="Path to Stage D v5 eval JSON.",
    )
    parser.add_argument(
        "--v6", type=Path, default=Path("eval_d_v6_chunked.json"),
        help="Path to Stage D v6 chunked eval JSON (used with --include-v6).",
    )
    parser.add_argument(
        "--include-v6", action="store_true",
        help="Add a third bar per joint for v6 chunked predictions.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("figures/stage_d_distribution_match.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    eval_v5 = load_eval(args.v5)
    eval_v6 = load_eval(args.v6) if args.include_v6 else None

    fig = build_figure(eval_v5, eval_v6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf")
    plt.close(fig)
    print(f"Saved figure: {args.out}\n")
    print_match_summary(eval_v5)


if __name__ == "__main__":
    main()