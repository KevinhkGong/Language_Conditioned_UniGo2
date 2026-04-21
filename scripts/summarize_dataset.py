#!/usr/bin/env python
"""
scripts/summarize_dataset.py

Produces dataset_summary.md from a directory of Stage D HDF5 episodes.
Designed for inclusion in the final project report.

Usage:
    python scripts/summarize_dataset.py data/real/stage_d/ \\
        --output dataset_summary.md

Produces:
    dataset_summary.md                      — main report
    dataset_summary_hist.png (optional)     — standoff histogram (if --png)
    dataset_summary_delta.png (optional)    — delta-vs-perturbation scatter
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Reuse per-episode validation
sys.path.insert(0, str(Path(__file__).parent))
from validate_episode import validate_episode


def _attr_to_str(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _classify(pert_norm: float) -> str:
    if pert_norm <= 1e-3:
        return "stage-c + stage-d no-perturb"
    if pert_norm <= 0.03:
        return "stage-d small-perturb"
    return "stage-d large-perturb"


def _read_episode(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        m = {
            "path": str(path),
            "episode_id":         _attr_to_str(a.get("episode_id", "")),
            "success_fk":         bool(a.get("success_fk", False)),
            "success_audio_live": bool(a.get("success_audio_live", False)),
            "success_target":     bool(a.get("success_target", False)),
            "contact_method":     _attr_to_str(a.get("contact_method", "none")),
            "perturbation":       np.asarray(a.get("perturbation_commanded",
                                                   np.zeros(2)), dtype=np.float32),
            "target_pos_base_at_standoff": np.asarray(
                a.get("target_pos_base_at_standoff", np.zeros(3)), dtype=np.float32),
            "duration_s":         float(a.get("duration_s", 0.0)),
            "k_p":                float(a.get("k_p", 0.0)),
            "button_height_m":    float(a.get("button_height_m", 0.0)),
            "color_detected":     _attr_to_str(a.get("color_detected", "")),
        }
        if "per_step" in f and "jacobian_pid_delta" in f["per_step"]:
            jpd = f["per_step/jacobian_pid_delta"][()]
            mags = np.linalg.norm(jpd, axis=1) if jpd.size else np.array([])
            m["max_delta_rad"] = float(np.max(mags)) if mags.size else 0.0
        else:
            m["max_delta_rad"] = 0.0
    return m


def _bucket(m: dict) -> str:
    return _classify(float(np.linalg.norm(m["perturbation"])))


def _fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate dataset_summary.md.")
    ap.add_argument("directory", type=str,
                    help="Path to data/real/stage_d/")
    ap.add_argument("--output", type=str, default="dataset_summary.md")
    ap.add_argument("--png", action="store_true",
                    help="Also save matplotlib figures alongside the .md")
    args = ap.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory.", file=sys.stderr)
        return 1

    # Gather training episodes (exclude tuning/)
    ep_paths = sorted(
        p for p in root.glob("episode_*.h5")
        if "tuning" not in p.parts
    )
    if not ep_paths:
        print(f"No episode_*.h5 in {root}.")
        return 1

    # Read each episode
    meta = []
    failed_validation: list[Path] = []
    for p in ep_paths:
        try:
            meta.append(_read_episode(p))
        except Exception as e:
            print(f"WARN: could not read {p}: {e}")
            failed_validation.append(p)

        # Run per-episode validation alongside
        res = validate_episode(p)
        if not res.pass_all():
            if p not in failed_validation:
                failed_validation.append(p)

    n = len(meta)
    if n == 0:
        print("No readable episodes.")
        return 1

    # ── Aggregate by bucket ─────────────────────────────────────────────────
    buckets: dict[str, list] = {}
    for m in meta:
        buckets.setdefault(_bucket(m), []).append(m)

    # ── Contact method distribution ─────────────────────────────────────────
    cm_counts: dict[str, int] = {}
    for m in meta:
        cm = m["contact_method"]
        cm_counts[cm] = cm_counts.get(cm, 0) + 1

    # ── Optional plots ──────────────────────────────────────────────────────
    hist_png_path = None
    delta_png_path = None
    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            output_base = Path(args.output).with_suffix("")

            # Histogram of target_pos_base_at_standoff (x, y, z)
            standoffs = np.stack([m["target_pos_base_at_standoff"] for m in meta])
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            for i, label in enumerate(("x", "y", "z")):
                axes[i].hist(standoffs[:, i], bins=20)
                axes[i].set_title(f"target_pos_base_at_standoff — {label}")
                axes[i].set_xlabel("m")
                axes[i].set_ylabel("count")
            fig.tight_layout()
            hist_png_path = Path(str(output_base) + "_hist.png")
            fig.savefig(hist_png_path, dpi=120)
            plt.close(fig)

            # Scatter: |perturbation| vs max |delta|
            fig, ax = plt.subplots(figsize=(6, 4.5))
            pert_norms = np.array(
                [float(np.linalg.norm(m["perturbation"])) for m in meta])
            delta_max = np.array([m["max_delta_rad"] for m in meta])
            ax.scatter(pert_norms * 100, delta_max, s=16, alpha=0.7)
            ax.set_xlabel("|perturbation|  (cm)")
            ax.set_ylabel("max |jacobian_pid_delta|  (rad)")
            ax.set_title("Correction magnitude vs perturbation")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            delta_png_path = Path(str(output_base) + "_delta.png")
            fig.savefig(delta_png_path, dpi=120)
            plt.close(fig)
        except ImportError:
            print("WARN: matplotlib not installed — skipping --png outputs.")

    # ── Write the markdown ──────────────────────────────────────────────────
    out = Path(args.output)
    with out.open("w") as f:
        f.write("# Stage D Dataset Summary\n\n")
        f.write(f"**Directory:** `{root}`  \n")
        f.write(f"**Episodes (excluding tuning/):** {n}  \n")
        f.write("\n")

        f.write("## Three-tier success rates by bucket\n\n")
        f.write("| Bucket | N | success_fk | success_audio_live | success_target |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for bucket, eps in sorted(buckets.items()):
            n_b = len(eps)
            fk = sum(m["success_fk"]         for m in eps) / n_b
            au = sum(m["success_audio_live"] for m in eps) / n_b
            tg = sum(m["success_target"]     for m in eps) / n_b
            f.write(f"| {bucket} | {n_b} | {_fmt_pct(fk)} | "
                    f"{_fmt_pct(au)} | {_fmt_pct(tg)} |\n")
        # Overall row
        fk = sum(m["success_fk"]         for m in meta) / n
        au = sum(m["success_audio_live"] for m in meta) / n
        tg = sum(m["success_target"]     for m in meta) / n
        f.write(f"| **Total** | **{n}** | **{_fmt_pct(fk)}** | "
                f"**{_fmt_pct(au)}** | **{_fmt_pct(tg)}** |\n")
        f.write("\n")

        f.write("## contact_method distribution\n\n")
        f.write("| method | count | fraction |\n")
        f.write("|---|---:|---:|\n")
        for cm, c in sorted(cm_counts.items(), key=lambda kv: -kv[1]):
            f.write(f"| {cm} | {c} | {_fmt_pct(c / n)} |\n")
        f.write("\n")

        f.write("## target_pos_base_at_standoff range\n\n")
        standoffs = np.stack([m["target_pos_base_at_standoff"] for m in meta])
        f.write(
            f"| axis | min | median | max |\n"
            f"|---|---:|---:|---:|\n"
            f"| x (m) | {standoffs[:,0].min():+.3f} | "
            f"{np.median(standoffs[:,0]):+.3f} | "
            f"{standoffs[:,0].max():+.3f} |\n"
            f"| y (m) | {standoffs[:,1].min():+.3f} | "
            f"{np.median(standoffs[:,1]):+.3f} | "
            f"{standoffs[:,1].max():+.3f} |\n"
            f"| z (m) | {standoffs[:,2].min():+.3f} | "
            f"{np.median(standoffs[:,2]):+.3f} | "
            f"{standoffs[:,2].max():+.3f} |\n"
        )
        f.write("\n")

        f.write("## K_p values used\n\n")
        kps = sorted({round(m["k_p"], 3) for m in meta})
        f.write(", ".join(f"{k}" for k in kps) + "\n\n")

        if hist_png_path or delta_png_path:
            f.write("## Figures\n\n")
            if hist_png_path:
                f.write(f"![standoff histogram]({hist_png_path.name})\n\n")
            if delta_png_path:
                f.write(f"![correction vs perturbation]({delta_png_path.name})\n\n")

        if failed_validation:
            f.write("## Episodes failing per-episode validation\n\n")
            for p in failed_validation:
                f.write(f"- `{p.name}`\n")
            f.write("\n")
            f.write("Run `python scripts/validate_episode.py <path>` for details.\n\n")

        f.write("---\n")
        f.write("_Generated by `scripts/summarize_dataset.py`._\n")

    print(f"Wrote {out}")
    if hist_png_path:
        print(f"Wrote {hist_png_path}")
    if delta_png_path:
        print(f"Wrote {delta_png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())