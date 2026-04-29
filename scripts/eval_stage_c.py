"""
scripts/eval_stage_c.py

Load a trained Stage C checkpoint and evaluate against the same val
split used during training. Prints per-dim MSE / MAE tables for the
waypoint head, per-dim metrics on the labeled subset of the standoff
head, and a full predictions table for manual inspection. Also writes
a JSON of per-sample results to ``<checkpoint_dir>/eval.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import build_stage_c_datasets
from src.models.stage_c import StageCBundle

logger = logging.getLogger("eval_stage_c")


DIM_LABELS_WP = [
    "lift_hip", "lift_thigh", "lift_calf",
    "extend_hip", "extend_thigh", "extend_calf",
    "press_hip", "press_thigh", "press_calf",
]
DIM_LABELS_SO = ["dx", "dy", "dyaw"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="models/stage_c")
    p.add_argument(
        "--data-dirs",
        type=str,
        default="data/real/stage_d_v2,data/real/stage_d_v3",
        help="Comma-separated list of episode directories (e.g., 'dir1,dir2').",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="DEPRECATED: use --data-dirs instead. Backward-compatible alias.",
    )
    p.add_argument("--val-fraction",   type=float, default=0.2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cpu")
    p.add_argument(
        "--gain-schedule-filter",
        type=str,
        default=None,
        help="If set, only load episodes where root attr 'gain_schedule' "
             "matches this exact string. Default: no filter.",
    )
    p.add_argument(
        "--intrinsics-version-filter",
        type=str,
        default=None,
        help="If set, only load episodes where root attr "
             "'camera_intrinsics_version' matches this exact string. "
             "Default: no filter.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write eval.json. If unset, writes to "
             "<checkpoint_dir>/eval.json (existing behaviour). Created "
             "if it does not exist.",
    )
    return p.parse_args()


def _fmt_row(vals: list[float], width: int = 8, prec: int = 4) -> str:
    return " ".join(f"{v:>{width}.{prec}f}" for v in vals)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device)

    # Load checkpoint ─────────────────────────────────────────────────
    ckpt_path = ckpt_dir / "stage_c.pt"
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    bundle = StageCBundle.fresh().to(device)
    bundle.load_state_dict(ckpt["bundle_state_dict"])
    bundle.eval()

    saved_val_ids: list[str] = ckpt.get("val_episode_ids", [])
    if saved_val_ids:
        logger.info(f"Checkpoint val episode ids: {saved_val_ids}")

    # Rebuild split ───────────────────────────────────────────────────
    if args.data_dir is not None:
        data_paths = [Path(args.data_dir)]
    else:
        data_paths = [Path(p.strip()) for p in args.data_dirs.split(",")
                      if p.strip()]
    logger.info(f"Data dirs: {[str(p) for p in data_paths]}")
    if args.gain_schedule_filter is not None:
        logger.info(f"gain_schedule_filter: {args.gain_schedule_filter!r}")
    if args.intrinsics_version_filter is not None:
        logger.info(
            f"intrinsics_version_filter: {args.intrinsics_version_filter!r}")
    _, val_ds = build_stage_c_datasets(
        data_paths,
        val_fraction=args.val_fraction,
        seed=args.seed,
        gain_schedule_filter=args.gain_schedule_filter,
        intrinsics_version_filter=args.intrinsics_version_filter,
    )
    live_val_ids = [s.episode_id for s in val_ds.samples]
    logger.info(f"Reproduced val episodes: {live_val_ids}")
    if saved_val_ids and set(saved_val_ids) != set(live_val_ids):
        logger.warning(
            "Checkpoint val ids do not match reproduced split "
            "(--val-fraction / --seed changed?). Using reproduced split.")

    # Inference ───────────────────────────────────────────────────────
    per_sample: list[dict] = []
    wp_sq_errs: list[np.ndarray] = []   # (9,)
    wp_abs_errs: list[np.ndarray] = []  # (9,)
    so_sq_errs: list[np.ndarray] = []   # (3,)  — labeled subset only
    so_abs_errs: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            x = sample["input"].unsqueeze(0).to(device)

            wp_pred_t = bundle.waypoint(x).squeeze(0).cpu().numpy()
            so_pred_t = bundle.standoff(x).squeeze(0).cpu().numpy()

            wp_true = torch.cat([
                sample["lift_waypoint_fr"],
                sample["extend_waypoint_fr"],
                sample["press_waypoint_fr"],
            ]).numpy()
            so_true = sample["standoff_offset"].numpy()

            wp_diff = wp_pred_t - wp_true
            wp_sq_errs.append(wp_diff ** 2)
            wp_abs_errs.append(np.abs(wp_diff))

            so_labeled = not np.isnan(so_true).any()
            if so_labeled:
                so_diff = so_pred_t - so_true
                so_sq_errs.append(so_diff ** 2)
                so_abs_errs.append(np.abs(so_diff))

            per_sample.append({
                "episode_id": sample["episode_id"],
                "input":      sample["input"].numpy().tolist(),
                "wp_pred":    wp_pred_t.tolist(),
                "wp_true":    wp_true.tolist(),
                "wp_abs_err": np.abs(wp_diff).tolist(),
                "so_pred":    so_pred_t.tolist(),
                "so_true":    [None if np.isnan(v) else float(v)
                               for v in so_true.tolist()],
                "so_labeled": bool(so_labeled),
            })

    # Aggregate ───────────────────────────────────────────────────────
    wp_mse = np.mean(np.stack(wp_sq_errs, axis=0), axis=0) \
        if wp_sq_errs else np.zeros(9)
    wp_mae = np.mean(np.stack(wp_abs_errs, axis=0), axis=0) \
        if wp_abs_errs else np.zeros(9)

    if so_sq_errs:
        so_mse = np.mean(np.stack(so_sq_errs, axis=0), axis=0)
        so_mae = np.mean(np.stack(so_abs_errs, axis=0), axis=0)
    else:
        so_mse = np.array([np.nan, np.nan, np.nan])
        so_mae = np.array([np.nan, np.nan, np.nan])

    # Print ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"Stage C eval  |  val samples: {len(val_ds)}  "
          f"|  standoff-labeled: {len(so_sq_errs)}")
    print("=" * 78)

    print("\nWaypoint head (9 dims) — per-dim over full val set:")
    print(f"  {'dim':<14} {'MSE':>10} {'MAE':>10}")
    for label, m, a in zip(DIM_LABELS_WP, wp_mse, wp_mae):
        print(f"  {label:<14} {m:>10.5f} {a:>10.5f}")
    print(f"  {'overall':<14} {wp_mse.mean():>10.5f} {wp_mae.mean():>10.5f}")

    print("\nStandoff head (3 dims) — labeled subset only:")
    print(f"  {'dim':<14} {'MSE':>10} {'MAE':>10}")
    for label, m, a in zip(DIM_LABELS_SO, so_mse, so_mae):
        m_str = "     nan" if np.isnan(m) else f"{m:>10.5f}"
        a_str = "     nan" if np.isnan(a) else f"{a:>10.5f}"
        print(f"  {label:<14} {m_str} {a_str}")

    # Per-sample predictions ──────────────────────────────────────────
    print("\nPer-sample val predictions:")
    print("-" * 78)
    for r in per_sample:
        print(f"[{r['episode_id']}]  target_pos_base="
              f"{[round(v,4) for v in r['input'][:3]]}")
        print(f"  wp_pred:  {_fmt_row(r['wp_pred'])}")
        print(f"  wp_true:  {_fmt_row(r['wp_true'])}")
        print(f"  wp_|err|: {_fmt_row(r['wp_abs_err'])}")
        so_true_str = (
            "[unlabeled]" if not r["so_labeled"]
            else f"[{', '.join(f'{v:+.4f}' for v in r['so_true'])}]")
        so_pred_str = ", ".join(f"{v:+.4f}" for v in r["so_pred"])
        print(f"  so_pred:  [{so_pred_str}]   so_true: {so_true_str}")
        print()

    # Save ────────────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ckpt_dir
    eval_path = out_dir / "eval.json"
    eval_path.write_text(json.dumps({
        "val_samples":         len(val_ds),
        "standoff_labeled_n":  len(so_sq_errs),
        "wp_mse_per_dim":      wp_mse.tolist(),
        "wp_mae_per_dim":      wp_mae.tolist(),
        "wp_mse_overall":      float(wp_mse.mean()),
        "wp_mae_overall":      float(wp_mae.mean()),
        "so_mse_per_dim":      [None if np.isnan(v) else float(v) for v in so_mse],
        "so_mae_per_dim":      [None if np.isnan(v) else float(v) for v in so_mae],
        "per_sample":          per_sample,
        "dim_labels_wp":       DIM_LABELS_WP,
        "dim_labels_so":       DIM_LABELS_SO,
    }, indent=2))
    print(f"Saved → {eval_path}")


if __name__ == "__main__":
    main()
