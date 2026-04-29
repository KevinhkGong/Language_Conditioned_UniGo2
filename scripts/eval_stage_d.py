"""
scripts/eval_stage_d.py

Load a trained Stage D checkpoint and evaluate on the same val split
used during training. Prints per-joint MSE/MAE, per-phase per-joint
MSE, prediction std per joint (used to confirm zero-weighted joints
learned nothing constructive), the predicted/true delta distribution
on the FR joints, and the top-5 worst FR-error samples. Writes the
full set of metrics to ``<checkpoint_dir>/eval.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import (
    STATE_LAYOUT,
    build_stage_d_datasets,
)
from src.models.stage_d import (
    STAGE_D_OUTPUT_DIM,
    StageDBundle,
    load_stage_d_bundle,
)

logger = logging.getLogger("eval_stage_d")


JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]
PHASE_NAMES = {0: "lift", 1: "extend", 2: "hold"}

# ``foot_to_target_error`` lives at indices [6, 9) in the 33-dim state.
FOOT_ERROR_SLICE = slice(6, 9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="models/stage_d")
    p.add_argument("--data-dirs",      default="data/real/stage_d_v2",
                   help="Comma-separated.")
    p.add_argument("--format-filter",  default=None, choices=[None, "v2", "v3"])
    p.add_argument("--val-fraction",   type=float, default=0.2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--phases",         default="0,1,2")
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


def _csv_paths(s: str) -> list[Path]:
    return [Path(p.strip()) for p in s.split(",") if p.strip()]


def _csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _fmt_row(vals: np.ndarray, width: int = 10, prec: int = 5) -> str:
    return " ".join(f"{float(v):>{width}.{prec}f}" for v in vals)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device)

    # ── Load bundle ──────────────────────────────────────────────────
    ckpt_path = ckpt_dir / "stage_d.pt"
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    bundle: StageDBundle = load_stage_d_bundle(ckpt_path, device=device)
    weights = bundle.joint_weights.detach().cpu().numpy().astype(np.float32)

    saved_val_ids: list[str] = ckpt.get("val_episode_ids", [])
    if saved_val_ids:
        logger.info(f"Checkpoint val episode ids: {saved_val_ids}")

    # ── Rebuild val split ────────────────────────────────────────────
    data_paths = _csv_paths(args.data_dirs)
    phases = _csv_ints(args.phases)
    if args.gain_schedule_filter is not None:
        logger.info(f"gain_schedule_filter: {args.gain_schedule_filter!r}")
    if args.intrinsics_version_filter is not None:
        logger.info(
            f"intrinsics_version_filter: {args.intrinsics_version_filter!r}")
    _, val_ds = build_stage_d_datasets(
        data_paths,
        val_fraction=args.val_fraction,
        seed=args.seed,
        phases=phases,
        format_filter=args.format_filter,
        gain_schedule_filter=args.gain_schedule_filter,
        intrinsics_version_filter=args.intrinsics_version_filter,
    )
    live_val_ids = sorted({s.episode_id for s in val_ds.samples})
    logger.info(f"Reproduced val episodes ({len(live_val_ids)}): {live_val_ids}")
    if saved_val_ids and set(saved_val_ids) != set(live_val_ids):
        logger.warning(
            "Checkpoint val ids do not match reproduced split "
            "(--val-fraction / --seed / --data-dirs changed?). "
            "Using reproduced split.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset is empty.")

    # ── Inference loop ───────────────────────────────────────────────
    raw_states = np.stack(
        [s.state_33d for s in val_ds.samples], axis=0).astype(np.float32)
    true_q = np.stack(
        [s.achieved_delta_q for s in val_ds.samples], axis=0).astype(np.float32)
    phase_ids = np.array(
        [s.phase for s in val_ds.samples], dtype=np.int64)
    episode_ids = [s.episode_id for s in val_ds.samples]

    normed = (raw_states - bundle.normalizer.mean) / bundle.normalizer.std
    x = torch.from_numpy(normed.astype(np.float32)).to(device)
    with torch.no_grad():
        pred_q = bundle.policy(x).cpu().numpy().astype(np.float32)

    err = pred_q - true_q                       # (N, 12)
    abs_err = np.abs(err)
    sq_err = err ** 2

    n_val = pred_q.shape[0]

    # ── Per-joint MSE / MAE ──────────────────────────────────────────
    per_joint_mse = sq_err.mean(axis=0)
    per_joint_mae = abs_err.mean(axis=0)

    # ── Prediction std per joint ─────────────────────────────────────
    pred_std  = pred_q.std(axis=0)
    pred_mean = pred_q.mean(axis=0)
    pred_min  = pred_q.min(axis=0)
    pred_max  = pred_q.max(axis=0)

    true_std  = true_q.std(axis=0)
    true_mean = true_q.mean(axis=0)
    true_min  = true_q.min(axis=0)
    true_max  = true_q.max(axis=0)

    # ── Per-phase per-joint MSE ──────────────────────────────────────
    per_phase_per_joint_mse = {}
    n_per_phase = {}
    for ph in (0, 1, 2):
        mask = phase_ids == ph
        n_per_phase[ph] = int(mask.sum())
        if n_per_phase[ph] == 0:
            per_phase_per_joint_mse[ph] = [float("nan")] * STAGE_D_OUTPUT_DIM
        else:
            per_phase_per_joint_mse[ph] = sq_err[mask].mean(axis=0).tolist()

    # ── Top-5 worst FR predictions ───────────────────────────────────
    fr_abs_sum = abs_err[:, 0:3].sum(axis=1)
    worst_idx = np.argsort(-fr_abs_sum)[:5]
    worst_records = []
    for rank, idx in enumerate(worst_idx, start=1):
        idx = int(idx)
        worst_records.append({
            "rank":          rank,
            "idx":           idx,
            "episode_id":    episode_ids[idx],
            "phase":         int(phase_ids[idx]),
            "phase_name":    PHASE_NAMES[int(phase_ids[idx])],
            "fr_pred":       pred_q[idx, 0:3].tolist(),
            "fr_true":       true_q[idx, 0:3].tolist(),
            "fr_abs_err":    abs_err[idx, 0:3].tolist(),
            "fr_abs_err_sum": float(fr_abs_sum[idx]),
            "foot_to_target_error":
                raw_states[idx, FOOT_ERROR_SLICE].tolist(),
        })

    # ── Print ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"Stage D eval  |  val samples: {n_val}  |  "
          f"episodes: {len(live_val_ids)}")
    print("=" * 78)

    print("\nTraining-time joint weights (from checkpoint):")
    print(f"  {'joint':<10} {'weight':>7}")
    for j, name in enumerate(JOINT_NAMES):
        marker = "" if weights[j] > 0 else "  ← zero-weighted at training"
        print(f"  {name:<10} {weights[j]:>7.2f}{marker}")

    print("\nPer-joint metrics over full val set:")
    print(f"  {'joint':<10} {'weight':>7} {'MSE':>10} {'MAE':>10} "
          f"{'pred_std':>10} {'true_std':>10}")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:<10} {weights[j]:>7.2f} "
              f"{per_joint_mse[j]:>10.5f} {per_joint_mae[j]:>10.5f} "
              f"{pred_std[j]:>10.5f} {true_std[j]:>10.5f}")
    print(f"  {'overall':<10} {'':>7} "
          f"{per_joint_mse.mean():>10.5f} {per_joint_mae.mean():>10.5f}")

    print("\nEffective vs prescribed weighting check:")
    for j, name in enumerate(JOINT_NAMES):
        w = float(weights[j])
        psd = float(pred_std[j])
        if w == 0.0:
            verdict = ("OK (no learning)" if psd < 1e-3
                       else "UNEXPECTED — non-trivial std despite weight=0")
        else:
            verdict = ("OK (learned)" if psd > 1e-4
                       else "WARN — weighted but pred_std≈0")
        print(f"  {name:<10} weight={w:.2f}  pred_std={psd:.5f}  → {verdict}")

    print("\nFR predicted vs true distribution:")
    print(f"  {'joint':<10} {'pred_mean':>10} {'pred_std':>10} "
          f"{'pred_min':>10} {'pred_max':>10}")
    for j in range(3):
        print(f"  {JOINT_NAMES[j]:<10} "
              f"{pred_mean[j]:>+10.5f} {pred_std[j]:>10.5f} "
              f"{pred_min[j]:>+10.5f} {pred_max[j]:>+10.5f}")
    print(f"  {'joint':<10} {'true_mean':>10} {'true_std':>10} "
          f"{'true_min':>10} {'true_max':>10}")
    for j in range(3):
        print(f"  {JOINT_NAMES[j]:<10} "
              f"{true_mean[j]:>+10.5f} {true_std[j]:>10.5f} "
              f"{true_min[j]:>+10.5f} {true_max[j]:>+10.5f}")

    print("\nPer-phase per-joint MSE (val):")
    header = f"  {'joint':<10}" + "".join(
        f" {PHASE_NAMES[ph]:>10}" for ph in (0, 1, 2))
    print(header)
    for j, name in enumerate(JOINT_NAMES):
        row = "  " + f"{name:<10}"
        for ph in (0, 1, 2):
            v = per_phase_per_joint_mse[ph][j]
            row += f" {v:>10.5f}"
        print(row)
    print(f"  n_per_phase: {n_per_phase}")

    print("\nTop-5 worst FR predictions (sorted by |hip|+|thigh|+|calf|):")
    print("-" * 78)
    for r in worst_records:
        print(f"[{r['rank']}] {r['episode_id']}  phase={r['phase_name']} "
              f"(idx={r['idx']})  |err|_sum={r['fr_abs_err_sum']:.4f}")
        print(f"    fr_pred:        {_fmt_row(np.asarray(r['fr_pred']))}")
        print(f"    fr_true:        {_fmt_row(np.asarray(r['fr_true']))}")
        print(f"    fr_|err|:       {_fmt_row(np.asarray(r['fr_abs_err']))}")
        print(f"    foot_to_target: "
              f"{_fmt_row(np.asarray(r['foot_to_target_error']))}")

    # ── Save ─────────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ckpt_dir
    eval_path = out_dir / "eval.json"
    eval_path.write_text(json.dumps({
        "val_samples":             n_val,
        "val_episode_ids":         live_val_ids,
        "joint_names":             JOINT_NAMES,
        "phase_names":             PHASE_NAMES,
        "joint_weights":           weights.tolist(),
        "per_joint_mse":           per_joint_mse.tolist(),
        "per_joint_mae":           per_joint_mae.tolist(),
        "pred_mean":               pred_mean.tolist(),
        "pred_std":                pred_std.tolist(),
        "pred_min":                pred_min.tolist(),
        "pred_max":                pred_max.tolist(),
        "true_mean":               true_mean.tolist(),
        "true_std":                true_std.tolist(),
        "true_min":                true_min.tolist(),
        "true_max":                true_max.tolist(),
        "per_phase_per_joint_mse": {str(k): v for k, v in
                                    per_phase_per_joint_mse.items()},
        "n_per_phase":             {str(k): v for k, v in n_per_phase.items()},
        "worst_fr":                worst_records,
        "state_layout":            STATE_LAYOUT,
    }, indent=2))
    print(f"\nSaved → {eval_path}")


if __name__ == "__main__":
    main()
