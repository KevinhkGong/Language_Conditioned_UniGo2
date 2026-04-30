"""
scripts/eval_stage_d_chunked.py

Load a trained chunked Stage D checkpoint and evaluate on the same val
split used during training. Mirrors ``eval_stage_d.py`` for the
chunked architecture, plus one chunked-specific metric: per-horizon-
offset MSE — the prediction quality at each of the K future steps the
model emits per call. Useful for asking whether longer chunks are
buying anything beyond the first few steps.

Writes ``eval.json`` next to the checkpoint (or to ``--output-dir``)
in the same schema as ``eval_stage_d.py`` plus the new
``per_horizon_per_joint_mse`` (shape K×12) and ``per_horizon_mse``
(shape K, weighted-MSE collapse).
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
    StageDChunkedBundle,
    load_stage_d_chunked_bundle,
)

logger = logging.getLogger("eval_stage_d_chunked")


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
    p.add_argument("--data-dirs",      default="data/real/stage_d_v3",
                   help="Comma-separated.")
    p.add_argument("--format-filter",  default=None, choices=[None, "v2", "v3"])
    p.add_argument("--val-fraction",   type=float, default=0.2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--phases",         default="0,1,2")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--chunk-size",     type=int,   default=None,
                   help="Override the chunk_size; default = checkpoint's value. "
                        "Must match the training-time chunk_size.")
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
             "<checkpoint_dir>/eval.json. Created if it does not exist.",
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

    # ── Load chunked bundle ──────────────────────────────────────────
    ckpt_path = ckpt_dir / "stage_d.pt"
    logger.info(f"Loading chunked checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    chunk_size = int(
        args.chunk_size
        if args.chunk_size is not None
        else ckpt.get("chunk_size", 0))
    if chunk_size <= 1:
        raise ValueError(
            f"This script is for chunked checkpoints. "
            f"Got chunk_size={chunk_size} — use eval_stage_d.py instead.")
    logger.info(f"chunk_size={chunk_size}")

    bundle: StageDChunkedBundle = load_stage_d_chunked_bundle(
        ckpt_path, device=device, chunk_size=chunk_size)
    weights = bundle.joint_weights.detach().cpu().numpy().astype(np.float32)

    saved_val_ids: list[str] = ckpt.get("val_episode_ids", [])
    if saved_val_ids:
        logger.info(f"Checkpoint val episode ids: {saved_val_ids}")

    # ── Rebuild val split with matching chunk_size ───────────────────
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
        chunk_size=chunk_size,
    )
    live_val_ids = sorted({s.episode_id for s in val_ds.samples})
    logger.info(f"Reproduced val episodes ({len(live_val_ids)}): {live_val_ids}")
    if saved_val_ids and set(saved_val_ids) != set(live_val_ids):
        logger.warning(
            "Checkpoint val ids do not match reproduced split "
            "(--val-fraction / --seed / --data-dirs / --chunk-size changed?). "
            "Using reproduced split.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset is empty.")

    # ── Inference loop ───────────────────────────────────────────────
    raw_states = np.stack(
        [s.state_33d for s in val_ds.samples], axis=0).astype(np.float32)
    true_chunks = np.stack(
        [s.action_chunk for s in val_ds.samples], axis=0).astype(np.float32)
    # true_chunks shape: (N, K, 12)
    phase_ids = np.array(
        [s.phase for s in val_ds.samples], dtype=np.int64)
    episode_ids = [s.episode_id for s in val_ds.samples]

    normed = (raw_states - bundle.normalizer.mean) / bundle.normalizer.std
    x = torch.from_numpy(normed.astype(np.float32)).to(device)
    with torch.no_grad():
        # Forward in batches to keep memory bounded for large val sets.
        BATCH = 1024
        preds = []
        for i in range(0, x.shape[0], BATCH):
            preds.append(bundle.policy(x[i:i + BATCH]).cpu())
        pred_chunks = torch.cat(preds, dim=0).numpy().astype(np.float32)
    # pred_chunks shape: (N, K, 12)

    err = pred_chunks - true_chunks            # (N, K, 12)
    abs_err = np.abs(err)
    sq_err = err ** 2

    n_val = pred_chunks.shape[0]

    # ── Per-joint MSE / MAE averaged over batch and K ────────────────
    per_joint_mse = sq_err.mean(axis=(0, 1))   # (12,)
    per_joint_mae = abs_err.mean(axis=(0, 1))  # (12,)

    # ── Per-horizon-offset MSE (chunked-specific) ────────────────────
    # For each k in [0, K), MSE averaged over N samples and 12 joints.
    per_horizon_mse = sq_err.mean(axis=(0, 2))                 # (K,)
    per_horizon_per_joint_mse = sq_err.mean(axis=0)            # (K, 12)
    # Weighted-MSE per horizon (collapses with the same weights used at training).
    weighted_per_horizon_mse = (sq_err * weights).sum(axis=2).mean(axis=0)  # (K,)

    # ── Step-0 only metrics (the "what gets applied at deployment" view) ──
    err0 = err[:, 0, :]                          # (N, 12)
    sq_err0 = err0 ** 2
    abs_err0 = np.abs(err0)
    per_joint_mse_step0 = sq_err0.mean(axis=0)   # (12,)
    per_joint_mae_step0 = abs_err0.mean(axis=0)  # (12,)

    # ── Prediction std per joint (over all N×K samples) ──────────────
    flat_pred = pred_chunks.reshape(-1, STAGE_D_OUTPUT_DIM)   # (N*K, 12)
    flat_true = true_chunks.reshape(-1, STAGE_D_OUTPUT_DIM)
    pred_std  = flat_pred.std(axis=0)
    pred_mean = flat_pred.mean(axis=0)
    pred_min  = flat_pred.min(axis=0)
    pred_max  = flat_pred.max(axis=0)
    true_std  = flat_true.std(axis=0)
    true_mean = flat_true.mean(axis=0)
    true_min  = flat_true.min(axis=0)
    true_max  = flat_true.max(axis=0)

    # ── Per-phase per-joint MSE (keyed on chunk-start phase) ─────────
    # Averaged over (N_phase, K).
    per_phase_per_joint_mse = {}
    n_per_phase = {}
    for ph in (0, 1, 2):
        mask = phase_ids == ph
        n_per_phase[ph] = int(mask.sum())
        if n_per_phase[ph] == 0:
            per_phase_per_joint_mse[ph] = [float("nan")] * STAGE_D_OUTPUT_DIM
        else:
            per_phase_per_joint_mse[ph] = sq_err[mask].mean(axis=(0, 1)).tolist()

    # ── Top-5 worst FR predictions at horizon 0 ──────────────────────
    fr_abs_sum_step0 = abs_err0[:, 0:3].sum(axis=1)
    worst_idx = np.argsort(-fr_abs_sum_step0)[:5]
    worst_records = []
    for rank, idx in enumerate(worst_idx, start=1):
        idx = int(idx)
        worst_records.append({
            "rank":              rank,
            "idx":               idx,
            "episode_id":        episode_ids[idx],
            "phase":             int(phase_ids[idx]),
            "phase_name":        PHASE_NAMES[int(phase_ids[idx])],
            "fr_pred_step0":     pred_chunks[idx, 0, 0:3].tolist(),
            "fr_true_step0":     true_chunks[idx, 0, 0:3].tolist(),
            "fr_abs_err_step0":  abs_err0[idx, 0:3].tolist(),
            "fr_abs_err_sum_step0": float(fr_abs_sum_step0[idx]),
            "foot_to_target_error":
                raw_states[idx, FOOT_ERROR_SLICE].tolist(),
        })

    # ── Print ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"Chunked Stage D eval  |  chunk_size={chunk_size}  |  "
          f"val samples: {n_val}  |  episodes: {len(live_val_ids)}")
    print("=" * 78)

    print("\nTraining-time joint weights (from checkpoint):")
    print(f"  {'joint':<10} {'weight':>7}")
    for j, name in enumerate(JOINT_NAMES):
        marker = "" if weights[j] > 0 else "  ← zero-weighted at training"
        print(f"  {name:<10} {weights[j]:>7.2f}{marker}")

    print("\nPer-joint metrics over full val set (averaged over N and K):")
    print(f"  {'joint':<10} {'weight':>7} {'MSE':>10} {'MAE':>10} "
          f"{'pred_std':>10} {'true_std':>10}")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:<10} {weights[j]:>7.2f} "
              f"{per_joint_mse[j]:>10.5f} {per_joint_mae[j]:>10.5f} "
              f"{pred_std[j]:>10.5f} {true_std[j]:>10.5f}")
    print(f"  {'overall':<10} {'':>7} "
          f"{per_joint_mse.mean():>10.5f} {per_joint_mae.mean():>10.5f}")

    print("\nPer-joint metrics at horizon 0 only (deployment-applied step):")
    print(f"  {'joint':<10} {'MSE':>10} {'MAE':>10}")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:<10} {per_joint_mse_step0[j]:>10.5f} "
              f"{per_joint_mae_step0[j]:>10.5f}")
    print(f"  {'overall':<10} "
          f"{per_joint_mse_step0.mean():>10.5f} "
          f"{per_joint_mae_step0.mean():>10.5f}")

    print("\nPer-horizon MSE (averaged over N and 12 joints):")
    print(f"  {'k':>3} {'unweighted_mse':>14} {'weighted_mse':>14}")
    for k in range(chunk_size):
        marker = ""
        if k == 0:
            marker = "  ← deployment-applied step"
        elif k == chunk_size - 1:
            marker = "  ← last step in chunk"
        print(f"  {k:>3d} {per_horizon_mse[k]:>14.6f} "
              f"{weighted_per_horizon_mse[k]:>14.6f}{marker}")

    # Quick degradation summary: ratio of last-step MSE to first-step MSE.
    if per_horizon_mse[0] > 0:
        ratio = float(per_horizon_mse[-1] / per_horizon_mse[0])
        print(f"  horizon-degradation ratio (mse[K-1] / mse[0]) = {ratio:.2f}×")

    print("\nFR per-horizon MSE (joints 0..2):")
    print(f"  {'k':>3} {'FR_hip':>10} {'FR_thigh':>10} {'FR_calf':>10}")
    for k in range(chunk_size):
        print(f"  {k:>3d} "
              f"{per_horizon_per_joint_mse[k, 0]:>10.5f} "
              f"{per_horizon_per_joint_mse[k, 1]:>10.5f} "
              f"{per_horizon_per_joint_mse[k, 2]:>10.5f}")

    print("\nFR predicted vs true distribution (over N×K samples):")
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

    print("\nPer-phase per-joint MSE (chunk-start phase, averaged over N×K):")
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

    print("\nTop-5 worst FR predictions at horizon 0 "
          "(sorted by |hip|+|thigh|+|calf|):")
    print("-" * 78)
    for r in worst_records:
        print(f"[{r['rank']}] {r['episode_id']}  phase={r['phase_name']} "
              f"(idx={r['idx']})  "
              f"|err|_sum_step0={r['fr_abs_err_sum_step0']:.4f}")
        print(f"    fr_pred_step0:  "
              f"{_fmt_row(np.asarray(r['fr_pred_step0']))}")
        print(f"    fr_true_step0:  "
              f"{_fmt_row(np.asarray(r['fr_true_step0']))}")
        print(f"    fr_|err|_step0: "
              f"{_fmt_row(np.asarray(r['fr_abs_err_step0']))}")
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
        "chunk_size":              chunk_size,
        "joint_names":             JOINT_NAMES,
        "phase_names":             PHASE_NAMES,
        "joint_weights":           weights.tolist(),
        # Aggregated over N and K (the trainer's eval_epoch_chunked metric).
        "per_joint_mse":           per_joint_mse.tolist(),
        "per_joint_mae":           per_joint_mae.tolist(),
        # The "what hits the joints at deployment" view.
        "per_joint_mse_step0":     per_joint_mse_step0.tolist(),
        "per_joint_mae_step0":     per_joint_mae_step0.tolist(),
        # Chunked-specific: prediction quality vs horizon.
        "per_horizon_mse":         per_horizon_mse.tolist(),
        "per_horizon_per_joint_mse": per_horizon_per_joint_mse.tolist(),
        "weighted_per_horizon_mse": weighted_per_horizon_mse.tolist(),
        # Distribution stats (over N×K).
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
        "worst_fr_step0":          worst_records,
        "state_layout":            STATE_LAYOUT,
    }, indent=2))
    print(f"\nSaved → {eval_path}")


if __name__ == "__main__":
    main()
