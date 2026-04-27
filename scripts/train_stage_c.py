"""
scripts/train_stage_c.py

Stage C trainer — fits StandoffMLP (5→3) and WaypointMLP (5→9) on real
episode data using independent AdamW optimizers.

Waypoint labels are present on every sample. Standoff labels are NaN on
~half of episodes (legacy sessions); those rows are masked out of the
standoff loss only.

Checkpoints go to ``--out-dir`` (default ``models/stage_c/``):
    stage_c.pt        PyTorch state_dict for the bundle
    config.json       CLI config + train/val episode id split
    training_log.json per-epoch loss curves (both heads + per-dim wp)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import (
    StageCDataset,
    build_stage_c_datasets,
    split_episodes_train_val,
)
from src.models.stage_c import StageCBundle

logger = logging.getLogger("train_stage_c")


# ──────────────────────────────────────────────────────────────────────
# CLI / config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_dirs:      list[str]
    epochs:         int
    batch_size:     int
    lr:             float
    weight_decay:   float
    val_fraction:   float
    seed:           int
    out_dir:        str
    device:         str
    log_every:      int
    no_save:        bool


def _parse_csv_paths(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
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
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--out-dir",      default="models/stage_c")
    p.add_argument("--device",       default="auto")
    p.add_argument("--log-every",    type=int,   default=10)
    p.add_argument("--no-save",      action="store_true")
    a = p.parse_args()

    if a.data_dir is not None:
        data_dirs = [a.data_dir]
    else:
        data_dirs = _parse_csv_paths(a.data_dirs)

    return TrainConfig(
        data_dirs=data_dirs,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        val_fraction=a.val_fraction,
        seed=a.seed,
        out_dir=a.out_dir,
        device=a.device,
        log_every=a.log_every,
        no_save=a.no_save,
    )


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ──────────────────────────────────────────────────────────────────────
# Collate: default_collate stumbles on the string episode_id field.
# We keep all the tensors stacked and the ids as a plain list.
# ──────────────────────────────────────────────────────────────────────

_TENSOR_KEYS = (
    "input",
    "standoff_offset",
    "lift_waypoint_fr",
    "extend_waypoint_fr",
    "press_waypoint_fr",
)


def stage_c_collate(batch: list[dict]) -> dict:
    out: dict = {}
    for k in _TENSOR_KEYS:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["episode_id"] = [b["episode_id"] for b in batch]
    return out


# ──────────────────────────────────────────────────────────────────────
# Loss helpers
# ──────────────────────────────────────────────────────────────────────

def _waypoint_target(batch: dict) -> torch.Tensor:
    """Stack the three 3-dim waypoints into a (B, 9) tensor."""
    return torch.cat([
        batch["lift_waypoint_fr"],
        batch["extend_waypoint_fr"],
        batch["press_waypoint_fr"],
    ], dim=1)


def _standoff_mask(target: torch.Tensor) -> torch.Tensor:
    """Return a (B,) bool mask of rows with no NaN in the standoff label."""
    return ~torch.isnan(target).any(dim=1)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """MSE averaged over valid rows × all dims. Returns (loss, n_valid)."""
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return pred.new_zeros(()), 0
    diff = pred[mask] - target[mask]
    loss = (diff * diff).mean()
    return loss, n_valid


# ──────────────────────────────────────────────────────────────────────
# Epoch loops
# ──────────────────────────────────────────────────────────────────────

def train_epoch(
    bundle: StageCBundle,
    loader: DataLoader,
    opt_standoff: torch.optim.Optimizer,
    opt_waypoint: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    bundle.train()
    wp_loss_sum = 0.0
    wp_batches = 0
    so_loss_sum = 0.0
    so_valid_total = 0

    for batch in loader:
        x = batch["input"].to(device)

        # Waypoint — always labeled.
        wp_target = _waypoint_target(batch).to(device)
        wp_pred = bundle.waypoint(x)
        wp_loss = nn.functional.mse_loss(wp_pred, wp_target)

        opt_waypoint.zero_grad(set_to_none=True)
        wp_loss.backward()
        opt_waypoint.step()
        wp_loss_sum += float(wp_loss.item())
        wp_batches += 1

        # Standoff — mask NaN rows.
        so_target = batch["standoff_offset"].to(device)
        mask = _standoff_mask(so_target)
        so_pred = bundle.standoff(x)
        so_loss, n_valid = _masked_mse(so_pred, so_target, mask)
        if n_valid > 0:
            opt_standoff.zero_grad(set_to_none=True)
            so_loss.backward()
            opt_standoff.step()
            so_loss_sum += float(so_loss.item()) * n_valid
            so_valid_total += n_valid

    return {
        "wp_loss":      wp_loss_sum / max(1, wp_batches),
        "standoff_loss": so_loss_sum / max(1, so_valid_total),
        "n_valid_standoff": so_valid_total,
    }


@torch.no_grad()
def eval_epoch(
    bundle: StageCBundle,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    bundle.eval()
    wp_loss_sum = 0.0
    wp_batches = 0
    so_loss_sum = 0.0
    so_valid_total = 0
    wp_sqerr_per_dim = torch.zeros(9, dtype=torch.float64, device=device)
    wp_count = 0

    for batch in loader:
        x = batch["input"].to(device)
        wp_target = _waypoint_target(batch).to(device)
        wp_pred = bundle.waypoint(x)
        wp_loss = nn.functional.mse_loss(wp_pred, wp_target)
        wp_loss_sum += float(wp_loss.item())
        wp_batches += 1
        wp_sqerr_per_dim += ((wp_pred - wp_target) ** 2).sum(dim=0).double()
        wp_count += wp_pred.shape[0]

        so_target = batch["standoff_offset"].to(device)
        mask = _standoff_mask(so_target)
        so_pred = bundle.standoff(x)
        so_loss, n_valid = _masked_mse(so_pred, so_target, mask)
        so_loss_sum += float(so_loss.item()) * n_valid
        so_valid_total += n_valid

    per_dim = (wp_sqerr_per_dim / max(1, wp_count)).cpu().numpy().tolist()
    return {
        "wp_loss":      wp_loss_sum / max(1, wp_batches),
        "standoff_loss": so_loss_sum / max(1, so_valid_total),
        "n_valid_standoff": so_valid_total,
        "wp_per_dim_mse": per_dim,   # length 9
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _episode_ids(ds: Dataset) -> list[str]:
    # StageCDataset stores StageCSample objects on .samples
    if hasattr(ds, "samples"):
        return [s.episode_id for s in ds.samples]
    return []


def main() -> None:
    cfg = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Device: {device}")

    # Data ────────────────────────────────────────────────────────────
    data_paths = [Path(p) for p in cfg.data_dirs]
    logger.info(f"Data dirs: {[str(p) for p in data_paths]}")
    train_ds, val_ds = build_stage_c_datasets(
        data_paths,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    logger.info(f"Train episodes: {len(train_ds)}  "
                f"Val episodes: {len(val_ds)}")

    # Record the episode split now so eval can reproduce it exactly
    # (same split function with same seed also works; we store ids for
    # belt-and-suspenders verification).
    train_ids = _episode_ids(train_ds)
    val_ids   = _episode_ids(val_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=stage_c_collate, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=stage_c_collate, drop_last=False)

    # Model ───────────────────────────────────────────────────────────
    bundle = StageCBundle.fresh().to(device)
    logger.info(f"StandoffMLP params: {count_params(bundle.standoff)}")
    logger.info(f"WaypointMLP params: {count_params(bundle.waypoint)}")

    opt_standoff = torch.optim.AdamW(
        bundle.standoff.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_waypoint = torch.optim.AdamW(
        bundle.waypoint.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict] = []

    # Train ───────────────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_epoch(
            bundle, train_loader, opt_standoff, opt_waypoint, device)
        val_stats = eval_epoch(bundle, val_loader, device)

        history.append({
            "epoch":                epoch,
            "train_wp_loss":        train_stats["wp_loss"],
            "train_standoff_loss":  train_stats["standoff_loss"],
            "train_n_valid_standoff": train_stats["n_valid_standoff"],
            "val_wp_loss":          val_stats["wp_loss"],
            "val_standoff_loss":    val_stats["standoff_loss"],
            "val_n_valid_standoff": val_stats["n_valid_standoff"],
            "val_wp_per_dim_mse":   val_stats["wp_per_dim_mse"],
        })

        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            logger.info(
                f"epoch {epoch:03d}/{cfg.epochs} | "
                f"wp train={train_stats['wp_loss']:.4f} "
                f"val={val_stats['wp_loss']:.4f} | "
                f"standoff train={train_stats['standoff_loss']:.5f} "
                f"val={val_stats['standoff_loss']:.5f} "
                f"(n_train={train_stats['n_valid_standoff']} "
                f"n_val={val_stats['n_valid_standoff']})"
            )

    # Final summary ───────────────────────────────────────────────────
    final = history[-1]
    dim_labels = [
        "lift_hip", "lift_thigh", "lift_calf",
        "extend_hip", "extend_thigh", "extend_calf",
        "press_hip", "press_thigh", "press_calf",
    ]
    logger.info("Final val waypoint MSE per output dim:")
    for name, mse in zip(dim_labels, final["val_wp_per_dim_mse"]):
        logger.info(f"  {name:<14} {mse:.5f}")

    # Save ────────────────────────────────────────────────────────────
    if cfg.no_save:
        logger.info("--no-save: skipping checkpoint write.")
        return

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "stage_c.pt"
    torch.save({
        "bundle_state_dict": bundle.state_dict(),
        "config":            asdict(cfg),
        "train_episode_ids": train_ids,
        "val_episode_ids":   val_ids,
    }, ckpt_path)
    logger.info(f"Checkpoint → {ckpt_path}")

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "training_log.json").write_text(
        json.dumps({
            "history":           history,
            "train_episode_ids": train_ids,
            "val_episode_ids":   val_ids,
            "dim_labels":        dim_labels,
        }, indent=2))
    logger.info(f"Config + log → {out_dir}")


if __name__ == "__main__":
    main()
