"""
scripts/train_stage_d.py

Stage D trainer — fits the 33→12 residual policy with a per-joint
weighted MSE loss. The default joint weighting is FR-only
([5,5,5,0,0,0,0,0,0,0,0,0]) since v2 episodes carry no useful
support-leg demonstrations; pass ``--joint-weights`` to switch to a
whole-body schedule once v3 data is available.

Checkpoints go to ``--out-dir`` (default ``models/stage_d/``):
    stage_d.pt        bundle_state_dict + config + train/val episode ids
    config.json       CLI config
    training_log.json per-epoch loss curves and per-joint MSE history
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import (
    StageDDataset,
    StateNormalizer,
    STATE_DIM,
    STATE_LAYOUT,
    build_stage_d_datasets,
)
from src.models.stage_d import (
    STAGE_D_OUTPUT_DIM,
    StageDBundle,
    StageDChunkedBundle,
)

logger = logging.getLogger("train_stage_d")


JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]
LEG_GROUPS = {
    "FR": (0, 3),
    "FL": (3, 6),
    "RL": (6, 9),
    "RR": (9, 12),
}
PHASE_NAMES = {0: "lift", 1: "extend", 2: "hold"}


# ──────────────────────────────────────────────────────────────────────
# CLI / config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_dirs:                 list[str]
    format_filter:             str | None
    epochs:                    int
    batch_size:                int
    lr:                        float
    weight_decay:              float
    joint_weights:             list[float]
    val_fraction:              float
    seed:                      int
    out_dir:                   str
    device:                    str
    log_every:                 int
    phases:                    list[int]
    no_save:                   bool
    gain_schedule_filter:      str | None
    intrinsics_version_filter: str | None
    chunk_size:                int


def _parse_csv_paths(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dirs",     default="data/real/stage_d_v3",
                   help="Comma-separated episode directories.")
    p.add_argument("--format-filter", default=None,
                   choices=[None, "v2", "v3"])
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--joint-weights",
                   default="3,3,3,1,1,1,1,1,1,1,1,1",
                   help="12 comma-separated non-negative floats.")
    p.add_argument("--val-fraction",  type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    # --output-dir is the canonical name; --out-dir kept as an alias for
    # backwards compatibility.
    p.add_argument("--out-dir", "--output-dir", dest="out_dir",
                   default="models/stage_d",
                   help="Where to save the checkpoint, config.json, and "
                        "training_log.json. Created if it does not exist.")
    p.add_argument("--device",        default="auto")
    p.add_argument("--log-every",     type=int,   default=1)
    p.add_argument("--phases",        default="0,1,2",
                   help="Comma-separated phase ints to include.")
    p.add_argument("--no-save",       action="store_true")
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
        "--chunk-size",
        type=int,
        default=1,
        help="Action chunk size. 1 (default) trains the legacy single-step "
             "MLP — bit-for-bit identical pre-change behaviour. >1 (e.g., 25 "
             "for 50ms at 500Hz) trains the chunked StageDChunkedPolicy with "
             "K future residuals; per-step joint weights are broadcast across "
             "K. Boundary samples (t+K > episode T) are dropped.",
    )
    a = p.parse_args()

    weights = _parse_csv_floats(a.joint_weights)
    if len(weights) != STAGE_D_OUTPUT_DIM:
        raise ValueError(
            f"--joint-weights must have {STAGE_D_OUTPUT_DIM} entries, "
            f"got {len(weights)}")
    if any(w < 0 for w in weights):
        raise ValueError(f"--joint-weights must be non-negative, got {weights}")

    phases = _parse_csv_ints(a.phases)
    if any(ph not in (0, 1, 2) for ph in phases):
        raise ValueError(f"--phases must be subset of {{0,1,2}}, got {phases}")

    if a.chunk_size < 1:
        raise ValueError(
            f"--chunk-size must be >= 1, got {a.chunk_size}")

    return TrainConfig(
        data_dirs=_parse_csv_paths(a.data_dirs),
        format_filter=a.format_filter,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        joint_weights=weights,
        val_fraction=a.val_fraction,
        seed=a.seed,
        out_dir=a.out_dir,
        device=a.device,
        log_every=a.log_every,
        phases=phases,
        no_save=a.no_save,
        gain_schedule_filter=a.gain_schedule_filter,
        intrinsics_version_filter=a.intrinsics_version_filter,
        chunk_size=a.chunk_size,
    )


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Dataset wrapper that applies normalization on the fly
# ──────────────────────────────────────────────────────────────────────

class NormalizedStageDDataset(Dataset):
    """Wrap a StageDDataset and z-score-normalize the ``state`` field."""

    def __init__(self, ds: StageDDataset, normalizer: StateNormalizer):
        self.ds = ds
        # Cache as torch tensors so normalization stays on CPU and fast.
        if normalizer.mean is None or normalizer.std is None:
            raise RuntimeError("normalizer must be fit before wrapping dataset")
        self.mean = torch.from_numpy(normalizer.mean.astype(np.float32))
        self.std  = torch.from_numpy(normalizer.std.astype(np.float32))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        s = self.ds[idx]
        s["state"] = (s["state"] - self.mean) / self.std
        return s


# ──────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────

_TENSOR_KEYS = ("state", "delta_fr", "delta_q")
_TENSOR_KEYS_CHUNKED = ("state", "delta_q_chunk")


def stage_d_collate(batch: list[dict]) -> dict:
    out: dict = {}
    for k in _TENSOR_KEYS:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["phase"]               = torch.stack(
        [b["phase"] for b in batch], dim=0)
    out["episode_id"]          = [b["episode_id"] for b in batch]
    out["data_format_version"] = [b["data_format_version"] for b in batch]
    return out


def stage_d_collate_chunked(batch: list[dict]) -> dict:
    out: dict = {}
    for k in _TENSOR_KEYS_CHUNKED:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["phase"]               = torch.stack(
        [b["phase"] for b in batch], dim=0)
    out["episode_id"]          = [b["episode_id"] for b in batch]
    out["data_format_version"] = [b["data_format_version"] for b in batch]
    return out


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """``mean over batch of sum_j w_j * (pred_j - target_j)^2``."""
    sq = (pred - target) ** 2
    return (sq * weights).sum(dim=1).mean()


def weighted_mse_chunk(pred: torch.Tensor, target: torch.Tensor,
                        weights: torch.Tensor) -> torch.Tensor:
    """Chunked variant: pred / target are (B, K, 12); weights (12,) broadcast.

    Returns the mean over both batch and K of the per-step weighted joint
    sum-of-squared-error. Reduces to ``weighted_mse`` when K=1.
    """
    sq = (pred - target) ** 2  # (B, K, 12)
    return (sq * weights).sum(dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────
# Epoch loops
# ──────────────────────────────────────────────────────────────────────

def train_epoch(
    bundle: StageDBundle,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    device: torch.device,
) -> dict:
    bundle.train()
    loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["state"].to(device)
        y = batch["delta_q"].to(device)
        pred = bundle.policy(x)
        loss = weighted_mse(pred, y, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item())
        n_batches += 1
    return {"weighted_loss": loss_sum / max(1, n_batches)}


@torch.no_grad()
def eval_epoch(
    bundle: StageDBundle,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
) -> dict:
    bundle.eval()
    loss_sum = 0.0
    n_batches = 0

    sqerr_per_joint = torch.zeros(
        STAGE_D_OUTPUT_DIM, dtype=torch.float64, device=device)
    n_total = 0

    # Per-phase × per-joint sqerr.
    sqerr_phase_joint = {
        ph: torch.zeros(STAGE_D_OUTPUT_DIM,
                        dtype=torch.float64, device=device)
        for ph in (0, 1, 2)
    }
    n_per_phase = {ph: 0 for ph in (0, 1, 2)}

    for batch in loader:
        x = batch["state"].to(device)
        y = batch["delta_q"].to(device)
        ph = batch["phase"].to(device)
        pred = bundle.policy(x)

        loss = weighted_mse(pred, y, weights)
        loss_sum += float(loss.item())
        n_batches += 1

        sq = (pred - y) ** 2  # (B, 12)
        sqerr_per_joint += sq.sum(dim=0).double()
        n_total += sq.shape[0]

        for ph_id in (0, 1, 2):
            mask = ph == ph_id
            if mask.any():
                sqerr_phase_joint[ph_id] += sq[mask].sum(dim=0).double()
                n_per_phase[ph_id] += int(mask.sum().item())

    per_joint_mse = (sqerr_per_joint / max(1, n_total)).cpu().numpy().tolist()
    per_phase_per_joint_mse = {
        ph: (sqerr_phase_joint[ph] / max(1, n_per_phase[ph]))
            .cpu().numpy().tolist()
        for ph in (0, 1, 2)
    }
    return {
        "weighted_loss":            loss_sum / max(1, n_batches),
        "per_joint_mse":            per_joint_mse,
        "per_phase_per_joint_mse":  per_phase_per_joint_mse,
        "n_per_phase":              n_per_phase,
        "n_total":                  n_total,
    }


# ──────────────────────────────────────────────────────────────────────
# Chunked epoch loops (parallel to the single-step versions above)
# ──────────────────────────────────────────────────────────────────────

def train_epoch_chunked(
    bundle: StageDChunkedBundle,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    device: torch.device,
) -> dict:
    bundle.train()
    loss_sum = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["state"].to(device)              # (B, 33)
        y = batch["delta_q_chunk"].to(device)      # (B, K, 12)
        pred = bundle.policy(x)                    # (B, K, 12)
        loss = weighted_mse_chunk(pred, y, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item())
        n_batches += 1
    return {"weighted_loss": loss_sum / max(1, n_batches)}


@torch.no_grad()
def eval_epoch_chunked(
    bundle: StageDChunkedBundle,
    loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
) -> dict:
    """Chunked-mode eval. ``per_joint_mse`` averages over batch and K.

    Per-phase breakdown is keyed on the chunk's *start* phase; the body of
    the chunk may span phase transitions.
    """
    bundle.eval()
    loss_sum = 0.0
    n_batches = 0

    sqerr_per_joint = torch.zeros(
        STAGE_D_OUTPUT_DIM, dtype=torch.float64, device=device)
    n_total = 0  # total time-step labels seen across (B, K)

    sqerr_phase_joint = {
        ph: torch.zeros(STAGE_D_OUTPUT_DIM,
                        dtype=torch.float64, device=device)
        for ph in (0, 1, 2)
    }
    n_per_phase = {ph: 0 for ph in (0, 1, 2)}

    for batch in loader:
        x = batch["state"].to(device)              # (B, 33)
        y = batch["delta_q_chunk"].to(device)      # (B, K, 12)
        ph = batch["phase"].to(device)             # (B,)
        pred = bundle.policy(x)                    # (B, K, 12)

        loss = weighted_mse_chunk(pred, y, weights)
        loss_sum += float(loss.item())
        n_batches += 1

        sq = (pred - y) ** 2                       # (B, K, 12)
        sqerr_per_joint += sq.sum(dim=(0, 1)).double()
        n_total += sq.shape[0] * sq.shape[1]

        for ph_id in (0, 1, 2):
            mask = ph == ph_id
            if mask.any():
                sqerr_phase_joint[ph_id] += sq[mask].sum(dim=(0, 1)).double()
                n_per_phase[ph_id] += int(mask.sum().item()) * sq.shape[1]

    per_joint_mse = (sqerr_per_joint / max(1, n_total)).cpu().numpy().tolist()
    per_phase_per_joint_mse = {
        ph: (sqerr_phase_joint[ph] / max(1, n_per_phase[ph]))
            .cpu().numpy().tolist()
        for ph in (0, 1, 2)
    }
    return {
        "weighted_loss":            loss_sum / max(1, n_batches),
        "per_joint_mse":            per_joint_mse,
        "per_phase_per_joint_mse":  per_phase_per_joint_mse,
        "n_per_phase":              n_per_phase,
        "n_total":                  n_total,
    }


# ──────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _episode_ids(ds: StageDDataset) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in ds.samples:
        if s.episode_id not in seen:
            seen.add(s.episode_id)
            out.append(s.episode_id)
    return out


def _print_normalizer_summary(norm: StateNormalizer) -> None:
    print(f"{'slice':<8} {'field':<24} {'mean':>10} {'std':>10}")
    for start, end, name in STATE_LAYOUT:
        m = float(norm.mean[start:end].mean())
        s = float(norm.std[start:end].mean())
        print(f"{start:>2}:{end:<4} {name:<24} {m:>+10.4f} {s:>10.4f}")


def _print_per_joint_table(per_joint_mse: list[float],
                           weights: list[float]) -> None:
    print(f"  {'joint':<10} {'weight':>7} {'val_MSE':>10}")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:<10} {weights[j]:>7.2f} {per_joint_mse[j]:>10.6f}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Device: {device}")

    weights_np = np.asarray(cfg.joint_weights, dtype=np.float32)
    print(f"Joint weights: {weights_np.tolist()}")
    fr_only = bool(np.allclose(weights_np[3:], 0.0)
                   and np.any(weights_np[0:3] > 0))
    if fr_only:
        print()
        print("=" * 72)
        print("FR-ONLY weighting — support-leg outputs receive no gradient.")
        print("Whole-body architecture is preserved; switch to a non-zero")
        print("weight on joints 3..11 once v3 data is collected.")
        print("=" * 72)
        print()

    # Data ─────────────────────────────────────────────────────────────
    data_paths = [Path(p) for p in cfg.data_dirs]
    if cfg.gain_schedule_filter is not None:
        logger.info(f"gain_schedule_filter: {cfg.gain_schedule_filter!r}")
    if cfg.intrinsics_version_filter is not None:
        logger.info(
            f"intrinsics_version_filter: {cfg.intrinsics_version_filter!r}")
    train_ds, val_ds = build_stage_d_datasets(
        data_paths,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        phases=tuple(cfg.phases),
        gain_schedule_filter=cfg.gain_schedule_filter,
        intrinsics_version_filter=cfg.intrinsics_version_filter,
        format_filter=cfg.format_filter,
        chunk_size=cfg.chunk_size,
    )
    chunked = cfg.chunk_size > 1
    if chunked:
        logger.info(
            f"chunk_size={cfg.chunk_size} → training StageDChunkedPolicy.")
    train_ep_ids = _episode_ids(train_ds)
    val_ep_ids   = _episode_ids(val_ds)
    logger.info(
        f"Train: {len(train_ds)} steps across {len(train_ep_ids)} episodes")
    logger.info(
        f"Val:   {len(val_ds)} steps across {len(val_ep_ids)} episodes")

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty — check --data-dirs and "
                           "--format-filter.")

    # Normalizer (fit on train only) ───────────────────────────────────
    print()
    print("Fitting StateNormalizer on train split…")
    normalizer = StateNormalizer()
    normalizer.fit(train_ds.samples)
    _print_normalizer_summary(normalizer)
    print()

    train_norm = NormalizedStageDDataset(train_ds, normalizer)
    val_norm   = NormalizedStageDDataset(val_ds,   normalizer)

    collate_fn = stage_d_collate_chunked if chunked else stage_d_collate
    train_loader = DataLoader(
        train_norm, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(
        val_norm, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False)

    # Model ────────────────────────────────────────────────────────────
    if chunked:
        bundle = StageDChunkedBundle.fresh(
            normalizer, weights_np, cfg.chunk_size).to(device)
        train_fn = train_epoch_chunked
        eval_fn = eval_epoch_chunked
    else:
        bundle = StageDBundle.fresh(normalizer, weights_np).to(device)
        train_fn = train_epoch
        eval_fn = eval_epoch
    weights_t = bundle.joint_weights.to(device)
    logger.info(f"StageDPolicy params: {count_params(bundle.policy)}")

    optimizer = torch.optim.AdamW(
        bundle.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1

    # Train ────────────────────────────────────────────────────────────
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        train_stats = train_fn(
            bundle, train_loader, optimizer, weights_t, device)
        val_stats   = eval_fn(bundle, val_loader, weights_t, device)
        ep_dt = time.time() - ep_t0

        if val_stats["weighted_loss"] < best_val:
            best_val = val_stats["weighted_loss"]
            best_epoch = epoch

        history.append({
            "epoch":                       epoch,
            "train_weighted_loss":         train_stats["weighted_loss"],
            "val_weighted_loss":           val_stats["weighted_loss"],
            "val_per_joint_mse":           val_stats["per_joint_mse"],
            "val_per_phase_per_joint_mse": val_stats["per_phase_per_joint_mse"],
            "val_n_per_phase":             val_stats["n_per_phase"],
            "epoch_seconds":               ep_dt,
        })

        if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            pj = val_stats["per_joint_mse"]
            sup_mean = float(np.mean(pj[3:12]))
            logger.info(
                f"epoch {epoch:03d}/{cfg.epochs} "
                f"({ep_dt:5.2f}s) | "
                f"train_w={train_stats['weighted_loss']:.4f} "
                f"val_w={val_stats['weighted_loss']:.4f} | "
                f"val per-joint (FR): "
                f"hip={pj[0]:.4f} thigh={pj[1]:.4f} calf={pj[2]:.4f} | "
                f"val per-joint (mean support): {sup_mean:.4f}"
            )

    train_dt = time.time() - t_start

    # Final summary ────────────────────────────────────────────────────
    final = history[-1]
    print()
    print("=" * 72)
    print(f"Training complete in {train_dt:.1f}s "
          f"({train_dt / max(1, cfg.epochs):.2f}s/epoch).")
    print(f"Best val_weighted_loss: {best_val:.6f} at epoch {best_epoch}.")
    print(f"Final train_weighted_loss: {final['train_weighted_loss']:.6f}")
    print(f"Final val_weighted_loss:   {final['val_weighted_loss']:.6f}")
    print("=" * 72)

    print("\nFinal per-joint val MSE (unweighted), grouped by leg:")
    pj = final["val_per_joint_mse"]
    for leg, (a, b) in LEG_GROUPS.items():
        leg_mean = float(np.mean(pj[a:b]))
        joints_str = "  ".join(
            f"{JOINT_NAMES[j]}={pj[j]:.5f}" for j in range(a, b))
        print(f"  {leg}: mean={leg_mean:.5f}   {joints_str}")
    print()

    print("Final per-phase per-joint val MSE (FR slice only):")
    print(f"  {'phase':<8} {'hip':>10} {'thigh':>10} {'calf':>10}")
    for ph in (0, 1, 2):
        pp = final["val_per_phase_per_joint_mse"][ph]
        print(f"  {PHASE_NAMES[ph]:<8} "
              f"{pp[0]:>10.5f} {pp[1]:>10.5f} {pp[2]:>10.5f}")

    # Save ─────────────────────────────────────────────────────────────
    if cfg.no_save:
        logger.info("--no-save: skipping checkpoint write.")
        return

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "stage_d.pt"
    torch.save({
        "bundle_state_dict": bundle.state_dict(),
        "config":            asdict(cfg),
        "train_episode_ids": train_ep_ids,
        "val_episode_ids":   val_ep_ids,
        "best_val":          best_val,
        "best_epoch":        best_epoch,
        "chunk_size":        int(cfg.chunk_size),
    }, ckpt_path)
    logger.info(f"Checkpoint → {ckpt_path}")

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "training_log.json").write_text(json.dumps({
        "history":           history,
        "joint_names":       JOINT_NAMES,
        "phase_names":       PHASE_NAMES,
        "joint_weights":     cfg.joint_weights,
        "train_episode_ids": train_ep_ids,
        "val_episode_ids":   val_ep_ids,
        "best_val":          best_val,
        "best_epoch":        best_epoch,
        "state_dim":         STATE_DIM,
    }, indent=2))
    logger.info(f"Config + log → {out_dir}")


if __name__ == "__main__":
    main()
