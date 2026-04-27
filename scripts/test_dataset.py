"""
scripts/test_dataset.py

End-to-end validation of src/data/dataset.py against the real episode
files under data/real/stage_d_v2/. Does not import any training code.

Outputs a summary to stdout and saves a fit StateNormalizer to
models/state_normalizer.json.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.dataset import (
    STATE_DIM,
    STATE_LAYOUT,
    StageDSample,
    StateNormalizer,
    build_stage_c_datasets,
    build_stage_d_datasets,
    load_episode,
    split_episodes_train_val,
)

EPISODES_DIR = Path("data/real/stage_d_v2")
EPISODES_DIR_V3 = Path("data/real/stage_d_v3")  # may not exist yet
NORMALIZER_PATH = Path("models/state_normalizer.json")

JOINT_NAMES_12 = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]


def _hdr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── File discovery and raw load diagnostics ──────────────────────
    _hdr("File discovery")
    paths = sorted(EPISODES_DIR.glob("episode_*.h5"))
    print(f"Directory: {EPISODES_DIR.resolve()}")
    print(f"Files found: {len(paths)}")

    loaded = 0
    skip_reasons: Counter[str] = Counter()
    # We re-open each file ourselves to attribute skip reasons. load_episode
    # already handles all the edge cases — here we just mirror its decision
    # and classify None returns.
    import h5py  # local import: only needed for classification here

    for p in paths:
        ep = load_episode(p)
        if ep is not None:
            loaded += 1
            continue
        # Classify skip reason.
        try:
            with h5py.File(p, "r") as f:
                attrs = f.attrs
                if "success_target" not in attrs:
                    skip_reasons["no success_target attr"] += 1
                elif not bool(np.asarray(attrs["success_target"]).item()):
                    skip_reasons["success_target=False"] += 1
                elif "per_step" not in f:
                    skip_reasons["no per_step group"] += 1
                elif f["per_step"]["timestamp"].shape[0] == 0:
                    skip_reasons["T=0"] += 1
                elif any(k not in f["per_step"] for k in
                         ("joint_tau_est", "imu_gyro", "imu_accel",
                          "achieved_delta_fr")):
                    skip_reasons["pre-v2 per_step"] += 1
                elif "phase_transitions" not in f or any(
                        k not in f["phase_transitions"]
                        for k in ("lift_end", "extend_end", "hold_end")):
                    skip_reasons["missing phase_transitions"] += 1
                else:
                    skip_reasons["other"] += 1
        except Exception as e:
            skip_reasons[f"open failed ({type(e).__name__})"] += 1

    print(f"Loaded successfully (success_target=True, v2 schema): {loaded}")
    print(f"Skipped: {len(paths) - loaded}")
    for reason, n in skip_reasons.most_common():
        print(f"  - {reason}: {n}")

    # ── Stage C ──────────────────────────────────────────────────────
    _hdr("Stage C datasets")
    c_train, c_val = build_stage_c_datasets(
        EPISODES_DIR, val_fraction=0.2, seed=42)
    print(f"Train size: {len(c_train)}")
    print(f"Val size:   {len(c_val)}")

    if len(c_train) > 0:
        s0 = c_train[0]
        print("First train sample shapes:")
        for k, v in s0.items():
            if hasattr(v, "shape"):
                print(f"  {k:<20} {tuple(v.shape)}  dtype={v.dtype}")
            else:
                print(f"  {k:<20} (non-tensor) {v!r}")

    def standoff_stats(ds) -> tuple[int, int]:
        nan_count = 0
        finite_count = 0
        for i in range(len(ds)):
            so = ds[i]["standoff_offset"].numpy()
            if np.isnan(so).any():
                nan_count += 1
            else:
                finite_count += 1
        return nan_count, finite_count

    for name, ds in (("train", c_train), ("val", c_val)):
        nan_c, fin_c = standoff_stats(ds)
        print(f"Stage C {name} standoff_offset: finite={fin_c}, nan={nan_c}")

    # ── Stage D ──────────────────────────────────────────────────────
    _hdr("Stage D datasets")
    d_train, d_val = build_stage_d_datasets(
        EPISODES_DIR, val_fraction=0.2, seed=42, phases=(0, 1, 2))
    print(f"Train steps: {len(d_train)}")
    print(f"Val steps:   {len(d_val)}")

    if len(d_train) > 0:
        s0 = d_train[0]
        print("First train sample shapes:")
        for k, v in s0.items():
            if hasattr(v, "shape"):
                shape = tuple(v.shape) if v.dim() > 0 else "()"
                print(f"  {k:<12} {shape}  dtype={v.dtype}")
            else:
                print(f"  {k:<12} (non-tensor) {v!r}")

    # Phase distribution across train.
    phase_counts: Counter[int] = Counter(
        int(d_train.samples[i].phase) for i in range(len(d_train)))
    phase_names = {0: "lift", 1: "extend", 2: "hold"}
    print("Stage D train phase distribution:")
    total = max(1, len(d_train))
    for ph in (0, 1, 2):
        n = phase_counts.get(ph, 0)
        print(f"  phase {ph} ({phase_names[ph]:<6}): {n:>6}  "
              f"({100.0 * n / total:5.1f}%)")

    # Achieved delta statistics across train.
    if len(d_train) > 0:
        deltas = np.stack(
            [s.achieved_delta_fr for s in d_train.samples], axis=0)
        print("Stage D achieved_delta_fr statistics (train):")
        for j, label in enumerate(("hip", "thigh", "calf")):
            col = deltas[:, j]
            print(f"  {label:<5}: mean={col.mean():+.4f}  "
                  f"std={col.std():.4f}  max|.|={np.abs(col).max():.4f}")

    # ── 12-dim achieved_delta_q checks ──────────────────────────────
    _hdr("Stage D achieved_delta_q (12 joints)")
    if len(d_train) == 0:
        print("No train samples — skipping 12-dim checks.")
    else:
        # Shape + dtype on first sample.
        s0 = d_train[0]
        assert s0["delta_q"].shape == (12,), \
            f"delta_q shape {tuple(s0['delta_q'].shape)} != (12,)"
        assert s0["delta_fr"].shape == (3,), \
            f"delta_fr shape {tuple(s0['delta_fr'].shape)} != (3,)"
        print(f"delta_q shape: {tuple(s0['delta_q'].shape)}, "
              f"dtype={s0['delta_q'].dtype}")

        # FR-slice equality on a random subset of ≥100 samples.
        n_check = min(100, len(d_train))
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(d_train), size=n_check, replace=False)
        mismatches = 0
        for idx in idxs:
            s = d_train[int(idx)]
            fr = s["delta_fr"].numpy()
            q  = s["delta_q"].numpy()
            if not np.array_equal(fr, q[0:3]):
                # NaN-tolerant compare: if both are NaN at same positions,
                # treat as equal.
                fr_nan = np.isnan(fr)
                q_nan  = np.isnan(q[0:3])
                if not (np.array_equal(fr_nan, q_nan)
                        and np.array_equal(fr[~fr_nan], q[0:3][~q_nan])):
                    mismatches += 1
        print(f"FR-slice equality check: {n_check - mismatches}/{n_check} "
              f"samples have delta_q[0:3] == delta_fr")
        assert mismatches == 0, \
            f"{mismatches}/{n_check} samples violate delta_q[0:3] == delta_fr"

        # Per-joint statistics over all 12 joints.
        deltas_q = np.stack(
            [s.achieved_delta_q for s in d_train.samples], axis=0)
        print(f"\nPer-joint statistics over {len(d_train)} train steps:")
        print(f"  {'joint':<10} {'mean':>10} {'std':>10} {'max|.|':>10}")
        for j, label in enumerate(JOINT_NAMES_12):
            col = deltas_q[:, j]
            mark = " *FR*" if j < 3 else ""
            print(f"  {label:<10} {col.mean():>+10.4f} "
                  f"{col.std():>10.4f} {np.abs(col).max():>10.4f}{mark}")

        fr_mag = float(np.abs(deltas_q[:, 0:3]).mean())
        sup_mag = float(np.abs(deltas_q[:, 3:12]).mean())
        print(f"\nMean |delta| FR (joints 0-2):     {fr_mag:.4f}")
        print(f"Mean |delta| support (joints 3-11): {sup_mag:.4f}")
        print(f"FR / support ratio:                {fr_mag / max(sup_mag, 1e-9):.2f}x")

    # ── data_format_version check ──────────────────────────────────
    _hdr("data_format_version (all current data must be v2)")
    versions = Counter(s.data_format_version for s in d_train.samples)
    versions.update(s.data_format_version for s in d_val.samples)
    print(f"Version counts across train+val samples: {dict(versions)}")
    assert set(versions) <= {"v2", "v3"}, \
        f"unexpected versions: {set(versions)}"
    assert "v3" not in versions, \
        f"no v3 data should exist yet, got {versions}"
    print("OK — every sample reports data_format_version == 'v2'.")

    # ── format_filter ──────────────────────────────────────────────
    _hdr("build_stage_d_datasets format_filter")
    d_train_v2, d_val_v2 = build_stage_d_datasets(
        EPISODES_DIR, val_fraction=0.2, seed=42, format_filter="v2")
    print(f'format_filter="v2" → train={len(d_train_v2)} val={len(d_val_v2)}')
    assert len(d_train_v2) == len(d_train), \
        f'v2 filter changed train count: {len(d_train_v2)} vs {len(d_train)}'
    assert len(d_val_v2) == len(d_val), \
        f'v2 filter changed val count: {len(d_val_v2)} vs {len(d_val)}'

    d_train_v3, d_val_v3 = build_stage_d_datasets(
        EPISODES_DIR, val_fraction=0.2, seed=42, format_filter="v3")
    print(f'format_filter="v3" → train={len(d_train_v3)} val={len(d_val_v3)}')
    assert len(d_train_v3) == 0 and len(d_val_v3) == 0, \
        "v3 filter should yield empty datasets (no v3 data exists yet)"
    print("OK — v3 filter is empty as expected.")

    # ── Multi-dir build (v3 dir may not exist) ──────────────────────
    _hdr("build_stage_d_datasets multi-dir (v2 + v3 even if v3 missing)")
    print(f"v3 dir exists: {EPISODES_DIR_V3.is_dir()}")
    d_train_multi, d_val_multi = build_stage_d_datasets(
        [EPISODES_DIR, EPISODES_DIR_V3], val_fraction=0.2, seed=42)
    print(f"multi-dir → train={len(d_train_multi)} val={len(d_val_multi)}")
    assert len(d_train_multi) == len(d_train), \
        f'multi-dir train count drift: {len(d_train_multi)} vs {len(d_train)}'
    assert len(d_val_multi) == len(d_val), \
        f'multi-dir val count drift: {len(d_val_multi)} vs {len(d_val)}'
    print("OK — multi-dir matches single-v2 (since v3 contributes 0 episodes).")

    # ── Normalizer fit + save ────────────────────────────────────────
    _hdr("StateNormalizer")
    if len(d_train) == 0:
        print("No train samples — cannot fit normalizer.")
    else:
        norm = StateNormalizer()
        norm.fit(d_train.samples)
        print(f"Fit on {len(d_train)} train samples. "
              f"state_dim={STATE_DIM}")
        print(f"{'slice':<8} {'field':<24} {'mean':>10} {'std':>10}")
        for start, end, name in STATE_LAYOUT:
            m = float(norm.mean[start:end].mean())
            s = float(norm.std[start:end].mean())
            print(f"{start:>2}:{end:<4} {name:<24} {m:>+10.4f} {s:>10.4f}")

        norm.save(NORMALIZER_PATH)
        print(f"Saved normalizer → {NORMALIZER_PATH.resolve()}")

        reloaded = StateNormalizer.load(NORMALIZER_PATH)
        assert np.allclose(reloaded.mean, norm.mean)
        assert np.allclose(reloaded.std, norm.std)
        print("Round-trip load verified.")

    # ── Train/val episode disjointness ──────────────────────────────
    _hdr("Train/val episode disjointness")
    train_paths, val_paths = split_episodes_train_val(
        paths, val_fraction=0.2, seed=42)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    for p in train_paths:
        ep = load_episode(p)
        if ep is not None:
            train_ids.add(ep["episode_id"])
    for p in val_paths:
        ep = load_episode(p)
        if ep is not None:
            val_ids.add(ep["episode_id"])
    overlap = train_ids & val_ids
    print(f"Train episode ids: {len(train_ids)}")
    print(f"Val   episode ids: {len(val_ids)}")
    print(f"Overlap: {len(overlap)}  (must be 0)")
    if overlap:
        raise AssertionError(
            f"Train and val share episodes: {sorted(overlap)}")
    print("OK — no episode appears in both splits.")


if __name__ == "__main__":
    main()
