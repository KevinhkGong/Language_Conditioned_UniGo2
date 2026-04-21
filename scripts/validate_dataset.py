#!/usr/bin/env python
"""
scripts/validate_dataset.py

Dataset-level validation implementing spec Section 11.2.
Runs per-episode validation on every episode_*.h5 in a directory (excluding
the tuning/ subdirectory), then computes aggregate statistics and checks
them against targets.

Usage:
    python scripts/validate_dataset.py data/real/stage_d/

Exit codes:
    0 — all aggregate thresholds met
    1 — at least one threshold missed (or per-episode validation failed)
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Reuse per-episode validation
sys.path.insert(0, str(Path(__file__).parent))
from validate_episode import validate_episode


# ──────────────────────────────────────────────
# Aggregate thresholds (spec Section 11.2)
# ──────────────────────────────────────────────

EXPECTED_TOTAL_EPISODES   = 50
MIN_SUCCESS_TARGET_RATE   = 0.60
MIN_SUCCESS_FK_RATE       = 0.70
MIN_FK_TARGET_AGREEMENT   = 0.80
MIN_AUDIO_CONTACT_RATE    = 0.60


def _attr_to_str(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _collect_episode_metadata(path: Path) -> dict:
    """Extract the aggregate-relevant fields from an episode HDF5."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        meta = {
            "path": str(path),
            "success_fk":         bool(attrs.get("success_fk", False)),
            "success_audio_live": bool(attrs.get("success_audio_live", False)),
            "success_target":     bool(attrs.get("success_target", False)),
            "contact_method":     _attr_to_str(attrs.get("contact_method", "none")),
            "target_pos_base_at_standoff": np.asarray(
                attrs.get("target_pos_base_at_standoff", np.zeros(3)), dtype=np.float32),
            "perturbation_commanded": np.asarray(
                attrs.get("perturbation_commanded", np.zeros(2)), dtype=np.float32),
            "duration_s":         float(attrs.get("duration_s", 0.0)),
        }

        # Phase durations from per_step — label array
        if "per_step" in f and "phase_label" in f["per_step"]:
            pl = f["per_step/phase_label"][()]
            meta["steps_total"] = int(pl.size)
            meta["steps_lift"]   = int(np.sum(pl == 0))
            meta["steps_extend"] = int(np.sum(pl == 1))
            meta["steps_hold"]   = int(np.sum(pl == 2))
        else:
            meta["steps_total"] = 0
            meta["steps_lift"]   = 0
            meta["steps_extend"] = 0
            meta["steps_hold"]   = 0

        # Foot-to-target error magnitude — use median non-nan during hold
        if "per_step" in f and "foot_to_target_error" in f["per_step"]:
            err = f["per_step/foot_to_target_error"][()]
            finite = np.isfinite(err).all(axis=1)
            if finite.any():
                mags = np.linalg.norm(err[finite], axis=1)
                meta["err_median_cm"] = float(np.median(mags) * 100.0)
                meta["err_p95_cm"]    = float(np.percentile(mags, 95) * 100.0)
            else:
                meta["err_median_cm"] = float("nan")
                meta["err_p95_cm"]    = float("nan")
        else:
            meta["err_median_cm"] = float("nan")
            meta["err_p95_cm"]    = float("nan")

        # Jacobian-PID delta magnitude — max during episode
        if "per_step" in f and "jacobian_pid_delta" in f["per_step"]:
            jpd = f["per_step/jacobian_pid_delta"][()]
            mags = np.linalg.norm(jpd, axis=1)
            meta["max_delta_rad"] = float(np.max(mags)) if mags.size else 0.0
        else:
            meta["max_delta_rad"] = 0.0

    return meta


def _classify_perturbation(pert: np.ndarray) -> str:
    """Bucket an episode into Stage C / Stage D category by perturbation magnitude."""
    norm = float(np.linalg.norm(pert))
    if norm <= 1e-3:
        return "no-perturb"
    if norm <= 0.03:
        return "small-perturb"
    return "large-perturb"


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a Stage D dataset directory.")
    ap.add_argument("directory", type=str,
                    help="Path to data/real/stage_d/ (or equivalent)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any per-episode check fails.")
    args = ap.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory.", file=sys.stderr)
        return 1

    # Gather episode files, excluding tuning/
    ep_paths = sorted(
        p for p in root.glob("episode_*.h5")
        if "tuning" not in p.parts
    )
    if not ep_paths:
        print(f"No episode_*.h5 files found in {root}.")
        return 1

    print("=" * 72)
    print(f"Dataset validation: {root}")
    print(f"Found {len(ep_paths)} training episodes (tuning/ excluded)")
    print("=" * 72)

    # ── Per-episode validation ──────────────────────────────────────────────
    per_ep_results = []
    for p in ep_paths:
        res = validate_episode(p)
        per_ep_results.append((p, res))
    n_failed = sum(1 for _, r in per_ep_results if not r.pass_all())
    print(f"Per-episode validation: {len(ep_paths) - n_failed}/{len(ep_paths)} pass")
    if n_failed > 0:
        print("Failing episodes:")
        for p, r in per_ep_results:
            if not r.pass_all():
                print(f"  {p}")
                if args.strict:
                    print(f"    (--strict enabled: run validate_episode.py "
                          f"{p} for detail)")

    # ── Aggregate stats ─────────────────────────────────────────────────────
    meta_list = []
    for p in ep_paths:
        try:
            meta_list.append(_collect_episode_metadata(p))
        except Exception as e:
            print(f"  WARNING: failed to read metadata from {p}: {e}")

    if not meta_list:
        print("No metadata readable — abort.")
        return 1

    n = len(meta_list)
    fk_rate    = sum(m["success_fk"]         for m in meta_list) / n
    audio_rate = sum(m["success_audio_live"] for m in meta_list) / n
    tgt_rate   = sum(m["success_target"]     for m in meta_list) / n
    audio_cm_rate = sum(m["contact_method"] == "audio" for m in meta_list) / n
    # Agreement between fk and target
    n_agree = sum(
        (m["success_fk"] == m["success_target"]) for m in meta_list
    )
    agreement = n_agree / n

    # Perturbation bucket breakdown
    buckets: dict[str, list] = {"no-perturb": [], "small-perturb": [], "large-perturb": []}
    for m in meta_list:
        buckets[_classify_perturbation(m["perturbation_commanded"])].append(m)

    # target_pos_base_at_standoff histogram
    standoff_arr = np.stack([m["target_pos_base_at_standoff"] for m in meta_list])
    x_min, x_max = float(standoff_arr[:, 0].min()), float(standoff_arr[:, 0].max())
    y_min, y_max = float(standoff_arr[:, 1].min()), float(standoff_arr[:, 1].max())
    z_min, z_max = float(standoff_arr[:, 2].min()), float(standoff_arr[:, 2].max())

    # Phase duration distribution (steps per episode — proxy for duration)
    steps_lift   = np.array([m["steps_lift"]   for m in meta_list])
    steps_extend = np.array([m["steps_extend"] for m in meta_list])
    steps_hold   = np.array([m["steps_hold"]   for m in meta_list])

    # Per-bucket mean error
    def bucket_stats(bucket_name: str) -> str:
        b = buckets[bucket_name]
        if not b:
            return f"    {bucket_name:<16}  (0 episodes)"
        errs = np.array([m["err_median_cm"] for m in b if np.isfinite(m["err_median_cm"])])
        return (
            f"    {bucket_name:<16}  "
            f"n={len(b):>3}  "
            f"fk={sum(m['success_fk'] for m in b)/len(b)*100:5.1f}%  "
            f"audio={sum(m['success_audio_live'] for m in b)/len(b)*100:5.1f}%  "
            f"target={sum(m['success_target'] for m in b)/len(b)*100:5.1f}%  "
            f"err_med={np.median(errs):.2f}cm" if errs.size else
            f"    {bucket_name:<16}  n={len(b):>3}  (no finite errors)"
        )

    # Delta vs perturbation correlation — spec 11.2 says delta magnitude should
    # grow with perturbation magnitude.
    pert_norms = np.array(
        [float(np.linalg.norm(m["perturbation_commanded"])) for m in meta_list]
    )
    delta_max  = np.array([m["max_delta_rad"] for m in meta_list])
    if pert_norms.std() > 1e-6 and delta_max.std() > 1e-6:
        corr = float(np.corrcoef(pert_norms, delta_max)[0, 1])
    else:
        corr = float("nan")

    # ── Print report ────────────────────────────────────────────────────────
    print()
    print("── Aggregate stats ─────────────────────────────────────────")
    print(f"  success_fk rate         : {fk_rate*100:.1f}%")
    print(f"  success_audio_live rate : {audio_rate*100:.1f}%")
    print(f"  success_target rate     : {tgt_rate*100:.1f}%")
    print(f"  fk / target agreement   : {agreement*100:.1f}%")
    print(f"  contact_method==audio   : {audio_cm_rate*100:.1f}%")
    print()
    print("── Perturbation buckets ───────────────────────────────────")
    for bucket in ("no-perturb", "small-perturb", "large-perturb"):
        print(bucket_stats(bucket))
    print()
    print("── target_pos_base_at_standoff range ─────────────────────")
    print(f"  x: [{x_min:+.3f}, {x_max:+.3f}]  (expect ~0.203 at no-perturb)")
    print(f"  y: [{y_min:+.3f}, {y_max:+.3f}]  (expect ~0.140 at no-perturb)")
    print(f"  z: [{z_min:+.3f}, {z_max:+.3f}]  (matches button height range)")
    print()
    print("── Phase durations (steps at 500 Hz) ─────────────────────")
    print(f"  lift   median={np.median(steps_lift):.0f}  "
          f"p5={np.percentile(steps_lift, 5):.0f}  "
          f"p95={np.percentile(steps_lift, 95):.0f}")
    print(f"  extend median={np.median(steps_extend):.0f}  "
          f"p5={np.percentile(steps_extend, 5):.0f}  "
          f"p95={np.percentile(steps_extend, 95):.0f}")
    print(f"  hold   median={np.median(steps_hold):.0f}  "
          f"p5={np.percentile(steps_hold, 5):.0f}  "
          f"p95={np.percentile(steps_hold, 95):.0f}")
    print()
    print("── Correction magnitude vs perturbation ─────────────────")
    print(f"  corr(|perturbation|, max|delta|) = {corr:+.3f}  "
          f"(expect positive)")
    print()

    # ── Threshold checks ────────────────────────────────────────────────────
    print("── Threshold checks ──────────────────────────────────────")
    thresholds = [
        ("episode count >= 50",
         n >= EXPECTED_TOTAL_EPISODES,
         f"got {n}"),
        (f"success_target rate >= {MIN_SUCCESS_TARGET_RATE*100:.0f}%",
         tgt_rate >= MIN_SUCCESS_TARGET_RATE,
         f"got {tgt_rate*100:.1f}%"),
        (f"success_fk rate >= {MIN_SUCCESS_FK_RATE*100:.0f}%",
         fk_rate >= MIN_SUCCESS_FK_RATE,
         f"got {fk_rate*100:.1f}%"),
        (f"fk/target agreement >= {MIN_FK_TARGET_AGREEMENT*100:.0f}%",
         agreement >= MIN_FK_TARGET_AGREEMENT,
         f"got {agreement*100:.1f}%"),
        (f"contact_method==audio >= {MIN_AUDIO_CONTACT_RATE*100:.0f}%",
         audio_cm_rate >= MIN_AUDIO_CONTACT_RATE,
         f"got {audio_cm_rate*100:.1f}%"),
    ]
    any_fail = False
    for label, ok, detail in thresholds:
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {label}  ({detail})")
        if not ok:
            any_fail = True

    # Per-episode failures count against strict mode
    if args.strict and n_failed > 0:
        any_fail = True

    print()
    print("OVERALL: " + ("PASS" if not any_fail else "FAIL"))
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())