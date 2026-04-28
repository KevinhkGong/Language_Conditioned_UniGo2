"""
scripts/verify_calibration.py

Quantify the systematic error in the current visual grounding pipeline by
comparing its 3D position estimate against:
    (a) a calibrated unprojection that uses newly obtained Go2X intrinsics
        and the plumb_bob distortion model, and
    (b) tape-measured ground truth in the robot base frame.

The script does NOT modify src/perception/grounding.py.  The calibrated
unprojection is implemented inline below for offline analysis only — the
production pipeline must remain unchanged because Stage D was trained
against its (biased) outputs.

Per physical button position the user enters tape-measured (x, y, z) in
the base frame; the script then captures N frames, runs grounding on each,
and computes both the existing biased estimate and a calibrated estimate
(same depth, same camera→base extrinsics, but undistorted-and-rebackprojected
through the candidate K, D).  Aggregated stats are saved to
data/calibration_check/<UTC timestamp>/ for offline review.

Usage:
    python scripts/verify_calibration.py
    python scripts/verify_calibration.py --positions 5 --frames 15
    python scripts/verify_calibration.py --output-dir data/calibration_check/run1
    python scripts/verify_calibration.py --no-save-frames

Ethernet setup (run before this script):
    sudo ip addr flush dev enx98fc84e68f1a
    sudo ip addr add 192.168.123.99/24 dev enx98fc84e68f1a
    sudo ip link set enx98fc84e68f1a up
    conda activate env_go2
"""

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

from src.perception.grounding import Go2Camera, VisualGrounder  # noqa: E402

logger = logging.getLogger(__name__)


# ── Candidate Go2X head-camera calibration ────────────────────────────────────
# Source: unofficial Go2X calibration, plumb_bob model, obtained April 2026.
# Treated as a candidate calibration for offline auditing only — the
# production pipeline still uses the URDF-derived pinhole intrinsics in
# src/perception/grounding.py.
K_CAL = np.array([
    [1310.77826,    0.0,    1018.71143],
    [   0.0,     1320.25059,  637.37672],
    [   0.0,        0.0,        1.0    ],
], dtype=np.float64)

D_CAL = np.array([-0.415971, 0.158898, -0.015395, -0.008031, 0.000000],
                 dtype=np.float64)  # plumb_bob: k1, k2, p1, p2, k3


# ── Camera → base extrinsics ──────────────────────────────────────────────────
# These constants MUST match those in src/perception/grounding.py.  They are
# duplicated here intentionally so this script remains a standalone audit
# tool that does not depend on the (possibly evolving) grounding module.
CAMERA_FORWARD = 0.327   # m
CAMERA_LATERAL = 0.000   # m
CAMERA_HEIGHT  = 0.043   # m
R_CAM_TO_BASE = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)
T_CAM_IN_BASE = np.array(
    [CAMERA_FORWARD, CAMERA_LATERAL, CAMERA_HEIGHT], dtype=np.float64,
)


# ── Tunables ──────────────────────────────────────────────────────────────────
MAX_CONSECUTIVE_NONE_FRAMES = 5     # > this many consecutive Nones aborts
BIAS_DENOM_EPS              = 0.01  # |y_cal| floor for reporting bias_factor


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated unprojection
# ─────────────────────────────────────────────────────────────────────────────

def unproject_calibrated(u: float, v: float, depth_m: float,
                         K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Unproject a single pixel to a 3D point in the camera optical frame.

    Uses the plumb_bob model: undistort the pixel back into the calibrated
    pinhole image plane (P=K), then back-project with K at depth Z.

    Args:
        u, v:    pixel coordinates of the centroid (in the original frame).
        depth_m: metric depth at that pixel.  This is whatever value the
                 caller obtained — typically the empirically corrected depth
                 from grounding.VisualGrounder._sample_depth.
        K:       (3, 3) intrinsic matrix.
        D:       (5,) plumb_bob distortion coefficients [k1, k2, p1, p2, k3].

    Returns:
        np.ndarray of shape (3,): [x, y, z] in the camera optical frame
        (z-forward, x-right, y-down).
    """
    pts = np.array([[[u, v]]], dtype=np.float64)            # (1, 1, 2)
    und = cv2.undistortPoints(pts, K, D, P=K)               # (1, 1, 2) px
    u_und = float(und[0, 0, 0])
    v_und = float(und[0, 0, 1])

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_cam = (u_und - cx) * depth_m / fx
    y_cam = (v_und - cy) * depth_m / fy
    z_cam = depth_m
    return np.array([x_cam, y_cam, z_cam], dtype=np.float64)


def camera_to_base(p_cam: np.ndarray) -> np.ndarray:
    """Apply the URDF-derived camera→base rigid transform.

    These constants are duplicated from src/perception/grounding.py — see
    the module-level note. Keep them in sync if grounding.py ever changes.
    """
    return R_CAM_TO_BASE @ p_cam + T_CAM_IN_BASE


# ─────────────────────────────────────────────────────────────────────────────
# User prompt
# ─────────────────────────────────────────────────────────────────────────────

def prompt_for_position(idx: int, total: int
                        ) -> Tuple[str, Optional[Tuple[float, float, float]]]:
    """Ask the user for a tape-measured (x, y, z) for one position.

    Returns:
        ("ok",   (x, y, z))  on a successful entry,
        ("skip", None)       if the user typed 'skip',
        ("quit", None)       if the user typed 'quit'.
    """
    while True:
        print(f"\nPosition {idx}/{total}: place button at the next location, "
              "then press Enter.")
        print("  Type 'skip' to skip this position, 'quit' to abort.")
        try:
            action = input("> ").strip().lower()
        except EOFError:
            return ("quit", None)
        if action == "quit":
            return ("quit", None)
        if action == "skip":
            return ("skip", None)

        try:
            coords = input("Enter tape-measured x y z in metres "
                           "(space separated): ").strip()
        except EOFError:
            return ("quit", None)
        parts = coords.split()
        if len(parts) != 3:
            print("  Need exactly 3 floats. Try again.")
            continue
        try:
            xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            print("  Could not parse one or more values as float. Try again.")
            continue
        return ("ok", xyz)


# ─────────────────────────────────────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray,
                   centroid: np.ndarray,
                   depth_m: float,
                   pos_old: np.ndarray,
                   pos_cal: np.ndarray,
                   truth: Tuple[float, float, float]) -> np.ndarray:
    """Mark the centroid pixel and write a single-line summary on the frame."""
    vis = frame.copy()
    cu, cv = int(round(centroid[0])), int(round(centroid[1]))
    cv2.circle(vis, (cu, cv), 8, (0, 255, 0), 2)

    line = (f"px=({cu},{cv}) depth={depth_m:.3f}m  "
            f"old_base=({pos_old[0]:.3f},{pos_old[1]:.3f},{pos_old[2]:.3f})  "
            f"cal_base=({pos_cal[0]:.3f},{pos_cal[1]:.3f},{pos_cal[2]:.3f})  "
            f"truth=({truth[0]:.3f},{truth[1]:.3f},{truth[2]:.3f})")
    cv2.putText(vis, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Per-position collection
# ─────────────────────────────────────────────────────────────────────────────

def _consume_warmup(camera: Go2Camera, n_warmup: int) -> None:
    """Discard n_warmup frames; raise RuntimeError on a long None streak."""
    consec_none = 0
    for _ in range(n_warmup):
        frame = camera.get_frame()
        if frame is None:
            consec_none += 1
            if consec_none > MAX_CONSECUTIVE_NONE_FRAMES:
                raise RuntimeError(
                    f"Camera returned None more than "
                    f"{MAX_CONSECUTIVE_NONE_FRAMES} times consecutively "
                    "during warmup. Check the network interface and the "
                    "Ethernet link to the Go2."
                )
        else:
            consec_none = 0


def collect_position(camera: Go2Camera,
                     grounder: VisualGrounder,
                     prompt: str,
                     pos_idx: int,
                     truth: Tuple[float, float, float],
                     n_frames: int,
                     n_warmup: int,
                     save_frames: bool,
                     output_dir: Path,
                     frames_writer: csv.writer,
                     csv_handle) -> dict:
    """Capture and process n_frames for one physical position.

    Each successful frame is written to frames_writer (and the underlying
    file is flushed) so partial progress survives a crash.

    Returns a dict suitable for inclusion in positions.json.
    """
    print(f"\n[Position {pos_idx}] truth=("
          f"{truth[0]:.3f}, {truth[1]:.3f}, {truth[2]:.3f}) — "
          f"warming up ({n_warmup} frames)…")
    _consume_warmup(camera, n_warmup)

    print(f"[Position {pos_idx}] capturing up to {n_frames} frames…")

    olds: List[np.ndarray] = []
    cals: List[np.ndarray] = []
    n_used = 0
    n_skipped = 0
    consec_none = 0
    frame_idx = 0

    while n_used + n_skipped < n_frames:
        frame = camera.get_frame()
        if frame is None:
            consec_none += 1
            if consec_none > MAX_CONSECUTIVE_NONE_FRAMES:
                raise RuntimeError(
                    f"Camera returned None more than "
                    f"{MAX_CONSECUTIVE_NONE_FRAMES} times consecutively. "
                    "Check the network interface and the Ethernet link "
                    "to the Go2."
                )
            continue
        consec_none = 0

        try:
            result = grounder.ground(frame, prompt)
        except Exception as exc:
            logger.warning("Grounding raised at pos %d frame %d: %s",
                           pos_idx, frame_idx, exc)
            n_skipped += 1
            frame_idx += 1
            continue

        # grounding.py contract: depth_m / position_base are None when the
        # raw depth is too close (< 0.15 m). Skip those frames per spec.
        if (result is None
                or result.position_base is None
                or result.depth_m is None
                or result.centroid_px is None):
            reason = ("no detection" if result is None
                      else "depth too close / partial result")
            logger.warning("Skipping pos %d frame %d: %s",
                           pos_idx, frame_idx, reason)
            n_skipped += 1
            frame_idx += 1
            continue

        u = float(result.centroid_px[0])
        v = float(result.centroid_px[1])
        depth_m = float(result.depth_m)
        confidence = (float(result.confidence)
                      if result.confidence is not None else None)

        pos_old = np.asarray(result.position_base, dtype=np.float64)
        p_cam_cal = unproject_calibrated(u, v, depth_m, K_CAL, D_CAL)
        pos_cal = camera_to_base(p_cam_cal)

        olds.append(pos_old)
        cals.append(pos_cal)

        frames_writer.writerow([
            pos_idx, frame_idx,
            f"{u:.3f}", f"{v:.3f}",
            f"{depth_m:.3f}",
            f"{confidence:.2f}" if confidence is not None else "",
            f"{pos_old[0]:.3f}", f"{pos_old[1]:.3f}", f"{pos_old[2]:.3f}",
            f"{pos_cal[0]:.3f}", f"{pos_cal[1]:.3f}", f"{pos_cal[2]:.3f}",
        ])
        csv_handle.flush()

        if save_frames:
            vis = annotate_frame(frame, result.centroid_px, depth_m,
                                 pos_old, pos_cal, truth)
            out_path = output_dir / f"frame_pos{pos_idx}_{frame_idx}.jpg"
            cv2.imwrite(str(out_path), vis)

        n_used += 1
        frame_idx += 1

    if n_skipped > n_frames // 2:
        logger.warning(
            "Position %d: %d/%d frames skipped (>half) — result uses %d frames.",
            pos_idx, n_skipped, n_frames, n_used,
        )

    if n_used == 0:
        return {
            "pos_idx": pos_idx,
            "ground_truth": [float(t) for t in truth],
            "n_frames_used": 0,
            "mean_old": None,
            "std_old":  None,
            "mean_cal": None,
            "std_cal":  None,
            "no_detection": True,
        }

    olds_arr = np.stack(olds, axis=0)
    cals_arr = np.stack(cals, axis=0)
    return {
        "pos_idx": pos_idx,
        "ground_truth": [float(t) for t in truth],
        "n_frames_used": int(n_used),
        "mean_old": olds_arr.mean(axis=0).tolist(),
        "std_old":  olds_arr.std(axis=0).tolist(),
        "mean_cal": cals_arr.mean(axis=0).tolist(),
        "std_cal":  cals_arr.std(axis=0).tolist(),
        "no_detection": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(records: List[dict]) -> str:
    """Build the aligned ASCII summary table as a single string."""
    header = (
        f"{'pos':>3}  "
        f"{'truth (x, y, z)':<26}  "
        f"{'old_base (x, y, z)':<26}  "
        f"{'cal_base (x, y, z)':<26}  "
        f"{'dy_old':>8}  {'dy_cal':>8}  {'bias_y':>8}"
    )
    sep = "-" * len(header)
    lines: List[str] = [header, sep]

    bias_factors:       List[float] = []
    cal_residual_norms: List[float] = []

    for rec in records:
        pos_idx = rec["pos_idx"]
        truth = rec["ground_truth"]
        truth_str = (f"({truth[0]:+.3f},{truth[1]:+.3f},{truth[2]:+.3f})")

        if rec.get("no_detection") or rec["mean_old"] is None:
            lines.append(
                f"{pos_idx:>3}  {truth_str:<26}  "
                f"{'no detection':<26}  {'no detection':<26}  "
                f"{'    n/a':>8}  {'    n/a':>8}  {'    n/a':>8}"
            )
            continue

        m_old = rec["mean_old"]
        m_cal = rec["mean_cal"]
        old_str = f"({m_old[0]:+.3f},{m_old[1]:+.3f},{m_old[2]:+.3f})"
        cal_str = f"({m_cal[0]:+.3f},{m_cal[1]:+.3f},{m_cal[2]:+.3f})"

        dy_old = m_old[1] - truth[1]
        dy_cal = m_cal[1] - truth[1]

        if abs(m_cal[1]) > BIAS_DENOM_EPS:
            bias = m_old[1] / m_cal[1]
            bias_str = f"{bias:+.3f}"
            bias_factors.append(bias)
        else:
            bias_str = "    n/a"

        residual = float(np.linalg.norm(np.array(m_cal) - np.array(truth)))
        cal_residual_norms.append(residual)

        lines.append(
            f"{pos_idx:>3}  {truth_str:<26}  "
            f"{old_str:<26}  {cal_str:<26}  "
            f"{dy_old:+8.3f}  {dy_cal:+8.3f}  {bias_str:>8}"
        )

    lines.append(sep)
    if bias_factors:
        lines.append(
            f"  mean bias_y = {np.mean(bias_factors):+.3f}  "
            f"(over {len(bias_factors)} positions)"
        )
    else:
        lines.append("  mean bias_y = n/a")

    if cal_residual_norms:
        lines.append(
            f"  mean |calibrated - truth| = "
            f"{np.mean(cal_residual_norms):.3f} m  "
            f"(over {len(cal_residual_norms)} positions)"
        )
    else:
        lines.append("  mean |calibrated - truth| = n/a")

    return "\n".join(lines) + "\n"


def write_artifacts(output_dir: Path, records: List[dict]) -> str:
    """Write summary.txt and positions.json. Returns the summary text."""
    text = build_summary(records)
    (output_dir / "summary.txt").write_text(text)
    with open(output_dir / "positions.json", "w") as f:
        json.dump(records, f, indent=2)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.output_dir is None:
        output_dir = _REPO_ROOT / "data" / "calibration_check" / timestamp
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  verify_calibration — Stage B grounding bias audit")
    print("=" * 65)
    print(f"  positions   : {args.positions}")
    print(f"  frames      : {args.frames}")
    print(f"  warmup      : {args.warmup}")
    print(f"  prompt      : \"{args.prompt}\"")
    print(f"  interface   : {args.interface}")
    print(f"  save frames : {not args.no_save_frames}")
    print(f"  output dir  : {output_dir}")
    print("=" * 65)

    print("\nLoading models and connecting to Go2 …")
    t0 = time.time()
    ChannelFactoryInitialize(0, args.interface)
    grounder = VisualGrounder()
    camera = Go2Camera(already_initialized=True)
    print(f"  ready in {time.time() - t0:.1f}s")

    frames_csv_path = output_dir / "frames.csv"
    save_frames = not args.no_save_frames
    records: List[dict] = []

    try:
        with open(frames_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "pos_idx", "frame_idx", "u_px", "v_px",
                "depth_m_corrected", "confidence",
                "x_old", "y_old", "z_old",
                "x_cal", "y_cal", "z_cal",
            ])
            csvfile.flush()

            for pos_idx in range(1, args.positions + 1):
                try:
                    action, xyz = prompt_for_position(pos_idx, args.positions)
                except KeyboardInterrupt:
                    print("\nInterrupted by user.")
                    break

                if action == "quit":
                    print("Quit requested — finalising artifacts.")
                    break
                if action == "skip" or xyz is None:
                    logger.info("Position %d skipped by user.", pos_idx)
                    continue

                try:
                    rec = collect_position(
                        camera=camera,
                        grounder=grounder,
                        prompt=args.prompt,
                        pos_idx=pos_idx,
                        truth=xyz,
                        n_frames=args.frames,
                        n_warmup=args.warmup,
                        save_frames=save_frames,
                        output_dir=output_dir,
                        frames_writer=writer,
                        csv_handle=csvfile,
                    )
                    records.append(rec)
                except RuntimeError as exc:
                    logger.error("Aborting collection: %s", exc)
                    break
                except KeyboardInterrupt:
                    print("\nInterrupted by user during collection.")
                    break
                except Exception:
                    logger.error(
                        "Unexpected error during position %d (continuing):\n%s",
                        pos_idx, traceback.format_exc(),
                    )
                    continue

                # Incremental write so a crash mid-run does not lose data.
                try:
                    write_artifacts(output_dir, records)
                except Exception:
                    logger.warning("Failed to write incremental artifacts:\n%s",
                                   traceback.format_exc())
    finally:
        try:
            camera.stop()
        except Exception:
            pass

    text = write_artifacts(output_dir, records)

    print("\n" + "=" * 65)
    print(text, end="")
    print("=" * 65)
    print(f"  summary    → {output_dir / 'summary.txt'}")
    print(f"  frames csv → {frames_csv_path}")
    print(f"  positions  → {output_dir / 'positions.json'}")
    if save_frames:
        print(f"  frames     → {output_dir}/frame_pos*_*.jpg")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Audit grounding bias against a candidate plumb_bob "
                     "calibration and tape-measured ground truth."),
    )
    parser.add_argument("--positions", type=int, default=3,
                        help="Number of physical positions to measure (default: 3).")
    parser.add_argument("--frames", type=int, default=10,
                        help="Frames per position (default: 10).")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup frames discarded per position (default: 5).")
    parser.add_argument("--prompt", type=str, default="red button",
                        help="Grounding prompt (default: \"red button\").")
    parser.add_argument("--interface", type=str, default="enx98fc84e68f1a",
                        help="Ethernet interface for DDS (default: enx98fc84e68f1a).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=("Output directory "
                              "(default: data/calibration_check/<UTC timestamp>)."))
    parser.add_argument("--no-save-frames", action="store_true",
                        help="Disable saving annotated per-frame JPEGs.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
