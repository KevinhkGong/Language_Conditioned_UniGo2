"""
scripts/verify_grounding.py

Diagnostic script to verify that the Grounded SAM2 + Depth Anything V2
perception pipeline can reliably detect and localize a physical wall-mounted
button from the Go2X's head camera over DDS/Ethernet.

Camera: Go2X built-in head camera (RGB, 1920x1080, 120° FOV) via VideoClient.
Depth:  Depth Anything V2 Metric Indoor (ViTL checkpoint).

Setup:
    - Robot standing in Sport Mode, facing the wall button at ~0.5m distance
    - Button mounted on wall at roughly body height (~0.45m above ground)
    - Ethernet connected and configured (see below)

Usage:
    python scripts/verify_grounding.py
    python scripts/verify_grounding.py --frames 50 --warmup 10
    python scripts/verify_grounding.py --frames 30 --output-dir data/real/run2

Ethernet setup (run before this script):
    sudo ip addr flush dev enx98fc84e68f1a
    sudo ip addr add 192.168.123.99/24 dev enx98fc84e68f1a
    sudo ip link set enx98fc84e68f1a up
    conda activate env_go2
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from src.perception.grounding import Go2Camera, GroundingResult, VisualGrounder

# ── Prompts to evaluate ───────────────────────────────────────────────────────
PROMPTS = [
    "button",
    "red button",
    "round button",
    "push button",
]

# Detection is considered successful above this confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# Expected rough button position in base frame (for sanity check printout)
# Wall button at ~0.5m forward, centered laterally, ~0.15m above base link
# (18 inches above ground ≈ 0.457m; base link ~0.30m above ground → z ≈ 0.15m)
EXPECTED_BASE_POS = np.array([0.50, 0.00, 0.15])
POSITION_WARN_THRESHOLD_M = 0.30   # warn if estimate deviates more than this


# ── Colour palette for overlay (BGR) ─────────────────────────────────────────
MASK_COLOUR   = (0,   200, 100)   # green-ish
BOX_COLOUR    = (0,   165, 255)   # orange
TEXT_COLOUR   = (255, 255, 255)
WARN_COLOUR   = (0,   0,   255)   # red — used when position looks wrong


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_mask_overlay(image: np.ndarray, mask: np.ndarray,
                      colour: tuple = MASK_COLOUR, alpha: float = 0.4) -> np.ndarray:
    """Blend a boolean mask onto the image."""
    overlay = image.copy()
    overlay[mask > 0] = colour
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_result_annotation(image: np.ndarray, result: GroundingResult,
                            prompt: str, frame_idx: int) -> np.ndarray:
    """
    Draw bounding box, mask, centroid, and 3D position text onto a copy of
    the image.  Returns the annotated copy.
    """
    vis = image.copy()

    # Mask overlay
    if result.mask is not None:
        vis = draw_mask_overlay(vis, result.mask)

    # Bounding box
    if result.bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in result.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOUR, 2)

    # Centroid dot
    if result.centroid_px is not None:
        cx, cy = int(result.centroid_px[0]), int(result.centroid_px[1])
        cv2.circle(vis, (cx, cy), 5, BOX_COLOUR, -1)

    # Position sanity check
    pos_ok = True
    if result.position_base is not None:
        dev = np.linalg.norm(result.position_base - EXPECTED_BASE_POS)
        pos_ok = dev < POSITION_WARN_THRESHOLD_M

    txt_colour = TEXT_COLOUR if pos_ok else WARN_COLOUR

    # Text lines (drawn from top-left of image)
    lines = [
        f"Frame {frame_idx:03d}  |  prompt: \"{prompt}\"",
        f"Confidence : {result.confidence:.3f}",
        f"Depth (m)  : {result.depth_m:.3f}" if result.depth_m else "Depth (m)  : --",
    ]
    if result.position_camera is not None:
        pc = result.position_camera
        lines.append(f"Cam  [m]   : x={pc[0]:.3f}  y={pc[1]:.3f}  z={pc[2]:.3f}")
    if result.position_base is not None:
        pb = result.position_base
        warn_str = "  ⚠ DEVIANT" if not pos_ok else ""
        lines.append(f"Base [m]   : x={pb[0]:.3f}  y={pb[1]:.3f}  z={pb[2]:.3f}{warn_str}")

    y_cursor = 24
    for line in lines:
        cv2.putText(vis, line, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_colour, 1, cv2.LINE_AA)
        y_cursor += 22

    return vis


def draw_no_detection(image: np.ndarray, prompt: str,
                      frame_idx: int, reason: str = "no detection") -> np.ndarray:
    """Draw a 'no detection' banner on the image."""
    vis = image.copy()
    lines = [
        f"Frame {frame_idx:03d}  |  prompt: \"{prompt}\"",
        f"RESULT: {reason.upper()}",
    ]
    y_cursor = 24
    for line in lines:
        cv2.putText(vis, line, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WARN_COLOUR, 1, cv2.LINE_AA)
        y_cursor += 22
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Main verification routine
# ─────────────────────────────────────────────────────────────────────────────

def run_verification(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Output directories ────────────────────────────────────────────────────
    output_dir   = Path(args.output_dir)
    samples_dir  = output_dir / "grounding_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    json_path    = output_dir / f"grounding_verification_{timestamp}.json"

    print("\n" + "=" * 65)
    print("  Grounded SAM2 + Depth Anything V2 — Wall Button Verification")
    print("=" * 65)
    print(f"  Frames to capture : {args.frames}")
    print(f"  Warmup frames     : {args.warmup}")
    print(f"  Prompts           : {PROMPTS}")
    print(f"  Confidence thresh : {CONFIDENCE_THRESHOLD}")
    print(f"  Output dir        : {output_dir}")
    print("=" * 65 + "\n")

    # ── Load models (slow — do once) ─────────────────────────────────────────
    print("[1/3] Loading GroundingDINO + SAM2 + Depth Anything V2 …")
    t0 = time.time()
    grounder = VisualGrounder()
    print(f"      Models loaded in {time.time() - t0:.1f}s\n")

    # ── Init DDS and open Go2 camera ──────────────────────────────────────────
    print("[2/3] Connecting to Go2 head camera via DDS …")
    ChannelFactoryInitialize(0, args.interface)
    camera = Go2Camera(already_initialized=True)
    print("      Camera connected.\n")

    # ── Warm-up: discard first N frames for auto-exposure ────────────────────
    print(f"[3/3] Warming up ({args.warmup} frames discarded) …")
    for _ in range(args.warmup):
        camera.get_frame()
    print("      Warm-up complete.\n")

    # ── Per-frame, per-prompt data store ──────────────────────────────────────
    # Structure:  per_frame[frame_idx][prompt] = { ... }
    per_frame = []

    # OpenCV window
    cv2.namedWindow("Grounding Verification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grounding Verification", 1280, 480)

    print("Running detection loop. Press 'q' to stop early.\n")
    print(f"{'Frame':>5}  {'Prompt':<22}  {'Conf':>6}  {'Depth':>7}  "
          f"{'Base x':>8}  {'Base y':>8}  {'Base z':>8}  Status")
    print("-" * 85)

    for frame_idx in range(args.frames):
        # Grab one RGB frame from Go2 head camera
        frame_data = camera.get_frame()
        if frame_data is None:
            print(f"  [{frame_idx:03d}] Camera returned None — skipping.")
            continue

        rgb_image = frame_data   # (H, W, 3) uint8 BGR

        frame_record = {
            "frame_idx": frame_idx,
            "prompts": {}
        }

        # Build a side-by-side display of all prompt results for this frame
        annotated_strips = []

        for prompt in PROMPTS:
            result = None
            status = "ok"

            try:
                result = grounder.ground(rgb_image, prompt)
            except Exception as exc:
                status = f"exception: {exc}"

            # Determine success
            detected = (
                result is not None
                and result.confidence is not None
                and result.confidence >= CONFIDENCE_THRESHOLD
            )

            if detected:
                strip = draw_result_annotation(rgb_image, result, prompt, frame_idx)
            else:
                reason = "low conf" if (result and result.confidence is not None) else "no det"
                strip  = draw_no_detection(rgb_image, prompt, frame_idx, reason)

            # Add prompt label bar at bottom of strip
            h, w = strip.shape[:2]
            bar = np.zeros((28, w, 3), dtype=np.uint8)
            label = f"\"{prompt}\""
            cv2.putText(bar, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1, cv2.LINE_AA)
            strip = np.vstack([strip, bar])
            annotated_strips.append(strip)

            # ── Log record ───────────────────────────────────────────────────
            prompt_record = {
                "detected": detected,
                "confidence": float(result.confidence) if (result and result.confidence) else None,
                "depth_m": float(result.depth_m) if (result and result.depth_m) else None,
                "centroid_px": [float(v) for v in result.centroid_px] if (result and result.centroid_px is not None) else None,
                "position_camera": [float(v) for v in result.position_camera] if (result and result.position_camera is not None) else None,
                "position_base": [float(v) for v in result.position_base] if (result and result.position_base is not None) else None,
                "status": status,
            }
            frame_record["prompts"][prompt] = prompt_record

            # ── Console row ──────────────────────────────────────────────────
            conf_str  = f"{result.confidence:.3f}" if (result and result.confidence) else "  ---"
            depth_str = f"{result.depth_m:.3f}m"   if (result and result.depth_m)    else "    ---"
            if result and result.position_base is not None:
                pb = result.position_base
                bx, by, bz = f"{pb[0]:.3f}", f"{pb[1]:.3f}", f"{pb[2]:.3f}"
            else:
                bx = by = bz = "    ---"
            det_str = "✓ det" if detected else "✗ miss"
            print(f"  {frame_idx:3d}  {prompt:<22}  {conf_str:>6}  {depth_str:>7}  "
                  f"{bx:>8}  {by:>8}  {bz:>8}  {det_str}")

        per_frame.append(frame_record)

        # ── Real-time display: tile all 4 prompt strips side by side ─────────
        # Resize strips to the same height before hstacking
        target_h = annotated_strips[0].shape[0]
        resized = []
        for strip in annotated_strips:
            if strip.shape[0] != target_h:
                strip = cv2.resize(strip, (strip.shape[1], target_h))
            resized.append(strip)

        tiled = np.hstack(resized)
        # Scale down to fit screen if needed
        screen_w = 1280
        if tiled.shape[1] > screen_w:
            scale = screen_w / tiled.shape[1]
            tiled = cv2.resize(tiled,
                               (screen_w, int(tiled.shape[0] * scale)))

        cv2.imshow("Grounding Verification", tiled)

        # ── Save annotated frame for best-confidence prompt ───────────────────
        best_prompt = max(
            PROMPTS,
            key=lambda p: (
                per_frame[-1]["prompts"][p]["confidence"] or 0.0
            )
        )
        best_strip = annotated_strips[PROMPTS.index(best_prompt)]
        save_path  = samples_dir / f"frame_{frame_idx:03d}_best.jpg"
        cv2.imwrite(str(save_path), best_strip)

        # Also save all 4 strips for frames that are multiples of 10
        if frame_idx % 10 == 0:
            all_path = samples_dir / f"frame_{frame_idx:03d}_all.jpg"
            cv2.imwrite(str(all_path), tiled)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n  [User pressed 'q' — stopping early.]\n")
            break

    cv2.destroyAllWindows()
    camera.stop()

    # ── Aggregate statistics ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Aggregate Statistics (per prompt)")
    print("=" * 65)

    stats_per_prompt = {}
    for prompt in PROMPTS:
        confs, depths = [], []
        pos_base_list = []
        n_detected, n_total = 0, 0

        for fr in per_frame:
            rec = fr["prompts"].get(prompt, {})
            n_total += 1
            if rec.get("detected"):
                n_detected += 1
                if rec["confidence"] is not None:
                    confs.append(rec["confidence"])
                if rec["depth_m"] is not None:
                    depths.append(rec["depth_m"])
                if rec["position_base"] is not None:
                    pos_base_list.append(rec["position_base"])

        detection_rate = n_detected / n_total if n_total > 0 else 0.0
        pos_arr = np.array(pos_base_list) if pos_base_list else None

        entry = {
            "n_total":        n_total,
            "n_detected":     n_detected,
            "detection_rate": detection_rate,
            "mean_confidence":  float(np.mean(confs))  if confs else None,
            "std_confidence":   float(np.std(confs))   if confs else None,
            "mean_depth_m":     float(np.mean(depths)) if depths else None,
            "std_depth_m":      float(np.std(depths))  if depths else None,
            "mean_position_base": pos_arr.mean(axis=0).tolist() if pos_arr is not None else None,
            "std_position_base":  pos_arr.std(axis=0).tolist()  if pos_arr is not None else None,
        }
        stats_per_prompt[prompt] = entry

        # Console summary
        print(f"\n  Prompt : \"{prompt}\"")
        print(f"    Detection rate : {n_detected}/{n_total}  ({detection_rate*100:.1f}%)")
        if confs:
            print(f"    Confidence     : mean={np.mean(confs):.3f}  std={np.std(confs):.3f}")
        if depths:
            print(f"    Depth          : mean={np.mean(depths):.3f}m  std={np.std(depths):.3f}m")
        if pos_arr is not None:
            m = pos_arr.mean(axis=0)
            s = pos_arr.std(axis=0)
            print(f"    Base pos mean  : x={m[0]:.3f}  y={m[1]:.3f}  z={m[2]:.3f}")
            print(f"    Base pos std   : x={s[0]:.3f}  y={s[1]:.3f}  z={s[2]:.3f}")
            dev = np.linalg.norm(m - EXPECTED_BASE_POS)
            flag = "  ⚠ DEVIANT (>{:.2f}m from expected)".format(POSITION_WARN_THRESHOLD_M) \
                   if dev > POSITION_WARN_THRESHOLD_M else "  ✓ within expected range"
            print(f"    Deviation from expected pos: {dev:.3f}m{flag}")

    # Best prompt recommendation
    best = max(PROMPTS,
               key=lambda p: stats_per_prompt[p]["detection_rate"])
    print(f"\n  ► Recommended prompt for run_baseline1.py : \"{best}\"")
    print(f"    (highest detection rate: "
          f"{stats_per_prompt[best]['detection_rate']*100:.1f}%)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "timestamp":        timestamp,
        "args": {
            "frames":      args.frames,
            "warmup":      args.warmup,
            "output_dir":  str(output_dir),
        },
        "prompts_tested":   PROMPTS,
        "expected_base_pos": EXPECTED_BASE_POS.tolist(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "per_frame":        per_frame,
        "stats_per_prompt": stats_per_prompt,
        "recommended_prompt": best,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  JSON report saved → {json_path}")
    print(f"  Annotated frames  → {samples_dir}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify Grounded SAM2 + Depth Anything V2 wall button detection on Go2X."
    )
    parser.add_argument(
        "--frames", type=int, default=30,
        help="Number of frames to capture and evaluate (default: 30)."
    )
    parser.add_argument(
        "--warmup", type=int, default=8,
        help="Number of initial frames to discard for auto-exposure warmup (default: 8)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/real",
        help="Directory for JSON report and annotated frame samples (default: data/real)."
    )
    parser.add_argument(
        "--interface", type=str, default="enx98fc84e68f1a",
        help="Ethernet interface for DDS (default: enx98fc84e68f1a)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_verification(args)