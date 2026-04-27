"""
scripts/capture_grounding_debug.py

One-shot grounding debug capture: grabs one frame from the Go2 head camera,
runs VisualGrounder, and saves raw + annotated frames to data/debug/grounding/.

Usage:
    python scripts/capture_grounding_debug.py
    python scripts/capture_grounding_debug.py --prompt "yellow panel"
    python scripts/capture_grounding_debug.py --no-camera-init   # skip ChannelFactoryInitialize
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_NETWORK_INTERFACE   = "enx98fc84e68f1a"
_CONFIDENCE_THRESHOLD = 0.5
_WARMUP_FRAMES        = 8


def _annotate(frame: np.ndarray, result, prompt: str) -> np.ndarray:
    vis = frame.copy()

    if result.mask is not None:
        overlay = vis.copy()
        overlay[result.mask] = (0, 200, 100)
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    if result.bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in result.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)

    if result.centroid_px is not None:
        cx, cy = int(result.centroid_px[0]), int(result.centroid_px[1])
        cv2.circle(vis, (cx, cy), 6, (0, 165, 255), -1)

    conf_str  = f"{result.confidence:.3f}" if result.confidence is not None else "--"
    depth_str = f"{result.depth_m:.3f}m"   if result.depth_m  is not None else "--"
    lines = [f"{prompt}  conf={conf_str}  depth={depth_str}"]
    if result.position_base is not None:
        pb = result.position_base
        lines.append(f"base x={pb[0]:.3f} y={pb[1]:.3f} z={pb[2]:.3f} m")

    y = 28
    for ln in lines:
        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="One-shot grounding debug capture from the Go2 head camera."
    )
    parser.add_argument("--prompt", type=str, default="red button",
                        help="Text prompt for VisualGrounder (default: 'red button')")
    parser.add_argument("--interface", type=str, default=_NETWORK_INTERFACE,
                        help="Ethernet interface for DDS (default: enx98fc84e68f1a)")
    parser.add_argument("--no-camera-init", action="store_true",
                        help="Skip ChannelFactoryInitialize (use when DDS is already up)")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / "data" / "debug" / "grounding"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts             = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path       = out_dir / f"raw_{ts}.png"
    annotated_path = out_dir / f"annotated_{ts}.png"

    # DDS init — skip if caller already called ChannelFactoryInitialize
    if not args.no_camera_init:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(0, args.interface)

    from src.perception.grounding import Go2Camera, VisualGrounder

    print(f"[capture] Initialising camera on {args.interface} …")
    camera = Go2Camera(already_initialized=True)

    print(f"[capture] Warming up ({_WARMUP_FRAMES} frames discarded) …")
    for _ in range(_WARMUP_FRAMES):
        camera.get_frame()
        time.sleep(0.05)

    print("[capture] Grabbing frame …")
    frame = camera.get_frame()
    camera.stop()

    if frame is None:
        print("[capture] ERROR: get_frame() returned None — check camera connection.")
        sys.exit(1)

    # get_frame() already returns BGR (cv2.IMREAD_COLOR); write directly.
    cv2.imwrite(str(raw_path), frame)
    print(f"[capture] Raw frame saved  → {raw_path}")

    print("[capture] Loading VisualGrounder (first run loads model weights ~30 s) …")
    grounder = VisualGrounder()

    print(f"[capture] Running grounder with prompt: \"{args.prompt}\" …")
    result = grounder.ground(frame, args.prompt)

    if result is None:
        print("[capture] No detection found.")
        print(f"[capture] Raw frame only → {raw_path}")
        sys.exit(0)

    conf = result.confidence if result.confidence is not None else float("nan")
    pos  = result.position_base

    threshold_label = "ABOVE" if conf >= _CONFIDENCE_THRESHOLD else "BELOW"
    print(f"\n[capture] confidence   : {conf:.3f}  "
          f"({threshold_label} threshold {_CONFIDENCE_THRESHOLD})")
    if result.depth_m is not None:
        print(f"[capture] depth        : {result.depth_m:.3f} m")
    else:
        print("[capture] depth        : None (target too close for reliable estimate)")
    if pos is not None:
        print(f"[capture] position_base: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}  m")
    else:
        print("[capture] position_base: None")

    # bbox is None but confidence is finite → low-confidence overlay, no box
    if result.bbox is None:
        annotated = frame.copy()
        msg = f"low confidence: {conf:.3f}"
        cv2.putText(annotated, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 100, 255), 1, cv2.LINE_AA)
    else:
        annotated = _annotate(frame, result, args.prompt)

    cv2.imwrite(str(annotated_path), annotated)
    print(f"[capture] Annotated frame → {annotated_path}")


if __name__ == "__main__":
    main()
