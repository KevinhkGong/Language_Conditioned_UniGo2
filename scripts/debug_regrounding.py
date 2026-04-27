"""
scripts/debug_grounding.py

Standalone grounding debug tool.
Captures one frame from Go2 camera, runs the grounder, saves annotated
image and detailed diagnostics.

Usage:
    python scripts/debug_grounding.py
    python scripts/debug_grounding.py --prompt "red button"
    python scripts/debug_grounding.py --num-frames 5
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from src.perception.grounding import VisualGrounder, Go2Camera


def annotate_frame(frame_rgb, result, prompt, success):
    """Overlay detection on frame."""
    vis = frame_rgb.copy()
    H, W = vis.shape[:2]

    # Header bar
    color = (0, 255, 0) if success else (0, 0, 255)
    status = "SUCCESS" if success else "FAIL"
    cv2.rectangle(vis, (0, 0), (W, 80), (40, 40, 40), -1)
    cv2.putText(vis, f"Prompt: '{prompt}'  |  {status}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if result is not None:
        conf = result.confidence
        cv2.putText(vis, f"Confidence: {conf:.3f}  (threshold: 0.50)",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if result.bbox is not None:
            x1, y1, x2, y2 = result.bbox.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            # Label
            label = f"{prompt} ({conf:.2f})"
            if result.depth_m is not None:
                label += f"  depth={result.depth_m:.2f}m"
            cv2.putText(vis, label, (x1, max(y1-10, 90)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if result.centroid_px is not None:
            cx, cy = result.centroid_px.astype(int)
            cv2.circle(vis, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(vis, f"({cx}, {cy})", (cx + 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Position in base frame
        if result.position_base is not None:
            pos = result.position_base
            text = f"Base: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m"
            cv2.putText(vis, text, (20, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(vis, "NO DETECTION", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)

    return vis


def visualize_mask(frame_rgb, mask):
    """Overlay segmentation mask."""
    vis = frame_rgb.copy()
    if mask is not None and mask.any():
        colored = np.zeros_like(vis)
        colored[mask] = [0, 255, 255]  # yellow
        vis = cv2.addWeighted(vis, 0.7, colored, 0.3, 0)
    return vis


def run_debug(prompt, num_frames, output_dir):
    """Capture N frames and save diagnostic output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Prompt: '{prompt}'")
    print(f"[DEBUG] Output: {output_dir}")

    print("[DEBUG] Initializing DDS…")
    ChannelFactoryInitialize(0, "enx98fc84e68f1a")

    print("[DEBUG] Loading camera…")
    camera = Go2Camera(already_initialized=True)

    print("[DEBUG] Loading grounder (SAM2 + GroundingDINO + DAV2)…")
    grounder = VisualGrounder()

    print(f"\n[DEBUG] Capturing {num_frames} frames…\n")

    for i in range(num_frames):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        print(f"--- Frame {i+1}/{num_frames} ---")
        frame = camera.get_frame()
        if frame is None:
            print("  [FAIL] Camera returned no frame!")
            continue

        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")
        print(f"  Frame mean: {frame.mean():.1f} (0=black, 255=white)")

        # Save raw frame (BGR for OpenCV)
        raw_path = output_dir / f"raw_{ts}.png"
        cv2.imwrite(str(raw_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Run grounding
        try:
            result = grounder.ground(frame, prompt=prompt)
        except Exception as e:
            print(f"  [EXCEPTION] Grounding raised: {e}")
            import traceback
            traceback.print_exc()
            continue

        if result is None:
            print("  [RESULT] None — no detection at all")
            print("  Possible causes:")
            print("    - Button out of frame / occluded")
            print("    - GroundingDINO found nothing matching prompt")
            print("    - Try different prompt wording")
            vis = frame.copy()
            cv2.putText(vis, "NO DETECTION", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            fail_path = output_dir / f"annotated_{ts}_FAIL.png"
            cv2.imwrite(str(fail_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            continue

        # We got a result
        print(f"  [RESULT] Detection found")
        print(f"    Confidence:    {result.confidence:.4f} (threshold: 0.50)")
        if result.bbox is not None:
            x1, y1, x2, y2 = result.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            print(f"    BBox:          [{x1}, {y1}, {x2}, {y2}]  (w={w}, h={h})")
        if result.centroid_px is not None:
            cx, cy = result.centroid_px.astype(int)
            print(f"    Centroid px:   ({cx}, {cy})")
        if result.depth_m is not None:
            print(f"    Depth:         {result.depth_m:.3f} m")
        else:
            print(f"    Depth:         None (below min_depth floor)")
        if result.position_camera is not None:
            pc = result.position_camera
            print(f"    Pos (camera):  [{pc[0]:+.3f}, {pc[1]:+.3f}, {pc[2]:+.3f}]")
        if result.position_base is not None:
            pb = result.position_base
            print(f"    Pos (base):    [{pb[0]:+.3f}, {pb[1]:+.3f}, {pb[2]:+.3f}]")
        else:
            print(f"    Pos (base):    None")

        success = (
            result is not None
            and result.confidence >= 0.5
            and result.position_base is not None
        )

        # Save annotated
        annotated = annotate_frame(frame, result, prompt, success)
        tag = "OK" if success else "FAIL"
        ann_path = output_dir / f"annotated_{ts}_{tag}.png"
        cv2.imwrite(str(ann_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # Save mask overlay
        if result.mask is not None:
            mask_vis = visualize_mask(frame, result.mask)
            mask_path = output_dir / f"mask_{ts}_{tag}.png"
            cv2.imwrite(str(mask_path), cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

        print(f"  Saved: {raw_path.name}, {ann_path.name}")
        print()

    print(f"\n[DEBUG] Done. Files saved in: {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="red button",
                    help="Grounding prompt (default: 'red button')")
    ap.add_argument("--num-frames", type=int, default=1,
                    help="How many frames to capture (default: 1)")
    ap.add_argument("--output-dir", default="data/debug/grounding",
                    help="Where to save debug outputs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    run_debug(args.prompt, args.num_frames, args.output_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)