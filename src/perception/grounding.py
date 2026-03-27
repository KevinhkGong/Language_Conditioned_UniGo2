"""
src/perception/grounding.py

Stage B: Open-vocabulary visual grounding using Grounded SAM2.

Pipeline:
    RealSense RGB-D frame
        → Grounding DINO  (text prompt → bounding box)
        → SAM2            (bounding box → segmentation mask)
        → Depth unproject (mask centroid + depth → 3D camera frame)
        → Extrinsic xform (camera frame → robot base frame)
        → GroundingResult

Usage:
    from src.perception.grounding import VisualGrounder, GroundingResult
    from src.language.intent_parser import TaskSpec

    grounder = VisualGrounder()
    result = grounder.ground(rgb, depth, task_spec)
    print(result.position_base)   # (x, y, z) in robot base frame

Requirements (env_go2):
    - sam2, groundingdino, pyrealsense2, torch, opencv-python
    - SAM2 checkpoint:    ~/Robotics/sam2/checkpoints/sam2.1_hiera_small.pt
    - GroundingDINO ckpt: ~/Robotics/weights/groundingdino_swint_ogc.pth
    - GroundingDINO cfg:  ~/Robotics/weights/GroundingDINO_SwinT_OGC.py
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import pyrealsense2 as rs

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────

_HOME = os.path.expanduser("~")

SAM2_CHECKPOINT = os.path.join(
    _HOME, "Robotics/sam2/checkpoints/sam2.1_hiera_small.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

GDINO_CHECKPOINT = os.path.join(
    _HOME, "Robotics/weights/groundingdino_swint_ogc.pth"
)
GDINO_CONFIG = os.path.join(
    _HOME, "Robotics/weights/GroundingDINO_SwinT_OGC.py"
)

# ──────────────────────────────────────────────
# Go2X RealSense D435i — nominal camera intrinsics
# (will be overridden by live intrinsics from RealSense SDK at runtime)
# ──────────────────────────────────────────────

NOMINAL_FX = 615.0   # focal length x (pixels)
NOMINAL_FY = 615.0   # focal length y (pixels)
NOMINAL_CX = 320.0   # principal point x
NOMINAL_CY = 240.0   # principal point y

# ──────────────────────────────────────────────
# Go2X RealSense — nominal extrinsic transform
# Camera frame → Robot base frame
#
# Coordinate conventions:
#   Camera:    x=right, y=down, z=forward
#   Robot base: x=forward, y=left, z=up
#
# Nominal mount position from Go2X URDF:
#   Camera is mounted ~0.28m forward of base origin,
#   ~0.08m above ground (z), tilted slightly downward (~10°)
#
# This is a hardcoded approximation — sufficient for ~1m range.
# Replace with calibrated values after checkerboard calibration.
# ──────────────────────────────────────────────

# Rotation: camera → base frame
# Camera z (forward) → base x (forward)
# Camera x (right)   → base -y (right = negative left)
# Camera y (down)    → base -z (down = negative up)
# With ~10° downward tilt (pitch)
_PITCH = np.radians(-10.0)   # camera tilted 10° down
_R_CAM_TO_BASE = np.array([
    [0,           -np.sin(_PITCH),  np.cos(_PITCH)],   # base x = cam z rotated
    [-1,           0,               0             ],   # base y = -cam x
    [0,           -np.cos(_PITCH), -np.sin(_PITCH)],   # base z = -cam y rotated
], dtype=np.float32)

# Translation: camera origin in base frame (metres)
_T_CAM_TO_BASE = np.array([0.28, 0.0, 0.08], dtype=np.float32)


# ──────────────────────────────────────────────
# GroundingDINO thresholds
# ──────────────────────────────────────────────

BOX_THRESHOLD  = 0.35   # minimum score to keep a detection
TEXT_THRESHOLD = 0.25   # minimum token score
MAX_DEPTH_M    = 3.0    # ignore depth readings beyond this (likely noise)
MIN_DEPTH_M    = 0.05   # ignore depth readings below this (too close)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class GroundingResult:
    """
    Full output of one grounding call.

    Attributes:
        found:              True if target was detected.
        target_description: Text prompt that was searched for.
        mask:               (H, W) bool array — SAM2 segmentation mask.
        bbox_xyxy:          (4,) float — bounding box in pixel coords [x1,y1,x2,y2].
        centroid_px:        (u, v) pixel coordinates of mask centroid.
        depth_m:            Median depth of mask region in metres.
        position_camera:    (3,) xyz in RealSense camera frame (x=right, y=down, z=fwd).
        position_base:      (3,) xyz in robot base frame (x=fwd, y=left, z=up).
        confidence:         GroundingDINO detection score (0–1).
    """
    found:               bool
    target_description:  str
    mask:                Optional[np.ndarray]   = None
    bbox_xyxy:           Optional[np.ndarray]   = None
    centroid_px:         Optional[Tuple[int,int]] = None
    depth_m:             Optional[float]         = None
    position_camera:     Optional[np.ndarray]    = None
    position_base:       Optional[np.ndarray]    = None
    confidence:          float                   = 0.0


# ──────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────

class VisualGrounder:
    """
    Grounded SAM2 pipeline for open-vocabulary target localization.

    Loads Grounding DINO + SAM2 once at init, then accepts RGB-D frames
    and text prompts at runtime via ground().

    Args:
        sam2_checkpoint:  Path to SAM2 .pt checkpoint file.
        sam2_config:      SAM2 config name (relative, resolved by hydra).
        gdino_checkpoint: Path to GroundingDINO .pth checkpoint.
        gdino_config:     Path to GroundingDINO config .py file.
        device:           'cuda' or 'cpu'.
    """

    def __init__(
        self,
        sam2_checkpoint:  str = SAM2_CHECKPOINT,
        sam2_config:      str = SAM2_CONFIG,
        gdino_checkpoint: str = GDINO_CHECKPOINT,
        gdino_config:     str = GDINO_CONFIG,
        device:           str = "cuda",
    ):
        self.device = device
        self._fx = NOMINAL_FX
        self._fy = NOMINAL_FY
        self._cx = NOMINAL_CX
        self._cy = NOMINAL_CY

        logger.info("Loading Grounding DINO...")
        self._gdino = load_model(gdino_config, gdino_checkpoint)
        self._gdino.eval()
        logger.info("Grounding DINO loaded.")

        logger.info("Loading SAM2...")
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self._sam2 = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 loaded.")

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """
        Update camera intrinsics from live RealSense stream.
        Call this once after starting the RealSense pipeline.

        Args:
            fx, fy: focal lengths in pixels.
            cx, cy: principal point in pixels.
        """
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        logger.debug(f"Intrinsics updated: fx={fx:.1f} fy={fy:.1f} "
                     f"cx={cx:.1f} cy={cy:.1f}")

    # ──────────────────────────────────────────────
    # Main grounding entry point
    # ──────────────────────────────────────────────

    def ground(
        self,
        rgb:   np.ndarray,
        depth: np.ndarray,
        text_prompt: str,
    ) -> GroundingResult:
        """
        Localize a target described by text_prompt in an RGB-D frame.

        Args:
            rgb:          (H, W, 3) uint8 BGR image from RealSense/OpenCV.
            depth:        (H, W) float32 depth image in metres.
            text_prompt:  Description of target e.g. "red button", "yellow cylinder".

        Returns:
            GroundingResult with all fields populated if target found,
            or GroundingResult(found=False) if not detected.
        """
        # ── 1. Grounding DINO: text → bounding box ───────────────────
        rgb_pil  = self._bgr_to_pil(rgb)
        boxes, scores, _ = self._run_gdino(rgb_pil, text_prompt)

        if boxes is None or len(boxes) == 0:
            logger.warning(f"Grounding DINO: no detection for '{text_prompt}'")
            return GroundingResult(found=False, target_description=text_prompt)

        # Take highest-confidence detection
        best_idx   = scores.argmax().item()
        best_score = scores[best_idx].item()
        best_box   = boxes[best_idx]   # (cx, cy, w, h) normalised

        # Convert to pixel xyxy
        H, W = rgb.shape[:2]
        bbox_xyxy = self._cxcywh_to_xyxy(best_box, W, H)

        logger.debug(f"DINO detection: score={best_score:.3f} "
                     f"box={bbox_xyxy.astype(int).tolist()}")

        # ── 2. SAM2: bounding box → segmentation mask ────────────────
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        mask, sam_score = self._run_sam2(rgb_rgb, bbox_xyxy)

        if mask is None:
            logger.warning("SAM2 returned no mask.")
            return GroundingResult(found=False, target_description=text_prompt)

        # ── 3. Mask centroid ─────────────────────────────────────────
        centroid = self._mask_centroid(mask)
        if centroid is None:
            logger.warning("Mask centroid could not be computed (empty mask).")
            return GroundingResult(found=False, target_description=text_prompt)

        u, v = centroid

        # ── 4. Depth at centroid ─────────────────────────────────────
        depth_m = self._sample_depth(depth, mask)
        if depth_m is None:
            logger.warning("No valid depth in mask region.")
            return GroundingResult(found=False, target_description=text_prompt)

        # ── 5. Unproject to 3D camera frame ──────────────────────────
        pos_cam = self._unproject(u, v, depth_m)

        # ── 6. Transform to robot base frame ─────────────────────────
        pos_base = _R_CAM_TO_BASE @ pos_cam + _T_CAM_TO_BASE

        logger.info(
            f"Grounding result: prompt='{text_prompt}' "
            f"score={best_score:.3f} depth={depth_m:.3f}m "
            f"pos_base=({pos_base[0]:.3f}, {pos_base[1]:.3f}, {pos_base[2]:.3f})"
        )

        return GroundingResult(
            found=True,
            target_description=text_prompt,
            mask=mask,
            bbox_xyxy=bbox_xyxy,
            centroid_px=(u, v),
            depth_m=depth_m,
            position_camera=pos_cam,
            position_base=pos_base,
            confidence=best_score,
        )

    # ──────────────────────────────────────────────
    # Internal: Grounding DINO
    # ──────────────────────────────────────────────

    def _run_gdino(self, rgb_pil, text_prompt: str):
        """Run Grounding DINO and return (boxes, scores, phrases)."""
        from groundingdino.util.inference import load_image
        import torchvision.transforms as T

        # Use GroundingDINO's own preprocessing transform
        transform = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(rgb_pil).to(self.device)

        # Ensure prompt ends with a period (DINO requirement)
        caption = text_prompt.strip()
        if not caption.endswith("."):
            caption += "."

        with torch.no_grad():
            boxes, scores, phrases = predict(
                model=self._gdino,
                image=image_tensor,
                caption=caption,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

        return boxes, scores, phrases

    # ──────────────────────────────────────────────
    # Internal: SAM2
    # ──────────────────────────────────────────────

    def _run_sam2(
        self,
        rgb_rgb: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Run SAM2 with a bounding box prompt.

        Args:
            rgb_rgb:   (H, W, 3) uint8 RGB image.
            bbox_xyxy: (4,) float bounding box [x1, y1, x2, y2] in pixels.

        Returns:
            (mask, score) — mask is (H, W) bool, score is float.
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._sam2.set_image(rgb_rgb)
            masks, scores, _ = self._sam2.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_xyxy[None, :],   # SAM2 expects (1, 4)
                multimask_output=False,
            )

        if masks is None or len(masks) == 0:
            return None, 0.0

        # masks shape: (N, H, W) — take first (multimask_output=False)
        mask  = masks[0].astype(bool)
        score = float(scores[0]) if scores is not None else 1.0
        return mask, score

    # ──────────────────────────────────────────────
    # Internal: geometry helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _bgr_to_pil(bgr: np.ndarray):
        """Convert BGR numpy array to PIL RGB image."""
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    @staticmethod
    def _cxcywh_to_xyxy(box: torch.Tensor, W: int, H: int) -> np.ndarray:
        """
        Convert normalised (cx, cy, w, h) to pixel (x1, y1, x2, y2).
        """
        cx, cy, w, h = box.cpu().numpy()
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Compute pixel centroid of a boolean mask using image moments.

        Returns:
            (u, v) integer pixel coordinates, or None if mask is empty.
        """
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            return None
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        return (u, v)

    @staticmethod
    def _sample_depth(depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """
        Compute median depth within the mask region.
        Filters out invalid readings (zeros, too close, too far).

        Args:
            depth: (H, W) float32 depth in metres.
            mask:  (H, W) bool mask.

        Returns:
            Median depth in metres, or None if no valid pixels.
        """
        values = depth[mask]
        valid  = values[(values > MIN_DEPTH_M) & (values < MAX_DEPTH_M)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def _unproject(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """
        Unproject a pixel (u, v) at depth d to a 3D point in camera frame.

        Camera frame convention: x=right, y=down, z=forward.

        Args:
            u, v:    Pixel coordinates.
            depth_m: Depth in metres.

        Returns:
            (3,) float32 array [Xc, Yc, Zc].
        """
        Xc = (u - self._cx) / self._fx * depth_m
        Yc = (v - self._cy) / self._fy * depth_m
        Zc = depth_m
        return np.array([Xc, Yc, Zc], dtype=np.float32)


# ──────────────────────────────────────────────
# RealSense helper
# ──────────────────────────────────────────────

class RealSenseCamera:
    """
    Thin wrapper around the RealSense D435i for the Go2X.

    Provides aligned RGB + depth frames and camera intrinsics.

    Usage:
        cam = RealSenseCamera()
        cam.start()
        rgb, depth = cam.get_frame()
        cam.stop()
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._pipeline = rs.pipeline()
        self._align    = None
        self._profile  = None

    def start(self) -> None:
        """Start the RealSense pipeline and enable depth+color streams."""
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)

        self._profile  = self._pipeline.start(config)
        self._align    = rs.align(rs.stream.color)

        # Get and return live intrinsics
        color_stream   = self._profile.get_stream(rs.stream.color)
        intrinsics     = color_stream.as_video_stream_profile().get_intrinsics()
        logger.info(
            f"RealSense started: {self.width}x{self.height}@{self.fps}fps | "
            f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
            f"cx={intrinsics.ppx:.1f} cy={intrinsics.ppy:.1f}"
        )
        self._intrinsics = intrinsics

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) from live camera profile."""
        i = self._intrinsics
        return i.fx, i.fy, i.ppx, i.ppy

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture one aligned RGB-D frame.

        Returns:
            rgb:   (H, W, 3) uint8 BGR image.
            depth: (H, W) float32 depth in metres.
        """
        frames        = self._pipeline.wait_for_frames()
        aligned       = self._align.process(frames)
        color_frame   = aligned.get_color_frame()
        depth_frame   = aligned.get_depth_frame()

        rgb   = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # RealSense depth is in millimetres (uint16) — convert to metres
        depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale

        return rgb, depth

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        self._pipeline.stop()
        logger.info("RealSense stopped.")


# ──────────────────────────────────────────────
# Visualisation helper (for debugging)
# ──────────────────────────────────────────────

def visualize_result(rgb: np.ndarray, result: GroundingResult) -> np.ndarray:
    """
    Draw bounding box, mask overlay, and centroid on an RGB image.

    Args:
        rgb:    (H, W, 3) BGR image.
        result: GroundingResult from VisualGrounder.ground().

    Returns:
        Annotated BGR image.
    """
    vis = rgb.copy()
    if not result.found:
        cv2.putText(vis, f"Not found: {result.target_description}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis

    # Mask overlay (green, semi-transparent)
    if result.mask is not None:
        overlay       = vis.copy()
        overlay[result.mask] = (0, 200, 0)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

    # Bounding box
    if result.bbox_xyxy is not None:
        x1, y1, x2, y2 = result.bbox_xyxy.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Centroid
    if result.centroid_px is not None:
        u, v = result.centroid_px
        cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)

    # Label
    label = (f"{result.target_description} "
             f"d={result.depth_m:.2f}m "
             f"conf={result.confidence:.2f}")
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 3D position in base frame
    if result.position_base is not None:
        pos = result.position_base
        pos_label = f"base: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"
        cv2.putText(vis, pos_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return vis


# ──────────────────────────────────────────────
# CLI smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """
    Live smoke test — streams from RealSense and runs Grounded SAM2.

    Usage:
        python src/perception/grounding.py --prompt "red button"
        python src/perception/grounding.py --prompt "yellow cylinder" --show
        python src/perception/grounding.py --image test.jpg --prompt "red button"
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Grounding smoke test")
    parser.add_argument("--prompt", type=str, default="red button",
                        help="Text prompt for grounding")
    parser.add_argument("--show", action="store_true",
                        help="Display results in an OpenCV window")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a static image (skips RealSense, depth=1m)")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to process (live mode)")
    args = parser.parse_args()

    print(f"\nLoading models (this takes ~10s)...")
    grounder = VisualGrounder()
    print("Models loaded.\n")

    if args.image:
        # ── Static image mode ────────────────────────────────────────
        print(f"Processing image: {args.image}")
        rgb   = cv2.imread(args.image)
        depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

        result = grounder.ground(rgb, depth, args.prompt)
        print(f"\nResult:")
        print(f"  found:         {result.found}")
        print(f"  confidence:    {result.confidence:.3f}")
        print(f"  centroid_px:   {result.centroid_px}")
        print(f"  depth_m:       {result.depth_m}")
        print(f"  position_base: {result.position_base}")

        if args.show:
            vis = visualize_result(rgb, result)
            # Before every cv2.imshow call, add:
            h, w = vis.shape[:2]
            scale = min(1280 / w, 720 / h, 1.0)
            if scale < 1.0:
                vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
            cv2.imshow("Grounding result", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        # ── Live RealSense mode ───────────────────────────────────────
        print("Starting RealSense...")
        cam = RealSenseCamera()
        cam.start()

        fx, fy, cx, cy = cam.get_intrinsics()
        grounder.update_intrinsics(fx, fy, cx, cy)

        print(f"Running grounding for {args.frames} frames. Prompt: '{args.prompt}'")
        for i in range(args.frames):
            rgb, depth = cam.get_frame()
            result     = grounder.ground(rgb, depth, args.prompt)

            print(f"\nFrame {i+1}/{args.frames}:")
            print(f"  found:         {result.found}")
            print(f"  confidence:    {result.confidence:.3f}")
            print(f"  depth_m:       {result.depth_m}")
            print(f"  position_base: {result.position_base}")

            if args.show:
                vis = visualize_result(rgb, result)
                cv2.imshow("Grounding", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cam.stop()
        if args.show:
            cv2.destroyAllWindows()

    print("\nSmoke test complete.")