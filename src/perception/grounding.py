"""
src/perception/grounding.py

Stage B — Open-Vocabulary Visual Grounding for the Go2X robot.

Pipeline:
    Text prompt + Go2 head camera RGB frame
        → GroundingDINO  : text  → bounding box
        → SAM2           : bbox  → pixel-level segmentation mask
        → Depth Anything V2 (metric indoor) : RGB → metric depth map (metres)
        → Unproject      : mask centroid + depth → 3D point in camera frame
        → Extrinsic xform: camera frame → robot base frame
        → GroundingResult

Camera: Go2X built-in head camera (RGB only, 1920×1080, 120° FOV)
        accessed via unitree_sdk2py VideoClient over DDS/Ethernet.

Depth:  Depth Anything V2 Metric Indoor (Hypersim-ViTL checkpoint).
        No RealSense or LiDAR required.

URDF-derived camera mounting (base frame origin):
    forward : +0.327 m
    lateral :  0.000 m
    height  : +0.043 m above base link
    pitch   :  0.0 rad  (camera faces horizontally)

Approximate intrinsics from 120° HFOV + 1920×1080:
    fx = fy = 960 / tan(60°) ≈ 554.3 px
    cx = 960, cy = 540

Ethernet setup (required every session before importing this module):
    sudo ip addr flush dev enx98fc84e68f1a
    sudo ip addr add 192.168.123.99/24 dev enx98fc84e68f1a
    sudo ip link set enx98fc84e68f1a up

Usage:
    # Quick static test (saves annotated image)
    python src/perception/grounding.py --image test.jpg --prompt "red button" --show

    # Live camera feed from Go2
    python src/perception/grounding.py --prompt "red button" --show --frames 30
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── Depth Anything V2 — metric indoor ────────────────────────────────────────
_DA2_REPO = Path.home() / "Robotics" / "Depth-Anything-V2" / "metric_depth"
if str(_DA2_REPO) not in sys.path:
    sys.path.insert(0, str(_DA2_REPO))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

# ── GroundingDINO ─────────────────────────────────────────────────────────────
import groundingdino.datasets.transforms as T  # noqa: E402
from groundingdino.models import build_model  # noqa: E402
from groundingdino.util.misc import clean_state_dict  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import get_phrases_from_posmap  # noqa: E402

# ── SAM2 ──────────────────────────────────────────────────────────────────────
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

# ── Camera intrinsics ────────────────────────────────────────────────────────
#
# Calibrated camera intrinsics (plumb_bob model, obtained April 2026).
# Validated against tape-measured ground truth at 3 positions:
# residual error ~2 cm laterally. Replaces the URDF-derived approximation
# below (fx=fy=554.3, cx=960, cy=540, no distortion) which gave
# ~12 cm error and ~2.37× lateral over-scaling for off-centre detections.
#
# CAMERA_D follows OpenCV's plumb_bob ordering: [k1, k2, p1, p2, k3].
CAMERA_K = np.array([
    [1310.77826,    0.     , 1018.71143],
    [   0.     , 1320.25059,  637.37672],
    [   0.     ,    0.     ,    1.     ],
], dtype=np.float64)
CAMERA_D = np.array(
    [-0.415971, 0.158898, -0.015395, -0.008031, 0.000000],
    dtype=np.float64,
)

# Calibration version tag — recorders / metadata writers can read this so
# downstream analysis can identify episodes captured under each intrinsics
# regime. Bump this string when CAMERA_K / CAMERA_D are re-calibrated.
CAMERA_INTRINSICS_VERSION = "calib_2026_04"

# Image dimensions — used by helpers that need full-frame size.
_IMG_W, _IMG_H = 1920, 1080

# ── Legacy URDF-derived approximation (HISTORICAL REFERENCE — DO NOT USE) ────
# Preserved so analyses or notebooks that hard-coded these values can still
# resolve them. The unprojection path now goes through CAMERA_K / CAMERA_D.
_HFOV_DEG_LEGACY   = 120.0
_FX_LEGACY = _FY_LEGACY = (_IMG_W / 2.0) / math.tan(
    math.radians(_HFOV_DEG_LEGACY / 2.0))   # ≈ 554.3
_CX_LEGACY         = _IMG_W / 2.0   # 960.0
_CY_LEGACY         = _IMG_H / 2.0   # 540.0

# URDF-derived extrinsics — camera in robot base frame
_CAM_FORWARD  =  0.327   # m
_CAM_LATERAL  =  0.000   # m
_CAM_HEIGHT   =  0.043   # m above base link
# Camera pitch = 0 rad (faces horizontally per URDF rpy = 0 0 0)
# Rotation: camera optical frame (z-forward, x-right, y-down)
#           → robot base frame   (x-forward, y-left, z-up)
# p_base_x =  p_cam_z
# p_base_y = -p_cam_x
# p_base_z = -p_cam_y
_R_CAM_TO_BASE = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)
_T_CAM_IN_BASE = np.array([_CAM_FORWARD, _CAM_LATERAL, _CAM_HEIGHT], dtype=np.float64)

# Depth Anything V2 checkpoint
_DA2_CKPT = Path.home() / "Robotics" / "weights" / "depth_anything_v2_metric_hypersim_vitl.pth"

# GroundingDINO weights
_GDINO_CKPT   = Path.home() / "Robotics" / "weights" / "groundingdino_swint_ogc.pth"
_GDINO_CFG    = Path.home() / "Robotics" / "weights" / "GroundingDINO_SwinT_OGC.py"

# SAM2 checkpoint
_SAM2_CKPT    = Path.home() / "Robotics" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
_SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Detection thresholds
_BOX_THRESHOLD  = 0.30
_TEXT_THRESHOLD = 0.25

# Depth sampling: take median over this fraction of mask pixels nearest centroid
_DEPTH_SAMPLE_RADIUS_PX = 15   # pixel radius around centroid for depth sampling


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    """Output of a single grounding call."""
    mask:            Optional[np.ndarray] = None   # (H, W) bool
    bbox:            Optional[np.ndarray] = None   # [x1, y1, x2, y2] in pixels
    centroid_px:     Optional[np.ndarray] = None   # [u, v] in pixels
    depth_m:         Optional[float]      = None   # metric depth at centroid
    position_camera: Optional[np.ndarray] = None   # [x,y,z] in camera frame
    position_base:   Optional[np.ndarray] = None   # [x,y,z] in robot base frame
    confidence:      Optional[float]      = None   # GroundingDINO score


# ─────────────────────────────────────────────────────────────────────────────
# Go2 Camera — replaces RealSenseCamera
# ─────────────────────────────────────────────────────────────────────────────

class Go2Camera:
    """
    Thin wrapper around the Go2X VideoClient that delivers BGR numpy frames
    from the robot's head camera over DDS/Ethernet.

    Requires ChannelFactoryInitialize to have been called before construction,
    OR pass already_initialized=False to call it here.

    Args:
        network_interface:   Ethernet interface name (default: enx98fc84e68f1a)
        already_initialized: If False, calls ChannelFactoryInitialize internally.
    """

    NETWORK_INTERFACE = "enx98fc84e68f1a"

    def __init__(self,
                 network_interface: str = NETWORK_INTERFACE,
                 already_initialized: bool = False):
        if not already_initialized:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, network_interface)

        from unitree_sdk2py.go2.video.video_client import VideoClient
        self._client = VideoClient()
        self._client.SetTimeout(3.0)
        self._client.Init()
        print(f"[Go2Camera] VideoClient initialised on {network_interface}")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Fetch one RGB frame from the Go2 head camera.

        Returns:
            BGR numpy array of shape (H, W, 3), or None on failure.
        """
        code, data = self._client.GetImageSample()
        if code != 0:
            print(f"[Go2Camera] GetImageSample error code: {code}")
            return None
        img = cv2.imdecode(
            np.frombuffer(bytes(data), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        return img

    def stop(self):
        """No-op — VideoClient has no explicit teardown."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Depth Anything V2 loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_depth_anything_v2(device: torch.device) -> DepthAnythingV2:
    """Load Depth Anything V2 Metric Indoor (ViTL) checkpoint."""
    model_cfg = {
        "encoder":      "vitl",
        "features":     256,
        "out_channels": [256, 512, 1024, 1024],
        "max_depth":    20,   # metres — indoor setting
    }
    model = DepthAnythingV2(**model_cfg)
    state = torch.load(str(_DA2_CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[DepthAnythingV2] Loaded ViTL metric indoor from {_DA2_CKPT}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# GroundingDINO loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_grounding_dino(device: torch.device):
    """Load GroundingDINO SwinT model."""
    args = SLConfig.fromfile(str(_GDINO_CFG))
    args.device = str(device)
    model = build_model(args)
    state = torch.load(str(_GDINO_CKPT), map_location="cpu")
    model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    model = model.to(device).eval()
    print(f"[GroundingDINO] Loaded SwinT from {_GDINO_CKPT}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Visual Grounder
# ─────────────────────────────────────────────────────────────────────────────

class VisualGrounder:
    """
    Main perception class.  Loads GroundingDINO, SAM2, and Depth Anything V2
    once at construction, then accepts RGB frames at runtime.

    Usage:
        grounder = VisualGrounder()
        result   = grounder.ground(bgr_frame, "red button")
        # result.position_base → [x, y, z] in robot base frame (metres)
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VisualGrounder] Using device: {self.device}")

        print("[VisualGrounder] Loading GroundingDINO …")
        self._gdino = _load_grounding_dino(self.device)

        print("[VisualGrounder] Loading SAM2 …")
        sam2_model = build_sam2(
            _SAM2_CFG,
            str(_SAM2_CKPT),
            device=self.device,
        )
        self._sam2 = SAM2ImagePredictor(sam2_model)

        print("[VisualGrounder] Loading Depth Anything V2 …")
        self._depth_model = _load_depth_anything_v2(self.device)

        # GroundingDINO image transform
        self._gdino_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print("[VisualGrounder] All models loaded.\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_grounding_dino(self, bgr: np.ndarray, prompt: str):
        """Run GroundingDINO and return (boxes_xyxy, scores, phrases)."""
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        img_t, _ = self._gdino_transform(pil, None)
        img_t = img_t.to(self.device).unsqueeze(0)

        caption = prompt.lower().strip()
        if not caption.endswith("."):
            caption += "."

        with torch.no_grad():
            outputs = self._gdino(img_t, captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]   # (N, 256)
        boxes  = outputs["pred_boxes"][0]               # (N, 4) cx,cy,w,h normalised

        scores = logits.max(dim=-1).values
        keep   = scores > _BOX_THRESHOLD

        boxes  = boxes[keep].cpu()
        scores = scores[keep].cpu()
        logits = logits[keep].cpu()

        if boxes.shape[0] == 0:
            return None, None, None

        # Convert normalised cx,cy,w,h → pixel x1,y1,x2,y2
        H, W = bgr.shape[:2]
        cx, cy, w, h = boxes.unbind(-1)
        x1 = ((cx - w / 2) * W).clamp(0, W).numpy()
        y1 = ((cy - h / 2) * H).clamp(0, H).numpy()
        x2 = ((cx + w / 2) * W).clamp(0, W).numpy()
        y2 = ((cy + h / 2) * H).clamp(0, H).numpy()
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Keep only highest-scoring detection
        best    = scores.argmax().item()
        box     = boxes_xyxy[best]
        score   = float(scores[best])

        # Phrase tokenisation
        tokeniser = self._gdino.tokenizer
        tokens    = tokeniser(caption)
        phrase    = get_phrases_from_posmap(
            logits[best] > _TEXT_THRESHOLD,
            tokens,
            tokeniser,
        )
        return box, score, phrase

    def _run_sam2(self, bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """Run SAM2 with a bounding box prompt; return best mask (H, W bool)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._sam2.set_image(rgb)
        masks, scores, _ = self._sam2.predict(
            box=box[None],
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None
        best = scores.argmax()
        return masks[best].astype(bool)

    def _run_depth(self, bgr: np.ndarray) -> np.ndarray:
        """
        Run Depth Anything V2 on the full BGR frame.
        Returns HxW float32 depth map in metres.
        """
        # infer_image expects BGR, returns numpy HxW float32
        depth = self._depth_model.infer_image(bgr)
        return depth.astype(np.float32)

    @staticmethod
    def _sample_depth(depth_map: np.ndarray,
                      centroid: np.ndarray,
                      mask: np.ndarray) -> Optional[float]:
        """
        Sample metric depth at the mask centroid and apply empirical correction.

        Depth Anything V2 overestimates on the Go2X 120 deg wide-angle lens.
        Calibrated at 3 distances (7.5in, 18.5in, 36in):
            actual = reported x 0.724 - 0.008

        Returns None if raw depth < 0.15m (too close, >5x error at 3in).
        Caller should freeze last valid estimate at contact range.
        """
        _CORRECTION_A  = 0.724
        _CORRECTION_B  = 0.008
        _MIN_RAW_DEPTH = 0.15

        H, W   = depth_map.shape
        cx, cy = int(centroid[0]), int(centroid[1])

        ys, xs = np.ogrid[:H, :W]
        radius_mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= _DEPTH_SAMPLE_RADIUS_PX ** 2
        combined    = radius_mask & mask & (depth_map > 0.05) & (depth_map < 18.0)

        if combined.sum() > 0:
            raw = float(np.median(depth_map[combined]))
        else:
            full = mask & (depth_map > 0.05) & (depth_map < 18.0)
            raw  = float(np.median(depth_map[full])) if full.sum() > 0 \
                   else float(np.median(depth_map[mask]))

        if raw < _MIN_RAW_DEPTH:
            return None

        return max(0.05, raw * _CORRECTION_A - _CORRECTION_B)

    @staticmethod
    def _unproject(u: float, v: float, depth: float) -> np.ndarray:
        """
        Unproject pixel (u, v) at metric depth to 3D point in camera frame.
        Camera optical convention: z-forward, x-right, y-down.

        Two-step pipeline:
          1. Undistort the pixel back into ``CAMERA_K``'s linear projection
             frame using the plumb_bob distortion coefficients.
          2. Unproject the corrected pixel using the calibrated focal lengths
             and principal point.

        ``cv2.undistortPoints`` with ``P=CAMERA_K`` returns undistorted pixel
        coordinates in the same K (rather than normalised image coordinates),
        so we can apply the standard pinhole formula on the result.
        """
        pts = np.asarray([[u, v]], dtype=np.float32).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts, CAMERA_K, CAMERA_D, P=CAMERA_K)
        u_corr = float(undist[0, 0, 0])
        v_corr = float(undist[0, 0, 1])

        fx = CAMERA_K[0, 0]
        fy = CAMERA_K[1, 1]
        cx = CAMERA_K[0, 2]
        cy = CAMERA_K[1, 2]

        x_cam = (u_corr - cx) * depth / fx
        y_cam = (v_corr - cy) * depth / fy
        z_cam = depth
        return np.array([x_cam, y_cam, z_cam], dtype=np.float64)

    @staticmethod
    def _camera_to_base(p_cam: np.ndarray) -> np.ndarray:
        """Transform 3D point from camera frame to robot base frame."""
        return _R_CAM_TO_BASE @ p_cam + _T_CAM_IN_BASE

    # ── Public API ────────────────────────────────────────────────────────────

    def ground(self, bgr_frame: np.ndarray, prompt: str) -> Optional[GroundingResult]:
        """
        Run the full grounding pipeline on one BGR frame.

        Args:
            bgr_frame: (H, W, 3) uint8 BGR image from Go2Camera.get_frame()
            prompt:    Text description of target, e.g. "red button"

        Returns:
            GroundingResult with all fields populated, or None on failure.
        """
        result = GroundingResult()

        # ── Stage 1: GroundingDINO ────────────────────────────────────────────
        box, score, phrase = self._run_grounding_dino(bgr_frame, prompt)
        if box is None:
            return None

        result.bbox       = box
        result.confidence = score

        # ── Stage 2: SAM2 ─────────────────────────────────────────────────────
        mask = self._run_sam2(bgr_frame, box)
        if mask is None or mask.sum() == 0:
            return None
        result.mask = mask

        # Mask centroid in pixel coordinates
        ys, xs         = np.where(mask)
        centroid_px    = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        result.centroid_px = centroid_px

        # ── Stage 3: Depth Anything V2 ───────────────────────────────────────
        depth_map      = self._run_depth(bgr_frame)
        depth_m        = self._sample_depth(depth_map, centroid_px, mask)
        result.depth_m = depth_m

        # Guard: too close for reliable depth — return partial result.
        # Caller (Stage D) should freeze the last valid position_base.
        if depth_m is None:
            return result

        # ── Stage 4: Unproject → camera frame ────────────────────────────────
        p_cam              = self._unproject(centroid_px[0], centroid_px[1], depth_m)
        result.position_camera = p_cam

        # ── Stage 5: Camera → base frame ─────────────────────────────────────
        p_base             = self._camera_to_base(p_cam)
        result.position_base = p_base

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _annotate(frame: np.ndarray, result: GroundingResult, prompt: str) -> np.ndarray:
    vis = frame.copy()
    if result.mask is not None:
        overlay        = vis.copy()
        overlay[result.mask] = (0, 200, 100)
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
    if result.bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in result.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
    if result.centroid_px is not None:
        cx, cy = int(result.centroid_px[0]), int(result.centroid_px[1])
        cv2.circle(vis, (cx, cy), 6, (0, 165, 255), -1)
    lines = [
        f"Prompt    : {prompt}",
        f"Confidence: {result.confidence:.3f}",
        f"Depth (m) : {result.depth_m:.3f}" if result.depth_m else "Depth (m) : --",
    ]
    if result.position_base is not None:
        pb = result.position_base
        lines.append(f"Base [m]  : x={pb[0]:.3f} y={pb[1]:.3f} z={pb[2]:.3f}")
    y = 28
    for ln in lines:
        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return vis


def main():
    parser = argparse.ArgumentParser(description="Grounded SAM2 + Depth Anything V2")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to static test image (skips live camera)")
    parser.add_argument("--prompt", type=str, default="red button",
                        help="Text prompt for GroundingDINO")
    parser.add_argument("--frames", type=int, default=30,
                        help="Number of live frames to process (live mode only)")
    parser.add_argument("--show",   action="store_true",
                        help="Display annotated frames in an OpenCV window")
    parser.add_argument("--interface", type=str,
                        default=Go2Camera.NETWORK_INTERFACE,
                        help="Ethernet interface for DDS")
    args = parser.parse_args()

    grounder = VisualGrounder()

    if args.image:
        # ── Static image mode ─────────────────────────────────────────────────
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read image: {args.image}")
            sys.exit(1)
        result = grounder.ground(frame, args.prompt)
        if result is None:
            print("No detection.")
        else:
            print(f"Confidence   : {result.confidence:.3f}")
            print(f"Depth (m)    : {result.depth_m:.3f}")
            print(f"Cam  [m]     : {result.position_camera}")
            print(f"Base [m]     : {result.position_base}")
            if args.show:
                vis = _annotate(frame, result, args.prompt)
                cv2.imshow("Grounding", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        # ── Live camera mode ──────────────────────────────────────────────────
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(0, args.interface)
        camera = Go2Camera(already_initialized=True)

        print(f"Warming up camera (8 frames discarded) …")
        for _ in range(8):
            camera.get_frame()
            time.sleep(0.05)

        print(f"Running {args.frames} frames with prompt: \"{args.prompt}\"\n")
        for i in range(args.frames):
            frame = camera.get_frame()
            if frame is None:
                continue
            result = grounder.ground(frame, args.prompt)
            if result is None:
                print(f"[{i:03d}] No detection.")
            else:
                pb = result.position_base
                print(f"[{i:03d}] conf={result.confidence:.3f}  "
                      f"depth={result.depth_m:.3f}m  "
                      f"base=[{pb[0]:.3f}, {pb[1]:.3f}, {pb[2]:.3f}]")
            if args.show and frame is not None:
                vis = _annotate(frame, result, args.prompt) if result else frame
                cv2.imshow("Grounding Live", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()