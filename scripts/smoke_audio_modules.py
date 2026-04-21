"""
scripts/smoke_audio_modules.py

Interactive smoke tests for Stage D audio modules:
1) AudioRecorder (5 s waveform capture)
2) AudioLiveDetector (silence calibration + clap detection)
3) ColorDetector (faster-whisper transcription color check)

Examples:
    python scripts/smoke_audio_modules.py --list-devices
    python scripts/smoke_audio_modules.py --device-index 1 --skip-whisper
    python scripts/smoke_audio_modules.py --device-index 1 --model-size small --whisper-device cpu --compute-type int8 --expected-color green
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

VALID_COLORS = ("red", "blue", "green", "yellow")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {label}{suffix}")
    RESULTS.append((label, condition, detail))
    return condition


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def test_audio_recorder(device_index: int | None, sample_rate: int) -> bool:
    from src.data.audio_recorder import AudioRecorder

    section("Test 1 — AudioRecorder (~5 s speech)")
    print(f"{INFO}  Default input device index: {device_index!r}")
    print("  You will have ~5 seconds to speak into the mic after START.\n")

    rec = AudioRecorder(sample_rate=sample_rate, device_index=device_index)
    rec.start_stream()
    try:
        input("  Press Enter when ready, then speak until you see DONE… ")
        rec.start_episode()
        print("  START — speak now…")
        time.sleep(5.0)
        print("  DONE.")
        audio = rec.end_episode()
    finally:
        rec.stop_stream()

    n_samples = int(len(audio))
    minimum = int(sample_rate * 4)
    return check(
        f"len(audio) > {sample_rate} * 4  (got {n_samples} samples)",
        n_samples > minimum,
        f"min ~{minimum}",
    )


def test_audio_live_detector(device_index: int | None, sample_rate: int) -> bool:
    from src.data.audio_live_detector import AudioLiveDetector

    section("Test 2 — AudioLiveDetector (silence + clap)")
    print(f"{INFO}  Default input device index: {device_index!r}")
    print("  Step A: stay quiet for 2 s while baseline calibrates.")
    print("  Step B: within the next 5 s, clap once (or make a sharp sound).\n")

    det = AudioLiveDetector(
        sample_rate=sample_rate,
        device_index=device_index,
        threshold_db=20.0,
        min_duration_ms=50,
    )
    det.start_stream()
    try:
        input("  Press Enter to start calibration (then stay silent)… ")
        det.calibrate_baseline(2.0)
        input("  Press Enter to arm detection, then clap within 5 s… ")
        det.start_episode()

        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            if det.sound_detected:
                break
            time.sleep(0.02)
        elapsed = time.monotonic() - t0
    finally:
        det.stop_stream()

    ok_detected = check("sound_detected is True after clap", det.sound_detected)

    t_det = det.detection_time_s
    if t_det is None:
        ok_time = check("detection_time_s is set when sound_detected", False, "got None")
    else:
        ok_time = check(
            "detection_time_s in [0, 6] s",
            0.0 <= t_det <= 6.0,
            f"t={t_det:.3f}s (wait loop ran {elapsed:.2f}s)",
        )
    return ok_detected and ok_time


def test_color_detector(
    device_index: int | None,
    sample_rate: int,
    whisper_device: str,
    compute_type: str,
    model_size: str,
    expected_color: str,
) -> bool:
    from src.data.audio_recorder import AudioRecorder
    from src.data.color_detector import ColorDetector

    section(f'Test 3 — ColorDetector (say "{expected_color}" ~3 s)')
    print(
        f"{INFO}  Whisper: model={model_size!r} device={whisper_device!r} "
        f"compute_type={compute_type!r}"
    )
    print(
        f"  Record yourself clearly saying the word {expected_color.upper()} "
        "for ~3 seconds.\n"
    )

    rec = AudioRecorder(sample_rate=sample_rate, device_index=device_index)
    rec.start_stream()
    try:
        input(f'  Press Enter when ready, then say "{expected_color}" for ~3 s… ')
        rec.start_episode()
        print("  RECORDING…")
        time.sleep(3.0)
        audio = rec.end_episode()
    finally:
        rec.stop_stream()

    cd = ColorDetector(model_size=model_size, device=whisper_device, compute_type=compute_type)
    cd.load()
    result = cd.detect(audio, sample_rate=sample_rate)

    transcript = result.get("transcript", "")
    detected = result.get("color_detected")
    print(f"\n  {INFO}  transcript: {transcript!r}")
    print(f"  {INFO}  color_detected: {detected!r}\n")

    return check(
        f'result["color_detected"] == "{expected_color}"',
        detected == expected_color,
        f"success_audio_transcript={result.get('success_audio_transcript')}",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Smoke tests for AudioRecorder, AudioLiveDetector, ColorDetector",
    )
    parser.add_argument("--device-index", type=int, default=None, help="Input device index")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (Hz)")
    parser.add_argument("--list-devices", action="store_true", help="List devices and exit")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip ColorDetector test")
    parser.add_argument(
        "--whisper-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Whisper compute device",
    )
    parser.add_argument("--compute-type", type=str, default=None, help="Whisper compute type")
    parser.add_argument("--model-size", type=str, default="tiny", help="Whisper model size")
    parser.add_argument(
        "--expected-color",
        choices=VALID_COLORS,
        default="red",
        help="Expected detected color for Test 3",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    whisper_device = resolve_device(args.whisper_device)
    compute_type = args.compute_type or default_compute_type(whisper_device)

    section("Stage D audio module smoke tests")
    print(f"{INFO}  device_index={args.device_index!r}  sample_rate={args.sample_rate}")

    ok1 = test_audio_recorder(args.device_index, args.sample_rate)
    ok2 = test_audio_live_detector(args.device_index, args.sample_rate)
    ok3 = True

    if args.skip_whisper:
        section("Test 3 — ColorDetector")
        print(f"  {INFO}  Skipped (--skip-whisper)")
    else:
        ok3 = test_color_detector(
            device_index=args.device_index,
            sample_rate=args.sample_rate,
            whisper_device=whisper_device,
            compute_type=compute_type,
            model_size=args.model_size,
            expected_color=args.expected_color,
        )

    section("Summary")
    for label, condition, _ in RESULTS:
        mark = "✓" if condition else "✗"
        state = "PASS" if condition else "FAIL"
        print(f"  {mark}  {label}: {state}")

    all_pass = ok1 and ok2 and (True if args.skip_whisper else ok3)
    print()
    if all_pass:
        print(f"  {PASS}  All run tests passed.")
        return 0
    print(f"  {FAIL}  One or more tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
