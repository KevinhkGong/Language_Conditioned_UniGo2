"""
scripts/debug_whisper.py

Diagnostic tool for Whisper audio pipeline.

Usage:
    # Debug a specific episode HDF5 file
    python scripts/debug_whisper.py --episode data/real/stage_d/episode_XXX.h5

    # Debug the most recent episode
    python scripts/debug_whisper.py --latest

    # List all episodes
    python scripts/debug_whisper.py --list
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_audio(audio, sr, label=""):
    """Print audio statistics."""
    if len(audio) == 0:
        print(f"  {label}  EMPTY AUDIO!")
        return
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    max_amp = float(np.abs(audio).max())
    zero_pct = float(np.mean(audio == 0) * 100)

    print(f"  {label}")
    print(f"    Duration:    {duration:.2f}s ({len(audio)} samples @ {sr}Hz)")
    print(f"    RMS:         {rms:.6f}")
    print(f"    Max ampl:    {max_amp:.4f}")
    print(f"    % zero:      {zero_pct:.1f}%")
    print(f"    % clipped:   {float(np.mean(np.abs(audio) > 0.99) * 100):.2f}%")

    # Segment-by-segment (5 segments)
    n_segs = 5
    seg_len = len(audio) // n_segs
    print(f"    Per-segment RMS:")
    for i in range(n_segs):
        start = i * seg_len
        end = start + seg_len
        seg_rms = float(np.sqrt(np.mean(audio[start:end] ** 2)))
        bar_len = int(seg_rms * 5000)
        bar = "█" * min(bar_len, 50)
        t_start = start / sr
        t_end = end / sr
        print(f"      [{t_start:5.2f} - {t_end:5.2f}s]  RMS={seg_rms:.6f}  {bar}")


def find_loud_segments(audio, sr, threshold_db=20.0, min_duration_ms=50):
    """Find loud segments that exceeded baseline threshold."""
    block_ms = 20
    block_size = int(sr * block_ms / 1000)
    baseline_samples = int(sr * 2.0)   # first 2s
    baseline = audio[:baseline_samples]
    baseline_rms = float(np.sqrt(np.mean(baseline ** 2)))
    baseline_rms = max(baseline_rms, 1e-6)

    blocks = []
    for start in range(0, len(audio) - block_size, block_size):
        block = audio[start:start + block_size]
        rms = float(np.sqrt(np.mean(block ** 2)))
        db = 20.0 * np.log10(rms / baseline_rms) if rms > 1e-8 else -120.0
        blocks.append((start / sr, rms, db))

    print(f"  Baseline RMS (first 2s): {baseline_rms:.6f}")
    print(f"  Threshold: {threshold_db} dB above baseline")
    print()
    print(f"  Segments > {threshold_db}dB above baseline:")

    in_loud = False
    loud_start = None
    segments = []

    for t, rms, db in blocks:
        if db > threshold_db:
            if not in_loud:
                loud_start = t
                in_loud = True
        else:
            if in_loud:
                duration = t - loud_start
                if duration * 1000 >= min_duration_ms:
                    segments.append((loud_start, t, duration))
                in_loud = False

    if in_loud and loud_start is not None:
        segments.append((loud_start, len(audio) / sr, len(audio)/sr - loud_start))

    if not segments:
        print("    [NONE — audio is uniformly quiet relative to baseline]")
    else:
        for s_start, s_end, dur in segments:
            print(f"    [{s_start:5.2f}s - {s_end:5.2f}s]  duration={dur*1000:.0f}ms")

    return segments


def run_whisper_diagnostics(audio, sr, model_path):
    """Run Whisper with multiple parameter sets and compare."""
    from faster_whisper import WhisperModel

    print("  Loading Whisper model (small)…")
    model = WhisperModel(model_path, device="cuda", compute_type="float16")

    # Normalize audio (boost quiet recordings)
    peak = float(np.abs(audio).max())
    if 0 < peak < 0.5:
        audio_norm = audio / peak * 0.9
        print(f"  Audio normalized: peak {peak:.3f} → 0.9")
    else:
        audio_norm = audio

    tests = [
        ("Default (beam=1, vad=on)", dict(
            beam_size=1,
            vad_filter=True,
            temperature=0.0,
        )),
        ("Beam=5, vad=on", dict(
            beam_size=5,
            vad_filter=True,
            temperature=0.0,
        )),
        ("Beam=5, vad=off", dict(
            beam_size=5,
            vad_filter=False,
            temperature=0.0,
        )),
        ("Beam=5, vad=off, low threshold", dict(
            beam_size=5,
            vad_filter=False,
            temperature=0.0,
            no_speech_threshold=0.05,
        )),
        ("Beam=5, vad=off, low threshold, hint", dict(
            beam_size=5,
            vad_filter=False,
            temperature=0.0,
            no_speech_threshold=0.05,
            condition_on_previous_text=False,
            initial_prompt="The button announces: red, blue, green, yellow.",
        )),
    ]

    results = []
    for name, params in tests:
        try:
            segments, info = model.transcribe(
                audio_norm,
                language="en",
                **params,
            )
            transcript = " ".join(s.text for s in segments).strip()
        except Exception as e:
            transcript = f"[ERROR: {e}]"

        results.append((name, transcript))
        print(f"    {name}:")
        print(f"      → {transcript!r}")

    print()
    colors = ["red", "blue", "green", "yellow"]
    print("  Color detection across configs:")
    for name, transcript in results:
        lower = transcript.lower()
        found = [c for c in colors if c in lower]
        found_str = ", ".join(found) if found else "(none)"
        print(f"    {name}: {found_str}")


def inspect_episode(path, run_whisper=True):
    """Full episode audio diagnostics."""
    print(f"\n{'='*70}")
    print(f"  EPISODE: {path}")
    print(f"{'='*70}")

    with h5py.File(path, "r") as f:
        # Metadata
        print(f"\n[METADATA]")
        attrs = dict(f.attrs)
        for k in ["success_fk", "success_audio_live", "success_target",
                  "audio_detection_time_s", "contact_method",
                  "color_detected", "color_expected", "audio_transcript",
                  "audio_sample_rate", "duration_s"]:
            if k in attrs:
                val = attrs[k]
                if isinstance(val, bytes):
                    val = val.decode()
                print(f"  {k:28s} = {val!r}")

        # Audio
        if "audio" not in f:
            print("\n[ERROR] No audio dataset in HDF5 file!")
            return

        audio = f["audio"][()]
        sr = int(attrs.get("audio_sample_rate", 16000))

    print(f"\n[AUDIO STATS]")
    analyze_audio(audio, sr, "Full recording")

    print(f"\n[LOUD SEGMENT DETECTION]")
    loud_segs = find_loud_segments(audio, sr)

    # Correlate with live detection time
    det_t = attrs.get("audio_detection_time_s", None)
    if det_t is not None and not np.isnan(det_t):
        print(f"\n  Live detector fired at t={det_t:.2f}s")
        matched = any(s <= det_t <= e for s, e, _ in loud_segs)
        if matched:
            print("    ✓ This matches a loud segment we found")
        else:
            print("    ✗ No loud segment at that time — check threshold calibration")

    # Save WAV
    wav_path = "/tmp/debug_episode.wav"
    wavfile.write(wav_path, sr, audio.astype(np.float32))
    print(f"\n[WAV SAVED]")
    print(f"  {wav_path}")
    print(f"  Listen with: pw-play {wav_path}  (or aplay {wav_path})")

    if run_whisper:
        print(f"\n[WHISPER DIAGNOSTICS]")
        model_path = os.path.expanduser("~/Robotics/weights/faster-whisper-small")
        if not os.path.exists(model_path):
            print(f"  [ERROR] Model path not found: {model_path}")
            print(f"  Skipping Whisper tests.")
            return
        run_whisper_diagnostics(audio, sr, model_path)


def latest_episode():
    """Find the most recent episode."""
    pattern = "data/real/stage_d/episode_*.h5"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No episodes found matching {pattern}")
        sys.exit(1)
    return files[-1]


def list_episodes():
    """List all episodes."""
    pattern = "data/real/stage_d/episode_*.h5"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[INFO] No episodes found")
        return
    print(f"Found {len(files)} episode(s):")
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  {f}  ({size_kb:.0f}KB)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode", help="Specific HDF5 episode file")
    g.add_argument("--latest", action="store_true",
                   help="Use the most recent episode")
    g.add_argument("--list", action="store_true",
                   help="List all episodes")
    ap.add_argument("--no-whisper", action="store_true",
                    help="Skip Whisper transcription tests")
    args = ap.parse_args()

    if args.list:
        list_episodes()
        return

    if args.latest:
        path = latest_episode()
    else:
        path = args.episode

    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    inspect_episode(path, run_whisper=not args.no_whisper)


if __name__ == "__main__":
    main()