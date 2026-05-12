#!/usr/bin/env bash
# Encode a short looping GIF for the README banner from a source MP4.
#
# Usage:
#   scripts/make_banner.sh <input.mp4> [--start SEC] [--duration SEC] [--width PX] [--fps N] [--out PATH]
#
# Defaults: --start 0 --duration 6 --width 720 --fps 15 --out figures/banner.gif
#
# Uses a two-pass palette (palettegen + paletteuse) for clean color, which is
# the standard recipe for high-quality GIFs from video.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input.mp4> [--start SEC] [--duration SEC] [--width PX] [--fps N] [--out PATH]" >&2
    exit 64
fi

INPUT="$1"; shift
START=0
DURATION=6
WIDTH=720
FPS=15
OUT="figures/banner.gif"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)    START="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --width)    WIDTH="$2"; shift 2 ;;
        --fps)      FPS="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 64 ;;
    esac
done

if [[ ! -f "$INPUT" ]]; then
    echo "input not found: $INPUT" >&2
    exit 66
fi

mkdir -p "$(dirname "$OUT")"

VF="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"

echo "Encoding: $INPUT  ->  $OUT"
echo "  start=${START}s  duration=${DURATION}s  width=${WIDTH}px  fps=${FPS}"

ffmpeg -y -ss "$START" -t "$DURATION" -i "$INPUT" -vf "$VF" -loop 0 "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
echo "Done: $OUT ($SIZE). If the file is too large, retry with --fps 12 --width 640."
