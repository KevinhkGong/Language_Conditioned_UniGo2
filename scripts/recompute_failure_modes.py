"""
scripts/recompute_failure_modes.py

Retrospective recompute of the ``failure_mode`` column in eval CSVs.

Walks every ``data/eval/eval_*.csv`` row and re-applies the corrected
``_classify_failure`` logic from ``run_methods.py`` (success = any contact
signal AND correct color word). Writes back in place.

Pre-fix, the classifier only counted FK-driven contact as success, so
audio-driven successes were silently mis-labelled ``miss``. This script
unwinds that.

Optional ``--force-success TRIAL_ID`` flags let you override specific
rows that succeeded physically but had no detector evidence (silent press,
mic dropout, etc.). Each forced row gets ``failure_mode=""`` and a note
appended to ``operator_notes``.

Usage:
    python scripts/recompute_failure_modes.py
        [--data-dir data/eval]
        [--dry-run]
        [--force-success 20260429_222106_core_method]
        [--force-success 20260428_174015_baseline_2]
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EXPECTED_COLOR = "red"


def classify_failure(
    *,
    success_fk: bool,
    audio_detected: bool,
    color_detected: Optional[str],
    timeout_phase: Optional[str],
) -> str:
    """Mirror of the corrected ``_classify_failure`` in run_methods.py."""
    contact_succ = bool(success_fk or audio_detected)
    color_match = (color_detected == EXPECTED_COLOR)
    color_other = bool(color_detected) and not color_match  # heard a non-red color word

    if contact_succ and color_match:
        return ""
    if audio_detected and color_other:
        return "wrong_color"
    if timeout_phase is not None:
        return "timeout"
    return "miss"


def _row_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/eval"),
                   help="Directory containing eval_*.csv files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change; do not write back.")
    p.add_argument("--force-success", action="append", default=[],
                   help="Trial ID to force-mark as success (failure_mode=''). "
                        "May be repeated. Example: --force-success "
                        "20260429_222106_core_method")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"Data dir not found: {args.data_dir}")

    forced = set(args.force_success)
    forced_seen: set[str] = set()

    files = sorted(args.data_dir.glob("eval_*.csv"))
    if not files:
        raise SystemExit(f"No eval_*.csv files in {args.data_dir}")

    n_rows = 0
    n_changed = 0
    n_forced = 0
    transitions: dict[tuple[str, str], int] = {}

    for f in files:
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        if not rows:
            continue
        if "failure_mode" not in fields:
            logger.warning(f"Skip {f.name}: no failure_mode column")
            continue

        for r in rows:
            n_rows += 1
            old = (r.get("failure_mode") or "").strip()
            tid = r.get("trial_id", "").strip()

            if tid in forced:
                new = ""
                forced_seen.add(tid)
                if old != new:
                    n_forced += 1
                    notes = r.get("operator_notes", "") or ""
                    extra = "manual_success_override"
                    if extra not in notes:
                        notes = (notes + " | " + extra).strip(" |")
                    r["operator_notes"] = notes
            else:
                # Re-apply the corrected classifier from the row's facts.
                fk_str = (r.get("contact_method") or "").strip()
                success_fk = (fk_str == "fk")
                audio_detected = _row_bool(r.get("success_audio_live", ""))
                color_detected = (r.get("color_detected") or "").strip() or None
                # The CSV doesn't carry a timeout_phase column. The legacy
                # classifier only ever returned "timeout" when a controller-
                # level timeout fired (rare). We can't recover that retro,
                # so pass None — affected rows will fall through to "miss"
                # which matches the pre-fix label for those cases anyway.
                new = classify_failure(
                    success_fk=success_fk,
                    audio_detected=audio_detected,
                    color_detected=color_detected,
                    timeout_phase=None,
                )

            if new != old:
                transitions[(old, new)] = transitions.get((old, new), 0) + 1
                if not (tid in forced):
                    n_changed += 1
                r["failure_mode"] = new

        if not args.dry_run:
            with open(f, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

    print()
    print(f"Scanned {len(files)} CSVs, {n_rows} rows total.")
    print(f"Auto-recomputed → {n_changed} row(s) changed by classifier fix.")
    print(f"Manually forced → {n_forced} row(s) overridden via --force-success.")
    if forced:
        missing = forced - forced_seen
        if missing:
            print(f"WARNING: --force-success ids not found: "
                  f"{sorted(missing)}")
    if transitions:
        print()
        print("failure_mode transitions:")
        for (old, new), n in sorted(transitions.items(), key=lambda x: -x[1]):
            print(f"  {old!r:<22} → {new!r:<22}  n={n}")
    if args.dry_run:
        print("\n(dry-run — no files modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
