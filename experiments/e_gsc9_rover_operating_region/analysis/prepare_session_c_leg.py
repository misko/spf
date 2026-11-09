#!/usr/bin/env python3
"""Render one immutable E-GSC9 session-C A/B/A-prime capture config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import yaml


LEG_STATE = {
    "a": ("no_pads", "no-extra-pads-control-a", -35),
    "b": ("pads_installed", "plus10db-per-arm-treatment-b", -25),
    "aprime": ("pads_removed", "no-extra-pads-reversal-aprime", -35),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--leg", choices=tuple(LEG_STATE), required=True)
    parser.add_argument("--physical-state", required=True)
    parser.add_argument("--operator-note", required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    args = parser.parse_args()

    expected_state, label_suffix, tx_gain_db = LEG_STATE[args.leg]
    if args.physical_state != expected_state:
        parser.error(
            f"leg {args.leg} requires --physical-state {expected_state!r}, "
            f"not {args.physical_state!r}"
        )
    if not args.operator_note.strip():
        parser.error("--operator-note must be non-empty")

    document = yaml.safe_load(args.base.read_text())
    calibration = document["calibration"]
    calibration["tx-gain-db"] = tx_gain_db
    calibration["setup-label"] = (
        f"e-gsc9c-r17-r18-v5-iio-usb-tx-fixed{abs(tx_gain_db)}db-gain-floor26-"
        f"30db-pad-tee-{label_suffix}"
    )
    calibration["notes"] = (
        str(calibration["notes"]).strip()
        + f" Physical A/B/A-prime leg={args.leg}; declared state={expected_state}; "
        + f"TX gain={tx_gain_db} dB; operator note={args.operator_note.strip()}"
    )
    rendered = yaml.safe_dump(document, sort_keys=False)

    if args.config_output.exists():
        existing = yaml.safe_load(args.config_output.read_text())
        if existing != document:
            raise ValueError(f"existing rendered config differs: {args.config_output}")
    else:
        write_atomic(args.config_output, rendered)

    repo = Path(__file__).resolve().parents[3]
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = {
        "schema": "spf.experiment.e_gsc9.session_c_physical_state",
        "schema_version": 1,
        "leg": args.leg,
        "physical_state": expected_state,
        "tx_gain_db": tx_gain_db,
        "operator_note": args.operator_note.strip(),
        "base_config": str(args.base.resolve()),
        "base_config_sha256": sha256(args.base),
        "rendered_config": str(args.config_output.resolve()),
        "rendered_config_sha256": sha256(args.config_output),
        "spf_git_sha": git_sha,
        "recorded_at_unix_ns": time.time_ns(),
    }
    if args.state_output.exists():
        existing = json.loads(args.state_output.read_text())
        comparable = {
            key: value
            for key, value in existing.items()
            if key != "recorded_at_unix_ns"
        }
        expected = {
            key: value for key, value in state.items() if key != "recorded_at_unix_ns"
        }
        if comparable != expected:
            raise ValueError(
                f"existing physical-state record differs: {args.state_output}"
            )
    else:
        write_atomic(
            args.state_output, json.dumps(state, indent=2, sort_keys=True) + "\n"
        )

    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
