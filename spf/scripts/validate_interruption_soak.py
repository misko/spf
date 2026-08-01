"""Validate and summarize a completed interrupted-capture soak tree."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


KERNEL_ERROR = re.compile(
    r"USB disconnect|error -71|device descriptor read|xhci.*error|I/O error|"
    r"not enough memory|Out of memory|Killed process|oom-kill|oom_reaper",
    re.IGNORECASE,
)
EXPECTED_CAPTURE_STATUS = {
    "sigint": "incomplete",
    "sigterm": "incomplete",
    "sigkill": "in_progress",
}
EXPECTED_RETURN_CODE = {"sigint": 130, "sigterm": 143, "sigkill": -9}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text())


def _strict_validate_clean_capture(
    path: Path, *, expected_frames: int, expected_receivers: int
) -> dict:
    """Load the strict IQ validator only when revalidation is requested."""

    from spf.scripts.validate_direct_usb_v7_zarr import validate_capture

    return validate_capture(path, expected_frames, expected_receivers)


def validate_soak(
    root: Path,
    *,
    expected_receivers: int,
    minimum_rounds: int,
    require_complete: bool,
    revalidate_clean_zarrs: bool = False,
) -> dict:
    if expected_receivers < 1:
        raise ValueError("expected_receivers must be positive")
    if minimum_rounds < 1:
        raise ValueError("minimum_rounds must be positive")
    rows_path = root / "rounds.tsv"
    if not rows_path.is_file():
        raise ValueError(f"missing round ledger: {rows_path}")
    with rows_path.open(newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if len(rows) < minimum_rounds:
        raise ValueError(
            f"only {len(rows)} completed rounds; require at least {minimum_rounds}"
        )
    if (root / "FAILED").exists():
        raise ValueError(f"soak recorded failure: {(root / 'FAILED').read_text().strip()}")
    if require_complete:
        pass_marker = root / "PASS"
        if not pass_marker.is_file() or pass_marker.read_text() != "PASS\n":
            raise ValueError("soak root lacks its exact PASS marker")
        if not (root / "result.env").is_file():
            raise ValueError("soak root lacks result.env")

    signals = {name: 0 for name in EXPECTED_CAPTURE_STATUS}
    serials: set[str] = set()
    interruption_frames = 0
    clean_frames = 0
    maximum_exit_seconds = 0.0
    maximum_release_probe_sessions = 0
    strictly_revalidated_clean_captures = 0
    completed_rounds = []

    for expected_round, row in enumerate(rows, start=1):
        round_number = int(row["round"])
        if round_number != expected_round:
            raise ValueError(
                f"round ledger is discontinuous: expected {expected_round}, "
                f"found {round_number}"
            )
        if int(row["status"]) != 0:
            raise ValueError(f"round {round_number} has nonzero status")
        specifications = row["cases"].split()
        round_root = root / f"round-{round_number:03d}"
        campaigns = sorted(round_root.glob("*_rover*"))
        if len(campaigns) != 1:
            raise ValueError(
                f"round {round_number} has {len(campaigns)} campaign directories"
            )
        campaign = campaigns[0]
        if (campaign / "PASS").read_text() != "PASS\n":
            raise ValueError(f"round {round_number} campaign lacks PASS")

        round_serials: set[str] | None = None
        for case_index, specification in enumerate(specifications, start=1):
            signal_name, threshold_text = specification.split(":", 1)
            threshold = int(threshold_text)
            if signal_name not in EXPECTED_CAPTURE_STATUS:
                raise ValueError(f"unsupported signal in ledger: {signal_name}")
            case_root = campaign / (
                f"case-{case_index:02d}-{signal_name}-{threshold}"
            )
            report = _load_json(
                case_root
                / "reports"
                / f"interruption-{signal_name}-{threshold}-records.json"
            )
            if report.get("status") != "pass":
                raise ValueError(f"case did not pass: {case_root}")
            if report.get("signal") != signal_name:
                raise ValueError(f"signal mismatch: {case_root}")
            if report.get("minimum_records") != threshold:
                raise ValueError(f"threshold mismatch: {case_root}")
            if report.get("capture_status") != EXPECTED_CAPTURE_STATUS[signal_name]:
                raise ValueError(f"capture status mismatch: {case_root}")
            if report.get("return_code") != EXPECTED_RETURN_CODE[signal_name]:
                raise ValueError(f"return code mismatch: {case_root}")
            counts = report.get("committed_after_interrupt", [])
            if len(counts) != expected_receivers or min(counts) < threshold:
                raise ValueError(f"unsafe committed prefix: {case_root}")
            case_serials = set(report.get("serials", []))
            if len(case_serials) != expected_receivers:
                raise ValueError(f"radio identity count mismatch: {case_root}")
            release_probe_sessions = report.get("release_probe_sessions")
            if not isinstance(release_probe_sessions, dict) or set(
                release_probe_sessions
            ) != case_serials:
                raise ValueError(f"release probe identity mismatch: {case_root}")
            for sessions in release_probe_sessions.values():
                if (
                    isinstance(sessions, bool)
                    or not isinstance(sessions, int)
                    or not 1 <= sessions <= 3
                ):
                    raise ValueError(f"invalid release probe sessions: {case_root}")
                maximum_release_probe_sessions = max(
                    maximum_release_probe_sessions, sessions
                )
            if round_serials is None:
                round_serials = case_serials
            elif case_serials != round_serials:
                raise ValueError(f"radio identity changed within round {round_number}")
            delta = case_root / "dmesg-delta.txt"
            if not delta.is_file():
                raise ValueError(f"missing kernel delta: {delta}")
            if KERNEL_ERROR.search(delta.read_text()):
                raise ValueError(f"kernel USB/memory error in {delta}")
            signals[signal_name] += 1
            interruption_frames += sum(counts)
            maximum_exit_seconds = max(
                maximum_exit_seconds, float(report.get("exit_seconds", 0.0))
            )

        validations = list((campaign / "clean-recovery").glob("validation.json"))
        if len(validations) != 1:
            raise ValueError(
                f"round {round_number} has {len(validations)} clean validations"
            )
        validation = _load_json(validations[0])
        if validation.get("status") != "pass" or validation.get("data_version") != 7:
            raise ValueError(f"clean V7 recovery failed in round {round_number}")
        if validation.get("receiver_count") != expected_receivers:
            raise ValueError(f"clean receiver count mismatch in round {round_number}")
        receivers = validation.get("receivers", {})
        validation_serials = {receiver["serial"] for receiver in receivers.values()}
        if validation_serials != round_serials:
            raise ValueError(f"clean recovery changed radio identity in round {round_number}")
        if revalidate_clean_zarrs:
            clean_zarrs = sorted((campaign / "clean-recovery").glob("*.zarr"))
            if len(clean_zarrs) != 1:
                raise ValueError(
                    f"round {round_number} has {len(clean_zarrs)} clean V7 stores"
                )
            strict_validation = _strict_validate_clean_capture(
                clean_zarrs[0],
                expected_frames=100,
                expected_receivers=expected_receivers,
            )
            strict_receivers = strict_validation.get("receivers", {})
            strict_serials = {
                receiver["serial"] for receiver in strict_receivers.values()
            }
            if strict_serials != round_serials:
                raise ValueError(
                    f"strict clean recovery changed radio identity in round "
                    f"{round_number}"
                )
            strictly_revalidated_clean_captures += 1
        round_clean_frames = sum(receiver["frames"] for receiver in receivers.values())
        clean_frames += round_clean_frames
        serials.update(validation_serials)
        completed_rounds.append(
            {
                "round": round_number,
                "cases": specifications,
                "clean_frames": round_clean_frames,
                "serials": sorted(validation_serials),
            }
        )

    return {
        "status": "pass",
        "root": str(root),
        "rounds": len(rows),
        "expected_receivers": expected_receivers,
        "signals": signals,
        "interruption_committed_frames": interruption_frames,
        "clean_recovery_frames": clean_frames,
        "maximum_signal_exit_seconds": maximum_exit_seconds,
        "maximum_release_probe_sessions": maximum_release_probe_sessions,
        "strictly_revalidated_clean_captures": strictly_revalidated_clean_captures,
        "serials": sorted(serials),
        "completed_rounds": completed_rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-receivers", type=int, required=True)
    parser.add_argument("--minimum-rounds", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--revalidate-clean-zarrs",
        action="store_true",
        help=(
            "re-read every clean recovery with the current strict V7 IQ, metadata, "
            "sequence, provenance and channel-distinctness validator"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_soak(
        args.root,
        expected_receivers=args.expected_receivers,
        minimum_rounds=args.minimum_rounds,
        require_complete=args.require_complete,
        revalidate_clean_zarrs=args.revalidate_clean_zarrs,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
