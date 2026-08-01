"""Validate and summarize a completed interrupted-capture soak tree."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


KERNEL_ERROR = re.compile(
    r"USB disconnect|error -71|device descriptor read|xhci.*error|I/O error",
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


def validate_soak(
    root: Path,
    *,
    expected_receivers: int,
    minimum_rounds: int,
    require_complete: bool,
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
    if require_complete:
        if (root / "PASS").read_text() != "PASS\n":
            raise ValueError("soak root lacks its exact PASS marker")
        if not (root / "result.env").is_file():
            raise ValueError("soak root lacks result.env")
    if (root / "FAILED").exists():
        raise ValueError(f"soak recorded failure: {(root / 'FAILED').read_text().strip()}")

    signals = {name: 0 for name in EXPECTED_CAPTURE_STATUS}
    serials: set[str] = set()
    interruption_frames = 0
    clean_frames = 0
    maximum_exit_seconds = 0.0
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
            if round_serials is None:
                round_serials = case_serials
            elif case_serials != round_serials:
                raise ValueError(f"radio identity changed within round {round_number}")
            delta = case_root / "dmesg-delta.txt"
            if not delta.is_file():
                raise ValueError(f"missing kernel delta: {delta}")
            if KERNEL_ERROR.search(delta.read_text()):
                raise ValueError(f"kernel USB error in {delta}")
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
        "serials": sorted(serials),
        "completed_rounds": completed_rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-receivers", type=int, required=True)
    parser.add_argument("--minimum-rounds", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_soak(
        args.root,
        expected_receivers=args.expected_receivers,
        minimum_rounds=args.minimum_rounds,
        require_complete=args.require_complete,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
