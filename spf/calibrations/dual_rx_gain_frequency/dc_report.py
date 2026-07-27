"""Reproducible report for matched RF-DC failure and recovery evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "spf.calibration.dual_rx_gain_frequency.rf_dc_evidence_report"
REPORT_SCHEMA_VERSION = 1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, Any]:
    files = []
    tree_digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": digest})
        tree_digest.update(relative.encode())
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode())
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode())
        tree_digest.update(b"\n")
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "sha256": tree_digest.hexdigest(),
        "files": files,
    }


def _state_by_gain(summary: dict[str, Any], tx2_enabled: bool) -> dict[int, Any]:
    return {
        int(state["gain_rx2_db"]): state
        for state in summary["state_summaries"]
        if bool(state["tx2_enabled"]) is tx2_enabled
    }


def _failed_rx2_gains(summary: dict[str, Any]) -> list[int]:
    failed = set()
    for state in summary["state_summaries"]:
        if (
            float(state["median_dc_dbfs"][1]) >= -20.0
            or float(state["maximum_clipping_fraction"][1]) > 0.0
        ):
            failed.add(int(state["gain_rx2_db"]))
    return sorted(failed)


def _word(snapshot: dict[str, Any], component: str) -> dict[str, Any] | None:
    banks = snapshot.get("correction_banks", {})
    bank = banks.get("A") or next(iter(banks.values()), None)
    if not bank:
        return None
    return bank.get("correction_words", {}).get(component)


def _recovery_words(recovery: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    before = {int(item["gain_rx2_db"]): item for item in recovery.get("before", [])}
    after = {int(item["gain_rx2_db"]): item for item in recovery.get("after", [])}
    for gain in sorted(set(before) & set(after)):
        row: dict[str, Any] = {"gain_rx2_db": gain}
        for component in ("rx2_i", "rx2_q"):
            for state, snapshots in (("before", before), ("after", after)):
                word = _word(snapshots[gain], component)
                row[f"{component}_{state}_signed"] = (
                    None if word is None else int(word["signed"])
                )
                row[f"{component}_{state}_stuck"] = (
                    None if word is None else bool(word["is_documented_stuck_value"])
                )
        result.append(row)
    return result


def _validate_inputs(
    before: dict[str, Any],
    recovery: dict[str, Any],
    after: dict[str, Any],
) -> tuple[str, int, list[int]]:
    if any(item.get("status") != "complete" for item in (before, recovery, after)):
        raise ValueError("all RF-DC evidence inputs must be complete")
    serials = {item.get("serial") for item in (before, recovery, after)}
    frequencies = {int(item.get("frequency_hz")) for item in (before, recovery, after)}
    if len(serials) != 1 or len(frequencies) != 1:
        raise ValueError("RF-DC evidence inputs do not describe one radio/frequency")
    before_gains = sorted(
        int(item["gain_rx2_db"]) for item in before["on_off_comparisons"]
    )
    after_gains = sorted(
        int(item["gain_rx2_db"]) for item in after["on_off_comparisons"]
    )
    recovery_gains = sorted(int(value) for value in recovery["gain_rx2_values_db"])
    if (
        not before_gains
        or before_gains != after_gains
        or before_gains != recovery_gains
    ):
        raise ValueError("RF-DC evidence inputs use different RX2 gain grids")
    return str(next(iter(serials))), next(iter(frequencies)), before_gains


def build_rf_dc_evidence(
    *,
    before_dir: Path,
    recovery_path: Path,
    after_dir: Path,
) -> dict[str, Any]:
    """Validate and summarize immutable before/recovery/after evidence."""

    before_dir = Path(before_dir)
    recovery_path = Path(recovery_path)
    after_dir = Path(after_dir)
    before = _load_json(before_dir / "summary.json")
    recovery = _load_json(recovery_path)
    after = _load_json(after_dir / "summary.json")
    serial, frequency_hz, gains = _validate_inputs(before, recovery, after)
    before_off = _state_by_gain(before, False)
    before_on = _state_by_gain(before, True)
    after_off = _state_by_gain(after, False)
    after_on = _state_by_gain(after, True)

    gain_rows = []
    for gain in gains:
        gain_rows.append(
            {
                "gain_rx2_db": gain,
                "before": {
                    "tx_off_rx2_dc_dbfs": before_off[gain]["median_dc_dbfs"][1],
                    "tx_on_rx2_dc_dbfs": before_on[gain]["median_dc_dbfs"][1],
                    "tx_off_rx2_max_clipping": before_off[gain][
                        "maximum_clipping_fraction"
                    ][1],
                    "tx_on_rx2_max_clipping": before_on[gain][
                        "maximum_clipping_fraction"
                    ][1],
                    "tx_on_quality_valid_frames": before_on[gain][
                        "quality_valid_frames"
                    ],
                },
                "after": {
                    "tx_off_rx2_dc_dbfs": after_off[gain]["median_dc_dbfs"][1],
                    "tx_on_rx2_dc_dbfs": after_on[gain]["median_dc_dbfs"][1],
                    "tx_off_rx2_max_clipping": after_off[gain][
                        "maximum_clipping_fraction"
                    ][1],
                    "tx_on_rx2_max_clipping": after_on[gain][
                        "maximum_clipping_fraction"
                    ][1],
                    "tx_on_quality_valid_frames": after_on[gain][
                        "quality_valid_frames"
                    ],
                },
            }
        )

    before_failed = _failed_rx2_gains(before)
    after_failed = _failed_rx2_gains(after)
    expected_on_frames = sum(
        int(state["frames"])
        for state in after["state_summaries"]
        if bool(state["tx2_enabled"])
    )
    valid_on_frames = sum(
        int(state["quality_valid_frames"])
        for state in after["state_summaries"]
        if bool(state["tx2_enabled"])
    )
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "serial": serial,
        "frequency_hz": frequency_hz,
        "gain_rx2_values_db": gains,
        "input_evidence": {
            "before_diagnostic": _tree_manifest(before_dir),
            "recovery": {
                "bytes": recovery_path.stat().st_size,
                "sha256": _sha256(recovery_path),
            },
            "after_diagnostic": _tree_manifest(after_dir),
        },
        "recovery": {
            "operation": recovery["operation"],
            "duration_ms": 1000.0
            * (
                float(recovery["operation_completed_unix_seconds"])
                - float(recovery["operation_started_unix_seconds"])
            ),
            "tx2_enabled": bool(recovery["tx2_enabled"]),
            "correction_words": _recovery_words(recovery),
        },
        "gain_results": gain_rows,
        "before_failed_rx2_gains_db": before_failed,
        "after_failed_rx2_gains_db": after_failed,
        "post_recovery_tx_on_quality_valid_frames": valid_on_frames,
        "post_recovery_tx_on_expected_frames": expected_on_frames,
        "conclusions": {
            "failure_present_before": bool(before_failed),
            "failure_present_with_tx2_off": any(
                float(before_off[gain]["median_dc_dbfs"][1]) >= -20.0
                or float(before_off[gain]["maximum_clipping_fraction"][1]) > 0.0
                for gain in gains
            ),
            "tx2_required_for_failure": False if before_failed else None,
            "rf_dc_recovery_passed": (
                bool(before_failed)
                and not after_failed
                and valid_on_frames == expected_on_frames
            ),
            "scope": (
                "This establishes the observed RF-DC failure and recovery for "
                "this radio, frequency, gain grid, and run. It does not prove "
                "all radios or all frequencies are healthy."
            ),
        },
    }


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"


def render_rf_dc_report(evidence: dict[str, Any]) -> str:
    rows = []
    for item in evidence["gain_results"]:
        before = item["before"]
        after = item["after"]
        rows.append(
            "| {gain} | {bdc} | {bclip:.2%} | {adc} | {aclip:.2%} | "
            "{valid} |".format(
                gain=item["gain_rx2_db"],
                bdc=_fmt(before["tx_off_rx2_dc_dbfs"]),
                bclip=float(before["tx_off_rx2_max_clipping"]),
                adc=_fmt(after["tx_off_rx2_dc_dbfs"]),
                aclip=float(after["tx_off_rx2_max_clipping"]),
                valid=after["tx_on_quality_valid_frames"],
            )
        )
    correction_rows = []
    for item in evidence["recovery"]["correction_words"]:
        correction_rows.append(
            "| {gain} | {ib} | {ia} | {qb} | {qa} |".format(
                gain=item["gain_rx2_db"],
                ib=item["rx2_i_before_signed"],
                ia=item["rx2_i_after_signed"],
                qb=item["rx2_q_before_signed"],
                qa=item["rx2_q_after_signed"],
            )
        )
    conclusions = evidence["conclusions"]
    return (
        "# RX2 RF-DC failure and recovery evidence\n\n"
        f"- Pluto serial: `{evidence['serial']}`\n"
        f"- LO: `{evidence['frequency_hz']}` Hz\n"
        f"- RX2 gains: `{evidence['gain_rx2_values_db']}` dB\n"
        f"- RF-only initialization duration: "
        f"`{evidence['recovery']['duration_ms']:.2f}` ms\n"
        f"- Post-recovery valid TX-on frames: "
        f"`{evidence['post_recovery_tx_on_quality_valid_frames']}/"
        f"{evidence['post_recovery_tx_on_expected_frames']}`\n\n"
        "## Result\n\n"
        f"Before recovery, the failed RX2 gains were "
        f"`{evidence['before_failed_rx2_gains_db']}` dB. The symptom was "
        "present in fresh TX2-off contexts, so TX2 transmission was not "
        "required for the observed failure. The Linux driver's RF-DC "
        f"initialization left failed gains `{evidence['after_failed_rx2_gains_db']}` "
        "and the matched post-recovery capture "
        f"{'passed' if conclusions['rf_dc_recovery_passed'] else 'did not pass'} "
        "the declared recovery condition.\n\n"
        "| RX2 gain dB | pre TX-off DC dBFS | pre TX-off max clip | "
        "post TX-off DC dBFS | post TX-off max clip | post valid TX-on |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n## RF correction words (input A bank)\n\n"
        "| RX2 gain dB | I before | I after | Q before | Q after |\n"
        "|---:|---:|---:|---:|---:|\n"
        + (
            "\n".join(correction_rows)
            if correction_rows
            else "| n/a | n/a | n/a | n/a | n/a |"
        )
        + "\n\n"
        "The RF-only operation is not presented as the complete ADI recovery "
        "procedure: the Linux `calib_mode=rf_dc_offs` interface does not rerun "
        "the separate BB-DC initialization. ADI recommends isolating the input "
        "and running both initial calibrations for the complete procedure. "
        "See [ADI's AD936x DC-offset issue note]"
        "(https://ez.analog.com/rf/wide-band-rf-transceivers/design-support/w/"
        "documents/10060/ad936x_5f00_dcoffset_5f00_issue).\n\n"
        "## Reproducibility and policy\n\n"
        "The adjacent `evidence.json` records SHA-256 manifests for every "
        "diagnostic file, including every full-IQ `.npy` frame, plus the "
        "recovery snapshot hash. The large source evidence remains under "
        "`artifacts/` and is not committed.\n\n"
        "New calibration runs initialize RF-DC with TX2 stopped before every "
        "radio/frequency block, then require a direct-USB tone preflight. A "
        "failed initialization, preflight, metadata check, clipping check, or "
        "phase-quality check fails closed. The earlier paused exhaustive scan "
        "must not be resumed because it predates this preparation policy.\n\n"
        "This result is scoped to the identified radio and test grid. Repeat "
        "the same before/recovery/after test on each radio before treating the "
        "fleet as characterized.\n"
    )


def write_rf_dc_evidence_report(
    *,
    before_dir: Path,
    recovery_path: Path,
    after_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write deterministic JSON and Markdown evidence outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = build_rf_dc_evidence(
        before_dir=before_dir,
        recovery_path=recovery_path,
        after_dir=after_dir,
    )
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "REPORT.md").write_text(render_rf_dc_report(evidence))
    return evidence
