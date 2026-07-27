"""CLI for direct-USB V7 dual-RX gain/frequency phase calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from .comparative_analysis import write_comparative_bundle
from .config import build_schedule, group_schedule_by_frequency
from .dc_offset import inspect_radio_rf_dc
from .model import fit_dataset, write_model
from .report import write_analysis_bundle
from .runner import load_calibration_document, probe_loopback, run_calibration
from .validate import validate_dataset, write_validation_report


def _run(args) -> int:
    _, config = load_calibration_document(args.config)
    total = config.measurements_per_radio * (
        len(args.serial) if args.serial else args.expected_radios
    )
    progress_bar = tqdm(
        total=total,
        unit="frame",
        disable=not sys.stderr.isatty(),
    )

    def progress(serial, entry, completed, expected):
        progress_bar.total = expected
        progress_bar.n = completed
        progress_bar.set_postfix(
            serial=serial[-6:],
            epoch=entry.epoch,
            frequency_mhz=f"{entry.lo_frequency_hz / 1e6:.3f}",
            gain1=entry.gain_rx1_db,
            gain2=entry.gain_rx2_db,
        )
        progress_bar.refresh()

    try:
        result = run_calibration(
            config_path=args.config,
            output_dir=args.output,
            ready_manifest_path=args.ready_manifest,
            serials=tuple(args.serial) if args.serial else None,
            progress=progress,
        )
    finally:
        progress_bar.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


def _validate(args) -> int:
    _, config = load_calibration_document(args.config)
    report = validate_dataset(
        args.dataset,
        config=config,
        expected_serial=args.serial,
        recompute_iq=not args.no_recompute_iq,
    )
    if args.output:
        write_validation_report(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cells"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


def _fit(args) -> int:
    _, config = load_calibration_document(args.config)
    model = fit_dataset(args.dataset, config=config)
    if args.output:
        write_model(args.output, model)
    print(
        json.dumps(
            {key: value for key, value in model.items() if key != "frequency_models"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if model["quality_valid_observations"] else 1


def _report(args) -> int:
    summary = write_analysis_bundle(
        validation_path=args.validation,
        model_path=args.model,
        output_dir=args.output_dir,
        plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _schedule(args) -> int:
    _, config = load_calibration_document(args.config)
    schedule = build_schedule(config)
    blocks = group_schedule_by_frequency(schedule)
    result = {
        "measurements_per_radio": len(schedule),
        "frequency_blocks": [
            {
                "epoch": block[0].epoch,
                "frequency_index": block[0].frequency_index,
                "frequency_hz": block[0].lo_frequency_hz,
                "gain_pairs": len(block),
            }
            for block in blocks
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _probe(args) -> int:
    result = probe_loopback(
        config_path=args.config,
        serial=args.serial,
        frequency_hz=args.frequency_hz,
        gain_db=args.gain_db,
        tx_channel=args.tx_channel,
        minimum_on_off_delta_db=args.minimum_on_off_delta_db,
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def _dc_registers(args) -> int:
    result = inspect_radio_rf_dc(serial=args.serial, uri=args.uri)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _compare_models(args) -> int:
    result = write_comparative_bundle(
        config_path=args.config,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run or resume calibration")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--serial", action="append", default=[])
    run_parser.add_argument(
        "--ready-manifest",
        type=Path,
        default=Path("/run/spf/direct_usb_ready.json"),
    )
    run_parser.add_argument(
        "--expected-radios",
        type=int,
        default=2,
        help="progress estimate when serials come from the ready manifest",
    )
    run_parser.set_defaults(function=_run)

    validate_parser = subparsers.add_parser(
        "validate", help="strictly validate one serial-specific V7 dataset"
    )
    validate_parser.add_argument("--config", type=Path, required=True)
    validate_parser.add_argument("--dataset", type=Path, required=True)
    validate_parser.add_argument("--serial")
    validate_parser.add_argument("--output", type=Path)
    validate_parser.add_argument("--no-recompute-iq", action="store_true")
    validate_parser.set_defaults(function=_validate)

    fit_parser = subparsers.add_parser(
        "fit", help="fit and cross-validate a circular additive phase model"
    )
    fit_parser.add_argument("--config", type=Path, required=True)
    fit_parser.add_argument("--dataset", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path)
    fit_parser.set_defaults(function=_fit)

    report_parser = subparsers.add_parser(
        "report", help="write Markdown, JSON, and phase-surface plots"
    )
    report_parser.add_argument("--validation", type=Path, required=True)
    report_parser.add_argument("--model", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser.add_argument("--no-plots", action="store_true")
    report_parser.set_defaults(function=_report)

    schedule_parser = subparsers.add_parser(
        "schedule", help="render the deterministic epoch/frequency schedule"
    )
    schedule_parser.add_argument("--config", type=Path, required=True)
    schedule_parser.set_defaults(function=_schedule)

    probe_parser = subparsers.add_parser(
        "probe", help="qualify TX2-on versus TX2-off loopback tone dominance"
    )
    probe_parser.add_argument("--config", type=Path, required=True)
    probe_parser.add_argument("--serial", required=True)
    probe_parser.add_argument("--frequency-hz", type=int)
    probe_parser.add_argument("--gain-db", type=int)
    probe_parser.add_argument(
        "--tx-channel",
        type=int,
        choices=(0, 1),
        default=1,
        help="diagnostic channel; production calibration requires TX2/channel 1",
    )
    probe_parser.add_argument("--minimum-on-off-delta-db", type=float, default=20.0)
    probe_parser.add_argument("--output", type=Path)
    probe_parser.set_defaults(function=_probe)

    dc_parser = subparsers.add_parser(
        "dc-registers",
        help="read and decode AD9361 RF DC correction words without mutation",
    )
    dc_selector = dc_parser.add_mutually_exclusive_group(required=True)
    dc_selector.add_argument("--serial")
    dc_selector.add_argument("--uri")
    dc_parser.add_argument("--output", type=Path)
    dc_parser.set_defaults(function=_dc_registers)

    compare_parser = subparsers.add_parser(
        "compare-models",
        help=(
            "write reproducible held-out model comparisons and preliminary "
            "per-radio calibrations from an existing V7 run"
        ),
    )
    compare_parser.add_argument("--config", type=Path, required=True)
    compare_parser.add_argument("--artifact-root", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.set_defaults(function=_compare_models)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
