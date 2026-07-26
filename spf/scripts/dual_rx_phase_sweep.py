"""Characterize Pluto+ RX1/RX2 phase across manual-gain combinations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from spf.bench.dual_rx_phase import (
    PlutoDualRxRadio,
    SweepConfig,
    ToneQualityThresholds,
    generate_report,
    resolve_pluto_uri,
    run_sweep,
    select_gain_values,
)


def _add_radio_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--uri", help="libiio URI, for example usb:1.9.5")
    selection.add_argument("--serial", help="stable Pluto USB serial")
    parser.add_argument("--lo-hz", type=int, required=True)
    parser.add_argument("--sample-rate-hz", type=int, default=2_000_000)
    parser.add_argument("--bandwidth-hz", type=int, default=1_000_000)
    parser.add_argument("--tone-offset-hz", type=float, default=100_000.0)
    parser.add_argument("--tone-search-width-hz", type=float, default=25_000.0)
    parser.add_argument("--buffer-size", type=int, default=65_536)
    parser.add_argument("--transient-samples", type=int, default=1_024)
    parser.add_argument("--phase-segments", type=int, default=8)
    parser.add_argument(
        "--phase-inversion-mitigation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--qec-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _config_from_args(args) -> SweepConfig:
    return SweepConfig(
        lo_hz=args.lo_hz,
        sample_rate_hz=args.sample_rate_hz,
        bandwidth_hz=args.bandwidth_hz,
        expected_tone_offset_hz=args.tone_offset_hz,
        tone_search_width_hz=args.tone_search_width_hz,
        buffer_size=args.buffer_size,
        transient_samples=args.transient_samples,
        phase_segments=args.phase_segments,
        repetitions=getattr(args, "repetitions", 3),
        captures_per_pair=getattr(args, "captures_per_pair", 1),
        random_seed=getattr(args, "random_seed", 20260726),
        randomize_pairs=getattr(args, "randomize_pairs", True),
        settle_seconds=getattr(args, "settle_ms", 25.0) / 1000.0,
        flush_buffers=getattr(args, "flush_buffers", 2),
        max_retries=getattr(args, "max_retries", 1),
        min_quality_valid_per_cell=getattr(args, "min_valid_per_cell", 2),
        max_across_repeat_phase_std_deg=getattr(
            args, "max_across_repeat_phase_std_deg", 5.0
        ),
        enable_phase_inversion_mitigation=args.phase_inversion_mitigation,
        enable_qec_tracking=args.qec_tracking,
        source_power_dbm=getattr(args, "source_power_dbm", None),
        setup_label=getattr(args, "setup_label", ""),
        notes=getattr(args, "notes", ""),
        quality=ToneQualityThresholds(
            min_tone_snr_db=getattr(args, "min_tone_snr_db", 15.0),
            min_tone_dbfs=getattr(args, "min_tone_dbfs", -70.0),
            max_tone_dbfs=getattr(args, "max_tone_dbfs", -3.0),
            max_clipping_fraction=getattr(args, "max_clipping_fraction", 0.0),
            min_coherence=getattr(args, "min_coherence", 0.98),
            max_within_capture_phase_std_deg=getattr(args, "max_phase_std_deg", 5.0),
        ),
    )


def _parse_explicit_gains(value: str | None):
    if value is None:
        return None
    try:
        return [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--gains must be comma-separated integer dB values"
        ) from error


def discover(args) -> int:
    uri = resolve_pluto_uri(uri=args.uri, serial=args.serial)
    config = _config_from_args(args)
    radio = PlutoDualRxRadio(uri, config)
    try:
        payload = {
            "identity": radio.identity(),
            "available_gains_db": radio.available_gains(),
            "config": config.as_json(),
            "source_frequency_hz": config.lo_hz + config.expected_tone_offset_hz,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        radio.close()
    return 0


def run(args) -> int:
    uri = resolve_pluto_uri(uri=args.uri, serial=args.serial)
    config = _config_from_args(args)
    radio = PlutoDualRxRadio(uri, config)
    try:
        gains = select_gain_values(
            radio.available_gains(),
            gain_start=args.gain_start,
            gain_end=args.gain_end,
            gain_step=args.gain_step,
            explicit=_parse_explicit_gains(args.gains),
        )
        measurement_count = (
            len(gains) ** 2 * config.repetitions * config.captures_per_pair
        )
        minimum_seconds = measurement_count * (
            config.settle_seconds
            + (config.flush_buffers + 1) * config.buffer_size / config.sample_rate_hz
        )
        print(
            f"URI={uri} gains={gains[0]}..{gains[-1]} ({len(gains)} states), "
            f"pairs={len(gains) ** 2}, measurements={measurement_count}, "
            f"ideal RF-time floor={minimum_seconds / 60:.1f} min"
        )
        print(
            f"Expected source frequency: "
            f"{config.lo_hz + config.expected_tone_offset_hz:.0f} Hz"
        )
        if not args.yes:
            response = input("Start/resume this sweep? [y/N] ")
            if response.lower() not in ("y", "yes"):
                print("Cancelled.")
                return 2
        progress_bar = tqdm(total=measurement_count, unit="capture")

        def progress(completed, total, entry):
            progress_bar.n = completed
            progress_bar.set_postfix(rep=entry[0], gain_rx1=entry[1], gain_rx2=entry[2])
            progress_bar.refresh()

        try:
            report = run_sweep(
                radio,
                config,
                gains,
                args.output,
                progress=progress,
            )
        except KeyboardInterrupt:
            report = generate_report(args.output)
            print("\nInterrupted; checkpoint and partial report are complete.")
        finally:
            progress_bar.close()
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "radio_serial",
                        "expected_measurements",
                        "completed_measurements",
                        "quality_valid_measurements",
                        "valid_cells",
                        "passing_cells",
                        "total_cells",
                        "phase_delta_span_deg",
                    )
                },
                indent=2,
            )
        )
        return 0 if report["status"] == "pass" else 1
    finally:
        radio.close()


def report(args) -> int:
    result = generate_report(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser(
        "discover", help="configure one Pluto and show identity/gain range"
    )
    _add_radio_arguments(discover_parser)
    discover_parser.set_defaults(function=discover)

    run_parser = subparsers.add_parser("run", help="run or resume a gain-pair sweep")
    _add_radio_arguments(run_parser)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--repetitions", type=int, default=3)
    run_parser.add_argument("--captures-per-pair", type=int, default=1)
    run_parser.add_argument("--random-seed", type=int, default=20260726)
    run_parser.add_argument(
        "--randomize-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument("--settle-ms", type=float, default=25.0)
    run_parser.add_argument("--flush-buffers", type=int, default=2)
    run_parser.add_argument("--max-retries", type=int, default=1)
    run_parser.add_argument(
        "--min-valid-per-cell",
        type=int,
        default=2,
        help="minimum quality-valid repeats required for every gain pair",
    )
    run_parser.add_argument(
        "--max-across-repeat-phase-std-deg",
        type=float,
        default=5.0,
        help="maximum circular phase standard deviation for a passing gain pair",
    )
    run_parser.add_argument(
        "--source-power-dbm",
        type=float,
        help="generator output before splitter/cables, for provenance",
    )
    run_parser.add_argument(
        "--setup-label",
        default="",
        help="short identifier for the splitter/cable/source setup",
    )
    run_parser.add_argument(
        "--notes",
        default="",
        help="free-form run notes stored in the immutable manifest",
    )
    run_parser.add_argument("--gain-start", type=int)
    run_parser.add_argument("--gain-end", type=int)
    run_parser.add_argument("--gain-step", type=int)
    run_parser.add_argument(
        "--gains", help="explicit comma-separated manual gains for a smoke sweep"
    )
    run_parser.add_argument("--min-tone-snr-db", type=float, default=15.0)
    run_parser.add_argument("--min-tone-dbfs", type=float, default=-70.0)
    run_parser.add_argument("--max-tone-dbfs", type=float, default=-3.0)
    run_parser.add_argument("--max-clipping-fraction", type=float, default=0.0)
    run_parser.add_argument("--min-coherence", type=float, default=0.98)
    run_parser.add_argument("--max-phase-std-deg", type=float, default=5.0)
    run_parser.add_argument(
        "--yes", action="store_true", help="skip the sweep confirmation prompt"
    )
    run_parser.set_defaults(function=run)

    report_parser = subparsers.add_parser(
        "report", help="regenerate reports from a checkpoint"
    )
    report_parser.add_argument("output", type=Path)
    report_parser.set_defaults(function=report)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
