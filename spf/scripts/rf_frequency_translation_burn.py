"""USB-only closed-loop RF frequency-translation qualification and burn-in."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import usb1

from spf.bench.dual_rx_phase import ToneQualityThresholds, analyze_common_tone
from spf.bench.rf_frequency_translation import (
    DEFAULT_DDS_OFFSET_HZ,
    DEFAULT_EMITTED_FREQUENCIES_HZ,
    DEFAULT_RX_LO_OFFSETS_HZ,
    TranslationCell,
    build_translation_cells,
    parse_hz_list,
    spectral_dominance,
)
from spf.direct_radio.iio_metadata import IioMetadataRx
from spf.direct_radio.tandem_agc import (
    RadioMetadataV5,
    TandemMode,
    TandemSessionRequestV1,
    TandemState,
)
from spf.direct_radio.usb_protocol import MetadataFlags
from spf.scripts.mute_pluto_tx import mute_sdr_tx, validate_loopback_safety

PLUTO_VENDOR_ID = 0x0456
PLUTO_PRODUCT_ID = 0xB673
EXPECTED_FIRMWARE = "v0.41-plutoplus-spf-tandem-agc-v8-rc1"
SAMPLE_RATE_HZ = 3_000_000
BANDWIDTH_HZ = 3_000_000
SAMPLES_PER_CHANNEL = 65_536
INITIAL_GAIN_DB = 40
SETTLE_SECONDS = 0.10
TEMPERATURE_DEADLINE_SECONDS = 2.0
MAX_FREQUENCY_ERROR_HZ = 250.0
MIN_GLOBAL_DOMINANCE_DB = 6.0
MIN_MIRROR_REJECTION_DB = 10.0
DEFAULT_TX_GAIN_DB = -30.0
DEFAULT_DDS_SCALE = 0.25
UNSAFE_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
    | MetadataFlags.GAIN_OBSERVATION_OVERFLOW
)
QUALITY_THRESHOLDS = ToneQualityThresholds(
    # The cabled AD9361 DDS path is spur-limited in some retune states. Keep
    # the repository's established 6 dB floor while independently enforcing
    # signed frequency, image rejection, dominance, coherence, and clipping.
    min_tone_snr_db=6.0,
    min_tone_dbfs=-75.0,
    max_tone_dbfs=-3.0,
    max_clipping_fraction=0.0,
    min_coherence=0.98,
    max_within_capture_phase_std_deg=5.0,
)


@dataclass(frozen=True, slots=True)
class UsbRadio:
    serial: str
    bus: int
    address: int
    port_path: tuple[int, ...]

    @property
    def uri(self) -> str:
        return f"usb:{self.bus}.{self.address}.5"


def discover_usb_radios() -> tuple[UsbRadio, ...]:
    """Discover runtime Pluto USB devices without scanning network backends."""

    context = usb1.USBContext()
    context.open()
    radios: list[UsbRadio] = []
    try:
        for device in context.getDeviceIterator(skip_on_error=True):
            if device.getDeviceAddress() == 0:
                continue
            if (
                device.getVendorID() != PLUTO_VENDOR_ID
                or device.getProductID() != PLUTO_PRODUCT_ID
            ):
                continue
            radios.append(
                UsbRadio(
                    serial=device.getSerialNumber(),
                    bus=device.getBusNumber(),
                    address=device.getDeviceAddress(),
                    port_path=tuple(device.getPortNumberList()),
                )
            )
    finally:
        context.close()
    return tuple(
        sorted(radios, key=lambda item: (item.bus, item.port_path, item.serial))
    )


def _close_context(sdr: Any) -> None:
    close = getattr(sdr._ctx, "close", None)
    if callable(close):
        close()


def _open_attested_sdr(radio: UsbRadio, expected_firmware: str):
    import adi

    sdr = adi.ad9361(uri=radio.uri)
    if sdr._ctx.attrs.get("hw_serial") != radio.serial:
        _close_context(sdr)
        raise RuntimeError(f"{radio.uri}: USB serial attestation failed")
    actual_firmware = sdr._ctx.attrs.get("fw_version")
    if actual_firmware != expected_firmware:
        _close_context(sdr)
        raise RuntimeError(
            f"{radio.serial}: firmware {actual_firmware!r} != {expected_firmware!r}"
        )
    if sdr._ctx.attrs.get("iio,buffer-metadata") != "2":
        _close_context(sdr)
        raise RuntimeError(f"{radio.serial}: ABI-2 metadata is unavailable")
    if sdr._ctx.find_device("tandem-agc") is None:
        _close_context(sdr)
        raise RuntimeError(f"{radio.serial}: tandem-agc device is unavailable")
    return sdr


def _configure_static(sdr: Any) -> None:
    mute_sdr_tx(sdr)
    sdr.rx_destroy_buffer()
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = SAMPLE_RATE_HZ
    sdr.rx_rf_bandwidth = BANDWIDTH_HZ
    sdr.tx_rf_bandwidth = BANDWIDTH_HZ
    sdr.rx_buffer_size = SAMPLES_PER_CHANNEL
    sdr._rxadc.set_kernel_buffers_count(2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = INITIAL_GAIN_DB
    sdr.rx_hardwaregain_chan1 = INITIAL_GAIN_DB


def _tone_analysis(signal: np.ndarray, expected_if_hz: int) -> dict[str, Any]:
    analysis = analyze_common_tone(
        signal,
        sample_rate_hz=SAMPLE_RATE_HZ,
        expected_tone_offset_hz=expected_if_hz,
        tone_search_width_hz=25_000,
        transient_samples=1_024,
        phase_segments=8,
        thresholds=QUALITY_THRESHOLDS,
    )
    dominance = spectral_dominance(
        np.asarray(signal)[:, 1_024:],
        sample_rate_hz=SAMPLE_RATE_HZ,
        expected_if_hz=expected_if_hz,
    )
    if not analysis["quality_valid"]:
        raise RuntimeError(f"tone quality failed: {analysis}")
    if abs(analysis["tone_frequency_error_hz"]) > MAX_FREQUENCY_ERROR_HZ:
        raise RuntimeError(f"signed IF frequency error is too large: {analysis}")
    if dominance["global_dominance_db"] < MIN_GLOBAL_DOMINANCE_DB:
        raise RuntimeError(f"expected IF is not the dominant non-DC peak: {dominance}")
    if dominance["mirror_rejection_db"] < MIN_MIRROR_REJECTION_DB:
        raise RuntimeError(f"signed IF image rejection is too low: {dominance}")
    return {**analysis, **dominance}


def _arm_fixed_emitter(
    sdr: Any, cell: TranslationCell, *, tx_gain_db: float, dds_scale: float
) -> dict[str, Any]:
    """Arm TX2 once and prove it before the RX-only sweep begins."""

    sdr.rx_destroy_buffer()
    sdr.rx_lo = cell.rx_lo_hz
    sdr.tx_lo = cell.tx_lo_hz
    if abs(int(sdr.rx_lo) - cell.rx_lo_hz) >= 10:
        raise RuntimeError("initial RX LO readback mismatch")
    if abs(int(sdr.tx_lo) - cell.tx_lo_hz) >= 10:
        raise RuntimeError("TX LO readback mismatch")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            mute_sdr_tx(sdr)
            sdr._ctrl.attrs["calib_mode"].value = "tx_quad"
            sdr.tx_hardwaregain_chan0 = -80
            sdr.tx_hardwaregain_chan1 = tx_gain_db
            sdr.dds_single_tone(cell.dds_offset_hz, dds_scale, channel=1)
            time.sleep(0.25)
            signal = np.asarray(sdr.rx())
            sdr.rx_destroy_buffer()
            analysis = _tone_analysis(signal, cell.expected_if_hz)
            return {"arm_attempts": attempt, "preflight": analysis}
        except Exception as error:  # noqa: BLE001 - bounded hardware arm retry
            last_error = error
    mute_sdr_tx(sdr)
    raise RuntimeError(f"TX2 emitter did not arm after three attempts: {last_error}")


def _validate_metadata(metadata: RadioMetadataV5) -> None:
    if metadata.tandem_state is not TandemState.ARMED_HOLD:
        raise RuntimeError(f"metadata tandem state is {metadata.tandem_state}")
    if metadata.tandem_fault_flags or metadata.flags & UNSAFE_FLAGS:
        raise RuntimeError(
            f"unsafe metadata: tandem=0x{metadata.tandem_fault_flags:x} "
            f"flags=0x{int(metadata.flags):x}"
        )
    if metadata.rx1_gain_index != metadata.rx2_gain_index:
        raise RuntimeError("paired RX gain indices diverged")
    if metadata.gain_events or metadata.tandem_transition_count:
        raise RuntimeError("HOLD metadata unexpectedly reports a gain transition")


def _capture_cell(
    sdr: Any,
    cell: TranslationCell,
    *,
    emitted_frequency_hz: int,
) -> dict[str, Any]:
    retune_started = time.monotonic_ns()
    sdr.rx_lo = cell.rx_lo_hz
    time.sleep(SETTLE_SECONDS)
    retune_elapsed_ns = time.monotonic_ns() - retune_started
    actual_rx_lo_hz = int(sdr.rx_lo)
    actual_tx_lo_hz = int(sdr.tx_lo)
    if abs(actual_rx_lo_hz - cell.rx_lo_hz) >= 10:
        raise RuntimeError("RX LO readback mismatch")
    if abs(actual_tx_lo_hz - cell.tx_lo_hz) >= 10:
        raise RuntimeError("fixed TX LO changed during RX sweep")

    receiver = IioMetadataRx(
        sdr,
        sample_rate_hz=SAMPLE_RATE_HZ,
        samples_per_channel=SAMPLES_PER_CHANNEL,
        tandem_request=TandemSessionRequestV1(
            mode=TandemMode.HOLD,
            initial_gain_db=INITIAL_GAIN_DB,
        ),
    )
    receiver.open()
    try:
        # The first complete frame after each retune is deliberately discarded.
        receiver.capture()
        capture_started = time.monotonic_ns()
        signal, metadata, capture_time = receiver.capture()
        _validate_metadata(metadata)
        deadline = time.monotonic() + TEMPERATURE_DEADLINE_SECONDS
        while (
            metadata.ad9361_temperature_mdeg_c is None and time.monotonic() < deadline
        ):
            signal, metadata, capture_time = receiver.capture()
            _validate_metadata(metadata)
        capture_elapsed_ns = time.monotonic_ns() - capture_started
    finally:
        receiver.close()
    temperature = metadata.ad9361_temperature_mdeg_c
    if temperature is None:
        raise RuntimeError("no valid cached AD9361 temperature within two seconds")
    if not -40_000 <= temperature <= 125_000:
        raise RuntimeError(
            f"AD9361 temperature is outside physical range: {temperature}"
        )

    analysis = _tone_analysis(signal, cell.expected_if_hz)
    return {
        **asdict(cell),
        "emitted_frequency_hz": emitted_frequency_hz,
        "actual_rx_lo_hz": actual_rx_lo_hz,
        "actual_tx_lo_hz": actual_tx_lo_hz,
        "retune_elapsed_ms": retune_elapsed_ns / 1_000_000,
        "capture_elapsed_ms": capture_elapsed_ns / 1_000_000,
        "temperature_mdeg_c": temperature,
        "ownership_epoch": metadata.ownership_epoch,
        "buffer_sequence": metadata.buffer_sequence,
        "first_sample_sequence": metadata.first_sample_sequence,
        "tandem_fault_flags": metadata.tandem_fault_flags,
        "capture_time": capture_time,
        "tone": analysis,
    }


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _append_jsonl(path: Path, document: dict[str, Any]) -> None:
    with path.open("a") as stream:
        json.dump(document, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record_cell(
    report: dict[str, Any], cells_path: Path, result: dict[str, Any]
) -> None:
    _append_jsonl(cells_path, result)
    summary = report["cell_summary"]
    summary["count"] += 1
    serial = result["serial"]
    summary["count_by_serial"][serial] = summary["count_by_serial"].get(serial, 0) + 1
    metrics = {
        "absolute_frequency_error_hz": abs(result["tone"]["tone_frequency_error_hz"]),
        "tone_snr_db": min(result["tone"]["tone_snr_db"]),
        "coherence": result["tone"]["coherence"],
        "global_dominance_db": result["tone"]["global_dominance_db"],
        "mirror_rejection_db": result["tone"]["mirror_rejection_db"],
        "temperature_mdeg_c": result["temperature_mdeg_c"],
        "retune_elapsed_ms": result["retune_elapsed_ms"],
        "capture_elapsed_ms": result["capture_elapsed_ms"],
    }
    for name, value in metrics.items():
        bounds = summary["metrics"].setdefault(
            name, {"minimum": value, "maximum": value}
        )
        bounds["minimum"] = min(bounds["minimum"], value)
        bounds["maximum"] = max(bounds["maximum"], value)


def _mute_radio(radio: UsbRadio, expected_firmware: str) -> dict[str, Any]:
    sdr = _open_attested_sdr(radio, expected_firmware)
    try:
        tx1, tx2 = mute_sdr_tx(sdr)
        tandem = sdr._ctx.find_device("tandem-agc")
        return {
            "serial": radio.serial,
            "uri": radio.uri,
            "tx1_gain_db": tx1,
            "tx2_gain_db": tx2,
            "tandem_state": int(tandem.attrs["state"].value),
            "tandem_fault_flags": int(tandem.attrs["fault_flags"].value),
        }
    finally:
        sdr.rx_destroy_buffer()
        _close_context(sdr)
        del sdr
        gc.collect()


def run_campaign(
    radios: tuple[UsbRadio, ...],
    *,
    report_path: Path,
    duration_seconds: float,
    emitted_frequencies_hz: tuple[int, ...] = DEFAULT_EMITTED_FREQUENCIES_HZ,
    rx_lo_offsets_hz: tuple[int, ...] = DEFAULT_RX_LO_OFFSETS_HZ,
    dds_offset_hz: int = DEFAULT_DDS_OFFSET_HZ,
    tx_gain_db: float = DEFAULT_TX_GAIN_DB,
    physical_attenuation_db: float = 0.0,
    dds_scale: float = DEFAULT_DDS_SCALE,
    expected_firmware: str = EXPECTED_FIRMWARE,
    random_seed: int = 20260822,
) -> dict[str, Any]:
    """Run at least one complete epoch, then stop at a duration boundary."""

    if not radios:
        raise ValueError("at least one USB radio is required")
    if duration_seconds < 0:
        raise ValueError("duration cannot be negative")
    effective_attenuation_db = validate_loopback_safety(
        physical_attenuation_db=physical_attenuation_db,
        strongest_tx_gain_db=tx_gain_db,
    )
    build_translation_cells(
        emitted_frequencies_hz,
        rx_lo_offsets_hz,
        dds_offset_hz=dds_offset_hz,
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "closed_loop_signed_rf_frequency_translation",
        "outcome": "started",
        "started_at_unix_ns": time.time_ns(),
        "duration_requested_seconds": duration_seconds,
        "expected_firmware": expected_firmware,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "bandwidth_hz": BANDWIDTH_HZ,
        "samples_per_channel": SAMPLES_PER_CHANNEL,
        "emitted_frequencies_hz": list(emitted_frequencies_hz),
        "rx_lo_offsets_hz": list(rx_lo_offsets_hz),
        "dds_offset_hz": dds_offset_hz,
        "tx_gain_db": tx_gain_db,
        "physical_attenuation_db": physical_attenuation_db,
        "effective_attenuation_db": effective_attenuation_db,
        "random_seed": random_seed,
        "radios": [asdict(radio) | {"uri": radio.uri} for radio in radios],
        "epochs_completed": 0,
        "cells_jsonl": str(report_path.with_suffix(".cells.jsonl")),
        "cell_summary": {"count": 0, "count_by_serial": {}, "metrics": {}},
    }
    cells_path = report_path.with_suffix(".cells.jsonl")
    cells_path.parent.mkdir(parents=True, exist_ok=True)
    with cells_path.open("w") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    _atomic_write_json(report_path, report)
    campaign_started = time.monotonic()
    active_error: BaseException | None = None
    try:
        report["initial_safety"] = [
            _mute_radio(radio, expected_firmware) for radio in radios
        ]
        _atomic_write_json(report_path, report)
        epoch = 0
        while epoch == 0 or time.monotonic() - campaign_started < duration_seconds:
            shuffled = duration_seconds > 0
            cells = build_translation_cells(
                emitted_frequencies_hz,
                rx_lo_offsets_hz,
                dds_offset_hz=dds_offset_hz,
                sample_rate_hz=SAMPLE_RATE_HZ,
                shuffle_seed=random_seed + epoch if shuffled else None,
            )
            by_emitter: dict[int, list[TranslationCell]] = {}
            for cell in cells:
                by_emitter.setdefault(cell.emitted_frequency_hz, []).append(cell)
            for radio in radios:
                sdr = _open_attested_sdr(radio, expected_firmware)
                try:
                    _configure_static(sdr)
                    for emitted_frequency_hz, carrier_cells in by_emitter.items():
                        first = carrier_cells[0]
                        arm = _arm_fixed_emitter(
                            sdr,
                            first,
                            tx_gain_db=tx_gain_db,
                            dds_scale=dds_scale,
                        )
                        try:
                            for cell in carrier_cells:
                                result = _capture_cell(
                                    sdr,
                                    cell,
                                    emitted_frequency_hz=emitted_frequency_hz,
                                )
                                result.update(
                                    {
                                        "epoch": epoch,
                                        "serial": radio.serial,
                                        "uri": radio.uri,
                                        "arm_attempts": arm["arm_attempts"],
                                    }
                                )
                                _record_cell(report, cells_path, result)
                                _atomic_write_json(report_path, report)
                        finally:
                            mute_sdr_tx(sdr)
                finally:
                    mute_sdr_tx(sdr)
                    sdr.rx_destroy_buffer()
                    _close_context(sdr)
                    del sdr
                    gc.collect()
            epoch += 1
            report["epochs_completed"] = epoch
            report["elapsed_seconds"] = time.monotonic() - campaign_started
            _atomic_write_json(report_path, report)
        report["outcome"] = "pass"
    except BaseException as error:
        active_error = error
        report["outcome"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        final_safety = []
        safety_errors = []
        for radio in radios:
            try:
                final_safety.append(_mute_radio(radio, expected_firmware))
            except Exception as error:  # noqa: BLE001 - attempt every final mute
                safety_errors.append(f"{radio.serial}: {type(error).__name__}: {error}")
        report["final_safety"] = final_safety
        report["final_safety_errors"] = safety_errors
        report["finished_at_unix_ns"] = time.time_ns()
        report["elapsed_seconds"] = time.monotonic() - campaign_started
        _atomic_write_json(report_path, report)
        if safety_errors and active_error is None:
            raise RuntimeError(f"final TX safety verification failed: {safety_errors}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--emitted-frequencies", default="433.92M,1.2G,2.45G,5.8G")
    parser.add_argument("--rx-lo-offsets", default="-900K,-350K,225K,800K")
    parser.add_argument("--dds-offset-hz", type=int, default=DEFAULT_DDS_OFFSET_HZ)
    parser.add_argument("--tx-gain-db", type=float, default=DEFAULT_TX_GAIN_DB)
    parser.add_argument("--physical-attenuation-db", type=float, default=0.0)
    parser.add_argument("--expected-firmware", default=EXPECTED_FIRMWARE)
    parser.add_argument("--random-seed", type=int, default=20260822)
    return parser


def main() -> int:
    def terminate(_signum, _frame) -> None:
        raise KeyboardInterrupt("termination requested; running verified TX cleanup")

    signal.signal(signal.SIGTERM, terminate)
    args = _parser().parse_args()
    radios = discover_usb_radios()
    if len(radios) != args.expected_count:
        raise SystemExit(
            f"expected exactly {args.expected_count} USB Plutos, found {len(radios)}: "
            f"{[radio.serial for radio in radios]}"
        )
    report = run_campaign(
        radios,
        report_path=args.report,
        duration_seconds=args.duration_seconds,
        emitted_frequencies_hz=parse_hz_list(args.emitted_frequencies),
        rx_lo_offsets_hz=parse_hz_list(args.rx_lo_offsets, signed=True),
        dds_offset_hz=args.dds_offset_hz,
        tx_gain_db=args.tx_gain_db,
        physical_attenuation_db=args.physical_attenuation_db,
        expected_firmware=args.expected_firmware,
        random_seed=args.random_seed,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("outcome", "epochs_completed", "elapsed_seconds")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
