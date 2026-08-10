"""Run a bounded, parallel direct-IP sample-rate ladder.

The ladder intentionally separates frame integrity from real-time headroom.
The Pluto IP gadget buffers a finite request before draining it over UDP, so a
high-rate rung can return every frame correctly even when the transport would
not sustain that RF rate continuously.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import socket
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iio
import numpy as np

from spf.sdrpluto.direct_ip_protocol import IpControlFlags
from spf.sdrpluto.direct_ip_protocol import (
    DEFAULT_UDP_DATAGRAM_BYTES,
    IP_FRAGMENT_HEADER_BYTES,
)
from spf.sdrpluto.direct_ip_receiver import PlutoDirectIpReceiver
from spf.sdrpluto.direct_usb_protocol import (
    GainObservationFlags,
    GAIN_OBSERVATION_BYTES,
    HEADER_PREFIX_BYTES_V3,
    MetadataFlags,
    RadioMetadataV3,
)


MIB = 1024 * 1024
REQUIRED_TRANSPORT_FLAGS = (
    IpControlFlags.BUFFERED_FINITE_RX | IpControlFlags.USB_CLASS_PACING
)


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelIpLadderSettings:
    hosts: tuple[str, ...]
    sample_rates_hz: tuple[int, ...]
    samples_per_channel: int = 524_288
    frames_per_request: int = 16
    cycles_per_rate: int = 3
    gain_observation_interval_samples: int = 2_048
    gain_observation_capacity: int = 256
    minimum_effective_receive_buffer_bytes: int = 256 * MIB
    network_interface: str = "eth0"
    stop_after_integrity_failure: bool = True

    def __post_init__(self) -> None:
        if len(self.hosts) < 2 or len(set(self.hosts)) != len(self.hosts):
            raise ValueError("parallel direct-IP ladder requires unique hosts")
        if not self.sample_rates_hz or any(rate <= 0 for rate in self.sample_rates_hz):
            raise ValueError("sample-rate ladder must contain positive rates")
        if tuple(sorted(set(self.sample_rates_hz))) != self.sample_rates_hz:
            raise ValueError("sample-rate ladder must be strictly increasing")
        if self.samples_per_channel <= 0:
            raise ValueError("samples per channel must be positive")
        if self.frames_per_request <= 0:
            raise ValueError("frames per request must be positive")
        if self.cycles_per_rate <= 0:
            raise ValueError("cycles per rate must be positive")
        if not 1 <= self.gain_observation_capacity <= 256:
            raise ValueError("gain-observation capacity must be in [1, 256]")


def parse_sample_rate_ladder(value: str) -> tuple[int, ...]:
    """Parse comma-separated Hz values, accepting compact ``3M`` notation."""

    rates: list[int] = []
    for token in value.split(","):
        item = token.strip().lower().replace("_", "")
        if not item:
            raise ValueError("sample-rate ladder contains an empty value")
        multiplier = 1
        if item.endswith("m"):
            multiplier = 1_000_000
            item = item[:-1]
        elif item.endswith("k"):
            multiplier = 1_000
            item = item[:-1]
        try:
            rate = int(float(item) * multiplier)
        except ValueError as error:
            raise ValueError(f"invalid sample rate: {token!r}") from error
        if rate <= 0:
            raise ValueError("sample rates must be positive")
        rates.append(rate)
    parsed = tuple(rates)
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("sample-rate ladder must be strictly increasing")
    return parsed


def direct_ip_sample_rate(host: str) -> int:
    context = iio.Context(f"ip:{host}")
    try:
        phy = context.find_device("ad9361-phy")
        if phy is None:
            raise RuntimeError(f"{host}: no ad9361-phy")
        channel = phy.find_channel("voltage0", True)
        if channel is None or "sampling_frequency" not in channel.attrs:
            raise RuntimeError(f"{host}: no RX sampling-frequency control")
        return int(channel.attrs["sampling_frequency"].value)
    finally:
        del context


def set_direct_ip_sample_rate(host: str, sample_rate_hz: int) -> int:
    context = iio.Context(f"ip:{host}")
    try:
        phy = context.find_device("ad9361-phy")
        if phy is None:
            raise RuntimeError(f"{host}: no ad9361-phy")
        channel = phy.find_channel("voltage0", True)
        if channel is None or "sampling_frequency" not in channel.attrs:
            raise RuntimeError(f"{host}: no RX sampling-frequency control")
        channel.attrs["sampling_frequency"].value = str(int(sample_rate_hz))
        return int(channel.attrs["sampling_frequency"].value)
    finally:
        del context


def direct_ip_identity(host: str) -> dict[str, str]:
    context = iio.Context(f"ip:{host}")
    try:
        return {
            "serial": context.attrs.get("hw_serial", ""),
            "firmware": context.attrs.get("fw_version", ""),
            "model": context.attrs.get("hw_model", ""),
        }
    finally:
        del context


def _rss_kib() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def read_udp_counters() -> dict[str, int]:
    lines = [line.split() for line in Path("/proc/net/snmp").read_text().splitlines()]
    for index in range(len(lines) - 1):
        if lines[index] and lines[index][0] == "Udp:":
            names = lines[index][1:]
            values = lines[index + 1][1:]
            return {name: int(value) for name, value in zip(names, values)}
    raise RuntimeError("/proc/net/snmp has no UDP counters")


def read_network_counters(interface: str) -> dict[str, int]:
    statistics = Path("/sys/class/net") / interface / "statistics"
    if not statistics.is_dir():
        raise RuntimeError(f"network interface does not exist: {interface}")
    names = (
        "rx_bytes",
        "rx_packets",
        "rx_errors",
        "rx_dropped",
        "rx_missed_errors",
        "tx_bytes",
        "tx_packets",
        "tx_errors",
        "tx_dropped",
    )
    return {name: int((statistics / name).read_text()) for name in names}


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - value for key, value in before.items()}


def _frame_summary(frame: Any, samples: int) -> dict[str, Any]:
    metadata = frame.metadata
    if not isinstance(metadata, RadioMetadataV3):
        raise RuntimeError("direct-IP ladder received non-v3 metadata")
    if metadata.samples_per_channel != samples or len(frame.iq_payload) != samples * 8:
        raise RuntimeError("direct-IP ladder received the wrong IQ payload size")
    required = (
        MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID
        | MetadataFlags.GAIN_OBSERVATIONS_VALID
    )
    if metadata.flags & required != required:
        raise RuntimeError("direct-IP frame lacks valid sample/gain metadata")
    forbidden = (
        MetadataFlags.DEVICE_IIO_OVERFLOW | MetadataFlags.GAIN_OBSERVATION_OVERFLOW
    )
    if metadata.flags & forbidden:
        raise RuntimeError(
            f"direct-IP frame reports overflow flags: {metadata.flags!r}"
        )
    if not metadata.gain_metadata_valid or not metadata.rssi_metadata_valid:
        raise RuntimeError("direct-IP frame has invalid gain or RSSI endpoints")
    if not metadata.gain_observations:
        raise RuntimeError("direct-IP frame has no gain observations")
    observation_required = (
        GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID
    )
    if any(
        observation.flags & observation_required != observation_required
        for observation in metadata.gain_observations
    ):
        raise RuntimeError("direct-IP frame has an invalid gain observation")

    raw = np.frombuffer(frame.iq_payload, dtype="<i2").reshape(samples, 4)
    channel_rms = []
    for i_offset in (0, 2):
        i_values = raw[:, i_offset].astype(np.float64)
        q_values = raw[:, i_offset + 1].astype(np.float64)
        rms = float(np.sqrt(np.mean(i_values * i_values + q_values * q_values)))
        if not np.isfinite(rms) or rms <= 0:
            raise RuntimeError("direct-IP frame contains an empty/non-finite channel")
        channel_rms.append(rms)
    return {
        "stream_id": metadata.stream_id,
        "buffer_sequence": metadata.buffer_sequence,
        "first_sample_sequence": metadata.first_sample_sequence,
        "gain_observation_count": len(metadata.gain_observations),
        "channel_rms": channel_rms,
    }


def _capture_one(
    receiver: Any, settings: ParallelIpLadderSettings, sample_rate_hz: int
) -> dict[str, Any]:
    capture = receiver.capture(
        samples_per_channel=settings.samples_per_channel,
        frame_count=settings.frames_per_request,
    )
    if len(capture.frames) != settings.frames_per_request:
        raise RuntimeError(
            f"received {len(capture.frames)}/{settings.frames_per_request} frames"
        )
    sequences = [frame.metadata.buffer_sequence for frame in capture.frames]
    if sequences != list(range(settings.frames_per_request)):
        raise RuntimeError(f"non-contiguous buffer sequences: {sequences}")
    sample_sequences = [
        frame.metadata.first_sample_sequence for frame in capture.frames
    ]
    if any(
        right - left != settings.samples_per_channel
        for left, right in zip(sample_sequences, sample_sequences[1:])
    ):
        raise RuntimeError("non-contiguous hardware sample sequences")
    stream_ids = {frame.metadata.stream_id for frame in capture.frames}
    if len(stream_ids) != 1:
        raise RuntimeError("capture contains multiple stream IDs")
    counters = {
        "duplicate_fragment_count": capture.duplicate_fragment_count,
        "expired_frame_count": capture.expired_frame_count,
        "rejected_frame_count": capture.rejected_frame_count,
        "receive_queue_overflow_count": capture.receive_queue_overflow_count,
    }
    if any(counters.values()):
        raise RuntimeError(f"direct-IP loss counters are non-zero: {counters}")
    payload_bytes = settings.samples_per_channel * 8 * settings.frames_per_request
    nominal_acquisition_seconds = (
        settings.samples_per_channel * settings.frames_per_request / sample_rate_hz
    )
    estimated_drain_seconds = capture.elapsed_seconds - nominal_acquisition_seconds
    estimated_drain_mibps = (
        payload_bytes / estimated_drain_seconds / MIB
        if estimated_drain_seconds > 0
        else None
    )
    return {
        "elapsed_seconds": capture.elapsed_seconds,
        "nominal_acquisition_seconds": nominal_acquisition_seconds,
        "estimated_drain_seconds": estimated_drain_seconds,
        "payload_bytes": payload_bytes,
        "payload_mibps": payload_bytes / capture.elapsed_seconds / MIB,
        "estimated_drain_mibps": estimated_drain_mibps,
        "effective_receive_buffer_bytes": (
            receiver.effective_data_receive_buffer_bytes
        ),
        **counters,
        "first_frame": _frame_summary(capture.frames[0], settings.samples_per_channel),
        "last_frame": _frame_summary(capture.frames[-1], settings.samples_per_channel),
    }


def _default_receiver_factory(host: str, settings: ParallelIpLadderSettings) -> Any:
    return PlutoDirectIpReceiver(
        remote_host=host,
        protocol_version=3,
        gain_observation_interval_samples=min(
            settings.gain_observation_interval_samples,
            settings.samples_per_channel,
        ),
        gain_observation_capacity=settings.gain_observation_capacity,
        minimum_effective_receive_buffer_bytes=(
            settings.minimum_effective_receive_buffer_bytes
        ),
    )


def _summarize_rung_performance(rung: dict[str, Any], hosts: tuple[str, ...]) -> None:
    """Summarize complete cycles, even when a later lifecycle cycle failed."""

    complete_cycles = [
        cycle
        for cycle in rung["cycles"]
        if not cycle["errors"] and all(host in cycle["radios"] for host in hosts)
    ]
    if not complete_cycles:
        return
    rung["minimum_payload_mibps_by_host"] = {
        host: min(cycle["radios"][host]["payload_mibps"] for cycle in complete_cycles)
        for host in hosts
    }
    estimated_drain_rates = {
        host: min(
            cycle["radios"][host]["estimated_drain_mibps"]
            for cycle in complete_cycles
            if cycle["radios"][host]["estimated_drain_mibps"] is not None
        )
        for host in hosts
    }
    rung["minimum_estimated_drain_mibps_by_host"] = estimated_drain_rates
    rung["minimum_realtime_headroom_by_host"] = {
        host: rate / rung["required_payload_mibps_per_radio"]
        for host, rate in estimated_drain_rates.items()
    }


def _is_control_rearm_failure(rung: dict[str, Any]) -> bool:
    """Recognize the firmware ignoring START after an earlier clean cycle."""

    if not any(not cycle["errors"] for cycle in rung["cycles"]):
        return False
    failed_cycles = [cycle for cycle in rung["cycles"] if cycle["errors"]]
    if not failed_cycles:
        return False
    errors = [error for cycle in failed_cycles for error in cycle["errors"].values()]
    return bool(errors) and all(
        "START_RX was not acknowledged" in error for error in errors
    )


def run_parallel_ip_ladder(
    settings: ParallelIpLadderSettings,
    *,
    get_sample_rate: Callable[[str], int] = direct_ip_sample_rate,
    set_sample_rate: Callable[[str, int], int] = set_direct_ip_sample_rate,
    get_identity: Callable[[str], dict[str, str]] = direct_ip_identity,
    receiver_factory: Callable[[str, ParallelIpLadderSettings], Any] = (
        _default_receiver_factory
    ),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the ladder and always restore each radio's original sample rate."""

    identities = {host: get_identity(host) for host in settings.hosts}
    serials = [identity.get("serial") for identity in identities.values()]
    if any(not serial for serial in serials) or len(set(serials)) != len(serials):
        raise RuntimeError(
            f"direct-IP hosts do not identify unique radios: {identities}"
        )
    original_rates = {host: get_sample_rate(host) for host in settings.hosts}
    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_name": socket.gethostname(),
        "transport": "direct_ip",
        "test": "parallel_sample_rate_ladder",
        "hosts": list(settings.hosts),
        "radio_identities": identities,
        "sample_rates_hz": list(settings.sample_rates_hz),
        "samples_per_channel": settings.samples_per_channel,
        "frames_per_request": settings.frames_per_request,
        "cycles_per_rate": settings.cycles_per_rate,
        "network_interface": settings.network_interface,
        "original_sample_rates_hz": original_rates,
        "rungs": [],
        "first_integrity_failure_hz": None,
        "first_control_rearm_failure_hz": None,
        "first_realtime_shortfall_hz": None,
    }
    v3_header_bytes = (
        HEADER_PREFIX_BYTES_V3
        + settings.gain_observation_capacity * GAIN_OBSERVATION_BYTES
        + 4
    )
    frame_bytes = settings.samples_per_channel * 8 + v3_header_bytes
    fragment_payload_bytes = DEFAULT_UDP_DATAGRAM_BYTES - IP_FRAGMENT_HEADER_BYTES
    report["v3_header_bytes"] = v3_header_bytes
    report["inner_frame_bytes"] = frame_bytes
    report["udp_datagrams_per_frame"] = (
        frame_bytes + fragment_payload_bytes - 1
    ) // fragment_payload_bytes

    def publish() -> None:
        if progress_callback is not None:
            progress_callback(report)

    publish()
    try:
        for sample_rate_hz in settings.sample_rates_hz:
            rung: dict[str, Any] = {
                "sample_rate_hz": sample_rate_hz,
                "required_payload_mibps_per_radio": sample_rate_hz * 8 / MIB,
                "cycles": [],
                "status": "pass",
            }
            report["rungs"].append(rung)
            try:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(settings.hosts)
                ) as executor:
                    configured = dict(
                        zip(
                            settings.hosts,
                            executor.map(
                                lambda host: set_sample_rate(host, sample_rate_hz),
                                settings.hosts,
                            ),
                        )
                    )
                rung["configured_sample_rates_hz"] = configured
                if any(rate != sample_rate_hz for rate in configured.values()):
                    raise RuntimeError(f"sample-rate readback mismatch: {configured}")

                with contextlib.ExitStack() as stack:
                    receivers = {
                        host: stack.enter_context(receiver_factory(host, settings))
                        for host in settings.hosts
                    }
                    for receiver in receivers.values():
                        if (
                            receiver.capabilities.flags & REQUIRED_TRANSPORT_FLAGS
                            != REQUIRED_TRANSPORT_FLAGS
                        ):
                            raise RuntimeError(
                                "radio lacks buffered/paced IP transport"
                            )
                    for cycle_index in range(settings.cycles_per_rate):
                        udp_before = read_udp_counters()
                        network_before = read_network_counters(
                            settings.network_interface
                        )
                        rss_before = _rss_kib()
                        cpu_before = time.process_time()
                        wall_before = time.monotonic()
                        results: dict[str, dict[str, Any]] = {}
                        errors: dict[str, str] = {}
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=len(receivers)
                        ) as executor:
                            futures = {
                                host: executor.submit(
                                    _capture_one,
                                    receiver,
                                    settings,
                                    sample_rate_hz,
                                )
                                for host, receiver in receivers.items()
                            }
                            for host, future in futures.items():
                                try:
                                    results[host] = future.result()
                                except (
                                    Exception
                                ) as error:  # preserve both radios' results
                                    errors[host] = f"{type(error).__name__}: {error}"
                        wall_seconds = time.monotonic() - wall_before
                        cpu_seconds = time.process_time() - cpu_before
                        udp_delta = _counter_delta(read_udp_counters(), udp_before)
                        network_delta = _counter_delta(
                            read_network_counters(settings.network_interface),
                            network_before,
                        )
                        payload_bytes = sum(
                            result["payload_bytes"] for result in results.values()
                        )
                        cycle = {
                            "cycle": cycle_index,
                            "wall_seconds": wall_seconds,
                            "process_cpu_seconds": cpu_seconds,
                            "process_cpu_percent": (
                                100 * cpu_seconds / wall_seconds if wall_seconds else 0
                            ),
                            "rss_before_kib": rss_before,
                            "rss_after_kib": _rss_kib(),
                            "aggregate_payload_bytes": payload_bytes,
                            "aggregate_payload_mibps": (
                                payload_bytes / wall_seconds / MIB
                                if wall_seconds
                                else 0
                            ),
                            "udp_counter_delta": udp_delta,
                            "network_counter_delta": network_delta,
                            "radios": results,
                            "errors": errors,
                        }
                        rung["cycles"].append(cycle)
                        publish()
                        if (
                            errors
                            or udp_delta.get("RcvbufErrors", 0) > 0
                            or udp_delta.get("MemErrors", 0) > 0
                        ):
                            raise RuntimeError(
                                f"parallel capture failed: errors={errors}, "
                                f"RcvbufErrors={udp_delta.get('RcvbufErrors', 0)}, "
                                f"MemErrors={udp_delta.get('MemErrors', 0)}"
                            )

                _summarize_rung_performance(rung, settings.hosts)
                if (
                    report["first_realtime_shortfall_hz"] is None
                    and min(rung["minimum_realtime_headroom_by_host"].values()) < 1.0
                ):
                    report["first_realtime_shortfall_hz"] = sample_rate_hz
            except Exception as error:
                _summarize_rung_performance(rung, settings.hosts)
                if _is_control_rearm_failure(rung):
                    rung["status"] = "control_rearm_failure"
                    if report["first_control_rearm_failure_hz"] is None:
                        report["first_control_rearm_failure_hz"] = sample_rate_hz
                else:
                    rung["status"] = "integrity_failure"
                    if report["first_integrity_failure_hz"] is None:
                        report["first_integrity_failure_hz"] = sample_rate_hz
                rung["error"] = f"{type(error).__name__}: {error}"
                if (
                    report["first_realtime_shortfall_hz"] is None
                    and rung.get("minimum_realtime_headroom_by_host")
                    and min(rung["minimum_realtime_headroom_by_host"].values()) < 1.0
                ):
                    report["first_realtime_shortfall_hz"] = sample_rate_hz
                publish()
                if settings.stop_after_integrity_failure:
                    break
    finally:
        restore_errors: dict[str, str] = {}
        for host, sample_rate_hz in original_rates.items():
            try:
                restored = set_sample_rate(host, sample_rate_hz)
                if restored != sample_rate_hz:
                    raise RuntimeError(
                        f"read back {restored}, expected {sample_rate_hz}"
                    )
            except Exception as error:
                restore_errors[host] = f"{type(error).__name__}: {error}"
        report["restored_sample_rates_hz"] = {
            host: get_sample_rate(host)
            for host in settings.hosts
            if host not in restore_errors
        }
        report["restore_errors"] = restore_errors
    report["completed_rungs"] = len(report["rungs"])
    report["integrity_pass"] = (
        report["first_integrity_failure_hz"] is None and not report["restore_errors"]
    )
    report["control_lifecycle_pass"] = report["first_control_rearm_failure_hz"] is None
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    publish()
    return report
