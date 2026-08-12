"""Request-driven libiio RX with frame-associated SPF V3 metadata."""

from __future__ import annotations

import gc
import time

from spf.direct_radio.sample_clock import (
    DEFAULT_SAMPLE_CLOCK_RATE_TOLERANCE_PPM,
    HostTimeAnchorMeasurement,
    capture_host_realtime_mapping,
    fit_sample_clock,
)
from spf.direct_radio.usb_protocol import (
    MetadataFlags,
    RadioMetadataV3,
    TimeAnchorFlags,
    TimeAnchorV1,
)


ADC_SAMPLE_COUNTER_LOW_REG = 0x800000B8
DEFAULT_METADATA_CAPACITY = 64 * 1024
INITIAL_TIME_ANCHOR_COUNT = 8
MAX_TIME_ANCHORS = 32
TIME_ANCHOR_WINDOW_NS = 10_000_000_000


class IioMetadataRx:
    """Adapt pyadi's existing RX conversion to a metadata-enabled IIO buffer.

    The patched Python libiio binding supplies ``MetadataBuffer``.  PyADI still
    performs its normal channel extraction and complex conversion; this class
    only selects the opt-in buffer, parses its opaque metadata, and maps the
    FPGA sample counter onto host time using ordinary IIO register reads.
    """

    def __init__(
        self,
        sdr,
        *,
        sample_rate_hz: int,
        samples_per_channel: int,
        metadata_capacity: int = DEFAULT_METADATA_CAPACITY,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if samples_per_channel <= 0:
            raise ValueError("samples_per_channel must be positive")
        if metadata_capacity <= 0:
            raise ValueError("metadata_capacity must be positive")
        self._sdr = sdr
        self._sample_rate_hz = int(sample_rate_hz)
        self._samples_per_channel = int(samples_per_channel)
        self._metadata_capacity = int(metadata_capacity)
        self._buffer = None
        self._time_anchors: list[HostTimeAnchorMeasurement] = []
        self._next_anchor_request_id = 1

    @property
    def is_open(self) -> bool:
        return self._buffer is not None

    def open(self) -> None:
        if self._buffer is not None:
            raise RuntimeError("IIO metadata RX is already open")
        import iio

        metadata_buffer_type = getattr(iio, "MetadataBuffer", None)
        if metadata_buffer_type is None:
            raise RuntimeError(
                "the installed libiio Python binding lacks MetadataBuffer; "
                "install the patched SPF libiio 0.25 or 0.26 binding"
            )

        # Let pyadi configure precisely the channels it would use for rx().
        # Its v0.x compatibility layer also creates an ordinary buffer, which
        # must be closed before the metadata owner is opened.
        self._sdr.rx_destroy_buffer()
        self._sdr._rx_init_channels()
        ordinary_buffer = self._sdr._rxbuf
        self._sdr._rxbuf = None
        del ordinary_buffer
        gc.collect()

        try:
            self._buffer = metadata_buffer_type(
                self._sdr._rxadc,
                self._samples_per_channel,
                self._metadata_capacity,
            )
            self._sdr._rxbuf = self._buffer
            self._refresh_time_anchors(initial=True)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        buffer = self._buffer
        self._buffer = None
        self._time_anchors = []
        if getattr(self._sdr, "_rxbuf", None) is buffer:
            self._sdr._rxbuf = None
        del buffer
        gc.collect()

    def capture(self):
        """Return pyadi IQ, parsed V3 metadata, and capture-time fields."""

        if self._buffer is None:
            raise RuntimeError("IIO metadata RX is not open")
        signal_matrix = self._sdr.rx()
        raw_metadata = self._buffer.metadata
        if raw_metadata is None:
            raise RuntimeError("metadata buffer refill returned no metadata")
        metadata = RadioMetadataV3.unpack(raw_metadata)
        if len(raw_metadata) != metadata.header_bytes:
            raise RuntimeError("metadata refill returned trailing bytes")
        self._validate_metadata(metadata)
        self._refresh_time_anchors(initial=False)
        return signal_matrix, metadata, self._capture_time(metadata)

    def _validate_metadata(self, metadata: RadioMetadataV3) -> None:
        if metadata.samples_per_channel != self._samples_per_channel:
            raise RuntimeError(
                "metadata sample count does not match the requested IIO buffer"
            )
        if metadata.iq_payload_bytes != self._samples_per_channel * 8:
            raise RuntimeError("metadata IQ byte count does not match dual-CS16 IIO")
        if metadata.enabled_scan_mask != 0x0F:
            raise RuntimeError("metadata scan mask does not match dual-channel IIO")
        if not metadata.flags & MetadataFlags.HARDWARE_SAMPLE_COUNTER_VALID:
            raise RuntimeError("IIO metadata lacks a valid FPGA sample counter")

    def _query_time_anchor(self) -> HostTimeAnchorMeasurement:
        request_id = self._next_anchor_request_id
        self._next_anchor_request_id = (request_id + 1) & 0xFFFFFFFFFFFFFFFF
        if self._next_anchor_request_id == 0:
            self._next_anchor_request_id = 1
        host_before_ns = time.monotonic_ns()
        sample_counter = int(
            self._sdr._rxadc.reg_read(ADC_SAMPLE_COUNTER_LOW_REG)
        ) & 0xFFFFFFFF
        host_after_ns = time.monotonic_ns()
        anchor = TimeAnchorV1(
            flags=(
                TimeAnchorFlags.COUNTER_INTERVAL_VALID
                | TimeAnchorFlags.MONOTONIC_INTERVAL_VALID
                | TimeAnchorFlags.COUNTER_LOW32
            ),
            request_id=request_id,
            radio_monotonic_before_ns=0,
            sample_counter_before=sample_counter,
            sample_counter_after=sample_counter,
            radio_monotonic_after_ns=0,
        )
        return HostTimeAnchorMeasurement(
            anchor=anchor,
            host_monotonic_before_ns=host_before_ns,
            host_monotonic_after_ns=host_after_ns,
            transport="iio",
        )

    def _refresh_time_anchors(self, *, initial: bool) -> None:
        count = INITIAL_TIME_ANCHOR_COUNT if initial else 1
        for index in range(count):
            self._time_anchors.append(self._query_time_anchor())
            if index + 1 < count:
                time.sleep(0.005)
        newest = self._time_anchors[-1].host_monotonic_after_ns
        cutoff = newest - TIME_ANCHOR_WINDOW_NS
        self._time_anchors = [
            item
            for item in self._time_anchors[-MAX_TIME_ANCHORS:]
            if item.host_monotonic_after_ns >= cutoff
        ]

    def _capture_time(self, metadata: RadioMetadataV3) -> dict[str, int | float | bool]:
        extended = [
            item.extend_near(metadata.first_sample_sequence)
            for item in self._time_anchors
        ]
        fit = fit_sample_clock(
            extended,
            nominal_sample_rate_hz=self._sample_rate_hz,
            maximum_rate_error_ppm=DEFAULT_SAMPLE_CLOCK_RATE_TOLERANCE_PPM,
        )
        realtime = capture_host_realtime_mapping()
        sample_start = metadata.first_sample_sequence
        sample_end = sample_start + metadata.samples_per_channel
        monotonic_start = fit.host_monotonic_ns(sample_start)
        monotonic_end = fit.host_monotonic_ns(sample_end)
        return {
            "sample_counter_end_exclusive": sample_end,
            "sample_time_valid": True,
            "sample_time_monotonic_start_ns": monotonic_start,
            "sample_time_monotonic_end_ns": monotonic_end,
            "sample_time_realtime_start_ns": realtime.realtime_ns(monotonic_start),
            "sample_time_realtime_end_ns": realtime.realtime_ns(monotonic_end),
            "sample_time_uncertainty_ns": max(
                fit.uncertainty_ns_at(sample_start),
                fit.uncertainty_ns_at(sample_end),
            )
            + realtime.uncertainty_ns,
            "sample_time_fitted_rate_hz": fit.fitted_sample_rate_hz,
            "sample_time_anchor_count": fit.anchor_count,
            "sample_time_max_round_trip_ns": fit.maximum_round_trip_ns,
            "sample_time_rate_tolerance_ppm": fit.maximum_rate_error_ppm,
        }
