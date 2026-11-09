"""Pluto TX2 loopback and finite direct-USB capture adapter."""

from __future__ import annotations

import time
from math import gcd

import numpy as np

from spf.calibrations.dual_rx_gain_frequency.config import CalibrationConfig
from spf.sdrpluto.direct_usb_protocol import (
    FIRST_CHANGE_UNAVAILABLE,
    MetadataFlags,
    RadioMetadataV2,
)
from spf.sdrpluto.direct_usb_receiver import (
    PlutoDirectUsbReceiver,
    iq_payload_to_complex64,
)
from spf.sdrpluto.sdr_controller import (
    PLUTO_USB_PRODUCT_ID,
    PLUTO_USB_VENDOR_ID,
    PlutoRxBuffer,
    SdrDeviceIdentity,
    _find_local_pluto_usb_device,
    _gain_series_arrays,
    _iq_power_dbfs,
)
from spf.direct_radio.iio_metadata import IioMetadataRx
from spf.direct_radio.usb_protocol import RadioMetadataV3


UNSAFE_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
)


def make_cyclic_tone(config: CalibrationConfig) -> np.ndarray:
    """Return an integer-period complex tone suitable for a cyclic TX buffer."""

    sample_rate = int(config.sample_rate_hz)
    tone_hz = int(config.tone_offset_hz)
    if float(tone_hz) != config.tone_offset_hz:
        raise ValueError("hardware tone offset must be an integer number of Hz")
    period = sample_rate // gcd(sample_rate, abs(tone_hz))
    sample_count = period
    while sample_count < 16_384:
        sample_count *= 2
    sample_index = np.arange(sample_count, dtype=np.float64)
    return (
        config.tx_digital_amplitude
        * np.exp(2j * np.pi * tone_hz * sample_index / sample_rate)
    ).astype(np.complex64)


class DirectUsbLoopbackRadio:
    """One Pluto configured through IIO and captured through direct USB."""

    def __init__(
        self,
        serial: str,
        config: CalibrationConfig,
        *,
        adi_module=None,
        direct_receiver_class=PlutoDirectUsbReceiver,
        scan_contexts=None,
        direct_protocol_version: int = 2,
        direct_receiver_options: dict | None = None,
    ):
        config.validate()
        if adi_module is None:
            import adi as adi_module
        if scan_contexts is None:
            import iio

            scan_contexts = iio.scan_contexts
        self.serial = serial
        self.config = config
        self.uri = self._resolve_uri(scan_contexts())
        self.sdr = adi_module.ad9361(uri=self.uri)
        actual_serial = self.sdr._ctx.attrs.get("hw_serial")
        if actual_serial != serial:
            raise RuntimeError(
                f"IIO serial mismatch: requested {serial}, opened {actual_serial}"
            )
        self.direct_protocol_version = int(direct_protocol_version)
        receiver_options = dict(direct_receiver_options or {})
        self.direct = direct_receiver_class(
            serial=serial,
            protocol_version=self.direct_protocol_version,
            **receiver_options,
        )
        self.direct.open()
        if self.direct.identity.serial != serial:
            self.direct.close()
            raise RuntimeError("direct USB identity does not match IIO identity")
        self._tone_active = False
        self._active_tx_gain = None
        try:
            self._configure_static()
        except Exception:
            self.close()
            raise

    def _resolve_uri(self, contexts: dict[str, str]) -> str:
        matches = [
            uri
            for uri, description in contexts.items()
            if uri.startswith("usb:") and f"serial={self.serial}" in description
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one USB-IIO URI for {self.serial}, found {matches}"
            )
        return matches[0]

    def _configure_static(self) -> None:
        sdr = self.sdr
        self.stop_tone()
        sdr.rx_destroy_buffer()
        sdr.rx_enabled_channels = [0, 1]
        sdr.sample_rate = int(self.config.sample_rate_hz)
        sdr.rx_rf_bandwidth = int(self.config.bandwidth_hz)
        sdr.tx_rf_bandwidth = int(self.config.bandwidth_hz)
        sdr.gain_control_mode_chan0 = "manual"
        sdr.gain_control_mode_chan1 = "manual"
        sdr.rx_buffer_size = int(self.config.buffer_size)
        sdr._rxadc.set_kernel_buffers_count(1)

        debug_attr = "adi,rx1-rx2-phase-inversion-enable"
        sdr._ctrl.debug_attrs[debug_attr].value = "1"
        register_0x22 = sdr._ctrl.reg_read(0x22)
        sdr._ctrl.reg_write(0x22, register_0x22 | (1 << 6))
        if not (sdr._ctrl.reg_read(0x22) & (1 << 6)):
            raise RuntimeError("failed to enable RX1/RX2 phase mitigation")

        for channel_name in ("voltage0", "voltage1"):
            channel = sdr._ctrl.find_channel(channel_name, is_output=False)
            if "quadrature_tracking_en" in channel.attrs:
                channel.attrs["quadrature_tracking_en"].value = "1"

        self.apply_rf_dc_offset_tracking()

    RF_DC_TRACKING_ATTR = "rf_dc_offset_tracking_en"

    def _read_rf_dc_offset_tracking(self) -> dict[str, str | None]:
        """Observed per-channel state of the continuous RF-DC tracking loop."""

        observed = {}
        for channel_name in ("voltage0", "voltage1"):
            channel = self.sdr._ctrl.find_channel(channel_name, is_output=False)
            attribute = channel.attrs.get(self.RF_DC_TRACKING_ATTR)
            observed[channel_name] = None if attribute is None else attribute.value
        return observed

    def apply_rf_dc_offset_tracking(self) -> dict[str, str | None]:
        """Force the RF-DC tracking loop to the configured state and verify it.

        The write is verified by reading the attribute back, and a mismatch is
        a hard error. The AD9361 driver silently ignores some attribute writes
        depending on chip state, and it re-asserts tracking across
        ``calib_mode`` writes and LO retunes. An unverified write would let a
        capture claim the tracking loop was disabled when it was never touched
        -- a false null indistinguishable from a real one on exactly the
        question this knob exists to answer.
        """

        requested = self.config.rf_dc_offset_tracking_en
        if requested is not None:
            desired = "1" if requested else "0"
            for channel_name in ("voltage0", "voltage1"):
                channel = self.sdr._ctrl.find_channel(channel_name, is_output=False)
                attribute = channel.attrs.get(self.RF_DC_TRACKING_ATTR)
                if attribute is None:
                    raise RuntimeError(
                        f"{self.serial}: {channel_name} does not expose "
                        f"{self.RF_DC_TRACKING_ATTR}; this firmware cannot run a "
                        "capture that pins the RF-DC tracking loop"
                    )
                attribute.value = desired
            observed = self._read_rf_dc_offset_tracking()
            mismatched = {
                name: value for name, value in observed.items() if value != desired
            }
            if mismatched:
                raise RuntimeError(
                    f"{self.serial}: RF-DC tracking readback does not match the "
                    f"requested {desired}: {mismatched}"
                )
            return observed
        return self._read_rf_dc_offset_tracking()

    def _prime_iio_rx_dma(self) -> None:
        """Initialize RX DMA after an LO retune and relinquish it to direct USB."""

        # The PlutoPlus direct-USB firmware shares the RX DMA hardware with the
        # standard IIO path. On the tested v0.38 image, the first direct RX
        # START can leave an already-armed cyclic TX buffer silent unless RX DMA
        # has first been initialized through IIO after the current LO is set.
        # This happens once per frequency block, before TX or direct streaming,
        # and never in the recorded frame loop.
        priming_frame = np.asarray(self.sdr.rx())
        self.sdr.rx_destroy_buffer()
        if priming_frame.shape != (2, self.config.buffer_size):
            raise RuntimeError(
                f"unexpected IIO RX priming shape: {priming_frame.shape}"
            )

    def available_gains(self) -> tuple[int, ...]:
        from spf.bench.dual_rx_phase import parse_gain_available

        values = []
        for channel_name in ("voltage0", "voltage1"):
            channel = self.sdr._ctrl.find_channel(channel_name, is_output=False)
            values.append(
                tuple(
                    parse_gain_available(channel.attrs["hardwaregain_available"].value)
                )
            )
        if values[0] != values[1]:
            raise RuntimeError(f"RX gain ranges differ: {values}")
        return values[0]

    def identity(self) -> SdrDeviceIdentity:
        direct = self.direct.identity
        capabilities = self.direct.capabilities
        return SdrDeviceIdentity(
            sdr_family="pluto",
            serial=self.serial,
            receiver_uri=self.uri,
            rx_transport="direct_usb",
            usb_vendor_id=PLUTO_USB_VENDOR_ID,
            usb_product_id=PLUTO_USB_PRODUCT_ID,
            usb_bus=direct.bus,
            usb_address=direct.address,
            usb_port_path=direct.port_path,
            direct_usb_interface=direct.interface,
            direct_usb_bulk_in_endpoint=direct.bulk_in_endpoint,
            direct_usb_bulk_out_endpoint=direct.bulk_out_endpoint,
            direct_usb_protocol_version=self.direct_protocol_version,
            direct_usb_protocol_min=capabilities.protocol_min,
            direct_usb_protocol_max=capabilities.protocol_max,
            direct_usb_supported_features=int(capabilities.supported_features),
            direct_usb_capability_flags=int(capabilities.capability_flags),
        )

    def configure_frequency(
        self, lo_frequency_hz: int, *, start_tone: bool = True
    ) -> None:
        self.stop_tone()
        self.sdr.rx_lo = int(lo_frequency_hz)
        self.sdr.tx_lo = int(lo_frequency_hz)
        if abs(int(self.sdr.rx_lo) - int(lo_frequency_hz)) >= 10:
            raise RuntimeError("RX LO readback mismatch")
        if abs(int(self.sdr.tx_lo) - int(lo_frequency_hz)) >= 10:
            raise RuntimeError("TX LO readback mismatch")
        # An LO retune can cross a gain-table band edge and re-assert the
        # driver's tracking defaults, so the pinned state is re-verified here.
        self.apply_rf_dc_offset_tracking()
        if start_tone:
            self.start_tone()
        time.sleep(self.config.frequency_settle_seconds)

    def set_gains(self, gain_rx1_db: int, gain_rx2_db: int) -> None:
        self.sdr.rx_hardwaregain_chan0 = int(gain_rx1_db)
        self.sdr.rx_hardwaregain_chan1 = int(gain_rx2_db)

    def set_tx_gain(self, tx_gain_db: float) -> None:
        if not self._tone_active:
            raise RuntimeError("cannot change TX gain while the tone is stopped")
        if self._active_tx_gain == float(tx_gain_db):
            return
        # Updating attenuation in place preserves the FPGA DDS and its phase
        # continuity throughout one frequency block.
        self.sdr.tx_hardwaregain_chan1 = float(tx_gain_db)
        actual = float(self.sdr.tx_hardwaregain_chan1)
        if not np.isclose(actual, tx_gain_db, atol=0.25):
            raise RuntimeError(
                f"TX2 attenuation readback {actual} != requested {tx_gain_db}"
            )
        self._active_tx_gain = actual

    def run_rf_dc_calibration(self) -> None:
        """Run the Linux driver's supported RF-DC initialization calibration."""

        if self._tone_active:
            raise RuntimeError("RF-DC calibration requires TX to be stopped")
        self.sdr._ctrl.attrs["calib_mode"].value = "rf_dc_offs"
        # The one-shot calibration is a separate mechanism from the tracking
        # loop, but the driver may re-enable tracking as a side effect of it.
        self.apply_rf_dc_offset_tracking()

    def run_tx_quadrature_calibration(self) -> None:
        """Force the supported AD9361 TX quadrature calibration.

        A successful IIO/LO configuration can occasionally leave the RF TX
        path silent while ENSM, LO-power, attenuation, and RF-port readbacks
        all look correct. The failure persists across TX buffer and DDS
        restarts in the same context. The driver's explicit ``tx_quad``
        calibration repairs that state deterministically, so every newly
        armed calibration tone runs it while TX is muted.
        """

        if self._tone_active:
            raise RuntimeError("TX quadrature calibration requires TX to be stopped")
        self.sdr._ctrl.attrs["calib_mode"].value = "tx_quad"

    def start_tone(
        self,
        tx_channel: int = 1,
        tx_gain_db: float | None = None,
        *,
        prime_after_arm: bool = False,
    ) -> None:
        if tx_channel not in (0, 1):
            raise ValueError("TX channel must be 0 or 1")
        if self._tone_active:
            raise RuntimeError("TX2 tone is already active")
        if self.config.tx_source != "fpga_dds":
            raise RuntimeError(f"unsupported TX source: {self.config.tx_source}")
        if tx_gain_db is None:
            tx_gain_db = self.config.tx_gain_db
        tone_hz = int(self.config.tone_offset_hz)
        if float(tone_hz) != self.config.tone_offset_hz:
            raise ValueError("FPGA DDS tone offset must be an integer number of Hz")
        dds_scale = float(self.config.tx_digital_amplitude) / float(2**15)
        try:
            self.run_tx_quadrature_calibration()
            self.sdr.disable_dds()
            self.sdr.tx_destroy_buffer()
            self.sdr.tx_cyclic_buffer = False
            self.sdr.tx_hardwaregain_chan0 = (
                float(tx_gain_db) if tx_channel == 0 else -80
            )
            self.sdr.tx_hardwaregain_chan1 = (
                float(tx_gain_db) if tx_channel == 1 else -80
            )
            self.sdr.tx_enabled_channels = []
            self.sdr.dds_single_tone(tone_hz, dds_scale, channel=tx_channel)
            self._tone_active = True
            self._active_tx_gain = float(tx_gain_db)
            if prime_after_arm:
                # Some Pluto+ runtime states require one IIO RX completion
                # after DDS arm before the first direct RX START.
                self._prime_iio_rx_dma()
        except Exception:
            self.stop_tone()
            raise

    def stop_tone(self) -> None:
        if not hasattr(self, "sdr"):
            return
        try:
            self.sdr.disable_dds()
        finally:
            try:
                self.sdr.tx_destroy_buffer()
            finally:
                self.sdr.tx_enabled_channels = []
                self.sdr.tx_hardwaregain_chan0 = -80
                self.sdr.tx_hardwaregain_chan1 = -80
                self.sdr.tx_cyclic_buffer = False
                self._tone_active = False
                self._active_tx_gain = None

    def discard(self, frame_count: int) -> None:
        if frame_count <= 0:
            return
        self.direct.capture(
            samples_per_channel=self.config.buffer_size,
            frame_count=frame_count,
        )

    def capture(self) -> PlutoRxBuffer:
        capture = self.direct.capture(
            samples_per_channel=self.config.buffer_size,
            frame_count=1,
        )
        if len(capture.frames) != 1:
            raise RuntimeError("direct USB did not return exactly one frame")
        return self._convert_wire_frame(capture.frames[0])

    def capture_after_discard(self, discard_frame_count: int) -> PlutoRxBuffer:
        """Capture discarded and recorded frames in one finite USB request."""

        if discard_frame_count < 0:
            raise ValueError("discard frame count cannot be negative")
        frame_count = discard_frame_count + 1
        capture = self.direct.capture(
            samples_per_channel=self.config.buffer_size,
            frame_count=frame_count,
        )
        if len(capture.frames) != frame_count:
            raise RuntimeError(
                f"direct USB returned {len(capture.frames)} frames, "
                f"expected {frame_count}"
            )
        return self._convert_wire_frame(capture.frames[-1])

    def _convert_wire_frame(self, wire_frame) -> PlutoRxBuffer:
        metadata = wire_frame.metadata
        if not isinstance(metadata, RadioMetadataV2):
            raise RuntimeError("V7 calibration requires radio metadata V2")
        unsafe = metadata.flags & UNSAFE_FLAGS
        if unsafe:
            raise RuntimeError(f"unsafe direct-USB metadata flags: 0x{int(unsafe):x}")
        if not metadata.gain_metadata_valid or not metadata.rssi_metadata_valid:
            raise RuntimeError("direct-USB gain/RSSI metadata is invalid")
        signal = iq_payload_to_complex64(
            wire_frame.iq_payload, metadata.samples_per_channel
        )
        first_change = np.asarray(
            [
                -1
                if metadata.rx1_first_change_sample == FIRST_CHANGE_UNAVAILABLE
                else metadata.rx1_first_change_sample,
                -1
                if metadata.rx2_first_change_sample == FIRST_CHANGE_UNAVAILABLE
                else metadata.rx2_first_change_sample,
            ],
            dtype=np.int32,
        )
        return PlutoRxBuffer(
            signal_matrix=signal,
            rssis=np.asarray(metadata.rssi_db_end, dtype=np.float64),
            gains=np.asarray(metadata.gain_db_end, dtype=np.float64),
            gain_index_start=np.full(2, 0xFF, dtype=np.uint8),
            gain_index_end=np.full(2, 0xFF, dtype=np.uint8),
            gain_metadata_valid=metadata.gain_metadata_valid,
            gain_endpoints_equal=np.asarray(
                metadata.gain_endpoints_equal, dtype=np.bool_
            ),
            gain_metadata_flags=int(metadata.flags),
            stream_id=metadata.stream_id,
            buffer_sequence=metadata.buffer_sequence,
            sample_sequence=metadata.first_sample_sequence,
            gain_start_read_duration_ns=metadata.gain_start_read_duration_ns,
            gain_end_read_duration_ns=metadata.gain_end_read_duration_ns,
            first_gain_change_sample=first_change,
            iq_power_dbfs=_iq_power_dbfs(signal),
            gain_db_start=np.asarray(metadata.gain_db_start, dtype=np.float32),
            gain_db_end=np.asarray(metadata.gain_db_end, dtype=np.float32),
            rssi_db_start=np.asarray(metadata.rssi_db_start, dtype=np.float32),
            rssi_db_end=np.asarray(metadata.rssi_db_end, dtype=np.float32),
            rssi_metadata_valid=metadata.rssi_metadata_valid,
            rssi_start_read_duration_ns=metadata.rssi_start_read_duration_ns,
            rssi_end_read_duration_ns=metadata.rssi_end_read_duration_ns,
        )

    def close(self) -> None:
        if getattr(self, "direct", None) is not None:
            self.direct.close()
            self.direct = None
        if getattr(self, "sdr", None) is not None:
            sdr = self.sdr
            try:
                self.stop_tone()
            finally:
                try:
                    sdr.rx_destroy_buffer()
                finally:
                    # Drop the final pyadi/libiio context reference now so a
                    # negotiated handoff retry can reclaim USB-IIO interface 5
                    # deterministically.
                    self.sdr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class IioLoopbackRadio(DirectUsbLoopbackRadio):
    """Pluto TX2 loopback captured through standard metadata-enabled libiio.

    ``uri`` may be either a discovered ``usb:`` URI or an explicit ``ip:``
    URI.  In both cases every call to :meth:`capture` performs one ordinary,
    request-driven IIO buffer refill; no independent producer or host-side IQ
    queue is introduced by this adapter.
    """

    def __init__(
        self,
        serial: str,
        config: CalibrationConfig,
        *,
        uri: str | None = None,
        adi_module=None,
        scan_contexts=None,
        metadata_receiver_class=IioMetadataRx,
    ):
        config.validate()
        if adi_module is None:
            import adi as adi_module
        if scan_contexts is None:
            import iio

            scan_contexts = iio.scan_contexts
        self.serial = serial
        self.config = config
        self.uri = uri or self._resolve_uri(scan_contexts())
        if not self.uri.startswith(("usb:", "ip:")):
            raise ValueError(f"unsupported IIO URI: {self.uri}")
        self.sdr = adi_module.ad9361(uri=self.uri)
        actual_serial = self.sdr._ctx.attrs.get("hw_serial")
        actual_serial = getattr(actual_serial, "value", actual_serial)
        if actual_serial != serial:
            self.sdr = None
            raise RuntimeError(
                f"IIO serial mismatch: requested {serial}, opened {actual_serial}"
            )
        self.direct = None
        self._tone_active = False
        self._active_tx_gain = None
        self._iio_rx = None
        try:
            self._configure_static()
            # MetadataBuffer may expose one buffer that was armed during open.
            # Seed that buffer with the experiment's reference gain so startup
            # cannot inherit a manual gain left by an earlier process.
            reference_gain = int(config.tx_reference_rx_gain_db)
            self.set_gains(reference_gain, reference_gain)
            self._iio_rx = metadata_receiver_class(
                self.sdr,
                sample_rate_hz=int(config.sample_rate_hz),
                samples_per_channel=int(config.buffer_size),
            )
            self._iio_rx.open()
        except Exception:
            self.close()
            raise

    def _prime_iio_rx_dma(self) -> None:
        """Complete one metadata IIO refill after arming DDS when requested."""

        self.capture()

    def identity(self) -> SdrDeviceIdentity:
        # These experiments require locally attached radios even for the IP
        # pass, so bind the network context to the same physical unit used in
        # the USB pass and to the boot-attested hardware fingerprint.
        usb_bus, usb_address, usb_port_path = _find_local_pluto_usb_device(self.serial)
        return SdrDeviceIdentity(
            sdr_family="pluto",
            serial=self.serial,
            receiver_uri=self.uri,
            rx_transport="iio",
            usb_vendor_id=PLUTO_USB_VENDOR_ID,
            usb_product_id=PLUTO_USB_PRODUCT_ID,
            usb_bus=usb_bus,
            usb_address=usb_address,
            usb_port_path=usb_port_path,
        )

    def discard(self, frame_count: int) -> None:
        for _ in range(max(0, int(frame_count))):
            self.capture()

    def capture(self) -> PlutoRxBuffer:
        if self._iio_rx is None:
            raise RuntimeError("IIO metadata RX is not open")
        signal, metadata, capture_time = self._iio_rx.capture()
        if not isinstance(metadata, RadioMetadataV3):
            raise RuntimeError("IIO calibration requires radio metadata V3")
        unsafe = metadata.flags & UNSAFE_FLAGS
        if unsafe:
            raise RuntimeError(f"unsafe IIO metadata flags: 0x{int(unsafe):x}")
        if not metadata.gain_metadata_valid or not metadata.rssi_metadata_valid:
            raise RuntimeError("IIO gain/RSSI metadata is invalid")
        first_change = np.asarray(
            [
                -1
                if metadata.rx1_first_change_sample == FIRST_CHANGE_UNAVAILABLE
                else metadata.rx1_first_change_sample,
                -1
                if metadata.rx2_first_change_sample == FIRST_CHANGE_UNAVAILABLE
                else metadata.rx2_first_change_sample,
            ],
            dtype=np.int32,
        )
        gain_series = _gain_series_arrays(metadata)
        observation_indices = gain_series["gain_observation_index"]
        if observation_indices.shape[0]:
            gain_index_start = observation_indices[0]
            gain_index_end = observation_indices[-1]
        else:
            gain_index_start = np.full(2, 0xFF, dtype=np.uint8)
            gain_index_end = np.full(2, 0xFF, dtype=np.uint8)
        return PlutoRxBuffer(
            signal_matrix=signal,
            rssis=np.asarray(metadata.rssi_db_end, dtype=np.float64),
            gains=np.asarray(metadata.gain_db_end, dtype=np.float64),
            gain_index_start=gain_index_start,
            gain_index_end=gain_index_end,
            gain_metadata_valid=metadata.gain_metadata_valid,
            gain_endpoints_equal=np.asarray(
                metadata.gain_endpoints_equal, dtype=np.bool_
            ),
            gain_metadata_flags=int(metadata.flags),
            stream_id=metadata.stream_id,
            buffer_sequence=metadata.buffer_sequence,
            sample_sequence=metadata.first_sample_sequence,
            gain_start_read_duration_ns=metadata.gain_start_read_duration_ns,
            gain_end_read_duration_ns=metadata.gain_end_read_duration_ns,
            first_gain_change_sample=first_change,
            iq_power_dbfs=_iq_power_dbfs(signal),
            gain_db_start=np.asarray(metadata.gain_db_start, dtype=np.float32),
            gain_db_end=np.asarray(metadata.gain_db_end, dtype=np.float32),
            rssi_db_start=np.asarray(metadata.rssi_db_start, dtype=np.float32),
            rssi_db_end=np.asarray(metadata.rssi_db_end, dtype=np.float32),
            rssi_metadata_valid=metadata.rssi_metadata_valid,
            rssi_start_read_duration_ns=metadata.rssi_start_read_duration_ns,
            rssi_end_read_duration_ns=metadata.rssi_end_read_duration_ns,
            **capture_time,
            **gain_series,
        )

    def capture_after_discard(self, discard_frame_count: int) -> PlutoRxBuffer:
        if discard_frame_count < 0:
            raise ValueError("discard frame count cannot be negative")
        self.discard(discard_frame_count)
        return self.capture()

    def close(self) -> None:
        receiver = getattr(self, "_iio_rx", None)
        self._iio_rx = None
        if receiver is not None:
            receiver.close()
        if getattr(self, "sdr", None) is not None:
            sdr = self.sdr
            try:
                self.stop_tone()
            finally:
                try:
                    sdr.rx_destroy_buffer()
                finally:
                    self.sdr = None
