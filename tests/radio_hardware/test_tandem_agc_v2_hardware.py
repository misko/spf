"""Tandem-AGC v2 ownership and attenuated TX2 loopback qualification."""

from __future__ import annotations

import errno
import gc
import json
import select
import shutil
import subprocess
import time
from contextlib import ExitStack
from itertools import pairwise

import numpy as np
import pytest

from spf.bench.dual_rx_phase import ToneQualityThresholds, analyze_common_tone
from spf.direct_radio.iio_metadata import IioMetadataRx
from spf.direct_radio.tandem_agc import (
    RadioMetadataV4,
    TandemEventDirection,
    TandemMode,
    TandemSessionRequestV1,
    TandemState,
)
from spf.direct_radio.usb_protocol import GainObservationFlags, MetadataFlags
from spf.scripts.mute_pluto_tx import (
    mute_attached_plutos,
    mute_sdr_tx,
    validate_loopback_safety,
)
from spf.scripts.pluto_multi_firmware import IsolatedPlutoNetwork

pytestmark = [pytest.mark.radio_hardware, pytest.mark.radio_tandem_agc]

SAMPLES_PER_CHANNEL = 65_536
SAMPLE_RATE_HZ = 3_000_000
RF_BANDWIDTH_HZ = 1_500_000
LO_HZ = 915_000_000
TONE_HZ = 100_000
# Keep equivalent detector margin for both supported safe loopbacks:
# 20 dB physical / -10 dB TX and 30 dB physical / 0 dB TX.
INITIAL_GAIN_DB = 30
AUTO_FRAMES_PER_STIMULUS = 24
MAX_AUTO_STIMULUS_ATTEMPTS = 3
AUTO_CAPTURE_PACE_SECONDS = 0.02
AUTO_QUALIFICATION_COOLDOWN_PERIODS = 16
TX_PATTERN_READY = b"SPF_TX_PATTERN_READY\n"
WATCHDOG_FAULT = 1 << 18
WATCHDOG_SETTLE_SECONDS = 6.5
UNSAFE_FLAGS = (
    MetadataFlags.DUMMY_GAINS
    | MetadataFlags.GAIN_READ_FAILED
    | MetadataFlags.RSSI_READ_FAILED
    | MetadataFlags.DEVICE_IIO_OVERFLOW
    | MetadataFlags.FPGA_EVENT_OVERFLOW
    | MetadataFlags.GAIN_OBSERVATION_OVERFLOW
)


@pytest.fixture(autouse=True)
def tx_safety_guard(attached_plutos):
    serials = [radio.serial for radio in attached_plutos]
    mute_attached_plutos(serials=serials, expected_count=len(serials))
    try:
        yield
    finally:
        mute_attached_plutos(serials=serials, expected_count=len(serials))


def _usb_uri(serial: str) -> str:
    import iio

    matches = [
        uri
        for uri, description in iio.scan_contexts().items()
        if uri.startswith("usb:") and f"serial={serial}" in description
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one USB-IIO URI for {serial}, found {matches}")
    return matches[0]


def _configure(sdr) -> None:
    sdr.tx_hardwaregain_chan0 = -80
    sdr.tx_hardwaregain_chan1 = -80
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = SAMPLE_RATE_HZ
    sdr.rx_rf_bandwidth = RF_BANDWIDTH_HZ
    sdr.tx_rf_bandwidth = RF_BANDWIDTH_HZ
    sdr.rx_lo = LO_HZ
    sdr.tx_lo = LO_HZ
    sdr.rx_buffer_size = SAMPLES_PER_CHANNEL
    sdr._rxadc.set_kernel_buffers_count(2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = INITIAL_GAIN_DB
    sdr.rx_hardwaregain_chan1 = INITIAL_GAIN_DB


def _mute_sdr(sdr) -> None:
    mute_sdr_tx(sdr)


def _assert_common(metadata: RadioMetadataV4, mode: TandemMode) -> None:
    expected_state = (
        TandemState.ARMED_HOLD if mode is TandemMode.HOLD else TandemState.ARMED_AUTO
    )
    assert metadata.tandem_state is expected_state
    assert metadata.tandem_fault_flags == 0
    assert not metadata.flags & UNSAFE_FLAGS
    assert metadata.rx1_gain_index == metadata.rx2_gain_index
    assert metadata.minimum_gain_db <= INITIAL_GAIN_DB <= metadata.maximum_gain_db
    assert metadata.minimum_gain_index <= metadata.rx1_gain_index
    assert metadata.rx1_gain_index <= metadata.maximum_gain_index
    for observation in metadata.gain_observations:
        required = (
            GainObservationFlags.VALID | GainObservationFlags.SAMPLE_INTERVAL_VALID
        )
        assert observation.flags & required == required
        assert observation.rx1_gain_index == observation.rx2_gain_index


def _open_receiver(sdr, mode: TandemMode) -> IioMetadataRx:
    request_options = {}
    if mode is TandemMode.AUTO:
        request_options["cooldown_periods"] = AUTO_QUALIFICATION_COOLDOWN_PERIODS
    receiver = IioMetadataRx(
        sdr,
        sample_rate_hz=SAMPLE_RATE_HZ,
        samples_per_channel=SAMPLES_PER_CHANNEL,
        tandem_request=TandemSessionRequestV1(
            mode=mode,
            initial_gain_db=INITIAL_GAIN_DB,
            **request_options,
        ),
    )
    receiver.open()
    return receiver


def _arm_verified_tx2_tone(sdr, strong_tx_gain_db: float) -> dict:
    thresholds = ToneQualityThresholds(
        min_tone_snr_db=6.0,
        min_tone_dbfs=-75.0,
        max_tone_dbfs=-3.0,
        max_clipping_fraction=0.0,
        min_coherence=0.98,
        max_within_capture_phase_std_deg=5.0,
    )
    analysis = None
    for arm_attempt in range(1, 4):
        _mute_sdr(sdr)
        sdr._ctrl.attrs["calib_mode"].value = "tx_quad"
        sdr.tx_hardwaregain_chan0 = -80
        sdr.tx_hardwaregain_chan1 = strong_tx_gain_db
        sdr.dds_single_tone(TONE_HZ, 0.25, channel=1)
        time.sleep(0.25)
        signal = np.asarray(sdr.rx())
        sdr.rx_destroy_buffer()
        analysis = analyze_common_tone(
            signal,
            sample_rate_hz=SAMPLE_RATE_HZ,
            expected_tone_offset_hz=TONE_HZ,
            tone_search_width_hz=25_000,
            transient_samples=1_024,
            phase_segments=8,
            thresholds=thresholds,
        )
        if analysis["quality_valid"]:
            analysis["arm_attempt_count"] = arm_attempt
            return analysis
    raise AssertionError(json.dumps(analysis, sort_keys=True))


def _start_remote_tx_pattern(
    host: str,
    serial: str,
    *,
    strong_tx_gain_db: float,
    weak_tx_gain_db: float,
    network: IsolatedPlutoNetwork | None = None,
) -> subprocess.Popen:
    if shutil.which("sshpass") is None:
        raise RuntimeError("sshpass is required for the bounded TX2 stimulus")
    command = f"""
set -eu
serial_path=/sys/kernel/config/usb_gadget/composite_gadget/strings/0x409/serialnumber
test "$(cat "$serial_path")" = {serial}
gain_path=/sys/bus/iio/devices/iio:device0/out_voltage1_hardwaregain
trap 'echo -80.000000 > "$gain_path"' EXIT HUP INT TERM
echo SPF_TX_PATTERN_READY
iteration=0
while test "$iteration" -lt 1; do
    echo {weak_tx_gain_db:.6f} > "$gain_path"
    usleep 150000
    echo {strong_tx_gain_db:.6f} > "$gain_path"
    usleep 400000
    iteration=$((iteration + 1))
done
"""
    ssh_command = [
        "sshpass",
        "-p",
        "analog",
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{host}",
        command,
    ]
    launcher = network.popen if network is not None else subprocess.Popen
    process = launcher(
        ssh_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], 5)
    if not readable or process.stdout.readline() != TX_PATTERN_READY:
        _stop_process(process)
        raise RuntimeError("bounded TX2 stimulus did not become ready")
    return process


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        # The remote shell's EXIT trap is the authoritative mute.  Closing the
        # local SSH client does not reliably deliver HUP to a non-PTY remote
        # shell, so let this short bounded pattern end by itself first.
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def _exercise_tandem_hold(uri: str, serial: str) -> None:
    import adi

    sdr = adi.ad9361(uri=uri)
    receiver = None
    try:
        assert sdr._ctx.attrs.get("hw_serial") == serial
        _configure(sdr)
        receiver = _open_receiver(sdr, TandemMode.HOLD)
        frames = [receiver.capture()[1] for _ in range(2)]
        assert all(isinstance(item, RadioMetadataV4) for item in frames)
        assert len({item.ownership_epoch for item in frames}) == 1
        for metadata in frames:
            _assert_common(metadata, TandemMode.HOLD)
            assert metadata.tandem_transition_count == 0
            assert not metadata.gain_events

        with pytest.raises(OSError) as blocked:
            sdr.rx_hardwaregain_chan0 = INITIAL_GAIN_DB + 1
        assert blocked.value.errno == errno.EBUSY

        # TX attenuation is outside the RX ownership transaction and must
        # remain writable for a separately guarded calibration source.
        sdr.tx_hardwaregain_chan1 = -70
        sdr.tx_hardwaregain_chan1 = -80

        receiver.close()
        receiver = None
        sdr.rx_hardwaregain_chan0 = INITIAL_GAIN_DB + 1
        sdr.rx_hardwaregain_chan0 = INITIAL_GAIN_DB
    finally:
        try:
            if receiver is not None:
                receiver.close()
        finally:
            _mute_sdr(sdr)
        sdr.rx_destroy_buffer()
        sdr._ctx.close()
    del sdr
    gc.collect()


def test_tandem_hold_owns_rx_and_restores_every_selected_radio(attached_plutos):
    for attached in attached_plutos:
        _exercise_tandem_hold(_usb_uri(attached.serial), attached.serial)


def test_tandem_hold_metadata_and_ownership_work_over_lan(
    attached_plutos, radio_lan_hosts
):
    for attached in attached_plutos:
        host = radio_lan_hosts.get(attached.serial)
        assert host is not None, f"no serial-bound LAN host for {attached.serial}"
        _exercise_tandem_hold(f"ip:{host}", attached.serial)


def test_tandem_stalled_owner_is_rolled_back(attached_plutos):
    import adi

    for attached in attached_plutos:
        sdr = adi.ad9361(uri=_usb_uri(attached.serial))
        receiver = None
        try:
            assert sdr._ctx.attrs.get("hw_serial") == attached.serial
            _configure(sdr)
            receiver = _open_receiver(sdr, TandemMode.HOLD)

            # Deliberately make no refill/status progress beyond the five-second
            # kernel deadline while the descriptor and process remain alive.
            time.sleep(WATCHDOG_SETTLE_SECONDS)
            tandem = sdr._ctx.find_device("tandem-agc")
            assert tandem is not None
            assert int(tandem.attrs["state"].value) == int(TandemState.FAULTED)
            assert int(tandem.attrs["fault_flags"].value) & WATCHDOG_FAULT

            # Automatic rollback must unlock RX controls even before the stale
            # userspace buffer is finally closed.
            sdr.rx_hardwaregain_chan0 = INITIAL_GAIN_DB + 1
            sdr.rx_hardwaregain_chan0 = INITIAL_GAIN_DB
        finally:
            try:
                if receiver is not None:
                    receiver.close()
            finally:
                _mute_sdr(sdr)
                sdr.rx_destroy_buffer()
                sdr._ctx.close()
        del sdr
        gc.collect()


@pytest.mark.radio_tx_loopback
def test_tandem_auto_events_are_paired_and_sample_aligned(
    attached_plutos, radio_lan_hosts, pytestconfig, radio_report_dir
):
    attenuation = pytestconfig.getoption("--radio-tx-loopback-attenuation-db")
    strong_tx_gain_db = pytestconfig.getoption("--radio-tx-strong-gain-db")
    weak_tx_gain_db = pytestconfig.getoption("--radio-tx-weak-gain-db")
    if weak_tx_gain_db >= strong_tx_gain_db:
        raise ValueError("weak TX gain must be below strong TX gain")
    validate_loopback_safety(
        physical_attenuation_db=attenuation,
        strongest_tx_gain_db=strong_tx_gain_db,
    )
    import adi

    report = {
        "attenuation_db": attenuation,
        "strong_tx_gain_db": strong_tx_gain_db,
        "weak_tx_gain_db": weak_tx_gain_db,
        "radios": [],
    }
    report_path = radio_report_dir / "tandem_agc_v2.json"
    for attached in attached_plutos:
        sdr = adi.ad9361(uri=_usb_uri(attached.serial))
        receiver = None
        stimulus = None
        try:
            assert sdr._ctx.attrs.get("hw_serial") == attached.serial
            _configure(sdr)
            tone_preflight = _arm_verified_tx2_tone(sdr, strong_tx_gain_db)
            sdr.tx_hardwaregain_chan1 = weak_tx_gain_db
            receiver = _open_receiver(sdr, TandemMode.AUTO)
            with ExitStack() as stimulus_stack:
                host = radio_lan_hosts.get(attached.serial)
                network = None
                stimulus_transport = "lan"
                if host is None:
                    network = stimulus_stack.enter_context(
                        IsolatedPlutoNetwork(attached.serial)
                    )
                    host = "192.168.2.1"
                    stimulus_transport = "serial-scoped-usb-network"
                frames = []
                all_events = []
                stimulus_attempt_count = 0
                for stimulus_attempt_count in range(
                    1, MAX_AUTO_STIMULUS_ATTEMPTS + 1
                ):
                    stimulus = _start_remote_tx_pattern(
                        host,
                        attached.serial,
                        strong_tx_gain_db=strong_tx_gain_db,
                        weak_tx_gain_db=weak_tx_gain_db,
                        network=network,
                    )
                    try:
                        for _ in range(AUTO_FRAMES_PER_STIMULUS):
                            metadata = receiver.capture()[1]
                            _assert_common(metadata, TandemMode.AUTO)
                            frames.append(metadata)
                            all_events.extend(metadata.gain_events)
                            # DMA may have several completed buffers queued, so
                            # refill calls alone do not prove that wall-clock
                            # stimulus time advanced.  Pace below one physical
                            # frame period while continuing to drain the FIFO.
                            time.sleep(AUTO_CAPTURE_PACE_SECONDS)
                    finally:
                        _stop_process(stimulus)
                        stimulus = None
                    directions = {event.direction for event in all_events}
                    if len(all_events) >= 4 and directions == {
                        TandemEventDirection.INCREASE,
                        TandemEventDirection.DECREASE,
                    }:
                        break
            receiver.close()
            receiver = None
            tandem = sdr._ctx.find_device("tandem-agc")
            assert tandem is not None
            assert int(tandem.attrs["state"].value) == int(TandemState.IDLE)
            assert int(tandem.attrs["fault_flags"].value) == 0
            assert int(tandem.attrs["overflow_count"].value) == 0
            event_diagnostics = {
                "frame_count": len(frames),
                "event_count": len(all_events),
                "first_transition_count": frames[0].tandem_transition_count,
                "last_transition_count": frames[-1].tandem_transition_count,
                "first_gain_indices": [
                    frames[0].rx1_gain_index,
                    frames[0].rx2_gain_index,
                ],
                "last_gain_indices": [
                    frames[-1].rx1_gain_index,
                    frames[-1].rx2_gain_index,
                ],
            }
            assert len(all_events) >= 4, (
                "bounded TX2 steps produced too few events: "
                f"{event_diagnostics}"
            )
            assert {event.direction for event in all_events} == {
                TandemEventDirection.INCREASE,
                TandemEventDirection.DECREASE,
            }, "bounded TX2 steps did not prove bidirectional AUTO control"

            for previous, current in pairwise(all_events):
                event_delta = (
                    current.event_sequence - previous.event_sequence
                ) & 0xFFFFFFFF
                assert 1 <= event_delta < (1 << 31)
                if event_delta == 1:
                    expected_delta = (
                        1
                        if current.direction is TandemEventDirection.INCREASE
                        else -1
                    )
                    assert (
                        current.rx1_gain_index - previous.rx1_gain_index
                        == expected_delta
                    )

            for previous, current in pairwise(frames):
                buffer_delta = current.buffer_sequence - previous.buffer_sequence
                assert buffer_delta >= 1
                # iiOD rejects, rather than fabricates, an IQ frame when every
                # periodic AUTO gain read is torn by an FPGA transition.  The
                # public buffer sequence must account exactly for such gaps.
                assert current.first_sample_sequence == (
                    previous.first_sample_sequence
                    + buffer_delta * previous.samples_per_channel
                )
                assert current.ownership_epoch == previous.ownership_epoch
                transition_delta = (
                    current.tandem_transition_count - previous.tandem_transition_count
                ) & 0xFFFFFFFF
                if buffer_delta == 1:
                    assert transition_delta == len(current.gain_events)
                else:
                    assert transition_delta >= len(current.gain_events)

            report["radios"].append(
                {
                    "serial": attached.serial,
                    "usb_uri": _usb_uri(attached.serial),
                    "stimulus_host": host,
                    "stimulus_transport": stimulus_transport,
                    "stimulus_attempt_count": stimulus_attempt_count,
                    "tone_preflight": tone_preflight,
                    "ownership_epoch": frames[0].ownership_epoch,
                    "frame_count": len(frames),
                    "event_count": len(all_events),
                    "increase_event_count": sum(
                        event.direction is TandemEventDirection.INCREASE
                        for event in all_events
                    ),
                    "decrease_event_count": sum(
                        event.direction is TandemEventDirection.DECREASE
                        for event in all_events
                    ),
                    "first_event_sequence": all_events[0].event_sequence,
                    "last_event_sequence": all_events[-1].event_sequence,
                }
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            _stop_process(stimulus)
            try:
                if receiver is not None:
                    receiver.close()
            finally:
                _mute_sdr(sdr)
                sdr.rx_destroy_buffer()
                sdr._ctx.close()
        del sdr
        gc.collect()

    assert len(report["radios"]) == len(attached_plutos)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
