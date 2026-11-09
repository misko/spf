"""Receive-only qualification of one explicitly selected, serial-attested radio.

Set SPF_TANDEM_TEST_URI and SPF_TANDEM_TEST_SERIAL to opt in. The ordinary
test suite never discovers radios or selects a default device for these tests.
"""

import errno
import json
import os
import select
import subprocess
import sys
import time

import pytest

URI = os.environ.get("SPF_TANDEM_TEST_URI")
SERIAL = os.environ.get("SPF_TANDEM_TEST_SERIAL")
pytestmark = pytest.mark.skipif(
    not (URI and SERIAL), reason="explicit radio selection required"
)


@pytest.fixture
def selected_radio(request):
    import iio

    from spf.sdrpluto.sdr_controller import PPlus, ReceiverConfig

    context = iio.Context(URI)
    assert context.attrs["hw_serial"] == SERIAL
    assert context.attrs["iio,buffer-metadata-legacy-agc"] == "1"
    context.close()
    frequency, samples = getattr(request, "param", (915000000, 262144))
    config = ReceiverConfig(
        lo=frequency,
        rf_bandwidth=2000000,
        sample_rate=2500000,
        intermediate=0,
        uri=f"pluto://{URI}",
        buffer_size=samples,
        gains=[20, 23],
        gain_control_modes=["manual", "manual"],
    )
    radio = PPlus(uri=config.uri, rx_config=config)
    try:
        radio.setup_rx_config()
        yield radio
    finally:
        radio.close()


def record(radio, mode, frame):
    row = dict(
        serial=SERIAL,
        uri=URI,
        frequency=radio.rx_config.lo,
        samples=radio.rx_config.buffer_size,
        mode=mode,
        gains=frame.gains.tolist(),
        epoch=frame.tandem_ownership_epoch,
        events=len(frame.gain_event_sample_sequence),
        sample_sequence=frame.sample_sequence,
        missing_samples=frame.missing_samples_before,
        sample_time_valid=frame.sample_time_valid,
    )
    if path := os.environ.get("SPF_TANDEM_TEST_REPORT"):
        with open(path, "a") as output:
            output.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    "selected_radio",
    [
        (frequency, samples)
        for frequency in (915000000, 2410000000, 5750000000)
        for samples in (262144, 524288)
    ],
    indirect=True,
)
def test_all_gain_modes_keep_metadata_and_release_ownership(selected_radio):
    radio = selected_radio
    epochs = []
    for mode in (
        "manual",
        "tandem",
        "slow_attack",
        "tandem",
        "fast_attack",
        "tandem",
        "manual",
    ):
        radio.set_gain_mode(mode)
        for _ in range(3):
            frame = radio.rx_with_metadata()
            assert (
                frame.gain_metadata_valid
                and frame.rssi_metadata_valid
                and frame.sample_time_valid
            )
            assert frame.requested_gain_modes == (mode, mode)
            assert (frame.tandem_ownership_epoch is not None) == (mode == "tandem")
            if mode == "tandem":
                assert frame.gains[0] == frame.gains[1]
            elif mode == "manual":
                assert frame.gains.tolist() == [20, 23]
            record(radio, mode, frame)
        if mode == "tandem":
            epochs.append(frame.tandem_ownership_epoch)
            # Bypass PyADI's deliberate no-op for non-manual cached gain modes.
            with pytest.raises(OSError) as blocked:
                radio.sdr._ctrl.find_channel("voltage0", False).attrs[
                    "hardwaregain"
                ].value = "17"
            assert blocked.value.errno == errno.EBUSY
        else:
            assert radio.sdr.gain_control_mode_chan0 == mode
            assert radio.sdr.gain_control_mode_chan1 == mode
    assert len(set(epochs)) == len(epochs)


def test_failed_enable_restores_unequal_manual_gains(selected_radio, monkeypatch):
    from spf.direct_radio.iio_metadata import IioMetadataRx

    radio = selected_radio
    radio.set_gain_mode("manual")
    original = IioMetadataRx.capture
    failed = False

    def fail_once(receiver):
        nonlocal failed
        if not failed and receiver._requested_tandem is not None:
            failed = True
            raise OSError("injected failure after acquiring tandem")
        return original(receiver)

    monkeypatch.setattr(IioMetadataRx, "capture", fail_once)
    with pytest.raises(OSError, match="injected failure"):
        radio.set_gain_mode("tandem")
    assert radio.rx_config.gain_control_modes == ["manual", "manual"]
    frame = radio.rx_with_metadata()
    assert frame.gains.tolist() == [20, 23]
    assert frame.tandem_ownership_epoch is None


def test_watchdog_releases_gain_and_stops_failed_capture(selected_radio):
    radio = selected_radio
    radio.set_gain_mode("tandem")
    # Deliberately stop servicing the owner; no heartbeat/status polling here.
    time.sleep(6.5)
    device = radio.sdr._ctx.find_device("tandem-agc")
    assert int(device.attrs["state"].value) == 4
    assert int(device.attrs["fault_flags"].value) != 0
    radio.sdr._ctrl.find_channel("voltage0", False).attrs["hardwaregain"].value = "17"
    with pytest.raises((OSError, RuntimeError)):
        radio.rx_with_metadata()
    assert radio.rx_config is None
    assert radio._iio_metadata_rx is None


def test_owner_process_exit_releases_the_radio(selected_radio):
    radio = selected_radio
    child_code = """
import os
from spf.sdrpluto.sdr_controller import PPlus, ReceiverConfig
config = ReceiverConfig(lo=915000000, rf_bandwidth=2000000, sample_rate=2500000,
    intermediate=0, uri='pluto://' + os.environ['SPF_TANDEM_TEST_URI'],
    buffer_size=262144, gains=[20,23], gain_control_modes=['tandem','tandem'])
radio = PPlus(uri=config.uri, rx_config=config)
radio.setup_rx_config()
radio.rx_with_metadata()
print('READY', flush=True)
while True:
    radio.rx_with_metadata()
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", child_code], stdout=subprocess.PIPE
    )
    try:
        assert select.select([process.stdout], [], [], 20)[0], (
            "child did not acquire the radio"
        )
        assert process.stdout.readline().strip() == b"READY"
        with pytest.raises(OSError) as blocked:
            radio.sdr._ctrl.find_channel("voltage0", False).attrs[
                "hardwaregain"
            ].value = "17"
        assert blocked.value.errno == errno.EBUSY
        process.kill()
        process.wait(timeout=10)
        deadline = time.monotonic() + 10
        while True:
            try:
                radio.sdr._ctrl.find_channel("voltage0", False).attrs[
                    "hardwaregain"
                ].value = "20"
                break
            except OSError as error:
                if error.errno != errno.EBUSY or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        radio.set_gain_mode("tandem")
        assert radio.rx_with_metadata().tandem_ownership_epoch is not None
        radio.set_gain_mode("manual")
        assert radio.rx_with_metadata().gains.tolist() == [20, 23]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_disconnected_transport_stops_capture_and_allows_reconnect(selected_radio):
    import copy

    from spf.sdrpluto.sdr_controller import PPlus

    # Dropping this private forwarding process disconnects only this test's
    # sockets. It does not change host networking or disconnect another radio.
    proxy_code = """
import asyncio, os
async def connection(reader, writer):
    remote_reader, remote_writer = await asyncio.open_connection(
        os.environ['SPF_TANDEM_TEST_URI'].removeprefix('ip:'), 30431)
    async def pump(source, destination):
        try:
            while chunk := await source.read(65536):
                destination.write(chunk)
                await destination.drain()
        finally:
            destination.close()
    await asyncio.gather(pump(reader, remote_writer), pump(remote_reader, writer))
async def main():
    server = await asyncio.start_server(connection, '127.0.0.1', 0)
    print(server.sockets[0].getsockname()[1], flush=True)
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""
    proxy = subprocess.Popen(
        [sys.executable, "-u", "-c", proxy_code], stdout=subprocess.PIPE
    )
    radio = None
    try:
        assert select.select([proxy.stdout], [], [], 10)[0]
        port = int(proxy.stdout.readline())
        config = copy.deepcopy(selected_radio.rx_config)
        config.uri = f"pluto://ip:127.0.0.1:{port}"
        config.gain_control_modes = ["tandem", "tandem"]
        radio = PPlus(uri=config.uri, rx_config=config)
        radio.sdr._ctx.set_timeout(2000)
        radio.setup_rx_config()
        assert radio.rx_with_metadata().tandem_ownership_epoch is not None
        proxy.kill()
        proxy.wait(timeout=10)
        with pytest.raises((OSError, RuntimeError)):
            radio.rx_with_metadata()
        assert radio.rx_config is None
        assert radio._iio_metadata_rx is None
        device = selected_radio.sdr._ctx.find_device("tandem-agc")
        deadline = time.monotonic() + 10
        while int(device.attrs["state"].value) not in (0, 4):
            assert time.monotonic() < deadline, "disconnected owner was not released"
            time.sleep(0.1)
        selected_radio.set_gain_mode("tandem")
        assert selected_radio.rx_with_metadata().tandem_ownership_epoch is not None
        selected_radio.set_gain_mode("manual")
        assert selected_radio.rx_with_metadata().gains.tolist() == [20, 23]
    finally:
        if proxy.poll() is None:
            proxy.kill()
            proxy.wait(timeout=10)
        # The disconnected context cannot acknowledge TX cleanup. The selected
        # direct connection's fixture performs that cleanup on the same radio.
        if radio is not None:
            radio.sdr = None
