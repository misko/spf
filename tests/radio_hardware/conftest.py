"""Explicit opt-in fixtures for tests that touch attached Pluto radios."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import usb1


PLUTO_VENDOR_ID = 0x0456
PLUTO_PRODUCT_ID = 0xB673


@dataclasses.dataclass(frozen=True, slots=True)
class AttachedPluto:
    serial: str
    bus: int
    address: int
    port_path: tuple[int, ...]


def pytest_addoption(parser):
    group = parser.getgroup("SPF attached radio hardware")
    group.addoption(
        "--radio-hardware",
        action="store_true",
        help="enable tests that claim and receive from attached Pluto radios",
    )
    group.addoption(
        "--radio-serial",
        action="append",
        default=[],
        help="test only this serial; repeat for multiple radios",
    )
    group.addoption(
        "--radio-expected-count",
        type=int,
        default=None,
        help="require exactly this many selected attached radios",
    )
    group.addoption(
        "--radio-direct-ip-transport",
        choices=("udp", "tcp"),
        default="udp",
        help=(
            "data transport for direct-IP gates. Defaults to udp so the "
            "existing numbers stay the baseline; 'tcp' fails closed rather "
            "than falling back, so a cell cannot silently run as udp"
        ),
    )
    group.addoption(
        "--radio-direct-ip-datagram-bytes",
        type=int,
        default=1_472,
        help=(
            "direct-IP chunk size. 65507 is the large-chunk cell, which "
            "isolates the effect of chunk size from the effect of transport"
        ),
    )
    group.addoption(
        "--radio-direct-ip-on-backlog",
        choices=("fail", "drop"),
        default="fail",
        help=(
            "backpressure policy. 'drop' sheds whole frames instead of "
            "stalling; run it only after the same cell passes on 'fail', "
            "since a drop-mode run cannot fail the way a fail-mode run does"
        ),
    )
    group.addoption(
        "--radio-samples",
        type=int,
        default=524_288,
        help="samples per channel in each direct-USB hardware frame",
    )
    group.addoption(
        "--radio-sample-rate",
        type=float,
        default=30_000_000.0,
        help=(
            "nominal device sample rate used only if the sample-clock gate "
            "cannot estimate the actual FPGA counter rate"
        ),
    )
    group.addoption(
        "--radio-direct-ip-burst-sample-rate",
        type=float,
        default=20_000_000.0,
        help=(
            "RF rate for the contiguous 16-frame direct-IP burn-in; the "
            "production single-frame/sample-clock gates remain at 30 MS/s"
        ),
    )
    group.addoption(
        "--radio-time-anchor-max-uncertainty-ms",
        type=float,
        default=5.0,
        help="maximum accepted GNSS-free frame-time uncertainty",
    )
    group.addoption(
        "--radio-cycles",
        type=int,
        default=10,
        help="repeated finite START/STOP cycles per radio",
    )
    group.addoption(
        "--radio-frames-per-request",
        type=int,
        default=3,
        help="frames in the contiguous-sequence request",
    )
    group.addoption(
        "--radio-max-rss-growth-mib",
        type=float,
        default=64.0,
        help="maximum host RSS growth during repeated capture",
    )
    group.addoption(
        "--radio-zarr",
        action="store_true",
        help="enable the hardware-backed V7 Zarr round-trip test",
    )
    group.addoption(
        "--radio-zarr-frames",
        type=int,
        default=3,
        help="frames per receiver in the hardware-backed V7 Zarr test",
    )
    group.addoption(
        "--radio-gain-series-v3",
        action="store_true",
        help="enable protocol-v3 sample-associated gain-series hardware gates",
    )
    group.addoption(
        "--radio-gain-observation-interval",
        type=int,
        default=2048,
        help="requested samples between protocol-v3 gain observations",
    )
    group.addoption(
        "--radio-gain-observation-capacity",
        type=int,
        default=256,
        help="fixed protocol-v3 gain-observation slots per frame",
    )
    group.addoption(
        "--radio-direct-ip",
        action="store_true",
        help="enable direct-IP hardware parity gates",
    )
    group.addoption(
        "--radio-direct-ip-host",
        default=None,
        help="unique LAN address of the selected Pluto direct-IP gadget",
    )
    group.addoption(
        "--radio-direct-ip-ladder",
        action="store_true",
        help="enable the bounded parallel two-radio direct-IP rate ladder",
    )
    group.addoption(
        "--radio-direct-ip-ladder-host",
        action="append",
        default=[],
        help="LAN address for the parallel IP ladder; repeat once per radio",
    )
    group.addoption(
        "--radio-direct-ip-ladder-rates",
        default="1M,1.25M,1.5M,2M,3M,6M,10M,15M,20M,25M,30M",
        help="strictly increasing comma-separated sample rates in Hz (M/K allowed)",
    )
    group.addoption(
        "--radio-direct-ip-ladder-cycles",
        type=int,
        default=3,
        help="parallel finite captures at every sample-rate rung",
    )
    group.addoption(
        "--radio-direct-ip-ladder-required-rate",
        type=float,
        default=3_000_000.0,
        help="highest rung which must preserve frame integrity for pytest to pass",
    )
    group.addoption(
        "--radio-direct-ip-ladder-continue-after-failure",
        action="store_true",
        help="continue with a fresh transport session at the next sample-rate rung",
    )
    group.addoption(
        "--radio-direct-ip-ladder-interface",
        default="eth0",
        help="host interface whose packet/drop counters are recorded",
    )
    group.addoption(
        "--radio-direct-ip-min-payload-mibps",
        type=float,
        default=20.0,
        help="minimum accepted end-to-end payload rate for the buffered IP burst",
    )
    group.addoption(
        "--radio-direct-ip-min-receive-buffer-mib",
        type=float,
        default=4.0,
        help="minimum effective host UDP receive buffer for the IP burst",
    )
    group.addoption(
        "--radio-tx-loopback",
        action="store_true",
        help="enable explicitly acknowledged, attenuated TX2 loopback tests",
    )
    group.addoption(
        "--radio-tx-loopback-attenuation-db",
        type=float,
        default=None,
        help=(
            "minimum physical attenuation from TX2 to either RX input; TX tests "
            "also require physical attenuation minus their strongest non-positive "
            "TX gain to total at least 30 dB"
        ),
    )
    group.addoption(
        "--radio-tx-lo-hz",
        type=int,
        default=2_412_000_000,
        help="RX/TX LO used by the cabled TX2 loopback test",
    )
    group.addoption(
        "--radio-tx-sample-rate",
        type=int,
        default=3_000_000,
        help="sample rate used by the cabled TX2 loopback test",
    )
    group.addoption(
        "--radio-tx-bandwidth",
        type=int,
        default=3_000_000,
        help="RX/TX RF bandwidth used by the cabled TX2 loopback test",
    )
    group.addoption(
        "--radio-tx-samples",
        type=int,
        default=65_536,
        help="samples per channel in each cabled TX2 loopback frame",
    )
    group.addoption(
        "--radio-tx-tone-hz",
        type=int,
        default=100_000,
        help="FPGA DDS tone offset used by the cabled TX2 loopback test",
    )
    group.addoption(
        "--radio-tx-gain-db",
        type=float,
        default=-10.0,
        help="nominal TX2 hardware gain used by the tone-quality gate",
    )
    group.addoption(
        "--radio-tx-weak-gain-db",
        type=float,
        default=-60.0,
        help="weak TX2 level used to exercise slow-attack AGC",
    )
    group.addoption(
        "--radio-tx-strong-gain-db",
        type=float,
        default=0.0,
        help="strong TX2 level used to exercise slow-attack AGC",
    )
    group.addoption(
        "--radio-interrupt",
        action="store_true",
        help="enable the real collector interruption/finalization test",
    )
    group.addoption(
        "--radio-interrupt-signal",
        choices=("sigint", "sigterm", "sigkill", "sigstop"),
        default="sigterm",
        help=(
            "signal used by the real collector interruption test; sigstop "
            "suspends longer than the USB/MAVLink deadline, then resumes"
        ),
    )
    group.addoption(
        "--radio-interrupt-min-records",
        type=int,
        default=2,
        help="minimum committed records per receiver before interruption",
    )
    group.addoption(
        "--radio-capture-config",
        type=Path,
        default=None,
        help="production V7 YAML for --radio-interrupt",
    )
    group.addoption(
        "--radio-device-mapping",
        type=Path,
        default=Path("/home/pi/device_mapping"),
        help="receiver-port mapping for --radio-interrupt",
    )
    group.addoption(
        "--radio-ready-manifest",
        type=Path,
        default=Path("/run/spf/direct_usb_ready.json"),
        help="boot readiness manifest for --radio-interrupt",
    )
    group.addoption(
        "--radio-soak",
        action="store_true",
        help="enable long-running attached-radio tests",
    )
    group.addoption(
        "--radio-burn-frequencies",
        default=(
            "868M,915M,1280M,1300M,1301M,2412M,2467.1M,"
            "4000M,4001M,5766M,5804M,5866M"
        ),
        help="comma-separated LO frequencies for the mixed USB/IP soak",
    )
    group.addoption(
        "--radio-crash-recovery",
        action="store_true",
        help="enable deliberate direct-USB daemon crash/rebind tests",
    )
    group.addoption(
        "--radio-rf-dc-tracking",
        action="store_true",
        help=(
            "enable tests that WRITE the AD9361 RF-DC tracking state; they "
            "restore the pre-test value, but they mutate chip configuration "
            "shared with every capture on this bench"
        ),
    )
    group.addoption(
        "--radio-report-dir",
        type=Path,
        default=None,
        help="optional directory for JSON hardware-test reports",
    )
    group.addoption(
        "--radio-iio-rate-ladder",
        default="1M,1.5M,2M,2.5M,3M,5M,10M,20M,30M",
        help="sample rates for the standard libiio USB/TCP benchmark",
    )
    group.addoption(
        "--radio-iio-rate-frames",
        type=int,
        default=12,
        help="timed frames per cell in the standard libiio USB/TCP benchmark",
    )
    group.addoption(
        "--radio-iio-rate-samples",
        type=int,
        default=262_144,
        help="samples per channel in each libiio USB/TCP benchmark frame",
    )


def pytest_collection_modifyitems(config, items):
    # When the complete ``tests`` tree is collected, pytest may discover this
    # nested conftest after command-line parsing.  In that case its options do
    # not exist in the global Config object.  Hardware tests must remain
    # fail-closed (skipped) rather than aborting collection with ValueError.
    hardware_enabled = config.getoption("--radio-hardware", default=False)
    zarr_enabled = config.getoption("--radio-zarr", default=False)
    gain_series_enabled = config.getoption("--radio-gain-series-v3", default=False)
    direct_ip_enabled = config.getoption("--radio-direct-ip", default=False)
    direct_ip_ladder_enabled = config.getoption(
        "--radio-direct-ip-ladder", default=False
    )
    tx_loopback_enabled = config.getoption("--radio-tx-loopback", default=False)
    interrupt_enabled = config.getoption("--radio-interrupt", default=False)
    soak_enabled = config.getoption("--radio-soak", default=False)
    crash_recovery_enabled = config.getoption("--radio-crash-recovery", default=False)
    rf_dc_tracking_enabled = config.getoption("--radio-rf-dc-tracking", default=False)
    hardware_skip = pytest.mark.skip(
        reason="requires explicit --radio-hardware and attached Pluto hardware"
    )
    zarr_skip = pytest.mark.skip(reason="requires explicit --radio-zarr")
    gain_series_skip = pytest.mark.skip(
        reason="requires explicit --radio-gain-series-v3"
    )
    direct_ip_skip = pytest.mark.skip(reason="requires explicit --radio-direct-ip")
    direct_ip_ladder_skip = pytest.mark.skip(
        reason="requires explicit --radio-direct-ip-ladder"
    )
    tx_loopback_skip = pytest.mark.skip(
        reason="requires explicit --radio-tx-loopback and an attenuated cable"
    )
    interrupt_skip = pytest.mark.skip(reason="requires explicit --radio-interrupt")
    soak_skip = pytest.mark.skip(reason="requires explicit --radio-soak")
    crash_recovery_skip = pytest.mark.skip(
        reason="requires explicit --radio-crash-recovery"
    )
    rf_dc_tracking_skip = pytest.mark.skip(
        reason="requires explicit --radio-rf-dc-tracking"
    )
    for item in items:
        if "radio_hardware" in item.keywords and not hardware_enabled:
            item.add_marker(hardware_skip)
        if "radio_zarr" in item.keywords and not zarr_enabled:
            item.add_marker(zarr_skip)
        if "radio_gain_series_v3" in item.keywords and not gain_series_enabled:
            item.add_marker(gain_series_skip)
        if "radio_direct_ip" in item.keywords and not direct_ip_enabled:
            item.add_marker(direct_ip_skip)
        if "radio_direct_ip_ladder" in item.keywords and not direct_ip_ladder_enabled:
            item.add_marker(direct_ip_ladder_skip)
        if "radio_tx_loopback" in item.keywords and not tx_loopback_enabled:
            item.add_marker(tx_loopback_skip)
        if "radio_interrupt" in item.keywords and not interrupt_enabled:
            item.add_marker(interrupt_skip)
        if "radio_soak" in item.keywords and not soak_enabled:
            item.add_marker(soak_skip)
        if "radio_crash_recovery" in item.keywords and not crash_recovery_enabled:
            item.add_marker(crash_recovery_skip)
        if "radio_rf_dc_tracking" in item.keywords and not rf_dc_tracking_enabled:
            item.add_marker(rf_dc_tracking_skip)


def _discover_attached_plutos() -> list[AttachedPluto]:
    context = usb1.USBContext()
    context.open()
    radios: list[AttachedPluto] = []
    try:
        for device in context.getDeviceIterator(skip_on_error=True):
            # Address zero is reserved for USB enumeration and cannot identify
            # a usable device. Some host controllers leave a libusb zombie at
            # bus 5/address 0 after a composite Pluto re-enumerates; attempting
            # to open its string descriptor raises LIBUSB_ERROR_NO_DEVICE and
            # must not hide the state of the real radios.
            if device.getDeviceAddress() == 0:
                continue
            if (
                device.getVendorID() != PLUTO_VENDOR_ID
                or device.getProductID() != PLUTO_PRODUCT_ID
            ):
                continue
            try:
                serial = device.getSerialNumber()
            except usb1.USBError as error:
                pytest.fail(
                    "Pluto is enumerated but its serial cannot be read: "
                    f"bus={device.getBusNumber()} address={device.getDeviceAddress()} "
                    f"error={error}"
                )
            radios.append(
                AttachedPluto(
                    serial=serial,
                    bus=device.getBusNumber(),
                    address=device.getDeviceAddress(),
                    port_path=tuple(device.getPortNumberList()),
                )
            )
    finally:
        context.close()
    return sorted(radios, key=lambda radio: (radio.port_path, radio.serial))


@pytest.fixture(scope="session")
def attached_plutos(pytestconfig) -> tuple[AttachedPluto, ...]:
    if not pytestconfig.getoption("--radio-hardware"):
        pytest.skip("attached-radio tests were not explicitly enabled")

    discovered = _discover_attached_plutos()
    requested = pytestconfig.getoption("--radio-serial")
    if requested:
        requested_set = set(requested)
        selected = [radio for radio in discovered if radio.serial in requested_set]
        missing = requested_set - {radio.serial for radio in selected}
        if missing:
            pytest.fail(f"requested Pluto serials are missing: {sorted(missing)}")
    else:
        selected = discovered

    expected = pytestconfig.getoption("--radio-expected-count")
    if expected is not None and len(selected) != expected:
        pytest.fail(
            f"expected exactly {expected} selected Pluto radios, found {len(selected)}: "
            f"{[radio.serial for radio in selected]}"
        )
    if not selected:
        pytest.fail("no attached Pluto radios were found")
    if len({radio.serial for radio in selected}) != len(selected):
        pytest.fail("attached Pluto serials are not unique")
    return tuple(selected)


@pytest.fixture(scope="session")
def direct_ip_transport_profile(pytestconfig) -> dict:
    """Transport settings for one cell of the direct-IP matrix.

    The whole point of passing these as one dict is that the gate bodies stay
    identical across cells; only the profile changes, so a difference in the
    result cannot come from a difference in the test.
    """

    return {
        "transport": pytestconfig.getoption("--radio-direct-ip-transport"),
        "max_datagram_bytes": pytestconfig.getoption(
            "--radio-direct-ip-datagram-bytes"
        ),
        "on_backlog": pytestconfig.getoption("--radio-direct-ip-on-backlog"),
    }


@pytest.fixture(scope="session")
def radio_report_dir(pytestconfig, tmp_path_factory) -> Path:
    configured = pytestconfig.getoption("--radio-report-dir")
    path = configured or tmp_path_factory.mktemp("radio_hardware_reports")
    path.mkdir(parents=True, exist_ok=True)
    return path
