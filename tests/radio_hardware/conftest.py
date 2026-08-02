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
        "--radio-samples",
        type=int,
        default=524_288,
        help="samples per channel in each direct-USB hardware frame",
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
        "--radio-crash-recovery",
        action="store_true",
        help="enable deliberate direct-USB daemon crash/rebind tests",
    )
    group.addoption(
        "--radio-report-dir",
        type=Path,
        default=None,
        help="optional directory for JSON hardware-test reports",
    )


def pytest_collection_modifyitems(config, items):
    # When the complete ``tests`` tree is collected, pytest may discover this
    # nested conftest after command-line parsing.  In that case its options do
    # not exist in the global Config object.  Hardware tests must remain
    # fail-closed (skipped) rather than aborting collection with ValueError.
    hardware_enabled = config.getoption("--radio-hardware", default=False)
    zarr_enabled = config.getoption("--radio-zarr", default=False)
    interrupt_enabled = config.getoption("--radio-interrupt", default=False)
    soak_enabled = config.getoption("--radio-soak", default=False)
    crash_recovery_enabled = config.getoption("--radio-crash-recovery", default=False)
    hardware_skip = pytest.mark.skip(
        reason="requires explicit --radio-hardware and attached Pluto hardware"
    )
    zarr_skip = pytest.mark.skip(reason="requires explicit --radio-zarr")
    interrupt_skip = pytest.mark.skip(reason="requires explicit --radio-interrupt")
    soak_skip = pytest.mark.skip(reason="requires explicit --radio-soak")
    crash_recovery_skip = pytest.mark.skip(
        reason="requires explicit --radio-crash-recovery"
    )
    for item in items:
        if "radio_hardware" in item.keywords and not hardware_enabled:
            item.add_marker(hardware_skip)
        if "radio_zarr" in item.keywords and not zarr_enabled:
            item.add_marker(zarr_skip)
        if "radio_interrupt" in item.keywords and not interrupt_enabled:
            item.add_marker(interrupt_skip)
        if "radio_soak" in item.keywords and not soak_enabled:
            item.add_marker(soak_skip)
        if "radio_crash_recovery" in item.keywords and not crash_recovery_enabled:
            item.add_marker(crash_recovery_skip)


def _discover_attached_plutos() -> list[AttachedPluto]:
    context = usb1.USBContext()
    context.open()
    radios: list[AttachedPluto] = []
    try:
        for device in context.getDeviceIterator(skip_on_error=True):
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
def radio_report_dir(pytestconfig, tmp_path_factory) -> Path:
    configured = pytestconfig.getoption("--radio-report-dir")
    path = configured or tmp_path_factory.mktemp("radio_hardware_reports")
    path.mkdir(parents=True, exist_ok=True)
    return path
