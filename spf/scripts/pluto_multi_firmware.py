"""Safely RAM-load direct-USB firmware on multiple attached Pluto radios.

Every Pluto exposes the same USB-network address (192.168.2.1). To address a
specific radio without unplugging its neighbours, this module temporarily
moves that radio's USB-network interface into a private network namespace.
USB-IIO and DFU continue to use the original physical USB path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid


PLUTO_VENDOR = "0456"
PLUTO_RUNTIME_PRODUCT = "b673"
PLUTO_DFU_PRODUCT = "b674"
PLUTO_HOST_ADDRESS = "192.168.2.1"
HOST_NAMESPACE_ADDRESS = "192.168.2.10/24"
DIRECT_USB_INTERFACE = 6
REQUIRED_UBOOT_ENV = {
    "attr_name": "compatible",
    "attr_val": "ad9361",
    "compatible": "ad9361",
    "mode": "2r2t",
}
DEFAULT_APPROVED_QSPI_DEVICE_FW = ("v0.37-dirty",)


class FirmwareError(RuntimeError):
    """A firmware operation failed its safety contract."""


@dataclasses.dataclass(frozen=True)
class UsbPluto:
    serial: str
    sysfs_name: str
    bus: int
    port_path: str
    direct_usb: bool
    address: int | None = None


@dataclasses.dataclass(frozen=True)
class InterfaceState:
    name: str
    address: str | None
    prefixlen: int | None
    route_metric: int | None


def _env_is_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _read(path: Path) -> str:
    return path.read_text().strip()


def _read_optional(path: Path) -> str | None:
    """Read a transient sysfs value, returning absent across re-enumeration."""

    try:
        return _read(path)
    except OSError:
        # A USB sysfs node can disappear after is_file()/exists() succeeds
        # while the device changes from runtime to DFU (or back). Treat that
        # normal re-enumeration window as absence and keep polling.
        return None


def _run(
    command: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        env=env,
    )


def _require_commands(commands: tuple[str, ...]) -> None:
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        raise FirmwareError(f"required commands are missing: {', '.join(missing)}")


def parse_uboot_environment(output: str) -> dict[str, str]:
    environment: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            environment[key] = value
    return environment


def parse_device_fw_version(output: str) -> str:
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "device-fw":
            return fields[1]
    raise FirmwareError("/opt/VERSIONS does not contain a device-fw version")


def _interface_class_path(device_path: Path, interface: int) -> Path:
    return device_path.parent / f"{device_path.name}:1.{interface}" / "bInterfaceClass"


def discover_runtime_plutos(
    usb_root: Path = Path("/sys/bus/usb/devices"),
) -> list[UsbPluto]:
    devices: list[UsbPluto] = []
    for device_path in usb_root.iterdir():
        vendor_path = device_path / "idVendor"
        product_path = device_path / "idProduct"
        serial_path = device_path / "serial"
        if not (
            vendor_path.is_file() and product_path.is_file() and serial_path.is_file()
        ):
            continue
        if _read(vendor_path).lower() != PLUTO_VENDOR:
            continue
        if _read(product_path).lower() != PLUTO_RUNTIME_PRODUCT:
            continue
        serial = _read(serial_path)
        if not serial:
            raise FirmwareError(f"{device_path.name}: Pluto has an empty USB serial")
        direct_class_path = _interface_class_path(device_path, DIRECT_USB_INTERFACE)
        direct_usb = (
            direct_class_path.is_file() and _read(direct_class_path).lower() == "ff"
        )
        devices.append(
            UsbPluto(
                serial=serial,
                sysfs_name=device_path.name,
                bus=int(_read(device_path / "busnum")),
                port_path=_read(device_path / "devpath"),
                direct_usb=direct_usb,
                address=int(_read(device_path / "devnum")),
            )
        )
    devices.sort(key=lambda device: (device.bus, device.port_path, device.serial))
    serials = [device.serial for device in devices]
    if len(serials) != len(set(serials)):
        raise FirmwareError(f"duplicate Pluto serials: {serials}")
    return devices


def _udev_properties(interface: str) -> dict[str, str]:
    result = _run(
        [
            "udevadm",
            "info",
            "--query=property",
            f"--path=/sys/class/net/{interface}",
        ],
        capture_output=True,
    )
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def find_network_interface(serial: str) -> str:
    matches: list[str] = []
    for interface_path in Path("/sys/class/net").iterdir():
        try:
            properties = _udev_properties(interface_path.name)
        except subprocess.CalledProcessError:
            continue
        if (
            properties.get("ID_VENDOR_ID", "").lower() == PLUTO_VENDOR
            and properties.get("ID_MODEL_ID", "").lower() == PLUTO_RUNTIME_PRODUCT
            and properties.get("ID_SERIAL_SHORT") == serial
        ):
            matches.append(interface_path.name)
    if len(matches) != 1:
        raise FirmwareError(
            f"expected one USB-network interface for Pluto {serial}; found {matches}"
        )
    return matches[0]


def _capture_interface_state(interface: str) -> InterfaceState:
    address_result = _run(
        ["ip", "-j", "address", "show", "dev", interface],
        capture_output=True,
    )
    address_data = json.loads(address_result.stdout)
    ipv4 = [
        address
        for address in address_data[0].get("addr_info", [])
        if address.get("family") == "inet"
    ]
    if len(ipv4) > 1:
        raise FirmwareError(
            f"{interface}: expected at most one IPv4 address before isolation; "
            f"found {ipv4}"
        )
    route_result = _run(
        ["ip", "-j", "route", "show", "dev", interface],
        capture_output=True,
    )
    routes = json.loads(route_result.stdout)
    metric = routes[0].get("metric") if routes else None
    return InterfaceState(
        name=interface,
        address=ipv4[0]["local"] if ipv4 else None,
        prefixlen=int(ipv4[0]["prefixlen"]) if ipv4 else None,
        route_metric=int(metric) if metric is not None else None,
    )


class IsolatedPlutoNetwork:
    """Temporarily isolate one Pluto USB-network interface by serial."""

    def __init__(self, serial: str):
        self.serial = serial
        self.namespace = f"spf-pluto-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.state: InterfaceState | None = None

    def __enter__(self) -> "IsolatedPlutoNetwork":
        interface = find_network_interface(self.serial)
        self.state = _capture_interface_state(interface)
        _run(["ip", "netns", "add", self.namespace])
        try:
            _run(["ip", "link", "set", interface, "netns", self.namespace])
            _run(["ip", "-n", self.namespace, "link", "set", "lo", "up"])
            _run(
                [
                    "ip",
                    "-n",
                    self.namespace,
                    "address",
                    "flush",
                    "dev",
                    interface,
                ]
            )
            _run(
                [
                    "ip",
                    "-n",
                    self.namespace,
                    "address",
                    "add",
                    HOST_NAMESPACE_ADDRESS,
                    "dev",
                    interface,
                ]
            )
            _run(
                [
                    "ip",
                    "-n",
                    self.namespace,
                    "link",
                    "set",
                    interface,
                    "up",
                ]
            )
        except Exception:
            self._restore()
            raise
        return self

    @property
    def interface(self) -> str:
        if self.state is None:
            raise FirmwareError("network namespace is not active")
        return self.state.name

    def run(self, command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        return _run(["ip", "netns", "exec", self.namespace, *command], **kwargs)

    def _interface_is_present(self) -> bool:
        if self.state is None:
            return False
        result = _run(
            ["ip", "-n", self.namespace, "link", "show", self.state.name],
            check=False,
            capture_output=True,
        )
        return result.returncode == 0

    def _restore(self) -> None:
        if self.state is None:
            return
        if self._interface_is_present():
            _run(
                [
                    "ip",
                    "netns",
                    "exec",
                    self.namespace,
                    "ip",
                    "link",
                    "set",
                    self.state.name,
                    "netns",
                    "1",
                ],
                check=False,
            )
            _run(["ip", "link", "set", self.state.name, "up"], check=False)
            if self.state.address is not None and self.state.prefixlen is not None:
                _run(
                    [
                        "ip",
                        "address",
                        "replace",
                        f"{self.state.address}/{self.state.prefixlen}",
                        "dev",
                        self.state.name,
                    ],
                    check=False,
                )
            if self.state.route_metric is not None and self.state.address is not None:
                subnet = ".".join(self.state.address.split(".")[:3]) + ".0/24"
                _run(
                    ["ip", "route", "del", subnet, "dev", self.state.name],
                    check=False,
                )
                _run(
                    [
                        "ip",
                        "route",
                        "replace",
                        subnet,
                        "dev",
                        self.state.name,
                        "src",
                        self.state.address,
                        "metric",
                        str(self.state.route_metric),
                    ],
                    check=False,
                )
        _run(["ip", "netns", "delete", self.namespace], check=False)
        self.state = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._restore()


PASSIVE_DEVICE_FACT_KEYS = (
    "device_tree_model",
    "memory_total_kib",
    "mtd0_size_bytes",
    "mtd1_size_bytes",
    "sd_present",
    "uboot_attr_name",
    "uboot_attr_val",
    "uboot_compatible",
    "uboot_mode",
    "device_fw",
    "linux_version",
    "uboot_version",
)


def parse_passive_device_facts(output: str) -> dict[str, str]:
    """Parse only explicitly allowlisted hardware facts from Pluto output."""

    facts: dict[str, str] = {}
    allowed = set(PASSIVE_DEVICE_FACT_KEYS)
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key not in allowed:
            continue
        if key in facts:
            raise FirmwareError(f"duplicate passive device fact: {key}")
        if len(value.encode("utf-8")) > 512:
            raise FirmwareError(f"passive device fact is too long: {key}")
        if any(ord(character) < 0x20 and character != "\t" for character in value):
            raise FirmwareError(f"passive device fact has control characters: {key}")
        facts[key] = value.strip()
    missing = allowed - set(facts)
    if missing:
        raise FirmwareError(f"passive device facts are incomplete: {sorted(missing)}")
    return facts


def read_passive_device_facts(
    serial: str,
    *,
    ssh_config: Path,
    ssh_password: str,
    timeout: float = 15,
) -> dict[str, str]:
    """Read a fixed allowlist without enabling RX, TX, DDS, or changing RF state."""

    remote_command = r"""
printf 'device_tree_model='
tr -d '\000' </proc/device-tree/model 2>/dev/null || true
printf '\n'
awk '/^MemTotal:/ { printf "memory_total_kib=%s\n", $2; found=1 } END { if (!found) print "memory_total_kib=" }' /proc/meminfo
for item in mtd0 mtd1; do
    printf '%s_size_bytes=' "$item"
    test -r "/sys/class/mtd/$item/size" && tr -d '\n' <"/sys/class/mtd/$item/size"
    printf '\n'
done
if test -e /sys/block/mmcblk0; then printf 'sd_present=true\n'; else printf 'sd_present=false\n'; fi
for item in attr_name attr_val compatible mode; do
    printf 'uboot_%s=' "$item"
    fw_printenv -n "$item" 2>/dev/null || true
    printf '\n'
done
awk '$1 == "device-fw" { print "device_fw=" $2; found=1; exit } END { if (!found) print "device_fw=" }' /opt/VERSIONS
awk '$1 == "linux" { print "linux_version=" $2; found=1; exit } END { if (!found) print "linux_version=" }' /opt/VERSIONS
awk '$1 == "uboot" { print "uboot_version=" $2; found=1; exit } END { if (!found) print "uboot_version=" }' /opt/VERSIONS
"""
    manager = MultiPlutoFirmwareManager(
        image=Path("/dev/null"),
        image_sha256="0" * 64,
        ssh_config=ssh_config,
        ssh_password=ssh_password,
        state_root=Path("/run/spf/passive-fingerprint-unused"),
        expected_count=1,
    )
    # USB-IIO and the direct gadget can enumerate several seconds before
    # Dropbear accepts connections after an embedded reboot. Fingerprinting is
    # still bound to this serial, but must wait through that normal bounded
    # readiness gap rather than making boot success timing-dependent.
    manager._wait_for_ssh(serial, timeout=60)
    result = manager._ssh(serial, remote_command, timeout=timeout)
    return parse_passive_device_facts(result.stdout)


class MultiPlutoFirmwareManager:
    def __init__(
        self,
        *,
        image: Path,
        image_sha256: str,
        ssh_config: Path,
        ssh_password: str,
        state_root: Path,
        expected_count: int,
        approved_qspi_versions: tuple[str, ...] = DEFAULT_APPROVED_QSPI_DEVICE_FW,
    ):
        self.image = image
        self.image_sha256 = image_sha256.lower()
        self.ssh_config = ssh_config
        self.ssh_password = ssh_password
        self.state_root = state_root
        self.expected_count = expected_count
        self.approved_qspi_versions = approved_qspi_versions

    def _check_root(self) -> None:
        if os.geteuid() != 0:
            raise FirmwareError("multi-radio firmware operations must run as root")

    def _check_image(self) -> None:
        if not self.image.is_file():
            raise FirmwareError(f"firmware image is missing: {self.image}")
        digest = hashlib.sha256(self.image.read_bytes()).hexdigest()
        if digest != self.image_sha256:
            raise FirmwareError(
                f"firmware SHA-256 mismatch: expected {self.image_sha256}, got {digest}"
            )

    def _devices(self) -> list[UsbPluto]:
        devices = discover_runtime_plutos()
        if len(devices) != self.expected_count:
            raise FirmwareError(
                f"expected {self.expected_count} runtime Plutos; found {len(devices)}"
            )
        return devices

    def _device(self, serial: str) -> UsbPluto:
        matches = [
            device for device in discover_runtime_plutos() if device.serial == serial
        ]
        if len(matches) != 1:
            raise FirmwareError(
                f"expected runtime Pluto {serial}; found {len(matches)} matches"
            )
        return matches[0]

    def _ssh_command(self, remote_command: str) -> list[str]:
        return [
            "env",
            f"SSHPASS={self.ssh_password}",
            "sshpass",
            "-e",
            "ssh",
            "-F",
            str(self.ssh_config),
            "-o",
            "ConnectTimeout=5",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ServerAliveInterval=2",
            "-o",
            "ServerAliveCountMax=2",
            f"root@{PLUTO_HOST_ADDRESS}",
            remote_command,
        ]

    def _ssh(
        self,
        serial: str,
        remote_command: str,
        *,
        check: bool = True,
        timeout: float = 15,
    ) -> subprocess.CompletedProcess[str]:
        with IsolatedPlutoNetwork(serial) as network:
            serial_result = network.run(
                self._ssh_command(
                    "cat /sys/kernel/config/usb_gadget/"
                    "composite_gadget/strings/0x409/serialnumber"
                ),
                capture_output=True,
                timeout=timeout,
            )
            actual_serial = serial_result.stdout.strip()
            if actual_serial != serial:
                raise FirmwareError(
                    f"USB-network identity mismatch: expected {serial}, "
                    f"reached {actual_serial}"
                )
            return network.run(
                self._ssh_command(remote_command),
                check=check,
                capture_output=True,
                timeout=timeout,
            )

    def _wait_product(
        self,
        sysfs_name: str,
        product: str,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        product_path = Path("/sys/bus/usb/devices") / sysfs_name / "idProduct"
        while time.monotonic() < deadline:
            actual = _read_optional(product_path)
            if actual is not None and actual.lower() == product:
                return
            time.sleep(0.25)
        actual = _read_optional(product_path)
        actual_description = actual if actual is not None else "absent"
        raise FirmwareError(
            f"{sysfs_name}: expected USB product {product}, "
            f"found {actual_description}"
        )

    def _wait_absent(self, sysfs_name: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        product_path = Path("/sys/bus/usb/devices") / sysfs_name / "idProduct"
        while time.monotonic() < deadline:
            if not product_path.exists():
                return
            time.sleep(0.25)
        raise FirmwareError(f"{sysfs_name}: Pluto did not disconnect during reboot")

    def _wait_for_ssh(self, serial: str, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self._ssh(serial, "true")
                return
            except (FirmwareError, subprocess.SubprocessError, OSError) as error:
                last_error = error
                time.sleep(1)
        raise FirmwareError(f"{serial}: SSH did not return: {last_error}")

    def _back_up(self, serial: str) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        destination = self.state_root / f"{timestamp}-{serial}-before-ram-boot.txt"
        result = self._ssh(
            serial,
            'printf "%s\\n" "--- /opt/VERSIONS ---"; cat /opt/VERSIONS; '
            'printf "%s\\n" "--- fw_printenv ---"; fw_printenv',
        )
        destination.write_text(result.stdout)
        print(f"Saved pre-load state: {destination}", flush=True)

    def _read_persistent_state(self, serial: str) -> tuple[str, dict[str, str], str]:
        result = self._ssh(
            serial,
            'printf "%s\\n" "--- /opt/VERSIONS ---"; cat /opt/VERSIONS; '
            'printf "%s\\n" "--- fw_printenv ---"; fw_printenv',
        )
        versions, separator, environment_output = result.stdout.partition(
            "--- fw_printenv ---"
        )
        if not separator:
            raise FirmwareError(f"{serial}: persistent-state response is incomplete")
        version = parse_device_fw_version(versions)
        environment = parse_uboot_environment(environment_output)
        return version, environment, result.stdout

    def _read_uboot_environment(self, serial: str) -> dict[str, str]:
        result = self._ssh(serial, "fw_printenv")
        return parse_uboot_environment(result.stdout)

    def _require_approved_qspi_version(self, serial: str, version: str) -> None:
        if version not in self.approved_qspi_versions:
            raise FirmwareError(
                f"{serial}: QSPI device-fw {version!r} is not approved; "
                f"expected one of {list(self.approved_qspi_versions)}"
            )

    @staticmethod
    def _environment_mismatches(
        environment: dict[str, str],
    ) -> dict[str, dict[str, str | None]]:
        return {
            key: {"expected": expected, "actual": environment.get(key)}
            for key, expected in REQUIRED_UBOOT_ENV.items()
            if environment.get(key) != expected
        }

    def _back_up_provisioning_state(
        self, device: UsbPluto, persistent_state: str
    ) -> Path:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        destination_dir = self.state_root / device.serial / "provisioning"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / (
            f"{timestamp}-{device.sysfs_name}-before-2r2t.txt"
        )
        destination.write_text(persistent_state)
        print(f"Saved pre-provisioning state: {destination}", flush=True)
        return destination

    def _iio_has_serial(self, serial: str) -> bool:
        result = _run(["iio_info", "-s"], capture_output=True, check=False)
        return any(
            serial in line and "[usb:" in line for line in result.stdout.splitlines()
        )

    def _iio_uri_for_serial(self, serial: str) -> str:
        result = _run(["iio_info", "-s"], capture_output=True, check=False)
        matches = []
        for line in result.stdout.splitlines():
            if serial not in line or "[usb:" not in line:
                continue
            start = line.rfind("[usb:")
            end = line.find("]", start)
            if start >= 0 and end > start:
                matches.append(line[start + 1 : end])
        if len(matches) != 1:
            raise FirmwareError(f"{serial}: expected one USB-IIO URI; found {matches}")
        return matches[0]

    def _verify_dual_rx(self, serial: str) -> None:
        uri = self._iio_uri_for_serial(serial)
        result = _run(
            ["iio_info", "-u", uri],
            capture_output=True,
            check=False,
            timeout=15,
        )
        if result.returncode != 0:
            raise FirmwareError(f"{serial}: failed to inspect USB-IIO context {uri}")
        in_rx_dma = False
        scan_channels = set()
        for line in result.stdout.splitlines():
            if line.startswith("\tiio:device"):
                in_rx_dma = "cf-ad9361-lpc" in line
                continue
            if not in_rx_dma:
                continue
            stripped = line.strip()
            for channel in range(4):
                if stripped.startswith(f"voltage{channel}:") and "(input," in stripped:
                    scan_channels.add(channel)
        if scan_channels != {0, 1, 2, 3}:
            raise FirmwareError(
                f"{serial}: dual RX is unavailable; cf-ad9361-lpc scan "
                f"channels are {sorted(scan_channels)}, expected [0, 1, 2, 3]"
            )

    def _verify_device(self, serial: str) -> None:
        device = self._device(serial)
        if not device.direct_usb:
            raise FirmwareError(f"{serial}: vendor direct-USB interface 6 is absent")
        if not self._iio_has_serial(serial):
            raise FirmwareError(f"{serial}: standard USB-IIO context is absent")
        self._verify_dual_rx(serial)
        if _env_is_true("SPF_PLUTO_SKIP_SSH_VERIFY"):
            # ssh-free per-boot verify: interface-6 (vendor gadget), the USB-IIO
            # context, and dual-RX are all confirmed above by USB serial without
            # touching the shared 192.168.2.1. Skip the ssh daemon-pid check.
            print(
                f"{serial}: interface-6 + USB-IIO + dual-RX present "
                f"(ssh daemon check skipped)",
                flush=True,
            )
            return
        result = self._ssh(
            serial,
            "pidof iiod >/dev/null && pidof sdr_usb_gadget >/dev/null",
            check=False,
        )
        if result.returncode != 0:
            raise FirmwareError(
                f"{serial}: iiod and sdr_usb_gadget are not both running"
            )

    # Number of times to (re)try the full RAM-load of a single radio. A first
    # attempt that flaps during runtime re-enumeration (the observed
    # "expected N runtime Plutos; found N-1" boot failure) then self-heals on the
    # retry instead of failing the whole service and stranding the rover until a
    # power cycle. Bounded so the worst case stays inside the unit's
    # TimeoutStartSec.
    _LOAD_ATTEMPTS = 2

    def _current_product(self, sysfs_name: str) -> str | None:
        product = _read_optional(
            Path("/sys/bus/usb/devices") / sysfs_name / "idProduct"
        )
        return product.lower() if product is not None else None

    def _load_device(self, device: UsbPluto) -> None:
        """RAM-load one radio, retrying the whole sequence on a transient flap."""
        last_error: BaseException | None = None
        for attempt in range(1, self._LOAD_ATTEMPTS + 1):
            try:
                self._load_device_once(device)
                return
            except (FirmwareError, subprocess.SubprocessError, OSError) as error:
                last_error = error
                print(
                    f"{device.serial}: RAM-load attempt "
                    f"{attempt}/{self._LOAD_ATTEMPTS} failed: {error}",
                    flush=True,
                )
                if attempt < self._LOAD_ATTEMPTS:
                    self._settle_before_retry(device)
        raise FirmwareError(
            f"{device.serial}: RAM load failed after {self._LOAD_ATTEMPTS} "
            f"attempts: {last_error}"
        )

    def _settle_before_retry(self, device: UsbPluto) -> None:
        """Best-effort return to a re-loadable state before another attempt."""
        try:
            if self._current_product(device.sysfs_name) == PLUTO_DFU_PRODUCT:
                # Stuck in DFU: execute/detach so it re-enumerates to runtime.
                _run(
                    [
                        "dfu-util",
                        "-p",
                        device.sysfs_name,
                        "-d",
                        f"{PLUTO_VENDOR}:{PLUTO_RUNTIME_PRODUCT},"
                        f"{PLUTO_VENDOR}:{PLUTO_DFU_PRODUCT}",
                        "-a",
                        "firmware.dfu",
                        "-e",
                    ],
                    check=False,
                )
            time.sleep(2)
        except (subprocess.SubprocessError, OSError) as error:  # noqa: BLE001
            print(f"{device.serial}: settle-before-retry note: {error}", flush=True)

    def _load_device_once(self, device: UsbPluto) -> None:
        # Tolerate a radio already parked in DFU (e.g. a retry after a partial
        # first attempt): skip the runtime backup/reboot and load directly.
        if self._current_product(device.sysfs_name) != PLUTO_DFU_PRODUCT:
            self._back_up(device.serial)
            print(
                f"{device.serial}: requesting exact volatile RAM boot at "
                f"{device.sysfs_name}",
                flush=True,
            )
            try:
                self._ssh(
                    device.serial,
                    "/usr/sbin/device_reboot ram",
                    check=False,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                pass
            self._wait_product(device.sysfs_name, PLUTO_DFU_PRODUCT, 30)
        else:
            print(
                f"{device.serial}: already in DFU at {device.sysfs_name}; "
                f"loading image directly",
                flush=True,
            )

        print(f"{device.serial}: loading verified image into RAM", flush=True)
        common = [
            "dfu-util",
            "-p",
            device.sysfs_name,
            "-d",
            f"{PLUTO_VENDOR}:{PLUTO_RUNTIME_PRODUCT},"
            f"{PLUTO_VENDOR}:{PLUTO_DFU_PRODUCT}",
            "-a",
            "firmware.dfu",
        ]
        _run([*common, "-D", str(self.image)])
        _run([*common, "-e"])

        # Runtime re-enumeration is a full embedded-Linux boot; allow more time
        # than the old 60 s to reduce spurious "found N-1" failures.
        self._wait_product(device.sysfs_name, PLUTO_RUNTIME_PRODUCT, 90)
        self._wait_for_ssh(device.serial)
        self._verify_device(device.serial)
        print(f"{device.serial}: PASS", flush=True)

    def load_all(self) -> None:
        self._check_root()
        _require_commands(("dfu-util", "iio_info", "ip", "ssh", "sshpass", "udevadm"))
        self._check_image()
        devices = self._devices()
        dfu_devices = [
            path
            for path in Path("/sys/bus/usb/devices").iterdir()
            if (path / "idVendor").is_file()
            and (path / "idProduct").is_file()
            and _read(path / "idVendor").lower() == PLUTO_VENDOR
            and _read(path / "idProduct").lower() == PLUTO_DFU_PRODUCT
        ]
        if dfu_devices:
            raise FirmwareError(
                f"refusing to start with existing DFU devices: {dfu_devices}"
            )
        self._load_devices(devices)
        self.verify_all()

    def _load_devices(self, devices: list[UsbPluto]) -> None:
        """Load every radio's RAM image.

        The two dominant boot-time costs are the two sequential embedded-Linux
        reboots each radio performs. Each radio is fully isolated — DFU is
        targeted by physical USB path (``dfu-util -p <sysfs>``) and every SSH
        runs in its own network namespace — so the loads run concurrently by
        default, roughly halving the wall-clock on a 2-radio rover. Set
        ``SPF_PLUTO_SEQUENTIAL_LOAD=1`` to fall back to serial loading if
        simultaneous USB re-enumeration is ever seen to interfere.
        """
        if len(devices) < 2 or _env_is_true("SPF_PLUTO_SEQUENTIAL_LOAD"):
            for device in devices:
                self._load_device(device)
            return

        errors: dict[str, BaseException] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {
                pool.submit(self._load_device, device): device for device in devices
            }
            for future in concurrent.futures.as_completed(futures):
                device = futures[future]
                try:
                    future.result()
                except BaseException as error:  # noqa: BLE001
                    errors[device.serial] = error
        if errors:
            raise FirmwareError(
                "parallel RAM load failed: "
                + "; ".join(f"{serial}: {error}" for serial, error in errors.items())
            )

    def verify_all(self) -> None:
        self._check_root()
        devices = self._devices()
        for device in devices:
            self._verify_device(device.serial)
            print(
                f"{device.serial}: PASS direct_usb path={device.sysfs_name}",
                flush=True,
            )
        print(f"PASS: verified {len(devices)} direct-USB Pluto radios", flush=True)

    def check_config_all(self) -> None:
        self._check_root()
        devices = self._devices()
        for device in devices:
            # /opt/VERSIONS describes the active image, which may be a volatile
            # RAM image retained across a Pi-only reboot. It cannot prove the
            # version stored in QSPI, so ordinary RAM boot must not gate on it.
            # Persistent writes retain the QSPI allowlist in
            # provision_config_all(); this read-only boot gate checks only the
            # U-Boot values needed by the exact image loaded immediately after.
            environment = self._read_uboot_environment(device.serial)
            mismatches = self._environment_mismatches(environment)
            if mismatches:
                raise FirmwareError(
                    f"{device.serial}: Pluto U-Boot environment mismatch: "
                    f"{mismatches}"
                )
            print(
                f"{device.serial}: PASS persistent ad9361/2r2t configuration",
                flush=True,
            )
        print(f"PASS: verified configuration on {len(devices)} Plutos", flush=True)

    def provision_config_all(self, *, dry_run: bool = False) -> None:
        self._check_root()
        _require_commands(("iio_info", "ip", "ssh", "sshpass", "udevadm"))
        devices = self._devices()
        for original in devices:
            device = self._device(original.serial)
            if device.direct_usb:
                raise FirmwareError(
                    f"{device.serial}: refuse persistent provisioning while "
                    "the temporary direct-USB RAM image is running"
                )
            if (
                device.bus != original.bus
                or device.port_path != original.port_path
                or device.sysfs_name != original.sysfs_name
            ):
                raise FirmwareError(
                    f"{device.serial}: USB identity changed before provisioning"
                )

            version, environment, persistent_state = self._read_persistent_state(
                device.serial
            )
            self._require_approved_qspi_version(device.serial, version)
            mismatches = self._environment_mismatches(environment)
            if not mismatches:
                self._verify_dual_rx(device.serial)
                print(
                    f"{device.serial}: already provisioned "
                    f"qspi={version} path={device.sysfs_name}; no writes",
                    flush=True,
                )
                continue

            print(
                f"{device.serial}: {'DRY-RUN ' if dry_run else ''}"
                f"provision qspi={version} path={device.sysfs_name} "
                f"changes={mismatches}",
                flush=True,
            )
            if dry_run:
                continue

            self._back_up_provisioning_state(device, persistent_state)
            assignments = "; ".join(
                f"fw_setenv {shlex.quote(key)} {shlex.quote(value)}"
                for key, value in REQUIRED_UBOOT_ENV.items()
            )
            self._ssh(device.serial, f"set -eu; {assignments}; sync")
            try:
                self._ssh(
                    device.serial,
                    "/usr/sbin/device_reboot reset",
                    check=False,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                pass
            self._wait_absent(device.sysfs_name, 30)
            self._wait_product(device.sysfs_name, PLUTO_RUNTIME_PRODUCT, 90)
            self._wait_for_ssh(device.serial, 60)

            returned = self._device(device.serial)
            if (
                returned.bus != device.bus
                or returned.port_path != device.port_path
                or returned.sysfs_name != device.sysfs_name
            ):
                raise FirmwareError(
                    f"{device.serial}: returned on unexpected USB path "
                    f"{returned.sysfs_name}; expected {device.sysfs_name}"
                )
            returned_version, returned_environment, _ = self._read_persistent_state(
                device.serial
            )
            self._require_approved_qspi_version(device.serial, returned_version)
            returned_mismatches = self._environment_mismatches(returned_environment)
            if returned_mismatches:
                raise FirmwareError(
                    f"{device.serial}: U-Boot provisioning did not persist: "
                    f"{returned_mismatches}"
                )
            self._verify_dual_rx(device.serial)
            print(
                f"{device.serial}: PASS provisioned ad9361/2r2t "
                f"path={device.sysfs_name}",
                flush=True,
            )
        if dry_run:
            print(
                f"PASS: dry-run inspected {len(devices)} Plutos; no writes",
                flush=True,
            )
        else:
            print(
                f"PASS: provisioned and verified {len(devices)} Plutos",
                flush=True,
            )

    def rollback_all(self) -> None:
        self._check_root()
        devices = self._devices()
        for original in devices:
            device = self._device(original.serial)
            if not device.direct_usb:
                print(f"{device.serial}: already running QSPI firmware; skipping")
                continue
            print(f"{device.serial}: resetting to unchanged QSPI firmware", flush=True)
            try:
                self._ssh(
                    device.serial,
                    "/usr/sbin/device_reboot reset",
                    check=False,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                pass
            self._wait_product(device.sysfs_name, PLUTO_RUNTIME_PRODUCT, 90)
            self._wait_for_ssh(device.serial)
            returned = self._device(device.serial)
            if returned.direct_usb:
                raise FirmwareError(
                    f"{device.serial}: direct interface remains after reset"
                )
            print(f"{device.serial}: PASS QSPI rollback", flush=True)
        print("PASS: all Plutos are running their installed QSPI firmware", flush=True)

    def restart_all(self) -> None:
        """Restart every direct-USB Pluto and prove durable identity on return."""

        self._check_root()
        _require_commands(("iio_info", "ip", "ssh", "sshpass", "udevadm"))
        devices = self._devices()
        for original in devices:
            if not original.direct_usb:
                raise FirmwareError(
                    f"{original.serial}: refuse restart soak without the "
                    "direct-USB interface"
                )
            print(
                f"{original.serial}: restarting {original.sysfs_name} "
                f"path={original.port_path}",
                flush=True,
            )
            try:
                self._ssh(
                    original.serial,
                    "sync; /usr/sbin/device_reboot reset",
                    check=False,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                # A disappearing SSH transport is the expected successful
                # response to device_reboot. The USB absence/presence gates
                # below determine whether the restart actually happened.
                pass
            self._wait_absent(original.sysfs_name, 30)
            self._wait_product(original.sysfs_name, PLUTO_RUNTIME_PRODUCT, 90)
            self._wait_for_ssh(original.serial, 60)

            returned = self._device(original.serial)
            if (
                returned.bus != original.bus
                or returned.port_path != original.port_path
                or returned.sysfs_name != original.sysfs_name
            ):
                raise FirmwareError(
                    f"{original.serial}: returned on unexpected USB identity "
                    f"{returned.sysfs_name}/bus={returned.bus}/path={returned.port_path}; "
                    f"expected {original.sysfs_name}/bus={original.bus}/"
                    f"path={original.port_path}"
                )
            if not returned.direct_usb:
                raise FirmwareError(
                    f"{original.serial}: direct-USB interface is absent after restart"
                )
            self._verify_device(original.serial)
            print(
                f"{original.serial}: PASS restart address "
                f"{original.address}->{returned.address}",
                flush=True,
            )
        print(f"PASS: restarted and verified {len(devices)} Plutos", flush=True)

    def status_all(self) -> None:
        devices = discover_runtime_plutos()
        for device in devices:
            print(
                f"serial={device.serial} usb={device.sysfs_name} "
                f"bus={device.bus} port={device.port_path} "
                f"direct_usb={str(device.direct_usb).lower()}"
            )
        print(f"runtime_count={len(devices)} expected_count={self.expected_count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "discover-count",
            "load-all",
            "verify-all",
            "check-config-all",
            "provision-config-all",
            "rollback-all",
            "restart-all",
            "status-all",
        ),
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-sha256")
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--ssh-password", default="analog")
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--approved-qspi-version",
        action="append",
        dest="approved_qspi_versions",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "discover-count":
        print(len(discover_runtime_plutos()))
        return 0
    missing = [
        name
        for name in (
            "image",
            "image_sha256",
            "ssh_config",
            "state_root",
            "expected_count",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            f"{args.command} requires: {', '.join('--' + name.replace('_', '-') for name in missing)}"
        )
    manager = MultiPlutoFirmwareManager(
        image=args.image,
        image_sha256=args.image_sha256,
        ssh_config=args.ssh_config,
        ssh_password=args.ssh_password,
        state_root=args.state_root,
        expected_count=args.expected_count,
        approved_qspi_versions=tuple(
            args.approved_qspi_versions or DEFAULT_APPROVED_QSPI_DEVICE_FW
        ),
    )
    try:
        method = getattr(manager, args.command.replace("-", "_"))
        if args.command == "provision-config-all":
            method(dry_run=args.dry_run)
        else:
            method()
    except (FirmwareError, subprocess.SubprocessError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
