"""Safely RAM-load direct-USB firmware on multiple attached Pluto radios.

Every Pluto exposes the same USB-network address (192.168.2.1). To address a
specific radio without unplugging its neighbours, this module temporarily
moves that radio's USB-network interface into a private network namespace.
USB-IIO and DFU continue to use the original physical USB path.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
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


class FirmwareError(RuntimeError):
    """A firmware operation failed its safety contract."""


@dataclasses.dataclass(frozen=True)
class UsbPluto:
    serial: str
    sysfs_name: str
    bus: int
    port_path: str
    direct_usb: bool


@dataclasses.dataclass(frozen=True)
class InterfaceState:
    name: str
    address: str
    prefixlen: int
    route_metric: int | None


def _read(path: Path) -> str:
    return path.read_text().strip()


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
    if len(ipv4) != 1:
        raise FirmwareError(
            f"{interface}: expected one IPv4 address before isolation; found {ipv4}"
        )
    route_result = _run(
        ["ip", "-j", "route", "show", "dev", interface],
        capture_output=True,
    )
    routes = json.loads(route_result.stdout)
    metric = routes[0].get("metric") if routes else None
    return InterfaceState(
        name=interface,
        address=ipv4[0]["local"],
        prefixlen=int(ipv4[0]["prefixlen"]),
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
            if self.state.route_metric is not None:
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
    ):
        self.image = image
        self.image_sha256 = image_sha256.lower()
        self.ssh_config = ssh_config
        self.ssh_password = ssh_password
        self.state_root = state_root
        self.expected_count = expected_count

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
            if product_path.is_file() and _read(product_path).lower() == product:
                return
            time.sleep(0.25)
        actual = _read(product_path) if product_path.is_file() else "absent"
        raise FirmwareError(
            f"{sysfs_name}: expected USB product {product}, found {actual}"
        )

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

    def _iio_has_serial(self, serial: str) -> bool:
        result = _run(["iio_info", "-s"], capture_output=True, check=False)
        return any(
            serial in line and "[usb:" in line for line in result.stdout.splitlines()
        )

    def _verify_device(self, serial: str) -> None:
        device = self._device(serial)
        if not device.direct_usb:
            raise FirmwareError(f"{serial}: vendor direct-USB interface 6 is absent")
        if not self._iio_has_serial(serial):
            raise FirmwareError(f"{serial}: standard USB-IIO context is absent")
        result = self._ssh(
            serial,
            "pidof iiod >/dev/null && pidof sdr_usb_gadget >/dev/null",
            check=False,
        )
        if result.returncode != 0:
            raise FirmwareError(
                f"{serial}: iiod and sdr_usb_gadget are not both running"
            )

    def _load_device(self, device: UsbPluto) -> None:
        if device.direct_usb:
            print(f"{device.serial}: direct firmware already present; verifying")
            self._verify_device(device.serial)
            return

        self._back_up(device.serial)
        print(
            f"{device.serial}: requesting volatile RAM boot at {device.sysfs_name}",
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

        self._wait_product(device.sysfs_name, PLUTO_RUNTIME_PRODUCT, 60)
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
        for device in devices:
            self._load_device(device)
        self.verify_all()

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
        choices=("load-all", "verify-all", "rollback-all", "status-all"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--ssh-password", default="analog")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manager = MultiPlutoFirmwareManager(
        image=args.image,
        image_sha256=args.image_sha256,
        ssh_config=args.ssh_config,
        ssh_password=args.ssh_password,
        state_root=args.state_root,
        expected_count=args.expected_count,
    )
    try:
        getattr(manager, args.command.replace("-", "_"))()
    except (FirmwareError, subprocess.SubprocessError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
