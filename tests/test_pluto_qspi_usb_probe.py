import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ENSURE_SCRIPT = (
    REPO_ROOT / "data_collection/rover/rover_v3.1/ensure_pluto_qspi.sh"
)
EXPECTED_FW = "v0.38_plutoplus_with_timestamping-9-g7b02"


def _add_runtime_pluto(root: Path, *, serial: str = "SERIAL_A") -> None:
    device = root / "1-1.1"
    device.mkdir(parents=True)
    (device / "idVendor").write_text("0456\n")
    (device / "idProduct").write_text("b673\n")
    (device / "serial").write_text(f"{serial}\n")
    (device / "busnum").write_text("1\n")
    (device / "devnum").write_text("3\n")


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
    path.chmod(0o755)


def _run_ensure(tmp_path: Path, *, iio_body: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    usb_root = tmp_path / "usb"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _add_runtime_pluto(usb_root)
    command_log = tmp_path / "commands.log"

    _write_fake_command(
        fake_bin / "iio_attr",
        'printf "iio_attr %s\\n" "$*" >>"$SPF_TEST_COMMAND_LOG"\n' + iio_body,
    )
    for command in ("mount", "umount", "eject"):
        _write_fake_command(
            fake_bin / command,
            f'printf "{command} %s\\n" "$*" >>"$SPF_TEST_COMMAND_LOG"\n'
            "exit 97\n",
        )

    firmware = tmp_path / "pluto.dfu"
    firmware.write_bytes(b"not needed on the matching fast path")
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "SPF_PLUTO_USB_ROOT": str(usb_root),
            "SPF_PLUTO_EXPECTED_DEVICE_FW": EXPECTED_FW,
            "SPF_FIRMWARE_DFU": str(firmware),
            "SPF_PLUTO_FRM": str(tmp_path / "pluto.frm"),
            "SPF_TEST_COMMAND_LOG": str(command_log),
        }
    )
    result = subprocess.run(
        ["bash", str(ENSURE_SCRIPT), "1"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = command_log.read_text().splitlines() if command_log.exists() else []
    return result, calls


def test_matching_active_firmware_uses_usb_iio_without_mounting(tmp_path):
    result, calls = _run_ensure(
        tmp_path,
        iio_body=f'printf "fw_version: {EXPECTED_FW}\\n"\n',
    )

    assert result.returncode == 0, result.stderr
    assert calls == ["iio_attr -T 2000 -u usb:1.3.5 -C fw_version"]
    assert "active firmware matches via USB-IIO; skip" in result.stdout
    assert "PASS: 1 Pluto(s) on expected active firmware" in result.stdout


def test_unreadable_usb_iio_fails_closed_without_mounting_or_flashing(tmp_path):
    result, calls = _run_ensure(tmp_path, iio_body="exit 1\n")

    assert result.returncode != 0
    assert calls == ["iio_attr -T 2000 -u usb:1.3.5 -C fw_version"]
    assert "active firmware unavailable over USB-IIO" in result.stderr
    assert "refusing to flash blind" in result.stderr


def test_qspi_script_documents_mass_storage_as_mismatch_only():
    source = ENSURE_SCRIPT.read_text()

    assert "Mass storage is opened only after an explicit USB-IIO mismatch" in source
    assert "iio_attr" in source
