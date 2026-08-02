import os
import subprocess
from pathlib import Path


LOADER = (
    Path(__file__).resolve().parents[1]
    / "data_collection"
    / "rover"
    / "rover_v3.1"
    / "load_direct_usb_firmware.sh"
)


def _run_usb_count(tmp_path: Path, lsusb_body: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    lsusb = fake_bin / "lsusb"
    lsusb.write_text("#!/bin/sh\n" + lsusb_body)
    lsusb.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; usb_count 0456:b674',
            "bash",
            str(LOADER),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_usb_count_treats_lsusb_no_match_as_zero(tmp_path):
    result = _run_usb_count(tmp_path, "exit 1\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


def test_usb_count_counts_matching_devices(tmp_path):
    result = _run_usb_count(
        tmp_path,
        "printf '%s\\n' 'device one' 'device two'\nexit 0\n",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"
