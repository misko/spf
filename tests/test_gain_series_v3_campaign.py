import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/radio_hardware/run_gain_series_v3_candidate.sh"


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)
    return path


def test_gain_series_candidate_campaign_orders_volatile_gates(tmp_path):
    shim = tmp_path / "bin"
    shim.mkdir()
    trace = tmp_path / "commands.log"
    image = tmp_path / "candidate.dfu"
    image.write_bytes(b"synthetic candidate image")
    report = tmp_path / "report"

    fake_python = _executable(
        shim / "python",
        'printf "%s\\n" "$*" >> "$SPF_V3_TEST_TRACE"\n',
    )
    _executable(
        shim / "sudo",
        '[[ "${1:-}" != "-n" ]] || shift\nexec "$@"\n',
    )
    _executable(
        shim / "iio_info",
        'printf "%s\\n" "synthetic IIO inventory"\n',
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{shim}:{environment['PATH']}",
            "SPF_V3_PYTHON": str(fake_python),
            "SPF_V3_EXPECTED_RADIOS": "2",
            "SPF_V3_PRODUCTION_RECORDS": "1",
            "SPF_V3_REPORT_ROOT": str(report),
            "SPF_V3_TEST_TRACE": str(trace),
        }
    )
    result = subprocess.run(
        [str(SCRIPT), str(image)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "QSPI was not modified" in result.stdout
    commands = trace.read_text().splitlines()
    expected_in_order = [
        "test_direct_usb_hardware.py",
        "check-config-all",
        "load-all",
        "test_direct_usb_hardware.py",
        "test_v3_usb_gain_observations",
        "test_v3_gain_series_round_trips_through_v7_zarr",
        "status-all",
    ]
    cursor = 0
    for expected in expected_in_order:
        while cursor < len(commands) and expected not in commands[cursor]:
            cursor += 1
        assert cursor < len(commands), (expected, commands)
        cursor += 1
    assert all("provision-config-all" not in command for command in commands)
    assert all("ensure_pluto_qspi" not in command for command in commands)
    assert (report / "baseline-v2.log").is_file()
    assert (report / "candidate-v3-production-zarr.log").is_file()
