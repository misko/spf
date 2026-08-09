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


def test_gain_series_candidate_campaign_requires_explicit_tx_and_mutes(tmp_path):
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
    _executable(shim / "iio_info", 'printf "synthetic IIO inventory\\n"\n')
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
        [
            str(SCRIPT),
            "--with-tx-loopback",
            "--loopback-attenuation-db=30",
            str(image),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = trace.read_text().splitlines()
    tx_test = [
        command
        for command in commands
        if "test_gain_series_v3_tx_loopback_hardware.py" in command
    ]
    assert len(tx_test) == 1
    assert "--radio-tx-loopback" in tx_test[0]
    assert "--radio-tx-loopback-attenuation-db=30" in tx_test[0]
    mute_commands = [
        command for command in commands if "spf.scripts.mute_pluto_tx" in command
    ]
    assert len(mute_commands) >= 3
    assert (report / "candidate-v3-tx2-loopback.log").is_file()


def test_gain_series_candidate_campaign_rejects_unacknowledged_tx(tmp_path):
    image = tmp_path / "candidate.dfu"
    image.write_bytes(b"synthetic candidate image")
    result = subprocess.run(
        [str(SCRIPT), "--with-tx-loopback", str(image)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires --loopback-attenuation-db" in result.stderr


def test_gain_series_candidate_campaign_accepts_direct_ip_serial_with_pipefail(
    tmp_path,
):
    shim = tmp_path / "bin"
    shim.mkdir()
    trace = tmp_path / "commands.log"
    image = tmp_path / "candidate.dfu"
    image.write_bytes(b"synthetic candidate image")
    report = tmp_path / "report"
    serial = "104000synthetic"

    fake_python = _executable(
        shim / "python",
        """if [[ "$*" == *"spf.scripts.resolve_pluto_ip"* ]]; then
    printf '%s\\n' 192.0.2.10
else
    printf '%s\\n' "$*" >> "$SPF_V3_TEST_TRACE"
fi
""",
    )
    _executable(
        shim / "sudo",
        '[[ "${1:-}" != "-n" ]] || shift\nexec "$@"\n',
    )
    _executable(shim / "iio_info", 'printf "synthetic IIO inventory\\n"\n')
    _executable(shim / "iio_attr", f'printf "hw_serial: {serial}\\n"\n')
    # Enough output after the early serial match makes grep -q close the pipe
    # and this producer exit 141.  The campaign must consume all lsusb output.
    _executable(
        shim / "lsusb",
        f"""printf '%s\\n' {serial}
for _index in $(seq 1 20000); do
    printf '%s\\n' synthetic-descriptor-line
done
""",
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
        [str(SCRIPT), str(image), "192.0.2.10"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "candidate-v3-direct-ip" in result.stdout
    assert any(
        "test_v3_direct_ip_uses_the_same_inner_frame" in command
        for command in trace.read_text().splitlines()
    )
