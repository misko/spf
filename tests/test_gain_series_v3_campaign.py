import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tests.radio_hardware.test_gain_series_v3_tx_loopback_hardware import (
    _single_channel_tone_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/radio_hardware/run_gain_series_v3_candidate.sh"


def test_internal_loopback_tone_metrics_accept_one_active_rx_channel():
    sample_rate_hz = 3_000_000
    tone_hz = 100_000
    samples = 30_000
    sample_index = np.arange(samples, dtype=np.float64)
    signal = np.zeros((2, samples), dtype=np.complex64)
    signal[1] = (
        512 * np.exp(2j * np.pi * tone_hz * sample_index / sample_rate_hz)
    ).astype(np.complex64)

    metrics = _single_channel_tone_metrics(
        signal,
        sample_rate_hz=sample_rate_hz,
        tone_hz=tone_hz,
        transient_samples=1_024,
    )

    assert metrics["strongest_channel"] == 1
    assert abs(metrics["frequency_error_hz"]) < 1e-3
    assert metrics["tone_dbfs"][1] == pytest.approx(-12.0412, abs=1e-3)
    assert metrics["tone_snr_db"][1] > 70.0


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
    assert len(tx_test) == 3
    assert all("--radio-tx-loopback" in command for command in tx_test)
    assert all(
        "--radio-tx-loopback-attenuation-db=30" in command for command in tx_test
    )
    load_commands = [command for command in commands if "load-all" in command]
    assert len(load_commands) == 3
    first_zarr = next(
        index
        for index, command in enumerate(commands)
        if "test_v3_gain_series_round_trips_through_v7_zarr" in command
    )
    assert all(commands.index(command) < first_zarr for command in tx_test)
    mute_commands = [
        command for command in commands if "spf.scripts.mute_pluto_tx" in command
    ]
    assert len(mute_commands) >= 5
    for epoch in range(1, 4):
        assert (report / f"candidate-v3-tx2-loopback-epoch-{epoch}.log").is_file()


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
