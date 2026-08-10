"""Opt-in parallel direct-IP sample-rate ladder for exactly two radios."""

from __future__ import annotations

import json

import pytest

from spf.scripts.direct_ip_parallel_ladder import (
    MIB,
    ParallelIpLadderSettings,
    parse_sample_rate_ladder,
    run_parallel_ip_ladder,
)


pytestmark = [
    pytest.mark.radio_hardware,
    pytest.mark.radio_gain_series_v3,
    pytest.mark.radio_direct_ip_ladder,
]


def test_parallel_direct_ip_sample_rate_ladder(pytestconfig, radio_report_dir):
    hosts = tuple(pytestconfig.getoption("--radio-direct-ip-ladder-host"))
    if len(hosts) != 2 or len(set(hosts)) != 2:
        pytest.fail(
            "parallel direct-IP ladder requires exactly two unique "
            "--radio-direct-ip-ladder-host values"
        )
    rates = parse_sample_rate_ladder(
        pytestconfig.getoption("--radio-direct-ip-ladder-rates")
    )
    cycles = pytestconfig.getoption("--radio-direct-ip-ladder-cycles")
    required_rate = int(
        pytestconfig.getoption("--radio-direct-ip-ladder-required-rate")
    )
    settings = ParallelIpLadderSettings(
        hosts=hosts,
        sample_rates_hz=rates,
        samples_per_channel=pytestconfig.getoption("--radio-samples"),
        frames_per_request=pytestconfig.getoption("--radio-frames-per-request"),
        cycles_per_rate=cycles,
        gain_observation_interval_samples=pytestconfig.getoption(
            "--radio-gain-observation-interval"
        ),
        gain_observation_capacity=pytestconfig.getoption(
            "--radio-gain-observation-capacity"
        ),
        minimum_effective_receive_buffer_bytes=int(
            pytestconfig.getoption("--radio-direct-ip-min-receive-buffer-mib") * MIB
        ),
        network_interface=pytestconfig.getoption("--radio-direct-ip-ladder-interface"),
        stop_after_integrity_failure=not pytestconfig.getoption(
            "--radio-direct-ip-ladder-continue-after-failure"
        ),
    )
    report_path = radio_report_dir / "direct_ip_parallel_sample_rate_ladder.json"

    def write_progress(report):
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    report = run_parallel_ip_ladder(settings, progress_callback=write_progress)

    assert not report["restore_errors"], report["restore_errors"]
    failed_integrity = [
        rung
        for rung in report["rungs"]
        if rung["sample_rate_hz"] <= required_rate
        and rung["status"] == "integrity_failure"
    ]
    assert not failed_integrity, (
        f"direct-IP frame integrity failed at or below required rate "
        f"{required_rate}: {failed_integrity}; report={report_path}"
    )
    failed_control = [
        rung
        for rung in report["rungs"]
        if rung["sample_rate_hz"] <= required_rate
        and rung["status"] == "control_rearm_failure"
    ]
    assert not failed_control, (
        f"direct-IP firmware failed repeated START/STOP at or below required rate "
        f"{required_rate}: {failed_control}; report={report_path}"
    )
