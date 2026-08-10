from types import SimpleNamespace

import pytest

import spf.scripts.direct_ip_parallel_ladder as ladder


def test_parse_sample_rate_ladder_accepts_compact_units():
    assert ladder.parse_sample_rate_ladder("3M, 6m, 12500k, 15_000_000") == (
        3_000_000,
        6_000_000,
        12_500_000,
        15_000_000,
    )


@pytest.mark.parametrize("value", ["", "3M,,6M", "6M,3M", "3M,3M", "0"])
def test_parse_sample_rate_ladder_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):
        ladder.parse_sample_rate_ladder(value)


class _FakeReceiver:
    def __init__(self, host, rates):
        self.host = host
        self.rates = rates
        self.capabilities = SimpleNamespace(flags=ladder.REQUIRED_TRANSPORT_FLAGS)
        self.effective_data_receive_buffer_bytes = 128 * ladder.MIB

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _fake_dependencies(monkeypatch, *, fail_at=None, rearm_fail_at=None):
    rates = {"radio-a": 1_000_000, "radio-b": 2_000_000}

    def get_rate(host):
        return rates[host]

    def set_rate(host, rate):
        rates[host] = rate
        return rate

    def factory(host, _settings):
        return _FakeReceiver(host, rates)

    def identity(host):
        return {"serial": f"serial-{host}", "firmware": "test", "model": "test"}

    attempts = {}

    def capture(receiver, settings, _sample_rate_hz):
        rate = receiver.rates[receiver.host]
        key = (receiver.host, rate)
        attempts[key] = attempts.get(key, 0) + 1
        if rearm_fail_at == rate and attempts[key] > 1:
            raise RuntimeError("START_RX was not acknowledged after 3 attempts")
        if fail_at is not None and rate >= fail_at and receiver.host == "radio-b":
            raise RuntimeError("synthetic loss")
        payload = settings.samples_per_channel * 8 * settings.frames_per_request
        # 30 MiB/s at 3 MS/s has headroom; 20 MiB/s at 6 MS/s does not.
        payload_mibps = 30.0 if rate <= 3_000_000 else 20.0
        return {
            "elapsed_seconds": payload / ladder.MIB / payload_mibps,
            "payload_bytes": payload,
            "payload_mibps": payload_mibps,
            "estimated_drain_mibps": payload_mibps,
        }

    monkeypatch.setattr(ladder, "_capture_one", capture)
    monkeypatch.setattr(
        ladder,
        "read_udp_counters",
        lambda: {"InDatagrams": 100, "RcvbufErrors": 0},
    )
    monkeypatch.setattr(
        ladder,
        "read_network_counters",
        lambda _interface: {"rx_bytes": 100, "rx_packets": 10, "rx_dropped": 0},
    )
    return rates, get_rate, set_rate, identity, factory


def test_parallel_ladder_records_realtime_shortfall_and_restores(monkeypatch):
    rates, get_rate, set_rate, identity, factory = _fake_dependencies(monkeypatch)
    progress = []
    settings = ladder.ParallelIpLadderSettings(
        hosts=("radio-a", "radio-b"),
        sample_rates_hz=(3_000_000, 6_000_000),
        samples_per_channel=1_024,
        frames_per_request=2,
        cycles_per_rate=2,
    )
    report = ladder.run_parallel_ip_ladder(
        settings,
        get_sample_rate=get_rate,
        set_sample_rate=set_rate,
        get_identity=identity,
        receiver_factory=factory,
        progress_callback=lambda report: progress.append(
            (len(report["rungs"]), report.get("integrity_pass"))
        ),
    )
    assert report["integrity_pass"] is True
    assert report["first_integrity_failure_hz"] is None
    assert report["first_realtime_shortfall_hz"] == 6_000_000
    assert report["udp_datagrams_per_frame"] > 0
    assert [rung["status"] for rung in report["rungs"]] == ["pass", "pass"]
    assert rates == {"radio-a": 1_000_000, "radio-b": 2_000_000}
    assert progress[0] == (0, None)
    assert progress[-1] == (2, True)


def test_parallel_ladder_stops_on_integrity_failure_and_restores(monkeypatch):
    rates, get_rate, set_rate, identity, factory = _fake_dependencies(
        monkeypatch, fail_at=6_000_000
    )
    settings = ladder.ParallelIpLadderSettings(
        hosts=("radio-a", "radio-b"),
        sample_rates_hz=(3_000_000, 6_000_000, 10_000_000),
        samples_per_channel=1_024,
        frames_per_request=2,
        cycles_per_rate=1,
    )
    report = ladder.run_parallel_ip_ladder(
        settings,
        get_sample_rate=get_rate,
        set_sample_rate=set_rate,
        get_identity=identity,
        receiver_factory=factory,
    )
    assert report["integrity_pass"] is False
    assert report["first_integrity_failure_hz"] == 6_000_000
    assert report["completed_rungs"] == 2
    assert report["rungs"][-1]["status"] == "integrity_failure"
    assert "synthetic loss" in report["rungs"][-1]["error"]
    assert rates == {"radio-a": 1_000_000, "radio-b": 2_000_000}


def test_parallel_ladder_classifies_rearm_and_continues(monkeypatch):
    rates, get_rate, set_rate, identity, factory = _fake_dependencies(
        monkeypatch, rearm_fail_at=3_000_000
    )
    settings = ladder.ParallelIpLadderSettings(
        hosts=("radio-a", "radio-b"),
        sample_rates_hz=(3_000_000, 6_000_000),
        samples_per_channel=1_024,
        frames_per_request=2,
        cycles_per_rate=2,
        stop_after_integrity_failure=False,
    )
    report = ladder.run_parallel_ip_ladder(
        settings,
        get_sample_rate=get_rate,
        set_sample_rate=set_rate,
        get_identity=identity,
        receiver_factory=factory,
    )
    assert report["integrity_pass"] is True
    assert report["control_lifecycle_pass"] is False
    assert report["first_integrity_failure_hz"] is None
    assert report["first_control_rearm_failure_hz"] == 3_000_000
    assert [rung["status"] for rung in report["rungs"]] == [
        "control_rearm_failure",
        "pass",
    ]
    assert report["rungs"][0]["minimum_payload_mibps_by_host"] == {
        "radio-a": 30.0,
        "radio-b": 30.0,
    }
    assert rates == {"radio-a": 1_000_000, "radio-b": 2_000_000}
