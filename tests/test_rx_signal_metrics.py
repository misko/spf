"""`rx_signal_metrics` — receive-path health, referred to the antenna.

The defect this pins: comparing raw |IQ| while AGC is running compares the
thing the AGC exists to equalise. Rover 1 on 2026-08-05 read 0.8 dB and 1.1 dB
apart and was reported PASS, while r1 ch0 was drawing 11.9 dB more gain to
reach that — 13.0 dB down at the antenna. Rover 4's r0 was reported OK at
6.0 dB when it was 18.5 dB down and its antenna was physically faulty.

Referring to the input (output dBFS minus gain) undoes the compensation.
"""

import numpy as np
import pytest

from spf.scripts import rx_signal_metrics as rsm


def channel(mag, gain, railed=0.0):
    out = 20 * np.log10(max(mag, 1e-12) / rsm.FULL_SCALE)
    return {
        "channel": 0,
        "mean_abs_iq": mag,
        "dbfs": out,
        "input_dbfs": out - gain,
        "mean_gain": gain,
        "railed_fraction": railed,
        "mean_rssi": 0.0,
        "frames_total": 100,
        "frames_sampled": 10,
        "zero_frame_fraction": 0.0,
    }


def receiver(ch0, ch1):
    a, b = channel(*ch0), channel(*ch1)
    a["channel"], b["channel"] = 0, 1
    return [a, b]


def analyse_fake(monkeypatch, receivers):
    monkeypatch.setattr(
        rsm, "zarr_open_from_lmdb_store", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        rsm, "_channel_stats", lambda group, frames: group
    )

    class Store(dict):
        pass

    store = Store({"receivers": receivers})
    monkeypatch.setattr(rsm, "zarr_open_from_lmdb_store", lambda *_a, **_k: store)
    return rsm.analyse("/fake/path.zarr", 10)


def test_agc_compensation_does_not_hide_a_weak_antenna(monkeypatch):
    """Rover 1, 2026-08-05: raw magnitudes 1.1 dB apart, 13 dB apart at the antenna."""
    report = analyse_fake(
        monkeypatch, {"r1": receiver((406.5, 51.2), (463.6, 39.3))}
    )
    finding = report["findings"][0]

    assert finding["imbalance_db"] == pytest.approx(13.0, abs=0.2)
    assert finding["output_imbalance_db"] == pytest.approx(1.1, abs=0.2)
    assert finding["gain_delta_db"] == pytest.approx(11.9, abs=0.2)
    assert finding["verdict"] == "IMBALANCED"


def test_the_old_metric_would_have_passed_that(monkeypatch):
    """Guard against regressing to raw-magnitude comparison."""
    report = analyse_fake(
        monkeypatch, {"r1": receiver((406.5, 51.2), (463.6, 39.3))}
    )
    finding = report["findings"][0]
    assert finding["output_imbalance_db"] < rsm.IMBALANCE_WARN_DB
    assert finding["imbalance_db"] >= rsm.IMBALANCE_WARN_DB


def test_rover4_both_channels_flagged(monkeypatch):
    """Physical inspection found BOTH ch0 antennas faulty; both must be flagged."""
    report = analyse_fake(
        monkeypatch,
        {
            "r0": receiver((290.1, 59.4, 0.68), (576.9, 46.8, 0.02)),
            "r1": receiver((46.9, 61.9, 0.99), (733.3, 51.1, 0.17)),
        },
    )
    by_rx = {f["receiver"]: f for f in report["findings"]}

    assert by_rx["r0"]["imbalance_db"] == pytest.approx(18.5, abs=0.3)
    assert by_rx["r0"]["verdict"] == "IMBALANCED"   # raw output said 6.0 dB -> OK
    assert by_rx["r1"]["imbalance_db"] == pytest.approx(34.7, abs=0.3)
    assert by_rx["r1"]["verdict"] == "DEAD"


def test_balanced_channels_still_pass(monkeypatch):
    report = analyse_fake(
        monkeypatch, {"r0": receiver((440.0, 40.0), (442.0, 39.8))}
    )
    assert report["findings"][0]["verdict"] == "OK"


def test_cross_receiver_comparison_is_input_referred(monkeypatch):
    """Radio-vs-antenna localisation must not be fooled by AGC either."""
    report = analyse_fake(
        monkeypatch,
        {
            "r0": receiver((290.1, 59.4), (576.9, 46.8)),
            "r1": receiver((46.9, 61.9), (733.3, 51.1)),
        },
    )
    cross = report["cross_receiver_db"]
    assert cross["ch0"] == pytest.approx(-18.3, abs=0.4)
    assert cross["ch1"] == pytest.approx(-2.2, abs=0.4)


def test_missing_gain_falls_back_to_raw_magnitude(monkeypatch):
    """A store without usable gains must still produce a verdict, not crash."""
    a, b = channel(100.0, float("nan")), channel(400.0, float("nan"))
    a["channel"], b["channel"] = 0, 1
    report = analyse_fake(monkeypatch, {"r0": [a, b]})
    assert report["findings"][0]["imbalance_db"] == pytest.approx(12.0, abs=0.3)
