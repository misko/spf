"""`rover radio signal` must measure the emitter, not the gaps between bursts.

The O4 is a bursty digital video transmitter -- `data_quality_plan.md` records
60-70% NaN as its healthy baseline. Two consequences the tool got wrong:

1. It converts a mean |IQ| over SAMPLED frames to dBFS and subtracts a mean gain
   over ALL frames. When the AGC rails during the gaps and drops during bursts,
   that single mean gain belongs to neither, and the "AT ANTENNA" figure is
   wrong by tens of dB. This is what produced the 2026-08-05 reading of ch0 at
   -83.2 dB "railed 90%", and three withdrawn conclusions built on it.

2. `railed_fraction` was `(gain == max(gain)).mean()` -- the fraction of frames
   at the maximum OBSERVED gain, not at the hardware ceiling. A channel whose
   gain never moves reports 100% regardless of its value. On 2026-08-06 it
   flagged rover 3 as "railed 100%" at a median gain of 29 dB.

   The detection was still worth having: a frozen gain means the channel has
   nothing varying to track, and that found a real antenna fault. So it is kept,
   under a name that says what it measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from spf.scripts.rx_signal_metrics import FULL_SCALE, _channel_stats


def _frames_from_dbfs(output_dbfs: np.ndarray, samples: int = 64) -> np.ndarray:
    """(T, 2, samples) complex whose per-channel mean|IQ| matches output_dbfs.

    Zero-mean, because real IQ is: a constant frame is pure DC, which the tool
    now (correctly) removes before measuring.
    """
    magnitude = FULL_SCALE * np.power(10.0, output_dbfs / 20.0)
    sign = np.where(np.arange(samples) % 2 == 0, 1.0, -1.0)
    return (magnitude[:, :, None] * sign[None, None, :]).astype(np.complex64)


class _FakeGroup(dict):
    """Minimal stand-in for a zarr receiver group."""


def _bursty_receiver(
    *,
    burst_fraction=0.3,
    total=100,
    burst_input_dbfs=(-50.0, -44.0),
    quiet_input_dbfs=-112.0,
    burst_gain=(36.0, 30.0),
    quiet_gain=(62.0, 62.0),
):
    """A receiver seeing a bursty emitter, with an AGC that rails in the gaps.

    Input-referred truth during a burst is `burst_input_dbfs`; the imbalance
    between the two channels is therefore exactly its difference. Between
    bursts both channels see the same receiver noise floor and the AGC rails.
    """
    rng = np.random.default_rng(0)
    loud = rng.random(total) < burst_fraction
    loud[0] = True  # guarantee at least one of each
    loud[1] = False

    gains = np.empty((total, 2))
    output = np.empty((total, 2))
    for channel in range(2):
        gains[loud, channel] = burst_gain[channel]
        gains[~loud, channel] = quiet_gain[channel]
        output[loud, channel] = burst_input_dbfs[channel] + burst_gain[channel]
        output[~loud, channel] = quiet_input_dbfs + quiet_gain[channel]

    return _FakeGroup(
        signal_matrix=_frames_from_dbfs(output),
        gains=gains,
        rssis=np.full((total, 2), 70.0),
    ), loud


def test_at_antenna_measures_the_burst_not_a_mean_over_the_gaps():
    """RED: mean-gain-over-all-frames is wrong by ~30 dB on a bursty emitter.

    Truth: ch0 -50 dBFS, ch1 -44 dBFS at the antenna during a burst -- a 6 dB
    imbalance. The old computation reports both ~30 dB lower, because it
    subtracts a gain averaged over frames the AGC had railed.
    """
    group, _loud = _bursty_receiver()

    stats = _channel_stats(group, frames=100)

    assert stats[0]["input_dbfs"] == pytest.approx(-50.0, abs=1.5), (
        "ch0 at-antenna should reflect the burst, not the silence between bursts"
    )
    assert stats[1]["input_dbfs"] == pytest.approx(-44.0, abs=1.5)


def test_the_imbalance_between_channels_survives_a_bursty_emitter():
    """The number the operator acts on is the difference; it must be right."""
    group, _loud = _bursty_receiver()

    stats = _channel_stats(group, frames=100)
    imbalance = stats[1]["input_dbfs"] - stats[0]["input_dbfs"]

    assert imbalance == pytest.approx(6.0, abs=1.0), (
        f"true imbalance is 6.0 dB, tool reported {imbalance:.1f} dB"
    )


def test_a_continuous_emitter_is_unaffected_by_burst_gating():
    """The fix must not change the answer where the old method was already right."""
    total = 100
    gains = np.tile(np.array([30.0, 30.0]), (total, 1))
    output = np.tile(np.array([-14.0, -20.0]), (total, 1))
    group = _FakeGroup(
        signal_matrix=_frames_from_dbfs(output),
        gains=gains,
        rssis=np.full((total, 2), 70.0),
    )

    stats = _channel_stats(group, frames=100)

    assert stats[0]["input_dbfs"] == pytest.approx(-44.0, abs=0.5)
    assert stats[1]["input_dbfs"] == pytest.approx(-50.0, abs=0.5)


# ------------------------------------------------------- railed vs frozen ----


def _fixed_gain_receiver(gain_value: float, total: int = 100):
    gains = np.full((total, 2), gain_value)
    output = np.full((total, 2), -14.0)
    return _FakeGroup(
        signal_matrix=_frames_from_dbfs(output),
        gains=gains,
        rssis=np.full((total, 2), 70.0),
    )


def test_a_constant_mid_scale_gain_is_not_railed():
    """RED: rover 3's B1 sat at a constant 29 dB and was reported 'railed 100%'."""
    stats = _channel_stats(_fixed_gain_receiver(29.0), frames=100)

    assert stats[0]["railed_fraction"] == pytest.approx(0.0), (
        "29 dB is nowhere near the AD9361 ceiling; it is not railed"
    )


def test_a_constant_gain_is_reported_as_frozen():
    """The detection was right even though the name was wrong -- keep it."""
    stats = _channel_stats(_fixed_gain_receiver(29.0), frames=100)

    assert stats[0]["gain_is_frozen"] is True
    assert stats[0]["distinct_gains"] == 1


def test_a_gain_actually_at_the_ceiling_is_railed():
    stats = _channel_stats(_fixed_gain_receiver(62.0), frames=100)

    assert stats[0]["railed_fraction"] == pytest.approx(1.0)
    assert stats[0]["gain_is_frozen"] is True


def test_a_healthy_varying_gain_is_neither_railed_nor_frozen():
    total = 100
    gains = np.stack(
        [np.linspace(26.0, 34.0, total), np.linspace(28.0, 36.0, total)], axis=1
    )
    group = _FakeGroup(
        signal_matrix=_frames_from_dbfs(np.full((total, 2), -14.0)),
        gains=gains,
        rssis=np.full((total, 2), 70.0),
    )

    stats = _channel_stats(group, frames=100)

    for channel in stats:
        assert channel["railed_fraction"] == pytest.approx(0.0)
        assert channel["gain_is_frozen"] is False


# ------------------------------------------------------------ DC exclusion ---
#
# ch1 on every rover carries an LO-leakage DC spur at -8 to -15 dBFS, 29-43 dB
# over its own noise floor, while ch0 carries none. mean|IQ| over the raw frame
# is therefore largely measuring that spur on one channel and not the other --
# an internal artifact being compared against antenna signal.


def _dc_offset_receiver(total: int = 60):
    """ch0: real signal, no DC. ch1: no signal, large DC. Equal raw magnitude."""
    samples = 64
    frames = np.zeros((total, 2, samples), dtype=np.complex64)
    # ch0 -- zero-mean signal, |IQ| = 400
    sign = np.where(np.arange(samples) % 2 == 0, 1.0, -1.0)
    frames[:, 0, :] = (400.0 * sign).astype(np.complex64)
    # ch1 -- pure DC at the same magnitude, carrying no information
    frames[:, 1, :] = np.complex64(400.0)
    return _FakeGroup(
        signal_matrix=frames,
        gains=np.full((total, 2), 30.0),
        rssis=np.full((total, 2), 70.0),
    )


def test_a_dc_spur_is_not_counted_as_antenna_signal():
    """RED: raw mean|IQ| cannot tell a DC offset from a received signal."""
    stats = _channel_stats(_dc_offset_receiver(), frames=60)

    assert stats[0]["input_dbfs"] > stats[1]["input_dbfs"] + 20.0, (
        "ch1 is pure DC and carries no signal, yet it measured as strong as "
        f"ch0: ch0={stats[0]['input_dbfs']:.1f} ch1={stats[1]['input_dbfs']:.1f}"
    )


# --------------------------------------------------- level vs actually tracking ---
#
# Real RO1 data, 2026-08-05: ch0 sits 21 dB BELOW ch1 at the antenna, and the
# tool called it DEAD -- yet ch0 swings 18.7 dB with the emitter's bursts while
# ch1 swings 2.4 dB. ch1 is the higher level and the one NOT following the
# transmitter. Level alone cannot tell a weak antenna from a strong local
# source, so the verdict must consider whether a channel tracks the emitter.


def _level_vs_tracking_receiver(total: int = 100):
    """ch0: weak but follows the bursts. ch1: strong, steady, ignores them."""
    rng = np.random.default_rng(1)
    loud = rng.random(total) < 0.35
    loud[0], loud[1] = True, False

    gains = np.full((total, 2), 40.0)
    output = np.empty((total, 2))
    output[loud, 0] = -78.0 + 40.0 + 18.0   # ch0 rises on a burst
    output[~loud, 0] = -78.0 + 40.0
    output[:, 1] = -57.0 + 40.0             # ch1 never moves

    return _FakeGroup(
        signal_matrix=_frames_from_dbfs(output),
        gains=gains,
        rssis=np.full((total, 2), 70.0),
    )


def test_a_channel_that_ignores_the_emitter_is_flagged_however_strong_it_is():
    from spf.scripts.rx_signal_metrics import analyse_receivers

    stats = _channel_stats(_level_vs_tracking_receiver(), frames=100)
    assert stats[0]["burst_swing_db"] > 10.0, "ch0 should track the bursts"
    assert stats[1]["burst_swing_db"] < 5.0, "ch1 should not"

    findings = analyse_receivers({"r0": stats})["findings"]
    entry = next(f for f in findings if f["receiver"] == "r0")

    assert entry.get("not_tracking_channel") == 1, (
        "the channel that does not follow the emitter is ch1, regardless of "
        f"it being the stronger one: {entry}"
    )
