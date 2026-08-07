"""Per-channel receive-path health for a v7 capture.

Answers one question: is every antenna actually receiving?

The structural validator (`validate_direct_usb_v7_zarr`) checks that frames
exist, are finite and have the right shape. A dead antenna passes all of that
— the samples are finite, the shape is right, there is simply no signal in
them. Rover 4 recorded 3000 such frames on 2026-08-04 with one channel 24 dB
below its sibling and its AGC railed on 99% of frames, and nothing complained.

**Comparisons are input-referred, not raw magnitude.** With AGC running, the
gain loop's whole job is to equalise output magnitude, so two channels can read
the same mean |IQ| while one of them is far weaker at the antenna and is simply
being amplified harder. Rover 1 on 2026-08-05 read 0.8 dB and 1.1 dB apart on
raw magnitude and passed -- while r1 ch0 was drawing 11.9 dB more gain to get
there, i.e. 13.0 dB down at the antenna. Referring to the input undoes the
AGC's compensation:

    input_dbfs = 20*log10(mean|IQ| / full_scale) - gain_dB

**The emitter is bursty, so the comparison is burst-gated.** The O4 is a digital
video transmitter and is silent in most frames (`data_quality_plan.md` records
60-70% NaN as its healthy baseline). Referring to the input per CAPTURE rather
than per FRAME is wrong twice over: the mean |IQ| includes the gaps, and the
mean gain belongs to neither the bursts nor the gaps when the AGC rails in
between. On 2026-08-05 that combination reported ch0 at -83.2 dB when the burst
value was near -50, and three conclusions were withdrawn because of it.

So input-referral happens per frame, frames carrying a transmission are selected
once per receiver (both channels burst together), and the channels are compared
only on those. A continuous emitter has no swing to gate on and every frame is
selected, so this is a no-op where the old behaviour was already right.

Per receiver and channel this reports mean |IQ|, output dBFS, input-referred
dBFS, mean gain, the fraction of frames railed at the 62 dB hardware ceiling,
whether the gain is FROZEN (never moved -- a different fault, and the one that
found rover 3's antenna), the burst swing, and mean RSSI. It then compares the
two channels of each receiver, and the same channel across receivers, which is
what localises a fault:

  - both channels of one receiver down, others fine  -> that radio
  - one channel down, its sibling the strongest      -> that antenna/cable
  - the same channel down on every receiver          -> systematic, not a fault

Reads READ-ONLY and samples a bounded number of frames, so it is fast on a
multi-gigabyte store.

Usage:
  python -m spf.scripts.rx_signal_metrics <zarr> [<zarr> ...] [--json out.json]
  python -m spf.scripts.rx_signal_metrics <zarr> --frames 40
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

# A healthy pair of antennas on one receiver sits within a few dB. The fleet
# baseline as of 2026-08-04 is 6-11 dB, which is already worth explaining; 20 dB
# is not an imbalance, it is a channel that is not connected to anything.
DEAD_CHANNEL_DB = 20.0
IMBALANCE_WARN_DB = 6.0
# ADC full scale for the Pluto's 12-bit signed samples.
FULL_SCALE = 2**11
# An AGC that is railed at maximum is not measuring, it is searching.
RAILED_WARN_FRACTION = 0.5
# AD9361 maximum RX gain. "Railed" means AT THIS, not merely at whatever the
# highest gain in the capture happened to be -- the old test was
# `gain == max(gain)`, which reports 100% for any channel whose gain never
# moves. On 2026-08-06 that flagged rover 3 at a constant 29 dB as "railed".
MAX_RX_GAIN_DB = 62.0
RAILED_GAIN_DB = MAX_RX_GAIN_DB - 1.0
# A channel following a bursty emitter moves tens of dB between burst and gap;
# one sitting on a steady local source barely moves at all.
NOT_TRACKING_SWING_DB = 5.0
TRACKING_SWING_DB = 10.0


def _burst_mask(input_dbfs: np.ndarray) -> np.ndarray:
    """Which frames carried a transmission, for a bursty emitter.

    The O4 transmits in bursts and is silent in most frames, so a statistic
    taken over every frame describes the silence. Both channels of a receiver
    burst together, so the frames are chosen once -- from whichever channel
    swings most between burst and gap -- and then applied to both, which keeps
    the channel comparison like-for-like.

    A continuous emitter has almost no swing, the threshold collapses onto the
    distribution, and every frame is selected. So this is a no-op exactly where
    the old all-frame behaviour was already correct.
    """
    if input_dbfs.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    low = np.nanpercentile(input_dbfs, 20, axis=0)
    high = np.nanpercentile(input_dbfs, 90, axis=0)
    driver = int(np.nanargmax(high - low))
    threshold = (low[driver] + high[driver]) / 2.0
    mask = input_dbfs[:, driver] >= threshold
    return mask if mask.any() else np.ones(input_dbfs.shape[0], dtype=bool)


def _channel_stats(group, frames: int) -> list[dict]:
    signal = group["signal_matrix"]
    total = signal.shape[0]
    step = max(1, total // max(frames, 1))

    all_gains = np.asarray(group["gains"][:])
    rssis = np.asarray(group["rssis"][:])
    rssis = rssis[np.isfinite(rssis).all(axis=1)]

    # Gains are taken at the SAME indices as the magnitudes. Previously the
    # magnitudes were sampled every `step` frames while the gain was averaged
    # over ALL of them, so the two described different sets of frames.
    magnitudes, sampled_gains, zero_frames, sampled = [], [], 0, 0
    for index in range(0, total, step):
        sampled += 1
        frame = signal[index]
        if np.abs(frame).sum() == 0:
            zero_frames += 1
            continue
        gain_row = all_gains[index] if index < len(all_gains) else None
        if gain_row is None or not np.isfinite(gain_row).all():
            continue
        # Measure the AC content, not the DC offset. Every ch1 in the fleet
        # carries an LO-leakage spur at -8 to -15 dBFS, 29-43 dB over its own
        # noise floor, while ch0 carries none -- so raw mean|IQ| compares an
        # internal artifact on one channel against antenna signal on the other.
        # DC is generated in the mixer and is not something an antenna received.
        centred = frame - frame.mean(axis=1, keepdims=True)
        magnitudes.append(np.abs(centred).mean(axis=1))
        sampled_gains.append(gain_row)
        if len(magnitudes) >= frames:
            break

    if magnitudes:
        magnitude_rows = np.asarray(magnitudes)
        gain_rows = np.asarray(sampled_gains)
        # Per frame, not per capture: with the AGC railing in the gaps and
        # dropping on a burst, a single mean gain belongs to neither, and
        # subtracting it puts "AT ANTENNA" tens of dB out. That is what
        # reported ch0 at -83.2 dB on 2026-08-05.
        output_rows = 20 * np.log10(
            np.maximum(magnitude_rows, 1e-12) / FULL_SCALE
        )
        input_rows = output_rows - gain_rows
        burst = _burst_mask(input_rows)
    else:
        magnitude_rows = np.zeros((1, 2))
        gain_rows = np.full((1, 2), np.nan)
        output_rows = np.full((1, 2), np.nan)
        input_rows = np.full((1, 2), np.nan)
        burst = np.ones(1, dtype=bool)

    out = []
    for channel in range(magnitude_rows.shape[1]):
        column = gain_rows[:, channel]
        finite_gain = column[np.isfinite(column)]
        in_burst = input_rows[burst, channel]
        out.append(
            {
                "channel": channel,
                "mean_abs_iq": float(magnitude_rows[burst, channel].mean()),
                "dbfs": float(np.nanmedian(output_rows[burst, channel])),
                "input_dbfs": float(np.nanmedian(in_burst)),
                "mean_gain": float(np.nanmean(gain_rows[burst, channel])),
                # At the hardware ceiling, not at "the biggest value seen".
                "railed_fraction": (
                    float((finite_gain >= RAILED_GAIN_DB).mean())
                    if finite_gain.size
                    else float("nan")
                ),
                # The old railed_fraction was really measuring this, and it did
                # find a real antenna fault that way -- so keep it, named for
                # what it is. A gain that never moves has nothing to track.
                "distinct_gains": int(np.unique(finite_gain).size),
                "gain_is_frozen": bool(np.unique(finite_gain).size == 1),
                "burst_frame_fraction": float(burst.mean()),
                "burst_swing_db": (
                    float(
                        np.nanpercentile(input_rows[:, channel], 90)
                        - np.nanpercentile(input_rows[:, channel], 20)
                    )
                    if magnitudes
                    else float("nan")
                ),
                "mean_rssi": (
                    float(rssis[:, channel].mean()) if len(rssis) else float("nan")
                ),
                "frames_total": int(total),
                "frames_sampled": int(sampled),
                "zero_frame_fraction": float(zero_frames / max(sampled, 1)),
            }
        )
    return out


def _db(numerator: float, denominator: float) -> float:
    return float(20 * np.log10(max(numerator, 1e-12) / max(denominator, 1e-12)))


def analyse(path: str, frames: int) -> dict:
    store = zarr_open_from_lmdb_store(path, mode="r")
    receivers = {}
    for name in sorted(store["receivers"].keys()):
        receivers[name] = _channel_stats(store["receivers"][name], frames)

    report = analyse_receivers(receivers)
    report["path"] = path
    return report


def analyse_receivers(receivers: dict) -> dict:
    """Turn per-channel stats into findings. Split out so it can be tested."""
    report = {"receivers": receivers, "findings": []}

    for name, channels in receivers.items():
        if len(channels) < 2:
            continue
        # Rank by what arrives at the antenna, not by what leaves the AGC.
        usable = [c for c in channels if np.isfinite(c["input_dbfs"])]
        if len(usable) < 2:
            weaker, stronger = sorted(channels, key=lambda c: c["mean_abs_iq"])
            spread = _db(stronger["mean_abs_iq"], weaker["mean_abs_iq"])
        else:
            weaker, stronger = sorted(usable, key=lambda c: c["input_dbfs"])
            spread = stronger["input_dbfs"] - weaker["input_dbfs"]
        report["receivers"][name] = channels
        entry = {
            "receiver": name,
            "imbalance_db": spread,
            "weak_channel": weaker["channel"],
            "gain_delta_db": weaker["mean_gain"] - stronger["mean_gain"],
            "output_imbalance_db": _db(
                stronger["mean_abs_iq"], weaker["mean_abs_iq"]
            ),
        }
        if spread >= DEAD_CHANNEL_DB:
            entry["verdict"] = "DEAD"
        elif spread >= IMBALANCE_WARN_DB:
            entry["verdict"] = "IMBALANCED"
        else:
            entry["verdict"] = "OK"
        # Level alone cannot tell a weak antenna from a strong LOCAL source.
        # Real RO1 data, 2026-08-05: ch0 sat 21 dB below ch1 and was called
        # DEAD -- yet ch0 swung 18.7 dB with the emitter's bursts while ch1
        # swung 2.4 dB. The higher-level channel was the one NOT receiving the
        # transmitter. A channel that does not move with the emitter is a
        # different fault from one that is simply quieter, and pointing an
        # operator at the wrong antenna costs a field session.
        swings = [c.get("burst_swing_db", float("nan")) for c in channels]
        if (
            all(np.isfinite(sw) for sw in swings)
            and min(swings) < NOT_TRACKING_SWING_DB
            and max(swings) >= TRACKING_SWING_DB
        ):
            entry["not_tracking_channel"] = int(
                channels[int(np.argmin(swings))]["channel"]
            )
            entry["not_tracking_swing_db"] = float(min(swings))

        report["findings"].append(entry)

    # Same channel across receivers: isolates a radio from an antenna.
    names = sorted(receivers)
    if len(names) == 2:
        for channel in (0, 1):
            a = receivers[names[0]][channel]
            b = receivers[names[1]][channel]
            if np.isfinite(a["input_dbfs"]) and np.isfinite(b["input_dbfs"]):
                delta = b["input_dbfs"] - a["input_dbfs"]
            else:
                delta = _db(b["mean_abs_iq"], a["mean_abs_iq"])
            report.setdefault("cross_receiver_db", {})[f"ch{channel}"] = delta

    return report


def render(report: dict) -> int:
    print(f"\n{report['path'].split('/')[-1]}")
    header = (f"{'rx':>4} {'ch':>3} {'mean|IQ|':>10} {'out dBFS':>9} {'gain':>6} "
              f"{'AT ANTENNA':>11} {'swing':>7} {'railed':>7} {'rssi':>7}")
    print(header)
    print("-" * len(header))
    for name, channels in report["receivers"].items():
        for c in channels:
            print(f"{name:>4} {c['channel']:>3} {c['mean_abs_iq']:>10.1f} "
                  f"{c['dbfs']:>9.1f} {c['mean_gain']:>6.1f} "
                  f"{c['input_dbfs']:>11.1f} "
                  f"{c.get('burst_swing_db', float('nan')):>6.1f} "
                  f"{c['railed_fraction']*100:>6.0f}% {c['mean_rssi']:>7.1f}")
    print("\n  'AT ANTENNA' = output dBFS minus gain. This is the comparison that"
          "\n  matters: the AGC equalises the output column, so two channels can"
          "\n  read the same mean |IQ| while one is far weaker at the antenna.")
    print("\n  'swing' = how far this channel moves between burst and gap. It is"
          "\n  the measure that says whether a channel is RECEIVING THE EMITTER at"
          "\n  all, as opposed to sitting on a steady local source: a channel fed by"
          "\n  the transmitter tracks its bursts, one fed by interference does not."
          "\n  A low swing beside a high level means the level is not antenna signal.")

    print()
    worst = "OK"
    for f in report["findings"]:
        print(f"  {f['receiver']}: ch{f['weak_channel']} is "
              f"{f['imbalance_db']:.1f} dB below its sibling at the antenna "
              f"-> {f['verdict']}")
        # Spell out the AGC's compensation when it is doing the hiding.
        if abs(f["gain_delta_db"]) >= 3.0:
            print(f"        (raw output differs by only "
                  f"{f['output_imbalance_db']:.1f} dB because the AGC gave ch"
                  f"{f['weak_channel']} {f['gain_delta_db']:+.1f} dB more gain)")
        if "not_tracking_channel" in f:
            print(f"        BUT ch{f['not_tracking_channel']} barely moves with "
                  f"the emitter ({f['not_tracking_swing_db']:.1f} dB swing): it is "
                  "NOT receiving the")
            print("        transmitter. Its level may look high -- that is a steady "
                  "local source,")
            print("        not antenna signal. Check THAT channel, not the quieter "
                  "one.")
        if f["verdict"] == "DEAD":
            worst = "DEAD"
        elif f["verdict"] == "IMBALANCED" and worst != "DEAD":
            worst = "IMBALANCED"

    cross = report.get("cross_receiver_db")
    if cross:
        print("\n  same channel across receivers, at the antenna "
              "(large gap on one channel only = antenna/cable, not the radio):")
        for channel, value in cross.items():
            print(f"    {channel}: {value:+.1f} dB")

    railed = [
        (n, c["channel"], c["railed_fraction"])
        for n, chans in report["receivers"].items()
        for c in chans
        if c["railed_fraction"] >= RAILED_WARN_FRACTION
    ]
    if railed:
        print(f"\n  AGC railed at the {MAX_RX_GAIN_DB:.0f} dB ceiling "
              "(searching, not measuring):")
        for name, channel, fraction in railed:
            print(f"    {name} ch{channel}: {fraction*100:.0f}% of frames")

    # Distinct from railing: a gain that never MOVES has nothing varying to
    # track. Rover 3's B1 sat at a constant 29 dB on 2026-08-06 and a replaced
    # antenna unfroze it, so this is an antenna check, not an AGC-mode one.
    frozen = [
        (n, c["channel"], c["mean_gain"])
        for n, chans in report["receivers"].items()
        for c in chans
        if c.get("gain_is_frozen")
    ]
    if frozen:
        print("\n  AGC frozen (gain never moved -- nothing varying to track;")
        print("  check that channel's antenna and cable):")
        for name, channel, gain in frozen:
            print(f"    {name} ch{channel}: constant {gain:.0f} dB")

    print()
    if worst == "DEAD":
        print("FAIL: a channel is not receiving. Swap that antenna, then its "
              "cable; if the sibling channel on the same radio is healthy the "
              "radio itself is fine.")
        return 1
    if worst == "IMBALANCED":
        print("WARN: channels differ by more than "
              f"{IMBALANCE_WARN_DB:.0f} dB at the antenna, though both are\n"
              "  receiving. Check the weaker channel's antenna and cable before\n"
              "  trusting phase measurements from this receiver.")
        return 0
    print("PASS: both channels receiving, and balanced at the antenna.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zarrs", nargs="+")
    parser.add_argument("--frames", type=int, default=24,
                        help="Frames to sample per receiver (bounded, for speed).")
    parser.add_argument("--json", help="Write the full report here.")
    args = parser.parse_args(argv)

    reports, status = [], 0
    for path in args.zarrs:
        try:
            report = analyse(path, args.frames)
        except Exception as error:
            print(f"ERROR: {path}: {error}", file=sys.stderr)
            status = 2
            continue
        reports.append(report)
        status = max(status, render(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2, sort_keys=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
