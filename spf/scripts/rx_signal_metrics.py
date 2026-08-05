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

Per receiver and channel this reports mean |IQ|, output dBFS, input-referred
dBFS, mean gain, the fraction of frames with gain railed at maximum, and mean
RSSI. It then compares the two channels of each receiver, and the same channel
across receivers, which is what localises a fault:

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


def _channel_stats(group, frames: int) -> list[dict]:
    signal = group["signal_matrix"]
    total = signal.shape[0]
    step = max(1, total // max(frames, 1))
    magnitudes, zero_frames, sampled = [], 0, 0
    for index in range(0, total, step):
        frame = signal[index]
        sampled += 1
        if np.abs(frame).sum() == 0:
            zero_frames += 1
            continue
        magnitudes.append(np.abs(frame).mean(axis=1))
        if len(magnitudes) >= frames:
            break

    gains = np.asarray(group["gains"][:])
    gains = gains[np.isfinite(gains).all(axis=1)]
    rssis = np.asarray(group["rssis"][:])
    rssis = rssis[np.isfinite(rssis).all(axis=1)]
    stacked = np.array(magnitudes) if magnitudes else np.zeros((1, 2))

    out = []
    for channel in range(stacked.shape[1]):
        magnitude = float(stacked[:, channel].mean())
        column = gains[:, channel] if len(gains) else np.array([np.nan])
        output_dbfs = float(20 * np.log10(max(magnitude, 1e-12) / FULL_SCALE))
        mean_gain = float(np.nanmean(column))
        # Undo the AGC. Output level alone says nothing about the antenna when
        # the gain loop is free to compensate for a weak one.
        input_dbfs = (
            output_dbfs - mean_gain if np.isfinite(mean_gain) else float("nan")
        )
        out.append(
            {
                "channel": channel,
                "mean_abs_iq": magnitude,
                "dbfs": output_dbfs,
                "input_dbfs": input_dbfs,
                "mean_gain": mean_gain,
                "railed_fraction": (
                    float((column == np.nanmax(column)).mean())
                    if len(gains)
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

    report = {"path": path, "receivers": receivers, "findings": []}

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
              f"{'AT ANTENNA':>11} {'railed':>7} {'rssi':>7}")
    print(header)
    print("-" * len(header))
    for name, channels in report["receivers"].items():
        for c in channels:
            print(f"{name:>4} {c['channel']:>3} {c['mean_abs_iq']:>10.1f} "
                  f"{c['dbfs']:>9.1f} {c['mean_gain']:>6.1f} "
                  f"{c['input_dbfs']:>11.1f} "
                  f"{c['railed_fraction']*100:>6.0f}% {c['mean_rssi']:>7.1f}")
    print("\n  'AT ANTENNA' = output dBFS minus gain. This is the comparison that"
          "\n  matters: the AGC equalises the output column, so two channels can"
          "\n  read the same mean |IQ| while one is far weaker at the antenna.")

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
        print("\n  AGC railed at maximum (searching, not measuring):")
        for name, channel, fraction in railed:
            print(f"    {name} ch{channel}: {fraction*100:.0f}% of frames")

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
