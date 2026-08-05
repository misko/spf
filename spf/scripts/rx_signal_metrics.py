"""Per-channel receive-path health for a v7 capture.

Answers one question: is every antenna actually receiving?

The structural validator (`validate_direct_usb_v7_zarr`) checks that frames
exist, are finite and have the right shape. A dead antenna passes all of that
— the samples are finite, the shape is right, there is simply no signal in
them. Rover 4 recorded 3000 such frames on 2026-08-04 with one channel 24 dB
below its sibling and its AGC railed on 99% of frames, and nothing complained.

Per receiver and channel this reports mean |IQ|, power in dBFS, mean gain, the
fraction of frames with gain railed at maximum, and mean RSSI. It then compares
the two channels of each receiver, and the same channel across receivers, which
is what localises a fault:

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
        out.append(
            {
                "channel": channel,
                "mean_abs_iq": magnitude,
                "dbfs": float(20 * np.log10(max(magnitude, 1e-12) / FULL_SCALE)),
                "mean_gain": float(np.nanmean(column)),
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
        weaker, stronger = sorted(channels, key=lambda c: c["mean_abs_iq"])
        spread = _db(stronger["mean_abs_iq"], weaker["mean_abs_iq"])
        report["receivers"][name] = channels
        entry = {"receiver": name, "imbalance_db": spread,
                 "weak_channel": weaker["channel"]}
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
            a = receivers[names[0]][channel]["mean_abs_iq"]
            b = receivers[names[1]][channel]["mean_abs_iq"]
            report.setdefault("cross_receiver_db", {})[f"ch{channel}"] = _db(b, a)

    return report


def render(report: dict) -> int:
    print(f"\n{report['path'].split('/')[-1]}")
    header = (f"{'rx':>4} {'ch':>3} {'mean|IQ|':>10} {'dBFS':>7} {'gain':>6} "
              f"{'railed':>7} {'rssi':>7}")
    print(header)
    print("-" * len(header))
    for name, channels in report["receivers"].items():
        for c in channels:
            print(f"{name:>4} {c['channel']:>3} {c['mean_abs_iq']:>10.1f} "
                  f"{c['dbfs']:>7.1f} {c['mean_gain']:>6.1f} "
                  f"{c['railed_fraction']*100:>6.0f}% {c['mean_rssi']:>7.1f}")

    print()
    worst = "OK"
    for f in report["findings"]:
        print(f"  {f['receiver']}: ch{f['weak_channel']} is "
              f"{f['imbalance_db']:.1f} dB below its sibling -> {f['verdict']}")
        if f["verdict"] == "DEAD":
            worst = "DEAD"
        elif f["verdict"] == "IMBALANCED" and worst != "DEAD":
            worst = "IMBALANCED"

    cross = report.get("cross_receiver_db")
    if cross:
        print("\n  same channel across receivers "
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
              f"{IMBALANCE_WARN_DB:.0f} dB but both are receiving.")
        return 0
    print("PASS: both channels receiving on every receiver.")
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
