#!/usr/bin/env python3
"""E-LNK1 metric 5 — does the transport change the measured phase?

H3, the decision-critical one: `angle(RX1) - angle(RX2)` on a fixed fixture must not
depend on how the samples reach the host. Any transport-dependent phase difference is a
defect and disqualifies that transport for SPF.

Two design points that decide whether the answer means anything:

1. **The RX setup is copied from the calibration path, attribute for attribute**
   (`spf/calibrations/dual_rx_gain_frequency/hardware.py:120-141`) -- including the
   RX1/RX2 phase-inversion debug attribute AND register 0x22 bit 6, plus
   `quadrature_tracking_en` on both channels and `set_kernel_buffers_count(1)`. The
   phase-inversion fix changes measured phase directly, so an arm that skipped it would
   look like transport corruption when it is really a configuration difference.

2. **The sample rate is deliberately low enough that no arm is throughput-starved.**
   E-LNK1's throughput half already showed every arm walls at ~2.9 MS/s. Running the
   phase test near that wall would let drops differ per arm and alias onto phase. At
   3 MS/s every arm streams contiguously, which isolates phase from throughput -- the
   thing H3 actually asks about.

Arms are interleaved within each repetition so fixture drift cannot alias onto arm.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import adi
import numpy as np

from spf.bench.dual_rx_phase import analyze_common_tone

SETUP = dict(
    sample_rate_hz=3_000_000,
    bandwidth_hz=3_000_000,
    lo_hz=868_000_000,
    buffer_size=65_536,
    rx_gain_db=41,
    tx_gain_db=0.0,
    tone_offset_hz=100_000.0,
    tone_search_width_hz=25_000.0,
    transient_samples=1_024,
    phase_segments=8,
    tx_digital_scale=0.25,
)


def configure(sdr: "adi.ad9361", *, with_source: bool = False) -> None:
    """Mirror hardware.py:120-141 exactly. Any divergence invalidates the comparison.

    ``with_source`` also arms the TX tone. It is used ONCE at the start of a run: the
    source is then left on for every arm, so the fixture is bit-identical across arms.
    Re-arming the DDS per measurement raced its own settling and produced captures with
    no tone at all -- the analyzer locked onto noise at 75-122 kHz instead of 100 kHz.
    """
    sdr.rx_destroy_buffer()
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(SETUP["sample_rate_hz"])
    sdr.rx_rf_bandwidth = int(SETUP["bandwidth_hz"])
    sdr.tx_rf_bandwidth = int(SETUP["bandwidth_hz"])
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_buffer_size = int(SETUP["buffer_size"])
    sdr._rxadc.set_kernel_buffers_count(1)

    attr = "adi,rx1-rx2-phase-inversion-enable"
    sdr._ctrl.debug_attrs[attr].value = "1"
    reg = sdr._ctrl.reg_read(0x22)
    sdr._ctrl.reg_write(0x22, reg | (1 << 6))
    if not (sdr._ctrl.reg_read(0x22) & (1 << 6)):
        raise RuntimeError("failed to enable RX1/RX2 phase mitigation")

    for name in ("voltage0", "voltage1"):
        ch = sdr._ctrl.find_channel(name, is_output=False)
        if "quadrature_tracking_en" in ch.attrs:
            ch.attrs["quadrature_tracking_en"].value = "1"

    sdr.rx_lo = int(SETUP["lo_hz"])
    sdr.tx_lo = int(SETUP["lo_hz"])
    sdr.rx_hardwaregain_chan0 = int(SETUP["rx_gain_db"])
    sdr.rx_hardwaregain_chan1 = int(SETUP["rx_gain_db"])
    if with_source:
        sdr.tx_hardwaregain_chan0 = -80.0
        sdr.tx_hardwaregain_chan1 = float(SETUP["tx_gain_db"])
        sdr.dds_single_tone(
            int(SETUP["tone_offset_hz"]), SETUP["tx_digital_scale"], channel=1
        )


def measure(uri: str, discard: int = 4) -> dict:
    sdr = adi.ad9361(uri=uri)
    try:
        configure(sdr)          # RX only -- the source stays as armed by arm_source()
        time.sleep(1.0)
        for _ in range(discard):
            sdr.rx()
        raw = sdr.rx()
        matrix = np.asarray([np.asarray(raw[0]), np.asarray(raw[1])])
        result = analyze_common_tone(
            matrix,
            sample_rate_hz=int(SETUP["sample_rate_hz"]),
            expected_tone_offset_hz=SETUP["tone_offset_hz"],
            tone_search_width_hz=SETUP["tone_search_width_hz"],
            transient_samples=int(SETUP["transient_samples"]),
            phase_segments=int(SETUP["phase_segments"]),
        )
        result["reg_0x22_bit6"] = bool(sdr._ctrl.reg_read(0x22) & (1 << 6))
        result["rx_lo_hz"] = int(sdr.rx_lo)
        result["sample_rate_hz"] = int(sdr.sample_rate)
        result["rx_gains_db"] = [
            float(sdr.rx_hardwaregain_chan0), float(sdr.rx_hardwaregain_chan1)]
        return result
    finally:
        try:
            sdr.rx_destroy_buffer()     # leave TX alone: the source is shared
        except Exception:
            pass
        del sdr


def _jsonable(result: dict) -> dict:
    """Keep every small analyzer field; drop bulky arrays."""
    out = {}
    for key, value in result.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 8:
            out[key] = [float(x) if isinstance(x, (int, float, np.floating)) else x
                        for x in value]
        elif isinstance(value, np.ndarray) and value.size <= 8:
            out[key] = [float(x) for x in value.ravel()]
    return out


def arm_source(uri: str) -> dict:
    """Arm the TX tone once, on one transport, and leave it on for the whole run."""
    sdr = adi.ad9361(uri=uri)
    try:
        configure(sdr, with_source=True)
        time.sleep(2.0)
        return {"armed_via": uri,
                "tx1_gain_db": float(sdr.tx_hardwaregain_chan0),
                "tx2_gain_db": float(sdr.tx_hardwaregain_chan1)}
    finally:
        del sdr


def mute_source(uri: str) -> None:
    sdr = adi.ad9361(uri=uri)
    try:
        sdr.tx_hardwaregain_chan1 = -80.0
    finally:
        del sdr


def circ_mean_deg(values_deg: list[float]) -> float:
    a = np.radians(np.asarray(values_deg))
    return float(np.degrees(np.angle(np.mean(np.exp(1j * a)))))


def circ_spread_deg(values_deg: list[float]) -> float:
    m = circ_mean_deg(values_deg)
    d = [(v - m + 180) % 360 - 180 for v in values_deg]
    return float(max(d) - min(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=uri, repeatable")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tx-gain-db", type=float, default=None,
                    help="override the source drive; needed because the direct-USB path "
                         "reports the same signal ~6.7 dB hotter than libiio, so a level "
                         "valid on one path can be 'too strong' on the other")
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arm)
    if args.tx_gain_db is not None:
        SETUP["tx_gain_db"] = args.tx_gain_db

    source_uri = next(iter(arms.values()))
    source = arm_source(source_uri)
    print(f"source armed once via {source_uri}: {source}")

    rows = []
    for rep in range(args.reps):
        for name, uri in arms.items():           # interleaved, not blocked by arm
            t0 = time.time()
            try:
                r = measure(uri)
                rows.append(dict(
                    arm=name, uri=uri, rep=rep, ok=True,
                    phase_difference_deg=float(np.degrees(r["phase_difference_rad"])),
                    quality_valid=bool(r["quality_valid"]),
                    quality_reasons=r.get("quality_reasons", []),
                    tone_frequency_hz=r.get("tone_frequency_hz"),
                    analyzer=_jsonable(r),
                    reg_0x22_bit6=r["reg_0x22_bit6"],
                    rx_lo_hz=r["rx_lo_hz"], sample_rate_hz=r["sample_rate_hz"],
                    rx_gains_db=r["rx_gains_db"],
                    elapsed_s=round(time.time() - t0, 3)))
                print(f"  rep {rep} {name:12} phase "
                      f"{rows[-1]['phase_difference_deg']:+8.3f} deg  "
                      f"valid={rows[-1]['quality_valid']}")
            except Exception as exc:  # keep going; a dead arm is a result
                rows.append(dict(arm=name, uri=uri, rep=rep, ok=False,
                                 error=f"{type(exc).__name__}: {exc}"))
                print(f"  rep {rep} {name:12} FAILED {type(exc).__name__}: {exc}")

    mute_source(source_uri)
    print("source muted")

    per_arm = {}
    for name in arms:
        ph = [r["phase_difference_deg"] for r in rows
              if r["arm"] == name and r.get("ok") and r.get("quality_valid")]
        if ph:
            per_arm[name] = dict(
                n=len(ph), mean_deg=round(circ_mean_deg(ph), 4),
                spread_deg=round(circ_spread_deg(ph), 4),
                stdev_deg=round(statistics.stdev(ph), 4) if len(ph) > 1 else 0.0,
                values_deg=[round(v, 4) for v in ph])
        else:
            per_arm[name] = dict(n=0, note="no quality-valid measurements")

    names = [n for n in arms if per_arm[n]["n"] > 0]
    within = max((per_arm[n]["spread_deg"] for n in names), default=None)
    between = None
    if len(names) > 1:
        ms = [per_arm[n]["mean_deg"] for n in names]
        between = round(max((m1 - m2 + 180) % 360 - 180 for m1 in ms for m2 in ms), 4)

    doc = dict(
        experiment="E-LNK1", metric="5 (phase agreement across transports)",
        hypothesis="H3", setup=SETUP, arms=arms, reps=args.reps,
        rx_setup_source="copied attribute-for-attribute from "
                        "spf/calibrations/dual_rx_gain_frequency/hardware.py:120-141",
        sample_rate_rationale="3 MS/s, well below the ~2.9 MS/s wall every arm hits, so "
                              "no arm is throughput-starved and drops cannot alias onto "
                              "phase",
        source=source, rows=rows, per_arm=per_arm,
        within_arm_repeatability_deg=within,
        max_between_arm_difference_deg=between,
        verdict=(None if between is None or within is None else
                 ("PASS - between-arm difference is within fixture repeatability"
                  if abs(between) <= within else
                  "FAIL - a transport shifts phase beyond fixture repeatability")),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(args.output, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwithin-arm repeatability (worst): {within} deg")
    print(f"max between-arm difference:      {between} deg")
    print(f"verdict: {doc['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
