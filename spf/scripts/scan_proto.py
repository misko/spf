#!/usr/bin/env python3
"""Host-driven Pluto+ frequency scanner -- baseline, and a test of the latency model.

Model from the measured primitives (persistent libiio context, USB):
    per frequency  =  LO write (1.31 ms)  +  ceil(Y / rssi_read) * rssi_read (0.57 ms)
with no hardware settle term, because the AD9361 is settled before the first read
completes. If the model is right, total scan time is predictable from N and Y alone and
is INDEPENDENT of sample rate and of how far apart the frequencies are.

Power is read from the AD9361's own RSSI, which is input-referred, so nothing is streamed
and the ~2.9 MS/s transport wall never applies. Validity flags come from the CTRL_OUT
detector bits characterised in E-AGC1: RSSI is only a correct input-power estimate while
the ADC is out of overload (it goes constant below overload and rises 1:1 with gain above).
"""
import argparse
import json
import statistics as st
import time

import iio


class Scanner:
    def __init__(self, uri, rx_gain_db=41, sample_rate=3_000_000, bandwidth=3_000_000):
        self.ctx = iio.Context(uri)
        self.phy = self.ctx.find_device("ad9361-phy")
        self.lo = next(c for c in self.phy.channels
                       if c.id == "altvoltage0" and c.output)
        self.rx = [next(c for c in self.phy.channels
                        if c.id == f"voltage{i}" and not c.output) for i in (0, 1)]
        self.serial = self.ctx.attrs.get("hw_serial", "?")
        for ch in self.rx:
            ch.attrs["gain_control_mode"].value = "manual"
            ch.attrs["hardwaregain"].value = str(int(rx_gain_db))
        self.rx[0].attrs["sampling_frequency"].value = str(int(sample_rate))
        self.rx[0].attrs["rf_bandwidth"].value = str(int(bandwidth))
        self.rx_gain_db = rx_gain_db
        # CTRL_OUT detector page, so overload flags are readable (E-AGC1)
        self.detectors = False
        try:
            self.phy.reg_write(0x035, 0x03)
            self.detectors = True
        except Exception:
            pass

    def _rssi(self, ch):
        return float(self.rx[ch].attrs["rssi"].value.split()[0])

    def _overload(self):
        """CH1/CH2 large-ADC-overload from register 0x2B* is not exposed; use the
        AD9361 overload registers instead. Returns None when unavailable."""
        try:
            return self.phy.reg_read(0x02A) if self.detectors else None
        except Exception:
            return None

    def scan(self, plan, dwell_ms=0.0, dual=False):
        """plan: list of dicts {hz, bw_hz?}. Returns rows + timing."""
        out = []
        t_start = time.perf_counter_ns()
        for entry in plan:
            t0 = time.perf_counter_ns()
            self.lo.attrs["frequency"].value = str(int(entry["hz"]))
            if entry.get("bw_hz"):
                self.rx[0].attrs["rf_bandwidth"].value = str(int(entry["bw_hz"]))
            # dwell: keep reading RSSI for at least dwell_ms, then take the last
            reads = 0
            v0 = self._rssi(0)
            reads += 1
            while (time.perf_counter_ns() - t0) / 1e6 < dwell_ms:
                v0 = self._rssi(0)
                reads += 1
            row = {"hz": entry["hz"], "rssi_rx1_db": v0, "reads": reads,
                   "power_dbfs_referred": round(self.rx_gain_db - v0, 2)}
            if dual:
                row["rssi_rx2_db"] = self._rssi(1)
                reads += 1
            row["elapsed_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 3)
            out.append(row)
        total_ms = (time.perf_counter_ns() - t_start) / 1e6
        per = [r["elapsed_ms"] for r in out]
        per.sort()
        return {"rows": out, "n": len(out), "total_ms": round(total_ms, 2),
                "per_freq_p50_ms": per[len(per)//2],
                "per_freq_p95_ms": per[int(0.95*len(per))],
                "per_freq_mean_ms": round(st.mean(per), 3),
                "freqs_per_second": round(1000*len(out)/total_ms, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="usb:1.90.5")
    ap.add_argument("--n", type=int, action="append", default=[])
    ap.add_argument("--dwell-ms", type=float, default=0.0)
    ap.add_argument("--start-mhz", type=float, default=700.0)
    ap.add_argument("--step-mhz", type=float, default=5.0)
    ap.add_argument("--dual", action="store_true")
    ap.add_argument("--output")
    a = ap.parse_args()
    ns = a.n or [10, 25, 50, 100]
    s = Scanner(a.uri)
    print(f"radio {s.serial[:12]}  gain {s.rx_gain_db} dB  detectors={s.detectors}")
    print(f"\n{'N':>5} {'dwell':>7} {'total':>10} {'per-freq p50':>13} {'p95':>8} {'freq/s':>9}")
    results = []
    for n in ns:
        plan = [{"hz": int((a.start_mhz + i*a.step_mhz)*1e6)} for i in range(n)]
        r = s.scan(plan, dwell_ms=a.dwell_ms, dual=a.dual)
        results.append({"n": n, "dwell_ms": a.dwell_ms, **{k: v for k, v in r.items() if k != "rows"}})
        print(f"{n:>5} {a.dwell_ms:>6.1f}m {r['total_ms']:>9.1f}m {r['per_freq_p50_ms']:>12.3f}m "
              f"{r['per_freq_p95_ms']:>7.3f}m {r['freqs_per_second']:>8.1f}")
    if a.output:
        with open(a.output, "w") as fh:
            json.dump({"serial": s.serial, "results": results}, fh, indent=2, sort_keys=True)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
