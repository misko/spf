#!/usr/bin/env python3
"""Benchmark the primitives a Pluto+ frequency scanner would be built from.

Separates three costs that get conflated:
  1. software cost of one attribute read/write on a PERSISTENT context
     (the 67 ms figure measured earlier was process spawn, not libiio)
  2. hardware cost of a full LO retune, i.e. write frequency -> RSSI reflects the new band
  3. hardware cost of an AD9361 fastlock recall for the same hop

Settle is measured end-to-end against a real observable: a tone is parked in-band at one
LO and far out of band at the other, so RSSI swings tens of dB and the crossing time is
unambiguous. That is what a scanner actually waits for -- not a datasheet lock time.
"""
import statistics as st
import sys
import time

import iio

URI = sys.argv[1] if len(sys.argv) > 1 else "usb:1.90.5"
F_ON = 868_000_000      # tone (TX2 DDS at +100 kHz) lands in band
F_OFF = 1_200_000_000   # tone far outside the 3 MHz RX bandwidth
N = 60

ctx = iio.Context(URI)
print(f"context: {ctx.attrs.get('hw_serial','?')[:12]}  fw {ctx.attrs.get('fw_version','?')}")
phy = ctx.find_device("ad9361-phy")
lo = next(c for c in phy.channels if c.id == "altvoltage0" and c.output)
rx0 = next(c for c in phy.channels if c.id == "voltage0" and not c.output)


def rssi():
    return float(rx0.attrs["rssi"].value.split()[0])


def set_lo(hz):
    lo.attrs["frequency"].value = str(int(hz))


def timeit(fn, n=200):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e3)   # microseconds
    ts.sort()
    return {"mean_us": st.mean(ts), "p50_us": ts[len(ts)//2],
            "p95_us": ts[int(0.95*len(ts))], "min_us": ts[0], "max_us": ts[-1]}


# ---------- 1. per-call software cost on a persistent context ----------
print("\n=== 1. attribute cost on a PERSISTENT context (microseconds) ===")
set_lo(F_ON); time.sleep(0.3)
for name, fn in (("read rssi", rssi),
                 ("write frequency (same value)", lambda: set_lo(F_ON)),
                 ("read frequency", lambda: lo.attrs["frequency"].value)):
    r = timeit(fn)
    print(f"  {name:30} p50 {r['p50_us']:8.1f}  p95 {r['p95_us']:8.1f}  min {r['min_us']:8.1f}")

# ---------- baseline RSSI at each LO, so the crossing threshold is real ----------
set_lo(F_ON);  time.sleep(0.5); on  = st.median([rssi() for _ in range(20)])
set_lo(F_OFF); time.sleep(0.5); off = st.median([rssi() for _ in range(20)])
print(f"\n  RSSI with tone in band  ({F_ON/1e6:.0f} MHz): {on:.2f} dB")
print(f"  RSSI with tone out band ({F_OFF/1e6:.0f} MHz): {off:.2f} dB")
if abs(on - off) < 10:
    print("  !! swing too small to time a crossing -- is the tone on?")
    sys.exit(1)
mid = (on + off) / 2
print(f"  crossing threshold: {mid:.2f} dB")


def settle(target_hz, want_below, recall=None, budget_s=0.5):
    """Hop, then poll RSSI until it crosses. Returns (us_to_cross, n_polls)."""
    t0 = time.perf_counter_ns()
    if recall is None:
        set_lo(target_hz)
    else:
        lo.attrs["fastlock_recall"].value = str(recall)
    polls = 0
    while True:
        v = rssi(); polls += 1
        ok = (v < mid) if want_below else (v > mid)
        if ok:
            return (time.perf_counter_ns() - t0) / 1e3, polls
        if (time.perf_counter_ns() - t0) / 1e9 > budget_s:
            return None, polls


# ---------- 2. full retune ----------
print("\n=== 2. FULL retune: write frequency -> RSSI reflects the new band ===")
res = {"to_on": [], "to_off": []}
for _ in range(N):
    set_lo(F_OFF); time.sleep(0.05)
    us, _ = settle(F_ON, want_below=True)       # tone in band -> RSSI drops
    if us: res["to_on"].append(us)
    set_lo(F_ON); time.sleep(0.05)
    us, _ = settle(F_OFF, want_below=False)     # tone gone -> RSSI rises
    if us: res["to_off"].append(us)
for k, v in res.items():
    if v:
        v.sort()
        print(f"  hop {k:7} n={len(v):3}  p50 {v[len(v)//2]/1000:7.2f} ms  "
              f"p95 {v[int(0.95*len(v))]/1000:7.2f} ms  min {v[0]/1000:7.2f} ms")

# ---------- 3. fastlock ----------
print("\n=== 3. FASTLOCK: store two profiles, recall instead of retuning ===")
set_lo(F_ON);  time.sleep(0.4); lo.attrs["fastlock_store"].value = "0"
set_lo(F_OFF); time.sleep(0.4); lo.attrs["fastlock_store"].value = "1"
print(f"  profile 0 @ {F_ON/1e6:.0f} MHz, profile 1 @ {F_OFF/1e6:.0f} MHz")
print(f"  recall cost alone: ", end="")
r = timeit(lambda: lo.attrs["fastlock_recall"].__setattr__("value", "0"), n=200)
print(f"p50 {r['p50_us']:.1f} us  p95 {r['p95_us']:.1f} us")
fl = {"to_on": [], "to_off": []}
for _ in range(N):
    lo.attrs["fastlock_recall"].value = "1"; time.sleep(0.05)
    us, _ = settle(None, want_below=True, recall=0)
    if us: fl["to_on"].append(us)
    lo.attrs["fastlock_recall"].value = "0"; time.sleep(0.05)
    us, _ = settle(None, want_below=False, recall=1)
    if us: fl["to_off"].append(us)
for k, v in fl.items():
    if v:
        v.sort()
        print(f"  hop {k:7} n={len(v):3}  p50 {v[len(v)//2]/1000:7.2f} ms  "
              f"p95 {v[int(0.95*len(v))]/1000:7.2f} ms  min {v[0]/1000:7.2f} ms")

# ---------- 4. how many profiles, and can they be reloaded from host? ----------
print("\n=== 4. fastlock profile capacity and host reload ===")
saved = {}
for slot in range(8):
    try:
        lo.attrs["fastlock_save"].value = str(slot)
        saved[slot] = lo.attrs["fastlock_save"].value
    except OSError as exc:
        print(f"  slot {slot}: save failed ({exc})")
print(f"  slots readable via fastlock_save: {sorted(saved)}")
if 0 in saved:
    print(f"  profile 0 bytes: {saved[0]}")
    r = timeit(lambda: lo.attrs["fastlock_load"].__setattr__("value", saved[0]), n=100)
    print(f"  fastlock_load (16 bytes from host): p50 {r['p50_us']:.1f} us  p95 {r['p95_us']:.1f} us")

set_lo(F_ON)
print("\ndone")
