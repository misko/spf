# Live SDR Capture Path — `sdr_controller.py` + `data_collector.py`

Deep dive (Passes 1→3 per [`DOCUMENTATION_GUIDE.md`](../../../DOCUMENTATION_GUIDE.md)) of the **live SDR
capture path**: driver layer (`spf/sdrpluto/sdr_controller.py`) and the threaded reader/collector
(`spf/data_collector.py`).

- **Pinned commit:** `574c4548e23c18d95ff3c2b72d3b63f0b1f42251` (2026-02-15). Line numbers below are
  against this commit.
- **Read-only:** no source was modified. Suspected bugs are recorded as evidence, never fixed here.
- Status legend: ✅ Active · ⚠️ Deprecated/legacy · 🔴 Broken (verified) · 🧪 Experimental/unused.
- Confidence: `✅ verified` (read the deciding lines) vs `🔍 analysis` (reasoned path not executed).
- Units/frames: angles are radians unless suffixed `_in_pis` (units of π) or noted **degrees**.
  `rx_spacing` is metres; `avg_phase_diff` is the RX0−RX1 phase difference in radians, wrapped to
  [−π, π). `heading` (V4) is **degrees** (0–360); `rx_theta_in_pis`/`rx_heading_in_pis` are in π units.

---

## The live path in one diagram

```
grbl_radio_collection.py (--mode rx via GrblDataCollectorRaw)        data-version 5
mavlink_radio_collection.py (--fake-drone or real, DroneDataCollectorRaw)  data-version 4
        │
        └── DataCollector.radios_to_online()                      data_collector.py:340
              ├── get_pplus(uri=...) for every radio   (registers in global pplus_online)
              ├── (if emitter type == "sdr") setup_rxtx(...)  → blast a tone, verify peak
              │       NB: in the live RX-only path the emitter is usually type != "sdr"
              │       ("its already blasting"), so this branch is skipped.
              ├── for each receiver: setup_rx(rx_config)        ← THE live RX bring-up
              │       → get_pplus → PPlus.setup_rx_config()  (AD9361 regs + warmup rx)
              └── prepare_threads()                              data_collector.py:413
                    ├── thread_class(pplus, ...).start_read_thread()   one daemon Thread / radio
                    │       thread runs ThreadedRX(Raw).read_forever()
                    └── collector_thread = Thread(run_collector_thread)
        │
        └── DataCollector.start() → run_collector_thread → run_inner_collector_thread
              pulls DataSnapshot* off each read_q (maxsize=1), submits write_to_record_matrix
              to a bounded ThreadPool → zarr (V4/V5) or memmap (V2).
```

**Which setup function the LIVE path uses:** `setup_rx` (sdr_controller.py:885) for receivers, and
`setup_rxtx` (sdr_controller.py:905) only when a co-located **SDR emitter** must be brought online
first. `setup_rxtx_and_phase_calibration` (sdr_controller.py:959) is **NOT on the live collection
path** — its sole caller is the standalone `--mode rxcal` CLI (sdr_controller.py:1256). See #6 below.

---

## Driver layer — `spf/sdrpluto/sdr_controller.py`

### `get_pplus(rx_config=None, tx_config=None, uri=None)` · `sdr_controller.py:231` · ✅ Active · ⚠️ "not thread safe"

**Summary.** The radio registry/factory. Resolves a URI, and either constructs a new driver
(`FakePPlus` / `BladeRFSdr` / `PPlus`, chosen by substring of the URI) or, if the URI is already
open, re-applies the config to the existing object. Returns the cached driver instance.

**Signature**
- Inputs: `rx_config` (`ReceiverConfig`|None), `tx_config` (`EmitterConfig`|None), `uri` (str|None).
  Exactly one of the three suffices to resolve the URI via `get_uri` (l.234).
- Returns: a driver object (`PPlus`-like) registered in the module global `pplus_online`.
- Side effects: **mutates module global `pplus_online`** (l.236–248); **opens hardware** on first
  construction; logging. Marked `# TODO not thread safe` (l.230).

**Dispatch** (l.237–246): `"fake" in uri` → `FakePPlus`; `"bladerf" in uri` → `BladeRFSdr`; else
`PPlus` (Pluto). So URIs like `pluto://ip:...`, `bladerf://...`, `fake-pluto` route correctly.

**Called from** (Pass 2)
- `data_collector.py:347` (`radios_to_online`) — `get_pplus(uri=radio_uri)` for *every* radio, to
  pre-open it. Return value discarded; the side effect (registry population + open) is the point.
- `setup_rx` (sdr_controller.py:890), `setup_rxtx` (l.908, 920),
  `setup_rxtx_and_phase_calibration` (l.970, 981) — each calls `get_pplus` then `.setup_rx_config()`
  (except #6, which calls the wrong method).
- Fan-in: ~6 call sites, all in this cluster.

**Contract**
- *Precondition:* serialized access — there is no lock around `pplus_online`. The repeat-config
  branch (l.248) asserts inside `set_config` that `self.rx_config is None` before assigning, so
  calling `get_pplus(rx_config=...)` **twice** on the same already-configured URI will `AssertionError`
  (set_config l.339/677). The live path avoids this by pre-opening with `uri=` only (no config) then
  configuring later through `setup_rx`.
- *Postcondition:* returned object is in `pplus_online[uri]`.

**Analysis (Pass 3)**
- *Threading hazard (🔍 analysis):* `get_pplus` is the shared-mutable-state choke point. In the
  live path it is only ever called **single-threaded from `radios_to_online`** before any reader
  thread is spawned (`prepare_threads` runs after all `setup_rx` calls). So although the function is
  genuinely not thread-safe, the **live path does not invoke it concurrently** — the reader threads
  never touch `get_pplus`/`pplus_online`; they hold their own `self.pplus` reference. The "not thread
  safe" risk is real only if a future caller calls `get_pplus` from multiple threads. See the
  *Threading & global state* section.
- See also the `set_config` double-config assertion above — a `get_pplus(rx_config=...)` on an
  already-configured live radio crashes, not silently misbehaves.

### `PPlus` (Pluto driver) · `sdr_controller.py:595` · ✅ Active

**Summary.** Production PlutoSDR (AD9361) driver. `__init__` opens `adi.ad9361(uri=...)`, immediately
quiets TX, sleeps 0.5 s, then (on a fresh open) destroys TX/RX buffers. Holds `self.sdr`,
`rx_config`, `tx_config`, `phase_calibration`.

**Key methods on the live path**
- `rx(num_samples=None)` (l.630) → `self.sdr.rx()`. Returns a length-2 list/array of complex128
  channel buffers. **Hardware read.** `num_samples` is ignored.
- `rssis()` (l.656) → 2-vector of per-channel RSSI parsed from the iio `voltage{0,1}` attr string
  (`value[:-3]` strips a unit suffix). **Hardware read.**
- `gains()` (l.664) → 2-vector of hardwaregain via `_get_iio_attr`. **Hardware read.**
- `soft_reset_radio()` (l.633) — **early `return`; the LO-toggle body below it is dead** (l.635–639
  never execute). So the periodic "recalibrate" in `read_forever` (every 100 samples) is a **no-op**
  on Pluto. ⚠️ effectively disabled. (✅ verified — `return` is the first statement.)
- `get_rssi_and_gain()` (l.641) — 4-vector [rssi0, rssi1, gain0, gain1]; has a leftover
  `# breakpoint()` (l.644, commented). **No live caller** (see #14 discussion).
- `setup_rx_config()` (l.703) — the real Pluto RX bring-up: destroys rx buffer, enables
  **`adi,rx1-rx2-phase-inversion-enable`** debug attr and sets reg 0x22 bit6, sets bandwidth /
  sample_rate / lo / FIR / gain modes / buffer size / kernel buffers, enables RX channels, powers
  down TX altvoltage1. Heavily `assert`-guarded against the RFIC silently rejecting a write.
- `__del__` (l.691) — destructor quiets TX and clears `tx_enabled_channels`. Touches `self.sdr`;
  if `__init__` failed before `self.sdr` was set this would raise in GC (🔍, low severity — open
  rarely fails that early).

**AD9361 phase-inversion calibration (Pluto).** The 2-element ULA phase model assumes RX0 and RX1
share a phase reference. The AD9361 inverts RX2's (the second channel's) phase under LNA bypass;
`setup_rx_config` corrects this by (a) enabling the ADI debug attr
`adi,rx1-rx2-phase-inversion-enable` = "1" (l.710–714) and (b) setting register 0x22 bit 6 (l.717–721).
Both are asserted to have taken effect. This is the per-channel hardware correction that makes
`avg_phase_diff = angle(rx0) − angle(rx1)` physically meaningful. (✅ verified the writes + asserts;
the *correctness* of bit semantics vs the AD9364 reference is taken on faith from the cited datasheet
URLs — 🔍 not independently verified against silicon.)

### `BladeRFSdr` · `sdr_controller.py:265` · ✅ Active

**Summary.** BladeRF 2.0 Micro driver. `__init__` resolves a `bladerf://...` URI (optionally mapping
a serial via `bladerf_serial_to_info`), opens the device, grabs RX/TX channels, disables TX.

**Phase-inversion calibration (BladeRF).** `setup_rx_config` (l.352) replicates the Pluto correction
at the RFIC register level via `bladerf_get/set_rfic_register` on `self.sdr.dev[0]`:
- reg 0x022 bit6 **set** — invert phase when LNA bypassed (l.371–379),
- reg 0x011 bit5 **set** — RX1/RX2 phase inversion (l.383–390),
- reg 0x189 bit5 **cleared** — disable inversion in RX2's DC-correction path (l.393–399).
Then per-channel bandwidth/sample_rate/frequency/gain-mode are set and asserted, and `sync_config`
arms RX_X2 MIMO streaming. (✅ verified the register ops; semantics 🔍 unverified vs silicon — the
in-code comments themselves are hedged "presumably", "Pseudo-code".)

**`rx()`** (l.491) — `sync_rx` into a byte buffer, reinterpret as int16, build complex, then
`np.vstack([samples[::2], samples[1::2]])` to de-interleave the two channels. Returns 2×N int16-complex.

**Issues found (BladeRF):**
- `rssis()` (l.312) does `np.array([ch0.rssi, ch1.rssi]).mean(axis=1)` (l.319). This **only works if
  `ch.rssi` is itself a sequence** (so the array is 2×k and `mean(axis=1)` yields a 2-vector). If
  `ch.rssi` is a scalar, the array is shape (2,) and `mean(axis=1)` raises
  `numpy.AxisError`. ⚠️ 🔍 analysis — depends on the bladerf python binding's `.rssi` return type,
  which I could not confirm in-repo (the `bladerf` package is an optional import, l.11–15).
  Reachability: only on real BladeRF hardware via `ThreadedRX.get_rx` (l.250). **No new
  KNOWN_ISSUES entry — unverified return type.** Flag for a maintainer with hardware.
- `gains()` (l.323) returns a hardcoded `np.array([0,0])` — gains are **stubbed**, not read. ⚠️
  Silently wrong-ish but documented as intentional (real read commented out). `no-crash`.
- `check_for_freq_peak()` (l.488) always returns `True` — the `setup_rxtx` peak verification is a
  no-op on BladeRF. ⚠️ stub.
- `BladeRFSdr.__init__` sets `self.sdr.rx = self.rx` (l.296) — monkey-patches the device's `rx`
  attribute to the wrapper method. Works, but note `ThreadedRX.get_rx` calls `self.pplus.sdr.rx()`
  (data_collector.py:249), which after this reassignment is the *wrapper* `BladeRFSdr.rx`, not the
  raw device `rx`. (✅ verified — this is how the BladeRF path reads samples through the threaded
  loop.)

### `FakePPlus` / `FakeSdr` · `sdr_controller.py:502` / `253` · 🧪 fake (test path)

Used whenever a URI contains `"fake"` (e.g. `fake-pluto`). `FakeSdr.rx` (l.257) sleeps 0.3 s and
returns a random 2×buffer complex matrix. `FakePPlus.rssis`/`gains` return `np.random.rand(2)`.
`soft_reset_radio` sleeps 0.1 s. This is the path exercised by `test_mavlink_radio_collect.py`
(the only integration test of the capture pipeline).

- **`FakePPlus.get_rssi_and_gain()` (l.530) — confirms KNOWN_ISSUE #14.** It reads `self.dev.lna_gain`
  / `self.dev.vga1` / `self.dev.vga2` (l.543–545) but `FakePPlus` only ever sets `self.sdr` (a
  `FakeSdr`), never `self.dev`. So this would raise `AttributeError`. **However it is DEAD code:** a
  repo-wide grep for `get_rssi_and_gain` finds zero callers (the live path calls `rssis()` and
  `gains()` separately in `ThreadedRX.get_rx`, data_collector.py:250–251). ✅ verified broken, ✅
  verified unreached. Severity: low (broken-but-dead).

### `setup_rx(rx_config, provided_pplus_rx=None)` · `sdr_controller.py:885` · ✅ Active — THE live RX bring-up

**Summary.** Brings one receiver online with **no TX/peak retries**: get/cache the driver, call
`setup_rx_config()`, sleep 2 s, then read-and-drop 10 buffers to flush stale samples, zero the
`phase_calibration`, return the driver.

**Called from:** `radios_to_online` (data_collector.py:402) per receiver; `--mode rx` CLI
(sdr_controller.py:1272). Return flows into `self.receiver_pplus[uri]` and is asserted non-None
(data_collector.py:404–410); `rx_config.rx_pos` is asserted non-None.

**Analysis.** `setup_rx` never returns None (it always returns `pplus_rx`), so the
`if pplus_rx is None: sys.exit(1)` guard at the call site (data_collector.py:404) is effectively
unreachable for `setup_rx`. (✅ verified — the function has a single `return pplus_rx`.) The
`phase_calibration = 0.0` reset (l.901) means **per-antenna phase calibration is not applied in the
live RX-only path** — calibration happens offline/downstream, not here.

### `setup_rxtx(rx_config, tx_config, leave_tx_on=False, provided_pplus_rx=None)` · `sdr_controller.py:905` · ✅ Active

**Summary.** Bring RX online, then loop (≤15 retries, honoring global `run_radios`): bring TX online,
blast `make_tone`, read-and-drop 400 buffers, and `check_for_freq_peak()`. On success returns
`(pplus_rx, pplus_tx)` (closing TX unless `leave_tx_on`); on exhaustion returns `(None, None)`.

**Called from:** `radios_to_online` (data_collector.py:383) **only when the emitter is an SDR**
(`emitter.type == "sdr"`, l.352) — i.e. the co-located-emitter bring-up; `--mode tx` CLI (l.1291).
In the common drone/wall RX-only configs the emitter is *not* type "sdr" ("its already blasting",
data_collector.py:401 comment), so this branch is skipped. After it returns, `radios_to_online`
calls `pplus_rx.close_rx()` (data_collector.py:388) — **note: if `setup_rxtx` returned `(None, None)`
this dereferences `None.close_rx()` → `AttributeError`** (🔍 analysis; reachable only when an SDR
emitter fails to verify within 15 retries; no None-guard at l.388 unlike the receiver loop's guard
at l.404). Low-moderate severity, hardware-only.

### `setup_rxtx_and_phase_calibration(...)` · `sdr_controller.py:959` · 🔴 Broken (verified) — confirms KNOWN_ISSUE #6

**Summary.** Inter-antenna phase calibration: bring RX up, refuse if another emitter is already
transmitting, bring RX+TX up, then average RX0−RX1 phase over `n_calibration_frames` (default 800)
until the std is under tolerance, storing the result in `pplus_rx.phase_calibration`.

**Confirms #6.** Lines 971 and 982 call `pplus_rx.setup_rx()`, but **no class in this module defines
`setup_rx`** — `PPlus`/`BladeRFSdr`/`FakePPlus` all define `setup_rx_config` (the module-level
*function* `setup_rx` is unrelated). So both calls raise `AttributeError`. ✅ verified (grep:
`def setup_rx_config` at l.352/579/703; `.setup_rx(` only at l.971/982 with no matching def).

**Reachability:** **NOT on the live collection path.** Its only caller is the standalone
`--mode rxcal` CLI block (sdr_controller.py:1256). Normal collection uses `setup_rx`/`setup_rxtx`,
which correctly call `setup_rx_config()`. So this crashes only if a user runs
`sdr_controller.py --mode rxcal` against real hardware. Severity: P2 (broken but off the live path).

**Additional latent bug in the same function (independent of #6):** the success branch is **inside**
the `while run_radios and retries < 20` loop but `retries` is **never incremented** (l.1001–1048). If
`phase_calibration_std >= tolerance` on every pass, the loop spins forever (l.1002) re-collecting 800
frames; `retries` stays 0 so the `< 20` bound is dead. The `pplus_tx.close()` + "failed" return at
l.1049–1051 is only reached if `run_radios` flips to False. (🔍 analysis — would only be observed if
#6 were fixed first, since the function crashes at l.971 before reaching the loop.)

### Other driver helpers (brief)
- `bladerf_serial_to_info` (l.33) — enumerate USB BladeRFs → {serial: uri}; prints + returns `{}` on
  `BladeRFError`. ✅
- `shutdown_radios` (l.212) — sets global `run_radios = False`. The cooperative-shutdown signal read
  by `setup_rxtx`, `setup_rxtx_and_phase_calibration`, `plot_recv_signal`. ✅
- `make_tone` (l.870) — cyclic TX tone buffer at the IF, length chosen to end on a full phase. ✅
- `ReceiverConfig` (l.74) — builds `rx_pos`/`rx_pos_rotated` via `ULADetector` when `rx_spacing` is
  set (l.107–119); otherwise `rx_pos = None`. The `rx_pos is not None` assertions in `ThreadedRX`
  and `setup_rx` depend on a non-None `rx_spacing`. ✅
- `plot_recv_signal` (l.1054) — live matplotlib debug viewer (CLI only). ✅

---

## Threaded reader / collector — `spf/data_collector.py`

### `ThreadedRX` · `data_collector.py:173` · ⚠️ legacy V2 base / ✅ base class

**Summary.** One reader per radio. Holds the driver (`self.pplus`), a **`read_q = queue.Queue(maxsize=1)`**
(a single-slot hand-off), a `run` flag, timing config. `read_forever` is the thread body.

**`read_forever()` (l.185) — the threaded capture loop.** ✅ Active.
- Precomputes steering vectors once (l.187).
- Loop `while self.run`:
  1. `data = self.get_data()` — read one sample (subclass-specific snapshot). On **any** exception,
     log, `read_q.put(None, timeout=5)`, set `run=False`, `continue` → loop exits next check (l.198–204).
  2. **Back-pressure put** (l.205–211): retry `read_q.put(data, timeout=0.5)` until it succeeds *or*
     `run` goes False, swallowing `queue.Full`. Because the queue is maxsize=1, **the producer blocks
     until the collector consumes the previous sample** — there is no silent buffer drop; instead the
     reader *paces itself* to the consumer.
  3. **Periodic recalibrate** (l.214–215): every 100th sample call `self.pplus.soft_reset_radio()`.
     On `PPlus` this is a **no-op** (early return, see driver section); on `FakePPlus` a 0.1 s sleep.
  4. **Rate limiting** (l.216–228): an EWMA of loop time (`alpha=0.9`); if `seconds_per_sample >= 0`
     and the average is below target, `time.sleep` the difference to hit a target sample period.

**Timing / drift notes (Pass 3).**
- The recalibrate branch and the rate-limit branch are mutually exclusive per-iteration (`elif`,
  l.216): on every 100th sample the loop **skips the rate-limit sleep** entirely. Minor cadence
  jitter every 100 samples. (✅ verified — `if idx%100==0 … elif idx>20`.)
- `idx > 20` guards the EWMA warmup, but on iterations 0–20 there is also no rate-limit sleep, so the
  first 20 samples come as fast as the hardware allows. Expected.
- `average_time_per_loop` measures only `get_data` + queue-put time (l.197/212), not the sleep — so
  the controller targets `seconds_per_sample` correctly without integrator wind-up. ✅
- **Default mismatch (🔍, benign):** `ThreadedRX.__init__` defaults `seconds_per_sample=0`
  (l.174), but `prepare_threads` passes `-1` when the yaml lacks `seconds-per-sample` (l.419) →
  rate-limiting **disabled** (the `>= 0` test fails). The grbl/mavlink mains inject a default
  (0.1 / fake yaml 5.0) so live runs do pace. No bug, just two different "defaults."

**`get_rx(max_retries=15)` (l.245) — raw read + rssi/gain with retries.** ✅ Active.
- Reads `self.pplus.sdr.rx()`, `self.pplus.rssis()`, `self.pplus.gains()`; returns a dict.
- On exception: log, `sleep(0.1)`, `tries += 1`, and `if tries > max_retries: sys.exit(1)`.
- **Off-by-one → `sys.exit(1)` is dead code (✅ verified):** the loop is `while tries < max_retries`.
  The 15th failing iteration increments `tries` 14→15, then tests `15 > 15` (False), then the
  `while 15 < 15` check ends the loop. `tries` can never exceed `max_retries` *inside* the loop, so
  the `sys.exit(1)` at l.261 never fires. Instead `get_rx` falls through and **returns `None`**
  (l.262). For `ThreadedRX.get_data` that None is turned into a `ValueError` (l.266–267); for
  `ThreadedRXRaw.get_data` (l.283) **the None is NOT checked** → `sdr_rx["signal_matrix"]` raises
  `TypeError` at l.288. Either way `read_forever`'s `except` catches it, queues `None`, and stops the
  thread — so the practical effect (thread aborts after 15 hardware failures) is the same; only the
  *intended* hard `sys.exit(1)` is unreachable. Severity: low (the library-level `sys.exit` everyone
  worries about can't actually fire here). ⚠️ silently-different-from-intended.

**`start_read_thread(thread=True)` (l.235)** — spawns a daemon `threading.Thread` (default) or a
`multiprocessing.Process` running `read_forever`. The Process variant would not share the in-process
`read_q` correctly with the collector (the collector reads `read_thread.read_q`, a plain
`queue.Queue`, which is **not** picklable across processes) — but the live path always uses the
default `thread=True` (data_collector.py:428 calls it with no arg). 🧪 the process branch is latent.

### `ThreadedRXRaw.get_data()` · `data_collector.py:283` · ✅ Active — V4/V5 live reader

Builds the live snapshot: `np.vstack(sdr_rx["signal_matrix"]).astype(complex64)`, timestamp =
`time.time() - time_offset`, `avg_phase_diff = get_avg_phase_fast2(signal_matrix)`, asserts
`rx_spacing > 0.001`, packs into `self.snapshot_class` (`DataSnapshotV4`/`V5`). Does **not** null-check
`get_rx()` (see above). Subclasses `ThreadedRXRawV4`/`V5` only set `snapshot_class`.

### `DataCollector.radios_to_online()` · `data_collector.py:340` · ✅ Active

**Summary.** The single entry that brings the whole rig online (callers: `grbl_radio_collection.py:179`,
`mavlink_radio_collection.py:305`, both `# blocking`). Steps:
1. Pre-open every radio URI with `get_pplus(uri=...)` (l.346–347) — single-threaded.
2. If `emitter.type == "sdr"`: build target rx/tx configs and `setup_rxtx(..., leave_tx_on=True)`,
   then `pplus_rx.close_rx()` (l.383–388). **No None-guard on the `setup_rxtx` return** (l.388) —
   see `setup_rxtx` note. Skipped for non-SDR emitters (the usual live case).
3. For each receiver: `setup_rx(rx_config)`; `sys.exit(1)` if None (dead guard, see `setup_rx`);
   store in `self.receiver_pplus`; assert `rx_pos is not None`.
4. `prepare_threads()` — spawn one reader thread per radio + the collector thread.

**Asserts (l.396–400):** receivers must not carry an `emitter-uri`, and the run must either lack
`skip_phase_calibration` or have it truthy — i.e. **the live path requires phase calibration be
skipped** (calibration is offline). ✅ verified.

### `DataCollector.run_inner_collector_thread()` · `data_collector.py:450` · ✅ Active

Bounded `ThreadPoolExecutorWithQueueSizeLimit(max_workers=2, maxsize=1)`. For each record index, for
each reader: `data = read_thread.read_q.get()` (**blocking** — back-pressures the readers), bail if
None, submit `write_to_record_matrix`. Keeps the in-flight `futures` list ≤4, draining and logging
any write exceptions (l.468–479). This is the consumer half of the maxsize=1 hand-off — there is **no
dropped-sample path**: a slow writer simply blocks the reader.

### `DataCollector.run_collector_thread()` · `data_collector.py:482` · ✅ Active

Runs the inner loop, sets `finished_collecting=True`, calls `close()` (zarr shrink / memmap flush),
then signals all readers `run=False` and joins them. Clean cooperative shutdown.

### `write_to_record_matrix` — the V4 `heading` mismatch · confirms KNOWN_ISSUE #16

**`DroneDataCollectorRaw.write_to_record_matrix` (data_collector.py:536) — V4.**
- Reads `current_pos_heading_and_time = position_controller.get_position_bearing_and_time()`
  (mavlink_controller.py:404 → `{"gps", "heading", "gps_time"}`, where `heading = msg.hdg/100` is
  **degrees** 0–360, mavlink_controller.py:807).
- Sets `data.heading = ...["heading"]` (l.540), `data.gps_long/lat/timestamp`.
- Writes to zarr by iterating `v4rx_f64_keys + v4rx_2xf64_keys` and `getattr(data, k)` (l.551–552).
  **`v4rx_f64_keys` contains `"heading"`** (v4_data.py:8), **not** `"rx_heading_in_pis"`.

**Verdict on #16 — confirmed, with nuance:**
- `DataSnapshotV4` (data_collector.py:83) inherits `rx_heading_in_pis: float = 0.0` from
  `DataSnapshotRaw` (l.79) and declares **no `heading` field**. Setting `data.heading = ...` adds a
  **dynamic attribute** to the dataclass instance; `getattr(data, "heading")` then succeeds, and the
  zarr key list also uses `"heading"`. So the **V4 write path is internally consistent and does NOT
  crash** — it stores degrees under the zarr key `heading`. (✅ verified end-to-end; the integration
  test `test_mavlink_radio_collect.py` asserts every `v4rx_f64_keys` field, including `heading`, is
  finite.)
- The real defect is a **latent field/units mismatch**: the declared dataclass field
  `rx_heading_in_pis` (units of π) is **never populated in V4** (stays 0.0, and isn't in
  `v4rx_f64_keys`, so it's never written), while the actually-written `heading` (degrees) is an
  **undeclared** attribute. The v4→v5 migration (`v4_tx_rx_to_v5.py:172–175`) bridges the two
  explicitly: it reads src key `heading` and stores `rx_heading_in_pis = heading/360*2`. So **the
  rest of the codebase consumes `rx_heading_in_pis` (π units); only the V4 raw on-disk format uses
  `heading` (degrees).**
- Severity: **silently-wrong-shaped, no crash.** ⚠️ `no-crash / verify downstream before any fix` —
  "fixing" the dataclass to write `rx_heading_in_pis` directly would change the V4 on-disk units
  (degrees → π) and break the `v4_tx_rx_to_v5` `/360*2` conversion and the `test_mavlink_radio_collect`
  key check. The mismatch is load-bearing.

**`GrblDataCollectorRaw.write_to_record_matrix` (l.593) — V5.** Reads tx/rx motor xy from the GRBL
controller, sets `tx_pos_*_mm`/`rx_pos_*_mm`, asserts `rx_lo > 1`, writes `v5rx_f64_keys`. V5 does
**not** touch heading (`rx_heading_in_pis` stays default 0.0 from the snapshot; the wall array has no
heading). Consistent.

**`GrblDataCollector.write_to_record_matrix` (l.645) — V2 legacy** (memmap via
`prepare_record_entry_v2`). Legacy path; uses `DataSnapshot`/`ThreadedRX`.

---

## Threading & global state

The capture path is multi-threaded but the shared mutable state is touched almost entirely **before**
threads start, which is what keeps it (mostly) safe in practice:

| State | Where mutated | Concurrency exposure on the live path |
|---|---|---|
| `pplus_online` (global dict) | `get_pplus` (l.236–248) | Populated **single-threaded** in `radios_to_online` (l.347, and via `setup_rx`/`setup_rxtx`) **before** `prepare_threads` spawns readers. Reader threads never call `get_pplus`; they hold `self.pplus`. So despite the `# TODO not thread safe`, the live path has **no concurrent access**. 🔍 Risk is purely future-proofing. |
| `run_radios` (global bool) | `shutdown_radios` (l.214) sets False | Read by `setup_rxtx`/`setup_rxtx_and_phase_calibration`/`plot_recv_signal`. A plain `bool` read/write — torn-read not a concern in CPython; cooperative shutdown is best-effort, not prompt. ✅ |
| `ThreadedRX.run` (per-instance bool) | reader thread + `run_collector_thread` (l.495) | Cross-thread flag set by the collector to stop readers; plain bool, GIL-atomic. ✅ |
| `read_q` (`queue.Queue(maxsize=1)`) | producer reader / consumer collector | The synchronization primitive — thread-safe by construction. maxsize=1 means **back-pressure, not drop**: a slow consumer stalls the producer; a slow producer stalls the consumer's `.get()`. ✅ |
| each driver's `self.sdr` | only that radio's reader thread | One reader per radio, so no two threads touch the same `self.sdr`. ✅ The collector thread never touches hardware. |

**Dropped-buffer / queue-overflow handling.** There is **no explicit drop logic** anywhere. The
maxsize=1 queue converts overflow into back-pressure (`read_forever` retries `put` with timeout until
the slot frees; `run_inner_collector_thread` blocks on `get`). Stale-sample avoidance is instead done
at the **hardware** layer: `setup_rx` reads-and-drops 10 buffers, `setup_rxtx` 400, and Pluto's kernel
buffer count is set low (`rx_buffers`, default 2–4) "to avoid stale data on Pluto"
(sdr_controller.py:772–774). The `soft_reset_radio` periodic "recalibrate" that *might* have flushed
the radio mid-run is a **no-op on Pluto** (early-return), so on real Pluto hardware there is no
in-loop flush — only the low kernel-buffer count guards against staleness. (✅ verified.)

**Multiprocessing path (latent).** `start_read_thread(thread=False)` and the unused
`ProcessPoolExecutorWithQueueSizeLimit` (l.41) hint at a process-based variant, but the live path is
all threads (`start_read_thread()` called with the default, l.428). The process variant shares a plain
`queue.Queue` that wouldn't cross the process boundary → 🧪 latent/broken if ever enabled.

---

## Tests & coverage

| Test | What it exercises | Gap |
|---|---|---|
| `tests/test_mavlink_radio_collect.py::test_mavlink_radio_collect` | **End-to-end fake path:** subprocess runs `mavlink_radio_collection.py --fake-drone`, V4, `fake-pluto` URIs → `radios_to_online`→`setup_rx`→`ThreadedRXRawV4`→`DroneDataCollectorRaw`→zarr. Asserts all `v4rx_f64_keys` (incl. `heading`) finite. | Fake only; never touches `PPlus`/`BladeRFSdr` real hardware, phase-inversion regs, `setup_rxtx`, or the retry/abort paths. |
| `tests/test_fake_data.py` | `create_fake_dataset` (synthetic zarr) — **does not** go through `DataCollector`/`ThreadedRX`. | No live-capture coverage. |
| `tests/test_rf.py` | rf.py math only (`circular_mean`, beamformer, percentile…). | Not capture path. |
| `tests/test_throughput.py` (sdrpluto) | misnamed standalone CLI, not pytest. | — |

**No direct unit test** covers: `get_pplus` dispatch/registry, `PPlus`/`BladeRFSdr` setup or phase
calibration, `setup_rxtx`, `setup_rxtx_and_phase_calibration` (#6), `ThreadedRX.get_rx` retry/abort,
the `read_q` back-pressure, or the V4 `heading` units. The threading and hardware logic is validated
only transitively through the single fake integration test.

---

## Cluster summary

| Symbol | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `get_pplus` | ✅ Active | ~6 | ❌ (fake integ only) | "Not thread safe" but live path calls it single-threaded pre-spawn; `set_config` re-config asserts. |
| `PPlus.setup_rx_config` | ✅ Active | 2 (setup_rx/rxtx) | ❌ | AD9361 phase-inversion cal (debug attr + reg 0x22 bit6), assert-guarded. |
| `PPlus.soft_reset_radio` | ⚠️ no-op | — | ❌ | Early `return`; in-loop "recalibrate" is a no-op on Pluto. |
| `BladeRFSdr.setup_rx_config` | ✅ Active | 1 | ❌ | RFIC reg 0x22/0x11/0x189 phase-inversion; `rssis().mean(axis=1)` 🔍 may crash on scalar rssi; `gains` stubbed [0,0]. |
| `FakePPlus.get_rssi_and_gain` | 🔴 dead | 0 | ❌ | #14 confirmed: undefined `self.dev`; **unreached**. |
| `setup_rx` | ✅ Active | 2 | ✅ (fake) | Live RX bring-up; never returns None (caller None-guard dead); zeros phase_calibration. |
| `setup_rxtx` | ✅ Active | 2 | ❌ | SDR-emitter bring-up; `(None,None)` return un-guarded at radios_to_online:388. |
| `setup_rxtx_and_phase_calibration` | 🔴 Broken | 1 (CLI rxcal) | ❌ | #6 confirmed: `.setup_rx()` missing; + `retries` never incremented (infinite loop). Off live path. |
| `ThreadedRX.read_forever` | ✅ Active | core | ✅ (fake) | maxsize=1 back-pressure (no drop); 100-sample recalibrate no-op on Pluto; rate-limit EWMA. |
| `ThreadedRX.get_rx` | ✅ Active | core | ✅ (fake) | `sys.exit(1)` is dead (off-by-one); returns None; Raw path doesn't null-check → TypeError, still aborts thread. |
| `radios_to_online` | ✅ Active | 2 | ✅ (fake) | Live entry; requires `skip_phase_calibration`; single-threaded radio open. |
| `DroneDataCollectorRaw.write_to_record_matrix` | ✅ Active | core (V4) | ✅ (fake) | #16 confirmed: writes dynamic `data.heading` (deg) under zarr key `heading`; `rx_heading_in_pis` unused in V4. No crash; units load-bearing. |

**Cross-cutting theme.** The live capture path is a back-pressured single-producer/single-consumer
pipeline per radio whose thread-safety holds because the genuinely-unsafe global state
(`pplus_online`) is only mutated single-threaded during bring-up, before any reader thread exists. The
recurring hazard class is **stubbed/dead safety machinery that looks active**: `soft_reset_radio`
no-op (so no in-loop flush on Pluto), the unreachable `sys.exit(1)`, the dead None-guards after
`setup_rx`, BladeRF `gains`/`check_for_freq_peak` stubs, and the `--mode rxcal` calibration path that
crashes on the first call (#6). The most subtle is the V4 `heading` (degrees) vs declared
`rx_heading_in_pis` (π units) split (#16) — internally consistent and bridged by the v4→v5
converter's `/360*2`, so it is **silently-units-divergent, not crashing, and dangerous to "fix."**

### NEW findings not currently in KNOWN_ISSUES.md (do not edit that file)
- `ThreadedRX.get_rx` `sys.exit(1)` (data_collector.py:261) is **dead** (off-by-one
  `tries > max_retries` after `while tries < max_retries`) — returns `None` instead; `ThreadedRXRaw.get_data`
  (l.283) **doesn't null-check** that None → `TypeError` at l.288. Net effect = thread aborts cleanly,
  but the intended hard exit can't fire. P3, ✅ verified.
- `setup_rxtx_and_phase_calibration` `retries` is never incremented (sdr_controller.py:1001–1048) →
  infinite calibration loop if std never converges. P2 (off live path, gated behind #6 crash). 🔍 analysis.
- `radios_to_online` does `pplus_rx.close_rx()` (data_collector.py:388) with **no None-guard** on the
  `setup_rxtx` return → `AttributeError` if the SDR emitter fails to verify within 15 retries. P2,
  hardware-only. 🔍 analysis.
- `BladeRFSdr.rssis().mean(axis=1)` (sdr_controller.py:319) `AxisError` **iff** `ch.rssi` is scalar —
  binding-dependent, unverified. P3, hardware-only. 🔍 analysis.
- `BladeRFSdr.gains` returns hardcoded `[0,0]` (l.323) and `check_for_freq_peak` always True (l.488) —
  intentional stubs, silently-wrong-ish. P3, ✅ verified.
- `PPlus.soft_reset_radio` early-return (sdr_controller.py:634) makes the 100-sample in-loop
  "recalibrate" a **no-op on Pluto**. P3, ✅ verified.
