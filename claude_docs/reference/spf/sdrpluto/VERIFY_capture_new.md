# Adversarial Verification — NEW capture-path claims

Read-only verification. Source pinned at current `main` (HEAD 574c454). Line numbers cite the live files
`spf/data_collector.py` and `spf/sdrpluto/sdr_controller.py`.

---

## Claim #1 — `ThreadedRX.get_rx` dead `sys.exit(1)` (off-by-one) → returns None; `ThreadedRXRaw.get_data` no null-check → `TypeError`

**VERDICT: CONFIRMED.** Severity **P3 (low)**. Reachable on real hardware (V4/V5 live paths).

Exact lines (`data_collector.py`):
```
245    def get_rx(self, max_retries=15) -> Dict[str, Any]:
246        tries = 0
247        while tries < max_retries:
...
257                time.sleep(0.1)
258                tries += 1
259                if tries > max_retries:
260                    logging.error("GIVE UP")
261                    sys.exit(1)
262        return None
```

- Loop bound is `while tries < max_retries` (l.247) and the guard is `tries > max_retries` (l.259).
  On the 15th failing iteration `tries` goes 14→15 (l.258); `15 > 15` is False (l.259), so `sys.exit(1)`
  (l.261) is **never executed**; then `while 15 < 15` ends the loop and `return None` (l.262) fires.
  The `sys.exit(1)` is **dead code** — exactly as claimed. (Note: the original doc cited "~:261"; the
  exact line is 261. ✅ matches.)
- Downstream split is real:
  - `ThreadedRX.get_data` (l.264) **does** null-check: `if sdr_rx is None: raise ValueError(...)` (l.266-267).
  - `ThreadedRXRaw.get_data` (l.283) **does NOT** null-check. Line 288: `np.vstack(sdr_rx["signal_matrix"])`
    on `sdr_rx=None` → `TypeError: 'NoneType' object is not subscriptable`. ✅ confirmed (doc cited ~:288 = exact).
- **Reachability on live hardware: YES.** `DroneDataCollectorRaw` → `ThreadedRXRawV4` (l.510) and
  `GrblDataCollectorRaw` → `ThreadedRXRawV5` (l.568); both subclass `ThreadedRXRaw`, so the live V4 (drone)
  and V5 (wall/grbl) capture paths use the **un-null-checked** `get_data`. Only the legacy V2
  `GrblDataCollector`→`ThreadedRX` (l.628) takes the checked branch. So 15 consecutive real RX-read
  failures on Pluto/BladeRF during a live collection hit the `TypeError`, not `sys.exit(1)`.
- Net effect is benign-ish: `read_forever`'s blanket `except` (l.198-204) catches the `TypeError`/`ValueError`,
  queues `None`, sets `run=False`, and the thread aborts cleanly. So both the intended `sys.exit(1)` and the
  accidental `TypeError` end the thread — the practical outcome is identical; only the *intended hard library
  exit* is unreachable. **silently-different-from-intended, not a corruption.**

---

## Claim #2 — `setup_rxtx_and_phase_calibration` never increments `retries` → infinite loop; but gated behind #6 `AttributeError`

**VERDICT: CONFIRMED (both parts).** Severity **P2 latent**, but **NOT reachable** because #6 crashes first.

Exact lines (`sdr_controller.py`):
```
970    pplus_rx = get_pplus(rx_config=rx_config)
971    pplus_rx.setup_rx()              # <-- AttributeError: no class defines setup_rx
...
1001    retries = 0
1002    while run_radios and retries < 20:
...
1036        if phase_calibration_std < tolerance:
...
1048            return pplus_rx, pplus_tx
1049    pplus_tx.close()
1050    logging.error(f"{rx_config.uri}: Phase calibration failed")
1051    return None, None
```

- **`retries` never incremented:** confirmed. Between `retries = 0` (l.1001) and the loop body
  (l.1002-1048) there is no `retries += 1` anywhere. grep for `retries` in this function shows only the
  init (l.1001) and the read in the `while` condition (l.1002). The only loop exit is the success
  `return` (l.1048) or `run_radios` flipping False. If `phase_calibration_std >= tolerance` on every pass,
  the loop spins forever; the `< 20` bound is dead. The "failed" return (l.1049-1051) is only reached when
  `run_radios` goes False. ✅ confirmed.
- **Gated behind #6:** confirmed. `pplus_rx.setup_rx()` at l.971 (and l.982) call a method that does not
  exist — `grep "def setup_rx_config" → l.352/579/703` (PPlus/BladeRFSdr/FakePPlus), and `def setup_rx(` only
  exists as the **module-level function** at l.885, never as an instance method. So l.971 raises
  `AttributeError` **before** the loop at l.1002 is ever entered. The infinite loop is therefore
  **unreachable as written** — it would only manifest if #6 were fixed first.
- **Live-path reachability: NONE.** Sole caller is the standalone `--mode rxcal` CLI; not on the
  `radios_to_online` collection path. Hardware + CLI-only, and crashes immediately. So the infinite loop is
  a **double-latent** bug (gated behind a crash, on a non-live path).

---

## Claim #3 — `radios_to_online` calls `pplus_rx.close_rx()` with no None-guard → `AttributeError`

**VERDICT: CONFIRMED.** Severity **P2**, hardware-only (SDR-emitter configs only).

Exact lines:
- `setup_rxtx` returns `(None, None)` on retry exhaustion: `sdr_controller.py:950 → return None, None`
  (after `while run_radios and retries < 15`, l.917).
- `data_collector.py`:
```
383            pplus_rx, _ = setup_rxtx(
384                rx_config=target_rx_config,
385                tx_config=target_tx_config,
386                leave_tx_on=True,
387            )
388            pplus_rx.close_rx()
```
  If `setup_rxtx` returned `(None, None)`, `pplus_rx is None` and l.388 does `None.close_rx()` →
  `AttributeError`. There is **no None-guard** here, unlike the receiver loop which guards at l.404
  (`if pplus_rx is None: ... sys.exit(1)`). ✅ confirmed — the asymmetry is real.
- **Reachability:** Only when `emitter.type == "sdr"` (l.342/352) — the co-located-SDR-emitter bring-up.
  In the common drone/wall RX-only configs the emitter is *not* type "sdr" ("its already blasting",
  l.401 comment) so this whole block is skipped. Triggered only if a real SDR emitter fails peak
  verification within 15 retries. Hardware + SDR-emitter-config-only, never on the fake test path.

---

## Claim #16 (RE-CONFIRM) — V4 `data.heading` (degrees) vs declared `rx_heading_in_pis` (π units): consistent, not corrupting?

**VERDICT: CONFIRMED — benign-but-consistent (no silent corruption on the V4 path).** Severity **P3
(silently-units-divergent, no crash).**

Evidence chain:
1. `DataSnapshotV4` (`data_collector.py:83`) inherits from `DataSnapshotRaw` and declares only
   `gps_timestamp/gps_lat/gps_long`. It has **no `heading` field**; it inherits
   `rx_heading_in_pis: float = 0.0` (l.79). ✅
2. `DroneDataCollectorRaw.write_to_record_matrix` (l.540): `data.heading = current_pos_heading_and_time["heading"]`
   — sets a **dynamic attribute** in **degrees** (mavlink `hdg/100`, 0-360). ✅
3. Zarr write loop (l.551-552): iterates `v4rx_f64_keys + v4rx_2xf64_keys` and does `getattr(data, k)`.
   `v4rx_f64_keys` contains `"heading"` (`v4_data.py:8`), **not** `"rx_heading_in_pis"`. So
   `getattr(data, "heading")` resolves the dynamic attr → **succeeds, no crash, no missing key**. The
   declared `rx_heading_in_pis` is never in the key list, so it is never written (stays the default 0.0). ✅
4. v4→v5 conversion (`spf/scripts/v4_tx_rx_to_v5.py:172-175`):
   ```
   172    if src_key == "rx_heading_in_pis":
   173        src_key = "heading"
   175        _src = (src[src_key][:] / 360) * 2
   ```
   When the V5 destination wants `rx_heading_in_pis`, the converter reads the V4 src key `heading`
   (degrees) and computes `(heading/360)*2` → π units. The bridge is **explicit and correct**. ✅

**Conclusion on #16: BENIGN-BUT-CONSISTENT, not a corruption.**
- The V4 on-disk format is internally consistent: degrees stored under zarr key `heading`; `rx_heading_in_pis`
  declared on the dataclass but never populated/written in V4.
- The rest of the codebase consumes `rx_heading_in_pis` (π units); only the V4 raw on-disk format uses
  `heading` (degrees), and the converter bridges via `/360*2`.
- The mismatch is **load-bearing**: "fixing" the dataclass to write `rx_heading_in_pis` directly would change
  the V4 on-disk units (deg→π) and silently break the converter's `/360*2` and the
  `test_mavlink_radio_collect` key check. Do **not** "fix" without auditing downstream.
- No silent corruption exists on the V4 write path. (The note in the original doc claiming the path is
  internally consistent is itself CONFIRMED.)

---

## Summary table

| Claim | Verdict | Real? | Trigger | Live-path reachable? | Severity |
|---|---|---|---|---|---|
| #1 dead `sys.exit(1)` → None → `TypeError` (Raw) | CONFIRMED | yes | 15 consecutive RX read failures | YES (V4/V5 hardware) | P3 low |
| #2 `retries` never incremented (infinite loop) | CONFIRMED | yes | std never converges | NO — gated behind #6 crash; `--mode rxcal` only | P2 double-latent |
| #3 `close_rx()` no None-guard → `AttributeError` | CONFIRMED | yes | SDR emitter fails peak in 15 retries | only SDR-emitter configs (hardware) | P2 |
| #16 V4 heading deg vs `rx_heading_in_pis` π | CONFIRMED — benign/consistent | yes | n/a (no crash) | live V4 path, no corruption | P3 units-divergent |

**No claim refuted. No silently-wrong corruption found on the V4 path** — #16 is a consistent, bridged
units split, not a data bug.
