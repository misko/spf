# P0 Issue #6 — `setup_rxtx_and_phase_calibration` calls non-existent `pplus_rx.setup_rx()`

**File:** `spf/sdrpluto/sdr_controller.py`
**Verdict: DOWNGRADE (real bug, but DEAD/standalone path — not P0).**

---

## 1. CONFIRM — the AttributeError is real

Lines 970–982:

```python
970    pplus_rx = get_pplus(rx_config=rx_config)
971    pplus_rx.setup_rx()                # <-- method call
...
981        pplus_rx = get_pplus(rx_config=rx_config)
982        pplus_rx.setup_rx()            # <-- method call
```

Both call `pplus_rx.setup_rx(...)` as a **method**.

`get_pplus(...)` returns a `PPlus` (class at :595) or `FakePPlus` (class at :502).
Method definitions found via grep:

| Class       | line | methods related to rx setup            |
|-------------|------|-----------------------------------------|
| `FakePPlus` | 502  | `setup_rx_config` (:579) — **no `setup_rx`** |
| `PPlus`     | 595  | `setup_rx_config` (:703) — **no `setup_rx`** |
| (an older class) | — | `setup_rx_config` (:352) |

There **is** a module-level function `setup_rx(rx_config, provided_pplus_rx=None)` at **:885** — a plain function, NOT a method. So `pplus_rx.setup_rx()` resolves no attribute on the instance.

**Conclusion:** `pplus_rx.setup_rx()` would raise `AttributeError: 'PPlus' object has no attribute 'setup_rx'` (same for `FakePPlus`). The intended call is almost certainly `pplus_rx.setup_rx_config()` (cf. the sibling functions `setup_rx`/`setup_rxtx` at :891 / :909 which correctly call `pplus_rx.setup_rx_config()`).

## 2. REACHABILITY — caller chain

Only ONE call site exists (grep across whole repo, `.py`):

- `spf/sdrpluto/sdr_controller.py:1256` — inside `if __name__ == "__main__":` (guard at :1153), under `if args.mode == "rxcal":` (:1252).

The function is **never imported anywhere** — grep for `setup_rxtx_and_phase_calibration` returns only its definition (:959) and that single in-module call (:1256). It is not in `data_collector.py`'s import list.

The **live data-collection path does NOT use it**:
- `grbl_radio_collection.py:179` and `mavlink_radio_collection.py:305` → `data_collector.radios_to_online()`.
- `data_collector.radios_to_online()` (:340) calls only the module-level `setup_rxtx` (:383) and `setup_rx` (:402) functions — both of which correctly call `pplus_rx.setup_rx_config()`. It never touches `setup_rxtx_and_phase_calibration`.

So the buggy function is reachable ONLY by running `python sdr_controller.py --mode rxcal ...` directly. It is a **standalone CLI calibration utility**, off the normal collection path.

## 3. TRIGGER

Runs only on the explicit standalone invocation `sdr_controller.py --mode rxcal`. It is NOT part of every capture and NOT part of normal `grbl_`/`mavlink_` collection. Normal collection avoids it entirely.

## 4. BLAST RADIUS

If someone runs the `rxcal` standalone mode, it crashes immediately at the first `pplus_rx.setup_rx()` (:971) — a hard `AttributeError` before any calibration/data is produced. Requires real SDR hardware to be reached at all (`get_pplus` opens a device).

CI/tests: **not reached.** Although `FakePPlus` also lacks `setup_rx`, nothing in tests imports or calls `setup_rxtx_and_phase_calibration`, and the `rxcal` branch is behind a `__main__` guard. No test would surface this.

## 5. REAL-WORLD / RECENCY

- No test or script references the function (grep clean outside the file).
- The buggy lines date to commit `9d00b7b` (2024-01-26). The file has since been touched (`ceec659` 2025-03-20) without fixing it — strong evidence this calibration entrypoint is not in active use; an active path would have crashed on first run.

## 6. SEVERITY VERDICT — DOWNGRADE

It is a **genuine bug** (guaranteed AttributeError if invoked), but:
- reachable only via a standalone `--mode rxcal` CLI entrypoint,
- never imported, never used by normal `grbl`/`mavlink` collection,
- never exercised in CI/tests,
- dormant in-tree since 2024 with no crash reports surfacing on the live path.

A P0 ("hard crash before any data collected on the live path") does not apply: the live collection path uses the correct module-level `setup_rx`/`setup_rxtx`. This is a **dead/standalone-utility bug → P2/P3**, fix-when-touched. (Keep it fixable since the `rxcal` utility is the intended way to compute phase calibration if anyone resurrects it.)

## 7. FIX SKETCH (do NOT apply)

In `setup_rxtx_and_phase_calibration`, replace both method calls with the existing instance method:

- :971 `pplus_rx.setup_rx()` → `pplus_rx.setup_rx_config()`
- :982 `pplus_rx.setup_rx()` → `pplus_rx.setup_rx_config()`

(Matches the correct usage in the neighboring `setup_rx`/`setup_rxtx` functions and is defined on both `PPlus` and `FakePPlus`.) Two-line change.
