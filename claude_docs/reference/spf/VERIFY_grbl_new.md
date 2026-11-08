# Adversarial Verification — GRBL new claims

Source pinned at working tree HEAD (`574c454` per doc). Read-only verification of three
claims in `grbl_and_planners.md`. Files inspected:
`spf/grbl/grbl_interactive.py`, `spf/motion_planners/planner.py`,
`spf/motion_planners/dynamics.py`, `spf/grbl_radio_collection.py`, `spf/data_collector.py`.

---

## Claim 1 — `to_steps` OOB raise swallowed by `run_planner`'s bare `except` → silent stop while collection records stale positions

**VERDICT: CONFIRMED. Severity: HIGH. Confidence: HIGH.**

Exact exception path traced line-by-line:

1. **`to_steps` raises on OOB.** `grbl_interactive.py:179` and `:196` —
   `raise PointOutOfBoundsException(...)` when `not self.unsafe and self.polygon is not None
   and not self.polygon.contains_point(...)` (raw target at 174-181, re-inverted point at
   191-196). Verified numerically: `to_steps([3500,2700])` raises (`OOB raw`).

2. **`move_to` does NOT guard `to_steps`.** `move_to` (369-391) calls
   `a_motor_steps, b_motor_steps = self.dynamics.to_steps(points[c])` at **line 376**, which is
   **outside** the only `try/except` in the method — that `try` (382-386) wraps **only**
   `self.s.write(...)`. So a `PointOutOfBoundsException` from line 376 propagates straight out of
   `move_to`. (Quoted: the `try:` is at 382, after the `to_steps` call at 376.)

3. **`move_to_iter` does not catch it.** `move_to_iter` (421-434) calls `self.move_to(next_points)`
   at line 428 with no `try`. Propagates.

4. **`run` does not catch it.** `run` (526-532) calls `self.controller.move_to_iter(...)` at line
   532 with no `try`. Propagates.

5. **`run_planner`'s bare `except Exception` swallows it.** `run_planner` (473-496): the call to
   `self.run()` is at line 490 inside `try:` (483) … `except Exception as e: logging.error(e)`
   (493-494). The except merely **logs**; control falls to line 495-496 (`time.sleep(10)`), the
   thread function returns, and **the daemon thread dies**. No re-raise, no `run_grbl=False`, no
   signal to the collector.

**Does collection keep recording stale positions? YES.** Verified the decoupling:

- `grbl_radio_collection.grbl_radio_main` (160-188): `gm.start()` spawns the `run_planner` daemon
  thread (`GRBLManager.__init__` 468, `start` 498-499); the **main thread** independently runs
  `data_collector.start()` then `while data_collector.is_collecting(): time.sleep(5)` (186-188).
  The two are separate threads with no shared liveness check.
- The GRBL data collectors read position **directly from the cached dict**, never re-polling:
  `GrblDataCollectorRaw.write_to_record_matrix` reads
  `self.position_controller.controller.position["xy"][...]` (data_collector.py:594-599); same in
  `GrblDataCollector` (646-651). No `update_status` call on the collection path.
- `position["xy"]` is **only** refreshed by `update_status` (grbl_interactive.py:345-353), and in a
  GRBL collection `update_status` is **only** invoked from inside the `run_planner` thread's motion
  loop (`move_to_iter`→`targets_far_out`→`distance_to_targets`→`update_status`, 429/414/408/316;
  plus `get_ready` 523). grep confirms no other live caller during collection
  (`get_position_bearing_and_time` at data_collector.py:538 is the **mavlink/drone** subclass, not
  GRBL).

Therefore, once the `run_planner` thread dies, `position["xy"]` **freezes at the last value**, and
every subsequent record written by the still-running collector is stamped with that stale
position — a **silently-wrong DATA corruption**, not just a stalled motor. `assert data.rx_lo > 1`
(605) is the only assertion in the writer and does not touch position, so nothing aborts the run.

**Reachability — is the OOB raise itself reachable in a real collection?** This is the one genuine
caveat. The raise requires a *planner-produced* point landing outside the polygon by more than the
`radius=1e-5` slack. Planner geometry is designed to stay in-bounds:
`binary_search_edge` (dynamics.py:81-91) catches `PointOutOfBoundsException` and bisects **inward**,
so bounce targets are pulled to the in-bounds side of the edge; the `1e-5` radius adds slack. So the
common case stays in bounds. But the path is *plausibly* reachable: `single_bounce` adds 5 % random
noise to the new direction (planner.py:59-63) and `bounce` adds another 5 % (75-77) **after** the
bisected bounce point is computed — the yielded `to_points` themselves come from the bisected
(in-bounds) segment, but `CirclePlanner` emits `angle_to_pos` points (planner.py:194-197) computed
purely from `circle_center`/`circle_radius` with **no bounds bisection at all** — if a configured
circle (radius 950 about `[2000,1500]`) grazes outside the polygon at any angle, `to_steps` raises
and the whole circle routine dies silently. The doc's "any planner that grazes the boundary"
framing is fair; circle routines are the most exposed because they never bounds-clamp their points.

Net: the swallow + stale-recording mechanism is **definitely real and the data-corruption
consequence is certain once triggered**; the trigger frequency depends on config geometry (circle
routines most at risk). HIGH severity stands — silent dataset poisoning with no operator signal.

---

## Claim 2 — `update_status` unbounded recursion on chatty/`ok`/malformed serial → `RecursionError`

**VERDICT: CONFIRMED. Severity: MEDIUM. Confidence: HIGH (by inspection; not reproduced on hardware).**

`update_status` (316-340): after `response = self.s.readline().decode().strip()` (326), it tries
`motor_position_str = response.split("|")[1]` (329). On `IndexError` (no `|` in the line) the
`except` (330-334) does:
- `if response.strip() == "ok" or response.strip() == "": return self.update_status(skip_write=not skip_write)` (331-332)
- else: `logging.warning(...); return self.update_status(skip_write=not skip_write)` (333-334)

**No depth cap, no retry counter, no max-iterations parameter** anywhere in the method or its
callers. `skip_write` only **toggles** write/no-write each level (so it re-prompts `?` on alternate
frames) — it does not bound recursion. Any device that emits a sustained stream of lines without a
`|` (firmware push messages `[MSG:...]`, `ALARM:n`, `error:n`, banner text, or just repeated `ok`)
adds one Python stack frame per line until the interpreter's recursion limit (~1000) → uncaught
`RecursionError`. Trigger confirmed; the branch at 334 catches **every** non-`MPos`, non-`ok`,
non-blank line, so even a single permanently-malformed `MPos` (e.g. a line with no `|`) recurses
forever.

**Reachable on real serial only — fake GRBL cannot trigger it. CONFIRMED.** The fake GRBL is
strictly request/response coupled through queues (`_run_fake_grbl` 296-310): each command enqueued
by a `write` yields exactly one response. A `?` always produces
`<Idle|WPos:...|FS:0,0|Ov:100,100,100>` (301-304), which contains `|`, so `split("|")[1]`
succeeds and never recurses. The fake emits no unsolicited chatter and never streams `ok` to a
`?`. (Minor correction to the doc's wording: it is not that "`OK` is consumed by the response
queue" so much as that the fake's response to `?` is *always* a well-formed status line; the
conclusion — untriggerable in tests — is correct.) So this is a real-hardware-only latent hazard,
untested by the suite (test_grbl.py drives only the fake). MEDIUM: real and uncaught, but requires
a misbehaving/chatty physical GRBL; not hit on the happy path.

---

## Claim 3 — `from_steps` positive-y-only branch mislabels sub-motor-line points, but is DEAD for shipped configs

**VERDICT: CONFIRMED (real bug, math correct; dead/unreachable in production). Severity: LOW. Confidence: HIGH.**

Math confirmed by reading and by numeric re-derivation:
- `from_steps` (144-171): `y = np.sqrt(r1*r1 - x*x)` (165) — the non-negative square root, then
  `self.pA + y_dir*y + x_dir*x` (168-169) with `y_dir=[-x_dir[1],x_dir[0]]`. The sign of `y` is
  **discarded**: any belt-length pair that geometrically corresponds to a point below the A–B motor
  line (`y<0`) is reflected to `+y`. Verified: feeding the belt lengths for `[2000,-1500]` returns
  `[2000, 1500]` — exact reflection across the motor line.

**Dead for shipped configs — CONFIRMED on both sub-claims:**
1. *Workspace is strictly y>0.* `home_bounding_box` y-values are `{400,450,500,1900,2700,2700}` —
   all ≥ 400 (verified). The motor line is `y=0` (`pA=[3568,0]`, `pB=[0,0]`). No in-bounds point
   has `y<0`.
2. *`to_steps` double-checks and rejects out-of-workspace targets first.* `to_steps` bounds-checks
   the raw target (174-181) **and** the re-inverted `from_steps` point (190-196) against the
   polygon; verified `to_steps([3500,2700])` raises. Since the only producer of motor steps for
   motion is `to_steps` (via `move_to`), and it refuses any point not inside the y>0 polygon, the
   belt lengths ever fed to `from_steps` in production always correspond to y>0 points, where the
   sign discard is a no-op (the correct branch *is* +y).

So the positive-y branch is a genuine latent correctness defect that is **unreachable for all
shipped GRBL/GPS configs** (one-sided, convex, y>0 boundaries; `to_steps` guards the entry).
test_grbl round-trip only covers in-bounds y>0 (doc's coverage table agrees). LOW severity:
real-but-dead. (Adjacent note, not part of this claim: the same line 165 `np.sqrt(r1²−x²)` is
unguarded → `NaN` on a malformed 4-float `MPos` in `update_status`; that NaN *would* flow into
`position["xy"]` and the dataset, but only on corrupt serial — separate from the y-sign issue.)

---

## Summary table

| Claim | Verdict | Real? | Trigger | Live-config reachable? | Severity |
|---|---|---|---|---|---|
| 1. swallowed OOB raise → silent stop + stale recording | CONFIRMED | yes | `to_steps` raises on a planner point outside polygon (circle routines most exposed; bounce is bisected inward) | yes — and once triggered the stale-position recording is **certain** | HIGH |
| 2. `update_status` unbounded recursion | CONFIRMED | yes | sustained non-`MPos` serial lines (`ok`/chatter/malformed) | real serial only (fake GRBL 1:1 coupled, cannot) | MEDIUM |
| 3. `from_steps` +y-only reflection | CONFIRMED real, DEAD | yes (math) | belt lengths for a `y<0` point | NO — workspace all y>0, `to_steps` guards entry | LOW |

**Bottom line on the most consequential claim (#1):** the swallowed-exception silent-stop is
genuinely reachable in a real collection. The swallow chain (376 raise → unguarded through
move_to/move_to_iter/run → bare `except` at 493 → daemon thread dies) is verified exactly as
described, and the collector's direct read of the frozen `controller.position["xy"]` cache
(data_collector.py:594-599/646-651) means a dead motion thread produces a dataset full of records
all stamped with the same last-known position, with **no error surfaced** to the operator. The only
thing gating it is how often a planner emits an out-of-polygon point — low for bounce (bisected
in-bounds), but unguarded for circle routines, so it is a live, silently-wrong data-integrity risk.
