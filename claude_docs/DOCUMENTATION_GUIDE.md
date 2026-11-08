# How to Document Code (SPF Standard)

This is the **method** for producing trustworthy code documentation in this repo. It is a
standard operating procedure, not a suggestion: documentation is written **only after** the code
is genuinely understood, and understanding is reached through **multiple disciplined passes**.

The artifacts this process produces (per-function/per-module reference docs) are the raw material
that later gets stitched into the high-level overviews in [`README.md`](README.md). Get the
function-level truth right first; the big picture assembles itself from accurate parts.

---

## First principles

1. **Understanding precedes writing.** Documentation is a *byproduct* of understanding the code,
   never a paraphrase of it. If you cannot state a function's contract without re-reading it, you
   are not done investigating.
2. **At least three passes before a word of prose.** Surface → contracts → internals/tests. You
   do not write documentation in pass 1.
3. **Read-only.** Do **not** change code while documenting. Suspected bugs are *recorded with
   evidence*, never fixed in this workflow. Fixes are a separate, later decision.
4. **Evidence or it didn't happen.** Every non-trivial claim cites `file:line`. Claims sourced
   from an LLM/agent/teammate are **verified against source** before they enter a doc. (We have
   already caught fabricated claims this way.)
5. **State uncertainty explicitly.** Mark anything unconfirmed as `⚠️ unverified` /
   `assumed` / `needs confirmation`. A known gap is fine; a confident wrong claim is not.
6. **Units and frames are never implicit.** Angles, units, and coordinate frames are stated and
   linked to [`00_concepts/conventions.md`](00_concepts/conventions.md). This is the #1 bug source.

---

## Scope: what is a "unit"

The atomic unit of documentation is a **function or method**. Documents are grouped **by module**
(one reference file per source module; see *Artifacts* below). Within a module, functions are
documented in **fan-in order** — most-called / public entry points first, leaf helpers last.

**Do not try to document everything at once.** Prioritize:

1. Public API / entry points (things imported elsewhere, CLI mains, exported classes).
2. Load-bearing internals (high fan-in — discovered in Pass 2).
3. Leaf helpers (document briefly, or by reference, once the above are done).

---

## The four passes

| Pass | Name | Goal | Writes prose? |
|---|---|---|---|
| 1 | Surface inventory | Capture signature-level truth: inputs, outputs, side effects | No (notes only) |
| 2 | Call graph & contracts | Where it's called, where returns flow, and the resulting contract | No (notes only) |
| 3 | Internals, correctness & tests | Verify the body honors the contract; map test coverage | No (notes only) |
| 4 | Synthesize & write | Produce the doc from the fixed template | **Yes** |

Each pass has an explicit **exit checklist**. Do not advance until it's met.

---

### Pass 1 — Surface inventory

**Goal:** for every in-scope function, record what it *appears* to take, return, and disturb —
from the signature and a single read of the body.

**Do:**
- Locate it: `file:line`, plus its module and the class it belongs to (if any).
- Write a one-line purpose (provisional).
- **Inputs:** each parameter — name, type (declared or inferred), meaning, units/frame, default.
- **Outputs:** return type/shape, meaning, units/frame.
- **Side effects:** anything that is *not* the return value —
  - mutation of arguments or `self`,
  - global / module / file / network / DB / GPU / hardware state,
  - I/O (prints, logs, writes, plots),
  - exceptions raised, `breakpoint()`/`assert`,
  - threads/processes spawned, randomness, time dependence.

**Exit checklist:**
- [ ] Every in-scope function has a stub with inputs, outputs, and side effects filled (deeper
      fields may say `TBD`).
- [ ] Any side effect that is surprising for the name is flagged for Pass 3.

---

### Pass 2 — Call graph & contracts

**Goal:** understand how the function is actually *used*, and derive its contract from real call
sites — not from wishful reading of the signature.

**Do:**
- **Find every call site** across the repo *and* tests (`grep`/search the symbol; include
  indirect references, partials, dispatch tables, config-string registries).
- For each call site record: **where it's called from** (caller `file:line`) and **where the
  return flows to** ("returns to" — what the caller does with the result next).
- **Check transitive reachability.** A direct call site is not proof of life: confirm each caller
  is *itself* reachable (has its own callers / is an entry point / a test). A function whose only
  caller is itself uncalled is **effectively dead** — flag it 🧪. (This is how the pilot caught
  `phase_diff_to_theta`: one caller, and that caller had none.)
- At each call site, read enough surrounding code to extract:
  - **Input constraints (preconditions):** what callers guarantee about arguments — ranges,
    shapes, dtypes, non-null, units/frame, ordering, sortedness, "already normalized", required
    prior setup/init, lock held, device.
  - **Output constraints (postconditions):** what callers *assume* about the return — shape,
    normalization (e.g. sums to 1), non-empty, sorted, value range, never-`None`.
  - **Invariants:** relationships that hold before *and* after (between args, with `self` state,
    with global/file state); conservation/identity properties.
- **Synthesize** the per-call-site observations into one contract: the *union* of preconditions
  callers rely on, the *intersection* the function must always satisfy. Note disagreements
  between call sites — they often reveal latent bugs or undocumented modes.

**Exit checklist:**
- [ ] Complete list of call sites (prod + tests) with caller location and return destination.
- [ ] A synthesized **Input constraints / Output constraints / Invariants** draft.
- [ ] Conflicting or surprising usages flagged for Pass 3.

---

### Pass 3 — Internals, correctness & tests

**Goal:** confirm the implementation actually honors the contract from Pass 2, and map what the
tests do and don't cover. **No code changes.**

**Do (correctness):**
- Read the body carefully; trace **every branch and loop** and each early return.
- For each precondition from Pass 2, confirm the body behaves; for each postcondition, confirm the
  body guarantees it on *all* paths.
- Actively hunt the usual failure modes:
  - angle/phase **wraparound** (±π), degrees vs radians, frame mix-ups,
  - off-by-one, empty input, single-element, `NaN`/`inf`, division by ~0,
  - **shape/dtype** mismatches, broadcasting surprises, float precision,
  - silent failure (swallowed exceptions, clamped-and-continue),
  - argument/`self` **mutation** the caller may not expect,
  - concurrency / shared-state / ordering hazards,
  - dead branches, contradictory asserts, leftover `breakpoint()`.
- Record every suspected issue as **evidence, not a fix**: `file:line`, what's wrong, the
  triggering condition, severity, and confidence.
- **Severity rubric** = *(does it misbehave)* × *(is the path reached)*. A `breakpoint()` /
  `pass` / `.copy()`-on-a-tensor is only P0 if a real caller or a checked-in config actually
  reaches it (use the Pass-2 transitive-reachability result). Broken-but-dead code is low
  severity. **Always tag a 🔴 claim with its reachability** ("reached via `caller`/`config`" vs
  "dead — no live caller").
- **Flag silently-wrong-result bugs distinctly from crashes.** A wrong value that does *not*
  raise (a swapped axis, a deg/rad mix) is more dangerous than a crash and more hazardous to
  "fix" — downstream code may have compensated for it. Mark these `⚠️ no-crash / verify
  downstream before any fix`, never as a routine bug.
- **The bar for asserting "broken":** would it actually raise / produce a wrong value, on a path
  that executes? If you cannot point at the line that fails *and* the caller that reaches it, it
  is a *suspicion* (🔍 analysis), not a confirmed bug.

**Do (tests):**
- Search `tests/` (and inline tests/notebooks) for the function name and its behaviors.
- Classify existing coverage: happy path / edge cases / error paths / specific modes.
- Identify **missing** coverage: which preconditions, postconditions, invariants, and edge cases
  are *not* exercised.

**Exit checklist:**
- [ ] Each contract clause is marked verified / partially verified / contradicted, with evidence.
- [ ] Issues log populated (or explicitly "none found", with what was checked).
- [ ] Test inventory + coverage-gap list complete.

---

### Pass 4 — Synthesize & write

Only now write prose, using the template below. Every claim must trace to evidence gathered in
Passes 1–3. Mark residual uncertainty. Keep it tight: the doc should let a reader use the
function correctly **without** reading its source.

---

## Output template (per function)

```markdown
### `function_name(...)`  ·  `path/to/module.py:LINE`  ·  ✅ Active

**Summary.** One short paragraph: what this function is for and why it exists. The mental model,
not the line-by-line.

**Signature**
- Inputs:
  - `arg` (type, units/frame) — meaning. Default: …
- Returns: (type/shape, units/frame) — meaning.
- Side effects: mutation / I/O / global state / exceptions / threads / randomness. ("None" is a
  valid and useful answer.)

**Called from** (Pass 2)
- `caller_a` (`file:line`) → return flows into … 
- `caller_b` (`file:line`) → …
- Tests: `tests/test_x.py:LINE`
- Fan-in: N call sites.

**Contract** (Pass 2)
- *Input constraints / preconditions:* …
- *Output constraints / postconditions:* …
- *Invariants:* …

**Tested by** (Pass 3)
- `tests/test_x.py::test_name` — covers …

**Analysis** (Pass 3)
- *Correctness:* what was verified; behaviors confirmed against the contract.
- *Issues found or considered:* `file:line` — description, trigger, severity, reachability, and a
  **confidence tier**: `✅ verified` (you read the lines that make it true) vs `🔍 analysis` (a
  reasoned race/path you did *not* execute). 🔍 items must pass an adversarial confirm before they
  are promoted to verified. (Read-only — not fixed.) Or: "None found; checked X, Y, Z."
- *Test coverage:* what the existing tests exercise.
- *Missing test coverage:* preconditions/edge cases/invariants not exercised.

**Open questions / TODO.** Anything unverified or needing a maintainer's confirmation.
```

A function with no surprises may compress several fields to one line — but every heading is
considered, even if the answer is "none."

### Module-level cluster summary

Every per-module reference file **ends with a summary table** indexing the functions it documents
— an at-a-glance map and a quick way to spot systemic gaps (e.g. "the most-depended-on functions
have the weakest tests"):

```markdown
| Function | Status | Fan-in | Direct test | Top finding |
|---|---|---|---|---|
| `foo` | ✅ Active | ~20+ | ❌ (indirect only) | High fan-in, no dedicated test. |
```

Follow it with one or two sentences naming any **cross-cutting theme** the cluster revealed.

---

## Conventions & quality bar

- **Status tags** (reuse repo-wide): ✅ Active · ⚠️ Deprecated/legacy · 🔴 Broken (verified) ·
  🧪 Experimental/unused.
- **Evidence:** `file:line` for every non-trivial claim. No `file:line`, no claim.
- **Uncertainty:** prefix unconfirmed statements with `⚠️ unverified` / `assumed`.
- **Units/frames:** always stated; link [`conventions.md`](00_concepts/conventions.md).
- **No fixes:** issues are documented, never patched, in this workflow.
- **Verify imported claims:** anything from an agent/LLM/teammate is checked against source first.

---

## Artifacts: where docs and notes live

- **Working notes** for Passes 1–3 are scratch — keep them per-function while investigating
  (a temp file or inline notes). They are *inputs* to Pass 4, not deliverables.
- **Finished reference docs** live under `claude_docs/reference/` mirroring the source tree:
  - `claude_docs/reference/spf/rf.md` documents `spf/rf.py`, one `###` block per function.
  - Split a single function into its own file only if it is very large (e.g. `beamformer`).
- The section **overviews** (`0X_*/overview.md`) link down into these reference docs; the
  reference docs are the verified substrate the overviews summarize.

---

## Parallelizing with agents (optional)

This process is expensive, so it parallelizes well — but with guardrails:

- **Pass 1** fans out cleanly: one agent per module produces the surface inventory. **Verify**
  before trusting — but verify the right thing (see Field notes: line numbers came back ~100%
  accurate; the errors were in *conclusions*, not coordinates).
- **Pass 2** also fans out (one agent per target function's call-graph), but the **synthesized
  contract is reviewed by one owner** to reconcile conflicting call sites.
- **Pass 3** (correctness) benefits from an *adversarial* second agent: one documents, an
  independent one tries to refute each contract clause and find an untested edge case.
- Always collapse agent output through a human/owner review before it becomes a finished doc.
- **Orchestration that kept context clean (worked well):** agents *write* their artifact to a file
  and *return only a short summary* (counts, grade, single biggest correction) — never paste the
  full inventory back to the orchestrator. Pin all evidence to a commit hash up front.

---

## Field notes — learnings from running this at scale

Earned by running the full process twice on this repo (a 5-group, ~650-symbol inventory wave and a
6-module deep-dive wave, each followed by an adversarial verification wave). These are the
non-obvious lessons; treat them as defaults.

1. **The dominant failure mode is the fabricated/overstated bug — not the wrong line number.**
   Across ~160 sampled citations the line-number hit rate was ~100%. Every *meaningful* verifier
   correction was semantic: a 🔴 "broken" claim that wasn't real (a fabricated `SyntaxError`, a
   fabricated `UnboundLocalError`), a "dead" function that was actually active (`rotate_dist`, 6
   import sites), or an understated side effect (an in-place mutation present in *two* branches, not
   one). **Implication:** locations are cheap to trust; conclusions are expensive. Point verification
   budget at claims with consequences (🔴 broken, 🧪 dead), not at re-checking coordinates.

2. **Verify in *both* directions.** Don't only ask "is this 🔴 real?" — also ask "is this 🧪-dead
   actually live?" The two error types are symmetric and both showed up. A "dead/unused" verdict
   needs a repo-wide grep for callers (and *transitive* reachability) exactly as much as a "broken"
   verdict needs a reproduction check.

3. **A "broken" verdict needs reachability, not just a bad-looking line.** The bugs that mattered
   mattered because they're *reached*: the cosine-scheduler crash only because a checked-in YAML
   sets `scheduler: cosine`; a missing method only because a live caller hits it. Conversely several
   genuinely broken functions are dead and therefore low priority. Severity = misbehavior ×
   reachability — always state both.

4. **Carry an explicit confidence tier on every correctness finding.** `✅ verified` (you read the
   deciding lines) vs `🔍 analysis` (a reasoned race or "this config would crash" you did not
   execute). The concurrency hazards were 🔍 until a second reader confirmed the exact lock scopes.
   Mixing the tiers is precisely how a reasoned guess gets laundered into a "known bug."

5. **The two-wave shape (draft → adversarial verify) is a standard stage for *every* artifact,**
   not just deep dives. We ran it for the inventory and the deep dives; both times the verify wave
   paid for itself by catching a confident-wrong claim that would otherwise have shipped. Ask each
   verifier for its **single biggest correction** — that one field surfaced the important deltas
   fastest, and grade A–F to make regressions visible.

6. **Reconciliation is a required step, and corrections propagate to multiple places.** A refuted
   claim must be struck everywhere it appears: the canonical issue list, the doc body, *and* the
   module cluster-summary table. Budget for a reconciliation pass after every verify wave — the
   artifacts are not done until the deltas are applied back.

7. **Keep one canonical, severity-ranked issue list** (e.g. `KNOWN_ISSUES.md`) separate from the
   per-module docs, with a confidence column. The per-module docs explain *how the code works*; the
   issue list is the actionable cross-cutting output and the thing a maintainer triages from.

8. **Pin evidence to a commit hash.** Inventories and line numbers drift the moment source changes;
   a doc that doesn't name the commit it was verified against silently rots.

## Definition of done (per function)

- [ ] Passes 1–3 exit checklists all met, with evidence.
- [ ] Doc written from the template; every claim has a `file:line` or an explicit uncertainty mark.
- [ ] Contract (inputs/outputs/invariants) stated precisely enough to *use* the function without
      reading its source.
- [ ] Call sites and test pointers listed; coverage gaps named.
- [ ] Issues (if any) logged with evidence; **no code was changed.**
- [ ] Units/frames stated and linked.

---

## Worked micro-example (illustrative — not yet verified)

Shows the *shape* of a finished entry. The specifics below are **placeholders** and must be
produced by the real passes before being trusted.

```markdown
### `phase_diff_to_theta(phase_diff, wavelength_spacing, ...)`  ·  `spf/rf.py:~721`  ·  ✅ Active

**Summary.** Inverts the array geometry to turn a measured phase difference φ into candidate
angle(s) of arrival θ via the arcsin relation. Because a 2-element array is front/back ambiguous,
it returns multiple θ candidates rather than one. (See beamforming_primer.md.)

**Signature**
- Inputs: `phase_diff` (rad, [-π,π]); `wavelength_spacing` = d/λ (dimensionless).
- Returns: θ candidates (rad). ⚠️ exact shape/ordering — confirm in Pass 1.
- Side effects: none (pure). ⚠️ confirm.

**Called from** — ⚠️ run Pass 2 (grep `phase_diff_to_theta`).
**Contract** — preconditions: φ already wrapped to [-π,π]? confirm at call sites.
**Tested by** — ⚠️ search tests/ (candidate: test_angles.py, test_rf.py).
**Analysis** — *Issues considered:* arcsin domain clipping when |φ| exceeds 2π·d/λ — verify the
clip exists and is correct. *Missing coverage:* behavior at the ±90° boundary.
```

Everything marked ⚠️ above is exactly what Passes 1–3 exist to resolve.
```
