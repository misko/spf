# P0 #3 — Cloud filter drivers import from `spf.scripts.run_filters_on_data*` (does not exist)

**Verdict: CONFIRMED P0 (code-level ImportError) — but the cloud pipeline appears effectively unused / dormant. The whole B2 filter pipeline is currently dead-on-arrival, including the Docker entrypoint, which is *separately* also broken by the same 2025-04-03 file move.**

Locations:
- `spf/filters/run_filters_on_data_b2.py:25`
- `spf/filters/run_filters_report_b2.py:10`

---

## 1. CONFIRM — imports are broken, no alternate resolution

Exact import lines:
- `spf/filters/run_filters_on_data_b2.py:25` →
  `from spf.scripts.run_filters_on_data import generate_configs_to_run, run_filter_jobs`
- `spf/filters/run_filters_report_b2.py:10` →
  `from spf.scripts.run_filters_on_data_b2 import get_all_items_by_bucket`

The target modules **do not exist**:
- `find . -name 'run_filters_on_data*'` / `run_filters_report*` → the real `.py` sources exist **only** under `spf/filters/`. `spf/scripts/` contains NO such `.py` files.
- `spf/scripts/` has **no `__init__.py`** (it resolves as an implicit/PEP-420 namespace package). `spf/filters/` also has **no `__init__.py`**. `spf/__init__.py` exists (0 bytes).

Empirical confirmation (run in repo):
- `python3 -c "import spf.scripts"` → succeeds (namespace package, `__file__` is `None`).
- `python3 -c "import spf.scripts.run_filters_on_data"` → **`ModuleNotFoundError: No module named 'spf.scripts.run_filters_on_data'`** (the submodule genuinely does not exist).
- `python3 -c "import spf.filters.run_filters_on_data"` → resolves the module path fine; fails only later on `import boto3` (`boto3` not installed in *this* local env, but it IS a declared dependency — see §4).

No alternate resolution exists:
- No `conftest.py` / `sys.path` manipulation references `scripts`.
- `pyproject.toml` declares `boto3` and `dotenv` as deps (lines 36–37) but defines no console-script or path remap that would recreate `spf.scripts.run_filters_on_data`.
- Stale bytecode `spf/scripts/__pycache__/run_filters_on_data*.cpython-312.pyc` (dated Mar 26 / Apr 3 2025) is a *fossil*: it was compiled while the source still lived at `spf/scripts/`. Python will NOT import a `.pyc` whose source `.py` is absent (no PEP 3147 sourceless fallback in `__pycache__`), so it does not rescue the import.

**Root cause:** commit `2276ed1 "move filters"` (2025-04-03) did a pure rename `R100 spf/scripts/run_filters_*.py → spf/filters/run_filters_*.py` (`git show --stat 2276ed1`: 5 files, 0 insertions/0 deletions). The intra-file `from spf.scripts...` imports were never updated to `from spf.filters...`. So both `_b2.py` files import their own (now-moved) siblings via the dead old package path.

---

## 2. REACHABILITY — is the cloud pipeline actually used?

Repo-wide grep for `run_filters_on_data_b2` / `run_filters_report_b2` / `spf.scripts.run_filters` (excluding `__pycache__` and `claude_docs`):

- **`docker/repo/Dockerfile:38`** (only one Dockerfile in the repo):
  `CMD /opt/spf-env/bin/python3 /spf/spf/scripts/run_filters_on_data_b2.py --config ... --parallel 192 ...`
- **`spf/filters/run_filters_report_b2.py:10`** — the broken import itself.
- **`spf/filters/run_filters_on_data_b2.py:25`** — the broken import itself.
- `spf/filters/run_filters.sh:10` references the **non-b2** local script, also at the stale `spf/scripts/run_filters_on_data.py` path (broken too, but local path — see §4).
- `read_particle.ipynb` imports `spf.scripts.run_filters_report` (also stale path; notebook, not pipeline).

No AWS Batch / job-queue YAML, no CI workflow, no cron, and no `.sh` other than the two above reference the b2 drivers. The only production trigger is the single Docker `CMD`.

**Assessment:** the B2/DynamoDB cloud filter pipeline is the *intended* at-scale path (it pulls datasets from B2, writes results to DynamoDB, `--parallel 192`), but it is **currently dormant/abandoned**: nothing that can actually launch it is wired up correctly, and the one launcher (Dockerfile CMD) is itself broken (§3).

---

## 3. TRIGGER — what invokes them

- `run_filters_on_data_b2.py`: invoked **only** by `docker/repo/Dockerfile:38` as the container `CMD`. **But that CMD path is also broken** — it points at `/spf/spf/scripts/run_filters_on_data_b2.py`, a path that no longer exists after the 2025-04-03 move (file is now `/spf/spf/filters/run_filters_on_data_b2.py`). So the container fails at `python3: can't open file` **before** Python even reaches the import error. The Dockerfile was last touched 2026-02-15 ("updates") and `git log -p` shows the CMD always used the `spf/scripts/...` path — it was never updated for the move.
- `run_filters_report_b2.py`: **no caller at all** — no Dockerfile, `.sh`, CI, or cron references it. It is a manually-run reporting script (`__main__` reads DynamoDB and writes CSV). It would `ImportError` on load if anyone ran it.

So there are **two independent breakages** stacked on the run driver: (a) container CMD path no longer exists; (b) even if path fixed, the in-module `spf.scripts...` import raises `ModuleNotFoundError`.

---

## 4. BLAST RADIUS — is the pipeline dead? Local alternative?

- **Cloud (B2/DynamoDB) path: fully dead.** Both `_b2.py` modules raise `ModuleNotFoundError` on import (confirmed empirically, §1), and the only launcher (Docker CMD) additionally has a stale exec path. The B2 dataset-pull + DynamoDB-write + report flow cannot run as committed.
- **Local path: also broken, but for a *different* reason** (out of scope of this P0, noted for completeness). `spf/filters/run_filters_on_data.py` is the local non-b2 runner and does NOT import from `spf.scripts` (only the `_b2` files do), so it imports fine. However the inventory notes a `TypeError` at its `__main__` (`run_filter_jobs(jobs)` called missing the required `nparallel` positional, `run_filters_on_data.py:~731` vs def `~627`). And its `.sh` launcher (`run_filters.sh:10`) calls the stale `spf/scripts/run_filters_on_data.py` path. So the local CLI path is degraded too, though the *library* import of `spf.filters.run_filters_on_data` itself works.

Net: there is **no currently-working end-to-end filter-evaluation entrypoint** wired in the repo. The importable, library-level function `run_filter_jobs` / `generate_configs_to_run` in `spf.filters.run_filters_on_data` is fine and could be called directly from Python.

---

## 5. REAL-WORLD — recency & evidence it once worked

- `git log` for both `_b2.py` files: last content change before the rename; since 2025-04-03 they were only *moved* (`git log -1`: 2025-04-03 19:46 for both). Active b2 development happened Dec 2024–Apr 2025 (`bd9fdb4 add boto3 b2 processing ... to amazon dynamodb`, `757418f b2 filter processing add ... routine`, `42237cf b2 fixes`, `7ee7201 fix for b2 run`).
- **Evidence it once worked at the old path:** stale `spf/scripts/__pycache__/run_filters_on_data*.cpython-312.pyc` (Mar–Apr 2025) — bytecode compiled while the source lived under `spf/scripts/`, i.e. the `spf.scripts.run_filters_on_data` import *did* resolve pre-move. Committed result dirs `spf/scripts/march1inference`, `spf/scripts/march8inference` (Mar 2025) predate the April move and are consistent with the pipeline producing inference output before the rename broke it.
- **Conclusion:** the imports were valid before 2025-04-03; the "move filters" rename silently broke them and they have sat broken for >1 year, which is itself strong evidence the cloud pipeline has not been run from this repo state since.

---

## 6. SEVERITY VERDICT — CONFIRMED P0 (with a dormancy caveat)

- **Code correctness: CONFIRMED P0.** Both cloud drivers raise `ModuleNotFoundError` at import time (verified by direct `python3 -c import ...`). An un-importable module that something is *configured* to run (Docker CMD) meets the P0 bar.
- **Operational urgency: effectively dormant.** The pipeline is not currently launched by any working trigger (the sole launcher's exec path is itself stale), there is no CI/Batch/cron wiring, and it has been broken for ~14 months with no fix. So while the defect is real and P0-class, nothing is silently failing in production *right now* because nothing is successfully starting it. Best characterized as **CONFIRMED P0 latent bug in a dormant pipeline** — must be fixed before the cloud path can be revived, but not actively burning.
- Do **not** downgrade to NON-ISSUE: the framing in `claude_docs/.../filters.md` that "the working path is the B2 one" is wrong — the B2 path does not import.

---

## 7. FIX SKETCH (describe only — not applied)

Minimal, mechanical fix — update the two stale package paths to the post-move location:

1. `spf/filters/run_filters_on_data_b2.py:25`
   `from spf.scripts.run_filters_on_data import ...` → `from spf.filters.run_filters_on_data import ...`
2. `spf/filters/run_filters_report_b2.py:10`
   `from spf.scripts.run_filters_on_data_b2 import ...` → `from spf.filters.run_filters_on_data_b2 import ...`

To actually make the cloud pipeline runnable again (companion fixes, beyond the strict import bug):
3. `docker/repo/Dockerfile:38` — change `/spf/spf/scripts/run_filters_on_data_b2.py` → `/spf/spf/filters/run_filters_on_data_b2.py` (else container fails to open the file).
4. (Local path, separate issue) `spf/filters/run_filters.sh:10` `spf/scripts/run_filters_on_data.py` → `spf/filters/run_filters_on_data.py`; and the `run_filter_jobs(jobs)` missing-`nparallel` `TypeError` in the non-b2 `__main__`.

Optional hardening: add `spf/filters/__init__.py` (and `spf/scripts/__init__.py`) so these are regular packages rather than relying on namespace-package resolution. Not strictly required for the import fix, since `spf.filters.run_filters_on_data` already resolves as a namespace submodule.

No behavioral/logic change is involved in items 1–3 — they are pure path corrections to repair the 2025-04-03 rename.
