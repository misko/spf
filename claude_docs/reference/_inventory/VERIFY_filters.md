# Adversarial Verification: `filters.md`

Verified against source in `/home/mouse9911/gits/spf/spf/filters/*.py` (13 files). READ-ONLY check.

## (a) Every 🔴 / 🧪 / dead claim → verdict + evidence

| # | Inventory claim | Verdict | Evidence (file:line, quoted) |
|---|---|---|---|
| 1 | LIVE `breakpoint()` at `particle_dual_radio_nn_filter.py:41` inside `PFSingleThetaDualRadioNN.observation()`, not commented | **CONFIRMED** | L40 `def observation(self, idx):` / L41 `        breakpoint()` / L42 `return self.ds.get_inference_for_idx(idx)[0]["paired"][0]`. Not commented. Only live breakpoint in all filter files (grep `^[^#]*breakpoint()` → single hit at L41). |
| 2 | `run_filters_on_data.py:731` calls `run_filter_jobs(jobs)` without required `nparallel` positional → TypeError | **CONFIRMED** | L731 (last line of file) `    run_filter_jobs(jobs)`. Def L626-628: `def run_filter_jobs(jobs, nparallel, debug=False, ...)` — `nparallel` has no default. Calling with one positional raises `TypeError`. |
| 3 | `run_PF_single_theta_dual_radio_NN` (L416) transitively blocked by breakpoint filter | **CONFIRMED** | L416 def; L437 `pf = PFSingleThetaDualRadioNN(nn_ds=nn_ds)` → its `observation()` hits breakpoint on first trajectory step. |
| 4 | `# breakpoint()` at `particle_dualradioXY_filter.py:99` is DEAD (commented), not live | **CONFIRMED** | L99 `        # breakpoint()` inside `metrics()`. Commented. |
| 5 | `config_to_job_params` uses `eval()` (~530) | **CONFIRMED** | L522 `def config_to_job_params(config):`; L530 `value = eval(value)  # TODO this might be dangerous`. Inventory body cites L530 exactly. |
| 6 | report/b2 files import runners from `spf.scripts.run_filters_on_data*` not `spf.filters.*` | **CONFIRMED (path)** but framing MISLEADING | `run_filters_on_data_b2.py:25` `from spf.scripts.run_filters_on_data import generate_configs_to_run, run_filter_jobs`; `run_filters_report_b2.py:10` `from spf.scripts.run_filters_on_data_b2 import get_all_items_by_bucket`. Path is `spf.scripts.*` as claimed. BUT see correction #1 below: those modules DO NOT EXIST — imports are broken, no mirrored copy. `run_filters_report.py` imports no runner (inventory did not claim it does). |
| 7 | `theta_phi_to_p` (filters.py:409) defined-but-unused | **CONFIRMED** | L409 `def theta_phi_to_p(theta, phi, full_p):`. Repo-wide grep for `theta_phi_to_p\b` (excluding `_vec` and the def) → ZERO callers. Dead. |
| 8 | `cannonical` defined-but-unused | **CONFIRMED** | Defined `run_filters_report.py:15`, `run_filters_report_b2.py:17`. Repo-wide grep → only the two defs, no callers. Dead (both). |
| 9 | `get_type` defined-but-unused | **CONFIRMED** | Defined `run_filters_report.py:65`, `run_filters_report_b2.py:83`. Grep in filters/ → only defs, no callers. Dead. |
| 10 | `report_workdir_to_csv` defined-but-unused | **PARTIALLY-CORRECT** | non-b2 `run_filters_report.py:104` IS called at L155 in `__main__` (NOT dead there — inventory's row at L322 marks ✅, consistent). b2 `run_filters_report_b2.py:124` is NOT called by its `__main__` (which uses `get_all_items_by_bucket`+`results_with_prefix`, L189/L200) → dead in b2 main, as inventory states (L344, L350). So the blanket "defined-but-unused" is only true for the b2 copy; inventory's nuanced per-file rows are correct. |

## (b) Line-number hit rate

20 rows spot-checked (symbol present at cited line): **20/20 = 100%.**
Notable: `particle_dualradio_filter.py:14` cites func `pf_single_theta_dual_radio_update`; L14 is the `@torch.jit.script` decorator, `def` on L15 — decorator-line convention, acceptable (not an error).

8-class matrix validated against class defs:
- `SPFKalmanFilter` ekf_single:19, `dim_x=2, dim_z=1` ✅ (matrix: 2-D, single) ✓
- `SPFPairedKalmanFilter` ekf_dual:21, `dim_x=2, dim_z=2` ✅ (2-D dual shared) ✓
- `SPFPairedXYKalmanFilter` ekf_XY:43, `dim_x=6, dim_z=2` ✅ (6-D) ✓
- `PFSingleThetaSingleRadio` :18 ✓; `PFSingleThetaDualRadio` :30 ✓
- `PFXYDualRadio` :12, particle init `mean=[[0,0,0,0,0]]` → 5-D `[theta,x,y,dx,dy]` ✅ ✓
- `PFSingleThetaSingleRadioNN` :22 ✓; `PFSingleThetaDualRadioNN` :30 (🔴) ✓
Matrix: **8/8 classes correct** (algo, state-dim, radios, likelihood, status all match source).

## (c) Missed / misclassified

- **BIGGEST CORRECTION — claim #6 framing is wrong:** `spf/scripts/run_filters_on_data.py` and `spf/scripts/run_filters_on_data_b2.py` **DO NOT EXIST** (`find` over repo finds the run_filters modules ONLY under `spf/filters/`; `spf/scripts/` has no such files and no `__init__.py`). The inventory hedges "Suggests these runner modules are also mirrored/expected under `spf/scripts/`; verify which copy is canonical" — there is NO mirrored copy. The `from spf.scripts.run_filters_on_data...` imports in both `_b2.py` files are therefore **broken (ImportError at runtime)**. This is a real bug the inventory under-states (it grades the b2 files ✅). Recommend marking both `_b2.py` modules 🔴 (un-importable) or noting the imports are dead/broken.
- Side-effect classifications spot-validated and correct: DynamoDB `boto3.resource("dynamodb").Table("filter_metrics")` at L73-74, `put_item` L171; B2 download / `load_dotenv` in `_b2.py`; `eval` self-flagged. No missed I/O found.
- Decorator-vs-def line for jit functions (e.g. pf_single_theta_dual_radio_update) is a benign convention, not an error.

## (d) Overall grade: **A−**

Rationale: Every 🔴 and dead-symbol claim CONFIRMED against source; 100% line-number hit rate; 8/8 class matrix correct; side-effects accurate. Marked down from A only for the under-stated `spf.scripts.*` import situation — the inventory left it as "verify which is canonical" when the modules simply don't exist and the imports are broken, a fact the adversarial check surfaced and the inventory should have resolved.
