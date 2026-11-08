# Adversarial verification — `rf.md` Clusters C & D

Scope: only the new `## Cluster C` (windowed circular statistics) and `## Cluster D`
(phase extraction) sections. Clusters A/B not re-checked. Source: `spf/rf.py` @ working tree,
call sites in `spf/dataset/segmentation.py`, `spf/dataset/spf_dataset.py`,
`spf/data_collector.py`, `spf/dataset/fake_dataset.py`, `tests/test_rf.py`.

Verdict legend: CONFIRMED / REFUTED / PARTIAL.

## (a) Priority claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Three percentile conventions; live=`fast_percentile_1d` (ceil, untested), tested=`fast_percentile` (floor, 0 prod callers), torch=`torch.quantile` | **CONFIRMED** | `fast_percentile_1d` `rf.py:348-355` uses `k=int(np.ceil((p/100)*n))-1`; callers `rf.py:114,328,602` (all live `_fast`/`_fast2`). `fast_percentile` `rf.py:82-90` uses `k=int(np.floor((p/100)*(n-1)))`, njit commented (`:81`); its only refs are `tests/test_rf.py:12,96` and import `tests/test_beamformer.py:18` — **zero production callers** (grep). Its test asserts vs `np.percentile(...,interpolation='lower')` (`test_rf.py:97`). torch path uses `torch.quantile(diff,1-trim/100)` (`rf.py:170,405,448`). All three confirmed. |
| 2 | `_fast2` passes scalar `pd[k]` to array fn `circular_diff_to_mean_single_fast` (`:612,625`) → crash; `@njit` commented; sole call site commented in segmentation | **CONFIRMED** | `rf.py:612` `diff = circular_diff_to_mean_single_fast(pd[k], mean_angle)` and `:625` same with `circ_mean` — both pass scalar `pd[k]`. `circular_diff_to_mean_single_fast` (`:234`) immediately does `n = angles.shape[0]` (`:235`) — scalar has no `.shape`. Decorator `# @njit(parallel=True)` commented `rf.py:556`. Sole call site `segmentation.py:661` is `# step_idxs2, step_stats2 = windowed_..._fast2(` — commented. All parts confirmed. (Note the correctly-typed scalar helper `circular_diff_to_mean_single_scalar` `:548` IS used correctly at `:585`, as doc states.) |
| 3 | `torch_mean_phase_mean` uses `np.sin/np.cos/np.arctan2` on torch input (broken) AND zero callers (dead) | **CONFIRMED** | `rf.py:424-429`: `_sin_angles = np.sin(angles)*weights`, `_cos_angles = np.cos(angles)*weights`, `cm = np.arctan2(...)` then `torch_pi_norm(cm)` — numpy ufuncs on torch tensor input. Grep for `torch_mean_phase_mean` across repo (non-doc): only the def at `rf.py:424`. Zero callers. Both confirmed (KNOWN_ISSUES #20). |
| 4 | Live chain: `windowed_..._fast`→`get_stats_for_signal_window`→{`circular_mean_single_fast`,`circular_stddev_fast`,`fast_median_abs`}, all on `fast_percentile_1d`; called by segmentation | **CONFIRMED** | `windowed_..._fast` `rf.py:657` calls `get_stats_for_signal_window` at `:671`. `get_stats_for_signal_window` `:686` calls `circular_mean_single_fast` (`:690`), `circular_stddev_fast` (`:691`), `fast_median_abs` (`:692`). `circular_mean_single_fast` uses `fast_percentile_1d` (`:328`); `circular_stddev_fast` uses it (`:114`). Segmentation: `get_all_windows_stats` `segmentation.py:641` → `windowed_..._fast` at `:658`; `get_all_windows_stats` called from `simple_segment` at `segmentation.py:726`. Chain exact. |
| 5 | `torch_get_phase_diff` indexes antennas on dim 1 (`[:,0]`/`[:,1]`), numpy `get_phase_diff` on dim 0; doc resolves as not-a-bug for `(batch,2,N)` | **CONFIRMED** | `torch_get_phase_diff` `rf.py:750-751`: `signal_matrix[:,0].angle() - signal_matrix[:,1].angle()` → dim 1. `get_phase_diff` `rf.py:744-746`: `signal_matrix[0]`/`signal_matrix[1]` → dim 0. Callers `spf_dataset.py:720,1414` pass `data["signal_matrix"][0]`; comment `spf_dataset.py:717` `# signal matrix ~ 1,1,2,524288`, assert `shape[0]==1` (`:719`), so after `[0]` shape is `(1,2,N)` → `[:,0/1]` selects antennas correctly. Resolution correct. |

## (b) Ordinary-claim spot-check (Cluster C + D), hit rate

| Claim | Verdict | Evidence |
|---|---|---|
| `windowed_..._fast` divisibility `assert (n_samples-window_size)%stride==0` at `:661` | CONFIRMED | `rf.py:661` exact. |
| `prange` under plain `@njit` (not parallel) at `:666` → serial | CONFIRMED | decorator `@njit` `rf.py:656`, `for step in prange(n_steps)` `:666`. |
| `get_stats_for_signal_window` slices `pd[start:end]`, `v[:,start:end]`, returns float32 array `:694` | CONFIRMED | `rf.py:687-694`. |
| `get_stats_for_signal` 2 prod callers: `segmentation.py:930` (recompute) + `windowed_..._fast` serial `:714` | CONFIRMED | `segmentation.py:930` `r=get_stats_for_signal(...)`; `rf.py:714` in `windowed_..._fast` (non-fast). |
| `circular_mean_single` (`:359`) dead, uses `np.percentile` | CONFIRMED | def `:359`, `np.percentile` `:374`; no callers (grep). |
| `circular_stddev` (`:134`) dead, `np.percentile` | CONFIRMED | def `:134`, `np.percentile` `:145`; no callers. |
| `circular_diff_to_mean_single_fast` returns float32, tested `test_rf.py:118` | CONFIRMED | `rf.py:236` `dtype=np.float32`; test `test_rf.py:108-118`. |
| `fast_median_abs`/`fast_abs_median` byte-identical; `fast_abs_median` only in `_fast2` `:643,644` | CONFIRMED | `rf.py:472-486` vs `528-544` identical logic; `fast_abs_median` used `:643,644`. |
| `fast_median_abs` tested `test_rf.py:139` | CONFIRMED | `test_rf.py:126 def test_fast_median_abs`, assert `:139`. |
| torch windowed-stats path (`torch_windowed_...` `:500`, `torch_get_stats_for_signal` `:460`, `torch_circular_stddev` `:157`) dead end-to-end | CONFIRMED | `torch_windowed_...` no callers; `torch_get_stats_for_signal` only called by it (`:521`); `torch_circular_stddev` only by `torch_get_stats_for_signal` (`:464`). |
| `mean_phase_mean` live at `segmentation.py:96` | CONFIRMED | `segmentation.py:96` `mean_phase_mean(angles=means, weights=weights)`. |
| `get_avg_phase` live `data_collector.py:159`; `get_avg_phase_fast2` live `:291` | CONFIRMED | `data_collector.py:159`, `:291` (`get_avg_phase_fast2`). |
| `get_avg_phase_fast` dead (0 callers) | CONFIRMED | grep: only def `rf.py:762`. |
| `torch_get_avg_phase_notrim` live `fake_dataset.py:289` | CONFIRMED | `fake_dataset.py:289`. |
| `torch_get_avg_phase` (trim) dead, wraps `torch.tensor(tuple)` `:795` | CONFIRMED | no callers; `rf.py:795` `torch.tensor(torch_circular_mean(...))`. |

Spot-check hit rate: **15/15 ordinary claims correct** plus 5/5 priority = effectively 100% on the
sampled set. Every line number checked resolved to the named construct (line numbers exact, no drift
observed at commit state).

## (c) Missed / misclassified

1. **`torch_circular_mean_noweight` mislabeled "dead / no callers" (🧪).** Cluster D table row says
   "🧪 dead | 0" and prose `:846` says "no callers 🧪". **REFUTED in part:** it IS called at
   `rf.py:461` inside `torch_get_stats_for_signal` (`trimmed_cm = torch_circular_mean_noweight(pd.reshape(1,-1), trim=trim)`).
   The mitigating truth: its only caller is itself dead (`torch_get_stats_for_signal` → only via the
   dead `torch_windowed_...`), so it is *transitively* dead — but the doc's stated fan-in of literally
   "0 callers" is inaccurate; correct statement is "1 caller, itself dead." Minor severity (conclusion
   of deadness stands), but the fan-in count is wrong. Note the Cluster C torch block (`:673`) DOES
   correctly say `torch_get_stats_for_signal` is "Called only by `torch_windowed_...`" — so the two
   sections are mutually slightly inconsistent about this sub-chain.

2. **Minor wording on `circular_stddev` njit:** doc calls `circular_stddev` (`:134`) a numpy-percentile
   `@njit` twin "differing in percentile backend" — correct, but worth noting `circular_stddev` internally
   calls `circular_diff_to_mean_single_fast` (`:135`), same as `_fast`; the ONLY difference is
   `np.percentile` (`:145`) vs `fast_percentile_1d`. Doc's characterization is accurate; no error.

3. No functions in C/D were missed. The `_fast2`/`circular_diff_to_mean_single_scalar` misuse, the
   four-way distance family, the two median twins, and the three percentile conventions are all captured.

No duplicated-variant misclassification beyond item 1.

## (d) Grade

**A−**

Rationale: All five priority claims CONFIRMED with exact evidence; 15/15 sampled ordinary claims correct;
line numbers exact; the three hard KNOWN_ISSUES (#20, #21, #37/percentile divergence) precisely
characterized including the subtle correct-vs-incorrect scalar-helper distinction in `_fast2`. The single
defect is the `torch_circular_mean_noweight` fan-in: documented as "0 callers" when it has 1 (itself
dead) — the deadness conclusion survives but the count is wrong and contradicts the doc's own Cluster C
torch block. Not an A because that is a verifiable factual slip in a claims table, not just prose hedging.
