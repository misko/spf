# Adversarial Verification — `v5spfdataset.md`

Target doc: `claude_docs/reference/spf/dataset/v5spfdataset.md`
Source: `spf/dataset/spf_dataset.py` (`v5spfdataset` + module collate fns)
Verdict: **Grade A.** Default-skeptic re-derivation from source confirmed every priority claim; only nits found.

## (a) Priority claims

| # | Claim | Verdict | Evidence (source line) |
|---|---|---|---|
| 1 | `__getitem__` returns `list[dict]` len `n_receivers` if `paired` else 1; `None` for temp past `valid_entries` | **CONFIRMED** | paired: `:1612-1615` `return [self.render_session(receiver_idx, idx, ...) for receiver_idx in range(self.n_receivers)]`; unpaired: `:1621` `return [self.render_session(receiver_idx, session_idx)]`. None: paired `:1607-1610` `if idx >= self.valid_entries[ridx]: return None`; unpaired `:1619-1620` `if self.temp_file and session_idx >= self.valid_entries[receiver_idx]: return None` |
| 2a | `y_rad`/`ground_truth_theta` = array-relative | **CONFIRMED** | `:1406` `data["y_rad"] = data["ground_truth_theta"]`; `:1587-1591` `torch_pi_norm(rx_to_tx_theta - (rx_theta_in_pis[:] + rx_heading_in_pis) * torch.pi)` |
| 2b | `craft_y_rad`/`craft_ground_truth_theta` = craft/heading-relative (heading NOT added) | **CONFIRMED** | `:1408` `data["craft_y_rad"] = data["craft_ground_truth_theta"]`; `:1546-1553` adds only `rx_theta_in_pis * torch.pi`, heading commented out `:1550` |
| 2c | `absolute_theta` = world | **CONFIRMED** | `:1374` assigns from `self.absolute_thetas`; `:1586` `absolute_thetas.append(torch_pi_norm(rx_to_tx_theta))` (raw arctan2) |
| 2d | `y_phi` = φ target | **CONFIRMED** | `:1407` `data["y_phi"] = data["ground_truth_phi"]` |
| 2e | `-sin` φ sign convention (~:1518) | **CONFIRMED (line nit)** | `:1513` `-torch.sin(...)`; ambiguity comment spans `:1518-1521`. The `-sin` token itself is `:1513`, not `:1518`; doc's `~:1518` points at the comment, acceptable as "~" |
| 3a | `get_segmentation` deletes `.pkl` then re-calls self on `UnpicklingError` | **CONFIRMED** | `:1753` `except pickle.UnpicklingError:` → `:1755` `os.remove(results_fn)` → `:1756-1760` `return self.get_segmentation(...)` |
| 3b | With `segment_if_not_exist=False`, deletes-then-raises | **CONFIRMED** | after delete, recursion reaches `:1683-1688`; `not segment_if_not_exist and not self.temp_file and not os.path.exists(results_fn)` now true → `:1688` `raise ValueError` |
| 3c | Corrupt `.yarr` (~:1732) is uncaught | **CONFIRMED** | `.yarr` open at `:1732` is inside the `try` (opens `:1719`) but the only `except` is `pickle.UnpicklingError` (`:1753`); a `.yarr` failure raises a different exception → not caught/deleted |
| 4 | `__getitem__` asserts vs `n_snapshots` while `__len__` returns `n_sessions` | **CONFIRMED** | `:1606` `assert idx < self.n_snapshots` (paired); `:1616` `assert idx < self.n_snapshots * self.n_receivers`; `__len__` `:1285-1287` returns `self.n_sessions` / `self.n_sessions * self.n_receivers` |
| 5 | `v5_collate_beamsegnet` dormant (only caller commented out) | **CONFIRMED (with nit)** | only call site is `simple_train.py:198` `# "collate_fn": v5_collate_beamsegnet,` (commented). It is additionally *imported but unused* in `benchmark_dataset.py:17` and `benchmark_dataset_linear.py:15` (those files actually call `v5_collate_keys_fast`). Doc said "only caller is a commented line" — true for calls; it omits the two dead imports. Verdict stands: dormant. |

## (b) Line-number hit rate

16/16 spot-checked lines land on the cited symbol (or the documented multi-line range start):
`:906, :909, :1024(comment, assert :1026-1030), :1056(guard, assert :1057-1060), :1131, :1237, :1254, :1369, :1398, :1441, :1457, :1595, :1636, :1688, :281, :297`. Cross-file refs also correct: `torch_pi_norm` rf.py:77, `SEGMENTATION_VERSION = 3.7` utils.py:14 (doc correctly flags inventory's "3.5" as stale). One trivial off-by-one: `to_bin` is utils.py:**250** (doc says :251 — body line). `-sin` is `:1513` (doc `~:1518`).

**Hit rate: 16/16 exact-or-range (100%); 2 cosmetic "~" pointers (to_bin :250 vs :251, -sin :1513 vs ~:1518).**

## (c) Missed / misclassified

- **Nit, claim 5:** `v5_collate_beamsegnet` is imported (dead) in `benchmark_dataset.py` and `benchmark_dataset_linear.py`. Doc's "only live caller is a commented line" is correct but could note the two stray imports. Does not change the 🧪 dormant verdict.
- **Shape table — all verified correct:** `gains (S,1,2)` (`:1369` `[:, None]` on cached `(S,2)`), `receiver_idx (1,S) int32` (`:1069-1071 expand(1, snapshots_per_session)`), `craft` recompute `:1398`, `rx_pos_xy` meters `:1457-1460`. No wrong shapes/frames found.
- **`v5_collate_all_fast` status "✅ Active (notebook only)":** caller `simple_train.py:197` is live (not commented), unlike beamsegnet's `:198`. Correctly distinguished by the doc.
- **No contract gaps found** beyond what the doc already flags (nondeterministic window subsample, `abs_signal_and_phase_diff` not flipped, NaN mean_phase, in-ctor QC raise, hardcoded 2-rx).

## (d) Overall grade: **A**

All five priority claims CONFIRMED. 100% line-number hit rate on spot-check. Only defects are cosmetic (two `~`/off-by-one pointers, one omitted pair of dead imports). The doc is unusually accurate and its KNOWN_ISSUES #8 refinement (delete-then-raise; `.yarr` uncaught; bounded recursion) is precisely correct against source.
