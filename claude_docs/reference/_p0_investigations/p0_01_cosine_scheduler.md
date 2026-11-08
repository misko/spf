# P0 Investigation #1: `load_optimizer` cosine branch → UnboundLocalError

**File:** `spf/scripts/train_single_point.py:377-397`
**Verdict (TL;DR):** DOWNGRADE to **P2**. Real bug, but only reachable by ONE opt-in config that is not used in any normal/CI/test path.

---

## 1. CONFIRM — does cosine leave `train_scheduler` unbound?

YES. The cosine branch is a pure `pass` (only commented-out code), so `train_scheduler` is never assigned, and the function unconditionally returns it.

`spf/scripts/train_single_point.py:377-397`:
```python
def load_optimizer(optim_config, params):
    optimizer = torch.optim.AdamW(
        params,
        lr=optim_config["learning_rate"],
        weight_decay=optim_config["weight_decay"],
    )
    if optim_config["scheduler"] == "step":
        train_scheduler = StepLR(
            optimizer,
            step_size=optim_config["scheduler_step"],
            gamma=0.5,
        )
    elif optim_config["scheduler"] == "cosine":
        # scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        # scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        # scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        pass
    else:
        raise ValueError(f"Invalid scheduler choice {optim_config['scheduler']}")

    return optimizer, train_scheduler   # <-- UnboundLocalError when scheduler == "cosine"
```
`train_scheduler` is only ever assigned inside the `"step"` branch (line 384). For `scheduler == "cosine"` the function reaches `return optimizer, train_scheduler` (line 397) with `train_scheduler` never bound → `UnboundLocalError: local variable 'train_scheduler' referenced before assignment`. (Note the commented code uses the name `scheduler`, not `train_scheduler`, so even uncommenting it would not fix the return.)

## 2. REACHABILITY — default scheduler + config census

- **Default scheduler = `"step"`**: `spf/scripts/train_utils.py:33` → `get_key_or_set_default(config, "optim/scheduler", "step")`. Configs that omit `scheduler:` get `step`, which hits the working branch.
- **Config census** (`grep -rn "scheduler" model_configs/`, 164 yaml files, 171 scheduler lines):
  - Files setting `scheduler: cosine`: **exactly 1**
    - `model_configs/dec3_single_config_sigma0p05_rerun_3p4_windows_beamformerwindow_nophase_nobeam_noempirical_nospacing_normalized_fix_bigger_4x.yaml:75`
  - Files setting `scheduler: step` explicitly: **0** (none set the value `step` literally; they rely on the default).
  - Files setting only `scheduler_step:` (→ default `scheduler: step`): all the rest (~170 lines across ~160 files).
- **Is the cosine config real/recent/used?** It is a one-off. Last touched by commit `0184f0f "some cleanup"` (no recent activity). No Python file references `bigger_4x` (`grep -rln "bigger_4x" --include=*.py` → empty). It is not referenced by any test, conftest, or driver script. The very name (`..._bigger_4x`) plus the commented-out scheduler body suggests it was an abandoned experiment.

## 3. TRIGGER — exactly what makes it fire

Fires iff BOTH:
1. The active config has `optim.scheduler == "cosine"` (only the one `bigger_4x.yaml` above), AND
2. `load_optimizer(config["optim"], m.parameters())` is called in the normal train path — `spf/scripts/train_single_point.py:1133`.

When triggered it crashes **immediately at optimizer construction time** (line 1133), before the training loop, before the first batch. The AdamW optimizer is built successfully; the error is raised on the `return` statement of `load_optimizer`. So it is a fail-fast crash at startup, not a mid-training or silent failure.

## 4. BLAST RADIUS — crash vs silently-wrong

**Hard crash, fail-fast, not silent.** One line: a single opt-in config crashes at optimizer-construction time (startup) with `UnboundLocalError`; no data loss, no silently-wrong training, and every other config is unaffected.

- Sole caller in the live path: `spf/scripts/train_single_point.py:1133` (`optimizer, scheduler = load_optimizer(...)`). The other reference at line 465 is commented out.
- **No try/except** wraps the call. Line 1133 sits in plain code in the model-loading sequence; the surrounding region has no exception handler swallowing it, so the error propagates and aborts the process. (If it somehow returned `None`, downstream `scheduler.step()` / `scheduler.get_last_lr()` at lines 1360/1333 would crash anyway — but it never gets that far.)

## 5. REAL-WORLD — tests / CI / conftest

- No test constructs an optimizer with `cosine`: `grep -rn "cosine" --include=*.py .` returns no test hits.
- No test/conftest references the `bigger_4x` cosine config.
- Tests/CI exercise only step-scheduler (default) configs, so CI never hits the cosine branch.

## 6. SEVERITY VERDICT — DOWNGRADE to P2

Reasoning = misbehavior × reachability:
- **Misbehavior:** moderate-high in isolation (hard crash), but it is fail-fast at startup, not silent corruption, and is trivially diagnosable.
- **Reachability:** essentially zero in practice — a single opt-in, non-default, unreferenced one-off config (`...bigger_4x.yaml`). Default is `step`; ~160 configs use the working path; no test/CI/driver selects cosine.

A crash reachable only by one opt-in config that nobody runs (and which itself looks abandoned, with its scheduler body commented out) is **not P0**. It is a latent bug. **P2** (cosmetic/latent; fix when convenient). Could argue P1 only if someone is about to actively use that config — no evidence of that.

## 7. FIX SKETCH (do NOT apply)

Minimal options, in order of preference:
- **Implement the intended cosine scheduler** in the branch, e.g. assign `train_scheduler = CosineAnnealingLR(optimizer, T_max=optim_config["scheduler_step"])` (import `CosineAnnealingLR` from `torch.optim.lr_scheduler`). This makes the one config actually work. Smallest correct fix.
- **OR**, if cosine is not wanted yet, replace the `pass` body with `raise NotImplementedError("cosine scheduler not implemented")` (or remove the branch so it falls to the existing `ValueError`). This converts the confusing `UnboundLocalError` into an explicit, intentional error.

Either way the bug is one branch in one function; no caller changes needed.
