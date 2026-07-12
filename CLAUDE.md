# SPF — project instructions for Claude

- **Before making any decisions** (data curation, training experiments, scanner/report
  changes, hardware/capture recommendations): **read `docs/learnings.md` first.** It holds
  the project's hard-won conclusions (e.g. what the `F:gain` flag really means, why
  sub-GHz g is untrustworthy, the frozen-val-set contract). Do not re-derive or contradict
  these without new evidence; when new evidence changes a conclusion, update
  `docs/learnings.md` in the same change.
- **Planned work lives in `docs/future_experiments.md`** — the queue of designed
  experiments (capture matrices, bench validations, training runs, scanner upgrades)
  with decision rules. Before proposing a new experiment, check whether it's already
  designed there; when one runs, record the outcome in `docs/learnings.md` and mark it
  in `docs/future_experiments.md`.
- Deeper background lives in `claude_docs/` (architecture docs, `KNOWN_ISSUES.md`,
  `03_datasets/data_quality_plan.md`, `04_training_inference/val_expansion_plan.md`).
- Artifacts are append-only: never edit existing split files or configs; new experiments
  get new manifests/configs with provenance-carrying names.
- The historical validation set is frozen forever; new val views are named
  `val_subset_groups` that must be subsets of it.
- Every generated PDF gets exported to PNG and visually verified (occlusion/collisions)
  before shipping; use a fresh agent to explain figures back when the change is substantial.
