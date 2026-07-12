# SPF — project instructions for Claude

- **Before making any decisions** (data curation, training experiments, scanner/report
  changes, hardware/capture recommendations): **read `docs/learnings.md` first.** It holds
  the project's hard-won conclusions (e.g. what the `F:gain` flag really means, why
  sub-GHz g is untrustworthy, the frozen-val-set contract). Do not re-derive or contradict
  these without new evidence; when new evidence changes a conclusion, update
  `docs/learnings.md` in the same change.
- Deeper background lives in `claude_docs/` (architecture docs, `KNOWN_ISSUES.md`,
  `03_datasets/data_quality_plan.md`, `04_training_inference/val_expansion_plan.md`).
- Artifacts are append-only: never edit existing split files or configs; new experiments
  get new manifests/configs with provenance-carrying names.
- The historical validation set is frozen forever; new val views are named
  `val_subset_groups` that must be subsets of it.
- Every generated PDF gets exported to PNG and visually verified (occlusion/collisions)
  before shipping; use a fresh agent to explain figures back when the change is substantial.
