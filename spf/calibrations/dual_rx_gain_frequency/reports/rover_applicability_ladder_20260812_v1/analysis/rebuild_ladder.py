"""Rebuild the model ladder from the committed machine-readable results, and
check it against the table published in the shipped package README §4.2."""
import json
import sys
from pathlib import Path

SCRATCH = Path(sys.argv[1])
A = json.loads((SCRATCH / "ladder_results_A_main.json").read_text())
M = json.loads((SCRATCH / "min_results.json").read_text())
W = json.loads((SCRATCH / "gsc1_wide_ladder.json").read_text())

SPLITS = {
    "LOEO": "LOEO leave-one-epoch-out",
    "LOFO": "LOFO leave-one-frequency-out",
    "LOBLK": "LOBLOCK leave-frequency-block-out",
    "LORO": "LORO leave-one-radio-out",
    "LOBAND": "LOBAND leave-one-gain-table-band-out",
}

# ---- stage A ladder, keyed by model name
stage_a = {}
for short, key in SPLITS.items():
    for row in A[key]:
        e = stage_a.setdefault(row["model"], {"params": row["params"]})
        e[short] = dict(
            coverage=row["coverage"],
            mae=row["mae_deg"],
            uneq=row.get("unequal_mae_deg"),
            p95=row["p95_deg"],
            mx=row["max_deg"],
            sup_mae=row.get("sup_mae_deg"),
            n=row["n"],
        )

# ---- pooled leave-one-frequency-out (README §4.2 lower table)
pooled = {}
for key in ("pooled_LOFO", "LOFO leave-one-frequency-out"):
    if key not in M:
        continue
    blk = M[key]
    it = blk.items() if isinstance(blk, dict) else ((r["model"], r) for r in blk)
    for name, row in it:
        pooled.setdefault(key, {})[name] = dict(
            params=row.get("params"),
            coverage=row.get("coverage"),
            mae=row.get("failclosed_mae_deg", row.get("mae_deg")),
            p95=row.get("failclosed_p95_deg", row.get("p95_deg")),
            sup_mae=row.get("sup_mae_deg"),
            uneq=row.get("unequal_mae_deg"),
        )

# ---- leave-one-gain-out coverage (README §4.5)
logo = {
    name: dict(params=r["params"], coverage=r["coverage"],
               sup_mae=r["sup_mae_deg"], sup_p95=r["sup_p95_deg"])
    for name, r in M["leave_one_gain_out_pooled"].items()
}
loband = {
    name: dict(params=r["params"], coverage=r["coverage"],
               failclosed_mae=r["failclosed_mae_deg"])
    for name, r in M["LOBAND leave-one-gain-table-band-out"].items()
} if isinstance(M.get("LOBAND leave-one-gain-table-band-out"), dict) else {
    r["model"]: dict(params=r["params"], coverage=r["coverage"],
                     failclosed_mae=r.get("mae_deg"))
    for r in M.get("LOBAND leave-one-gain-table-band-out", [])
}

# ---- the independent 53-LO wide survey ladder (2026-08-07)
wide = {}
for key, rows in W["results"].items():
    for row in rows:
        wide.setdefault(row["model"], {"params": row["params"]})[key] = dict(
            coverage=row["coverage"], mae=row["mae_deg"],
            uneq=row.get("unequal_mae_deg"), p95=row["p95_deg"],
        )

# ---- check against the README table
README_42 = {
    # model prefix : (params, LOEO, LOEO uneq, LOFO, LOFO uneq, LOBLK, LORO)
    "L00": (0, 6.65, 8.31, 6.65, 8.31, 6.65, 6.65),
    "L01": (3, 5.12, 6.40, 5.16, 6.45, 5.64, 5.13),
    "L11": (12, 2.99, 3.74, 3.08, 3.85, 3.14, 3.05),
    "L14": (15, 2.85, 3.56, 2.99, 3.73, 3.25, 2.90),
    "L16": (21, 2.42, 3.02, 2.50, 3.12, 2.70, 2.49),
    "L18": (21, 2.54, 3.18, 2.70, 3.37, 3.49, 2.71),
    "L26": (27, 2.08, 2.60, 2.26, 2.83, 2.47, 2.22),
    "L27": (49, 1.68, 2.10, 1.85, 2.32, 3.52, 1.91),
    "L29": (45, 2.75, 3.44, 3.10, 3.88, 4.03, 2.92),
    "L30": (8, 3.49, 4.37, 3.54, 4.42, 3.66, 3.52),
    "L31": (20, 2.45, 3.06, 2.58, 3.22, 2.79, 2.54),
    "L33": (43, 1.81, 2.26, 1.99, 2.48, 3.58, 2.00),
    "L23": (678, 0.99, 1.23, None, None, None, None),
    "L24": (1356, 0.62, 0.77, None, None, None, None),
}
checks = []
for prefix, vals in README_42.items():
    match = [k for k in stage_a if k.startswith(prefix + " ")]
    if not match:
        checks.append(dict(rung=prefix, status="NOT FOUND in committed JSON"))
        continue
    k = match[0]
    e = stage_a[k]
    got = (
        e["params"],
        round(e["LOEO"]["mae"], 2), round(e["LOEO"]["uneq"], 2),
        round(e["LOFO"]["mae"], 2), round(e["LOFO"]["uneq"], 2),
        round(e["LOBLK"]["mae"], 2), round(e["LORO"]["mae"], 2),
    )
    want = vals
    diffs = []
    labels = ["params", "LOEO", "LOEO_uneq", "LOFO", "LOFO_uneq", "LOBLK", "LORO"]
    for lab, g, w in zip(labels, got, want):
        if w is None:
            continue
        if lab == "params":
            if g != w:
                diffs.append(f"{lab}: README {w} vs JSON {g}")
        elif abs(g - w) > 0.005:
            diffs.append(f"{lab}: README {w} vs JSON {g}")
    checks.append(dict(rung=prefix, model=k, json_values=dict(zip(labels, got)),
                       readme_values=dict(zip(labels, want)),
                       status="match" if not diffs else "MISMATCH",
                       differences=diffs))

out = dict(
    stage_a_ladder=stage_a,
    pooled=pooled,
    leave_one_gain_out_pooled=logo,
    leave_one_band_out_pooled=loband,
    wide_survey_ladder_53lo=wide,
    wide_survey_meta={k: W[k] for k in
                      ("source", "convention", "caveats", "n_rows", "n_los",
                       "baseline_all_deg", "baseline_uneq_deg", "state_coverage")},
    readme_section_4_2_verification=checks,
)
(SCRATCH / "ladder_rebuilt.json").write_text(json.dumps(out, indent=1))
print("models in stage-A ladder:", len(stage_a))
for c in checks:
    print(c["status"], c["rung"], c.get("differences", ""))
print("\nwide-survey ladder splits:", list(W["results"].keys()))
print("wide models:", list(wide))
print("\npooled keys:", list(pooled))
