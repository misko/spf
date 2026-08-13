"""Emit the candidate coefficient sets, with full provenance, and check the comb."""

from __future__ import annotations

import hashlib
import json
import subprocess
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SPF_REPO",
                           Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitlib as FL  # noqa: E402
import union as U  # noqa: E402
from spf.calibrations.gain_state_phase_model_v1.fit_from_extracted import (  # noqa: E402
    check_comb_conditioning,
)
from spf.calibrations.gain_state_phase_model_v1.gain_tables import (  # noqa: E402
    default_tables,
)

WT = REPO
SRC = REPO / "spf/calibrations/dual_rx_gain_frequency/reports"
INPUTS = [
    SRC / "equal_gain_diagonal_20260811_v1/additive_cross_1040007c4a94.json",
    SRC / "equal_gain_diagonal_20260811_v1/additive_cross_104000bac495.json",
    SRC / "e_gsc7_iio_20260812_v1/analysis.json",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tab = default_tables()
    git_sha = subprocess.run(
        ["git", "-C", str(WT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    input_manifest = [
        {"path": str(p.relative_to(WT)), "sha256": sha256(p), "bytes": p.stat().st_size}
        for p in INPUTS
    ]

    report = {"git_sha": git_sha, "inputs": input_manifest, "sets": {}}

    variants = {
        "l31_gsc6_gsc7_r18_20260812_v1": dict(radios=("R18",)),
        "l31_gsc6_gsc7_pooled_20260812_v1": dict(radios=("R17", "R18")),
    }
    for name, kw in variants.items():
        f = U.build(include_gsc7=True, **kw)
        cond = check_comb_conditioning(f["lo_hz"])
        model = FL.fit_rung(
            f["lo_hz"], f["g1"], f["g2"], f["D"], rf_hz=f["rf_hz"],
            static_fields=FL.L31_FIELDS, n_ripples=2, name="L31", tables=tab,
        )
        model.provenance.update({
            "source_report": "spf/calibrations/dual_rx_gain_frequency/reports/"
                             "l31_gsc6_gsc7_union_20260812_v1",
            "ladder_rung": "L31 MIN (RF words: LNA, MIXER, TIA) + 2 ripples per "
                           "LNA state, NO categorical baseband-LPF family",
            "datasets": ["E-GSC6 equal_gain_diagonal_20260811_v1",
                         "E-GSC7 e_gsc7_iio_20260812_v1"],
            "radios": sorted(set(f["radio"].tolist())),
            "n_los": int(len(np.unique(f["lo_hz"]))),
            "lo_hz": sorted(float(x) for x in np.unique(f["lo_hz"])),
            "gains_db": sorted(set(f["g1"].tolist()) | set(f["g2"].tolist())),
            "anchor": "measured equal-gain cell at 26 dB, per (radio, LO); "
                      "inherited from the additive-cross fit's reference cell "
                      "and NOT re-derivable per epoch",
            "phase_convention": "angle(RX1) - angle(RX2), radians",
            "spf_git_sha": git_sha,
            "input_sha256": {m["path"]: m["sha256"] for m in input_manifest},
            "comb_conditioning": cond,
            "reconstruction": "FITTED-COEFFICIENT RECONSTRUCTION, NOT FRAMES. "
                              "E-GSC6 rows are its committed per-arm additive fit "
                              "(own residual 0.70 deg training / 0.71-0.75 deg "
                              "held-out); E-GSC7 rows are its committed "
                              "shared-effect steps at 5766 MHz ONLY. Every "
                              "holdout number is optimistic by roughly the "
                              "additive fit's own residual.",
            # Updated 2026-08-13 after E-GSC8. The carrier-transfer clause this
            # field used to carry cited E-GSC7 H5, which tested 5766 -> 5300 MHz,
            # a carrier the rover does not use. E-GSC8 tested the 74 MHz hop that
            # matters and it passes, so that clause is withdrawn rather than
            # restated. Two reasons remain, and either alone is sufficient.
            "DEPLOYMENT_STATUS": "NOT DEPLOYABLE -- see the report, including "
                                 "its 2026-08-13 addendum. Two reasons, either "
                                 "sufficient: (1) this is a single-radio fit "
                                 "with no leave-one-radio-out and no "
                                 "leave-one-epoch-out evidence, and the pooled "
                                 "two-radio fit is worse than applying no "
                                 "correction (7.347 deg LOFO vs a 7.323 deg "
                                 "anchor-only baseline); (2) the rover corpus "
                                 "has no usable equal-gain anchor, which this "
                                 "model is defined as a residual to. "
                                 "NOT a reason any more: carrier transfer. "
                                 "E-GSC8 measured 5766 -> 5840 MHz at 0.451 deg "
                                 "RMS on the clean radio (95% CI 0.329-0.554), "
                                 "and the damaged radio's 2.842 deg is 98% a "
                                 "constant -2.819 deg offset that cancels in "
                                 "this model's D = H(s1) - H(s2) form.",
        })
        path = out / f"{name}.json"
        model.save(path)
        report["sets"][name] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "tau_ns": [t * 1e9 for t in model.tau_seconds],
            "n_rows": model.provenance["n_rows"],
            "n_columns": model.provenance["n_columns"],
            "rank": model.provenance["rank"],
            "comb_conditioning": cond,
            "supported_levels": {k: [int(x) for x in v]
                                 for k, v in model.supported_levels.items()},
        }
        print(f"{name}: tau={report['sets'][name]['tau_ns']} ns  "
              f"cols={model.provenance['n_columns']} "
              f"rank={model.provenance['rank']}")
        print(f"  comb: kappa={cond['condition_number']:.2f} -> {cond['verdict']}")

    (out / "emit_manifest.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote {out}/emit_manifest.json")


if __name__ == "__main__":
    main(sys.argv[1])
