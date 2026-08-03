"""Build the analysed-input manifest required by MODEL_FITTING_AND_EVALUATION.md §10.

Hashes the scalar evidence actually analysed (the extracted per-stage npz), plus the
campaign-level provenance files. Records source Zarr identity without hashing multi-GB
LMDB stores: path, byte size, mtime, and the V7 attrs that pin firmware/serial.

Read-only. Writes only the manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import spflib as S

SCRATCH = Path(__file__).resolve().parent
CAMPAIGNS = [
    Path("/mnt/qnap01/mouse9911/share/spf_campaigns/spectroscopy_20260730_full"),
    Path("/mnt/qnap01/mouse9911/share/spf_campaigns/spectroscopy_20260730_full_r2"),
]


def sha256_file(p: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


REPO = Path(
    subprocess.run(["git", "rev-parse", "--show-toplevel"],
                   cwd=Path(__file__).resolve().parent, capture_output=True,
                   text=True).stdout.strip()
    or "/home/mouse9911/gits/spf"
)


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()


def main():
    man = {
        "schema": "spf.calibration.gain_state_phase_model.inputs",
        "schema_version": 1,
        "analysis_date": "2026-08-02",
        "software": {
            "spf_git_sha": git("rev-parse", "HEAD"),
            "spf_git_dirty_tracked": bool(
                git("status", "--porcelain", "--untracked-files=no")
            ),
            "python": "3.12",
            "pins": "numpy<2, zarr<=2.18.4, numcodecs<0.16, lmdb, scipy, matplotlib",
        },
        "radios": {
            "treated_historical_17": S.TREATED,
            "control_historical_18": S.CONTROL,
        },
        "stages_used_by_analysis": {
            "model_ladder_and_diagnostics": ["A"],
            "pooled_gain_coverage": ["A", "F", "E_tx_0", "rate_pilot"],
            "transfer_tests": ["A", "B", "C", "D", "G"],
            "excluded_abandoned_attempts": [
                "F_unsupported_negative_gain_attempt_20260730",
                "G_usb_iio_timeout_attempt_20260730",
            ],
            "excluded_other": [
                "E_tx_n7", "E_tx_n14", "E_tx_n21", "E_tx_n28",
                "E_tx_n80 (TX-muted floor control)",
                "E_anchor_* (single-frame thermal anchors)",
            ],
        },
        "campaign_provenance": [],
        "analysed_scalar_inputs": [],
        "source_zarr_stores": [],
        "analysis_code": [],
        "generated_results": [],
    }

    # campaign-level provenance files (small, hashable)
    for root in CAMPAIGNS:
        for name in ("gain_table_audit.json", "campaign_plan.json"):
            p = root / name
            if p.exists():
                man["campaign_provenance"].append({
                    "path": str(p),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })

    # the scalar evidence actually analysed
    ext = SCRATCH / "extracted"
    for npz in sorted(ext.rglob("*.npz")):
        rel = npz.relative_to(ext)
        d = np.load(npz)
        man["analysed_scalar_inputs"].append({
            "campaign": rel.parts[0],
            "stage": rel.parts[1],
            "serial": rel.stem,
            "frames": int(len(d["sweep_completed"])),
            "completed": int(d["sweep_completed"].sum()),
            "quality_valid": int(d["sweep_quality_valid"].sum()),
            "bytes": npz.stat().st_size,
            "sha256": sha256_file(npz),
        })

    # source Zarr identity (not hashed: multi-GB LMDB)
    for root in CAMPAIGNS:
        sd = root / "stages"
        if not sd.exists():
            continue
        for stage in sorted(sd.iterdir()):
            for serial in sorted(p for p in stage.iterdir() if p.is_dir()):
                z = serial / "calibration.v7.zarr" / "data.mdb"
                if not z.exists():
                    continue
                man["source_zarr_stores"].append({
                    "campaign": root.name,
                    "stage": stage.name,
                    "serial": serial.name,
                    "path": str(z.parent),
                    "data_mdb_bytes": z.stat().st_size,
                    "data_mdb_mtime_unix": int(z.stat().st_mtime),
                })

    report = (REPO / "spf/calibrations/dual_rx_gain_frequency/reports"
              / "gain_state_phase_model_20260802_v1")
    for p in sorted((report / "analysis").glob("*.py")):
        man["analysis_code"].append({
            "file": f"analysis/{p.name}",
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    for p in sorted(report.glob("*.json")):
        man["generated_results"].append({
            "file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p),
        })
    for p in sorted(report.glob("*.png")):
        man["generated_results"].append({
            "file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p),
        })

    out = report / "inputs_manifest.json"
    out.write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {out}")
    print(f"  campaign provenance : {len(man['campaign_provenance'])}")
    print(f"  scalar inputs       : {len(man['analysed_scalar_inputs'])}")
    print(f"  source zarr stores  : {len(man['source_zarr_stores'])}")
    print(f"  analysis code files : {len(man['analysis_code'])}")
    print(f"  generated results   : {len(man['generated_results'])}")
    used = {"A", "B", "C", "D", "F", "G", "E_tx_0", "rate_pilot"}
    man["frame_counts"] = {
        "all_extracted_stores": {
            "scheduled": sum(x["frames"] for x in man["analysed_scalar_inputs"]),
            "completed": sum(x["completed"] for x in man["analysed_scalar_inputs"]),
            "quality_valid": sum(x["quality_valid"] for x in man["analysed_scalar_inputs"]),
        },
        "stages_used_by_this_analysis": {
            "scheduled": sum(x["frames"] for x in man["analysed_scalar_inputs"] if x["stage"] in used),
            "completed": sum(x["completed"] for x in man["analysed_scalar_inputs"] if x["stage"] in used),
            "quality_valid": sum(x["quality_valid"] for x in man["analysed_scalar_inputs"] if x["stage"] in used),
        },
    }
    out.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man["frame_counts"], indent=1))
    tot = sum(x["frames"] for x in man["analysed_scalar_inputs"])
    don = sum(x["completed"] for x in man["analysed_scalar_inputs"])
    qv = sum(x["quality_valid"] for x in man["analysed_scalar_inputs"])
    print(f"  frames scheduled/completed/quality-valid: {tot}/{don}/{qv}")


if __name__ == "__main__":
    main()
