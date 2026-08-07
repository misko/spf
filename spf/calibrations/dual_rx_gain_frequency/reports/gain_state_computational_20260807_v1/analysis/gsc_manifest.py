"""Build inputs_manifest.json for the E-GSC computational report.

Hashes every analysed scalar input, every committed artifact this analysis read,
every analysis script, and every generated result. Source Zarr stores are
identified by path/size/mtime rather than hashed, per this directory's
convention -- they are multi-GB LMDB and live outside Git.

Read-only with respect to the campaigns. Writes only the manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent
REPO = Path("/home/mouse9911/gits/spf")
REPORT = (
    REPO / "spf/calibrations/dual_rx_gain_frequency/reports"
    / "gain_state_computational_20260807_v1"
)
CAMPAIGNS = [
    Path("/mnt/qnap01/mouse9911/share/spf_campaigns/spectroscopy_20260730_full"),
    Path("/mnt/qnap01/mouse9911/share/spf_campaigns/spectroscopy_20260730_full_r2"),
    Path("/mnt/qnap01/mouse9911/share/spf_campaigns/gain_state_followups_20260807_v1"),
]
COMMITTED_INPUTS = [
    "spf/calibrations/dual_rx_gain_frequency/reports/"
    "wide_integer_gain_cross_band_20260730_v1/model_matrix.json",
    "spf/calibrations/gain_state_phase_model_v1/gain_tables_audited.json",
    "spf/calibrations/gain_state_phase_model_v1/coefficients/l26_stage_a_v1.json",
    "spf/calibrations/gain_state_phase_model_v1/coefficients/l26_pooled_v1.json",
    "spf/calibrations/gain_state_phase_model_v1/coefficients/l30_pooled_v1.json",
    "spf/calibrations/gain_state_phase_model_v1/coefficients/l31_pooled_v1.json",
    "spf/calibrations/gain_state_phase_model_v1/model.py",
    "spf/calibrations/gain_state_phase_model_v1/gain_tables.py",
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


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def main():
    man = {
        "schema": "spf.calibration.gain_state_computational.inputs",
        "schema_version": 1,
        "analysis_date": "2026-08-07",
        "report": "spf/calibrations/dual_rx_gain_frequency/reports/"
                  "gain_state_computational_20260807_v1",
        "software": {
            "spf_git_sha": git("rev-parse", "HEAD"),
            "spf_git_dirty_tracked": bool(
                git("status", "--porcelain", "--untracked-files=no")
            ),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "interpreter": sys.executable,
        },
        "radios": {
            "treated_historical_17": "104000bac4950008230026001b440a003a",
            "control_historical_18": "1040007c4a94000211000b009186843ef2",
        },
        "stages_used": {
            "E-GSC2 identifiability": ["A", "E_CAL3_PROSPECTIVE_DENSE"],
            "E-GSC3 gap decomposition": [
                "A", "B", "C", "D", "G", "E_CAL3_PROSPECTIVE_DENSE",
            ],
            "E-GSC5 selection (A-G only)": [
                "A", "F", "E_tx_0", "rate_pilot",
            ],
            "E-GSC5 confirmation (prospective)": [
                "E_CAL3_PROSPECTIVE_DENSE", "E_CAL2_LOW", "E_CAL2_MIDDLE",
                "E_CAL2_HIGH",
            ],
            "E-GSC1 / E-GSC4 (committed coefficients only)": [
                "wide_integer_gain_cross_band_20260730_v1/model_matrix.json",
            ],
        },
        "campaign_provenance": [],
        "committed_repo_inputs": [],
        "analysed_scalar_inputs": [],
        "source_zarr_stores": [],
        "analysis_code": [],
        "generated_results": [],
    }

    for root in CAMPAIGNS:
        for name in (
            "gain_table_audit.json", "gain_table_audit_final.json",
            "campaign_plan.json",
        ):
            p = root / name
            if p.exists():
                man["campaign_provenance"].append(
                    {
                        "campaign": root.name,
                        "path": str(p),
                        "bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    }
                )

    for rel in COMMITTED_INPUTS:
        p = REPO / rel
        man["committed_repo_inputs"].append(
            {"file": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        )

    ext = SCRATCH / "extracted"
    for npz in sorted(ext.rglob("*.npz")):
        rel = npz.relative_to(ext)
        d = np.load(npz)
        man["analysed_scalar_inputs"].append(
            {
                "campaign": rel.parts[0],
                "stage": rel.parts[1],
                "serial": rel.stem,
                "frames": int(len(d["sweep_completed"])),
                "completed": int(d["sweep_completed"].sum()),
                "quality_valid": int(d["sweep_quality_valid"].sum()),
                "bytes": npz.stat().st_size,
                "sha256": sha256_file(npz),
            }
        )

    for root in CAMPAIGNS:
        sd = root / "stages"
        if not sd.exists():
            continue
        for stage in sorted(sd.iterdir()):
            if not stage.is_dir():
                continue
            for serial in sorted(p for p in stage.iterdir() if p.is_dir()):
                z = serial / "calibration.v7.zarr" / "data.mdb"
                if not z.exists():
                    continue
                man["source_zarr_stores"].append(
                    {
                        "campaign": root.name,
                        "stage": stage.name,
                        "serial": serial.name,
                        "path": str(z.parent),
                        "data_mdb_bytes": z.stat().st_size,
                        "data_mdb_mtime_unix": int(z.stat().st_mtime),
                        "opened": "read-only (zarr.LMDBStore readonly=True, "
                                  "lock=False); never written",
                    }
                )

    for p in sorted((REPORT / "analysis").glob("*.py")):
        man["analysis_code"].append(
            {
                "file": f"analysis/{p.name}",
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    for pat in ("*.json", "*.png", "*.md"):
        for p in sorted(REPORT.glob(pat)):
            if p.name == "inputs_manifest.json":
                continue
            man["generated_results"].append(
                {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            )

    man["absent_raw_stores"] = {
        "checked_on": "2026-08-07",
        "note": "the wide 53-LO integer-gain survey and the 2.4 GHz integer-gain "
                "runs no longer have raw stores on this machine",
        "expected_locations": [
            "artifacts/dual_rx_gain_frequency/"
            "overnight_wide_integer_gain_cross_20260730_special_17_18_v1/",
            "artifacts/dual_rx_gain_frequency/"
            "integer_gain_cross_2p4_20260729_special_17_18_v1/",
        ],
        "verified_absent": True,
        "search_performed": "artifacts/ contains only direct_usb_gain_metadata and "
                            "direct_usb_stability; no *.v7.zarr exists under "
                            "/mnt/{4tb_ssd,data,md0,md1,md2,ssd,usb_drive,backblaze}; "
                            "/mnt/qnap01/.../spf_campaigns holds only the two "
                            "spectroscopy campaigns and the 2026-08-07 follow-up",
    }

    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / "inputs_manifest.json"
    out.write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {out}")
    for k in (
        "campaign_provenance", "committed_repo_inputs", "analysed_scalar_inputs",
        "source_zarr_stores", "analysis_code", "generated_results",
    ):
        print(f"  {k:26s}: {len(man[k])}")


if __name__ == "__main__":
    main()
