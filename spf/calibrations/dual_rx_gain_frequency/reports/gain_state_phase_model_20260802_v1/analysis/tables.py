"""Emit markdown tables for the report."""

from __future__ import annotations

import json
from pathlib import Path

# results live beside the report when run from analysis/, else in cwd
_HERE = Path(__file__).resolve().parent
_R = _HERE.parent if (_HERE.parent / "ladder_results_A_main.json").exists() else Path(".")
MAIN = json.load(open(_R / "ladder_results_A_main.json"))
MIN = json.load(open(_R / "min_results.json"))
COMB = json.load(open(_R / "comb_results.json"))
BAND = json.load(open(_R / "band_results.json"))

SPLITS = ["LOEO leave-one-epoch-out", "LOFO leave-one-frequency-out",
          "LOBLOCK leave-frequency-block-out", "LORO leave-one-radio-out"]
SHORT = {"LOEO leave-one-epoch-out": "LOEO",
         "LOFO leave-one-frequency-out": "LOFO",
         "LOBLOCK leave-frequency-block-out": "LOBLK",
         "LORO leave-one-radio-out": "LORO",
         "LOBAND leave-one-gain-table-band-out": "LOBAND"}


def main_table():
    idx = {}
    for sp in SPLITS + ["LOBAND leave-one-gain-table-band-out"]:
        for r in MAIN[sp]:
            idx.setdefault(r["model"], {})[SHORT[sp]] = r
    order = sorted(idx, key=lambda m: m.split()[0])
    print("| Model | Params | LOEO MAE / P95 | LOFO MAE / P95 | LOBLK MAE | "
          "LORO MAE | cov (LOFO) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m in order:
        d = idx[m]

        def cell(k, p95=True):
            r = d.get(k)
            if r is None:
                return "—"
            if r["coverage"] < 0.005:
                return "**unsupported**"
            s = f"{r['mae_deg']:.2f}"
            if p95:
                s += f" / {r['p95_deg']:.2f}"
            return s

        pr = d.get("LOEO") or d.get("LOFO") or next(iter(d.values()))
        cov = d["LOFO"]["coverage"] if "LOFO" in d else float("nan")
        print(f"| {m} | {pr['params']} | {cell('LOEO')} | {cell('LOFO')} | "
              f"{cell('LOBLK', False)} | {cell('LORO', False)} | {cov:.2f} |")


def min_table():
    print("\n\n### Minimal models, pooled A+F+E+pilot leave-one-frequency-out\n")
    rows = MIN.get("pooled_LOFO", [])
    print("| Model | Params | MAE | P95 | coverage |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['model']} | {r['params']} | {r['mae_deg']:.2f} | "
              f"{r['p95_deg']:.2f} | {r['coverage']:.2f} |")

    print("\n\n### Leave-one-PROBE-GAIN-out, pooled (can the model predict an "
          "unmeasured requested gain?)\n")
    g = MIN.get("leave_one_gain_out_pooled", {})
    print("| Model | Params | coverage | fail-closed MAE | supported MAE | "
          "supported P95 |")
    print("|---|---:|---:|---:|---:|---:|")
    for k, v in g.items():
        print(f"| {k} | {v['params']} | {v['coverage']:.3f} | "
              f"{v['failclosed_mae_deg']:.2f} | "
              f"{v.get('sup_mae_deg', float('nan')):.2f} | "
              f"{v.get('sup_p95_deg', float('nan')):.2f} |")


def band_table():
    print("\n\n### Pooled leave-one-gain-table-band-out (extrapolation across "
          "a whole band)\n")
    print("| Model | Params | coverage | fail-closed MAE | P95 |")
    print("|---|---:|---:|---:|---:|")
    for r in BAND["pooled_LOBAND"]:
        print(f"| {r['model']} | {r['params']} | {r['coverage']:.2f} | "
              f"{r['mae_deg']:.2f} | {r['p95_deg']:.2f} |")


def comb_table():
    print("\n\n### Calibration comb spacing (stage A, fail-closed MAE, deg)\n")
    widths = ["2750", "1375", "688", "344", "172", "96"]
    keys = ["2blocks", "4blocks", "8blocks", "16blocks", "32blocks", "57blocks"]
    print("| Model | " + " | ".join(f"{w} MHz gap" for w in widths) + " |")
    print("|---|" + "---:|" * len(widths))
    for name, d in COMB["comb"].items():
        print(f"| {name} | " + " | ".join(f"{d[k]:.2f}" for k in keys) + " |")

    print("\n\n### Which parameters must be radio-specific?\n")
    print("| Variant | Params | LOEO MAE | LORO MAE | LORO coverage |")
    print("|---|---:|---:|---:|---:|")
    for k, v in COMB["radio_specificity"].items():
        print(f"| {k} | {v['loeo']['params']} | {v['loeo']['mae_deg']:.3f} | "
              f"{v['loro']['mae_deg']:.3f} | {v['loro']['coverage']:.2f} |")


if __name__ == "__main__":
    main_table()
    min_table()
    band_table()
    comb_table()
