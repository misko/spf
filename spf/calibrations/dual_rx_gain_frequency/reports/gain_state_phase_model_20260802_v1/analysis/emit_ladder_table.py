"""Emit the report's section 4 ladder table from the regenerated results, and
splice it into REPORT.md at the <!--LADDER_TABLE--> marker."""

from __future__ import annotations

import json
from pathlib import Path

REPORT = Path("/home/mouse9911/gits/spf/spf/calibrations/dual_rx_gain_frequency"
              "/reports/gain_state_phase_model_20260802_v1")
MAIN = json.load(open(REPORT / "ladder_results_A_main.json"))

SHORT = {"LOEO leave-one-epoch-out": "LOEO",
         "LOFO leave-one-frequency-out": "LOFO",
         "LOBLOCK leave-frequency-block-out": "LOBLK",
         "LORO leave-one-radio-out": "LORO",
         "LOBAND leave-one-gain-table-band-out": "LOBAND"}

# the rungs shown in the report body; the rest are in full_result_tables.md
SHOWN = ["L00", "L01", "L03", "L05", "L06", "L08", "L11", "L14", "L16", "L18",
         "L26", "L27", "L29", "L30", "L31", "L33", "L21", "L22", "L23", "L24"]
BOLD = {"L16", "L26", "L27", "L33"}


def main():
    idx = {}
    for sp, rows in MAIN.items():
        for r in rows:
            idx.setdefault(r["model"], {})[SHORT[sp]] = r

    lines = [
        "| Model | Params | LOEO MAE / uneq | LOFO MAE / uneq | LOBLK MAE | "
        "LORO MAE | LOBAND |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in SHOWN:
        name = next((m for m in idx if m.split()[0] == key), None)
        if name is None:
            continue
        d = idx[name]

        def cell(k, pair=True):
            r = d.get(k)
            if r is None:
                return "—"
            if r["coverage"] < 0.005:
                return "**fails closed**"
            if pair:
                return f"{r['mae_deg']:.2f} / {r['unequal_mae_deg']:.2f}"
            return f"{r['mae_deg']:.2f}"

        p = d["LOEO"]["params"]
        label = f"**{name}**" if key in BOLD else name
        params = f"**{p}**" if key in BOLD else str(p)
        lines.append(
            f"| {label} | {params} | {cell('LOEO')} | {cell('LOFO')} | "
            f"{cell('LOBLK', False)} | {cell('LORO', False)} | "
            f"{cell('LOBAND', False)} |"
        )
    table = "\n".join(lines)

    rp = REPORT / "REPORT.md"
    s = rp.read_text()
    assert "<!--LADDER_TABLE-->" in s, "marker missing"
    s = s.replace("<!--LADDER_TABLE-->", table)
    rp.write_text(s)
    print(table)


if __name__ == "__main__":
    main()
