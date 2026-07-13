"""Generate the rover v3.1 system schematic (rover_v3p1_schematic.png).

Sources: rover_v3.1/README.md (LPD setpoints, Cytron wiring, USB map, SiK NetIDs),
drone_run.sh / setup.sh (boot + network), rover3_base_parameters.params (ArduPilot),
capture_configs/*.yaml (radios), lab log (buck-regulator redesign, switch replacement).
Edges: red = power, blue = data/control, green = RF, gray = mechanical/PWM.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

POW, DATA, RF, MECH = "#c0392b", "#2b6cb0", "#2e8b57", "#555555"


def box(ax, x, y, w, h, title, lines=(), fc="#f8f8f8", ec="#333", title_c="#111"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                       facecolor=fc, edgecolor=ec, linewidth=1.2)
    )
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=title_c)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.045 - i * 0.0245, ln, ha="center", va="top",
                fontsize=7.6, color="#222")
    return (x, y, w, h)


def edge(ax, p0, p1, color, lw=1.8, style="-", label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=lw, linestyle=style,
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly, label,
                fontsize=7.2, color=color, ha="center")


def right(b):  # (x,y) middle of right edge
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


fig, ax = plt.subplots(figsize=(17, 10.5))
ax.set_xlim(0, 1.62)
ax.set_ylim(0, 1.0)
ax.axis("off")
ax.set_title("SPF Rover v3.1 — system schematic  (red=power, blue=data/control, green=RF, gray=drive)",
             fontsize=13, pad=14)

# ------------------------------------------------------------------ power column
batt = box(ax, 0.02, 0.72, 0.20, 0.20, "Li-ion pack (3S)",
           ["~11.1 V nominal / 12.6 V full", "velcro battery compartment",
            "(cell count inferred from", "LPD setpoints)"], fc="#fdeaea")
sw = box(ax, 0.02, 0.575, 0.20, 0.095, "Main power switch",
         ["(all replaced Apr 2025", "after field failures)"], fc="#fdeaea")
lpd = box(ax, 0.02, 0.39, 0.20, 0.135, "Low-power disconnect 30 A",
          ["warn 11.2 V / cut 11.1 V", "sole battery protection", "⚠ 0.1 V hysteresis"],
          fc="#fdeaea")
buck = box(ax, 0.02, 0.20, 0.20, 0.135, "Buck regulators ×N",
           ["10–32 V → 5 V USB", "(Feb-2024 redesign,", "no custom soldering)"],
           fc="#fdeaea")
fans = box(ax, 0.02, 0.06, 0.20, 0.09, "Cooling fans", ["Cooler Guys"], fc="#fdeaea")

edge(ax, bottom(batt), top(sw), POW)
edge(ax, bottom(sw), top(lpd), POW)
edge(ax, bottom(lpd), top(buck), POW, label="pack V", lx=-0.045)
edge(ax, bottom(buck), top(fans), POW, label="5 V", lx=-0.03)

# ------------------------------------------------------------------ drive
cytron = box(ax, 0.32, 0.70, 0.24, 0.22, "Cytron MDDS30 motor driver",
             ["30 A dual channel", "DIP 3,4,6 = HIGH",
              "MLA/MLB → left pair", "MRA/MRB → right pair"], fc="#f2f2f2")
ml = box(ax, 0.65, 0.83, 0.17, 0.09, "Left motors ×2", ["GoBilda Recon"], fc="#eee")
mr = box(ax, 0.65, 0.70, 0.17, 0.09, "Right motors ×2", ["skid steer"], fc="#eee")
edge(ax, right(lpd), (0.32, 0.79), POW, label="pack V", lx=0, ly=0.015)
edge(ax, (0.56, 0.855), left(ml), MECH)
edge(ax, (0.56, 0.745), left(mr), MECH)

# ------------------------------------------------------------------ autopilot
fc_ = box(ax, 0.32, 0.36, 0.24, 0.27, "Flight controller (ArduPilot Rover)",
          ["params enforced at boot", "(load + diff, per-rover SYSID)",
           "CRUISE 1.5 m/s, TURN_R 5 m",
           "⚠ no BATT_* monitor/failsafe", "⚠ FS_THR disabled"],
          fc="#eaf0fa")
gps = box(ax, 0.62, 0.55, 0.185, 0.10, "GPS + compass",
          ["cable routed away from", "Pi/SDR (EMI!)  magcal era"], fc="#eaf0fa")
rc = box(ax, 0.62, 0.43, 0.185, 0.10, "FrSky X8R RC",
         ["Taranis: arm / mode /", "shutdown / mag-cal"], fc="#eaf0fa")
sik = box(ax, 0.62, 0.31, 0.185, 0.10, "SiK telemetry",
          ["NetID 25/32/39", "↔ base mavproxy"], fc="#eaf0fa")
buzz = box(ax, 0.62, 0.21, 0.185, 0.078, "Buzzer", ["field status chirps"], fc="#eaf0fa")
edge(ax, (0.44, 0.63), (0.44, 0.70), MECH, label="PWM (skid)", lx=0.065)
edge(ax, right(fc_), left(gps), DATA)
edge(ax, right(fc_), left(rc), DATA)
edge(ax, right(fc_), left(sik), DATA)
edge(ax, right(fc_), left(buzz), DATA)
edge(ax, (0.22, 0.245), (0.32, 0.44), POW, label="5 V", lx=0.01, ly=0.02)

# ------------------------------------------------------------------ compute
pi = box(ax, 0.88, 0.36, 0.26, 0.27, "Raspberry Pi 4",
         ["eth0 192.168.1.4x (bench)", "Wi-Fi DISABLED (2.4 GHz)",
          "boot: git self-update → params", "→ GPS time → capture loop",
          "USB2 = Radio A | USB1 = Radio B"], fc="#eafaf0")
us = box(ax, 0.88, 0.21, 0.26, 0.09, "Ultrasonic ranger",
         ["GPIO 2/3 obstacle stop"], fc="#eafaf0")
edge(ax, (0.22, 0.28), (0.88, 0.47), POW, label="5 V USB", lx=0.0, ly=0.015)
edge(ax, (0.56, 0.54), (0.88, 0.54), DATA, label="MAVLink", lx=-0.13, ly=0.012)
edge(ax, bottom(pi), top(us), DATA)

# ------------------------------------------------------------------ radios
ra = box(ax, 1.24, 0.70, 0.245, 0.185, "PlutoPlus SDR — Radio A",
         ["2r2t fw (checked at boot)", "2-el array 35/43/47 mm",
          "5.766 GHz, 30 MS/s, 3 MHz BW", "slow_attack AGC"], fc="#eafaf0")
rb = box(ax, 1.24, 0.47, 0.245, 0.185, "PlutoPlus SDR — Radio B",
         ["mounted at different θ", "(rover 2: single radio)",
          "usb-gadget net 192.168.2.1"], fc="#eafaf0")
tx = box(ax, 1.24, 0.24, 0.245, 0.16, "Emitter (rover 2 only)",
         ["ESP32 'tone blaster' /", "5.8 GHz video TX", "the tracked signal"],
         fc="#fef7e0")
ant = box(ax, 1.24, 0.05, 0.245, 0.13, "Antennas",
          ["2 per radio (phase pair)", "+ GPS patch + SiK whip + RC"],
          fc="#ffffff")
edge(ax, right(pi), left(ra), DATA, label="USB2", lx=0, ly=0.02)
edge(ax, (1.14, 0.55), left(rb), DATA, label="USB1", lx=-0.01, ly=0.02)
edge(ax, right(ra), (1.55, 0.79), RF)
edge(ax, right(rb), (1.55, 0.56), RF)
edge(ax, right(tx), (1.55, 0.32), RF, style="--")
ax.text(1.555, 0.68, "RF", color=RF, fontsize=9, rotation=90)

# roles footnote
ax.text(0.02, 0.012,
        "Roles: rover1 = RX dual 35mm 'bounce' | rover2 = TX + single RX 'circle' | rover3 = RX dual 43mm 'bounce'."
        "  Boot chain: mavlink_controller.service → drone_run.sh (infinite mission loop, GPS time resync)."
        "  ⚠ = review findings 2026-07-13.",
        fontsize=8.2, color="#333")

fig.tight_layout()
out = __file__.replace("make_schematic.py", "rover_v3p1_schematic.png")
fig.savefig(out, dpi=140)
print("wrote", out)
