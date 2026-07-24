"""Generate the Taranis Q X7 control map (taranis_q_controls.png).

Recreates the operator field-guide slide ("Taranis Q controller" deck, linked from
project_spf.pdf p.57) as a repo-generated figure, with channel/switch annotations added.
Sources: README.md Taranis mixer map (CH<->switch), rover3_base_parameters.params
(MODE_CH 8, MODE1=0 Manual / MODE4=11 RTL / MODE6=15 Guided — boot-enforced by
drone_run.sh), rover3_rc_servo_parameters.params (RC5_OPTION 153 ArmDisarm,
RC6_OPTION 300 Scripting1/inert), spf/mavlink/mavlink_controller.py L897-917
(Pi-side CH7/9/10/12 handlers).
Arrow colors: gray = consumed by the ArduPilot FC, red = consumed by the Pi.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

FC_C, PI_C = "#555555", "#c0392b"
BODY, SILVER, DARK = "#2b2b2b", "#c9c9c9", "#111111"

fig, ax = plt.subplots(figsize=(12.8, 8))
ax.set_xlim(0, 1.6)
ax.set_ylim(0, 1.0)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "Taranis Q X7 — rover v3.1 control map  (gray = ArduPilot FC, red = Raspberry Pi)",
    fontsize=13, pad=12,
)


def toggle(x, y, dx, dy, cap_c=SILVER):
    """A toggle switch: dark base, angled stem, round cap. Tip at (x+dx, y+dy)."""
    ax.add_patch(Circle((x, y), 0.016, facecolor=DARK, edgecolor=SILVER, lw=1.2, zorder=5))
    ax.plot([x, x + dx], [y, y + dy], color=SILVER, lw=3.5,
            solid_capstyle="round", zorder=4)
    ax.add_patch(Circle((x + dx, y + dy), 0.011, facecolor=cap_c, edgecolor="#888",
                        lw=0.8, zorder=5))
    return (x + dx, y + dy)


def callout(text, tx, ty, px, py, color, ha="center", fs=10.5, astart=None):
    """Label + arrow. astart overrides the arrow launch point (else the text center)."""
    ax.text(tx, ty, text, ha=ha, va="center", fontsize=fs, fontweight="bold",
            color=color, linespacing=1.35)
    ax0, ay0 = astart if astart else (tx, ty)
    ax.add_patch(FancyArrowPatch((ax0, ay0), (px, py), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=2.2,
                                 shrinkA=0 if astart else 26, shrinkB=4, zorder=6))


# ------------------------------------------------------------------ transmitter body
ax.add_patch(Polygon(
    [(0.40, 0.14), (0.44, 0.66), (0.50, 0.74), (0.66, 0.70), (0.94, 0.70),
     (1.10, 0.74), (1.16, 0.66), (1.20, 0.14), (1.10, 0.08), (0.50, 0.08)],
    closed=True, facecolor=BODY, edgecolor="#000", lw=1.5, joinstyle="round"))
# antenna (folded, up-right from the right shoulder)
ax.add_patch(Polygon([(1.06, 0.735), (1.075, 0.76), (1.42, 0.87), (1.415, 0.845)],
                     closed=True, facecolor=DARK, edgecolor="#000", lw=1))

# gimbal sticks
for cx in (0.62, 0.98):
    ax.add_patch(Circle((cx, 0.44), 0.115, facecolor="#1c1c1c", edgecolor=SILVER, lw=3.5))
    ax.add_patch(Circle((cx, 0.44), 0.030, facecolor="#3a3a3a", edgecolor=SILVER, lw=1.5))
    ax.add_patch(Circle((cx, 0.44), 0.012, facecolor=SILVER, edgecolor="#777", lw=0.8))

# center face: brand, power, LCD
ax.text(0.80, 0.685, "FrSky", ha="center", fontsize=8, color="#ddd", fontweight="bold")
ax.add_patch(FancyBboxPatch((0.775, 0.55), 0.05, 0.05, boxstyle="round,pad=0.006",
                            facecolor="#1c1c1c", edgecolor=SILVER, lw=1.2))
ax.add_patch(Circle((0.80, 0.572), 0.0115, facecolor="none", edgecolor=SILVER, lw=1.4))
ax.plot([0.80, 0.80], [0.578, 0.590], color=SILVER, lw=1.6, solid_capstyle="round")
ax.text(0.80, 0.315, "X7", ha="center", fontsize=11, color="#eee", fontweight="bold")
ax.text(0.80, 0.275, "Taranis Q", ha="center", fontsize=9.5, color="#f5a623",
        fontstyle="italic", fontweight="bold")
ax.add_patch(FancyBboxPatch((0.68, 0.125), 0.24, 0.115, boxstyle="round,pad=0.008",
                            facecolor="#cfe3ee", edgecolor="#000", lw=1.2))
ax.text(0.80, 0.183, "OPEN TX", ha="center", va="center", fontsize=12,
        color="#1a6b9f", fontweight="bold")
# page/exit button + rotary
ax.add_patch(Circle((0.545, 0.175), 0.032, facecolor="#1c1c1c", edgecolor=SILVER, lw=1.5))
ax.text(0.545, 0.175, "PAGE\nEXIT", ha="center", va="center", fontsize=4.2, color="#ccc")
ax.add_patch(Circle((1.055, 0.175), 0.032, facecolor=SILVER, edgecolor="#777", lw=1.5))
# S1/S2 pots on the top face
for cx in (0.68, 0.92):
    ax.add_patch(Circle((cx, 0.715), 0.017, facecolor="#1c1c1c", edgecolor=SILVER, lw=1.4))

# ------------------------------------------------------------------ switches
sa = toggle(0.455, 0.700, -0.030, 0.045)   # left outer face — flight mode
sf = toggle(0.525, 0.745, -0.012, 0.055)   # left shoulder — arm
us = toggle(0.600, 0.720, 0.000, 0.052)    # left inner — ultrasonic
sc = toggle(1.000, 0.720, 0.000, 0.052)    # right inner — compass cal
sd = toggle(1.145, 0.700, 0.030, 0.045)    # right outer face — reboot FC
sh = toggle(1.085, 0.752, 0.014, 0.055)    # right shoulder (antenna base) — shutdown

# ------------------------------------------------------------------ callouts
callout("Arm / Disarm\nCH5 (SF)", 0.30, 0.90, *sf, FC_C)
callout("Ultrasonic ON/OFF\nCH12", 0.62, 0.945, *us, PI_C)
callout("Shutdown (momentary)\nCH9 (SH)", 1.24, 0.945, *sh, PI_C)
callout("Flight mode — CH8 (SA)\nswitch pos 1 / 4 / 6 =\nManual / RTL / Guided",
        0.155, 0.57, *sa, FC_C, astart=(0.265, 0.635))
callout("Reboot flight controller\nCH7 (SD)", 1.44, 0.62, *sd, PI_C, fs=10)
callout("Compass calibration\nCH10 (SC)", 1.38, 0.38, *sc, PI_C)

# ------------------------------------------------------------------ footnotes
ax.text(0.02, 0.055,
        "Sticks CH1–4 = Ail / Ele / Thr / Rud (RCMAP 1-4).  CH6 (S2 pot): RC6_OPTION 300"
        " (Scripting1) — inert, no Lua script in-tree.  CH12 switch id not recorded in-tree.",
        fontsize=8.2, color="#333")
ax.text(0.02, 0.015,
        "Mode order = boot-enforced rover3_base_parameters.params (MODE_CH 8: slot1=Manual(0),"
        " slot4=RTL(11), slot6=Guided(15)).  Pi-side actions: mavlink_controller.py"
        " handle_RC_CHANNELS (L897-917).",
        fontsize=8.2, color="#333")

fig.tight_layout()
out = __file__.replace("make_taranis_map.py", "taranis_q_controls.png")
fig.savefig(out, dpi=150)
print("wrote", out)
