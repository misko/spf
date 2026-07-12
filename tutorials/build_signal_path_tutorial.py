#!/usr/bin/env python3
"""
Generate `spf_signal_path.pdf` — a mathematical overview of the SPF signal path:
transmission -> propagation/array geometry -> receiving & down-conversion ->
phase difference -> beamforming -> the SPF processing stack.

Pure matplotlib (mathtext + PdfPages), no LaTeX install required.
Run:  /home/mouse9911/virtual-envs/spf/bin/python tutorials/build_signal_path_tutorial.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

# ---------- page geometry (A4 portrait, inches) ----------
PAGE_W, PAGE_H = 8.27, 11.69
LEFT, RIGHT, TOP, BOT = 0.085, 0.93, 0.945, 0.06
BODY = 10.5
INK = "#1a1a1a"
ACC = "#1f5fa8"
MUT = "#666666"


import re as _re
def _wrap_tokens(text, wrap):
    # keep `$...$` (with internal spaces + trailing punct) as one atomic token
    tokens = _re.findall(r"\$[^$]*\$[.,;:)]?|\S+", text)
    lines, line = [], ""
    for t in tokens:
        if line and len(line) + len(t) + 1 > wrap:
            lines.append(line); line = t
        else:
            line = (line + " " + t).strip()
    if line:
        lines.append(line)
    return lines


class Doc:
    def __init__(self, path):
        self.pdf = PdfPages(path)
        self.pageno = 0
        self._new_page(first=True)

    def _new_page(self, first=False):
        if not first:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = TOP
        self.pageno += 1

    def _footer(self):
        self.fig.text(0.5, 0.032, f"SPF · The Signal Path · page {self.pageno}",
                      ha="center", fontsize=8, color=MUT)

    def _room(self, need):
        if self.y - need < BOT:
            self._new_page()

    def spacer(self, h=0.012):
        self.y -= h

    def h1(self, text):
        self._room(0.075)
        self.y -= 0.006
        self.fig.text(LEFT, self.y, text, fontsize=15.5, fontweight="bold",
                      color=ACC, va="top")
        self.y -= 0.010
        self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [self.y, self.y],
                                       color=ACC, lw=0.8, alpha=0.5))
        self.y -= 0.024

    def h2(self, text):
        self._room(0.06)
        self.y -= 0.004
        self.fig.text(LEFT, self.y, text, fontsize=12, fontweight="bold",
                      color=INK, va="top")
        self.y -= 0.028

    def p(self, text, gap=0.010, size=BODY, color=INK, indent=0.0):
        lines = _wrap_tokens(text, 96)
        lh = 0.0182 * (size / BODY)
        self._room(lh * len(lines) + gap)
        for ln in lines:
            self.fig.text(LEFT + indent, self.y, ln, fontsize=size, color=color,
                          va="top")
            self.y -= lh
        self.y -= gap

    def bullet(self, text):
        self.p(r"$\bullet$  " + text, gap=0.006, indent=0.012)

    def eq(self, mathstr, size=13):
        self._room(0.040)
        self.y -= 0.006
        self.fig.text(0.5, self.y, "$" + mathstr + "$", fontsize=size,
                      color=INK, ha="center", va="top")
        self.y -= 0.030
        self.y -= 0.006

    def note(self, text):
        # boxed side-note
        lines = _wrap_tokens(text, 92)
        lh = 0.017
        h = lh * len(lines) + 0.022
        self._room(h + 0.01)
        top = self.y
        self.fig.patches.append(
            Rectangle((LEFT, top - h), RIGHT - LEFT, h, transform=self.fig.transFigure,
                      facecolor="#eef4fb", edgecolor=ACC, lw=0.6, alpha=0.9, zorder=0))
        yy = top - 0.014
        for ln in lines:
            self.fig.text(LEFT + 0.012, yy, ln, fontsize=9.5, color="#123", va="top")
            yy -= lh
        self.y = top - h - 0.012

    def diagram(self, draw_fn, h=0.24, w=0.72, pad_top=0.026, pad_bot=0.034):
        self._room(h + pad_top + pad_bot + 0.01)
        self.y -= pad_top
        left = LEFT + (RIGHT - LEFT - w) / 2
        ax = self.fig.add_axes([left, self.y - h, w, h])
        draw_fn(ax)
        self.y -= h + pad_bot

    def end(self):
        self._footer()
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.pdf.close()


# ---------------- diagrams ----------------
def diag_geometry(ax):
    ax.axis("off")
    ax.set_xlim(-1, 10); ax.set_ylim(-1.5, 6)
    # two elements on x-axis
    e0, e1 = (2.0, 0.0), (6.0, 0.0)
    for (x, y), lab in [(e0, "el. 0"), (e1, "el. 1")]:
        ax.plot(x, y, "o", ms=9, color=ACC)
        ax.text(x, y - 0.7, lab, ha="center", fontsize=9)
    ax.annotate("", xy=e1, xytext=e0,
                arrowprops=dict(arrowstyle="<->", color=MUT))
    ax.text(4.0, -0.55, r"$d$ (spacing)", ha="center", fontsize=10, color=MUT)
    # incoming plane-wave direction from angle theta (from array normal, +y)
    th = np.deg2rad(35)
    d = np.array([np.sin(th), np.cos(th)])  # unit propagation dir toward array
    # wavefronts (perpendicular to d)
    perp = np.array([np.cos(th), -np.sin(th)])
    for k, c in [(6.2, "#bbb"), (5.2, "#bbb"), (4.2, ACC)]:
        base = np.array([4.0, 0.3]) + d * k
        p1 = base + perp * 3.2
        p2 = base - perp * 3.2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, lw=1.4)
    # arrow of propagation
    a0 = np.array([4.0, 0.3]) + d * 6.6
    ax.annotate("", xy=np.array([4.0, 0.3]) + d * 3.6, xytext=a0,
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.6))
    ax.text(a0[0] + 0.1, a0[1] + 0.1, "plane wave", color=ACC, fontsize=9)
    # normal + theta
    ax.plot([4, 4], [0, 4.6], ls=":", color=MUT, lw=1)
    ax.text(4.05, 4.7, "array normal", color=MUT, fontsize=8)
    ax.annotate(r"$\theta$", xy=(4.55, 2.0), fontsize=13, color=INK)
    # extra path to element 1
    foot = e1 - perp * np.dot((np.array(e1) - np.array(e0)), perp)
    ax.plot([e0[0], foot[0]], [e0[1], foot[1]], color="#c0392b", lw=1.2)
    ax.text(4.6, 0.75, r"extra path $=d\sin\theta$", color="#c0392b", fontsize=9)
    ax.set_title(r"Fig. 1  Plane wave on a 2-element array:  "
                 r"$\phi=2\pi\,(d/\lambda)\sin\theta$", fontsize=10)


def diag_downconv(ax):
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    # RF axis
    ax.annotate("", xy=(9.4, 1.0), xytext=(0.4, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.text(9.5, 1.0, "freq", fontsize=9, va="center")
    ax.text(0.3, 0.5, "RF", fontsize=9, color=MUT)
    ax.plot([6.6], [1.0], marker="^", ms=10, color="#c0392b")
    ax.text(6.6, 1.35, r"$f_c$ (tone)", ha="center", fontsize=9, color="#c0392b")
    ax.plot([6.0, 6.0], [1.0, 2.0], ls=":", color=ACC)
    ax.plot([6.0], [1.0], marker="v", ms=9, color=ACC)
    ax.text(6.0, 0.55, r"$f_{LO}$", ha="center", fontsize=9, color=ACC)
    # baseband axis
    ax.annotate("", xy=(9.4, 3.6), xytext=(0.4, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.text(9.5, 3.6, "freq", fontsize=9, va="center")
    ax.text(0.3, 3.1, "baseband", fontsize=9, color=MUT)
    ax.plot([0.6, 0.6], [3.6, 4.4], ls=":", color="#999")
    ax.text(0.6, 4.5, "DC", ha="center", fontsize=8, color=MUT)
    ax.plot([1.2], [3.6], marker="^", ms=10, color="#c0392b")
    ax.text(1.2, 3.9, r"$\Delta f=f_c-f_{LO}$", ha="left", fontsize=9, color="#c0392b")
    ax.annotate("", xy=(1.2, 3.4), xytext=(6.6, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=MUT, ls="--", lw=1))
    ax.text(3.4, 2.2, r"mix by $e^{-j2\pi f_{LO}t}$ + LPF", fontsize=9, color=MUT,
            rotation=0)
    ax.set_title(r"Fig. 2  Down-conversion shifts $f_c\!\to\!\Delta f$; "
                 r"the spatial phase $\phi$ rides through unchanged", fontsize=10)


def diag_beamformer(ax):
    th = np.linspace(-np.pi, np.pi, 65)
    true = np.deg2rad(30)
    # bimodal (front/back) response
    p = np.exp(-((np.sin(th) - np.sin(true)) ** 2) / (2 * 0.08 ** 2))
    p = p / p.max()
    ax.plot(np.rad2deg(th), p, color=ACC, lw=1.8)
    ax.axvline(np.rad2deg(true), color="#c0392b", ls="--", lw=1, label=r"true $\theta$")
    ax.axvline(np.rad2deg(np.pi - true), color="#888", ls=":", lw=1,
               label=r"mirror $\pi-\theta$")
    ax.set_xlabel(r"steering angle $\theta$ (deg)", fontsize=9)
    ax.set_ylabel(r"power $P(\theta)$", fontsize=9)
    ax.set_xlim(-180, 180); ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(r"Fig. 3  Beamformer power spectrum: peak at $\theta$ and its "
                 r"front/back mirror", fontsize=10)


def diag_stack(ax):
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    boxes = [
        ("raw IQ\n(2 x N)", "#eef4fb"),
        ("detrend +\nwindow", "#eef4fb"),
        ("per-window\ndelay-and-sum", "#dcebfb"),
        ("windowed\nbeamformer\n+ circ. stats", "#dcebfb"),
        ("NN\n(single/paired)", "#e8f6ec"),
        (r"$P(\theta)$", "#e8f6ec"),
    ]
    x = 0.2
    w = 1.5
    for i, (t, c) in enumerate(boxes):
        ax.add_patch(Rectangle((x, 0.9), w, 1.2, facecolor=c, edgecolor="#456",
                               lw=0.8))
        ax.text(x + w / 2, 1.5, t, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 0.12, 1.5), xytext=(x + w - 0.02, 1.5),
                        arrowprops=dict(arrowstyle="-|>", color="#456"))
        x += w + 0.13
    ax.set_title("Fig. 4  The SPF processing stack (stage-1 path)", fontsize=10)


def diag_iq(ax):
    ax.axis("off"); ax.set_xlim(0, 10.4); ax.set_ylim(0, 6)
    # RF in + split
    ax.annotate("", xy=(1.3, 3), xytext=(0.2, 3),
                arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.text(0.15, 3.4, "RF in", fontsize=8, color=MUT)
    ax.plot([1.3, 1.3], [1.5, 4.5], color=INK, lw=1)
    ax.plot(1.3, 3, "o", ms=3, color=INK)
    for yy, lo, br, sgn in [(4.5, r"$\cos\,\omega_{LO}t$", "I", 1),
                            (1.5, r"$-\sin\,\omega_{LO}t$", "Q", -1)]:
        ax.plot([1.3, 2.55], [yy, yy], color=INK, lw=1)
        ax.add_patch(plt.Circle((2.85, yy), 0.27, fill=False, color=INK, lw=1.2))
        ax.text(2.85, yy, r"$\times$", ha="center", va="center", fontsize=12)
        ax.annotate("", xy=(2.85, yy + sgn * 0.32), xytext=(2.85, yy + sgn * 0.82),
                    arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1))
        ax.text(2.85, yy + sgn * 1.02, lo, ha="center", va="center",
                fontsize=8, color=ACC)
        ax.plot([3.12, 3.75], [yy, yy], color=INK, lw=1)
        ax.add_patch(Rectangle((3.75, yy - 0.27), 0.95, 0.54, fill=False,
                               color=INK, lw=1))
        ax.text(4.22, yy, "LPF", ha="center", va="center", fontsize=8)
        ax.plot([4.7, 5.5], [yy, yy], color=INK, lw=1)
        ax.text(5.6, yy, br + r"$(t)$", fontsize=10, va="center", color="#c0392b")
    ax.text(1.75, 3, r"$90^\circ$", ha="center", fontsize=8, color=MUT)
    # combine -> z
    ax.plot([6.15, 6.7], [4.5, 3.15], color=INK, lw=0.8)
    ax.plot([6.15, 6.7], [1.5, 2.85], color=INK, lw=0.8)
    ax.text(7.35, 3, r"$z=I+jQ$", fontsize=11, va="center", ha="center")
    # phasor inset
    ox, oy = 8.9, 3.0
    ax.annotate("", xy=(ox + 1.3, oy), xytext=(ox - 0.4, oy),
                arrowprops=dict(arrowstyle="-|>", color=MUT))
    ax.annotate("", xy=(ox, oy + 1.7), xytext=(ox, oy - 1.1),
                arrowprops=dict(arrowstyle="-|>", color=MUT))
    ax.text(ox + 1.35, oy - 0.28, r"$I$", fontsize=9, color=MUT)
    ax.text(ox + 0.12, oy + 1.7, r"$Q$", fontsize=9, color=MUT)
    zx, zy = ox + 0.95, oy + 1.15
    ax.annotate("", xy=(zx, zy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.6))
    ax.plot([zx, zx], [oy, zy], ls=":", color="#999", lw=0.8)
    ax.plot([ox, zx], [zy, zy], ls=":", color="#999", lw=0.8)
    ax.text(zx + 0.05, zy + 0.05, r"$z=A\,e^{j\phi}$", fontsize=9, color=ACC)
    ax.text((ox + zx) / 2 - 0.1, oy - 0.32, r"$I$", fontsize=8, color="#777")
    ax.text(zx + 0.08, (oy + zy) / 2, r"$Q$", fontsize=8, color="#777")
    ax.set_title(r"Fig. 5  Quadrature (IQ) demodulator: two LO copies $90^\circ$ "
                 r"apart give the complex sample $z=I+jQ=A\,e^{j\phi}$",
                 fontsize=9.5)


def diag_phase_flow(ax):
    ax.axis("off"); ax.set_xlim(0, 10.6); ax.set_ylim(0, 3)
    boxes = [
        ("antennas\n$d\\sin\\theta$", "#eef4fb"),
        ("RF carrier\n$\\phi=2\\pi(d/\\lambda)\\sin\\theta$", "#dcebfb"),
        ("IQ streams\n$\\angle(z_1 z_0^{*})=\\phi$", "#dcebfb"),
        ("beamformer\npeak at\n$\\phi(\\theta)=\\phi$", "#e8f6ec"),
    ]
    x, w = 0.2, 2.4
    for i, (t, c) in enumerate(boxes):
        ax.add_patch(Rectangle((x, 0.8), w, 1.35, facecolor=c, edgecolor="#456",
                               lw=0.8))
        ax.text(x + w / 2, 1.48, t, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 0.12, 1.48), xytext=(x + w - 0.02, 1.48),
                        arrowprops=dict(arrowstyle="-|>", color="#456"))
        x += w + 0.13
    ax.text(5.3, 0.3, r"the same $\phi$ at every stage -- only its representation "
            r"changes", ha="center", fontsize=8.5, color=MUT)
    ax.set_title("Fig. 6  One phase difference, four representations", fontsize=9.5)


def diag_triangle(ax):
    from matplotlib.patches import Arc
    ax.axis("off"); ax.set_xlim(-0.4, 9.2); ax.set_ylim(-0.2, 5)
    th = np.deg2rad(35)
    p = np.array([np.sin(th), -np.cos(th)])   # propagation direction (down-right)
    w = np.array([np.cos(th), np.sin(th)])    # wavefront direction (up-right)
    A0 = np.array([2.3, 1.1]); A1 = np.array([6.8, 1.1])
    dd = float((A1 - A0) @ p)                 # = d*sin(theta)
    F = A1 - dd * p                           # foot on A0's wavefront
    # baseline + elements
    ax.plot([A0[0], A1[0]], [A0[1], A1[1]], color=INK, lw=1)
    ax.plot(*A0, "o", ms=8, color=ACC); ax.plot(*A1, "o", ms=8, color=ACC)
    ax.text(A0[0], A0[1] - 0.42, "el. 0", ha="center", fontsize=9)
    ax.text(A1[0], A1[1] - 0.42, "el. 1", ha="center", fontsize=9)
    ax.annotate("", xy=(A1[0], 0.35), xytext=(A0[0], 0.35),
                arrowprops=dict(arrowstyle="<->", color=MUT))
    ax.text((A0[0] + A1[0]) / 2, 0.02, r"$d$", ha="center", fontsize=11, color=MUT)
    # wavefront through el.0
    wf0 = A0 - 0.8 * w; wf1 = F + 0.7 * w
    ax.plot([wf0[0], wf1[0]], [wf0[1], wf1[1]], color="#999", lw=1.3)
    ax.text(wf1[0] + 0.08, wf1[1] + 0.02, "wavefront\nthrough el. 0", fontsize=7.5,
            color="#777", va="center")
    # extra path el.1 -> foot
    ax.plot([A1[0], F[0]], [A1[1], F[1]], color="#c0392b", lw=1.9)
    ax.text(F[0] - 0.12, (A1[1] + F[1]) / 2 + 0.05, r"$\Delta d=d\sin\theta$",
            color="#c0392b", fontsize=10.5, ha="right")
    # parallel incoming rays
    for tgt in (A0, A1):
        ax.annotate("", xy=tgt - 0.16 * p, xytext=tgt - 2.7 * p,
                    arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.2))
    s = A0 - 2.7 * p
    ax.text(s[0] - 0.30, s[1] + 0.42, "incoming\nwave", color=ACC, fontsize=8,
            va="center", ha="center")
    # right-angle mark at foot
    ra = 0.24
    c1 = F - ra * p; c2 = F + ra * w; c3 = F - ra * p + ra * w
    ax.plot([c1[0], c3[0]], [c1[1], c3[1]], color="#c0392b", lw=0.8)
    ax.plot([c2[0], c3[0]], [c2[1], c3[1]], color="#c0392b", lw=0.8)
    # theta between baseline and wavefront at el.0
    ax.add_patch(Arc(A0, 1.5, 1.5, angle=0, theta1=0, theta2=np.rad2deg(th),
                     color=INK, lw=1))
    ax.text(A0[0] + 0.95, A0[1] + 0.26, r"$\theta$", fontsize=13)
    ax.set_title(r"Fig. 1b  Extra path = projection of baseline $d$ onto the ray "
                 r"$= d\sin\theta$", fontsize=9.5)


# ---------------- build ----------------
def build(path):
    d = Doc(path)
    f = d.fig
    # ---- title page ----
    f.text(0.5, 0.80, "The SPF Signal Path", ha="center", fontsize=27,
           fontweight="bold", color=ACC)
    f.text(0.5, 0.745, "From Emission to Angle — a Mathematical Overview",
           ha="center", fontsize=14, color=INK)
    f.text(0.5, 0.70, "transmission  ·  propagation  ·  down-conversion  ·  "
           "phase difference  ·  beamforming", ha="center", fontsize=10, color=MUT)
    d.fig.add_axes([0.24, 0.40, 0.52, 0.24]).axis("off")
    diag_geometry(d.fig.axes[-1])
    f.text(0.5, 0.30, r"Core identity:   $\phi \;=\; 2\pi\,\frac{d}{\lambda}\,"
           r"\sin\theta \;=\; 2\pi f_c\,\frac{d\sin\theta}{c}$",
           ha="center", fontsize=14)
    f.text(0.5, 0.22, "Signal Processing Fun (SPF)  ·  generated from "
           "tutorials/build_signal_path_tutorial.py", ha="center", fontsize=9,
           color=MUT)
    d._new_page()

    # ---- 1 setup ----
    d.h1("1  Setup and notation")
    d.p("SPF locates a radio emitter by measuring, at a moving receiver, the "
        "phase difference between two closely-spaced antennas. One receiver = two "
        "antenna elements a distance $d$ apart, sampled coherently. The whole "
        "chain reduces to one geometric fact (Fig. 1) carried faithfully through "
        "the radio hardware and the software stack.")
    for sym, mean in [
        (r"$f_c$", "carrier (transmit) frequency; $\\lambda=c/f_c$ is the wavelength"),
        (r"$d$", "physical spacing between the two antenna elements (metres)"),
        (r"$d/\lambda$", "spacing in wavelengths; production spans $\\approx0.12-1.55$ (above $0.5$, $\\phi$ aliases -- see Section 8)"),
        (r"$\theta$", "angle of arrival, measured from the array normal (broadside)"),
        (r"$\phi$", "inter-element phase difference — the raw observable"),
        (r"$f_{LO}$", "receiver local-oscillator frequency (the 'base clock' of mixing)"),
        (r"$\Delta f$", "baseband offset $=f_c-f_{LO}$; SPF sets it to the 'intermediate' freq"),
        (r"$f_s$", "ADC sampling rate (SPF: 30 MHz on a 3 MHz signal)"),
    ]:
        d.p(sym + r"$\;\;$ " + mean, gap=0.004, indent=0.01)
    d.note("Key idea, stated once: the inter-antenna phase $\\phi$ is a "
           "carrier-frequency effect. It is set at RF, survives down-conversion "
           "unchanged, and is essentially independent of the sampling clock $f_s$. "
           "The sampling clock only has to satisfy Nyquist for the signal bandwidth.")

    # ---- 2 transmission ----
    d.h1("2  Transmission")
    d.p("The emitter radiates a narrowband tone at carrier $f_c$. Written as a "
        "real passband signal and, equivalently, in complex analytic form:")
    d.eq(r"s(t)=A\cos(2\pi f_c t+\psi)\quad\Leftrightarrow\quad "
         r"\tilde s(t)=A\,e^{\,j(2\pi f_c t+\psi)}")
    d.p("$A$ is amplitude, $\\psi$ an arbitrary start phase. The absolute phase "
        "$\\psi$ is unknowable and irrelevant — only differences will matter. In "
        "SPF the emitter is deliberately offset from the receiver LO by a known "
        "'intermediate' frequency, so the tone lands at $\\Delta f$ in baseband "
        "(never at DC, which would collide with LO/DC leakage in the AD9361).")

    # ---- 3 propagation / geometry ----
    d.h1("3  Propagation and array geometry")
    d.p("A plane wave arriving from angle $\\theta$ reaches element 1 over an "
        "extra path length $\\Delta d=d\\sin\\theta$ (Fig. 1). Extra path becomes "
        "extra phase at the carrier:")
    d.eq(r"\phi=\frac{2\pi\,\Delta d}{\lambda}=\frac{2\pi d\sin\theta}{\lambda}"
         r"=2\pi\,\frac{d}{\lambda}\sin\theta")
    d.p("This is $\\mathtt{theta\\_to\\_phi}$ in the code. Equivalently "
        "$\\phi=2\\pi f_c\\,\\tau$ with $\\tau=d\\sin\\theta/c$ the inter-element "
        "time delay. Note it is the CARRIER $f_c$ that scales the phase.")
    d.note("Why phase, not time-of-arrival: with $d=0.035$ m the delay is "
           "$\\tau=d/c\\approx117$ ps. At $f_s=30$ MHz one sample is 33 ns, so "
           "$\\tau\\approx0.0035$ samples — unresolvable in time at ANY realistic "
           "ADC rate. But at $d/\\lambda=0.35$ the phase is $2\\pi(0.35)\\approx126^\\circ$: "
           "large and easy to read. Phased-array DF lives in the phase domain.")
    d.p("Because $\\sin\\theta=\\sin(\\pi-\\theta)$, a single pair cannot separate "
        "$\\theta$ from $\\pi-\\theta$: the front/back ambiguity (Fig. 3), resolved "
        "later by fusing two differently-oriented receivers.")

    # ---- 3b derivation: where d sin(theta) comes from ----
    d._new_page()
    d.h1("Where the extra path d sin(theta) comes from")
    d.p(r"Section 3 used $\phi=2\pi(d/\lambda)\sin\theta$ without deriving it. It "
        r"rests on two facts: a wave's phase advances with distance travelled, and "
        r"geometry makes one antenna's path longer by $d\sin\theta$.")
    d.h2(r"1.  Phase advances 2*pi per wavelength")
    d.p(r"A travelling wave repeats every wavelength $\lambda$ -- that is what "
        r"wavelength MEANS. So moving an extra distance $\Delta d$ along the "
        r"propagation direction advances the phase by that fraction of a cycle:")
    d.eq(r"\Delta\phi=2\pi\,\frac{\Delta d}{\lambda}")
    d.p(r"Two equivalent views. Spatial: a frozen snapshot of the wave is a sinusoid "
        r"in space of period $\lambda$, so stepping $\Delta d$ changes the cosine's "
        r"argument by $(2\pi/\lambda)\Delta d$. Temporal: the far antenna receives the "
        r"wave later by $\tau=\Delta d/c$, and a carrier at $f_c$ delayed by $\tau$ "
        r"gains phase $2\pi f_c\tau=2\pi\,\Delta d/\lambda$. Same answer -- extra "
        r"distance, extra delay, extra phase.")
    d.h2(r"2.  Geometry: the extra path is d sin(theta)")
    d.p(r"The source is far, so wavefronts arrive flat and the two rays are parallel. "
        r"Take the wavefront (a constant-phase line, perpendicular to the ray) that "
        r"just touches element 0; the wave must travel an extra $\Delta d$ to reach "
        r"element 1 (Fig. 1b). In the right triangle with hypotenuse = baseline $d$ "
        r"and $\Delta d$ opposite the angle $\theta$:")
    d.diagram(diag_triangle, h=0.235, w=0.72)
    d.eq(r"\sin\theta=\frac{\Delta d}{d}\quad\Rightarrow\quad \Delta d=d\sin\theta")
    d.p(r"(The triangle's angle is $\theta$ because the ray is $\theta$ off the "
        r"normal, so the wavefront -- perpendicular to the ray -- is $\theta$ off the "
        r"baseline.)")
    d.h2(r"3.  Combine")
    d.eq(r"\phi=\frac{2\pi}{\lambda}\,\Delta d=2\pi\,\frac{d}{\lambda}\sin\theta")
    d.bullet(r"Broadside $\theta=0$: $\sin\theta=0$, $\Delta d=0$, $\phi=0$ -- the "
             r"wavefront reaches both antennas at once (equidistant from the source).")
    d.bullet(r"Endfire $\theta=90^\circ$: $\sin\theta=1$, $\Delta d=d$, $\phi$ is "
             r"maximal -- the wave travels straight along the array axis.")
    d.bullet(r"Hence $\sin\theta$ (zero at broadside, max at endfire); it would be "
             r"$\cos\theta$ if $\theta$ were measured from the array axis, not the normal.")
    d.p(r"Example (SPF): $f_c=2.4$ GHz $\Rightarrow \lambda=12.5$ cm, $d=3.5$ cm "
        r"($d/\lambda=0.28$). At $\theta=30^\circ$, $\Delta d=1.75$ cm $=0.14\lambda "
        r"\Rightarrow \phi\approx50^\circ$; at $\theta=90^\circ$, "
        r"$\phi\approx101^\circ$. A couple of cm of path becomes tens of degrees "
        r"because it is a sizable fraction of the small wavelength.")
    d.note(r"The $\lambda$ here is the CARRIER wavelength ($\lambda=c/f_c$). $\Delta "
           r"d$ is a fixed number of metres set by geometry; converting metres to "
           r"phase divides by the carrier wavelength -- which is why the phase lives "
           r"at the carrier, and a higher $f_c$ gives more phase per degree of angle.")

    # ---- 4 receiving / downconversion ----
    d.h1("4  Receiving and down-conversion (high freq $\\rightarrow$ low freq)")
    d.p("Each element $k\\in\\{0,1\\}$ sees the same tone plus its own spatial "
        "phase $\\phi_k$ (with $\\phi_0=0$, $\\phi_1=\\phi$):")
    d.eq(r"\tilde s_k(t)=A\,e^{\,j(2\pi f_c t+\psi+\phi_k)}")
    d.p("The receiver multiplies by the local oscillator $e^{-j2\\pi f_{LO}t}$ "
        "(the same LO for both channels) and low-pass filters. Mixing shifts the "
        "spectrum down by $f_{LO}$ (Fig. 2):")
    d.eq(r"b_k(t)=\mathrm{LPF}\{\tilde s_k(t)\,e^{-j2\pi f_{LO}t}\}"
         r"=A\,e^{\,j(2\pi\Delta f\,t+\psi+\phi_k)},\quad \Delta f=f_c-f_{LO}")
    d.diagram(diag_downconv, h=0.205)
    d.p("This is the crux the question asks about: the down-conversion moves the "
        "signal from $f_c$ to $\\Delta f$, but the spatial phase $\\phi_k$ appears "
        "as an ADDITIVE CONSTANT on the baseband phasor — identical before and "
        "after the mixer. The mixer changes the frequency, not the spatial phase. "
        "So $\\phi$ is 'placed' as a fixed per-channel offset that is simply carried "
        "along into baseband.")

    # ---- 5 phase difference ----
    d.h1("5  The phase difference through the stack")
    d.p("Form the phase difference of the two baseband channels (this is "
        "$\\mathtt{get\\_phase\\_diff}$ in $\\mathtt{rf.py}$):")
    d.eq(r"\Delta\phi(t)=\angle b_1(t)-\angle b_0(t)"
         r"=[2\pi\Delta f t+\psi+\phi_1]-[2\pi\Delta f t+\psi+\phi_0]"
         r"=\phi")
    d.p("Everything common to both channels cancels at every sample: the LO/"
        "baseband rotation $2\\pi\\Delta f t$, the source phase $\\psi$, and the "
        "unknown absolute offset. Only the geometry-bearing $\\phi$ survives. "
        "This is why (a) the intermediate offset $\\Delta f$ is harmless — it "
        "cancels sample-by-sample; and (b) absolute phase is discarded — the "
        "pipeline only ever needs the DIFFERENCE.")
    d.p("SPF then averages $\\Delta\\phi$ over a window with a trimmed CIRCULAR "
        "mean (angles wrap at $\\pm\\pi$), giving a robust per-window phase and a "
        "spread (stddev) used as a confidence weight.")
    d.note(r"Sign convention in the implementation: the code computes the OPPOSITE "
           r"difference -- $\mathtt{get\_phase\_diff}$ (rf.py:745) returns "
           r"$\angle b_0-\angle b_1=-\phi$ -- and the dataset's ground-truth target "
           r"carries a matching $-\sin\theta$ (spf_dataset.py:1513), so the two minus "
           r"signs cancel end-to-end. Which physical antenna is element 0 is set by "
           r"cabling, not code; the pair (measurement sign, target sign) is what must "
           r"stay consistent, and every filter likewise negates the spacing.")
    d.note("The one term that does NOT cancel is a per-channel LO phase mismatch "
           "between the two RX chains. It survives as a fixed offset — the "
           "$\\mathtt{phi\\_drift}$ / AD9361 RX1-RX2 phase inversion — which SPF "
           "removes with a calibration step (and models with opposite sign on the "
           "two receivers in the synthetic generator).")

    # ---- 6 beamforming ----
    d.h1("6  From phase to angle: beamforming")
    d.p("Rather than invert $\\phi\\to\\theta$ directly (which is 2-to-1), SPF "
        "SCANS candidate angles. For each $\\theta_i$ build a steering vector that "
        "predicts the phase a wave from $\\theta_i$ would imprint, apply it to the "
        "signal, and measure output power:")
    d.eq(r"a(\theta_i)=e^{-j\,2\pi(d/\lambda)\sin\theta_i},\qquad "
         r"P(\theta_i)=|\,a(\theta_i)\cdot x\,|")
    d.p(r"(No conjugate is applied at multiply time -- the compensating minus sign "
        r"is already baked into the $e^{-j}$ steering definition; the code does a "
        r"plain product, rf.py:1266. And in the implementation the steering phase is "
        r"split $\pm\phi/2$ across the two elements at $\pm d/2$ rather than all on "
        r"element 1 -- an equivalent global-phase choice, rf.py:1020-1023.)")
    d.p("$P(\\theta)$ peaks where the steering phase matches the measured phase "
        "(Fig. 3). With a 2-element array the response is bimodal — a peak at the "
        "true $\\theta$ and at its mirror $\\pi-\\theta$. SPF evaluates this on a "
        "grid of $n_\\theta=65$ angles (linspace includes both $\\pm\\pi$ endpoints, "
        "which are the same physical angle).")
    d.diagram(diag_beamformer, h=0.235)

    # ---- 7 the stack ----
    d.h1("7  How it flows through the SPF stack")
    d.p("The raw IQ is processed in windows: detrend, then a per-window "
        "delay-and-sum beamformer produces a $(n_{windows}\\times 65)$ power map "
        "(the $\\mathtt{windowed\\_beamformer}$), alongside per-window circular "
        "phase mean/stddev ($\\mathtt{all\\_windows\\_stats}$). These features — "
        "NOT the raw IQ — feed the neural network, which outputs a distribution "
        "over the 65 angle bins.")
    d.diagram(diag_stack, h=0.16, w=0.86)
    d.p("Symmetry is handled deliberately: the network is trained invariant to "
        "left/right ($\\theta\\to-\\theta$, data augmentation) and its target is "
        "folded front/back ($\\theta\\to\\mathrm{sign}(\\theta)\\pi-\\theta$), because "
        "a single pair genuinely cannot resolve either. A second, differently-"
        "oriented receiver (the paired model) breaks the ambiguity.")

    # ---- 8 artifacts + numbers ----
    d.h1("8  Hardware artifacts and real SPF numbers")
    d.p("The idealized $b_k(t)$ picks up real-world perturbations. Those SPF "
        "models explicitly (and why):")
    d.bullet("thermal / phase noise $\\Rightarrow$ Gaussian on $\\phi$ (broadens the peak)")
    d.bullet("unknown absolute phase $\\Rightarrow$ random $\\psi$ on both elements (cancels)")
    d.bullet("inter-radio LO offset $\\Rightarrow$ fixed $\\mathtt{phi\\_drift}$ (calibrated)")
    d.bullet("intermittent emitter $\\Rightarrow$ noise windows (what segmentation detects)")
    d.p("Not modeled (levers for realism): multipath, Doppler, IQ imbalance, DC "
        "offset, ADC quantization, AGC dynamics, real packet modulation.")
    d.h2("Representative receiver parameters")
    for a, b in [
        ("carrier $f_c$", r"$2.4$ or $5.766\ \mathrm{GHz}$"),
        ("wavelength $\\lambda=c/f_c$", r"$\approx 12.5$ / $5.2\ \mathrm{cm}$"),
        ("spacing $d$ / ratio $d/\\lambda$", r"$0.025-0.08\ \mathrm{m}$ / $0.12-1.55$ ($>0.5$ aliased for many sets)"),
        ("sample rate $f_s$", r"$30\ \mathrm{MHz}$ (signal BW $\approx 3\ \mathrm{MHz}$)"),
        ("inter-element delay $\\tau_{max}$", r"$\approx 117\ \mathrm{ps}\;(\ll$ one sample$)$"),
        ("angle grid $n_\\theta$", r"$65$ bins on $[-\pi,\pi]$"),
    ]:
        d.p(a + r"$\;\;\rightarrow\;\;$" + b, gap=0.004, indent=0.01)

    # ---- 9 recap ----
    d.h1("9  One-page recap")
    d.p("Geometry sets the observable at the carrier:")
    d.eq(r"\phi=2\pi\,(d/\lambda)\sin\theta")
    d.p("Down-conversion by a shared LO moves $f_c\\to\\Delta f$ but carries "
        "$\\phi$ through as an additive constant:")
    d.eq(r"b_k(t)=A\,e^{\,j(2\pi\Delta f t+\psi+\phi_k)}")
    d.p("The inter-channel difference cancels all common terms and recovers pure "
        "geometry:")
    d.eq(r"\angle b_1-\angle b_0=\phi\;\;\Rightarrow\;\;\theta\ \mathrm{(up\ to\ "
         r"front/back)}")
    d.p("Sampling clock: needs only $f_s\\geq$ signal bandwidth (Nyquist) plus "
        "channel COHERENCE (shared LO + clock). It does not set, and is not set "
        "by, the phase $\\phi$.")

    # ---- Appendix A : IQ modulation ----
    d._new_page()
    d.h1("Appendix A  IQ modulation: why the samples are complex")
    d.p("Every stage above treats each baseband sample as one complex number. "
        "That number is produced by IQ (in-phase / quadrature) demodulation, and "
        "it is the reason SPF can read phase at all.")
    d.h2("The idea")
    d.p(r"Any narrowband passband signal is a slowly-varying complex envelope "
        r"$z(t)=I(t)+jQ(t)$ riding on the carrier:")
    d.eq(r"s(t)=I(t)\cos(2\pi f_c t)-Q(t)\sin(2\pi f_c t)"
         r"=\mathrm{Re}\{\,z(t)\,e^{j2\pi f_c t}\,\}")
    d.p(r"The two real branches carry the whole story -- $I=A\cos\phi$ (in-phase), "
        r"$Q=A\sin\phi$ (quadrature) -- so amplitude and phase fall straight out:")
    d.eq(r"|z|=\sqrt{I^{2}+Q^{2}}=A,\qquad \angle z=\arctan(Q/I)=\phi")
    d.p(r"A receiver recovers $I$ and $Q$ by mixing the antenna voltage with two "
        r"copies of the local oscillator $90^\circ$ apart ($\cos$ and $-\sin$), then "
        r"low-pass filtering each branch (Fig. 5). The AD9361 in the Pluto does exactly "
        r"this and hands software a stream of complex samples -- SPF's "
        r"$\mathtt{signal\_matrix}$ is $\mathtt{complex64}$.")
    d.diagram(diag_iq, h=0.205, w=0.86)
    d.h2("Why use it -- hardware")
    d.bullet(r"Half the ADC rate: a complex sample at $f_s$ spans a full bandwidth "
             r"$f_s$ ($-f_s/2$ to $+f_s/2$); a real ADC needs $2f_s$ for the same span.")
    d.bullet("Zero-/low-IF front end: mixing straight to baseband removes the "
             "high-frequency IF stage and its filters -- what lets an SDR fit on one chip.")
    d.bullet(r"Image rejection for free: a real mixer folds the signal and its mirror "
             r"image together; the $90^\circ$ I/Q relationship separates $+$ and $-$ "
             r"frequencies and cancels the image, no dedicated filter needed.")
    d.bullet("Amplitude AND phase in one shot: both ADC channels together deliver the "
             "full complex envelope every sample -- no phase reconstruction.")
    d.h2("Why use it -- signal processing")
    d.bullet(r"Instantaneous phase is free: $\phi=\arctan(Q/I)$. SPF's core operation "
             r"-- the inter-antenna difference $\angle z_1-\angle z_0$ "
             r"($\mathtt{get\_phase\_diff}$) -- exists only because samples are complex.")
    d.bullet(r"Positive and negative frequencies stay distinct (one-sided spectrum): a "
             r"tone above the LO is told from one below -- what makes the intermediate "
             r"offset $\Delta f$ unambiguous.")
    d.bullet(r"Frequency shifts and beamforming are just complex multiplies: "
             r"down-conversion is $\times\,e^{-j\omega t}$; a steering vector is "
             r"$e^{-j\phi(\theta)}$ applied to the complex signal (exactly "
             r"$\mathtt{rf.py}$ steering vectors $\cdot$ signal matrix).")
    d.bullet(r"Common-mode rotations cancel cleanly: the LO phase and the $\Delta f\,t$ "
             r"rotation are shared multiplicative phases that drop out of the difference "
             r"(the cancellation in Section 5) -- natural arithmetic in the complex plane.")
    d.note(r"Without IQ you would sample a real passband signal: 2x the ADC rate, an "
           r"image-reject filter, and a Hilbert transform to recover instantaneous phase "
           r"-- and you still could not tell $+\Delta f$ from $-\Delta f$. IQ is what "
           r"makes phase-difference direction-finding practical.")

    # ---- Appendix B : tracing the phase difference through the stack ----
    d._new_page()
    d.h1("Appendix B  From spacing d to the beamformer peak")
    d.p(r"This appendix follows one quantity -- the inter-antenna phase difference "
        r"$\phi$ -- from the physical spacing $d$ to where it becomes a peak in the "
        r"beamformer. It is the SAME $\phi$ at every stage; only its representation "
        r"changes (Fig. 6).")
    d.diagram(diag_phase_flow, h=0.135, w=0.94)
    d.h2("1.  At the antennas (RF)")
    d.p(r"A wave from angle $\theta$ reaches element 1 over an extra path "
        r"$d\sin\theta$ -- an extra carrier phase")
    d.eq(r"\phi=2\pi\,(d/\lambda)\sin\theta")
    d.p(r"so the two antenna voltages differ by exactly $\phi$:")
    d.eq(r"s_0=A\,e^{j(2\pi f_c t+\psi)},\qquad s_1=A\,e^{j(2\pi f_c t+\psi+\phi)}")
    d.p(r"The spacing $d$ enters once, here, at the carrier.")
    d.h2("2.  Into the IQ samples (down-conversion)")
    d.p(r"Both channels are mixed by the SAME LO $e^{-j2\pi f_{LO}t}$ and low-pass "
        r"filtered. The frequency drops to $\Delta f=f_c-f_{LO}$; the spatial phase "
        r"$\phi$ is untouched:")
    d.eq(r"z_0=A\,e^{j(2\pi\Delta f\,t+\psi)},\qquad "
         r"z_1=A\,e^{j(2\pi\Delta f\,t+\psi+\phi)}")
    d.p(r"In the complex IQ stream $\phi$ is the constant angle by which $z_1$ leads "
        r"$z_0$ -- equivalently the phase of their cross-product, where the common "
        r"LO/rotation terms cancel:")
    d.eq(r"z_1\,z_0^{*}=A^{2}e^{j\phi}\quad\Rightarrow\quad \phi=\angle(z_1 z_0^{*})")
    d.p(r"So $d$ has trickled from a carrier phase into a fixed rotation between two "
        r"complex sample streams.")
    d.h2("3.  Reading phi optimally")
    d.p(r"To recover $\phi$ from noisy samples, combine COHERENTLY over the window: "
        r"the maximum-likelihood estimate is the angle of the summed cross-product,")
    d.eq(r"\hat\phi=\angle\left(\sum_t z_1(t)\,z_0^{*}(t)\right)")
    d.p(r"Summing the complex products BEFORE taking the angle amplitude-weights each "
        r"sample by its SNR -- the optimal translation of antenna phase into a number. "
        r"SPF's per-sample $\mathtt{get\_phase\_diff}$ + trimmed circular mean is a "
        r"robust near-equivalent.")
    d.h2("4.  Into the beamformer (where phi shows up)")
    d.p(r"The beamformer scans candidate angles. For $\theta_i$ it applies the modeled "
        r"steering phase $\phi(\theta_i)=2\pi(d/\lambda)\sin\theta_i$ to element 1 and "
        r"coherently sums the two elements:")
    d.eq(r"y(\theta_i)=z_0+e^{-j\phi(\theta_i)}z_1,\qquad "
         r"P(\theta_i)=|y(\theta_i)|")
    d.p(r"Substituting $z_1=z_0\,e^{j\phi}$ from stage 2:")
    d.eq(r"P(\theta_i)=|z_0|\,\left|1+e^{j(\phi-\phi(\theta_i))}\right|"
         r"=2\,|z_0|\,\left|\cos\!\left(\frac{\phi-\phi(\theta_i)}{2}\right)\right|")
    d.p(r"(The code returns this magnitude, not its square -- "
        r"$\mathtt{beamformer\_given\_steering\_nomean}$, rf.py:1302-1306; the argmax "
        r"over $\theta_i$ is identical either way.) The response is maximal exactly "
        r"when the modeled phase matches the measured one, "
        r"$\phi(\theta_i)=\phi$ (i.e. $\sin\theta_i=\sin\theta$). THAT is where $\phi$ "
        r"appears: as the LOCATION of the beamformer peak. Crucially the same $d$ sits "
        r"in both the physical $\phi$ and the steering $\phi(\theta_i)$, so they must "
        r"use the identical spacing ($\mathtt{rx\_wavelength\_spacing}$) or the peak "
        r"lands at a biased angle.")
    d.note(r"One quantity, four faces: an extra path (metres) -> a carrier phase (RF) "
           r"-> a fixed rotation between two IQ streams (baseband) -> the argmax of a "
           r"power spectrum (beamformer). 'Optimal' at each step means coherent "
           r"summation -- exactly what both the ML phase estimate and the beamformer do.")

    d.end()
    return d.pageno


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "spf_signal_path.pdf")
    n = build(out)
    print(f"wrote {out}  ({n} pages)")
