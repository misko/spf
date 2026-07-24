"""Render the rover buzzer tones to WAV (tones/<name>.wav) — stdlib only.

The tone strings live in spf/mavlink/mavlink_controller.py (`tones` dict) and are
sent to the flight controller as MAVLink PLAY_TUNE; the FC's ToneAlarm plays them
on the piezo. This script AST-extracts that dict (no heavy imports, cannot drift
from the code) and synthesizes each tune with a square wave to approximate the
piezo timbre. Format: QBasic PLAY / MML as implemented by ArduPilot's MMLPlayer —
T tempo (quarters/min), L default length (1/n of a whole), O octave, < > octave
shift, A-G notes (#/+ sharp, - flat), P/R rests, dots, MN/ML/MS articulation
(sound 7/8, 8/8, 3/4 of the note; MF/MB ignored). Run: python3 make_tones.py
"""

import ast
import math
import os
import struct
import wave

RATE = 22050
AMP = 0.45
RAMP_S = 0.003  # attack/release to avoid clicks

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLLER = os.path.join(HERE, "..", "..", "..", "spf", "mavlink", "mavlink_controller.py")
OUT_DIR = os.path.join(HERE, "tones")

SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def extract_tones(path):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(getattr(t, "id", None) == "tones" for t in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError(f"tones dict not found in {path}")


def parse_mml(s):
    """Yield (freq_hz_or_None, sound_s, gap_s) per note/rest."""
    s = s.replace(" ", "").upper()
    i, tempo, length, octave, art = 0, 120, 4, 4, 7 / 8
    out = []

    def read_int(default=None):
        nonlocal i
        n = ""
        while i < len(s) and s[i].isdigit():
            n += s[i]
            i += 1
        return int(n) if n else default

    def read_dots():
        nonlocal i
        d = 0
        while i < len(s) and s[i] == ".":
            d += 1
            i += 1
        return d

    def dur(n, dots):
        d = (240.0 / tempo) / n
        return d * (2 - 0.5 ** dots)

    while i < len(s):
        c = s[i]
        i += 1
        if c == "M":
            m = s[i]
            i += 1
            art = {"N": 7 / 8, "L": 1.0, "S": 3 / 4}.get(m, art)  # MF/MB: no-op
        elif c == "T":
            tempo = read_int(tempo)
        elif c == "L":
            length = read_int(length)
        elif c == "O":
            octave = read_int(octave)
        elif c == "<":
            octave = max(0, octave - 1)
        elif c == ">":
            octave = min(8, octave + 1)
        elif c in ("P", "R"):
            d = dur(read_int(length), read_dots())
            out.append((None, 0.0, d))
        elif c == "N":
            n = read_int(0)
            d = dur(length, read_dots())
            if n == 0:
                out.append((None, 0.0, d))
            else:
                f = 440.0 * 2 ** ((n - 46) / 12)  # N46 = A4
                out.append((f, d * art, d * (1 - art)))
        elif c in SEMITONE:
            semi = SEMITONE[c]
            if i < len(s) and s[i] in "#+":
                semi += 1
                i += 1
            elif i < len(s) and s[i] == "-":
                semi -= 1
                i += 1
            d = dur(read_int(length), read_dots())
            f = 440.0 * 2 ** ((octave - 4) + (semi - 9) / 12)
            out.append((f, d * art, d * (1 - art)))
        # anything else: ignore (ArduPilot is lenient)
    return out


def synth(notes):
    samples = []
    for freq, sound_s, gap_s in notes:
        n = int(sound_s * RATE)
        ramp = min(int(RAMP_S * RATE), n // 2)
        for j in range(n):
            v = AMP if math.sin(2 * math.pi * freq * j / RATE) >= 0 else -AMP
            if j < ramp:
                v *= j / ramp
            elif j > n - ramp:
                v *= (n - j) / ramp
            samples.append(v)
        samples.extend([0.0] * int(gap_s * RATE))
    return samples


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(v * 32767)) for v in samples))


def first_note_freq(samples):
    """Zero-crossing estimate over the steady part of the first sounded note."""
    a, b = int(0.01 * RATE), int(0.09 * RATE)
    seg = samples[a:b]
    crossings = sum(1 for j in range(1, len(seg)) if seg[j - 1] < 0 <= seg[j])
    return crossings / ((b - a) / RATE)


if __name__ == "__main__":
    tones = extract_tones(CONTROLLER)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'tone':<11} {'dur(s)':>7} {'first-note(Hz)':>15}  notes")
    for name, tune in sorted(tones.items()):
        mml = tune.decode() if isinstance(tune, bytes) else tune
        notes = parse_mml(mml)
        samples = synth(notes)
        write_wav(os.path.join(OUT_DIR, f"{name}.wav"), samples)
        seq = " ".join("rest" if f is None else f"{f:.0f}" for f, _, _ in notes)
        print(f"{name:<11} {len(samples)/RATE:>7.2f} {first_note_freq(samples):>15.1f}  {seq}")
    print(f"wrote {len(tones)} wavs to {OUT_DIR}/")
