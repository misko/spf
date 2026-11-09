"""Generate the E-GSC9 capture configs. Deterministic; no hardware needed."""
from pathlib import Path

OUT = Path("/tmp/claude-1000/-home-mouse9911-gits-spf/fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/gsc9/out")
FW = """pluto-firmware:
  release-tag: v0.38-plutoplus-spf-libiio-metadata-v5
  device-fw: v0.38-plutoplus-spf-libiio-metadata-v5
  asset-name: plutoplus-spf-libiio-metadata-v5-d7c87a9a2809-pluto.dfu
  image-url: https://github.com/misko/plutosdr-fw/releases/download/v0.38-plutoplus-spf-libiio-metadata-v5/plutoplus-spf-libiio-metadata-v5-d7c87a9a2809-pluto.dfu
  image-sha256: 948b46506febacb087f3955be86015e074f8c0e3370a9dfc6a942e735d97f882
  firmware-git-sha: d7c87a9a28094ee6f0b23cb47df9ff737b5a69d8
  gadget-git-sha: ab270f9e3128187372f27de887be65353f9e195d
  boot-mode: qspi
"""
GMIN, GMAX, REF = 23, 62, 62
GAINS = list(range(GMIN, GMAX + 1))
LOS = [5766000000, 5840000000]


def held_out_full():
    return [(a, b) for a in GAINS if a != REF for b in GAINS if b != REF]


def rover_priority_cells(k):
    """Highest-rover-mass cells with neither arm at the reference, from the census."""
    import pickle, collections
    H = pickle.load(open("/tmp/claude-1000/-home-mouse9911-gits-spf/fc21bd4f-704c-4541-ac00-783c1cec096d/scratchpad/gsc9/rover_hist.pkl", "rb"))
    tot = collections.Counter()
    for lo in (5766000000, 5840000000):
        c, n = H[f"all_{lo}"]
        for cell, v in c.items():
            tot[cell] += v / n           # equal weight per carrier
    diag = [(g, g) for g in GAINS if g != REF]
    ranked = [c for c, _ in tot.most_common()
              if c[0] != REF and c[1] != REF and GMIN <= c[0] <= GMAX and GMIN <= c[1] <= GMAX]
    out, seen = [], set()
    for c in diag + ranked:
        if c not in seen:
            seen.add(c); out.append(c)
        if len(out) >= k:
            break
    return out


def emit(name, header, pairs, reps, freqs, tx_gain, seed, label, notes):
    lines = [header, "data-version: 7", FW.rstrip(), "calibration:",
             f"  frequencies-hz: [{', '.join(str(f) for f in freqs)}]",
             f"  gains-db: [{', '.join(str(g) for g in GAINS)}]",
             "  schedule-design: additive_cross",
             f"  schedule-reference-gain-db: {REF}",
             "  held-out-gain-pairs:"]
    lines += [f"    - [{a}, {b}]" for a, b in pairs]
    lines += [
        f"  repetitions: {reps}",
        "  sample-rate-hz: 30000000", "  bandwidth-hz: 3000000", "  buffer-size: 65536",
        "  tone-offset-hz: 100000", "  tone-search-width-hz: 25000",
        "  transient-samples: 1024", "  phase-segments: 8",
        "  settle-seconds: 0.025", "  frequency-settle-seconds: 0.5",
        "  discard-frames-after-gain: 1", "  max-retries: 1",
        "  rf-dc-calibration-policy: before_each_frequency_block",
        f"  random-seed: {seed}", "  tx-source: fpga_dds",
        f"  tx-gain-db: {tx_gain}", "  tx-gain-policy: fixed",
        f"  tx-reference-rx-gain-db: 49", "  tx-min-gain-db: -80", "  tx-max-gain-db: 0",
        "  tx-digital-amplitude: 16384",
        f"  min-quality-valid-per-cell: {max(1, reps // 2 + 1)}",
        "  max-across-repeat-phase-std-deg: 5",
        f"  setup-label: {label}", f"  notes: >-\n    {notes}",
        "  quality:", "    min-tone-snr-db: -5", "    min-tone-dbfs: -65",
        "    max-tone-dbfs: -6", "    max-clipping-fraction: 0",
        "    min-coherence: 0.98", "    max-within-capture-phase-std-deg: 5",
    ]
    p = OUT / name
    p.write_text("\n".join(lines) + "\n")
    return p


A = emit("e_gsc9_rover_region_grid.yaml",
         "# E-GSC9 session A -- full 40x40 gain grid over the rover's operating region.\n"
         "# 1600 cells x 2 LOs x 5 epochs = 16000 frames/radio, 32000 for the pair.",
         held_out_full(), 5, LOS, -29, 20260814,
         "e-gsc9-r17-r18-v5-iio-usb-tx-fixed-29db-30db-pad-tee-rx1-rx2",
         "E-GSC9 session A, preregistered in experiments/e_gsc9_rover_operating_region. "
         "Full ordered cross of gains 23..62 at both rover carriers. Reference 62 makes the "
         "mandatory additive cross the rover's own g1=62 locus; the 1521 held-out cells are a "
         "genuine out-of-sample additivity test. TX is FIXED so no cell-dependent source level "
         "can confound the gain index.")

B = emit("e_gsc9_session_transfer.yaml",
         "# E-GSC9 session B -- session-to-session transfer control, >=12 h after session A.\n"
         "# 279 cells x 2 LOs x 3 epochs = 1674 frames/radio, 3348 for the pair.",
         rover_priority_cells(200), 3, LOS, -29, 20260815,
         "e-gsc9b-r17-r18-v5-iio-usb-tx-fixed-29db-30db-pad-tee-rx1-rx2",
         "E-GSC9 session B. Same bench, same fixed TX, after a power cycle and >=12 h. "
         "Measures how much of session A's LUT survives a session boundary and whether "
         "re-measuring only the 79-cell cross restores it.")

C = emit("e_gsc9_pad_discriminator.yaml",
         "# E-GSC9 session C (conditional) -- +10 dB pad at each RX port, TX raised 10 dB.\n"
         "# Identical cell set to session B. 279 cells x 2 LOs x 3 epochs.",
         rover_priority_cells(200), 3, LOS, -19, 20260816,
         "e-gsc9c-r17-r18-v5-iio-usb-tx-fixed-19db-30db-pad-tee-plus10db-per-arm",
         "E-GSC9 session C. Adds 10 dB of attenuation between the SMA tee and each RX port, "
         "which cuts round-trip cross-arm coupling by 20 dB while leaving every gain state "
         "identical. If the additivity residual is unchanged it is not harness coupling.")
print(A, B, C, sep="\n")


# ---- step-0 level ladder: 3 TX levels bracketing the design value ----
LADDER_G=[23,35,45,49,56,62]
def emit_ladder(tx,seed):
    lines=[f"# E-GSC9 step-0 level ladder at TX {tx} dB. 36 cells x 2 LOs x 1 epoch = 72 frames/radio.",
           "# Purpose: measure tone_dbfs(gain) on THIS bench before freezing session A's tx-gain-db.",
           "data-version: 7", FW.rstrip(), "calibration:",
           f"  frequencies-hz: [{', '.join(str(f) for f in LOS)}]",
           f"  gains-db: [{', '.join(str(g) for g in LADDER_G)}]",
           "  schedule-design: cartesian",
           "  repetitions: 1","  sample-rate-hz: 30000000","  bandwidth-hz: 3000000","  buffer-size: 65536",
           "  tone-offset-hz: 100000","  tone-search-width-hz: 25000","  transient-samples: 1024",
           "  phase-segments: 8","  settle-seconds: 0.025","  frequency-settle-seconds: 0.5",
           "  discard-frames-after-gain: 1","  max-retries: 1",
           "  rf-dc-calibration-policy: before_each_frequency_block",
           f"  random-seed: {seed}","  tx-source: fpga_dds",
           f"  tx-gain-db: {tx}","  tx-gain-policy: fixed","  tx-reference-rx-gain-db: 49",
           "  tx-min-gain-db: -80","  tx-max-gain-db: 0","  tx-digital-amplitude: 16384",
           "  min-quality-valid-per-cell: 1","  max-across-repeat-phase-std-deg: 5",
           f"  setup-label: e-gsc9-level-ladder-tx{abs(tx)}db-30db-pad-tee-rx1-rx2",
           "  notes: >-\n    E-GSC9 step 0. Diagnostic only; never fitted. Measures the bench level law\n    tone_dbfs = 0.98*tx + g + K per (radio, LO, arm) so session A's fixed TX can be\n    chosen from measurement instead of from the 2026-08-13 archive.",
           "  quality:","    min-tone-snr-db: -20","    min-tone-dbfs: -85","    max-tone-dbfs: -3",
           "    max-clipping-fraction: 0","    min-coherence: 0.9","    max-within-capture-phase-std-deg: 10"]
    q=OUT/f"e_gsc9_level_ladder_tx{abs(tx)}.yaml"; q.write_text("\n".join(lines)+"\n"); return q
for tx,seed in ((-35,20260810),(-29,20260811),(-23,20260812)):
    print(emit_ladder(tx,seed))
