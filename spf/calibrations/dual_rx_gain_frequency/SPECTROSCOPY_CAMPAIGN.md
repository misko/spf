# Controlled A–G spectroscopy campaign

This campaign tests whether the frequency ripple is external to the Pluto,
assigns candidate delay components with a known jumper, measures connector and
thermal repeatability, tests fixed TX-level dependence, and fills low-gain
coverage. It reuses the production direct-USB V7 calibration runner; it does
not introduce another IQ capture path.

The source manifest is
[`configs/spectroscopy_campaign.yaml`](configs/spectroscopy_campaign.yaml).
The common acquisition contract is
[`configs/spectroscopy_campaign_base.yaml`](configs/spectroscopy_campaign_base.yaml).
Resolved stage configs are generated inside the chosen output root and are
hashed before any capture starts.

## Design and exact counts

Every science stage uses the additive-cross schedule. For `N` gains this
records `2N-1` ordered pairs around a reference gain, rather than an `N × N`
Cartesian surface.

| Stage | Frequencies | Gains | Pairs/LO | Epochs | Frames/radio |
|---|---:|---:|---:|---:|---:|
| Rate pilot | 10 | 3 | 5 | 1 | 50 |
| A, B, C, D, or G | 113 | 3 | 5 | 3 | 1,695 each |
| Each fixed-TX E treatment | 2 | 14 | 27 | 3 | 162 |
| Seven E thermal anchors | 1 | 1 | 1 | 1 | 7 total |
| F common low gain plus TIA boundary | 6 | 12 | 23 | 3 | 414 |
| **Total** | | | | | **9,918** |

The 113-point spectroscopy comb is 400–5900 MHz in 50 MHz steps plus the
genuinely additional 1301 and 4001 MHz probes. The 1300 and 4000 MHz probes
are already members of the comb and are not duplicated.

The seven small E anchor roots replace the proposed “anchor every ten cells.”
Keeping anchors in separate datasets avoids treating TX=0 controls as
observations from the current fixed-TX treatment. It still provides a
before/after time series for every treatment. If ten-cell interleaving is
later essential, add a first-class anchor flag to the V7 schedule and all
model readers; do not silently duplicate an ordinary treatment cell.

## Pass/fail gates

| Gate | Pass | Fail/stop |
|---|---|---|
| Render | Every generated config validates; counts above match; hashes recorded | Duplicate coordinates, unknown keys, invalid gain/reference, or count mismatch |
| Firmware preparation | Exactly two serials, boot-verified pinned RAM image, protocol v2 metadata | Wrong count, firmware SHA, missing direct USB, or invalid fingerprint |
| Gain-table audit | Both radios report 77-row `FULL` tables with exact ranges and pinned 231-byte SHA256 values | Any type, range, row, firmware, or table-byte mismatch |
| Rate pilot | Complete V7 validation and end-to-end time ≤1.3 seconds per recorded frame | Stop before A; do not improvise a reduced design inside the same run root |
| A | All frames structurally complete; ordinary quality validation passes | Repair baseline setup before touching cables |
| B | Treated radio `.17` RX1 has the documented nominal 11 dB pad stack; its RX2 and control radio `.18` are untouched; validation passes or an explicit quality waiver records the failure | Control arm or harness changed, missing operator note, or invalid data |
| C | Pad stack removed; nominal 30 cm jumper installed on treated `.17` RX1 only; control radio `.18` remains unchanged; validation passes or an explicit quality waiver records the failure | Unknown jumper/torque, changed RX2 arm, or changed control radio |
| D | Original harness restored; validation passes | A–D repeatability cannot be bounded, so B/C conclusions are provisional |
| E | Six separate fixed-level roots plus seven TX=0 anchors complete | Missing treatment/anchor or non-muted `-80 dB` control |
| F | All requested gains available in each active table and overlaps validate | Any unavailable gain or overlap inconsistency |
| G | Starts at least eight hours after A and original harness remains installed | Early start or unrecorded harness change |

The muted E root deliberately disables the “clean tone required” preflight.
Metadata and structural V7 checks still apply, but tone-quality failure is
expected and recorded rather than promoted to a stage failure.

## Runbook

Choose a unique output root. Never reuse it for a different physical session.

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  render \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION
```

RAM-load and fingerprint both attached radios:

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  prepare
```

Read and verify the active low, middle, and high full gain tables. This changes
the LO to three passive probe frequencies but never enables TX. The audit reads
the driver's 4 KiB sysfs binary attribute locally over the serial-isolated
Pluto USB-network path; Python libiio's ordinary 1 KiB attribute accessor is
too small for the 77-row table and is deliberately not used:

```bash
sudo -E /home/pi/spf-virtualenv/bin/python \
  -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  audit \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION/gain_table_audit.json
```

The audit needs root/CAP_NET_ADMIN only to create a short-lived, serial-isolated
USB-network namespace for each Pluto. It writes the non-secret audit JSON with
read permissions for the normal unprivileged campaign runner.

Record the baseline-harness checkpoint and run the rate pilot:

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  approve \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION \
  --stage rate_pilot \
  --operator OPERATOR \
  --note "Original 30 dB attenuator and splitter harness confirmed"

python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  run-stage \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION \
  --stage rate_pilot
```

The pilot writes its measured end-to-end seconds/frame into
`stages/rate_pilot/stage_result.json`. A failed rate gate prevents every later
stage from starting.

For stages with a cable checkpoint, approve only after the physical work:

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  approve \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION \
  --stage A \
  --operator OPERATOR \
  --note "Original harness; torque recorded in session log"

python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  run-stage \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION \
  --stage A
```

Repeat `approve` where required and `run-stage` in manifest order. Stages
without physical changes, including the E treatment/anchor sequence and F,
need only `run-stage`. The campaign refuses missing dependencies, changed
resolved configs, missing approvals, a failed rate gate, firmware mismatch,
or an early G start.

If an experimental treatment is structurally complete but fails only its
quality or repeatability gate, preserve that failure and add an explicit
waiver before continuing:

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  waive-quality \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION \
  --stage B \
  --operator OPERATOR \
  --note "Complete treatment dataset retained; document failed cells and rationale"
```

The waiver is bound to the exact SHA256 of `stage_result.json`. It permits only
a complete capture whose radio validations are `pass` or `fail_quality`, with
at least one `fail_quality`. It cannot waive missing frames, partial captures,
schema failures, corrupt data, or a changed result file. The failed stage
remains `failed`; downstream status reports show `quality_waived: true`.

Inspect resumable state at any time:

```bash
python -m spf.calibrations.dual_rx_gain_frequency.spectroscopy_campaign \
  --manifest spf/calibrations/dual_rx_gain_frequency/configs/spectroscopy_campaign.yaml \
  status \
  --output artifacts/dual_rx_gain_frequency/spectroscopy_SESSION
```

Each stage stores:

```text
stages/STAGE/
├── SERIAL/calibration.v7.zarr
├── SERIAL/observations.jsonl
├── SERIAL/preflight.jsonl
├── SERIAL/validation.json
├── run_result.json
└── stage_result.json
```

The V7 stores retain radio serial, firmware/gadget SHAs, hardware fingerprint,
requested and observed gains, fixed/adaptive TX gain, IQ, and quality flags.
The campaign plan, resolved config hashes, gain-table audit, operator approvals,
and wall-clock timing remain beside the datasets.
