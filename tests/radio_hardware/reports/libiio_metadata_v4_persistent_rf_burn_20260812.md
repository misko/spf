# libiio metadata v4 persistent flash and RF burn — 2026-08-12

## Disposition

**PASS for persistent QSPI deployment, reboot persistence, IP/USB metadata RX,
LO/bandwidth retuning, and cabled RF signal quality on both radios.**

The accepted campaign contains 144 fresh main-burn sessions and 576 metadata
frames.  IP and USB each carried 72 sessions.  A final post-reboot smoke added
four sessions and eight frames.  No accepted session needed a transport/context
retry, and every frame passed IQ shape, capture-index, gain/RSSI metadata,
gain-observation, overflow, RF-state readback, and spectral-tone checks.

This is not a claim that TX setup was one-shot clean.  The known Pluto+ silent
first DDS-arm state appeared during setup and is recorded below.  RX metadata
frames were not accepted until the cabled signal passed the preflight gate.

## Persistent candidate

- DFU: `plutoplus-spf-libiio-metadata-v4-0ad6c9410afd-pluto.dfu`
- SHA-256: `5acd2073c2a14b39e047c6328040562d49c8334237994bb926ea7e518139cea1`
- Firmware: `v0.38-plutoplus-spf-gain-series-v4-rc16-12-g0ad6c`
- Buildroot: `libiio-metadata-v4-source/buildroot-final-v5`
- Radio iiOD/libiio: `0.25`
- QSPI `fit_size`: `c274fb`

Each radio was checked by serial immediately before entering serial-flash DFU
mode.  Only the `firmware.dfu` alternate on the exact physical path was
written, followed by DFU detach:

| Radio | Serial | Physical USB path |
|---|---|---|
| A | `104000bac4950008230026001b440a003a` | `1-1.1` |
| B | `1040007c4a94000211000b009186843ef2` | `1-1.2` |

Both radios independently rebooted from QSPI after flashing, passed the burn,
then rebooted from QSPI again.  The final reboot retained the firmware,
Buildroot, `fit_size`, metadata capability, and radio-side libiio 0.25 identity.
DHCP moved the final LAN addresses to `192.168.1.157` (A) and
`192.168.1.185` (B); these addresses are observations, not persistent IDs.

## RF fixture and checks

The existing fixture connects each radio's TX2 through a declared 30 dB
attenuated splitter to that radio's RX1 and RX2.  The runner used a +100 kHz
FPGA DDS tone, 3 MS/s, manual RX gain 26 dB, and TX2 gain -10 dB.  TX1 remained
at -80 dB.  Every exit path disabled DDS and returned TX1/TX2 to -80 dB.

The shuffled matrix covered each combination of:

- LO: 868, 915, 1280, 2412, 4000, and 5804 MHz;
- RX and TX RF bandwidth: 0.8, 1.5, and 3.0 MHz;
- capture transport: IP and USB;
- patched host libiio: 0.25 and 0.26.

Each final RF state was read back.  Bandwidth, sample rate, and kernel-buffer
count had to match exactly; RX/TX LO had to be within 10 Hz.  Each frame then
had to contain dual-channel CS16 IQ, increasing capture index, valid hardware
sample counter, valid gain and RSSI endpoints, a nonempty gain-observation
sequence, and no dummy-gain, read-failure, device-overflow, or FPGA-event-
overflow flag.

The spectral gate required at least 6 dB tone SNR on each channel, at least
-75 dBFS tone level, at least 0.90 cross-channel coherence, and at most 8
degrees within-frame phase standard deviation.

## Results

| Host | Radio | Sessions | Frames | IP / USB | Minimum SNR | Minimum coherence | Maximum phase std | Maximum frequency error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | A | 54 | 216 | 27 / 27 | 20.492 dB | 0.9999963 | 0.0960 deg | 21.092 Hz |
| 0.25 | B | 54 | 216 | 27 / 27 | 20.764 dB | 0.9999958 | 0.1000 deg | 21.034 Hz |
| 0.26 | A | 18 | 72 | 9 / 9 | 20.723 dB | 0.9999970 | 0.0816 deg | 20.987 Hz |
| 0.26 | B | 18 | 72 | 9 / 9 | 20.560 dB | 0.9999967 | 0.0885 deg | 21.024 Hz |
| **Total** | **A+B** | **144** | **576** | **72 / 72** | **20.492 dB** | **0.9999958** | **0.1000 deg** | **21.092 Hz** |

The maximum sample-time uncertainty was 12.312 ms in one radio-B 0.25
session; all other per-report maxima were 1.994 ms or lower.  This is consistent
with the already documented EP0 timing sensitivity under shared USB-bus bulk
activity and did not affect IQ, FPGA sample counters, metadata, or spectral
quality.

After the final QSPI reboot, each radio passed one 915 MHz/1.5 MHz session over
IP and one over USB.  Their minimum SNRs were 29.581 and 29.331 dB, respectively.
Final readback showed TX1/TX2 at -80 dB and all eight DDS enables at zero.

## TX setup observation

The first DDS arm was silent in 74 of the 144 main sessions.  A preflight IIO
frame detected this before any metadata frame was accepted.  Repeating the
documented TX-quadrature calibration and DDS arm once in the same context
recovered all 74; no cell required a third arm or a fresh transport/context
session.  Radio/host breakdown of second-arm sessions was:

| Host | Radio A | Radio B |
|---|---:|---:|
| 0.25 | 32 / 54 | 28 / 54 |
| 0.26 | 10 / 18 | 4 / 18 |

TX-quadrature calibration at the requested 0.8 MHz bandwidth also reproduced a
silent TX state on both radios.  Calibrating at 1.5 MHz, then applying and
read-verifying the requested final 0.8 MHz filter, produced the expected tone.

This setup behavior is not caused by the new RX metadata/TCP framing and did
not corrupt an accepted RX frame, but it is not a clean one-shot TX result.
Any production feature that relies on immediately valid TX after a retune
should retain a signal preflight/re-arm or investigate the AD9361 calibration
sequence separately.  RX-only operation is unaffected by the DDS-arm retry.

## Short-frame regression

One preliminary 4096-sample, two-kernel-buffer USB smoke attempt returned
`ENODATA`.  It was not reproduced after the final reboot: 20 fresh processes
(10 per radio), each performing ten refills with the exact same 4096-sample and
two-buffer settings, completed all 200 refills.  The 576 main-burn frames and
eight post-reboot frames also completed without `ENODATA`.

## Evidence

- Radio A, host 0.25 JSON SHA-256: `1cb26ae2b68d2291ad52315af523825cf68dfe00b8550431e40d714a9c044434`
- Radio B, host 0.25 JSON SHA-256: `7797dc1eb0c136bf1457f0890ba7f4434d4a1e4279d1d15d5bf8dcb831f9f801`
- Radio A, host 0.26 JSON SHA-256: `fc3de99e495b62cbfdf439eb2edb4a949e160cf1b90b664f6298396b20de3672`
- Radio B, host 0.26 JSON SHA-256: `e3e00c8e1eeec4598005610d8c5a610a9e5f67c39cd5aeb4ebd354cd677d7f25`
- Radio A post-reboot JSON SHA-256: `7ec93313dff851750942df16e9a56552fb3bafac1ae8179b15eae71e2d00a1c3`
- Radio B post-reboot JSON SHA-256: `93d96a38419ff65714f29cb48b6fd075db0db8aaf5b37632e942cb223aa845c3`

Reusable focused runner:
`tests/radio_hardware/iio_metadata_rf_burn.py`.

The full pytest suite was not run.
