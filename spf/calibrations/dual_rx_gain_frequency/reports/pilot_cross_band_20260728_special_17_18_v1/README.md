# Historical-IP special-radio automated pilot

## Outcome

The first end-to-end use of the automated new-radio calibration command
completed successfully on the two radios with historical labels
`192.168.1.17` and `192.168.1.18`.

The automation performed, in order:

1. persistent AD9361/2R2T verification;
2. checksum-pinned direct-USB firmware RAM loading on both radios;
3. post-enumeration device mapping;
4. direct-USB protocol/capability verification;
5. passive post-firmware hardware fingerprinting;
6. per-serial TX2-off/TX2-on loopback probing;
7. a 324-frame V7 cross-band pilot per radio; and
8. full stored-IQ recomputation and validation.

No experimental firmware was written to QSPI.

## Reproduction

The successful run used clean SPF commit
`1882aeaa2b08603acf2fa939c0b8866b259450a3`:

```bash
python -m spf.calibrations.dual_rx_gain_frequency automate \
  --config \
    spf/calibrations/dual_rx_gain_frequency/configs/pilot_cross_band.yaml \
  --output \
    artifacts/dual_rx_gain_frequency/pilot_cross_band_20260728_special_17_18_v1 \
  --expected-radios 2 \
  --resume
```

`--resume` was used because an earlier attempt had safely stopped before
firmware loading when it exposed virtualenv interpreter-path handling. Commit
`1882aea` fixed that integration issue and added a regression test. No dataset
or radio mutation preceded the successful attempt.

The complete preparation, capture, and validation took approximately
11 minutes. The run root is:

```text
artifacts/dual_rx_gain_frequency/pilot_cross_band_20260728_special_17_18_v1
```

## Radio identity and provenance

Historical IP is a human provenance label only. Serial and V7 hardware
fingerprint remain authoritative.

| Historical IP | Pluto serial | USB path after RAM load | Stable fingerprint SHA-256 |
| --- | --- | --- | --- |
| `192.168.1.17` | `104000bac4950008230026001b440a003a` | `1.1` | `f3a958fea8fa43336986404c1d196ce2551b4aecb8b1edde401087eb86c45c99` |
| `192.168.1.18` | `1040007c4a94000211000b009186843ef2` | `1.2` | `854599ff8d81be79799ab0752e233cea0bc6f39f214406b66c7c7103efca70ae` |

Both V7 stores record:

- direct-USB protocol version 2;
- firmware verified `true`;
- firmware image SHA-256
  `0a6a8939b31babed2ad7093d83941ebc809323d69804adcd8da5bcae0e48d3e9`;
- firmware Git SHA `7b02276519a802aed83d47b6672c46e578ce4de0`;
- gadget Git SHA `a1e6417d07188bd72be70692e28c5d6ae9a5ec62`;
- SPF Git SHA `1882aeaa2b08603acf2fa939c0b8866b259450a3`;
- software dirty flag `false`; and
- one distinct passive hardware fingerprint per physical serial.

## Probe results

Both radios selected the first direct-RX/DDS handoff and passed at 868 MHz,
26 dB equal gain, using TX2 FPGA DDS.

| Historical IP | RX1 TX-on/off delta | RX2 TX-on/off delta | RX1 tone SNR | RX2 tone SNR | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `.17` | 70.33 dB | 67.06 dB | 13.79 dB | 30.83 dB | Pass |
| `.18` | 69.41 dB | 72.39 dB | 31.90 dB | 31.77 dB | Pass |

The observed tone was at approximately `+100.246 kHz`, matching the configured
`+100 kHz` DDS offset within the capture resolution.

## Capture and validation

| Historical IP | Stored frames | Gain metadata valid | RSSI metadata valid | Quality-valid frames | Passing cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| `.17` | 324/324 | 324/324 | 324/324 | 245/324 | 82/108 |
| `.18` | 324/324 | 324/324 | 324/324 | 243/324 | 81/108 |
| Total | 648/648 | 648/648 | 648/648 | 488/648 | 163/216 |

Both datasets have shape `complex64[324, 2, 65536]`. Validation read every IQ
frame back from LMDB and recomputed the stored tone, phase, and quality
metrics. There were no capture errors and no preflight-failure sidecars.

The validation status is `fail_quality`, not a capture-integrity failure. The
pilot intentionally includes the complete ordered Cartesian grid for gains
`[-1, 26, 62]`. Extreme asymmetric pairs can make one channel too weak and
unstable, so those cells remain recorded but unsupported.

Per-frequency passing-cell counts were consistent:

| Frequency region | `.17` | `.18` |
| --- | ---: | ---: |
| 868 MHz through 5.766 GHz | 7/9 at each frequency | 7/9 at each frequency |
| 5.804 GHz | 7/9 | 6/9 |
| 5.838 GHz | 6/9 | 6/9 |
| 5.866 GHz | 6/9 | 6/9 |

This pattern supports proceeding to the dense design: both equal and moderate
gain cells work across all three AD9361 gain-table regions, while the expected
extreme weak-channel cells fail closed.

## Evidence hashes

The large V7 stores remain under the Git-ignored `artifacts/` tree. These
hashes bind this report to the compact orchestration and validation evidence:

| Evidence file relative to run root | SHA-256 |
| --- | --- |
| `automation_plan.json` | `42b64d2ed21b82a0fb02094972725da3e2aa86003b9edfd8a0350574a4fbac2c` |
| `automation_result.json` | `a5f805fdfb6339cefc1b95067cf783454da351a55f7b0063a341a172d9762f1f` |
| `run_result.json` | `00671f5efd850831e62dcc5982f251b513089fad6a954999ba389abe38134473` |
| `.17/probe.json` | `bf1c5f7907591c9b5321f39f8c7b9a4eb2044817ea740e01a15f4d5c14af768f` |
| `.17/validation.json` | `f8908abc111470c22aa70c2dba8b2b53c3e8a023d9e6e92cd369d812f3356234` |
| `.18/probe.json` | `500270a2c0726e921cde55898fd476b95813ae4495e8406ba25b832fd7395e43` |
| `.18/validation.json` | `86878a409713dacdeb2b60944786399ac77c73c7e5f7da97004d73204d2e5f19` |

## Next step

Use a new output root with `survey_cross_band.yaml` to collect the complete
17-by-17 gain grid for these same two serials. The automated command should
again perform the RAM load and readiness sequence from scratch. Do not reuse
this pilot root for the dense survey.
