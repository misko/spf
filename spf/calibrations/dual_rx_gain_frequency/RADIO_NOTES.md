# Physical radio notes

This is the human-maintained registry for historical radio labels that are not
part of the hardware fingerprint. The Pluto serial is the authoritative key.
An IP address in this file is historical provenance only: it may be reassigned,
may no longer be reachable, and must not be used as the radio's current
identity or connection target.

Calibration software and analysis must continue to identify a radio from the
serial and stored V7 hardware fingerprint.

## Historical IP mapping

| Pluto serial | Historical IP | Calibration dataset roots | Notes |
| --- | --- | --- | --- |
| `1040007c4a94000211000b009186843ef2` | `192.168.1.18` | `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/pilot_cross_band_20260728_special_17_18_v1/1040007c4a94000211000b009186843ef2/calibration.v7.zarr`<br>`/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/survey_cross_band_20260728_special_17_18_v1/1040007c4a94000211000b009186843ef2/calibration.v7.zarr` | First inventoried alone on 2026-07-28 before SPF RAM loading. Persistent `ad9361`/`2r2t` verification passed. Dense survey: 10,404/10,404 frames, 10,180 quality-valid, 3,392/3,468 passing cells. Runtime model configs are keyed by this serial under `spf/calibrations/models/`. |
| `104000bac4950008230026001b440a003a` | `192.168.1.17` | `/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/pilot_cross_band_20260728_special_17_18_v1/104000bac4950008230026001b440a003a/calibration.v7.zarr`<br>`/mnt/qnap01/mouse9911/spf/calibration_data/raw/dual_rx_gain_frequency/survey_cross_band_20260728_special_17_18_v1/104000bac4950008230026001b440a003a/calibration.v7.zarr` | Identified as the newly attached radio on 2026-07-28 before SPF RAM loading. Persistent `ad9361`/`2r2t` verification passed. Dense survey: 10,404/10,404 frames, 10,185 quality-valid, 3,395/3,468 passing cells. Runtime model configs are keyed by this serial under `spf/calibrations/models/`. |

## Initial read-only inventory

The following observations establish which physical radio received the
historical label. USB device numbers are included only as contemporaneous
evidence and are not stable identifiers.

| Field | Observed value |
| --- | --- |
| Inventory date | 2026-07-28 |
| Attached Pluto count | 1 |
| Pluto serial | `1040007c4a94000211000b009186843ef2` |
| Historical IP supplied by operator | `192.168.1.18` |
| USB sysfs path | `1-1.2` |
| USB bus/device at inventory | `001/071` |
| USB product before RAM loading | `PlutoSDR (ADALM-PLUTO)` |
| USB device release | `0510` |
| USB-IIO URI before RAM loading | `usb:1.71.5` |
| QSPI/runtime firmware before RAM loading | `v0.37-dirty` |
| Persistent radio configuration | PASS: `ad9361`, `2r2t` |
| Standard USB-IIO | Present |
| SPF vendor direct-USB interface 6 | Absent before RAM loading |
| `iiod` | Running |
| `sdr_usb_gadget` | Absent before RAM loading |

The second special radio was then attached alongside the already identified
`.18` radio. Subtracting the known serial from the two-device inventory made
the new mapping unambiguous:

| Field | `.17` radio | `.18` radio at second inventory |
| --- | --- | --- |
| Inventory date | 2026-07-28 | 2026-07-28 |
| Pluto serial | `104000bac4950008230026001b440a003a` | `1040007c4a94000211000b009186843ef2` |
| Historical IP | `192.168.1.17` | `192.168.1.18` |
| USB sysfs path | `1-1.1` | `1-1.2` |
| USB bus/device at inventory | `001/073` | `001/074` |
| USB-IIO URI before RAM loading | `usb:1.73.5` | `usb:1.74.5` |
| USB product before RAM loading | `PlutoSDR (ADALM-PLUTO)` | `PlutoSDR (ADALM-PLUTO)` |
| USB device release | `0510` | `0510` |
| Approved QSPI/runtime firmware | `v0.37-dirty` | `v0.37-dirty` |
| Persistent radio configuration | PASS: `ad9361`, `2r2t` | PASS: `ad9361`, `2r2t` |
| Standard USB-IIO | Present | Present |
| SPF vendor direct-USB interface 6 | Absent before RAM loading | Absent before RAM loading |

## Updating this registry

When collecting from a registered radio:

1. Match the attached device to the exact serial above.
2. Complete the normal post-firmware readiness and fingerprint checks.
3. Let the V7 writer store the serial and hardware fingerprint in the Zarr.
4. Replace `Pending` with the pilot and dense run roots, separated with
   `<br>` when more than one is present.
5. Do not change the historical IP merely because the radio is currently
   reached at another address.

For a new historical mapping, attach that radio alone, gather the same
read-only inventory, and add a new row. Never infer the mapping from USB
enumeration order after multiple radios are attached.
