# Direct-USB protocol v2 compatibility result

Date: 2026-07-25

## Outcome

The compatibility-first delivery passed on one Pluto+ and a Raspberry Pi 5.
Each direct-USB frame carried device-local RX1/RX2 gain in dB and RSSI in the
same fixed transfer as dual-channel IQ. The existing Python `rx()`, `rssis()`,
`gains()` API and existing v4 Zarr schema were preserved.

The Pluto was RAM-booted for testing. It was never flashed, and a normal reboot
restored stock QSPI firmware afterward.

## Identity and source

```text
Pluto serial                 104000f6ad020002fdff3a00bba2f096a1
USB physical path            1-1.1
vendor interface             6
bulk IN / OUT                0x89 / 0x07
stock QSPI version            v0.37-dirty
tested RAM image SHA-256      f3cd4d689e7c9ad392edc00eeb6d20da178900fb092eb6afe38a8e003ddbfdf4
published firmware commit     dd6b1f4db710abc20693888db08e8da2427e0dc3
published Buildroot commit    6d5b0298364dc03ae9fb1c0754b83355960b4d63
published gadget commit       54610e01c6fd6a69df77f148ea0dc88f9cb18063
```

The commits were created after the accepted image and capture the tested source
content. A clean rebuild changes embedded Git version text and therefore does
not reproduce the image byte-for-byte.

## Wire and hardware smoke

```text
protocol range               1..2
v2 feature mask              gain endpoints, CRC32, sample sequence,
                             gain dB, RSSI endpoints
v2 header                    96 bytes
Rover IQ payload             4,194,304 bytes
complete transfer            4,194,400 bytes
IQ result                    (2, 524288), complex64
example gain end             [62.0, 62.0] dB
example RSSI end             [125.25, 125.25] dB magnitude
```

Protocol v1 remained available and a two-frame rollback smoke returned raw
indices with sequences 0 and 1.

## Existing Python and Zarr path

The real `PPlus` compatibility path passed:

```python
signal_matrix = pplus.rx()
rssis = pplus.rssis()
gains = pplus.gains()
```

Results:

```text
signal_matrix                (2, 524288), complex64
rssis                        (2,), float64
gains                        (2,), float64
host IIO metadata reads      zero after direct receive
```

The normal `mavlink_radio_collection.py` and `DroneDataCollectorRaw` v4 path
recorded and reopened 100 frames:

```text
signal_matrix                (100, 2, 524288), complex64
gains                        (100, 2), float64
rssis                        (100, 2), float64
gain range                   46..62 dB
RSSI range                   71.5..125.25 dB magnitude
first-to-last                52.6000 s
median interval              0.494120 s
median frame rate            2.02380 Hz
logical IQ rate              15.2091 MiB/s
```

The v2 and IIO v4 receiver groups had the same 13 array keys and identical
shapes and dtypes for `signal_matrix`, `gains`, and `rssis`.

## Gain and RSSI checks

Five-frame manual captures returned:

```text
configured gain              reported gain
20 / 20 dB                   20 / 20 dB on every frame
20 / 40 dB                   20 / 40 dB on every frame
```

One stable out-of-stream standard IIO diagnostic matched the v2 values exactly:

```text
RX1 RSSI / gain              86.75 dB / 20 dB
RX2 RSSI / gain              102.75 dB / 40 dB
```

The one-hour AGC soak observed independent channel gain motion. Endpoint flags
were derived from the Pluto's raw indices, not from equality of rounded dB.
Equal endpoints are not represented as proof of in-buffer stability.

A remotely controlled RF attenuator was not present, so a calibrated stepped
input test was not performed. RSSI sign and scaling were instead verified
against the same local Linux-driver attributes used by the legacy host path.
RSSI remains a positive dB magnitude, not dBm.

## Soak and resource behavior

An early 60-second run exposed a python-libusb1 receive-buffer reference cycle.
Explicit transfer close plus a generation-zero collection at the ownership
boundary fixed it.

Final runs:

```text
10 minutes                  1,200 frames
one hour                    7,200 frames
one-hour elapsed            3600.00065 s
one-hour rate               1.99999964 Hz
initial-to-final RSS growth 17,844 KiB
maximum RSS                 354,444 KiB
gain-read maximum           850,500 ns
RSSI-read maximum           710,220 ns
gain endpoint changes       RX1 0, RX2 5,087
gain-read failures          0
RSSI-read failures          0
invalid frame/header/IQ     0
```

Memory reached a stable plateau in both the 10-minute and one-hour runs; it did
not grow with frame count.

## Tests and build

```text
focused SPF tests           163 passed
fake-drone regression       3 passed
gadget C tests              5 passed
ARM cross-build             passed
remote-pinned Buildroot     fetched, cross-built, installed
Zarr validator              passed 100/100 frames
```

The full repository test run was stopped in an unrelated, CPU-heavy numerical
section after reaching 58% without failures. It is not used as evidence for the
firmware result; the complete transport, collector, dataset, and protocol test
set is the scoped acceptance suite above.

## Rollback

After the soak, a normal device reset restored:

```text
firmware                     v0.37-dirty
standard USB-IIO             present and capturing
custom vendor interface      absent
serial                       unchanged
```

Rollback passed and QSPI was not modified.

The requirement-by-requirement audit, including the remaining calibrated-RSSI
and coherent-phase bench work, is in:

```text
docs/direct_usb_gain_completion_audit.md
```
