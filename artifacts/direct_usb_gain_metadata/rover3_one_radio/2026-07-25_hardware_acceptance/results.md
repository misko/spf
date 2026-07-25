# Pluto+ direct-USB gain metadata hardware acceptance

Date: 2026-07-25

## Hardware and tested image

- Pluto serial: `104000f6ad020002fdff3a00bba2f096a1`
- USB physical path: `1-1.1`
- Direct interface: 6
- Bulk IN/OUT: `0x89` / `0x07`
- Installed QSPI firmware before and after the test: `v0.37-dirty`
- Tested RAM image:
  `/home/pi/spf-direct-usb/plutosdr-fw/build/pluto.dfu`
- Tested image SHA-256:
  `fd8910295643b6f72d8aa30d0fa179f813a891eba452ac1605bbc529794c548a`
- RAM firmware banner:
  `v0.38_plutoplus_with_timestamping-1-ga098-dirty`
- Gadget source snapshot commit:
  `eaf850d846d8183e2345374c3d732d457ef8f8ba`
- Buildroot development-logging snapshot:
  `9ca05caa5daada1715599fc89396aa5821daea08`
- Firmware build-source snapshot commit:
  `7c4eda5ca78cc9cc002dd649561a560ce76aa5e2`
- SPF base commit plus this working tree:
  `d5b4bf4878133114a058ca09f8f39112147fe184`

The source snapshot commits were made after the accepted image was built and
therefore capture its source content, while the tested image retains the
embedded `a098-dirty` version string and the SHA-256 above.

The image was loaded with DFU into RAM. No QSPI write was performed.

## Protocol and small-frame smoke

Capability negotiation returned:

```text
protocol range             1..1
supported features         gain endpoints, header CRC32, sample sequence
maximum finite frames      16
capability flags           finite RX
dummy-gain capability      absent
```

One 4,096-sample, two-channel frame passed:

```text
header                     80 bytes, CRC valid
IQ payload                 32,768 bytes
shape/type                 (2, 4096), complex64
gain start                 [76, 76]
gain end                   [76, 74]
gain metadata valid        true
dummy gains                false
both IQ channels nonzero   true
```

## Exact Rover frame

The radio was configured through pyadi/USB-IIO before direct RX:

```text
LO                         5.766 GHz
sample rate                30 MS/s
RF bandwidth               3 MHz
gain mode                  slow_attack
kernel RX buffers          4
samples/channel/frame      524,288
```

One direct frame passed:

```text
framed USB bytes           4,194,384
shape/type                 (2, 524288), complex64
capture time               0.403 s
gain start/end             [76,76] / [76,76]
gain metadata valid        true
gain read durations        486,205 ns / 498,432 ns
legacy gain/RSSI           NaN / NaN
```

The on-device diagnostic log confirmed full-table mode, exact transfer size,
finite completion, clean STOP, and zero gain-read failures.

## Normal 100-frame Rover capture

Dataset:

```text
artifacts/direct_usb_gain_metadata/rover3_one_radio/
  2026-07-25_zarr_direct_usb_100/
```

Strict validation passed:

```text
frames                     100
shape/type                 (100, 2, 524288), complex64
first-to-last time         53.250 s
median frame interval      498.440 ms
median frame rate          2.006 Hz
logical IQ rate            15.024 MiB/s
unique explicit streams    100
endpoint-changed frames    68
unsafe metadata flags      0
gain-read failures         0
device overflow flags      0
legacy gain/RSSI reads     absent (stored as NaN)
start read p50 / p99       488,133 / 549,695 ns
end read p50 / p99         494,664 / 535,522 ns
```

Each normal collector snapshot is one negotiated finite START, so each has a
new nonzero stream ID and sequence zero. Resets are therefore explicit rather
than hidden discontinuities.

The matched USB-IIO baseline was 2.058 Hz and 16.46 logical MiB/s. Direct USB
met the 0.5-second Rover cadence with about 2.5% lower frame rate while adding
the gain metadata and removing per-frame host gain/RSSI reads.

## Manual gain and channel mapping

Two additional normal 100-frame v6 datasets passed strict validation:

```text
manual 20/20 dB:
  start/end raw indices    [34,34] in every frame
  endpoint changes         0 / 100

manual 20/40 dB:
  start/end raw indices    [34,54] in every frame
  endpoint changes         0 / 100
```

Intentional changes were then made through the retained control-only IIO
interface during bounded three-frame streams:

```text
RX1-only toggling:
  observed                 [54,34] -> [34,34]
  flags                    RX1 changed=true, RX2 changed=false

RX2-only toggling:
  observed                 [34,64] -> [34,34] -> [34,64]
  flags                    RX1 changed=false, RX2 changed=true
```

This proves independent register mapping and exact agreement between endpoint
comparisons and channel-specific flags. It does not imply that equal endpoints
prove gain stability inside a frame.

## IQ order and rollback

With manual gains 0/60 dB:

```text
direct USB indices         [14,74]
direct power dBFS          [-66.74, -9.43]
USB-IIO power dBFS         [-66.69, -47.44]
stronger channel           channel 1 in both transports
```

Direct RX was stopped before the USB-IIO capture; an attempted competing IIO
RX buffer correctly returned `EBUSY`.

A normal reboot after RAM testing restored QSPI `v0.37-dirty`. The custom
capability query was no longer discoverable and standard two-channel USB-IIO
captured a valid `(2,524288)` frame. Rollback passed.

## Host robustness finding

The Pi has a 16 MiB USBFS pool. Four queued 4,194,384-byte transfers exceed it,
so the production client intentionally queues one Rover frame. A partial
multi-transfer allocation initially exposed a cleanup edge case; the receiver
now cancels every successfully submitted transfer even when a later submission
fails before START. A regression test covers this path.

No common coherent CW was attached, so phase values from the ambient AGC
capture are not interpreted as phase characterization. FPGA CTRL_OUT
sample-aligned event capture remains a separate future project.

Final focused verification:

```text
SPF Python tests            154 passed
gadget protocol/gain tests  3 passed with -Wall -Wextra -Werror
git diff whitespace check   passed
```
