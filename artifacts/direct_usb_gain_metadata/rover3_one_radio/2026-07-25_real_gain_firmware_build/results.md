# Real-gain direct-USB RAM image build

Date: 2026-07-25

Result: **PASS — source, host tests, ARM cross-build, and RAM image build**

Runtime status: **pending a physical Pluto reset before RAM boot**

## Source identity

- Firmware base: `a098f87b2302dde8bc3f442133800f6386512960`
- Direct USB gadget base:
  `7339ab52678459da4c52a457f3259d52e7adf007`
- Firmware tree: `/home/pi/spf-direct-usb/plutosdr-fw`
- Gadget tree: `/home/pi/spf-direct-usb/pluto-sdr-usb-gadget`

Both external trees are intentionally dirty. The gadget changes add protocol
v1, finite RX, local gain reads, and tests. The firmware tree selects that
local gadget through `buildroot/local.mk`, applies the already-documented XSA
and cross-toolchain compatibility changes, and enables gadget logging by
default only in this development RAM image.

## Gain implementation

The device-local helper:

- finds `ad9361-phy`;
- verifies `adi,split-gain-table-mode-enable` is false;
- reads RX1 register `0x2b0` and RX2 register `0x2b5`;
- masks both values with `0x7f`;
- invalidates the pair as `[0xff, 0xff]` if either read fails;
- records pair-read duration using `CLOCK_MONOTONIC_RAW`.

The read thread caches one pair before the receive loop. Immediately after each
successful `iio_buffer_refill()`, it reads the next pair and emits:

```text
gain_index_start = prior pair
gain_index_end   = current pair
```

Endpoint equality is only an observed comparison; it is not represented as
proof of in-frame stability.

## Validation

Host-native builds and tests passed with warnings treated as errors:

```text
test_spf_gain_metadata
test_sdr_usb_gadget_protocol
test_spf_gain_read
```

The full gadget also built natively with CMake. Buildroot explicitly rebuilt
the local override with:

```sh
make -C buildroot sdr_usb_gadget-rebuild
make SKIP_LEGAL=1 build/pluto.dfu
```

The installed gadget is a stripped 32-bit ARM EABI5 executable. Static string
inspection confirms that the final root filesystem contains the full-table
guard and gain-read diagnostics. Extracting `usr/sbin/sdr_usb_gadget` directly
from the final `build/rootfs.cpio.gz` produces the same SHA-256 as the staged
target binary, proving the intended gadget is inside the RAM image.

## Output identity

```text
fd8910295643b6f72d8aa30d0fa179f813a891eba452ac1605bbc529794c548a  build/pluto.dfu
96d78b916a72dd5c4861096f95eea1a8986369b73d94a6aa5966153aad82f03e  build/pluto.itb
9fb14894d0180b9de95e44777ede9d2e1f93c9c8b5c2a719a933a53d0fa29e04  buildroot/output/target/usr/sbin/sdr_usb_gadget
```

`build/pluto.dfu` is 13,730,911 bytes.

## Current hardware boundary

The previous dummy-metadata RAM image successfully enumerated and answered the
protocol capability request. Its first finite bulk capture timed out. While
enabling live diagnostics, stopping the FunctionFS gadget daemon removed the
whole Pluto composite device before it could be restarted.

Recovery paths attempted from the Pi:

- direct child-port power cycle;
- complete upstream USB2 hub power cycle;
- USB reset/re-enumeration checks;
- USB-network, ACM serial, and IIO discovery;
- the Rover power-board supervisor at I2C address `0x36`.

The Pluto remains externally powered and exposes no control path. The I2C
supervisor is not present on this assembled rover. QSPI was never modified, so
a physical Pluto power cycle returns the known v0.37 image.

After that reset, the next safe operation is another DFU RAM boot of the exact
image above, followed by inspection of `/var/log/sdr_usb_gadget.log` and a
small-frame finite RX test before the 524,288-sample test.
