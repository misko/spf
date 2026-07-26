# Pluto+ direct-USB runtime baseline

Date: 2026-07-24
Overall result: **CORRECTED — RAM firmware and rollback pass; the original
rate tests used USB-IIO rather than the custom direct-USB transport**

No firmware was persistently flashed.

## Device under test

- USB path before RAM boot: `usb:1.3.5`
- USB serial: `104000f6ad020002fdff3a00bba2f096a1`
- Original QSPI firmware: `v0.37-dirty`
- Device-tree model: `Analog Devices PlutoSDR Rev.C (Z7010/AD9363)`
- U-Boot overrides: `mode=2r2t`, `attr_val=ad9361`,
  `refclk_source=internal`

The full pre-boot `fw_printenv` output was captured before rebooting.

## RAM boot

Artifact:

```text
/home/pi/spf-direct-usb/plutosdr-fw/build/pluto.dfu
SHA-256 2c814c38190ceb61ec58ddc4214af2a914c376a4336d1958e382504d68298a55
```

Procedure:

1. Requested `/usr/sbin/device_reboot ram` on the original image.
2. Confirmed USB DFU ID `0456:b674`.
3. Confirmed the RAM-only DFU alternates:
   - alternate 0: `dummy.dfu`
   - alternate 1: `firmware.dfu`
4. Uploaded `pluto.dfu` only to alternate 1.
5. Issued DFU detach/execute.

The transfer completed with `DFU state(2) = dfuIDLE` and status 0.

## Enumeration and control-path result

**PASS**

The RAM image re-enumerated as `0456:b673`. Standard USB-IIO worked:

```text
fw_version: v0.38_plutoplus_with_timestamping-1-ga098-dirty
hw_model: Analog Devices PlutoSDR Rev.C (Z7010-AD9361)
IIO backend: 0.25
Linux: 5.15.0-gd798b0d821b8
```

The composite device exposed:

- interface 5: standard IIO FunctionFS;
- interface 6: vendor-specific `sdrgadget`;
- bulk IN endpoint `0x89`;
- bulk OUT endpoint `0x07`.

On-device process and mount checks passed:

```text
/usr/sbin/iiod -D -n 3 -F /dev/iio_ffs
/usr/sbin/sdr_usb_gadget /dev/sdr_gadget_ffs
iio_ffs on /dev/iio_ffs type functionfs
sdr_gadget_ffs on /dev/sdr_gadget_ffs type functionfs
```

The host RNDIS interface did not receive an IPv4 address automatically. A
temporary `192.168.2.10/24` address was assigned to `eth1` for SSH inspection.
This did not affect direct USB streaming.

## Reference host client

- `pgreenland/SoapyPlutoSDR` branch `sdr_gadget_timestamping`
- SHA: `2bbf77152d4e6d30c6630807fdfc8a869a528cf3`
- Built locally against SoapySDR 0.8.1, libiio 0.24, and libusb 1.0.26.
- The module was loaded from its build directory and was not installed.

## Dual-channel streaming correction

The tests below omitted the required Soapy device argument `direct=1`.
Inspection and explicit retesting on 2026-07-25 showed that the reference
branch does not select the custom gadget merely because it is present.
Consequently, these numbers describe standard USB-IIO streaming on the RAM
firmware and must not be cited as direct-USB performance.

See `../2026-07-25/timestamp_validation.md` for an explicitly direct test and
end-to-end timestamp validation.

### 1 MS/s

**PASS for USB-IIO, not direct USB**

Two independent runs successfully started two-channel CS16 RX through the
direct USB gadget:

```text
Num channels: 2
Element size: 4 bytes per complex channel
observed: 0.99 MS/s
observed: 7.9 MB/s
```

The byte rate matches:

```text
1e6 samples/s * 2 RX * I/Q * 2 bytes = 8 MB/s
```

The gadget daemon remained alive and a second stream started successfully,
passing the basic stop/restart lifecycle check.

### 6 MS/s

**FAIL for the requested rate through USB-IIO; not a direct-USB measurement**

Requested dual-channel CS16 rate:

```text
6 MS/s = 48 MB/s payload
```

Observed:

```text
approximately 2.33 MS/s
approximately 18.6 MB/s
```

The existing reference stack therefore does not sustain the handoff's 6 MS/s
example rate on this development host. The current Rover SPF configuration of
30 MS/s dual-channel CS16 would require 240 MB/s and cannot cross USB 2.0.

This is a throughput/design blocker for production rates, but not a blocker
for implementing and validating the metadata vertical slice at a lower rate.
Before Rover deployment, one or more of the following is required:

- lower the sample rate;
- decimate on the Pluto FPGA/ARM;
- transfer a genuinely packed representation on the wire;
- optimize the gadget/host implementation and retest, within USB 2 limits.

Changing Soapy's returned format to CS8 or CS12 does not by itself reduce USB
wire bandwidth because the reference gadget still transports the IIO CS16
containers before host conversion.

## Rollback

**PASS**

A normal reboot from the RAM image returned to the untouched QSPI image:

```text
USB URI after rollback: usb:1.7.5
fw_version: v0.37-dirty
```

This demonstrates that the development image was RAM-only and that the known
firmware remained bootable.
