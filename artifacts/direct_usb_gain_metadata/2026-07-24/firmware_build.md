# Pluto+ direct-USB firmware build evidence

Date: 2026-07-24
Result: **PASS — build and static validation only; nothing was booted or flashed**

## Source identity

- `pgreenland/plutosdr-fw`: `a098f87b2302dde8bc3f442133800f6386512960`
- Buildroot: `af0a100f5479ccbf46a8ca0cfcb3907e87f3e9ce`
- HDL: `be89a77d3fd0b344419377fac6fab8cfc7a66ad8`
- Quantulum HDL: `d70102267713f5bbc99805be5f4f08b0a07766cb`
- Linux: `d798b0d821b85ebd51ecffbfa68d8e4d69b77132`
- U-Boot: `1ff0468e9bea29b0a768a7bf52db8d025c521b9a`
- Direct USB gadget: `7339ab52678459da4c52a457f3259d52e7adf007`

The firmware tree is outside the SPF repository at
`/home/pi/spf-direct-usb/plutosdr-fw`.

## PDF checksum investigation

The two archives called out in `spf.pdf` were independently reproduced using
this firmware branch's pinned Buildroot Git downloader. Both match the
repository's original hash files:

- `ad936x_ref_cal`:
  `26aedd8021fa939ab2f53e55904d869207265242fef7ad86aa4673e219b7cbef`
- `libiio`:
  `e791ad1cf35aef08fc6e2b6b0dcdd1cc21d36cf287d81fa14adb088c6c1d4c49`

Therefore the alternate hashes recorded in the PDF were not applied and no
Buildroot package hash file was changed.

## Required local build compatibility changes

Two changes were required in the external firmware tree's top-level
`Makefile`:

1. The release XSA URL was changed from the now-missing Analog Devices asset
   to the matching `pgreenland/plutosdr-fw` release asset.
2. `CROSS_COMPILE` was made overridable and its default changed from
   `arm-linux-gnueabihf-` to `arm-none-linux-gnueabihf-`, matching the pinned
   Arm GNU 2021.07 toolchain installed by this Buildroot version.

Vivado is not installed on this host. The build used the verified release
`system_top.xsa`, whose HDL content is from the same release lineage as the
checked-out branch.

Build command:

```sh
make SKIP_LEGAL=1
```

`SKIP_LEGAL=1` was used because a legal-source download endpoint for this old
Buildroot release is no longer available. It skips the separate legal-source
archive, not package download hash verification. The Pluto mass-storage image
still requires `LICENSE.html`, so that page was generated from the legal
metadata already collected in `buildroot/output/legal-info` and copied into
`buildroot/board/pluto/msd/` before the successful resume.

One normal root-filesystem package, `libdaemon-0.14.tar.gz`, also had a dead
primary URL. It was obtained from the Buildroot source mirror and accepted
only after matching the branch's pinned SHA-256:

```text
fd23eb5f6f986dcc7e708307355ba3289abe03cc381fc47a80bca4a50aa6b834
```

## Produced artifacts

```text
2c814c38190ceb61ec58ddc4214af2a914c376a4336d1958e382504d68298a55  build/pluto.dfu
d1cf801fa214b0e5a77aa09946e1bdc9a5a30298abce0f7989e8ef4a8ecf9f74  build/pluto.itb
8f65f014ef95db23192a7167ddf5cf93c92c6737f55263ec9b0566b6c7238906  build/pluto.frm
39906c16971dd47d02e37e972a283c8650cbdf2512bcd80126064dd38b9eab27  build/uboot-env.dfu
e07af4a31973e332f1c7b19a20b8d9527df6ccf91d3b805db417e0164981be3a  build/system_top.xsa
2cf8eda39cc52e70888931bd40e40b6dd9d9e8d6921b0d40425b4e6a6ad91d09  build/plutosdr-fw-v0.38_plutoplus_with_timestamping-1-ga098-dirty.zip
```

The `dirty` suffix records the two deliberate Makefile compatibility changes
above.

## Static validation

- `build/pluto.dfu` and `build/pluto.itb` are valid FIT/device-tree images.
- FIT inspection reports kernel, gzip ramdisk, FPGA bitstream, and RevA/RevB/
  RevC device-tree configurations.
- The XSA is a readable ZIP archive and contains `system_top.bit`,
  `system.hwh`, and the Zynq PS7 initialization sources.
- `buildroot/output/target/usr/sbin/sdr_usb_gadget` is an executable,
  stripped 32-bit ARM EABI5 binary.
- `S23udc` creates both `ffs.iio_ffs` and `ffs.sdr_gadget_ffs`, starts both
  `iiod` and `sdr_usb_gadget`, and preserves the dual control/data-function
  architecture required by the handoff.

## Remaining gate

This evidence does not prove runtime compatibility with the attached Pluto+.
Before any persistent update:

1. Back up the current U-Boot environment and retain the known-good image.
2. Confirm DFU recovery.
3. RAM-boot `build/pluto.dfu`.
4. Verify both standard IIO and the custom direct-USB interface enumerate.
5. Prove two-channel IQ and rollback.

Persistent flashing remains out of scope until those checks pass.
