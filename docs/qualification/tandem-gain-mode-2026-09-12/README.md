# Tandem gain-mode qualification — 2026-09-12

Radio `104473b80a16000de6ff2000f8a6beca79` was flashed using PPU with the
latest published full release, `v0.49-plutoplus-spf-iq-direct-async-v4`, then
tested with the legacy-AGC metadata provider extension and SPF gain-mode API.
Its current DHCP address is `192.168.1.173`.

## Deployment attestation

- Official DFU SHA256: `f45524f4765d5743144703ff6f4541084ff1ab9b1ce20a77f3f6fa820a1f84b6`.
- Read-back FIT SHA256: `77f899610548d486aab2c83c4dc7170532d470b115d2bd0e8fc43e72b3bfca67`.
- Provider source: libiio `f7bdb33235ad6670a63c8eed253743fa7fd23102`.
- Installed provider SHA256: `7b1abdfffcf1c9ebda365f4a81976d829d9d2c4c64b93301a98020579a3432af`.
- Base host/radio libiio: `5cb2389719d46d12463daa0371d1fda19eb25fa7`.
- Kernel UAPI: `7176508dd84bde78c62d8790bbd17957fdda12d7`.
- Metadata source: `3294365ff44da26b261be4a2ccb241b7896d23ad`.

The provider is installed in `/mnt/jffs2/spf-gain-mode/iiod`; the accompanying
`autorun.sh` reproduces the tested persistent overlay. It verifies both the
provider and stock executable hashes. The stock backup remains in
`/mnt/jffs2/spf-gain-mode/iiod.release`. Disabling this autorun hook and rebooting
restores stock v0.49. The base FIT and bootloader were not changed by the overlay.

PPU's flash receipt `0a165712-18a5-4387-89ce-f59ee0335429` reports an unknown
final outcome because DHCP moved the radio from `.179` to `.162`. The later
verification reboot moved it to `.173`, also timing out the old-address probe.
These receipt limitations were reconciled separately with PPU's read-only
serial, firmware, boot-ID, and FIT-hash checks at the new addresses. SSH keys
were enrolled through PPU for the selected serial. Original receipts were
retained unchanged in the private deployment evidence directory.

The final attestation confirms both buffers disabled, TX LO powered down, DDS
disabled, and attenuation at -80 dB. Tandem state and ownership epoch are zero,
fault flags are clear, and both RX channels are left in manual mode.

## Results

- 128 software regression tests passed.
- 10 hardware tests passed after reboot in 83.64 seconds. A final metadata
  validation cleanup change passed a further hardware rollback smoke test.
- C provider serialization/request tests passed with address and undefined
  behavior sanitizers.
- 46 PPU diagnostic-profile tests passed for the small v0.49 recognition fix.

The mode matrix covered 915 MHz, 2.41 GHz, and 5.75 GHz, with 262144 and 524288
samples per channel at 2.5 MSPS. Every combination ran manual → tandem → slow
→ tandem → fast → tandem → manual. The attached 126 frame records contain 18
distinct tandem ownership epochs and 138 gain events, with zero reported
missing samples. Unequal manual gains of 20/23 dB were restored correctly.

Failure tests covered failed enable after ownership acquisition, watchdog
expiry, abrupt owner-process termination, and a dropped private TCP proxy
connection followed by reconnection. Unit tests additionally cover invalid
requests before mutation, rollback failure stopping RX, corrupt metadata,
recording compatibility, and preserving the metadata buffer when PyADI clears
its reference after a recoverable startup discard. That last regression was
found and fixed during the post-reboot hardware run.

Qualification was receive-only over LAN IIO. No controlled RF source/harness was
specified, so amplitude-response sweeps and phase/coherence benefit remain
unqualified. USB transport was not exercised. This is a qualified extension on
top of v0.49, not a new published firmware release.

See [usage and test commands](../../tandem_gain_modes.md). The matching host
library and Python binding must be active; an older system libiio is insufficient.
The test host used the exact pinned library in an isolated environment. PPU
main's environment validator expects a newer development library; its installer
receipt is not evidence of validation for this release-pinned environment.
