# Passive Pluto hardware compatibility fingerprints

Rover data-version-7 captures record a versioned, per-receiver hardware
compatibility fingerprint. The fingerprint helps distinguish physical radios
and detects operational mismatches; it is not proof that a board is genuine or
counterfeit.

The workflow is strictly passive. It never starts RX or TX DMA, enables a DDS,
changes LO/gain/bandwidth, or performs a 2R2T RF functional test. It may report
that the persistent configuration is `2r2t`, but it never reports that 2R2T was
functionally verified.

## Boot ordering and failure policy

`prepare_direct_usb_boot.sh` performs the following fail-closed sequence:

1. Delete `/run/spf/direct_usb_ready.json` before parsing configuration.
2. Enumerate every attached Pluto by USB serial and physical path.
3. Verify the persistent `ad9361`/`2r2t` configuration.
4. RAM-load the exact image and SHA-256 named by the Rover YAML.
5. Wait for the same serials to re-enumerate.
6. Regenerate `/home/pi/device_mapping`.
7. Verify USB-IIO, direct USB, protocol v2 and required metadata capabilities.
8. Query the firmware's passive hardware-identity endpoint.
9. Read a fixed allowlist of USB, IIO, device-tree, memory, flash and U-Boot
   configuration facts.
10. Atomically publish a session-bound ready manifest.

No capture service is authorized until every configured radio passes. The
manifest is bound to the Pi boot ID, firmware build, radio serial, USB path and
fingerprint session. A reboot or firmware reload invalidates it.

Inspect and verify the result with:

```bash
sudo systemctl status spf-pluto-direct-usb.service
python3 -m json.tool /run/spf/direct_usb_ready.json
python3 -m spf.scripts.pluto_ready_manifest verify \
  --rover-id "$(cat /home/pi/rover_id)"
```

Each radio row contains `hardware_fingerprint`. Two attached radios must have
different `stable_fingerprint_sha256` values.

## Identity and privacy boundary

The required board anchor is the SPI-NOR factory UniqueID. The Pluto firmware
already uses that value as its USB/IIO serial, so it is available
programmatically and is already part of the receiver provenance. SPF also
stores a domain-separated HMAC-SHA256 of it in the stable fingerprint. This
does not prove board manufacture; it binds the observed compatibility facts to
the same flash device and catches duplicate identities.

The protocol reserves an optional field for the Zynq-7000 programmable-logic
Device DNA. The production v1 image does not require it: reading that value
needs additional FPGA logic, and the Zynq chip identity does not identify the
PCB vendor. If a later HDL build supplies Device DNA, SPF HMACs it and never
stores the raw value.

The default key path is:

```text
/etc/spf/hardware_fingerprint_hmac.key
```

It must be a regular file with mode `0600` and contain at least 32 random bytes.
The boot process creates a local key atomically if one has not been
provisioned. To keep HMAC identities stable when radios move between Rovers,
provision the same fleet key on every Rover through the deployment secret
channel. Never commit the key.

Only these on-device categories are collected:

- device-tree model;
- total RAM;
- allowlisted MTD sizes and SD presence;
- `attr_name`, `attr_val`, `compatible`, and `mode` U-Boot values;
- allowlisted firmware/kernel/U-Boot version fields.

The collector does not dump the U-Boot environment, flash, process environment,
Wi-Fi credentials, SSH material, or access tokens.

## V7 recording semantics

Every new V7 receiver group copies its matching sanitized fingerprint once as:

```text
hardware_fingerprint_schema_version = 1
hardware_fingerprint_v1 = { ... }
```

For a live capture it must contain:

```text
fingerprint_timing = "post_firmware_before_recording"
acquisition_binding = true
passive_observation = true
tx_operations_performed = false
```

The recorder refuses to create a V7 capture if the manifest is missing, stale,
belongs to another boot or firmware session, or does not match the receiver's
serial, bus, address and physical USB path.

Validate a completed capture with:

```bash
python3 -m spf.scripts.validate_direct_usb_v7_zarr \
  CAPTURE.v7.zarr \
  --expected-frames 100 \
  --expected-receivers 2
```

## Historical calibration backfill

Backfill is additive. It does not rewrite IQ or analysis arrays and it never
claims that a current observation was collected at acquisition time.

First run a dry-run:

```bash
python3 -m \
  spf.calibrations.dual_rx_gain_frequency.backfill_hardware_fingerprint \
  artifacts/dual_rx_gain_frequency \
  --ready-manifest /run/spf/direct_usb_ready.json \
  --report /tmp/calibration-fingerprint-dry-run.json
```

Review every proposed serial/path, then apply:

```bash
python3 -m \
  spf.calibrations.dual_rx_gain_frequency.backfill_hardware_fingerprint \
  artifacts/dual_rx_gain_frequency \
  --ready-manifest /run/spf/direct_usb_ready.json \
  --report /tmp/calibration-fingerprint-apply.json \
  --apply
```

Eligible historical fingerprints are labelled:

```text
fingerprint_timing = "post_run_backfill"
acquisition_binding = false
matched_by = "pluto_serial"
```

The utility hashes every physically stored Zarr array schema and materialized
chunk before and after mutation while excluding only attributes. This avoids
expanding unwritten preallocated zero frames. It also checks shape and
completed-frame counts, preserves the original acquisition firmware
attributes, refuses ambiguous serials or conflicting fingerprints, and is
idempotent. The report contains the original root/receiver attributes for each
mutated store.

Do not backfill a calibration store unless the same serial is physically
attached and present exactly once in a verified ready-manifest-v2 session.

## Rollback

Fingerprint-capable firmware is always qualified through RAM boot. If
preparation fails, leave the ready manifest absent and reboot or use the
existing rollback command to return to the unchanged QSPI image. Do not flash
QSPI as part of fingerprint deployment.
