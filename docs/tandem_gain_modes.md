# Selecting tandem AGC

Use the existing gain-mode setting:

```yaml
rx-transport: iio
rx-gain-mode: tandem
buffer-size: 524288
```

Select `manual`, `slow_attack`, or `fast_attack` to disable tandem. The CLI's
`--rx-mode` accepts the same four values. Existing defaults are unchanged.
Tandem requires both RX channels; mixed tandem/legacy per-channel settings and
the separate `direct_usb` protocol are rejected. USB and LAN libiio connections
use the same IIO session implementation.

For an already configured `PPlus` receiver:

```python
radio.set_gain_mode("tandem")
frame = radio.rx_with_metadata()
assert frame.tandem_ownership_epoch is not None

radio.set_gain_mode("slow_attack")
radio.set_gain_mode("manual", gains=[20, 23])
```

Mode changes restart capture under the receiver's RX lock. Validation occurs
before closing the previous session. The provider releases its ownership before
legacy gain writes; the new configuration is published only after a valid frame
confirms the requested ownership and policy. Failures restore the previous
configuration and metadata capture where possible. If restoration fails, RX is
stopped explicitly. A metadata capture failure also stops RX; it cannot fall back
silently to ordinary IQ. Reapplying a healthy mode verifies the session without
restarting it.

Manual gains are validated in dB against the active band's
`hardwaregain_available` range. Tandem uses a common initial gain of 20 dB and
0–62 dB limits, with the existing session policy: 1024-sample power measurement,
three-period low-power dwell, and 16-period cooldown. Policy timing stays fixed
when frame size changes. Both retained refill windows must fit the 64-event
capacity; 262144 and 524288 samples are hardware-qualified. Oversized tandem
frames fail before arming.

## Required radio and host components

- Base firmware: [v0.49 Pluto+ release](https://github.com/misko/plutosdr-fw/releases/tag/v0.49-plutoplus-spf-iq-direct-async-v4).
- Host libiio: `5cb2389719d46d12463daa0371d1fda19eb25fa7`, installed using
  `install_spf_libiio.sh` after the Python dependencies. This is the source pin
  in `packaging/libiio/versions.sh`.
- Radio iiOD: the legacy-AGC metadata extension in the libiio
  [legacy metadata extension PR](https://github.com/misko/libiio/pull/7),
  based on that same release source.

**Stock v0.49 alone does not provide metadata while tandem is off.** The extension
advertises `iio,buffer-metadata-legacy-agc=1` and accepts a small `SPFL` request
that leaves AD9361 gain control unchanged. It retains gain observations, RSSI,
temperature, FPGA timestamps, and exact gap accounting without acquiring tandem
ownership. It does not claim FPGA gain events for legacy AGC.

The 104-byte tandem request and the firmware's kernel ownership/watchdog logic
remain unchanged. No AD9361 register controller is added to SPF. `IioMetadataRx`
now defaults to the explicit legacy request; callers wanting tandem must pass a
`TandemSessionRequestV1`.

The qualification deployment installs the provider through the radio's JFFS2
autorun hook. The hook checks both the provider hash and the original v0.49 iiOD
hash; it leaves an independently upgraded release executable untouched. The
base firmware FIT remains the official release image. The original iiOD is
retained at `/mnt/jffs2/spf-gain-mode/iiod.release` for rollback.

## Recording and tests

Returned frames include requested gain modes, tandem ownership epoch, and exact
missing-sample count. V7 recordings add an optional `gain_control_schema_version`
extension with those values. Epoch zero denotes no tandem owner. Requested mode
codes are manual=0, slow=1, fast=2, tandem=3, unknown=255, also recorded in the
store attributes. Existing V7 recordings remain readable.

Run ordinary tests without touching a radio:

```bash
python -m pytest tests/test_gain_control.py tests/test_iio_metadata.py \
  tests/test_tandem_agc.py tests/test_libiio_dependency_contract.py
```

The hardware suite requires both explicit selectors and checks the serial before
opening the receiver for control. With the matched host library/binding active:

```bash
SPF_TANDEM_TEST_URI=ip:RADIO_ADDRESS \
SPF_TANDEM_TEST_SERIAL=EXACT_SERIAL \
SPF_TANDEM_TEST_REPORT=/private/hardware-frames.jsonl \
python -m pytest --noconftest tests/test_tandem_gain_modes_hardware.py
```

The suite exercises all public modes across three gain-table bands and two frame
sizes, paired tandem ownership/events, unequal manual gain restoration, enable
failure, watchdog expiry, process termination, and a dropped proxy connection
followed by reconnection. It is receive-only. Controlled RF amplitude response
and phase benefit require a separately specified signal source/harness.

[Qualification results and deployment attestation](qualification/tandem-gain-mode-2026-09-12/README.md).
