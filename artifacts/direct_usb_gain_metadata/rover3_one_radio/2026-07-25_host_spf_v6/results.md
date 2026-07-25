# Direct-USB host, SPF collector, and v6 schema evidence

Date: 2026-07-25

Result: **PASS without hardware; real-radio acceptance remains pending**

## Implemented path

```text
Rover YAML
  -> ReceiverConfig(rx_transport="direct_usb")
  -> pyadi configuration only
  -> PlutoDirectUsbReceiver
  -> bounded queued libusb bulk-IN
  -> strict protocol-v1 parser
  -> PlutoRxBuffer
  -> ThreadedRXRawV6
  -> DroneDataCollectorRawV6
  -> LMDB-backed Zarr v6
```

The direct collector obtains IQ and metadata in one call. Legacy per-frame
`rssis()` and `gains()` calls are not present in this path. Those legacy v6
arrays are NaN in direct mode; `iq_power_dbfs` is computed from the frame IQ.

## Host ownership and transfer behavior

- Selects the Pluto by USB serial or physical port path.
- Discovers the vendor-specific interface and bulk endpoints.
- Queries and validates protocol capabilities before START.
- Claims only the custom interface; standard USB-IIO remains available for
  radio configuration.
- Queues a bounded set of exact-size asynchronous bulk-IN transfers before
  issuing START.
- Rejects short, corrupt, incompatible, missing, extra, or discontinuous
  frames.
- Cancels and drains pending transfers on failure.
- Sends STOP and releases/reattaches interface ownership during cleanup.

## Schema

The v6 receiver group preserves all v4 fields and adds typed arrays for:

```text
gain_index_start             uint8[2]
gain_index_end               uint8[2]
gain_metadata_valid          bool
gain_endpoints_equal         bool[2]
gain_metadata_flags          uint16
stream_id                    uint64
buffer_sequence              uint64
sample_sequence              uint64
gain_start_read_duration_ns  uint32
gain_end_read_duration_ns    uint32
first_gain_change_sample     int32[2]
iq_power_dbfs                float32[2]
```

Runtime USB serial, bus, port path, interface, endpoint, protocol version, and
capability flags are stored as receiver-group attributes. Firmware/gadget
source identity and the RAM-image SHA are retained in the saved YAML.

## Test result

```text
149 passed, 8 warnings in 111.38s
```

The focused combined run covered:

- 134 protocol tests, including the shared golden byte vector and arbitrary
  USB fragmentation;
- synchronous and queued/asynchronous host transfer behavior;
- exact RX1-I/Q, RX2-I/Q CS16 conversion;
- a collector hot-path guard that raises if direct mode attempts remote RSSI
  or hardware-gain reads;
- v4 and v5 dataset regression;
- typed v6 create/write/close/reopen round-trip;
- normal `mavlink_radio_collection.py` subprocess captures for v4 and v6 with
  fake radios.

Warnings are existing Zarr LMDB deprecation and W&B Sentry deprecation
warnings, not failures.

## Strict output validator

`spf/scripts/validate_direct_usb_gain_zarr.py` validates a completed 100-frame
capture and reports median frame cadence, logical IQ MiB/s, endpoint-change
count, stream count, and gain-read latency percentiles. A synthetic
524,288-sample v6 capture passed this validator at 2.0 frames/s.
