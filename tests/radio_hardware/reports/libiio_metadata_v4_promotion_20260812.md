# libiio frame-metadata v4 promotion evidence

Date: 2026-08-12

## Scope and immutable candidate

The qualified radio baseline is libiio/iiOD 0.25. The only supported patched
host clients are libiio 0.25 and 0.26.

| Component | Revision |
|---|---|
| Firmware | `0ad6c9410afd1a5a5c641e5f1a93b429d406117e` |
| Buildroot | `684ecbcbe44bc82043caae7091f12c02cbd02d8b` |
| Radio/host libiio 0.25 | `c26258bfa33098c2b215e19cf85d448e89499b1a` |
| Host libiio 0.26 | `d5695c3eaa9cec99cc6f7b2c91565555044b907a` |
| Shared sampler/serializer | `ab270f9e3128187372f27de887be65353f9e195d` |
| SPF host | `e7d9563` |

GitHub Actions run `31573289191` built, independently verified, and attested
the candidate. The exact RAM-booted DFU SHA-256 was:

```text
5acd2073c2a14b39e047c6328040562d49c8334237994bb926ea7e518139cea1
```

Both radios were identified by serial before every RAM boot:

- `104000bac4950008230026001b440a003a` on physical USB port `1-1.1`;
- `1040007c4a94000211000b009186843ef2` on physical USB port `1-1.2`.

Nothing was flashed persistently. Both radios were restored to their existing
`gain-series-v4-rc17-source/buildroot-final-v2` images after qualification.

## Critical short-frame race gate

Each radio ran 20 fresh USB client processes. Every process used one kernel
buffer and completed ten 1024-sample metadata refills. All 400 refills returned
8192 IQ bytes, valid V3 metadata, at least one gain observation, and a strictly
increasing capture index. There were no `ENODATA` failures.

Ordinary and metadata RX passed over LAN and USB with each patched host version.
The real SPF `IioMetadataRx` adapter also passed every radio/transport/version
combination with dual-channel IQ, capture index and time, gain sequence, and
RSSI start/end.

## Slow-host bounded-backlog gates

Radio A was sampled once, left without another refill request for the indicated
duration, and then drained for `N+1` frames. The table lists the warm-up index
separately from the post-stall indices.

| Stall | N | Warm-up | Post-stall capture indices |
|---:|---:|---:|---|
| 30 s | 1 | `0` | `900124, 900236` |
| 30 s | 4 | `0` | `1, 2, 3, 900130, 900271` |
| 60 s | 1 | `0` | `1800124, 1800233` |
| 60 s | 4 | `0` | `1, 2, 78, 1800198, 1800334` |
| 300 s | 1 | `0` | `9000122, 9000257` |
| 300 s | 4 | `0` | `1, 44, 119, 9000228, 9000362` |

For `N=1`, no old frame remained after the delivered warm-up frame. For `N=4`,
only three pre-stall frames remained. The following frame jumped by the elapsed
radio time. Thus the complete stale-data bound, including the warm-up frame,
was exactly `N`; there was no userspace IQ history or growing TCP backlog.

During the five-minute `N=1` stall the sampler remained at three CPU ticks,
iiOD RSS remained 2708--2712 KiB, and both TCP queues stayed empty. During the
five-minute `N=4` stall the sampler remained at four ticks and iiOD RSS remained
2712 KiB.

## Deliberately stalled TCP response

A raw `OPENM`/`READBUFM` client requested a 2 MiB frame with a requested 4096
byte receive buffer (8192 bytes after Linux doubling), then did not read for 30
seconds. The observed host receive queue was 6542 bytes and the radio send queue
was 82536 bytes. iiOD RSS stayed at 2844 KiB. The radio could not finish the
current response and did not issue another capture request.

The client then closed without reading or sending `CLOSE`. The sampler thread
was removed and a fresh metadata client immediately completed ten frames.
Twenty additional clients closed immediately after sending `READBUFM`; all
cleaned up, after which a 0.26 client completed 20 metadata frames and iiOD was
back to its three baseline threads at 2844 KiB RSS.

## Throughput

Each timed row copied 64 dual-channel frames of 262144 samples after two warm-up
frames. IQ payload rate excludes the small metadata record but includes the same
host-side IQ copy in both modes.

| Host | Transport | Ordinary MB/s | Metadata MB/s | Retained throughput |
|---|---|---:|---:|---:|
| 0.25 | LAN | 47.365 | 42.971 | 90.7% |
| 0.25 | USB | 22.515 | 21.832 | 97.0% |
| 0.26 | LAN | 47.294 | 42.847 | 90.6% |
| 0.26 | USB | 22.486 | 21.860 | 97.2% |

The host versions have indistinguishable behavior. Metadata costs about 9% on
LAN and 3% on USB for these large frames; it does not change IQ layout or byte
count.

## Restart, rollback, and compatibility

Radio A completed three independently identified RAM boots of the attested
candidate. Candidate boots advertised `iio,buffer-metadata=1`, reported
`libiio-metadata-v4-source/buildroot-final-v5`, and completed metadata capture.
Each reboot into persistent firmware restored `buildroot-final-v2`.

On the final stock state, patched 0.26 received `ENOSYS` for metadata creation
and then completed an 8192-byte ordinary refill on the same synchronized
connection. Earlier qualification demonstrated the same behavior with both
patched versions and both radios.

## Commands and remaining promotion boundary

The reusable probes are in
`tests/radio_hardware/iio_metadata_qualification.py`. They are selected
individually; the full pytest suite was not run on the Pi.

The RAM-only data-path gates are complete. Persistent deployment and a physical
power-cycle qualification remain a separate, explicit promotion decision.
Section 11 of `~/libiio_extension_and_testing_plan.md` records the decision not
to add a second diagnostic schema: per-frame diagnostics, typed errors, and
external qualification measurements cover the feature without expanding its
production API.
