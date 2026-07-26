# Direct USB gain metadata discovery

Date: 2026-07-24

## Source state

Exact revisions are recorded in `source_shas.txt`.

The SPF worktree was already dirty before this project began. Pre-existing
untracked files were preserved. Project-created files at this point are
`plan.md` and this evidence directory.

Reference repositories were cloned outside the SPF worktree under:

```text
/home/pi/spf-direct-usb/
```

## Development host

```text
Linux devpi 6.6.20+rpt-rpi-v8 aarch64
Python 3.11.2
libiio 0.24
```

`cmake` and `dfu-util` were not installed at discovery time.

## Attached radio

```text
USB VID:PID: 0456:b673
USB URI: usb:1.3.5
IP URI: ip:pluto.local / 192.168.2.1
serial: 104000f6ad020002fdff3a00bba2f096a1
IIO model: Analog Devices PlutoSDR Rev.C (Z7010-AD9361)
device-tree model: Analog Devices PlutoSDR Rev.C (Z7010/AD9363)
firmware: v0.37-dirty
kernel: 5.10.0-98725-g3eae70065be9-dirty
mode: 2r2t
compatible override: ad9361
```

The software identity does not reveal the physical PlutoPlus PCB revision.
That remains a Step 0 blocker and requires visual inspection of the PCB or
authoritative inventory data.

The existing composite USB device exposes:

- RNDIS control/data;
- mass storage;
- ACM control/data;
- standard IIO FunctionFS.

It does not expose a separate vendor-specific SDR streaming interface.

## Current radio state during benchmark

```text
sample rate: 30.72 MS/s
RF bandwidth: 18 MHz
LO: 2.4 GHz
RX1 mode: slow_attack
RX2 mode: slow_attack
reported hardware gains: 69 dB, 67 dB
split gain table mode: disabled
external-LNA all-index gain table mode: disabled
MGC split-table control-input mode: disabled
```

## Persistent pyadi latency baseline

The repository virtual environment and one persistent `adi.ad9361` context were
used. Each measurement contains 100 timed calls after three warm-up calls.
The client buffer contained 32,768 samples per enabled channel. RF settings
were read but not changed.

| Operation | median | p95 | p99 | mean | max |
|---|---:|---:|---:|---:|---:|
| `sdr.rx()` | 18.214 ms | 18.508 ms | 18.764 ms | 18.230 ms | 18.935 ms |
| both RSSI attributes | 1.002 ms | 1.103 ms | 1.130 ms | 1.018 ms | 1.159 ms |
| both `hardwaregain` attributes | 1.878 ms | 2.115 ms | 2.197 ms | 1.857 ms | 2.299 ms |

The two metadata calls add approximately 2.88 ms, or 15.8% of the median RX
call time in this test.

This does not reproduce the multi-second latency reported for the Rover setup.
The earlier standalone `iio_attr` timing included process startup and context
creation and must not be used as the in-process SPF result. The Rover hardware,
USB topology, configured buffer size, and exact collector process still require
their own benchmark.

## Step 0 gate status

Status: **BLOCKED**

Satisfied:

- exact source and submodule revisions recorded;
- device serial and current USB path recorded;
- firmware and software-visible hardware identity recorded;
- 100-call in-process latency distributions recorded;
- existing USB interface inventory recorded.

Outstanding:

- establish the physical PlutoPlus PCB revision;
- repeat the latency baseline on every Rover radio/topology in scope;
- capture a complete pre-change U-Boot environment during the firmware-safety
  step.
