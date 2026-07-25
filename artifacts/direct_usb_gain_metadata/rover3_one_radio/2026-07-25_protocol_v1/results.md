# Direct-USB gain protocol v1 host/device test

Date: 2026-07-25

## Result

**PASS**

The Python parser and C gadget header agree on the canonical 80-byte
little-endian golden vector:

```text
53474d3101005000
0700000017040000
f0debc9a78563412
0700000000000000
0004000000000000
0800000040000000
0f0000000100022a
2b292b00b0040000
14050000ffffffff
ffffffff796afe5d
```

## Commands

```sh
source /home/pi/spf-virtualenv/bin/activate
pytest -q tests/test_direct_usb_protocol.py

cc -std=c11 -Wall -Wextra -Werror -pedantic \
  /home/pi/spf-direct-usb/pluto-sdr-usb-gadget/test_spf_gain_metadata.c \
  -o /tmp/test_spf_gain_metadata
/tmp/test_spf_gain_metadata
```

## Results

- Python: 121 passed
- C: exit status 0 with warnings treated as errors
- Header size: 80 bytes
- CRC: CRC-32/ISO-HDLC over the header with the CRC field zeroed
- Python and C golden bytes: identical

The Python tests cover single-byte and arbitrary fragmentation, concatenated
frames, short and extra payload, isolated bad magic/version/header size,
nonzero reserved data, bad CRC, unsupported scan mask/sample format/channel
count, integer overflow, invalid gain sentinels, unknown feature/flag bits,
sequence gaps and resets, stream changes, invalid FPGA fields, dummy metadata,
and fail-closed endpoint comparison.
