#!/usr/bin/env python3
"""Assemble E-AGC1 key=value output into a JSON artifact, stamped with host time."""
import json
import subprocess
import sys
import time

SERIALS = {
    "192.168.1.165": "104000bac4950008230026001b440a003a",
    "192.168.1.175": "1040007c4a94000211000b009186843ef2",
}
LABELS = {"192.168.1.165": "R17", "192.168.1.175": "R18"}


def run_remote(host, script_path):
    with open(script_path, "rb") as handle:
        subprocess.run(
            ["sshpass", "-p", "analog", "ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", f"root@{host}",
             "cat > /tmp/agc1_step.sh"],
            stdin=handle, check=True, capture_output=True, timeout=60,
        )
    out = subprocess.run(
        ["sshpass", "-p", "analog", "ssh", "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", f"root@{host}", "sh /tmp/agc1_step.sh"],
        check=True, capture_output=True, text=True, timeout=600,
    )
    return out.stdout


def parse(text):
    doc, multi = {}, {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line or line.startswith("Warning:"):
            continue
        key, _, value = line.partition("=")
        if key in ("claimed_line", "trial"):
            multi.setdefault(key, []).append(value)
        else:
            doc[key] = value
    doc.update(multi)
    return doc


def decode(doc):
    def reg(name):
        raw = doc.get(f"reg_{name}")
        return int(raw, 16) if raw else None

    fb, fc, fe = reg("0x0FB"), reg("0x0FC"), reg("0x0FE")
    i1, i2 = reg("0x2B0"), reg("0x2B5")
    out = {}
    if fb is not None:
        out["pin_control_armed_bits"] = fb & 0x3
        out["pin_control_disarmed_as_shipped"] = (fb & 0x3) == 0
        out["reg_0FB_other_bits_set"] = f"0x{fb & ~0x3 & 0xFF:02x}"
        out["bare_0x03_write_would_clobber"] = (fb & ~0x3 & 0xFF) != 0
    if fc is not None:
        out["manual_increment_step"] = ((fc >> 5) & 0x7) + 1
    if fe is not None:
        out["manual_decrement_step"] = ((fe >> 5) & 0x7) + 1
        out["peak_overload_wait_time"] = fe & 0x1F
    if i1 is not None:
        out["rx1_gain_index"] = i1 & 0x7F
    if i2 is not None:
        out["rx2_gain_index"] = i2 & 0x7F
    return out


def main():
    script, host, step, name, dest = sys.argv[1:6]
    text = run_remote(host, script)
    doc = {
        "experiment": "E-AGC1",
        "step": int(step),
        "name": name,
        "host_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "radio_label": LABELS.get(host, "?"),
        "radio_serial": SERIALS.get(host, "?"),
        "radio_ip": host,
        "note": "radio has no RTC; its own clock reads 1970 + uptime",
    }
    doc.update(parse(text))
    dec = decode(doc)
    if dec:
        doc["decoded"] = dec
    with open(dest, "w") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(doc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
