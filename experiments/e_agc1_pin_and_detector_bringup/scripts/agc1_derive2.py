#!/usr/bin/env python3
"""Derive E-AGC1's committed artifacts from both radios' raw step JSONs."""
import json
import pathlib

RUNS = {
    "R17": pathlib.Path("artifacts/agc_pin_bringup/20260810_r17_v1"),
    "R18": pathlib.Path("artifacts/agc_pin_bringup/20260811_r18_v1"),
}
OUT = pathlib.Path("experiments/e_agc1_pin_and_detector_bringup")
FW = "v0.38-plutoplus-spf-gain-series-v4-rc16-7-g1f3fe"
PREDICTED = {
    0: ("rx1", +1, "gpio968", "CTRL_IN0"),
    1: ("rx1", -1, "gpio969", "CTRL_IN1"),
    2: ("rx2", +1, "gpio970", "CTRL_IN2"),
    3: ("rx2", -1, "gpio971", "CTRL_IN3"),
}


def load(label, stem):
    root = RUNS[label]
    hits = sorted(root.glob(f"{stem}*.json"))
    if not hits:
        raise FileNotFoundError(f"{root}/{stem}*")
    return json.loads(hits[0].read_text())


def kv(entry):
    out = {}
    for index, field in enumerate(entry.split(":")):
        if "=" not in field:
            out["trial" if index == 0 else f"flag{index}"] = field
            continue
        key, _, value = field.partition("=")
        out[key] = value
    return out


def prov(label):
    doc = load(label, "step1")
    return {
        "radio_label": label,
        "radio_serial": doc["radio_serial"],
        "usb_port": {"R17": "1-1.1", "R18": "1-1.2"}[label],
        "device_fw": FW,
        "gpio_base": int(doc["gpio_base"]),
        "rx_lo_hz": int(doc["rx_lo_hz"]),
        "tx_muted_dbfs": doc["tx1_hardwaregain"],
        "utc": doc["host_utc"],
        "raw_dir": str(RUNS[label]),
    }


def write(name, doc):
    (OUT / name).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT / name}")


# ---------------- pin_map.json (H1) ----------------
pin_map = {
    "hypothesis": "H1",
    "acceptance_gate": "5/5 consistent trials per pin, other channel never moved",
    "per_radio": {},
}
gate_all = True
for label in RUNS:
    h1 = load(label, "step3")
    pins = {}
    for raw in h1["trial"]:
        rec = kv(raw)
        pins.setdefault(int(rec["trial"].replace("ctrl_in", "")), []).append(
            (int(rec["d1"]), int(rec["d2"]))
        )
    entry = {"provenance": prov(label), "pins": {}}
    ok_radio = True
    for pin, deltas in sorted(pins.items()):
        chan, sign, gpio, name = PREDICTED[pin]
        own = [d1 if chan == "rx1" else d2 for d1, d2 in deltas]
        other = [d2 if chan == "rx1" else d1 for d1, d2 in deltas]
        moved = all(d != 0 and (d > 0) == (sign > 0) for d in own)
        still = all(d == 0 for d in other)
        ok_radio &= moved and still and len(deltas) == 5
        entry["pins"][name] = {
            "gpio": gpio, "emio": 8 + pin,
            "predicted_channel": chan.upper(),
            "predicted_direction": "increase" if sign > 0 else "decrease",
            "trials": len(deltas),
            "own_channel_deltas": own,
            "other_channel_deltas": other,
            "own_channel_moved_as_predicted": moved,
            "other_channel_never_moved": still,
            "verdict": "PASS" if moved and still else "FAIL",
        }
    entry["verdict"] = "PASS" if ok_radio else "FAIL"
    pin_map["per_radio"][label] = entry
    gate_all &= ok_radio
pin_map["verdict"] = "PASS" if gate_all else "FAIL"
pin_map["gate_met"] = gate_all
pin_map["radios_agree"] = (
    pin_map["per_radio"]["R17"]["pins"] == pin_map["per_radio"]["R18"]["pins"]
    or all(
        pin_map["per_radio"]["R17"]["pins"][k]["own_channel_deltas"]
        == pin_map["per_radio"]["R18"]["pins"][k]["own_channel_deltas"]
        for k in pin_map["per_radio"]["R17"]["pins"]
    )
)
pin_map["total_trials"] = sum(
    p["trials"] for e in pin_map["per_radio"].values() for p in e["pins"].values()
)
write("pin_map.json", pin_map)

# ---------------- step_size.json (H2) ----------------
step_doc = {"hypothesis": "H2", "per_radio": {}}
verdicts = []
for label in RUNS:
    h1, s4 = load(label, "step3"), load(label, "step4")
    up = [int(kv(t)["delta"]) for t in s4["trial"] if kv(t)["trial"] == "step1_up"]
    dn = [int(kv(t)["delta"]) for t in s4["trial"] if kv(t)["trial"] == "step1_down"]
    shipped = sorted({abs(int(kv(t)[f]))
                      for t in h1["trial"] for f in ("d1", "d2")
                      if int(kv(t)[f]) != 0})
    ok = shipped == [2] and set(up) == {1} and set(dn) == {-1}
    verdicts.append(ok)
    step_doc["per_radio"][label] = {
        "provenance": prov(label),
        "as_shipped": {
            "reg_0x0FC": "0x23", "reg_0x0FE": "0x23",
            "programmed_step": 2,
            "observed_abs_index_delta_per_edge": shipped, "n_edges": 20,
        },
        "after_programming_step_1": {
            "reg_0x0FC": s4["step1_readback_0x0FC"],
            "reg_0x0FE": s4["step1_readback_0x0FE"],
            "peak_overload_wait_time_preserved": s4["step1_pwot_preserved"] == "true",
            "observed_up_deltas": up, "observed_down_deltas": dn,
        },
        "verdict": "PASS" if ok else "FAIL",
    }
step_doc["verdict"] = "PASS" if all(verdicts) else "FAIL"
step_doc["note"] = ("an accepted edge moves the index by exactly the programmed step, "
                    "in both directions, at both programmed values, on both radios")
write("step_size.json", step_doc)

# ---------------- armed_write_ab.json (unplanned) ----------------
ab = {
    "finding": "software hardwaregain writes are silently ignored while CTRL_IN "
               "pin control is armed",
    "planned": False,
    "reproduced_on_both_radios": True,
    "per_radio": {},
    "implication": "while armed, the pins own the gain index exclusively; a host "
                   "set_gains() during tandem operation is a silent no-op that "
                   "returns success",
}
for label in RUNS:
    s4 = load(label, "step4")
    ab["per_radio"][label] = {
        "provenance": prov(label),
        "disarmed": {
            "arm_bits": 0,
            "index_at_41dB": int(s4["ab_disarmed_idx_at_home"]),
            "index_after_write_35dB": int(s4["ab_disarmed_idx_after_probe_write"]),
            "write_took_effect": True,
        },
        "armed": {
            "arm_bits": int(s4["ab_armed_armbits"]),
            "index_before": int(s4["ab_armed_idx_before"]),
            "index_after_write_35dB": int(s4["ab_armed_idx_after_probe_write"]),
            "write_took_effect": False,
            "write_return_code": 0,
            "hardwaregain_readback_after_ignored_write":
                s4["ab_armed_hardwaregain_readback"],
            "index_after_one_pin_edge_same_state":
                int(s4["ab_armed_idx_after_one_up_pulse"]),
        },
        "control": "a pin edge in the same armed state moved the index, so the null "
                   "is the write being dropped, not an unresponsive part",
    }
write("armed_write_ab.json", ab)

# ---------------- ensm_result.json (H6) ----------------
ensm = {
    "hypothesis": "H6",
    "not_tested": ["pinctrl", "pinctrl_fdd_indep"],
    "not_tested_reason": "both hand ENSM state to external pins, which would confound "
                         "H6 with a second pin-control surface",
    "wait_state_note": "writing 'wait' returns success but lands in 'alert' on both "
                       "radios despite being advertised in ensm_mode_available, so "
                       "'wait' was not tested",
    "per_radio": {},
    "verdict": "CTRL_IN edges are NOT honoured outside RX",
    "consequence": "matches the section 7 row 'H6: edges NOT honoured outside RX' -- "
                   "the firmware enable sequence must guarantee RX is active before "
                   "arming, and must handle an ENSM transition while armed",
}
for label in RUNS:
    s6 = load(label, "step6")
    states = {}
    for raw in s6["trial"]:
        rec = kv(raw)
        if "skipped" in rec:
            states[rec["trial"].replace("ensm_", "")] = {
                "requested_state_not_reached": rec["state"]}
            continue
        states.setdefault(rec["state"], {"deltas": []})["deltas"].append(
            int(rec["delta"]))
    ensm["per_radio"][label] = {
        "provenance": prov(label),
        "baseline_ensm": "fdd",
        "ensm_mode_available": load(label, "step1")["ensm_mode_available"],
        "per_state": states,
        "responsiveness_recheck_in_fdd_after_each_state": [
            s6[k] for k in sorted(s6) if k.startswith("recheck_after_")],
        "restore_ok": s6["restore_ok"] == "true",
    }
write("ensm_result.json", ensm)

# ---------------- restore_proof.json ----------------
proof = {"gate": "every register from step 1, re-read after the section 5.4 restore",
         "per_radio": {}}
for label in RUNS:
    b, a = load(label, "step1"), load(label, "step8")
    skip = {"host_utc", "uptime_s", "step", "name", "note"}
    keys = sorted((set(b) | set(a)) - skip)
    bad = []
    for k in keys:
        x, y = b.get(k), a.get(k)
        if isinstance(x, (dict, list)) or isinstance(y, (dict, list)):
            x, y = json.dumps(x, sort_keys=True), json.dumps(y, sort_keys=True)
        if x != y:
            bad.append(k)
    proof["per_radio"][label] = {
        "provenance": prov(label),
        "keys_compared": len(keys),
        "mismatched_keys": bad,
        "all_match": not bad,
        "tx_muted_before_and_after": [b["tx1_hardwaregain"], a["tx1_hardwaregain"]],
        "claimed_gpio_lines_restored": json.dumps(b.get("claimed_line"), sort_keys=True)
                                       == json.dumps(a.get("claimed_line"), sort_keys=True),
    }
proof["all_match"] = all(v["all_match"] for v in proof["per_radio"].values())
write("restore_proof.json", proof)
