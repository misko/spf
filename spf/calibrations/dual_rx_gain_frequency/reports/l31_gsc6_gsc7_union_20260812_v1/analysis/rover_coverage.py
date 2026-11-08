"""Measure, per carrier, what fraction of the 2026 rover corpus each coefficient
set can actually predict -- fail-closed, never extrapolated.

READ-ONLY. Every store is opened with ``zarr_open_from_lmdb_store(path, mode="r")``.
Nothing under /mnt is written.

Deduplication: the merge names are ``<RX capture>.<TX capture>.zarr`` and some RX
captures were merged against more than one TX capture, so statistics are
deduplicated on the RX-capture prefix, exactly as
``rover_applicability_ladder_20260812_v1`` section 3.1 does. Without that dedup
every quantity is silently re-weighted by TX-partner multiplicity.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SPF_REPO",
                           Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(REPO))

from spf.calibrations.gain_state_phase_model_v1.model import (  # noqa: E402
    GainStatePhaseModel,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store  # noqa: E402

MERGED = Path("/mnt/qnap01/mouse9911/rovers_2026/merged")
CARRIERS = (5_766_000_000, 5_840_000_000)


def scan() -> dict:
    """One pass over the corpus; returns per-carrier gain-pair histograms.

    UNIT OF ACCOUNT. The model predicts ``D`` for exactly one ``(g1, g2)`` pair,
    and there is exactly one such pair per (receiver stream, frame): each stream
    carries ``gains`` of shape (T, 2), the RX1/RX2 arm pair. So the denominator
    used throughout is the **arm-pair observation** -- one per receiver stream
    per frame. Capture-frames are recorded separately so the numbers can be
    reconciled against a report that counts them differently.
    """
    seen_rx: set[str] = set()
    pairs: dict[int, Counter] = defaultdict(Counter)
    gee: dict[int, dict] = defaultdict(lambda: {"unstable": 0, "n": 0})
    exact_equal: dict[int, int] = Counter()
    capture_frames: dict[int, int] = Counter()
    streams: dict[int, set] = defaultdict(set)
    captures: dict[int, set] = defaultdict(set)
    per_capture = []
    skipped = []

    for zp in sorted(MERGED.glob("*.zarr")):
        rx_prefix = zp.name.split(".")[0]
        if rx_prefix in seen_rx:
            continue
        seen_rx.add(rx_prefix)
        try:
            z = zarr_open_from_lmdb_store(str(zp), mode="r")
            rxs = list(z["receivers"].keys())
        except Exception as exc:                      # noqa: BLE001
            skipped.append({"zarr": zp.name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        cap = {"rx_prefix": rx_prefix, "receivers": rxs, "carriers": {}}
        for r in rxs:
            try:
                rr = z[f"receivers/{r}"]
                lo = np.asarray(rr["rx_lo"][:], dtype=float)
                gains = np.asarray(rr["gains"][:], dtype=float)
                try:
                    ge = np.asarray(rr["gain_endpoints_equal"][:])
                except Exception:                      # noqa: BLE001
                    ge = None
            except Exception as exc:                  # noqa: BLE001
                skipped.append({"zarr": zp.name, "receiver": r,
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            lo_i = np.round(lo).astype(np.int64)
            for carrier in CARRIERS:
                m = lo_i == carrier
                if not m.any():
                    continue
                n = int(m.sum())
                cap["carriers"].setdefault(str(carrier), 0)
                cap["carriers"][str(carrier)] += n
                capture_frames[carrier] += n
                streams[carrier].add(f"{rx_prefix}|{r}")
                captures[carrier].add(rx_prefix)
                gg = gains[m]
                ok = np.isfinite(gg).all(axis=1)
                gi = np.round(gg[ok]).astype(np.int64)
                for a, b in gi:
                    pairs[carrier][(int(a), int(b))] += 1
                exact_equal[carrier] += int((gi[:, 0] == gi[:, 1]).sum())
                if ge is not None:
                    gearr = np.asarray(ge)[m]
                    gee[carrier]["unstable"] += int(
                        (gearr == 0).any(axis=1).sum() if gearr.ndim == 2
                        else (gearr == 0).sum()
                    )
                    gee[carrier]["n"] += n
        per_capture.append(cap)
    return {
        "pairs": pairs, "gee": gee, "exact_equal": exact_equal,
        "capture_frames": capture_frames,
        "n_streams": {c: len(s) for c, s in streams.items()},
        "n_captures": {c: len(s) for c, s in captures.items()},
        "per_capture": per_capture, "skipped": skipped,
        "n_rx_captures": len(seen_rx),
    }


def support_table(pairs: Counter, model: GainStatePhaseModel, lo_hz: float) -> dict:
    total = sum(pairs.values())
    supported = 0
    guarded = 0
    reasons: Counter = Counter()
    for (g1, g2), n in pairs.items():
        p = model.predict(lo_hz, g1, g2)
        if p.supported:
            supported += n
            if p.guarded:
                guarded += n
        else:
            reasons[p.reason] += n
    return {
        "frames": total,
        "supported": supported,
        "supported_fraction": supported / total if total else 0.0,
        "guarded_by_rule5": guarded,
        "guarded_fraction": guarded / total if total else 0.0,
        "correcting": supported - guarded,
        "correcting_fraction": (supported - guarded) / total if total else 0.0,
        "top_refusal_reasons": [
            {"reason": r, "frames": c, "fraction": c / total}
            for r, c in reasons.most_common(5)
        ],
    }


def main(out_path: str, extra_models: list[str]):
    sc = scan()
    out: dict = {
        "corpus": str(MERGED),
        "read_only": True,
        "n_zarrs_globbed": len(list(MERGED.glob("*.zarr"))),
        "n_distinct_rx_captures": sc["n_rx_captures"],
        "skipped": sc["skipped"],
        "carriers": {},
    }

    names = ["l26_pooled_v1", "l26_stage_a_v1", "l30_pooled_v1",
             "l31_pooled_v1"] + extra_models
    models = {}
    for nm in names:
        if Path(nm).exists():
            models[Path(nm).stem] = GainStatePhaseModel.load(nm)
        else:
            models[nm] = GainStatePhaseModel.load_named(nm)

    grand = sum(sum(sc["pairs"][c].values()) for c in CARRIERS)
    for carrier in CARRIERS:
        pr = sc["pairs"][carrier]
        arms_total = sum(pr.values())
        # frames = arm-observations / 2 receivers... keep the arm-observation
        # denominator explicit rather than dividing and hoping
        blob = {
            "arm_pair_observations": arms_total,
            "receiver_stream_frames": int(sc["capture_frames"][carrier]),
            "n_receiver_streams": int(sc["n_streams"].get(carrier, 0)),
            "n_rx_captures": int(sc["n_captures"].get(carrier, 0)),
            "share_of_corpus": arms_total / grand if grand else 0.0,
            "distinct_gain_pairs": len(pr),
            "exact_equal_gain_fraction":
                sc["exact_equal"][carrier] / arms_total if arms_total else 0.0,
            "gain_endpoints_unstable_fraction":
                (sc["gee"][carrier]["unstable"] / sc["gee"][carrier]["n"]
                 if sc["gee"][carrier]["n"] else None),
            "support": {},
        }
        for nm, m in models.items():
            blob["support"][nm] = support_table(pr, m, float(carrier))
            s = blob["support"][nm]
            print(f"{carrier/1e6:.0f} MHz {nm:34s} "
                  f"supported {s['supported_fraction']*100:6.2f}%  "
                  f"correcting {s['correcting_fraction']*100:6.2f}%")
        out["carriers"][str(carrier)] = blob

    Path(out_path).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
