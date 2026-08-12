"""Independent re-measurement for the refreshed model-ladder report.

Read-only. Three parts:
  A. rover corpus applicability   (deduplicated by RX capture)
  B. E-GSC7 paired tests from the committed analysis.json
  C. the segmentation fold non-commutation demonstration
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from spf.calibrations.gain_state_phase_model_v1 import GainStatePhaseModel
from spf.calibrations.gain_state_phase_model_v1.gain_tables import (
    GainTables,
    band_for_lo,
)
from spf.scripts.zarr_utils import zarr_open_from_lmdb_store

OUTDIR = Path(sys.argv[1])
OUTDIR.mkdir(parents=True, exist_ok=True)
gt = GainTables()

MERGED_DIR = Path("/mnt/qnap01/mouse9911/rovers_2026/merged")

out = {}

# =========================================================== A. rover corpus
merged = sorted(MERGED_DIR.glob("*.zarr"))
# merged names are "<RX capture>.<TX capture>.zarr"; the same RX capture can be
# merged against more than one TX capture, so dedup on the RX prefix or every
# statistic is silently re-weighted by how many TX partners a capture happened
# to have.
by_rx = {}
for p in merged:
    name = p.name
    i = name.index(".rover_")
    by_rx.setdefault(name[:i], []).append(p)

streams = []  # one per (rx capture, receiver)
for rxcap, paths in sorted(by_rx.items()):
    p = sorted(paths)[0]  # canonical: first merged file for this RX capture
    z = zarr_open_from_lmdb_store(str(p), mode="r")
    for r in sorted(z["receivers"].keys()):
        g = z["receivers"][r]
        streams.append(
            dict(
                rx_capture=rxcap,
                receiver=r,
                merged_file=p.name,
                n_merged_partners=len(paths),
                gains=np.asarray(g["gains"][:]),
                lo=np.asarray(g["rx_lo"][:]),
                gee=np.asarray(g["gain_endpoints_equal"][:]),
                gmv=np.asarray(g["gain_metadata_valid"][:]),
                gstart=np.asarray(g["gain_db_start"][:]),
                gend=np.asarray(g["gain_db_end"][:]),
            )
        )

corpus = dict(
    merged_zarr_files=len(merged),
    distinct_rx_captures=len(by_rx),
    rx_captures_merged_more_than_once=sum(
        1 for v in by_rx.values() if len(v) > 1
    ),
    receiver_streams_after_dedup=len(streams),
)

G = np.concatenate([s["gains"] for s in streams])
LO = np.concatenate([s["lo"] for s in streams])
GEE = np.concatenate([s["gee"] for s in streams])
GMV = np.concatenate([s["gmv"] for s in streams])
GST = np.concatenate([s["gstart"] for s in streams])
GEN = np.concatenate([s["gend"] for s in streams])
CAP = np.concatenate(
    [np.full(s["gains"].shape[0], i) for i, s in enumerate(streams)]
)

corpus["total_frames_after_dedup"] = int(G.shape[0])
corpus["lo_histogram_hz"] = {
    str(int(k)): int(v) for k, v in sorted(Counter(LO.tolist()).items())
}
corpus["gain_metadata_valid_fraction"] = float(GMV.mean())

EGSC6 = [-1, 8, 20, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 40, 41, 45, 49, 50,
         51, 52, 62]
EGSC7 = [26, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

COEF_SETS = ("l26_pooled_v1", "l26_stage_a_v1", "l30_pooled_v1", "l31_pooled_v1")
models = {name: GainStatePhaseModel.load_named(name) for name in COEF_SETS}
coefdir = Path("spf/calibrations/gain_state_phase_model_v1/coefficients")
fitted_levels = {}
for name in COEF_SETS:
    c = json.loads((coefdir / f"{name}.json").read_text())
    fitted_levels[name] = dict(
        lna=sorted(int(k) for k in c["h_deg"]["lna"]),
        mixer=sorted(int(k) for k in c["h_deg"]["mixer"]),
        tia=sorted(int(k) for k in c["h_deg"]["tia"]),
        lpf=sorted(int(k) for k in c["h_deg"]["lpf"]),
        n_columns=c["provenance"]["n_columns"],
        rank=c["provenance"]["rank"],
    )
corpus["fitted_levels"] = fitted_levels


def levels_of(gain_list, band):
    d = defaultdict(set)
    for gi in gain_list:
        st = gt.state(band, gi)
        if st is None:
            continue
        d["lna"].add(st.lna)
        d["mixer"].add(st.mixer)
        d["tia"].add(st.tia)
        d["lpf"].add(st.lpf)
    return {k: sorted(v) for k, v in d.items()}


per_lo = {}
for lo in sorted(set(LO.tolist())):
    m = LO == lo
    g = G[m]
    n = int(m.sum())
    band = band_for_lo(lo)
    dec = {}
    for gi in sorted({int(round(x)) for x in np.unique(g)}):
        st = gt.state(band, gi)
        dec[gi] = None if st is None else (st.lna, st.mixer, st.tia, st.lpf)
    L = np.array([[dec[int(round(x))][0] for x in g[:, 0]],
                  [dec[int(round(x))][0] for x in g[:, 1]]])
    M = np.array([[dec[int(round(x))][1] for x in g[:, 0]],
                  [dec[int(round(x))][1] for x in g[:, 1]]])
    T = np.array([[dec[int(round(x))][2] for x in g[:, 0]],
                  [dec[int(round(x))][2] for x in g[:, 1]]])
    P = np.array([[dec[int(round(x))][3] for x in g[:, 0]],
                  [dec[int(round(x))][3] for x in g[:, 1]]])
    rf_equal = (L[0] == L[1]) & (M[0] == M[1]) & (T[0] == T[1])
    diff = np.abs(g[:, 0] - g[:, 1])
    gee = GEE[m]
    mv = np.abs(GEN[m] - GST[m])

    def cov(levels, require_lpf):
        ok = (np.isin(L[0], levels["lna"]) & np.isin(L[1], levels["lna"])
              & np.isin(M[0], levels["mixer"]) & np.isin(M[1], levels["mixer"])
              & np.isin(T[0], levels["tia"]) & np.isin(T[1], levels["tia"]))
        if require_lpf:
            ok = ok & np.isin(P[0], levels["lpf"]) & np.isin(P[1], levels["lpf"])
        return dict(all_frames=float(ok.mean()),
                    needing_correction=float(ok[~rf_equal].mean()))

    entry = dict(
        band=band,
        n_frames=n,
        n_rx_captures=int(len({streams[i]["rx_capture"] for i in np.unique(CAP[m])})),
        n_receiver_streams=int(len(np.unique(CAP[m]))),
        gain_unequal_fraction=float((diff > 0).mean()),
        median_abs_gain_diff_db_unequal_only=float(np.median(diff[diff > 0]))
        if (diff > 0).any() else None,
        median_abs_gain_diff_db_all=float(np.median(diff)),
        rf_words_equal_fraction=float(rf_equal.mean()),
        needs_correction_fraction=float((~rf_equal).mean()),
        n_frames_needing_correction=int((~rf_equal).sum()),
        mixer_levels_invoked=sorted(set(M[0].tolist()) | set(M[1].tolist())),
        lna_levels_invoked=sorted(set(L[0].tolist()) | set(L[1].tolist())),
        tia_levels_invoked=sorted(set(T[0].tolist()) | set(T[1].tolist())),
        lpf_levels_invoked=sorted(set(P[0].tolist()) | set(P[1].tolist())),
        mixer_pair_top=[[list(k), int(v), float(v) / n]
                        for k, v in Counter(zip(M[0].tolist(), M[1].tolist())).most_common(8)],
        mixer_pair_top_needing_correction=[
            [list(k), int(v), float(v) / int((~rf_equal).sum())]
            for k, v in Counter(zip(M[0][~rf_equal].tolist(),
                                    M[1][~rf_equal].tolist())).most_common(8)],
        gain_endpoints_equal_fraction_both_arms=float(gee.all(axis=1).mean()),
        gain_endpoints_unequal_fraction_both_arms=float(1 - gee.all(axis=1).mean()),
        gain_endpoints_equal_fraction_per_arm=[float(gee[:, 0].mean()),
                                               float(gee[:, 1].mean())],
        gain_endpoints_unequal_fraction_any_arm_elementwise=float((~gee).mean()),
        within_buffer_move_db=dict(
            mean=float(mv.mean()), median=float(np.median(mv)),
            p95=float(np.percentile(mv, 95)), max=float(mv.max()),
            frames_with_any_move=float((mv > 0).any(axis=1).mean()),
        ),
        gain_endpoints_unequal_by_capture=sorted(
            float(1 - GEE[m][CAP[m] == c].all(axis=1).mean())
            for c in np.unique(CAP[m])
        ),
    )
    # coefficient-set support
    sup = {}
    pair_counts = Counter((int(round(a)), int(round(b))) for a, b in g.tolist())
    rng = np.random.default_rng(20260812)
    idx = rng.choice(n, size=min(400, n), replace=False)
    for name, model in models.items():
        w_ok = 0
        for (a, b), c in pair_counts.items():
            if model.predict(lo, a, b).supported:
                w_ok += c
        s_ok = sum(
            1 for i in idx
            if model.predict(lo, int(round(g[i, 0])), int(round(g[i, 1]))).supported
        )
        sup[name] = dict(
            frame_weighted_supported=int(w_ok),
            frame_weighted_fraction=float(w_ok) / n,
            sampled_400_supported=int(s_ok),
            sampled_400_fraction=float(s_ok) / len(idx),
        )
    entry["coefficient_set_support"] = sup
    # what a refit on E-GSC6 / E-GSC6+7 gain sets would cover
    entry["refit_coverage"] = {
        "e_gsc6": dict(levels=levels_of(EGSC6, band),
                       rf_only=cov(levels_of(EGSC6, band), False),
                       with_lpf=cov(levels_of(EGSC6, band), True)),
        "e_gsc6_plus_e_gsc7": dict(
            levels=levels_of(EGSC6 + EGSC7, band),
            rf_only=cov(levels_of(EGSC6 + EGSC7, band), False),
            with_lpf=cov(levels_of(EGSC6 + EGSC7, band), True)),
        "l26_pooled_v1_fitted": dict(
            rf_only=cov(fitted_levels["l26_pooled_v1"], False),
            with_lpf=cov(fitted_levels["l26_pooled_v1"], True)),
    }
    per_lo[str(int(lo))] = entry

corpus["per_lo"] = per_lo
out["rover_corpus"] = corpus

# ====================================================== B. E-GSC7 paired tests
a = json.loads(
    subprocess.run(
        ["git", "show",
         "main:spf/calibrations/dual_rx_gain_frequency/reports/"
         "e_gsc7_iio_20260812_v1/analysis.json"],
        capture_output=True, text=True, check=True).stdout
)
res = {(e["radio"], e["transport"]): e for e in a["results_5766"]}
thr = a["h1_threshold_deg"]

gsc7 = dict(h1_threshold_deg=thr, h2_expected_deg=a["h2_expected_deg"])
gsc7["step_counts"] = {f"{r}_{t}": len(v["steps_deg"]) for (r, t), v in res.items()}
gsc7["resolved_steps"] = {
    f"{r}_{t}": int(sum(1 for s in v["steps_deg"] if abs(s) > thr))
    for (r, t), v in res.items()
}
gsc7["step_sums_deg"] = {f"{r}_{t}": float(sum(v["steps_deg"]))
                         for (r, t), v in res.items()}

# B1. paired transport agreement, per step, per radio (n = 10 pairs)
for radio in ("R17", "R18"):
    u = np.array(res[(radio, "usb")]["steps_deg"])
    p = np.array(res[(radio, "ip")]["steps_deg"])
    w = stats.wilcoxon(u, p)
    gsc7.setdefault("paired_usb_vs_ip_steps", {})[radio] = dict(
        n=len(u),
        mean_abs_difference_deg=float(np.abs(u - p).mean()),
        max_abs_difference_deg=float(np.abs(u - p).max()),
        wilcoxon_statistic=float(w.statistic),
        wilcoxon_p=float(w.pvalue),
    )

# B2. is the mean |step| resolvable? one-sample sign test against the threshold
for (radio, transport), v in res.items():
    s = np.abs(np.array(v["steps_deg"]))
    n_above = int((s > thr).sum())
    binom = stats.binomtest(n_above, len(s), 0.5, alternative="greater")
    gsc7.setdefault("steps_vs_resolution_threshold", {})[f"{radio}_{transport}"] = dict(
        n=len(s), n_above_threshold=n_above,
        median_abs_step_deg=float(np.median(s)),
        mean_abs_step_deg=float(s.mean()),
        binomial_p_majority_above=float(binom.pvalue),
    )

# B3. does the 5766 curve transfer? cross-LO rms vs same-LO cross-transport rms
rows = []
for (radio, transport), v in res.items():
    same_lo = a["transport_agreement"][radio]["curve_rms_deg"]
    for lo, e in v["cross_lo_curve_error"].items():
        rows.append(dict(radio=radio, transport=transport, lo_hz=int(lo),
                         cross_lo_rms_deg=e["rms_deg"],
                         cross_lo_max_deg=e["maximum_deg"],
                         same_lo_cross_transport_rms_deg=same_lo))
cl = np.array([r["cross_lo_rms_deg"] for r in rows])
sl = np.array([r["same_lo_cross_transport_rms_deg"] for r in rows])
w = stats.wilcoxon(cl, sl, alternative="greater")
gsc7["cross_lo_transfer"] = dict(
    rows=rows,
    n_pairs=len(rows),
    n_cross_lo_worse=int((cl > sl).sum()),
    median_cross_lo_rms_deg=float(np.median(cl)),
    median_same_lo_rms_deg=float(np.median(sl)),
    wilcoxon_statistic=float(w.statistic),
    wilcoxon_p_cross_lo_greater=float(w.pvalue),
)
# R18 only (the clean radio), 4 LOs x 2 transports
r18 = [r for r in rows if r["radio"] == "R18"]
cl18 = np.array([r["cross_lo_rms_deg"] for r in r18])
sl18 = np.array([r["same_lo_cross_transport_rms_deg"] for r in r18])
w18 = stats.wilcoxon(cl18, sl18, alternative="greater")
gsc7["cross_lo_transfer_R18_only"] = dict(
    n_pairs=len(r18), n_cross_lo_worse=int((cl18 > sl18).sum()),
    median_cross_lo_rms_deg=float(np.median(cl18)),
    same_lo_rms_deg=float(sl18[0]),
    wilcoxon_statistic=float(w18.statistic),
    wilcoxon_p_cross_lo_greater=float(w18.pvalue),
)
gsc7["rf_word_audit"] = a["rf_word_audit"]
out["e_gsc7"] = gsc7

# ================================================ C. segmentation fold algebra
def fold(theta):
    t = np.array(theta, dtype=float, copy=True)
    m = np.abs(t) > np.pi / 2
    t[m] = np.sign(t[m]) * np.pi - t[m]
    return t


rng = np.random.default_rng(7)
theta = rng.uniform(-np.pi, np.pi, 200000)
for c_deg in (2.0, 5.0, 10.0, 20.0):
    c = np.deg2rad(c_deg)
    correct_first = fold(theta - c)          # correction applied per window
    post_hoc = fold(theta) - c               # correction applied to stored mean
    err = np.rad2deg(np.abs(correct_first - post_hoc))
    outside = np.abs(theta) > np.pi / 2
    out.setdefault("segmentation_fold", {})[f"correction_{c_deg:g}deg"] = dict(
        fraction_theta_outside_half_pi=float(outside.mean()),
        mean_abs_disagreement_deg=float(err.mean()),
        disagreement_deg_inside=float(np.abs(err[~outside]).max()),
        disagreement_deg_outside_median=float(np.median(err[outside])),
        disagreement_deg_outside_max=float(err[outside].max()),
        ratio_outside_to_correction=float(np.median(err[outside]) / c_deg),
    )
out["segmentation_fold"]["source"] = (
    "spf/dataset/segmentation.py: mean_phase = trim_mean("
    "reduce_theta_to_positive_y(all_windows_stats[0])[mask], 0.1); fold is "
    "spf/rf.py reduce_theta_to_positive_y, theta -> sign(theta)*pi - theta for "
    "|theta| > pi/2 (read-only inspection; segmentation.py was not modified)"
)

(OUTDIR / "analysis.json").write_text(json.dumps(out, indent=1, sort_keys=True))
print("wrote", OUTDIR / "analysis.json")
print(json.dumps(corpus["lo_histogram_hz"], indent=1))
for lo, e in per_lo.items():
    print(lo, "n=", e["n_frames"], "caps=", e["n_rx_captures"],
          "uneq=", round(e["gain_unequal_fraction"], 4),
          "needcorr=", round(e["needs_correction_fraction"], 4),
          "gee0=", round(e["gain_endpoints_unequal_fraction_both_arms"], 4),
          "l26=", e["coefficient_set_support"]["l26_pooled_v1"]["frame_weighted_fraction"],
          "gsc6cov=", round(e["refit_coverage"]["e_gsc6"]["rf_only"]["needing_correction"], 4))
print(json.dumps(gsc7["cross_lo_transfer_R18_only"], indent=1))
