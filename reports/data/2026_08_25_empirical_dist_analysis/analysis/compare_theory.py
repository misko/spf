#!/usr/bin/env python3
"""Compare every empirical phase/bearing table with two-element-array theory.

This analysis is intentionally read-only with respect to its inputs.  It loads
the canonical empirical tables, fits low-dimensional theoretical descriptions,
and writes only CSV/JSON/PNG/PDF report products beneath ``--output-dir``.

The table contract is ``table[phi_bin, theta_bin] = P(theta | phi)``.  The
physical model starts with the repo's two-element far-field equation

    phi = wrap(phi0 - 2*pi*rho*g*sin(theta - theta0)), rho = d/lambda,

and a circular (von Mises) phase-noise model.  A uniform theta prior converts
``P(phi | theta)`` to a directly comparable theoretical ``P(theta | phi)``.

Two fits are reported:

* nominal: repo sign, g=1, theta0=phi0=0; only phase-noise width is fitted;
* calibrated: effective g, theta0, phi0, and phase-noise width are fitted.

Both are fitted to pooled-radio ``r/nosym``.  The same parameters are then
passed through the producer's exact index-based symmetry transform before
comparison with ``r/sym``.  This keeps physical fit and as-built table behavior
separate.  It does not silently re-fit away symmetry-operator error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.optimize import differential_evolution, minimize_scalar  # noqa: E402
from scipy.special import i0e, i1e  # noqa: E402


PROVENANCE_KEY = "__provenance__"
MODEL_VERSION = "two_element_vm_midpoint_v1"
FIT_BOUNDS = {
    "log_kappa": [-5.0, 5.5],
    "effective_spacing_gain": [0.25, 3.0],
    "theta_mount_shift_rad": [-0.9, 0.9],
    "phase_offset_rad": [-math.pi, math.pi],
}


@dataclass(frozen=True)
class Params:
    kappa: float
    spacing_gain: float = 1.0
    theta_shift: float = 0.0
    phase_offset: float = 0.0
    sign: int = -1


def parse_args() -> argparse.Namespace:
    repo = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "empirical_dists/full_20260809_v1.pkl").exists()
    )
    report = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--table",
        type=Path,
        default=repo / "empirical_dists/full_20260809_v1.pkl",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=repo / "empirical_dists/full.pkl",
    )
    p.add_argument("--output-dir", type=Path, default=report)
    p.add_argument("--maxiter", type=int, default=70)
    p.add_argument("--popsize", type=int, default=9)
    return p.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def spacing_keys(table: dict) -> list[str]:
    return sorted(k for k in table if k != PROVENANCE_KEY)


def split_key(key: str) -> tuple[str, float]:
    device, spacing = key.replace("SDRDEVICE.", "").rsplit("_", 1)
    return device, float(spacing)


def wrap(x: np.ndarray | float) -> np.ndarray:
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def bin_centers(n: int) -> np.ndarray:
    edges = np.linspace(-np.pi, np.pi, n + 1)
    return (edges[:-1] + edges[1:]) / 2


def normalize_rows(m: np.ndarray) -> np.ndarray:
    out = np.asarray(m, dtype=np.float64).copy()
    denom = out.sum(axis=1, keepdims=True)
    np.divide(out, denom, out=out, where=denom > 0)
    return out


def apply_repo_symmetry(joint_theta_phi: np.ndarray) -> np.ndarray:
    """Exact odd-bin transform used by create_empirical_p_dist.py."""
    h = np.asarray(joint_theta_phi)
    bins = h.shape[0]
    half = h[: math.ceil(bins / 2)] + np.flip(h[math.floor(bins / 2) :])
    half = half + np.flip(half, axis=0)
    return np.vstack([half[:-1], np.flip(half)])


def theory_matrices(rho: float, n: int, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """Return physical/nosym and producer-folded/sym P(theta|phi) matrices."""
    theta = bin_centers(n)
    phi = bin_centers(n)
    mu = wrap(
        params.phase_offset
        + params.sign
        * 2
        * np.pi
        * rho
        * params.spacing_gain
        * np.sin(theta - params.theta_shift)
    )
    delta = wrap(phi[:, None] - mu[None, :])
    # Subtracting one is a harmless shared factor and prevents overflow.
    likelihood_phi_theta = np.exp(params.kappa * (np.cos(delta) - 1.0))
    nosym = normalize_rows(likelihood_phi_theta)
    joint_theta_phi = likelihood_phi_theta.T
    folded_joint = apply_repo_symmetry(joint_theta_phi)
    sym = normalize_rows(folded_joint.T)
    return nosym, sym


def theory_matrices_integrated(
    rho: float, n: int, params: Params, quadrature: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-integrated theory for controlled discretization diagnostics.

    Main all-key fitting uses the much faster bin-midpoint approximation.  This
    sub-bin quadrature is used where the quantity of interest *is* a binning
    effect (the symmetry diagnostic), so that sharp/high-spacing ridges are not
    accidentally sampled at a favorable or unfavorable center.
    """
    centers = bin_centers(n)
    width = 2 * np.pi / n
    offsets = ((np.arange(quadrature) + 0.5) / quadrature - 0.5) * width
    joint_theta_phi = np.zeros((n, n), dtype=np.float64)
    for theta_offset in offsets:
        mu = wrap(
            params.phase_offset
            + params.sign
            * 2
            * np.pi
            * rho
            * params.spacing_gain
            * np.sin(centers + theta_offset - params.theta_shift)
        )
        for phase_offset in offsets:
            delta = wrap((centers + phase_offset)[:, None] - mu[None, :])
            joint_theta_phi += np.exp(
                params.kappa * (np.cos(delta) - 1.0)
            ).T
    joint_theta_phi /= quadrature**2
    nosym = normalize_rows(joint_theta_phi.T)
    sym = normalize_rows(apply_repo_symmetry(joint_theta_phi).T)
    return nosym, sym


def valid_rows(empirical: np.ndarray) -> np.ndarray:
    return np.isfinite(empirical).all(axis=1) & (empirical.sum(axis=1) > 1e-12)


def mean_row_js(empirical: np.ndarray, model: np.ndarray) -> float:
    keep = valid_rows(empirical)
    p = normalize_rows(empirical[keep])
    q = normalize_rows(model[keep])
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log(p / m), 0).sum(axis=1)
        kl_qm = np.where(q > 0, q * np.log(q / m), 0).sum(axis=1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def matrix_metrics(empirical: np.ndarray, model: np.ndarray) -> dict[str, float]:
    keep = valid_rows(empirical)
    p = normalize_rows(empirical[keep])
    q = normalize_rows(model[keep])
    tv_rows = 0.5 * np.abs(p - q).sum(axis=1)
    a, b = p.ravel(), q.ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else math.nan
    map_agreement = float(np.mean(np.argmax(p, axis=1) == np.argmax(q, axis=1)))
    model_map_mass = float(np.mean(p[np.arange(len(p)), np.argmax(q, axis=1)]))
    return {
        "mean_row_tv": float(tv_rows.mean()),
        "median_row_tv": float(np.median(tv_rows)),
        "max_row_tv": float(tv_rows.max()),
        "mean_row_js_nats": mean_row_js(p, q),
        "flattened_corr": corr,
        "map_bin_agreement": map_agreement,
        "empirical_mass_at_model_map": model_map_mass,
        "n_nonempty_phi_rows": int(keep.sum()),
    }


def fit_nominal(empirical: np.ndarray, rho: float, sign: int) -> tuple[Params, float]:
    n = empirical.shape[0]

    def objective(log_kappa: float) -> float:
        p, _ = theory_matrices(rho, n, Params(kappa=math.exp(log_kappa), sign=sign))
        return mean_row_js(empirical, p)

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=tuple(FIT_BOUNDS["log_kappa"]),
        options={"xatol": 1e-5},
    )
    params = Params(kappa=math.exp(float(result.x)), sign=sign)
    return params, float(result.fun)


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def fit_calibrated(
    empirical: np.ndarray,
    rho: float,
    key: str,
    nominal: Params,
    maxiter: int,
    popsize: int,
) -> tuple[Params, float, str]:
    n = empirical.shape[0]

    def unpack(x: np.ndarray) -> Params:
        return Params(
            kappa=math.exp(float(x[0])),
            spacing_gain=float(x[1]),
            theta_shift=float(x[2]),
            phase_offset=float(wrap(x[3])),
            sign=-1,
        )

    def objective(x: np.ndarray) -> float:
        p, _ = theory_matrices(rho, n, unpack(x))
        return mean_row_js(empirical, p)

    x0 = np.array([math.log(nominal.kappa), 1.0, 0.0, 0.0])
    bounds = [
        tuple(FIT_BOUNDS["log_kappa"]),
        tuple(FIT_BOUNDS["effective_spacing_gain"]),
        tuple(FIT_BOUNDS["theta_mount_shift_rad"]),
        tuple(FIT_BOUNDS["phase_offset_rad"]),
    ]
    result = differential_evolution(
        objective,
        bounds,
        seed=stable_seed(key),
        x0=x0,
        maxiter=maxiter,
        popsize=popsize,
        tol=2e-5,
        polish=True,
        updating="immediate",
        workers=1,
    )
    fitted = unpack(result.x)
    fit_obj = float(result.fun)
    nominal_obj = objective(x0)
    if not np.isfinite(fit_obj) or fit_obj > nominal_obj + 1e-10:
        return nominal, float(nominal_obj), "nominal_fallback"
    return fitted, fit_obj, "ok" if result.success else "maxiter_polished"


def kappa_to_circular_sigma_deg(kappa: float) -> float:
    ratio = float(i1e(kappa) / i0e(kappa))
    ratio = float(np.clip(ratio, 1e-15, 1.0))
    return math.degrees(math.sqrt(max(0.0, -2.0 * math.log(ratio))))


def max_bearing_solutions(rho: float) -> int:
    phase = np.linspace(-np.pi, np.pi, 20001, endpoint=False)
    ks = np.arange(-math.ceil(rho) - 2, math.ceil(rho) + 3)
    counts = np.zeros_like(phase, dtype=int)
    for k in ks:
        s = -(phase + 2 * np.pi * k) / (2 * np.pi * rho)
        counts += np.abs(s) < 1.0 - 1e-9
    return int(2 * counts.max())


def fit_all(
    table: dict,
    maxiter: int,
    popsize: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Params]]]:
    prov_keys = table.get(PROVENANCE_KEY, {}).get("keys", {})
    uniform_cache: dict[int, np.ndarray] = {}
    rows: list[dict] = []
    models: dict[str, dict[str, Params]] = {}
    keys = spacing_keys(table)
    for idx, key in enumerate(keys, 1):
        print(f"fit {idx:02d}/{len(keys)} {key}", flush=True)
        device, rho = split_key(key)
        empirical = as_array(table[key]["r"]["nosym"])
        empirical_sym = as_array(table[key]["r"]["sym"])
        n = empirical.shape[0]
        uniform = uniform_cache.setdefault(n, np.full((n, n), 1.0 / n))

        nominal, nominal_obj = fit_nominal(empirical, rho, sign=-1)
        opposite, opposite_obj = fit_nominal(empirical, rho, sign=+1)
        calibrated, calibrated_obj, fit_status = fit_calibrated(
            empirical, rho, key, nominal, maxiter=maxiter, popsize=popsize
        )
        nom_nosym, nom_sym = theory_matrices(rho, n, nominal)
        cal_nosym, cal_sym = theory_matrices(rho, n, calibrated)
        opp_nosym, _ = theory_matrices(rho, n, opposite)

        m_uniform = matrix_metrics(empirical, uniform)
        m_nom = matrix_metrics(empirical, nom_nosym)
        m_cal = matrix_metrics(empirical, cal_nosym)
        m_opp = matrix_metrics(empirical, opp_nosym)
        m_sym_nom = matrix_metrics(empirical_sym, nom_sym)
        m_sym_cal = matrix_metrics(empirical_sym, cal_sym)

        row = {
            "key": key,
            "device": device,
            "d_lambda": rho,
            "n_datasets": int(prov_keys.get(key, {}).get("n_datasets", 0)),
            "spatially_aliased": bool(rho > 0.5),
            "max_theoretical_bearing_solutions": max_bearing_solutions(rho),
            "repo_sign_wins": bool(nominal_obj < opposite_obj),
            "nominal_kappa": nominal.kappa,
            "nominal_phase_sigma_deg": kappa_to_circular_sigma_deg(nominal.kappa),
            "opposite_sign_kappa": opposite.kappa,
            "cal_kappa": calibrated.kappa,
            "cal_phase_sigma_deg": kappa_to_circular_sigma_deg(calibrated.kappa),
            "cal_effective_spacing_gain": calibrated.spacing_gain,
            "cal_effective_d_lambda": rho * calibrated.spacing_gain,
            "cal_theta_shift_deg": math.degrees(calibrated.theta_shift),
            "cal_phase_offset_deg": math.degrees(calibrated.phase_offset),
            "cal_fit_status": fit_status,
            "uniform_nosym_tv": m_uniform["mean_row_tv"],
            "nominal_nosym_tv": m_nom["mean_row_tv"],
            "nominal_nosym_js_nats": m_nom["mean_row_js_nats"],
            "nominal_nosym_corr": m_nom["flattened_corr"],
            "opposite_nosym_tv": m_opp["mean_row_tv"],
            "opposite_nosym_js_nats": m_opp["mean_row_js_nats"],
            "cal_nosym_tv": m_cal["mean_row_tv"],
            "cal_nosym_js_nats": m_cal["mean_row_js_nats"],
            "cal_nosym_corr": m_cal["flattened_corr"],
            "cal_nosym_map_agreement": m_cal["map_bin_agreement"],
            "cal_nosym_empirical_mass_at_map": m_cal[
                "empirical_mass_at_model_map"
            ],
            "cal_tv_skill_vs_uniform": 1.0
            - m_cal["mean_row_tv"] / m_uniform["mean_row_tv"],
            "nominal_sym_tv": m_sym_nom["mean_row_tv"],
            "nominal_sym_js_nats": m_sym_nom["mean_row_js_nats"],
            "nominal_sym_corr": m_sym_nom["flattened_corr"],
            "cal_sym_tv": m_sym_cal["mean_row_tv"],
            "cal_sym_js_nats": m_sym_cal["mean_row_js_nats"],
            "cal_sym_corr": m_sym_cal["flattened_corr"],
            "cal_sym_map_agreement": m_sym_cal["map_bin_agreement"],
            "nosym_nonempty_phi_rows": m_cal["n_nonempty_phi_rows"],
            "sym_nonempty_phi_rows": m_sym_cal["n_nonempty_phi_rows"],
            "fit_objective_nominal": nominal_obj,
            "fit_objective_calibrated": calibrated_obj,
            "fit_objective_opposite_sign": opposite_obj,
        }
        rows.append(row)
        models[key] = {
            "nominal": nominal,
            "opposite": opposite,
            "calibrated": calibrated,
        }
    frame = pd.DataFrame(rows).sort_values(["d_lambda", "device"]).reset_index(drop=True)
    return frame, models


def variant_metrics(table: dict, models: dict[str, dict[str, Params]]) -> pd.DataFrame:
    rows = []
    for key in spacing_keys(table):
        device, rho = split_key(key)
        for radio in ("r", "r0", "r1"):
            for symmetry in ("nosym", "sym"):
                empirical = as_array(table[key][radio][symmetry])
                n = empirical.shape[0]
                nom = theory_matrices(rho, n, models[key]["nominal"])[
                    symmetry == "sym"
                ]
                cal = theory_matrices(rho, n, models[key]["calibrated"])[
                    symmetry == "sym"
                ]
                mn = matrix_metrics(empirical, nom)
                mc = matrix_metrics(empirical, cal)
                rows.append(
                    {
                        "key": key,
                        "device": device,
                        "d_lambda": rho,
                        "radio": radio,
                        "symmetry": symmetry,
                        "parameters_fitted_to": "r/nosym",
                        "nominal_mean_row_tv": mn["mean_row_tv"],
                        "nominal_mean_row_js_nats": mn["mean_row_js_nats"],
                        "nominal_flattened_corr": mn["flattened_corr"],
                        "cal_mean_row_tv": mc["mean_row_tv"],
                        "cal_mean_row_js_nats": mc["mean_row_js_nats"],
                        "cal_flattened_corr": mc["flattened_corr"],
                        "cal_map_bin_agreement": mc["map_bin_agreement"],
                        "nonempty_phi_rows": mc["n_nonempty_phi_rows"],
                    }
                )
    return pd.DataFrame(rows).sort_values(["d_lambda", "device", "radio", "symmetry"])


def old_new_metrics(
    table: dict, baseline: dict, models: dict[str, dict[str, Params]]
) -> pd.DataFrame:
    rows = []
    for key in sorted(set(spacing_keys(table)) & set(spacing_keys(baseline))):
        device, rho = split_key(key)
        old = as_array(baseline[key]["r"]["sym"])
        new = as_array(table[key]["r"]["sym"])
        cal = theory_matrices(rho, old.shape[0], models[key]["calibrated"])[1]
        direct = matrix_metrics(old, new)
        mo = matrix_metrics(old, cal)
        mn = matrix_metrics(new, cal)
        rows.append(
            {
                "key": key,
                "device": device,
                "d_lambda": rho,
                "old_to_new_mean_row_tv": direct["mean_row_tv"],
                "old_to_new_corr": direct["flattened_corr"],
                "old_cal_theory_tv": mo["mean_row_tv"],
                "new_cal_theory_tv": mn["mean_row_tv"],
                "new_minus_old_cal_theory_tv": mn["mean_row_tv"]
                - mo["mean_row_tv"],
            }
        )
    return pd.DataFrame(rows).sort_values("old_to_new_mean_row_tv", ascending=False)


def cross_device_metrics(table: dict) -> pd.DataFrame:
    by_rho: dict[float, dict[str, str]] = {}
    for key in spacing_keys(table):
        device, rho = split_key(key)
        by_rho.setdefault(rho, {})[device] = key
    rows = []
    for rho, devices in sorted(by_rho.items()):
        if not {"PLUTO", "BLADERF2"}.issubset(devices):
            continue
        for symmetry in ("nosym", "sym"):
            a = as_array(table[devices["PLUTO"]]["r"][symmetry])
            b = as_array(table[devices["BLADERF2"]]["r"][symmetry])
            m = matrix_metrics(a, b)
            rows.append(
                {
                    "d_lambda": rho,
                    "symmetry": symmetry,
                    "pluto_key": devices["PLUTO"],
                    "bladerf2_key": devices["BLADERF2"],
                    "mean_row_tv": m["mean_row_tv"],
                    "flattened_corr": m["flattened_corr"],
                    "map_bin_agreement": m["map_bin_agreement"],
                }
            )
    return pd.DataFrame(rows)


def write_full_table(frame: pd.DataFrame, path: Path) -> None:
    cols = [
        "key",
        "n_datasets",
        "max_theoretical_bearing_solutions",
        "uniform_nosym_tv",
        "nominal_nosym_tv",
        "cal_nosym_tv",
        "cal_nosym_corr",
        "cal_sym_tv",
        "cal_sym_corr",
        "cal_effective_spacing_gain",
        "cal_theta_shift_deg",
        "cal_phase_offset_deg",
        "cal_phase_sigma_deg",
    ]
    headers = [
        "key",
        "n",
        "max bearings",
        "uniform TV",
        "nominal TV",
        "cal TV",
        "cal corr",
        "prod-sym TV",
        "prod-sym corr",
        "g",
        "theta0 deg",
        "phi0 deg",
        "phase sigma deg",
    ]
    lines = [
        "# All-key theory comparison",
        "",
        "Primary fit target is pooled-radio `r/nosym`. `prod-sym` applies the same fit",
        "through the repository's exact symmetry transform and compares it with `r/sym`.",
        "TV is mean total-variation distance per observed-phase row (0 is identical, 1 is",
        "disjoint). `corr` is the flattened Pearson correlation.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for _, r in frame[cols].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r['key'].replace('SDRDEVICE.', '')}`",
                    f"{int(r['n_datasets'])}",
                    f"{int(r['max_theoretical_bearing_solutions'])}",
                    f"{r['uniform_nosym_tv']:.3f}",
                    f"{r['nominal_nosym_tv']:.3f}",
                    f"{r['cal_nosym_tv']:.3f}",
                    f"{r['cal_nosym_corr']:.3f}",
                    f"{r['cal_sym_tv']:.3f}",
                    f"{r['cal_sym_corr']:.3f}",
                    f"{r['cal_effective_spacing_gain']:.3f}",
                    f"{r['cal_theta_shift_deg']:+.1f}",
                    f"{r['cal_phase_offset_deg']:+.1f}",
                    f"{r['cal_phase_sigma_deg']:.1f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


DEVICE_COLORS = {"PLUTO": "#2d6cdf", "BLADERF2": "#e67e22"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 14,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def heatmap(ax, m: np.ndarray, title: str, vmax=None, cmap="magma"):
    image = ax.imshow(
        m,
        origin="lower",
        extent=[-180, 180, -180, 180],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("bearing theta (deg)")
    ax.set_ylabel("observed phase phi (deg)")
    ax.grid(False)
    return image


def fig_theory_geometry(fig_dir: Path) -> None:
    rhos = [0.30, 0.48, 0.90, 1.35]
    theta = np.linspace(-np.pi, np.pi, 1200)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    for c, rho in enumerate(rhos):
        unwrapped = -2 * np.pi * rho * np.sin(theta)
        axes[0, c].plot(np.degrees(theta), np.degrees(unwrapped), lw=1.3)
        for y in (-180, 180):
            axes[0, c].axhline(y, color="0.45", ls=":", lw=0.8)
        axes[0, c].set_title(f"d/lambda={rho:.2f}")
        axes[0, c].set_xlabel("bearing theta (deg)")
        axes[0, c].set_ylabel("unwrapped phase (deg)")
        model, _ = theory_matrices(
            rho,
            65,
            Params(kappa=25.0, sign=-1),
        )
        heatmap(axes[1, c], model, f"ideal P(theta | phi), max {max_bearing_solutions(rho)} bearings")
    fig.suptitle(
        "Two-element far-field theory: wrapping creates required multimodality above d/lambda=0.5"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "theoretical_geometry.png", dpi=180)
    plt.close(fig)


def fig_fleet_summary(frame: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    labels = ["uniform", "nominal", "calibrated"]
    cols = ["uniform_nosym_tv", "nominal_nosym_tv", "cal_nosym_tv"]
    data = [frame[c].to_numpy() for c in cols]
    bp = ax.boxplot(data, tick_labels=labels, widths=0.55, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#b4b4b4", "#78a7e8", "#2d6cdf"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    rng = np.random.default_rng(20260825)
    for i, values in enumerate(data, 1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(values)), values, s=9, c="0.15", alpha=0.35)
    ax.set_ylabel("mean row TV (lower is better)")
    ax.set_title("Pooled r/nosym: geometry carries real information")

    ax = axes[0, 1]
    for device, group in frame.groupby("device"):
        sizes = 18 + 12 * np.log1p(group["n_datasets"].to_numpy())
        ax.scatter(
            group["d_lambda"],
            group["cal_nosym_tv"],
            s=sizes,
            color=DEVICE_COLORS[device],
            alpha=0.78,
            label=device,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axvline(0.5, ls=":", c="0.35", label="alias threshold")
    for _, r in frame.nlargest(5, "cal_nosym_tv").iterrows():
        ax.annotate(
            r["key"].replace("SDRDEVICE.", ""),
            (r["d_lambda"], r["cal_nosym_tv"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("d/lambda")
    ax.set_ylabel("calibrated mean row TV")
    ax.set_title("Residual mismatch by spacing (marker size = source count)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(frame["cal_nosym_tv"], frame["cal_sym_tv"], c=frame["d_lambda"], cmap="viridis", s=38)
    lim = [0, max(frame["cal_nosym_tv"].max(), frame["cal_sym_tv"].max()) * 1.04]
    ax.plot(lim, lim, ls="--", c="0.4", lw=1)
    ax.set(xlim=lim, ylim=lim)
    ax.set_xlabel("physical r/nosym calibrated TV")
    ax.set_ylabel("as-built r/sym calibrated TV")
    ax.set_title("Same physical fit carried through producer symmetry")

    ax = axes[1, 1]
    ax.scatter(
        frame["fit_objective_opposite_sign"],
        frame["fit_objective_nominal"],
        c=[DEVICE_COLORS[d] for d in frame["device"]],
        s=36,
        alpha=0.8,
    )
    lo = min(frame["fit_objective_opposite_sign"].min(), frame["fit_objective_nominal"].min())
    hi = max(frame["fit_objective_opposite_sign"].max(), frame["fit_objective_nominal"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", c="0.4")
    ax.set_xlabel("opposite-sign row JS (nats)")
    ax.set_ylabel("repo-sign row JS (nats)")
    wins = int(frame["repo_sign_wins"].sum())
    ax.set_title(f"Sign negative control: repo sign wins {wins}/{len(frame)} keys")
    fig.suptitle("Fleet-wide empirical-to-theory agreement")
    fig.tight_layout()
    fig.savefig(fig_dir / "fleet_fit_summary.png", dpi=180)
    plt.close(fig)


def fig_fitted_parameters(frame: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = [
        ("cal_effective_spacing_gain", "effective spacing gain g", 1.0),
        ("cal_phase_sigma_deg", "fitted circular phase sigma (deg)", None),
        ("cal_theta_shift_deg", "fitted theta mount shift (deg)", 0.0),
        ("cal_phase_offset_deg", "fitted phase offset (deg)", 0.0),
    ]
    for ax, (col, ylabel, reference) in zip(axes.ravel(), specs):
        for device, group in frame.groupby("device"):
            ax.scatter(
                group["d_lambda"],
                group[col],
                color=DEVICE_COLORS[device],
                s=35,
                alpha=0.76,
                label=device,
            )
        if reference is not None:
            ax.axhline(reference, ls="--", c="0.35", lw=1)
        ax.axvline(0.5, ls=":", c="0.6", lw=0.8)
        ax.set_xlabel("d/lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(col.replace("cal_", "").replace("_", " "))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Best-fit effective parameters (descriptive; aliasing makes some non-identifiable)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fitted_parameters.png", dpi=180)
    plt.close(fig)


def select_representative_keys(table: dict) -> list[str]:
    preferred = [
        "SDRDEVICE.PLUTO_0.12208",
        "SDRDEVICE.PLUTO_0.48083",
        "SDRDEVICE.BLADERF2_0.48083",
        "SDRDEVICE.PLUTO_0.67317",
        "SDRDEVICE.PLUTO_0.90397",
        "SDRDEVICE.PLUTO_0.91964",
        "SDRDEVICE.PLUTO_1.34727",
        "SDRDEVICE.PLUTO_0.56296",
    ]
    return [k for k in preferred if k in table]


def fig_representative(
    table: dict, frame: pd.DataFrame, models: dict[str, dict[str, Params]], fig_dir: Path
) -> None:
    keys = select_representative_keys(table)
    by_key = frame.set_index("key")
    fig, axes = plt.subplots(len(keys), 4, figsize=(15, 3.05 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        _, rho = split_key(key)
        empirical = as_array(table[key]["r"]["nosym"])
        nominal, _ = theory_matrices(rho, empirical.shape[0], models[key]["nominal"])
        calibrated, _ = theory_matrices(rho, empirical.shape[0], models[key]["calibrated"])
        vmax = np.quantile(np.hstack([empirical.ravel(), nominal.ravel(), calibrated.ravel()]), 0.995)
        heatmap(axes[row, 0], empirical, f"{key.replace('SDRDEVICE.','')} empirical", vmax=vmax)
        heatmap(
            axes[row, 1],
            nominal,
            f"nominal, TV={by_key.loc[key, 'nominal_nosym_tv']:.3f}",
            vmax=vmax,
        )
        heatmap(
            axes[row, 2],
            calibrated,
            f"calibrated, TV={by_key.loc[key, 'cal_nosym_tv']:.3f}",
            vmax=vmax,
        )
        heatmap(
            axes[row, 3],
            np.abs(empirical - calibrated),
            f"absolute residual, corr={by_key.loc[key, 'cal_nosym_corr']:.3f}",
            cmap="viridis",
        )
    fig.suptitle(
        "Representative pooled r/nosym tables: empirical, nominal theory, calibrated theory, residual",
        y=0.998,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.986])
    fig.savefig(fig_dir / "representative_heatmaps.png", dpi=170)
    plt.close(fig)


def fig_symmetry_bias(frame: pd.DataFrame, fig_dir: Path) -> None:
    sigma = 0.2
    # Invert circular sigma approximately by a one-dimensional search.
    def loss(logk):
        return abs(kappa_to_circular_sigma_deg(math.exp(logk)) - math.degrees(sigma))

    kappa = math.exp(minimize_scalar(loss, bounds=(-2, 8), method="bounded").x)
    values = []
    for rho in frame["d_lambda"]:
        physical, folded = theory_matrices_integrated(
            float(rho), 65, Params(kappa=kappa), quadrature=9
        )
        values.append(matrix_metrics(physical, folded)["mean_row_tv"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].scatter(frame["d_lambda"], values, c=frame["d_lambda"], cmap="viridis", s=36)
    axes[0].set_xlabel("d/lambda")
    axes[0].set_ylabel("mean row TV: correct ideal vs repo-folded ideal")
    axes[0].set_title("Confirmed transform bias for 65 bins, phase sigma=0.2 rad")
    example_rho = 1.54880
    physical, folded = theory_matrices_integrated(
        example_rho, 65, Params(kappa=kappa), quadrature=9
    )
    diff = np.abs(physical - folded)
    inset = axes[1]
    heatmap(inset, diff, f"absolute theoretical distortion at d/lambda={example_rho:.5f}", cmap="viridis")
    fig.suptitle("The index symmetry operator changes an already symmetric physical model")
    fig.tight_layout()
    fig.savefig(fig_dir / "symmetry_operator_bias.png", dpi=180)
    plt.close(fig)


def fig_cross_device(table: dict, cross: pd.DataFrame, fig_dir: Path) -> None:
    sym = cross[cross["symmetry"] == "sym"].sort_values("d_lambda")
    if sym.empty:
        return
    fig, axes = plt.subplots(len(sym), 3, figsize=(12.5, 3.4 * len(sym)), squeeze=False)
    for row, (_, r) in enumerate(sym.iterrows()):
        a = as_array(table[r["pluto_key"]]["r"]["sym"])
        b = as_array(table[r["bladerf2_key"]]["r"]["sym"])
        vmax = np.quantile(np.hstack([a.ravel(), b.ravel()]), 0.995)
        heatmap(axes[row, 0], a, f"PLUTO d/lambda={r['d_lambda']:.5f}", vmax=vmax)
        heatmap(axes[row, 1], b, "BLADERF2 (theory predicts same matrix)", vmax=vmax)
        heatmap(
            axes[row, 2],
            np.abs(a - b),
            f"abs difference; TV={r['mean_row_tv']:.3f}, corr={r['flattened_corr']:.3f}",
            cmap="viridis",
        )
    fig.suptitle("Negative control: identical d/lambda, different device/corpus")
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_device_same_spacing.png", dpi=180)
    plt.close(fig)


def fig_atlas(
    table: dict, frame: pd.DataFrame, models: dict[str, dict[str, Params]], fig_dir: Path
) -> None:
    by_key = frame.set_index("key")
    keys = frame["key"].tolist()
    pdf_path = fig_dir / "all_48_keys_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_index, start in enumerate(range(0, len(keys), 6)):
            page_keys = keys[start : start + 6]
            fig, axes = plt.subplots(len(page_keys), 3, figsize=(12, 2.75 * len(page_keys)), squeeze=False)
            for row, key in enumerate(page_keys):
                _, rho = split_key(key)
                empirical = as_array(table[key]["r"]["nosym"])
                calibrated, _ = theory_matrices(rho, empirical.shape[0], models[key]["calibrated"])
                vmax = np.quantile(np.hstack([empirical.ravel(), calibrated.ravel()]), 0.995)
                heatmap(axes[row, 0], empirical, f"{key.replace('SDRDEVICE.','')} empirical", vmax=vmax)
                heatmap(
                    axes[row, 1],
                    calibrated,
                    f"calibrated theory; TV={by_key.loc[key, 'cal_nosym_tv']:.3f}",
                    vmax=vmax,
                )
                heatmap(
                    axes[row, 2],
                    np.abs(empirical - calibrated),
                    f"absolute residual; corr={by_key.loc[key, 'cal_nosym_corr']:.3f}",
                    cmap="viridis",
                )
            fig.suptitle(
                f"All-key atlas — pooled r/nosym — page {page_index + 1}",
                y=0.998,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.986])
            pdf.savefig(fig, dpi=150)
            if page_index == 0:
                fig.savefig(fig_dir / "all_keys_atlas_page1.png", dpi=160)
            plt.close(fig)


def fig_production_atlas(
    table: dict, frame: pd.DataFrame, models: dict[str, dict[str, Params]], fig_dir: Path
) -> None:
    """All production r/sym tables beside as-built and physical theory."""
    by_key = frame.set_index("key")
    keys = frame["key"].tolist()
    pdf_path = fig_dir / "all_48_keys_production_sym_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_index, start in enumerate(range(0, len(keys), 6)):
            page_keys = keys[start : start + 6]
            fig, axes = plt.subplots(
                len(page_keys), 4, figsize=(15.8, 2.75 * len(page_keys)), squeeze=False
            )
            for row, key in enumerate(page_keys):
                _, rho = split_key(key)
                empirical = as_array(table[key]["r"]["sym"])
                physical, folded = theory_matrices(
                    rho, empirical.shape[0], models[key]["calibrated"]
                )
                vmax = np.quantile(
                    np.hstack([empirical.ravel(), folded.ravel(), physical.ravel()]),
                    0.995,
                )
                heatmap(
                    axes[row, 0],
                    empirical,
                    f"{key.replace('SDRDEVICE.','')} production r/sym",
                    vmax=vmax,
                )
                heatmap(
                    axes[row, 1],
                    folded,
                    f"as-built sym theory; TV={by_key.loc[key, 'cal_sym_tv']:.3f}",
                    vmax=vmax,
                )
                heatmap(
                    axes[row, 2],
                    physical,
                    "physical theory before index fold",
                    vmax=vmax,
                )
                heatmap(
                    axes[row, 3],
                    np.abs(empirical - folded),
                    f"production residual; corr={by_key.loc[key, 'cal_sym_corr']:.3f}",
                    cmap="viridis",
                )
            fig.suptitle(
                f"All-key production atlas — pooled r/sym — page {page_index + 1}",
                y=0.998,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.986])
            pdf.savefig(fig, dpi=150)
            if page_index == 0:
                fig.savefig(fig_dir / "production_sym_atlas_page1.png", dpi=160)
            plt.close(fig)


def fig_old_new(old_new: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ordered = old_new.sort_values("old_to_new_mean_row_tv", ascending=True)
    y = np.arange(len(ordered))
    colors = [DEVICE_COLORS[d] for d in ordered["device"]]
    axes[0].barh(y, ordered["old_to_new_mean_row_tv"], color=colors, alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([k.replace("SDRDEVICE.", "") for k in ordered["key"]], fontsize=6)
    axes[0].set_xlabel("old-to-current mean row TV (r/sym)")
    axes[0].set_title("44 shared keys: direct table movement")
    axes[1].scatter(
        old_new["old_cal_theory_tv"],
        old_new["new_cal_theory_tv"],
        c=[DEVICE_COLORS[d] for d in old_new["device"]],
        s=35,
    )
    hi = max(old_new["old_cal_theory_tv"].max(), old_new["new_cal_theory_tv"].max()) * 1.03
    axes[1].plot([0, hi], [0, hi], ls="--", c="0.4")
    for _, r in old_new.head(4).iterrows():
        axes[1].annotate(
            r["key"].replace("SDRDEVICE.", ""),
            (r["old_cal_theory_tv"], r["new_cal_theory_tv"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_xlabel("old table vs current calibrated theory TV")
    axes[1].set_ylabel("current table vs current calibrated theory TV")
    axes[1].set_title("Did the rebuild move shared keys toward the current fit?")
    fig.suptitle("Baseline full.pkl versus current full_20260809_v1.pkl")
    fig.tight_layout()
    fig.savefig(fig_dir / "old_vs_current_theory.png", dpi=180)
    plt.close(fig)


def write_metadata(
    args: argparse.Namespace, table: dict, frame: pd.DataFrame, path: Path
) -> None:
    metadata = {
        "analysis_model_version": MODEL_VERSION,
        "generated_utc_note": "Generated on the report date; inputs are read-only.",
        "table": {"path": str(args.table.resolve()), "sha256": sha256(args.table)},
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sha256": sha256(args.baseline),
        },
        "n_current_keys": len(spacing_keys(table)),
        "n_repo_sign_wins": int(frame["repo_sign_wins"].sum()),
        "fit_bounds": FIT_BOUNDS,
        "fit_target": "pooled r/nosym P(theta|phi), fixed consumer coordinates",
        "nominal_model": "repo sign; g=1; theta0=0; phi0=0; fit kappa",
        "calibrated_model": "repo sign; fit g, theta0, phi0, kappa",
        "discretization": "65 phase-bin and bearing-bin centers (midpoint approximation)",
        "conditional_prior": "uniform theta prior",
        "metrics": {
            "tv": "mean per-nonempty-phase-row total variation; [0,1]",
            "js": "mean per-nonempty-phase-row Jensen-Shannon divergence; natural log",
            "corr": "flattened Pearson correlation over nonempty empirical phase rows",
        },
        "optimizer": {
            "differential_evolution_maxiter": args.maxiter,
            "population_multiplier": args.popsize,
            "deterministic_seed": "sha256(key) first 32 bits",
        },
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    table = load_pickle(args.table)
    baseline = load_pickle(args.baseline)
    frame, models = fit_all(table, maxiter=args.maxiter, popsize=args.popsize)
    variants = variant_metrics(table, models)
    old_new = old_new_metrics(table, baseline, models)
    cross = cross_device_metrics(table)

    frame.to_csv(out / "metrics_all_keys.csv", index=False, float_format="%.10g")
    variants.to_csv(out / "metrics_all_variants.csv", index=False, float_format="%.10g")
    old_new.to_csv(out / "metrics_old_vs_current.csv", index=False, float_format="%.10g")
    cross.to_csv(out / "metrics_cross_device.csv", index=False, float_format="%.10g")
    write_full_table(frame, out / "ALL_KEYS_TABLE.md")
    write_metadata(args, table, frame, out / "analysis_metadata.json")

    fig_theory_geometry(fig_dir)
    fig_fleet_summary(frame, fig_dir)
    fig_fitted_parameters(frame, fig_dir)
    fig_representative(table, frame, models, fig_dir)
    fig_symmetry_bias(frame, fig_dir)
    fig_cross_device(table, cross, fig_dir)
    fig_atlas(table, frame, models, fig_dir)
    fig_production_atlas(table, frame, models, fig_dir)
    fig_old_new(old_new, fig_dir)
    print(f"wrote report products to {out}")


if __name__ == "__main__":
    main()
