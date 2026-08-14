"""In-memory gain-phase correction wrapper around a v5spfdataset.

THE UNDERLYING ROVER DATA IS NEVER MODIFIED. This class opens nothing, writes
nothing, and holds a corrected COPY of ``mean_phase`` in RAM. It is the same
pattern ``v5spfdataset.v4_to_v5`` already uses -- "wrap readonly so we can modify
on the fly" -- applied one level up.

Everything except the phase is delegated to the wrapped dataset by ``__getattr__``,
so the filters cannot tell the difference and need no changes.

WHICH FIELD IS CORRECTED, AND WHY IT MATTERS
--------------------------------------------
Only ``mean_phase`` (and its per-item twin ``mean_phase_segmentation``) is
corrected. That field is ``arctan2(sum sin*w, sum cos*w)`` -- a circular mean with
NO fold -- and an additive phase correction commutes with it exactly (measured at
0.00e+00 deg). The other candidate, ``weighted_windows_stats[0]``, applies
``reduce_theta_to_positive_y`` and does NOT commute: correcting through a fold
turns a 10 deg error into 20. Correcting the wrong field would be silently wrong.

MODES
-----
``arm_lut``   subtract the model's predicted RX1-minus-RX2 gain-phase offset
``constant``  subtract only the per-receiver MEAN of that prediction, which tests
              whether a per-session constant is already absorbed downstream
``shuffled``  negative control: the same correction VALUES, permuted across gain
              states, so magnitude and distribution are preserved but the gain
              mapping is destroyed. If this scores like ``arm_lut`` the effect is
              not the correction.

Unsupported ``(frequency, g1, g2)`` cells fail closed to a zero correction, which
is the shipped model's own strict-mode contract.
"""

from __future__ import annotations

import numpy as np
import torch

from spf.calibrations.models import UnsupportedPhaseModelInput, load_phase_model

MODES = ("arm_lut", "constant", "shuffled")


class PhaseCorrectedDataset:
    """Read-only, in-memory phase correction over a v5spfdataset."""

    def __init__(self, ds, mode: str, model_path: str, shuffle_seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"unknown phase_correction mode {mode!r}; use {MODES}")
        self._ds = ds
        self.phase_correction_mode = mode
        self._model = load_phase_model(model_path)
        self._shuffle_seed = shuffle_seed
        self._corr = self._build_corrections()
        self.mean_phase = self._corrected_mean_phase()

    # ---- delegation: everything not defined here comes from the wrapped dataset
    def __getattr__(self, name):
        return getattr(self._ds, name)

    def __len__(self):
        return len(self._ds)

    # ---- correction construction -------------------------------------------
    def _build_corrections(self) -> dict[int, np.ndarray]:
        """Per-receiver array of corrections in radians; 0 where unsupported."""
        out = {}
        for ridx in range(self._ds.n_receivers):
            keys = self._ds.cached_keys[ridx]
            gains = np.asarray(keys["gains"], dtype=float)
            lo = np.asarray(keys["rx_lo"], dtype=float)
            n = gains.shape[0]
            corr = np.zeros(n, dtype=np.float64)
            supported = np.zeros(n, dtype=bool)
            cache: dict[tuple, float | None] = {}
            for i in range(n):
                g1, g2 = gains[i, 0], gains[i, 1]
                if not (np.isfinite(g1) and np.isfinite(g2) and np.isfinite(lo[i])):
                    continue
                key = (int(round(lo[i])), int(round(g1)), int(round(g2)))
                if key not in cache:
                    try:
                        cache[key] = float(self._model.predict_phase_offset(
                            frequency_hz=key[0], gain_rx1_db=key[1], gain_rx2_db=key[2],
                            allow_float32_frequency_alias=True))
                    except (UnsupportedPhaseModelInput, ValueError, KeyError):
                        cache[key] = None
                v = cache[key]
                if v is not None:
                    corr[i] = v
                    supported[i] = True
            if self.phase_correction_mode == "constant":
                # one number for the whole receiver: the circular mean of the
                # corrections actually applicable to this capture
                if supported.any():
                    mu = float(np.angle(np.mean(np.exp(1j * corr[supported]))))
                    corr = np.where(supported, mu, 0.0)
            elif self.phase_correction_mode == "shuffled":
                # permute the mapping from gain-state to correction, keeping the
                # multiset of correction values identical
                rng = np.random.default_rng(self._shuffle_seed + ridx)
                uniq = sorted({k for k, v in cache.items() if v is not None})
                vals = np.array([cache[k] for k in uniq], dtype=float)
                perm = rng.permutation(len(uniq))
                remap = {k: float(vals[perm[j]]) for j, k in enumerate(uniq)}
                for i in range(n):
                    g1, g2 = gains[i, 0], gains[i, 1]
                    if not supported[i]:
                        continue
                    corr[i] = remap[(int(round(lo[i])), int(round(g1)), int(round(g2)))]
            out[ridx] = corr
            self._supported_fraction = float(supported.mean())
        return out

    def _corrected_mean_phase(self):
        mp = {}
        for ridx in range(self._ds.n_receivers):
            src = self._ds.mean_phase[f"r{ridx}"]
            corr = torch.as_tensor(self._corr[ridx], dtype=src.dtype)
            # wrap into (-pi, pi]; NaN and inf sentinels pass through untouched
            finite = torch.isfinite(src)
            out = src.clone()
            out[finite] = torch.remainder(
                src[finite] - corr[finite] + torch.pi, 2 * torch.pi) - torch.pi
            mp[f"r{ridx}"] = out
        return mp

    # ---- per-item path (used only when ds.temp_file is set) -----------------
    def __getitem__(self, idx):
        item = self._ds[idx]
        try:
            out = list(item)
        except TypeError:
            return item
        for ridx, entry in enumerate(out):
            if not isinstance(entry, dict) or "mean_phase_segmentation" not in entry:
                continue
            e = dict(entry)                       # copy: never mutate the cache
            v = e["mean_phase_segmentation"]
            c = float(self._corr[ridx][idx]) if idx < len(self._corr[ridx]) else 0.0
            t = torch.as_tensor(v)
            fin = torch.isfinite(t)
            t = t.clone()
            t[fin] = torch.remainder(t[fin] - c + torch.pi, 2 * torch.pi) - torch.pi
            e["mean_phase_segmentation"] = t
            out[ridx] = e
        return out
