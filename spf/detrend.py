import torch

# mostly chatgpt


def detrend_1d(
    x: torch.Tensor,
    remove: bool = True,
    window_size: int = 1024,
    merge_similar: bool = False,
    slope_threshold: float = 1e-3,
    intercept_threshold: float = 1e-2,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
):
    """
    Detrend a 1D real-valued PyTorch tensor by removing a linear component in each segment.

    Parameters
    ----------
    x : torch.Tensor
        1D real tensor of shape (N,).
    remove : bool
        If True, actually remove the linear trend. If False, do nothing (pass x through).
        This allows turning off detrending easily.
    window_size : int
        Size of each fixed window.
    merge_similar : bool
        If True, attempt to merge adjacent windows if their slope/intercept are within thresholds.
    slope_threshold : float
        Max allowed slope difference for merging adjacent windows.
    intercept_threshold : float
        Max allowed intercept difference for merging adjacent windows.
    pad_mode : str
        Padding mode if length(x) isn't divisible by window_size.
        Options: ['constant', 'reflect', 'replicate']. Default 'constant'.
    pad_value : float
        Value used for 'constant' padding. Default 0.0.

    Returns
    -------
    x_detrended : torch.Tensor
        1D real tensor, same shape as x, with linear trends removed if remove=True.
    slopes : torch.Tensor
        Slopes for each final segment after merges. Shape: (num_segments,).
    intercepts : torch.Tensor
        Intercepts for each final segment. Shape: (num_segments,).
    """
    # Early-out if we are not removing anything
    if remove is False:
        return x.clone(), torch.tensor([]), torch.tensor([])

    if x.dim() != 1:
        raise ValueError("x must be 1D")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not x.is_floating_point():
        raise ValueError("x must be a floating-point (real) tensor")

    device = x.device
    dtype = x.dtype
    N = x.shape[0]

    # 1. Segment into windows (with optional padding)
    def segment_1d(signal: torch.Tensor, wsize: int):
        length = signal.shape[0]
        num_wins = (length + wsize - 1) // wsize  # ceiling
        total_req = num_wins * wsize
        pad_len = total_req - length
        if pad_len > 0:
            if pad_mode == "constant":
                pad_vals = torch.full((pad_len,), pad_value, dtype=dtype, device=device)
            elif pad_mode in ["reflect", "replicate"]:
                if length == 0:
                    raise ValueError("Cannot pad empty tensor with reflect/replicate")
                if pad_mode == "reflect":
                    pad_vals = signal[-pad_len:].flip(0)
                else:  # replicate
                    pad_vals = signal[-1:].repeat(pad_len)
            else:
                raise ValueError(f"Unsupported pad_mode={pad_mode}")
            padded = torch.cat((signal, pad_vals), dim=0)
        else:
            padded = signal
            pad_len = 0
        x_wins = padded.view(num_wins, wsize)
        return x_wins, pad_len

    windows, pad_len = segment_1d(x, window_size)
    num_wins = windows.shape[0]

    # 2. Fit slope/intercept for each window
    def fit_slopes_intercepts(win: torch.Tensor):
        """
        Given shape=(nw, wsize), fit y(t)=m*t + b in batch.
        Return slopes, intercepts, fitted_trend
        """
        nw, wsize = win.shape
        tvec = (
            torch.linspace(0, 1, steps=wsize, device=device).unsqueeze(0).repeat(nw, 1)
        )
        # X shape (nw, wsize, 2)
        X = torch.stack((tvec, torch.ones_like(tvec)), dim=2)
        y = win.unsqueeze(2)  # (nw, wsize, 1)
        sol = torch.linalg.lstsq(X, y).solution.squeeze(2)  # shape (nw, 2)
        slopes_ = sol[:, 0]
        intercepts_ = sol[:, 1]
        # Reconstruct
        trend = (X @ sol.unsqueeze(-1)).squeeze(-1)
        return slopes_, intercepts_, trend

    s_all, i_all, trend_all = fit_slopes_intercepts(windows)

    # 3. Build segments list
    segments = []
    for idx in range(num_wins):
        seg = {
            "idx": idx,
            "data": windows[idx],
            "length": window_size,
            "slope": s_all[idx].item(),
            "intercept": i_all[idx].item(),
        }
        segments.append(seg)

    # 4. (Optional) Merge similar windows
    def can_merge(segA, segB):
        dslope = abs(segA["slope"] - segB["slope"])
        dinter = abs(segA["intercept"] - segB["intercept"])
        return (dslope <= slope_threshold) and (dinter <= intercept_threshold)

    def merge_two(segA, segB):
        combined_data = torch.cat((segA["data"], segB["data"]), dim=0)
        combined_len = segA["length"] + segB["length"]
        # Re-fit
        w_reshaped = combined_data.unsqueeze(0)
        s_, i_, _ = fit_slopes_intercepts(w_reshaped)
        segA["data"] = combined_data
        segA["length"] = combined_len
        segA["slope"] = s_[0].item()
        segA["intercept"] = i_[0].item()
        return segA

    if merge_similar:
        merged_list = []
        i = 0
        while i < len(segments):
            if i == len(segments) - 1:
                merged_list.append(segments[i])
                i += 1
            else:
                segA = segments[i]
                segB = segments[i + 1]
                if can_merge(segA, segB):
                    merged_seg = merge_two(segA, segB)
                    merged_list.append(merged_seg)
                    i += 2
                else:
                    merged_list.append(segA)
                    i += 1
        segments = merged_list

    # 5. Final pass to re-fit after merges
    #    (If you want multiple merge passes, you can repeat until stable. We do single pass.)
    final_segments = []
    for seg in segments:
        w_reshaped = seg["data"].unsqueeze(0)
        s_, i_, _ = fit_slopes_intercepts(w_reshaped)
        seg["slope"] = s_[0].item()
        seg["intercept"] = i_[0].item()
        final_segments.append(seg)

    # 6. Build final detrended output
    #    We'll accumulate each final segment's detrended data
    all_pieces = []
    for seg in final_segments:
        length = seg["length"]
        tlocal = torch.linspace(0, 1, steps=length, device=device)
        fit_line = seg["slope"] * tlocal + seg["intercept"]
        detrended = seg["data"] - fit_line
        all_pieces.append(detrended)

    x_detrended = torch.cat(all_pieces, dim=0)
    # remove padding
    if pad_len > 0 and x_detrended.shape[0] > N:
        x_detrended = x_detrended[:N]

    # prepare slopes/intercepts as final arrays
    slope_list = []
    intercept_list = []
    for seg in final_segments:
        slope_list.append(seg["slope"])
        intercept_list.append(seg["intercept"])

    slopes_t = torch.tensor(slope_list, dtype=torch.float32, device=device)
    intercepts_t = torch.tensor(intercept_list, dtype=torch.float32, device=device)

    return x_detrended, slopes_t, intercepts_t


def merge_dynamic_windows(
    x: torch.Tensor,
    remove_real: bool = False,
    remove_imag: bool = True,
    window_size: int = 1024,
    merge_similar: bool = False,
    slope_threshold_I: float = 1e-3,
    intercept_threshold_I: float = 1e-2,
    slope_threshold_Q: float = 1e-3,
    intercept_threshold_Q: float = 1e-2,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
):
    """
    Detrend a 1D complex tensor by separately calling `detrend_1d` on the real and imaginary parts.

    This allows, for example, removing a linear trend only from the imaginary part but leaving
    the real part untouched if remove_real=False.

    Parameters
    ----------
    x : torch.Tensor
        1D complex tensor of shape (N,).
    remove_real : bool
        If True, remove linear trend from the real part. If False, leave real part as-is.
    remove_imag : bool
        If True, remove linear trend from the imaginary part. If False, leave imaginary part as-is.
    window_size : int
        The base window size for segmentation.
    merge_similar : bool
        If True, merges adjacent windows with similar slope/intercept in their respective channels.
        Note: We might use separate thresholds for real vs. imaginary.
    slope_threshold_I : float
        Real-part slope difference threshold for merging.
    intercept_threshold_I : float
        Real-part intercept difference threshold for merging.
    slope_threshold_Q : float
        Imag-part slope difference threshold for merging.
    intercept_threshold_Q : float
        Imag-part intercept difference threshold for merging.
    pad_mode : str
        Padding mode for segmenting the real and imaginary parts.
    pad_value : float
        Padding value for constant mode.

    Returns
    -------
    x_detrended : torch.Tensor
        1D complex tensor, same shape as x, with the specified detrending applied to real/imag parts.
    slopes_real : torch.Tensor
        Slopes of the real part's final segments if remove_real=True, else empty.
    intercepts_real : torch.Tensor
        Intercepts of the real part's final segments if remove_real=True, else empty.
    slopes_imag : torch.Tensor
        Slopes of the imag part's final segments if remove_imag=True, else empty.
    intercepts_imag : torch.Tensor
        Intercepts of the imag part's final segments if remove_imag=True, else empty.
    """
    if not torch.is_complex(x):
        raise ValueError("x must be a complex tensor")
    if x.dim() != 1:
        raise ValueError("x must be 1D")

    x_r = x.real
    x_i = x.imag

    # Detrend real part
    r_detrended, r_slopes, r_intercepts = detrend_1d(
        x_r,
        remove=remove_real,
        window_size=window_size,
        merge_similar=merge_similar,
        slope_threshold=slope_threshold_I,
        intercept_threshold=intercept_threshold_I,
        pad_mode=pad_mode,
        pad_value=pad_value,
    )

    # Detrend imaginary part
    i_detrended, i_slopes, i_intercepts = detrend_1d(
        x_i,
        remove=remove_imag,
        window_size=window_size,
        merge_similar=merge_similar,
        slope_threshold=slope_threshold_Q,
        intercept_threshold=intercept_threshold_Q,
        pad_mode=pad_mode,
        pad_value=pad_value,
    )

    # Recombine into a single complex 1D tensor
    x_detrended = torch.complex(r_detrended, i_detrended)

    return x_detrended, r_slopes, r_intercepts, i_slopes, i_intercepts


import numpy as np


def detrend_1d_np(
    x: np.ndarray,
    remove: bool = True,
    window_size: int = 1024,
    merge_similar: bool = False,
    slope_threshold: float = 1e-3,
    intercept_threshold: float = 1e-2,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
):
    """
    Detrend a 1D real-valued NumPy array by removing a linear component in each segment.

    Parameters
    ----------
    x : np.ndarray
        1D real array of shape (N,).
    remove : bool
        If True, subtract the linear trend from each window. If False, do nothing (output x unchanged).
    window_size : int
        Size of each base window for segmentation.
    merge_similar : bool
        If True, merge adjacent windows whose slope/intercept are within thresholds.
    slope_threshold : float
        Max allowed slope difference for merging adjacent windows.
    intercept_threshold : float
        Max allowed intercept difference for merging adjacent windows.
    pad_mode : str
        How to pad x if x.size isn't divisible by window_size. Choices:
        ['constant', 'reflect', 'replicate']. (Replicate is implemented by repeating
        the last element, 'reflect' by reversing a slice.)
    pad_value : float
        Value used if pad_mode='constant'.

    Returns
    -------
    x_detrended : np.ndarray
        1D real array, same shape as x, with linear trends removed (if remove=True).
    slopes : np.ndarray
        Slopes for each final segment. Shape (num_segments,).
    intercepts : np.ndarray
        Intercepts for each final segment. Shape (num_segments,).
    """
    x = np.asanyarray(x, dtype=float)  # ensure float
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    N = x.size

    # If not removing, simply return x as-is plus empty slopes/intercepts
    if not remove:
        return x.copy(), np.array([]), np.array([])

    # 1. Segment the array into windows (optionally pad)
    def segment_1d(data: np.ndarray, wsize: int):
        length = data.size
        num_wins = (length + wsize - 1) // wsize  # ceiling
        total_req = num_wins * wsize
        pad_len = total_req - length
        if pad_len > 0:
            if pad_mode == "constant":
                pad_vals = np.full((pad_len,), pad_value, dtype=data.dtype)
            elif pad_mode in ["reflect", "replicate"]:
                if length == 0:
                    raise ValueError(
                        "Cannot pad an empty array with reflect/replicate."
                    )
                if pad_mode == "reflect":
                    pad_vals = data[-pad_len:][::-1]
                else:  # replicate
                    pad_vals = np.repeat(data[-1], pad_len)
            else:
                raise ValueError(f"Unsupported pad_mode={pad_mode}")

            data_padded = np.concatenate((data, pad_vals), axis=0)
        else:
            data_padded = data
            pad_len = 0
        # Reshape
        x_wins = data_padded.reshape((-1, wsize))
        return x_wins, pad_len

    windows, pad_len = segment_1d(x, window_size)
    num_wins = windows.shape[0]

    # 2. Fit slope/intercept for each window
    def fit_slopes_intercepts_np(wins: np.ndarray):
        """
        Given shape=(nw, wsize), fit y(t) = slope*t + intercept for each row (in batch).
        Return slopes, intercepts, fitted trends.
        """
        nw, wsize = wins.shape
        # Create t in [0,1]
        tvec = np.linspace(0, 1, wsize)[None, :]  # shape (1,wsize)
        tvec = np.repeat(tvec, nw, axis=0)  # shape (nw,wsize)
        # Build design matrix X: shape (nw, wsize, 2)
        # We'll flatten each row for np.linalg.lstsq
        # But let's do this row-by-row in a loop for simplicity.
        slopes_out = np.zeros(nw, dtype=float)
        intercepts_out = np.zeros(nw, dtype=float)
        trend_out = np.zeros_like(wins)

        for i in range(nw):
            # A = np.stack([tvec[i], np.ones(wsize)], axis=1) # shape (wsize, 2)
            A = np.column_stack((tvec[i], np.ones(wsize)))
            y = wins[i]  # shape (wsize,)
            sol, residuals, rank, svals = np.linalg.lstsq(A, y, rcond=None)
            slope_i, intercept_i = sol
            slopes_out[i] = slope_i
            intercepts_out[i] = intercept_i
            # reconstruct fitted line
            fitted = A @ sol
            trend_out[i] = fitted
        return slopes_out, intercepts_out, trend_out

    s_all, i_all, trends_all = fit_slopes_intercepts_np(windows)

    # 3. Build a list of segments
    segments = []
    for idx in range(num_wins):
        seg = {
            "idx": idx,
            "data": windows[idx],  # shape (window_size,)
            "length": window_size,
            "slope": s_all[idx],
            "intercept": i_all[idx],
        }
        segments.append(seg)

    # 4. (Optional) Merge segments if merge_similar is True
    def can_merge(segA, segB):
        dslope = abs(segA["slope"] - segB["slope"])
        dinter = abs(segA["intercept"] - segB["intercept"])
        return (dslope <= slope_threshold) and (dinter <= intercept_threshold)

    def merge_two(segA, segB):
        # Combine data, re-fit
        combined_data = np.concatenate((segA["data"], segB["data"]), axis=0)
        combined_len = segA["length"] + segB["length"]
        # Re-fit slope
        # Fit with np.linalg.lstsq
        wsize_comb = combined_len
        tloc = np.linspace(0, 1, wsize_comb)
        A = np.column_stack((tloc, np.ones(wsize_comb)))
        sol, _, _, _ = np.linalg.lstsq(A, combined_data, rcond=None)
        slope_c, intercept_c = sol
        segA["data"] = combined_data
        segA["length"] = combined_len
        segA["slope"] = slope_c
        segA["intercept"] = intercept_c
        return segA

    if merge_similar:
        merged_list = []
        i = 0
        while i < len(segments):
            if i == len(segments) - 1:
                merged_list.append(segments[i])
                i += 1
            else:
                segA = segments[i]
                segB = segments[i + 1]
                if can_merge(segA, segB):
                    merged_seg = merge_two(segA, segB)
                    merged_list.append(merged_seg)
                    i += 2
                else:
                    merged_list.append(segA)
                    i += 1
        segments = merged_list

    # re-fit after merges (single pass)
    final_segments = []
    for seg in segments:
        data_ = seg["data"]
        length_ = data_.size
        tloc = np.linspace(0, 1, length_)
        A = np.column_stack((tloc, np.ones(length_)))
        sol, _, _, _ = np.linalg.lstsq(A, data_, rcond=None)
        slope_f, intercept_f = sol
        seg["slope"] = slope_f
        seg["intercept"] = intercept_f
        final_segments.append(seg)

    # 5. Build final detrended array
    all_parts = []
    for seg in final_segments:
        length_ = seg["length"]
        tloc = np.linspace(0, 1, length_)
        fitted = seg["slope"] * tloc + seg["intercept"]
        d_ = seg["data"] - fitted
        all_parts.append(d_)

    x_detrended = np.concatenate(all_parts, axis=0)
    # remove padding
    if pad_len > 0 and x_detrended.size > N:
        x_detrended = x_detrended[:N]

    slopes_list = []
    intercepts_list = []
    for seg in final_segments:
        slopes_list.append(seg["slope"])
        intercepts_list.append(seg["intercept"])
    slopes_arr = np.array(slopes_list, dtype=float)
    intercepts_arr = np.array(intercepts_list, dtype=float)

    return x_detrended, slopes_arr, intercepts_arr


def merge_dynamic_windows_np(
    x: np.ndarray,
    remove_real: bool = False,
    remove_imag: bool = True,
    window_size: int = 1024,
    merge_similar: bool = False,
    slope_threshold_I: float = 1e-3,
    intercept_threshold_I: float = 1e-2,
    slope_threshold_Q: float = 1e-3,
    intercept_threshold_Q: float = 1e-2,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
):
    """
    Detrend a 1D complex NumPy array by separately calling detrend_1d_np on the real and imaginary parts.

    This allows, for example, removing a linear trend only from the imaginary part but leaving
    the real part untouched if remove_real=False.

    Parameters
    ----------
    x : np.ndarray
        1D complex array of shape (N,).
    remove_real : bool
        If True, remove linear trend from the real part. If False, leave real part as-is.
    remove_imag : bool
        If True, remove linear trend from the imaginary part. If False, leave imaginary part as-is.
    window_size : int
        The base window size for segmentation.
    merge_similar : bool
        If True, merges adjacent windows with similar slope/intercept for each part.
    slope_threshold_I : float
        Real-part slope difference threshold for merging.
    intercept_threshold_I : float
        Real-part intercept difference threshold for merging.
    slope_threshold_Q : float
        Imag-part slope difference threshold for merging.
    intercept_threshold_Q : float
        Imag-part intercept difference threshold for merging.
    pad_mode : str
        Padding mode for segmenting the real and imaginary parts.
    pad_value : float
        Padding value for constant mode.

    Returns
    -------
    x_detrended : np.ndarray
        1D complex array, same shape as x, with the specified detrending applied to real/imag parts.
    slopes_real : np.ndarray
        Slopes of the real part's final segments if remove_real=True, else empty array.
    intercepts_real : np.ndarray
        Intercepts of the real part's final segments if remove_real=True, else empty array.
    slopes_imag : np.ndarray
        Slopes of the imag part's final segments if remove_imag=True, else empty array.
    intercepts_imag : np.ndarray
        Intercepts of the imag part's final segments if remove_imag=True, else empty array.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if not np.iscomplexobj(x):
        raise ValueError("x must be a complex NumPy array.")

    # Real and imaginary parts
    real_part = x.real
    imag_part = x.imag

    # Detrend real part
    r_detrended, r_slopes, r_inters = detrend_1d_np(
        real_part,
        remove=remove_real,
        window_size=window_size,
        merge_similar=merge_similar,
        slope_threshold=slope_threshold_I,
        intercept_threshold=intercept_threshold_I,
        pad_mode=pad_mode,
        pad_value=pad_value,
    )

    # Detrend imaginary part
    i_detrended, i_slopes, i_inters = detrend_1d_np(
        imag_part,
        remove=remove_imag,
        window_size=window_size,
        merge_similar=merge_similar,
        slope_threshold=slope_threshold_Q,
        intercept_threshold=intercept_threshold_Q,
        pad_mode=pad_mode,
        pad_value=pad_value,
    )

    # Recombine into a single complex array
    x_detrended = r_detrended + 1j * i_detrended

    return x_detrended, r_slopes, r_inters, i_slopes, i_inters


def detrend_np(v):
    v0, _, _, _, _ = merge_dynamic_windows_np(
        v[0],
        remove_real=True,
        remove_imag=True,
        window_size=1024,
        merge_similar=True,
        slope_threshold_I=1e-3,
        intercept_threshold_I=1e-2,
        slope_threshold_Q=1e-3,
        intercept_threshold_Q=1e-2,
    )
    v1, _, _, _, _ = merge_dynamic_windows_np(
        v[1],
        remove_real=True,
        remove_imag=True,
        window_size=1024,
        merge_similar=True,
        slope_threshold_I=1e-3,
        intercept_threshold_I=1e-2,
        slope_threshold_Q=1e-3,
        intercept_threshold_Q=1e-2,
    )
    return np.vstack([v0, v1])
