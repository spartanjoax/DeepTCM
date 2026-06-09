################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Signal preprocessing pipeline for raw TCM sensor data.

Main entry point: apply_full_preprocessing().
"""
import numpy as np
from helpers.helper import (
    apply_moving_average,
    apply_high_pass_filter,
    apply_low_pass_filter,
    apply_rms,
    augment_signal,
    convert_to_cwt,
    convert_to_fsst,
    convert_to_wsst,
    convert_to_stft,
)


def apply_signal_preprocessing(
    sig,
    noise_reduction="none",
    ma_window_size=10,
):
    """
    Applies signal level operations: Winsorization, Detrending, and Filtering.

    Args:
        sig (np.ndarray): Signal to process.
        noise_reduction (str): Filter type.
        ma_window_size (int): Window size for MA or RMSE.

    Returns:
        np.ndarray: processed_signal
    """
    curr_sig = sig

    if noise_reduction == "moving_average":
        curr_sig = apply_moving_average(curr_sig, window_size=ma_window_size)
    elif noise_reduction == "high_pass":
        curr_sig = apply_high_pass_filter(curr_sig)
    elif noise_reduction == "low_pass":
        curr_sig = apply_low_pass_filter(curr_sig)
    elif noise_reduction == "rms":
        curr_sig = apply_rms(curr_sig, window_size=10)

    return curr_sig


def apply_full_preprocessing(
    x,
    y,
    hps,
    split="train",
    preproc_stats=None,
    sliding_window_config=None,
    undersample_to=None,
):
    """
    Unified function to apply the full preprocessing pipeline:
    1. Signal denoising
    2. Windowing / Augmentation
    3. Scalogram Conversion

    Args:
        x: Input data. Can be signal array or list [signal, info].
        y: Target data.
        hps (dict): Dictionary of hyperparameters (noise_reduction, scalogram, etc.).
        split (str): 'train', 'val', or 'test'. Used to determine if generation should happen.
        preproc_stats (tuple): (p1, p99) tuple for preprocessing statistics. Required if winsorize=True and split!='train'.
        sliding_window_config (dict): Dictionary with 'window_size' and 'stride'.
        undersample_to (int | None): If set, uniformly downsample each run to this many
            time points before windowing.  Applies to **both** flat and run-grouped paths.

    Returns:
        Tuple:
            x_signal   : windowed signal array
            y          : target array
            x_proc     : process-parameter columns — shape (N, P), or None
            out_stats  : preprocessing statistics dict
    """

    noise_reduction = hps.get("noise_reduction", "none")
    scalogram_type = hps.get("scalogram", "none")
    ma_window_size = hps.get("ma_window_size", 10)
    scaler_type = hps.get("scaler_type", "robust")
    downsample_factor = hps.get("downsample_factor", 1)
    
    sw_size = sliding_window_config.get("window_size", 250)
    sw_stride = sliding_window_config.get("stride", 125)

    x_signal = None
    x_info = None

    if isinstance(x, list) and len(x) == 2:
        x_signal = x[0]
        x_info = x[1]
    else:
        x_signal = x

    computed_stats = {}
    x_signal = apply_signal_preprocessing(
        x_signal,
        noise_reduction=noise_reduction,
        ma_window_size=ma_window_size,
    )

    if scaler_type == "minmax":
        if split == "train":
            mm_min = x_signal.min(axis=(0, 1), keepdims=True)   # (1, 1, C)
            mm_max = x_signal.max(axis=(0, 1), keepdims=True)
            mm_range = np.where(mm_max - mm_min > 1e-8, mm_max - mm_min, 1.0)
            computed_stats["mm_min"] = mm_min
            computed_stats["mm_max"] = mm_max
        else:
            mm_min = preproc_stats.get("mm_min", 0.0) if preproc_stats else 0.0
            mm_max = preproc_stats.get("mm_max", 1.0) if preproc_stats else 1.0
            mm_range = np.where(mm_max - mm_min > 1e-8, mm_max - mm_min, 1.0)
        x_signal = (x_signal - mm_min) / mm_range

    if undersample_to is not None and (downsample_factor is None or downsample_factor <= 1):
        run0_len = len(x_signal[0]) if isinstance(x_signal, list) else x_signal.shape[1]
        downsample_factor = max(1, round(run0_len / undersample_to))

    if downsample_factor is not None and downsample_factor > 1:
        if isinstance(x_signal, np.ndarray):
            x_signal = x_signal[:, ::downsample_factor, :]
        else:
            x_signal = [run[::downsample_factor] for run in x_signal]

    if x_info is None:
        x_info_temp = np.zeros((len(x_signal), 1))
    else:
        x_info_temp = x_info

    x_signal_aug, x_info_aug, y_aug = augment_signal(
        x_signal,
        x_info_temp,
        y,
        window_size=sw_size,
        stride=sw_stride,
        description=f"Augmenting ({split})",
    )

    if x_info is None:
        x_info_aug = None

    _jitter = hps.get("jitter_sigma", 0.0)
    if split == "train" and _jitter > 0:
        x_signal_aug = x_signal_aug + np.random.normal(
            0, _jitter, x_signal_aug.shape
        ).astype(x_signal_aug.dtype)

    _fs = 250.0 / max(1, downsample_factor)
    if scalogram_type == "cwt":
        x_signal_aug = convert_to_cwt(x_signal_aug, fs=_fs, description=f"CWT ({split})")
    elif scalogram_type == "fsst":
        x_signal_aug = convert_to_fsst(x_signal_aug, fs=_fs, description=f"FSST ({split})")
    elif scalogram_type == "wsst":
        x_signal_aug = convert_to_wsst(x_signal_aug, fs=_fs, description=f"WSST ({split})")
    elif scalogram_type == "stft":
        x_signal_aug = convert_to_stft(x_signal_aug, fs=_fs, description=f"STFT ({split})")
    elif scalogram_type == "none":
        x_signal_aug = np.expand_dims(x_signal_aug, axis=2)

    if scalogram_type in ["cwt", "fsst", "wsst", "stft"]:
        _have_precomputed = (
            preproc_stats is not None
            and "scalo_mean" in preproc_stats
            and "scalo_std" in preproc_stats
        )
        if split == "train" and not _have_precomputed:
            scalo_mean = np.mean(x_signal_aug, axis=(0, 1, 2), keepdims=True)
            scalo_std = np.std(x_signal_aug, axis=(0, 1, 2), keepdims=True) + 1e-8
        else:
            scalo_mean = preproc_stats.get("scalo_mean", 0.0) if preproc_stats else 0.0
            scalo_std = preproc_stats.get("scalo_std", 1.0) if preproc_stats else 1.0
        computed_stats["scalo_mean"] = scalo_mean
        computed_stats["scalo_std"] = scalo_std

        x_signal_aug = (x_signal_aug - scalo_mean) / scalo_std

    return x_signal_aug, y_aug, x_info_aug, computed_stats
