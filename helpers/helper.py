################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Training loop utilities, metric computation, logging, and visualisation helpers.
"""

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from scipy.signal import  butter, filtfilt, detrend, stft 
from ssqueezepy import (
    ssq_stft,
    ssq_cwt,
    cwt,
)
from torch import sqrt, mean
from scipy.ndimage import uniform_filter1d

import numpy as np
import matplotlib.pyplot as plt

from fnmatch import fnmatch
import os
import json
import time
import pandas as pd
from tqdm import tqdm

def get_scores(y_true, y_pred):
    """
    Calculates regression metrics (RMSE, R2, MAE, MAPE).

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        tuple: (rmse, r2, mae, mape)
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, r2, mae, mape


def root_mse(output, target):
    """
    Calculates PyTorch Root Mean Squared Error.

    Args:
        output (torch.Tensor): Output tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: RMSE value.
    """
    return sqrt(mean((output - target) ** 2))


def get_file_list(root=".", pattern="*.*"):
    """
    Recursively finds files matching a pattern.

    Args:
        root (str): Root directory to search. Defaults to '.'.
        pattern (str): Glob pattern to match. Defaults to '*.*'.

    Returns:
        list: List of matching file paths.
    """
    file_list = []
    for path, _, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                file_list.append(os.path.join(path, name))
    return file_list


def apply_rms(x, window_size=10):
    """
    Apply Root Mean Square envelope to the signal.

    Args:
        x (np.ndarray): Input signal. Shape (N, L, C) or (L, C).
        window_size (int): Windows size for the moving average used in RMS. Defaults to 10.

    Returns:
        np.ndarray: RMS signal.
    """
    x_squared = np.power(x, 2)
    x_mean_squared = apply_moving_average(x_squared, window_size=window_size)
    x_rms = np.sqrt(x_mean_squared)
    return x_rms


def apply_high_pass_filter(x, cutoff=0.05, fs=1.0, order=5):
    """
    Apply a high-pass Butterworth filter to the signals.

    Args:
        x (np.ndarray): Input signal. Shape (N, Time, Channels) or (Time, Channels).
        cutoff (float): Cutoff frequency (normalized if fs=1.0, else in Hz). Defaults to 0.05.
        fs (float): Sampling frequency. Defaults to 1.0.
        order (int): Filter order. Defaults to 5.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)

    axis = 1 if x.ndim == 3 else 0
    return filtfilt(b, a, x, axis=axis)


def apply_low_pass_filter(x, cutoff=0.15, fs=1.0, order=5):
    """
    Apply a low-pass Butterworth filter to the signals.

    Args:
        x (np.ndarray): Input signal. Shape (N, Time, Channels) or (Time, Channels).
        cutoff (float): Cutoff frequency. Defaults to 0.15.
        fs (float): Sampling frequency. Defaults to 1.0.
        order (int): Filter order. Defaults to 5.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    axis = 1 if x.ndim == 3 else 0
    return filtfilt(b, a, x, axis=axis)


def apply_moving_average(x, window_size=10):
    """
    Apply moving average filter along the time axis.

    Args:
        x (np.ndarray): Input signal. Shape (N, Time, Channels) or (Time, Channels)
        window_size (int): Size of the moving average window. Defaults to 10.

    Returns:
        np.ndarray: Smoothed signal.
    """
    axis = 1 if x.ndim == 3 else 0

    return uniform_filter1d(x, size=window_size, axis=axis, mode="nearest")

def augment_signal(
    x, x_process_params, y, window_size=1000, stride=1, description="Augmenting signals"
):
    """
    Augments the signal by creating sliding window segments.

    Args:
        x (np.ndarray): Input signals (list of signals or array).
        x_process_params (np.ndarray): Process parameters corresponding to x.
        y (np.ndarray): Targets corresponding to x.
        window_size (int): Size of the sliding window. Defaults to 1000.
        stride (int): Stride of the sliding window. Defaults to 1.

    Returns:
        tuple: (inputs, params, labels) as numpy arrays.
    """
    inputs = []
    labels = []
    params = []
    for i in tqdm(range(len(x)), desc=description):
        data = np.array(x[i])
        for j in range(0, data.shape[0] - window_size + 1, stride):
            inputs.append(data[j : j + window_size, :])
            labels.append(y[i])
            params.append(x_process_params[i])
    inputs = np.array(inputs)
    labels = np.array(labels)
    params = np.array(params)
    return inputs, params, labels

def plot_signals(x, signal_list, signal_chanel=1):
    """
    Plots specific channels from the input data.

    Args:
        x (np.ndarray): Input signal batch.
        signal_list (list): List of indices to plot.
        signal_chanel (int): Mode of plotting.
    """
    print(f"Plotting x with shape: {x[0].shape}")
    
    num_subplots = max(min(len(signal_list), 30), 4)
    print(f"Subplots: {num_subplots} - Rows: {(num_subplots + 2) // 3}")

    fig, axes = plt.subplots(
        (num_subplots + 2) // 3, 3, figsize=(12, 3 * ((num_subplots + 2) // 3))
    )

    for i in range(num_subplots):
        row, col = divmod(i, 3)
        if signal_chanel == 1:
            axes[row, col].plot(x[0][i, :])
        elif signal_chanel == 2:
            axes[row, col].plot(x[0][:, i])
        elif signal_chanel == 3:
            print(f"channel {i} mean", x[0][:, :, i].mean())
            axes[row, col].imshow(x[0][:, :, i])
        axes[row, col].set_title(f"Signal {signal_list[i]}")
        if len(signal_list) - 1 == i:
            break

    fig.suptitle("Sample 1: Individual Signal Plots", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"X_plots.png")
    plt.show()


def convert_to_cwt(
    x,
    fs=250.0,
    wavelet="morlet",
    description="CWT generation",
    device="cpu",
    scales="log-piecewise",
    nv=16,
):
    """
    Converts signals to Continuous Wavelet Transform (CWT).

    Args:
        x (np.ndarray): Input data. Shape (N, W, C).
        fs (float): Sampling frequency in Hz. Must match the effective rate after
            any decimation (raw NASA = 250 Hz; ds=2 → 125 Hz).
        wavelet (str): Wavelet family. Defaults to 'morlet' (analytic Morlet).
            Morlet is chosen on domain grounds: milling signals contain periodic
            tooth-pass impulses in the vibration and AE channels, and the Morlet
            wavelet is structurally similar to such impulse components, making it
            a natural basis for detecting wear-related transients [1].
            Jáuregui et al. validated Morlet specifically for cutting-force and
            vibration TCM, citing its balance between time and frequency
            resolutions [2].
        description (str): Description for progress bar.
        device (str): 'cpu' or 'gpu'.
        scales (str): Scale distribution. 'log-piecewise' is recommended [3]: it
            uses logarithmic spacing with coarsening at very high scales to avoid
            redundancy while retaining coverage of the full frequency range.
        nv (int): Voices per octave (frequency bins per doubling). nv=16 is the
            documented minimum for reliable TF localisation [3]. nv=8 (previous
            value) was a memory shortcut that under-resolved closely spaced
            frequency components; at ds=2, w=500 each window is only ~750 kB so
            the memory argument no longer applies.

    Returns:
        np.ndarray: CWT magnitude array. Shape (N, Scales, W, C).
    """
    if device == "gpu":
        os.environ["SSQ_GPU"] = "1"
    else:
        os.environ["SSQ_GPU"] = "0"

    Wx, _ = cwt(x[0, :, 0], wavelet=wavelet, scales=scales, nv=nv, fs=fs)
    n_scales_actual = Wx.shape[0]

    cwt_out = np.empty(
        (x.shape[0], n_scales_actual, x.shape[1], x.shape[2]), dtype=np.float32
    )

    for c in tqdm(range(x.shape[2]), desc=description):
        Wx, _ = cwt(x[:, :, c], wavelet=wavelet, scales=scales, nv=nv, fs=fs)
        cwt_out[:, :, :, c] = np.abs(Wx)

    return cwt_out


def convert_to_fsst(
    x,
    fs=250.0,
    n_fft=64,
    win_len=None,
    description="FSST generation",
    device="cpu",
):
    """
    Apply Fourier Synchrosqueezed Transform (FSST) to signals.

    Args:
        x (np.ndarray): Input data. Shape (N, W, C).
        fs (float): Sampling frequency in Hz. Must match effective rate after
            decimation (raw NASA = 250 Hz; ds=2 → 125 Hz).
        n_fft (int): FFT length, controlling the number of frequency bins
            (F = n_fft//2 + 1 = 33 bins). Defaults to 64.
            Rationale: the milling spindle motor (smcAC/smcDC) dominates at
            low frequencies (0–10 Hz); vibration channels carry energy up to
            ~50 Hz; AE is broadband but most discriminative wear-related content
            is below 40 Hz at 125 Hz effective rate. n_fft=64 gives 33 bins
            spanning 0–62.5 Hz with 1.95 Hz/bin resolution — sufficient. The
            previous default (n_fft=None → 500) produced 251 bins, most above
            the informative band, at 8× the compute cost.
            ssq_stft hop_len is hardcoded to 1 (required for synchrosqueezing
            invertibility [1]), so T always equals the window length regardless
            of n_fft. Limiting n_fft is the only way to reduce the output size.
        win_len (int / None): Length of the analysis window. Defaults to
            n_fft//8 (ssqueezepy default), i.e. 8 samples = 64 ms at 125 Hz.
            A short window gives good time localisation at the cost of frequency
            resolution — appropriate for non-stationary spindle/AE bursts [2].
        description (str): Description for progress bar.
        device (str): 'cpu' or 'gpu'.

    Returns:
        np.ndarray: FSST magnitude array. Shape (N, F, T, C).
    """
    if device == "gpu":
        os.environ["SSQ_GPU"] = "1"
    else:
        os.environ["SSQ_GPU"] = "0"

    Tx, *_ = ssq_stft(x[0, :, 0], fs=fs, n_fft=n_fft, win_len=win_len)
    F, T = Tx.shape

    fsst_out = np.empty((x.shape[0], F, T, x.shape[2]), dtype=np.float32)

    for c in tqdm(range(x.shape[2]), desc=description):
        Tx, *_ = ssq_stft(x[:, :, c], fs=fs, n_fft=n_fft, win_len=win_len)
        fsst_out[:, :, :, c] = np.abs(Tx)

    return fsst_out


def convert_to_wsst(
    x,
    fs=250.0,
    wavelet="morlet",
    scales="log-piecewise",
    nv=16,
    description="WSST generation",
    device="cpu",
):
    """
    Apply Wavelet Synchrosqueezed Transform (WSST / SSQ-CWT) to signals.

    Args:
        x (np.ndarray): Input data. Shape (N, W, C).
        fs (float): Sampling frequency in Hz. Must match effective rate after
            decimation (raw NASA = 250 Hz; ds=2 → 125 Hz). Passed to ssq_cwt
            so that returned frequencies are in physical Hz, not normalised units.
        wavelet (str): Wavelet family. Defaults to 'morlet' — same rationale as
            convert_to_cwt: impulse similarity and validated TF balance for
            milling TCM [1, 2].
        scales (str): Scale distribution. 'log-piecewise' recommended [3].
        nv (int): Voices per octave. nv=16 is the documented minimum [3].
            Previous value of 8 was a memory shortcut; no longer justified at
            ds=2, w=500 window sizes.
        description (str): Description for progress bar.
        device (str): 'cpu' or 'gpu'.

    Returns:
        np.ndarray: WSST magnitude array. Shape (N, F, T, C).
    """
    if device == "gpu":
        os.environ["SSQ_GPU"] = "1"
    else:
        os.environ["SSQ_GPU"] = "0"

    Tx, *_ = ssq_cwt(x[0, :, 0], wavelet=wavelet, scales=scales, fs=fs, nv=nv)
    F, T = Tx.shape

    wsst_out = np.empty((x.shape[0], F, T, x.shape[2]), dtype=np.float32)

    for c in tqdm(range(x.shape[2]), desc=description):
        Tx, *_ = ssq_cwt(x[:, :, c], wavelet=wavelet, scales=scales, fs=fs, nv=nv)
        wsst_out[:, :, :, c] = np.abs(Tx)

    return wsst_out


def convert_to_stft(
    x,
    fs=250.0,
    nperseg=64,
    noverlap=32,
    description="STFT generation",
    device="cpu",  # Kept for signature compatibility; scipy.signal.stft runs on CPU
):
    """
    Apply Short-Time Fourier Transform (STFT) to signals.

    Uses scipy.signal.stft (imported at module level).

    Args:
        x (np.ndarray): Input data. Shape (N, W, C).
        fs (float): Sampling frequency in Hz. Must match effective rate after
            decimation (raw NASA = 250 Hz; ds=2 → 125 Hz).
        nperseg (int): Analysis window length in samples. Defaults to 64
            (512 ms at 125 Hz). Chosen to capture at least 2–3 cycles of the
            lowest informative frequency (~2 Hz tooth-pass at typical feed rates)
            while maintaining adequate time resolution for non-stationary wear
            transients. A 64-sample window gives F = 33 frequency bins spanning
            0–62.5 Hz with 1.95 Hz/bin resolution [1].
        noverlap (int): Samples of overlap between consecutive frames. Defaults
            to 32 (50% overlap). 50% overlap with a Hann window satisfies the
            COLA (Constant Overlap-Add) condition, ensuring uniform weighting
            across the signal [2]. For a 500-sample window this yields T ≈ 15
            time frames vs 500 with the previous noverlap=127 (hop=1) — the
            earlier setting was maximally redundant and added no information.
        description (str): Description for progress bar.
        device (str): Unused; kept for API consistency with other convert_to_*.

    Returns:
        np.ndarray: STFT magnitude array. Shape (N, F, T, C).
    """
    f, t, Zxx = stft(x[0, :, 0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    F = f.shape[0]
    T = t.shape[0]

    stft_out = np.empty((x.shape[0], F, T, x.shape[2]), dtype=np.float32)

    for c in tqdm(range(x.shape[2]), desc=description):
        f, t, Zxx = stft(x[:, :, c], fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
        stft_out[:, :, :, c] = np.abs(Zxx)

    return stft_out


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        """Serialize numpy float32/float64 values to Python float for JSON compatibility."""
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def evaluate_model(
    model,
    x,
    y,
    folder,
    run_name,
    history=None,
    training_time=None,
    model_results=None,
    y_min=0,
    y_max=1,
    norm_target=False,
    clip_min=0.0,
    clip_max=0.45,
    prediction_level="window",
    save_model=True,
):
    """
    Evaluates model, generates plots, and saves results/history to JSON.

    Args:
        model: Keras model to evaluate.
        x (np.ndarray): Input features.  For run-level models this is the
            padded array ``(N_runs, N_r_max, L, C)``; ``model.predict`` handles
            both cases identically.
        y (np.ndarray): Target labels.  Shape ``(N_windows,)`` for window-level
            or ``(N_runs,)`` for run-level.
        folder (str): Output folder for results.
        run_name (str): Name of the run.
        history: Training history object or dict.
        training_time (float): Training time in seconds.
        model_results (dict): Dictionary to append results to.
        y_min (float): Minimum target value (for denormalization).
        y_max (float): Maximum target value (for denormalization).
        norm_target (bool): Whether targets were normalized.
        clip_min (float, optional): Lower bound for clipping predictions.
        clip_max (float, optional): Upper bound for clipping predictions.
        prediction_level (str): ``"window"`` (default) or ``"run"``.
            Purely informational — stored in the result dict so downstream
            analysis can distinguish the two evaluation granularities.
        save_model (bool): If False, skip saving the ``.keras`` file but still
            save the JSON and plots.  Use False during LOCO calibration folds
            where only epoch statistics are needed, not the model weights.

    Returns:
        dict: Updated model_results dictionary.
    """
    if model_results is None:
        model_results = {}

    start_time = time.time()
    ypred = model.predict(x)
    total_time = time.time() - start_time

    model_results["eval_time"] = total_time
    if training_time is not None:
        model_results["time"] = training_time
        model_results["training_time"] = training_time

    if norm_target:
        ypred_orig = ypred * (y_max - y_min) + y_min
        y_test_orig = y * (y_max - y_min) + y_min
    else:
        ypred_orig = ypred
        y_test_orig = y

    if clip_min is not None and clip_max is not None:
        ypred_orig = np.clip(ypred_orig, clip_min, clip_max)

    rmse, r2, mae, mape = get_scores(y_test_orig, ypred_orig)
    print(f"RMSE: {rmse:.5f}\nR2: {r2:.5f}\nMAE: {mae:.5f}\nMAPE: {mape:.5f}")

    file_name = f"{folder}/DL_{run_name}_{rmse:.4f}.keras"
    if save_model:
        model.save(file_name)
    else:
        file_name = None

    model_results["model_file"] = file_name
    model_results["prediction_level"] = prediction_level
    model_results["r2_score"] = r2
    model_results["rmse"] = rmse
    model_results["mae"] = mae
    model_results["mape"] = mape

    if history is not None:
        hist_dict = history.history if hasattr(history, "history") else history
        model_results["history"] = hist_dict

        plt.figure()
        if "loss" in hist_dict:
            plt.plot(hist_dict["loss"], label="Training")
        if "val_loss" in hist_dict:
            plt.plot(hist_dict["val_loss"], label="Validation")
        plt.title(f"Loss - {run_name}")
        plt.legend()
        plt.savefig(f"{folder}/DL_{run_name}_training_loss.png")
        plt.close()

    results_df = pd.DataFrame(
        {"Ground truth": np.squeeze(y_test_orig), "Prediction": np.squeeze(ypred_orig)}
    )
    results_df = results_df.sort_values(by=["Ground truth"], ignore_index=True)

    plt.figure()
    plt.plot(results_df["Prediction"], label="Prediction")
    plt.plot(results_df["Ground truth"], label="Ground truth")
    plt.legend(loc="upper left")
    plt.title(f"Preds - {run_name}")
    plt.savefig(f"{folder}/DL_{run_name}_test_pred.png")
    plt.close()

    json_path = f"{folder}/DL_{run_name}.json"
    with open(json_path, "w") as outfile:
        outfile.write(json.dumps(model_results, indent=4, cls=NumpyFloatValuesEncoder))

    return model_results


__all__ = [
    "get_scores",
    "root_mse",
    "get_file_list",
    "apply_moving_average",
    "augment_signal",
    "plot_signals",
    "convert_to_cwt",
    "convert_to_fsst",
    "convert_to_wsst",
    "apply_high_pass_filter",
    "apply_low_pass_filter",
    "apply_rms",
    "evaluate_model",
    "NumpyFloatValuesEncoder",
]
