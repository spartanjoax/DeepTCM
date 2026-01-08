# -*- coding: utf-8 -*-

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.signal import resample, butter, filtfilt, detrend
from ssqueezepy import ssq_stft, ssq_cwt, cwt # https://github.com/AlexanderVNikitin/tsgm/tree/main?tab=readme-ov-file
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

from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import gamma_soft_dtw

from stockwell import st

import warnings
warnings.filterwarnings("ignore")

def get_scores(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, r2, mae, mape

def root_mse(output, target):
    return sqrt(mean((output - target)**2))

def get_file_list(root='.', pattern='*.*'):
    file_list = []
    for path, _, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                file_list.append(os.path.join(path, name))
    return file_list

def apply_detrend(x, type='linear'):
    """
    Remove linear trend.
    x: (N, L, C) or (L, C)
    """
    is_batched = x.ndim == 3
    axis = 1 if is_batched else 0
    return detrend(x, axis=axis, type=type)

def apply_rms(x, window_size=10):
    """
    Apply Root Mean Square envelope to the signal.
    x: (N, L, C) or (L, C)
    """
    # RMS is sqrt(mean(x^2))
    x_squared = np.power(x, 2)
    x_mean_squared = apply_moving_average(x_squared, window_size=window_size)
    x_rms = np.sqrt(x_mean_squared)
    return x_rms
    
def apply_high_pass_filter(x, cutoff=0.05, fs=1.0, order=5):
    """
    Apply a high-pass Butterworth filter to the signals.
    x: (N, Time, Channels) or (Time, Channels)
    cutoff: Cutoff frequency (normalized if fs=1.0, else in Hz)
    fs: Sampling frequency
    order: Filter order
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    x_out = []
     
    for i in tqdm(range(len(x)), desc="Applying High Pass"):
        sample = x[i]
        # filtfilt applies along last axis by default? No, we want time axis (0)
        # sample shape (Time, Channels). We filter along axis 0.
        filtered_sample = filtfilt(b, a, sample, axis=0)
        x_out.append(filtered_sample)
        
    return np.array(x_out)

def apply_low_pass_filter(x, cutoff=0.15, fs=1.0, order=5):
    """
    Apply a low-pass Butterworth filter to the signals.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    x_out = []
    
    for i in tqdm(range(len(x)), desc="Applying Low Pass"):
        sample = x[i]
        filtered_sample = filtfilt(b, a, sample, axis=0)
        x_out.append(filtered_sample)
        
    return np.array(x_out)

def apply_moving_average(x, window_size=10):
    """
    Apply moving average filter along the time axis.
    x: (N, Time, Channels) or (Time, Channels)
    """
    is_batched = x.ndim == 3
    # Time axis is 1 for batched (N, T, C), 0 for unbatched (T, C)
    axis = 1 if is_batched else 0
    
    # uniform_filter1d computes the mean in the window
    return uniform_filter1d(x, size=window_size, axis=axis, mode='nearest')
    
def augment_signal(x, x_process_params, y, window_size=1000, stride=1):
    inputs = []
    labels = []
    params = []
    for i in tqdm(range(len(x))):
        data = np.array(x[i])
        for j in range(0, data.shape[0]-window_size+1, stride):
            inputs.append(data[j:j + window_size, :])
            labels.append(y[i])
            params.append(x_process_params[i])
    inputs = np.array(inputs)
    labels = np.array(labels)
    params = np.array(params)
    return inputs, params, labels

def downsample(signal, to_len):
    down_signal = resample(signal, to_len)
    return down_signal

def plot_signals(x, signal_list, signal_chanel=1):
    print(f'Plotting x with shape: {x[0].shape}')
    # Determine the number of subplots based on the second dimension, at least 2 rows.
    num_subplots = max(min(len(signal_list), 30), 4)
    print(f'Subplots: {num_subplots} - Rows: {(num_subplots + 2) // 3}')

    # Create a figure with appropriate subplots
    fig, axes = plt.subplots((num_subplots + 2) // 3, 3, figsize=(12, 3 * ((num_subplots + 2) // 3)))

    # Plot each channel of the first sample on a separate subplot
    for i in range(num_subplots):
        row, col = divmod(i, 3)
        if signal_chanel == 1:
            axes[row, col].plot(x[0][i, :])
        elif signal_chanel == 2:
            axes[row, col].plot(x[0][:, i])
        elif signal_chanel == 3:
            print(F'channel {i} mean', x[0][:, :, i].mean())
            axes[row, col].imshow(x[0][:, :, i])
        axes[row, col].set_title(f"Signal {signal_list[i]}")
        if len(signal_list) - 1 == i:
            break

    # Adjust layout (optional)
    fig.suptitle("Sample 1: Individual Signal Plots", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"X_plots.png")
    plt.show()

def plot_signals_dict(x, signal_list):
    # Determine the number of subplots based on the second dimension
    num_subplots = min(len(signal_list), 30)
    print(f'Subplots: {num_subplots} - Rows: {(num_subplots + 2) // 3}')

    # Create a figure with appropriate subplots
    fig, axes = plt.subplots((num_subplots + 2) // 3, 3, figsize=(12, 3 * ((num_subplots + 2) // 3)))

    # Plot each channel of the first sample on a separate subplot
    for i in range(num_subplots):
        row, col = divmod(i, 3)
        axes[row, col].plot(x[0][signal_list[i]])
        axes[row, col].set_title(f"Signal {signal_list[i]}")

    # Adjust layout (optional)
    fig.suptitle("Sample 1: Individual Signal Plots", fontsize=12)
    plt.tight_layout()

    plt.show()

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation for each channel in the dataset.

    Args:
        dataset (Dataset): PyTorch Dataset object.

    Returns:
        tuple: mean (np.array), std (np.array) for each channel.
    """
    channel_sums = None
    channel_squared_sums = None
    total_samples = 0

    for sample, _ in dataset:
        # sample shape: (signal_length, num_channels)
        signal_length, num_channels = sample['x'].shape

        if channel_sums is None:
            channel_sums = np.zeros(num_channels)
            channel_squared_sums = np.zeros(num_channels)

        channel_sums += sample['x'].numpy().sum(axis=0)  # Sum across signal_length for each channel
        channel_squared_sums += (sample['x'].numpy() ** 2).sum(axis=0)
        total_samples += signal_length  # Accumulate total signal_length (not samples)

    mean = channel_sums / total_samples
    variance = (channel_squared_sums / total_samples) - (mean ** 2)
    std = np.sqrt(variance)

    return mean, std


def compute_min_max(dataset):
    """
    Compute the minimum and maximum for each feature in the dataset.

    Args:
        dataset (Dataset): PyTorch Dataset object.

    Returns:
        tuple: min (np.array), max (np.array) for each feature.
    """
    all_data = []

    for sample, _ in dataset:
        all_data.append(sample['proc_data'].numpy())  # Convert tensor to NumPy array for easier manipulation

    # Compute min and max for each feature (along the last axis)
    min = np.min(all_data, axis=0)
    max = np.max(all_data, axis=0)

    return min, max

def augment_dataset_softdtw(x, x_proc, y, num_synthetic_ts, max_neighbors=5):     
    # synthetic train set and labels 
    synthetic_x = []
    synthetic_x_proc = []
    synthetic_y = []

    if len(x) == 1 :
        # skip if there is only one time series per set
        print('Cannot create synthetic data with only one sample.')
    else:
        # compute appropriate gamma for softdtw for the entire class
        gamma = gamma_soft_dtw(x)
        
        # loop through the number of synthetic examples needed
        for _ in tqdm(range(num_synthetic_ts)):
            # Choose a random representative for the class
            random_index = np.random.choice(len(x), size=1, replace=False)
            random_representative = x[random_index]
            
            # Choose a random number of neighbours (between 1 and max_neighbors)
            random_number_of_neighbors = int(np.random.uniform(1, max_neighbors, size=1))
            knn = KNeighborsTimeSeries(n_neighbors=random_number_of_neighbors+1, metric='softdtw', metric_params={'gamma': gamma}).fit(x)
            random_neighbor_distances, random_neighbor_indices = knn.kneighbors(X=random_representative, return_distance=True)
            
            random_neighbor_indices = random_neighbor_indices[0]
            random_neighbor_distances = random_neighbor_distances[0]
            nearest_neighbor_distance = np.sort(random_neighbor_distances)[1]

            random_neighbors = x[random_neighbor_indices]
            neighbor_targets = y[random_neighbor_indices]
            neighbor_proc = x_proc[random_neighbor_indices]
            
            # Choose a random weight vector (and then normalise it)
            weights = np.exp(np.log(0.5) * random_neighbor_distances / nearest_neighbor_distance)
            weights /= np.sum(weights)

            # Compute tslearn.barycenters.softdtw_barycenter with weights=random weights and gamma value specific to neighbors
            random_neighbors_gamma = gamma_soft_dtw(random_neighbors)
            generated_sample = softdtw_barycenter(random_neighbors, weights=weights, gamma=random_neighbors_gamma)
            
            # Compute synthetic target as weighted average of neighbour targets
            generated_target = np.sum(weights * neighbor_targets)
            # Compute synthetic process parameters as a weighted average of neighbour parameters
            generated_process_params = np.sum(weights[:, None] * neighbor_proc, axis=0)
            
            # Compute synthetic target as weighted average of neighbour targets
            synthetic_x.append(generated_sample)
            synthetic_y.append(generated_target)
            synthetic_x_proc.append(generated_process_params)

    # return the synthetic set     
    return np.array(synthetic_x), np.array(synthetic_x_proc), np.array(synthetic_y)

def convert_to_cwt(x, wavelet='morlet', description='CWT generation', device='cpu', scales='log-piecewise'):
    """
    data shape: (N, W, C) - N = Number of samples, W = Width of the sample (time), C = Channel
    returns: CWT numpy array with shape: (N, S, W, C) - S = Scales (height), W = Width
    """   
    # Set GPU/CPU mode
    if device == 'gpu':
        os.environ['SSQ_GPU'] = '1'
    else:
        os.environ['SSQ_GPU'] = '0'

    # Run once to get dimensions
    Wx, _ = cwt(x[0, :, 0], wavelet=wavelet, scales=scales)
    n_scales_actual = Wx.shape[0]
    
    cwt_out = np.empty((x.shape[0], n_scales_actual, x.shape[1], x.shape[2]), dtype=np.float32)
    
    for c in tqdm(range(x.shape[2]), desc=description):
        Wx, _ = cwt(x[:, :, c], wavelet=wavelet, scales=scales)
        # Log transform for better neural network features
        cwt_out[:, :, :, c] = np.log(np.abs(Wx) + 1e-6)
            
    return cwt_out
    
def convert_to_stransform(x, lo=0, hi=125, gamma=1, description='S-transform generation'):
    """
    Source: https://ieeexplore.ieee.org/abstract/document/492555
    data shape: (N, W, C) - N = Number of samples, W = Width of the sample (time), C = Channel
    returns: S-transform numpy array with shape: (N, H, W, C) - H = Height (frequency)
    """
    cwt = np.empty((x.shape[0], hi+1, x.shape[1], x.shape[2]), dtype=x.dtype)
    for n in tqdm(range(x.shape[0]), desc=description):
        for c in range(x.shape[2]):
            stransf = st.st(x[n, :, c], hi=hi, lo=lo, gamma=gamma)
            cwt[n, :, :, c] = np.abs(stransf)
    return cwt

def convert_to_fsst(x, fs=1.0, window=None, n_fft=None, win_len=None, description='FSST generation', device='cpu'):
    """
    Apply Fourier Synchrosqueezed Transform (FSST) to signals using ssqueezepy.
    data shape: (N, W, C)
    returns: FSST magnitude array with shape: (N, F, T, C)
    Args:
        window: STFT windowing kernel. Default None.
        n_fft: FFT length. Default None.
        win_len: Length of window. Default None.
    """
    # Set GPU/CPU mode
    if device == 'gpu':
        os.environ['SSQ_GPU'] = '1'
    else:
        os.environ['SSQ_GPU'] = '0'

    # Run once to get output shape
    # ssq_stft returns tuple, first element is Tx
    Tx, *_ = ssq_stft(x[0, :, 0], fs=fs, window=window, n_fft=n_fft, win_len=win_len)
    F, T = Tx.shape
    
    # Pre-allocate. Note: output might be complex
    fsst_out = np.empty((x.shape[0], F, T, x.shape[2]), dtype=np.float32)
    
    for c in tqdm(range(x.shape[2]), desc=description):
        Tx, *_ = ssq_stft(x[:, :, c], fs=fs, window=window, n_fft=n_fft, win_len=win_len)
        fsst_out[:, :, :, c] = np.abs(Tx)
            
    return fsst_out

def convert_to_wsst(x, fs=1.0, wavelet='morlet', scales='log-piecewise', description='WSST generation', device='cpu'):
    """
    Apply Wavelet Synchrosqueezed Transform (WSST) to signals using ssqueezepy.
    data shape: (N, W, C)
    returns: WSST magnitude array with shape: (N, F, T, C)
    Args:
        scales: Scales to use for CWT. Default 'log-piecewise'.
    """
    # Set GPU/CPU mode
    if device == 'gpu':
        os.environ['SSQ_GPU'] = '1'
    else:
        os.environ['SSQ_GPU'] = '0'

    print(f'WSST x shape: {x.shape}')

    # Run once to get output shape
    Tx, *_ = ssq_cwt(x[0, :, 0], wavelet=wavelet, scales=scales, fs=fs)
    F, T = Tx.shape
    
    # Pre-allocate
    wsst_out = np.empty((x.shape[0], F, T, x.shape[2]), dtype=np.float32)
    
    for c in tqdm(range(x.shape[2]), desc=description):
        Tx, *_ = ssq_cwt(x[:, :, c], wavelet=wavelet, scales=scales, fs=fs)
        wsst_out[:, :, :, c] = np.abs(Tx)
            
    return wsst_out
    
class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def evaluate_model(model, x, y, folder, run_name, history=None, training_time=None, model_results=None, y_min=0, y_max=1, norm_target=False):
    """
    Evaluates model, generates plots, and saves results/history to JSON.
    """
    if model_results is None:
        model_results = {}
        
    start_time = time.time()
    ypred = model.predict(x)
    total_time = time.time() - start_time
    
    model_results['eval_time'] = total_time
    if training_time is not None:
        model_results['time'] = training_time # Compatible with train.py expectation
        model_results['training_time'] = training_time
        
    # Inverse MinMax predictions and targets
    if norm_target:
        ypred_orig = ypred * (y_max - y_min) + y_min
        y_test_orig = y * (y_max - y_min) + y_min
    else:
        ypred_orig = ypred
        y_test_orig = y

    rmse, r2, mae, mape = get_scores(y_test_orig, ypred_orig)
    print(f'RMSE: {rmse:.5f}\nR2: {r2:.5f}\nMAE: {mae:.5f}\nMAPE: {mape:.5f}')

    file_name = f'{folder}/DL_{run_name}_{rmse:.4f}.keras'
    model.save(file_name)
    
    model_results['model_file'] = file_name
    model_results['r2_score'] = r2
    model_results['rmse'] = rmse
    model_results['mae'] = mae
    model_results['mape'] = mape
    
    # Save History Plot
    if history is not None:
        hist_dict = history.history if hasattr(history, 'history') else history
        model_results['history'] = hist_dict
        
        plt.figure()
        if 'loss' in hist_dict: plt.plot(hist_dict['loss'], label='Training')
        if 'val_loss' in hist_dict: plt.plot(hist_dict['val_loss'], label='Validation')
        plt.title(f"Loss - {run_name}")
        plt.legend()
        plt.savefig(f"{folder}/DL_{run_name}_training_loss.png")
        plt.close()

    # Save Prediction Plot
    results_df = pd.DataFrame({'Ground truth' : np.squeeze(y_test_orig), 'Prediction' : np.squeeze(ypred_orig)})
    results_df = results_df.sort_values(by=['Ground truth'], ignore_index=True)

    plt.figure()
    plt.plot(results_df['Prediction'], label='Prediction')
    plt.plot(results_df['Ground truth'], label='Ground truth')
    plt.legend(loc='upper left')
    plt.title(f"Preds - {run_name}")
    plt.savefig(f"{folder}/DL_{run_name}_test_pred.png")
    plt.close()
    
    # Export Results to JSON
    json_path = f'{folder}/DL_{run_name}.json'
    with open(json_path, "w") as outfile:
        outfile.write(json.dumps(model_results, indent=4, cls=NumpyFloatValuesEncoder))
        
    return model_results

__all__ = [
    'get_scores',
    'root_mse',
    'get_file_list',
    'apply_moving_average',
    'augment_signal',
    'downsample',
    'plot_signals',
    'plot_signals_dict',
    'compute_mean_std',
    'compute_min_max',
    'augment_dataset_softdtw',
    'convert_to_cwt',
    'convert_to_stransform',
    'convert_to_fsst',
    'convert_to_wsst',
    'apply_high_pass_filter',
    'apply_low_pass_filter',
    'apply_rms',
    'apply_detrend',
    'evaluate_model',
    'NumpyFloatValuesEncoder'
]