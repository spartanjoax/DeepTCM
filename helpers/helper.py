# -*- coding: utf-8 -*-

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.signal import resample
from ssqueezepy import ssq_stft
from torch import sqrt, mean

import numpy as np
import matplotlib.pyplot as plt

from fnmatch import fnmatch
import os
from tqdm import tqdm

from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import gamma_soft_dtw

import pywt
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
    
def apply_moving_average(x, window_size=10):
    x_aug = []
    for j in tqdm(range(len(x))):
        x_aug.append([])
        for i in range(len(x[j])):
            x_aug[j].append([])
            x_aug[j][i] = np.convolve(x[j][i], np.ones(window_size)/window_size, mode="same")
            x_aug[j][i] = x_aug[j][i][window_size:len(x_aug[j][i])-window_size]
        x_aug[j] = np.array(x_aug[j])
    return x_aug
    
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

def convert_to_cwt(x, n_scales=200, wavelet='morl', description='CWT generation'):
    """
    data shape: (N, W, C) - N = Number of samples, W = Width of the sample (time), C = Channel
    returns: CWT numpy array with shape: (N, S, W, C) - S = Scales (height), W = Width
    """   
    scales = np.arange(1, n_scales+1)#np.logspace(np.log10(1), np.log10(n_scales), n_scales) # 1 … n_scales  (same for both)
    cwt = np.empty((x.shape[0], n_scales, x.shape[1], x.shape[2]), dtype=x.dtype)
    for n in tqdm(range(x.shape[0]), desc=description):
        for c in range(x.shape[2]):
            coef, _ = pywt.cwt(x[n, :, c], scales, wavelet)
            # Step 1: Compute magnitude
            magnitude = np.abs(coef)
            # Step 2: Scale normalization (1/sqrt(scale))
            scale_norm = magnitude / np.sqrt(scales[:, np.newaxis])
            # Step 3: Log transform
            log_transformed = np.log(scale_norm + 1e-6)
            cwt[n, :, :, c] = log_transformed
    return cwt
    
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

def convert_to_fsst(x, fs=1.0, description='FSST generation'):
    """
    Apply Fourier Synchrosqueezed Transform (FSST) to signals using ssqueezepy.
    data shape: (N, W, C)
    returns: FSST magnitude array with shape: (N, F, T, C)
    """
    # Run once to get output shape
    # ssq_stft returns tuple, first element is Tx
    Tx, *_ = ssq_stft(x[0, :, 0])
    F, T = Tx.shape
    
    # Pre-allocate. Note: output might be complex
    fsst_out = np.empty((x.shape[0], F, T, x.shape[2]), dtype=np.float32)
    
    for n in tqdm(range(x.shape[0]), desc=description):
        for c in range(x.shape[2]):
            Tx, *_ = ssq_stft(x[n, :, c])
            fsst_out[n, :, :, c] = np.abs(Tx)
            
    return fsst_out
    
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
    'convert_to_fsst'
]