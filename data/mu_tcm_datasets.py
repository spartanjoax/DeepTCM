################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""
MU-TCM Dataset loader for the Mondragon Unibertsitatea CNC face-milling dataset.

Key differences from NASA_Dataset:
  - Variable-length runs (1000–4250 samples at 250 Hz) → always windowed
  - 24 signals (16 internal at 250 Hz + 8 external downsampled to 250 Hz)
  - Process params: [ap, fz, material, Vc] (ap fixed at 1.5 for all runs)
  - No 'paper' split — random run-level split only
  - VB cap at 0.4 mm (per Ch06 CL pipeline)
  - Condition identifier: (material, Vc, fz) → 8 unique conditions
"""

import gc
import os
import ntpath
import warnings

import numpy as np
import pandas as pd
import scipy.io
from joblib import load, dump
from scipy.ndimage import uniform_filter1d
from scipy.signal import decimate
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from helpers import (
    augment_signal,
    get_file_list,
)

# Min-max bounds for process parameter normalisation.
MU_PROC_MIN = [0.1, 0.05, 1, 50]
MU_PROC_MAX = [1.5, 0.5, 2, 200]
N_BASE_PROC = 4

allowed_signal_group = [
    "DC", "DC_AE", "DC_table", "DC_Vib",
    "table", "all",
    "AC", "AC_AE", "AC_table", "AC_Vib",
    "internals", "internals_v2", "internals_v3", "ACDC",
]

_CONDITION_KEYS = sorted([
    (1, 100, 0.1), (1, 100, 0.2), (1, 200, 0.1), (1, 200, 0.2),  # CI
    (2, 50, 0.05), (2, 50, 0.1), (2, 100, 0.05), (2, 100, 0.1),  # SS
])
_CONDITION_MAP = {k: i for i, k in enumerate(_CONDITION_KEYS)}


def _condition_id(material, vc, fz):
    """Map (material, Vc, fz) to integer condition ID.

    Args:
        material (int): Material code (1 = GG30 cast iron, 2 = SS316L stainless steel).
        vc (int): Cutting speed in m/min.
        fz (float): Feed per tooth in mm.

    Returns:
        int: Condition ID from ``_CONDITION_MAP``, or -1 if not found.
    """
    key = (int(material), int(vc), float(round(fz, 3)))
    return _CONDITION_MAP.get(key, -1)


class MU_TCM_Dataset(Dataset):

    material_map = {
        "CastIron.GG30": 1,
        "StainlessSteel.316L": 2,
        "": 0,
    }

    signals_250 = [
        "CV3_S", "CV3_X", "CV3_Y", "CV3_Z",
        "FREAL", "POS_S", "POS_X", "POS_Y", "POS_Z",
        "SREAL", "TV2_S", "TV2_X", "TV2_Y", "TV2_Z", "TV50", "TV51",
    ]
    signals_50k = ["Ax", "Ay", "Az", "Fx", "Fy", "Fz"]
    signals_1M = ["AE_F", "AE_RMS"]
    signal_list = sorted(signals_250 + signals_50k + signals_1M)

    proc_variable_list = ["ap", "fz", "material", "Vc"]

    def __init__(
        self,
        transformX=None,
        transformProc=None,
        split="train",
        split_ratios=(0.48, 0.12, 0.4),
        seed=42,
        signal_group="all",
        debug_plots=False,
        sliding_window_size=500,
        sliding_window_stride=50,
        filters=None,
        apply_averaging=True,
        avg_window_size=10,
        mu_data_folder="data/MU-TCM/",
        stats_csv_path="data/MU-TCM/signals_stats.csv",
        custom_train_runs=None,
        custom_val_runs=None,
    ):
        """
        PyTorch Dataset for the MU-TCM face-milling dataset.

        Windowing is always applied (variable-length runs require it).

        Args:
            transformX: Signal normalisation transform (applied eagerly).
            transformProc: Process-parameter normalisation transform.
            split: One of 'train', 'val', 'test'.
            split_ratios: (train, val, test) proportions.
            seed: Random seed for splitting.
            signal_group: Signal subset name.
            debug_plots: Print signal statistics.
            sliding_window_size: Window size in samples (applied to 250 Hz signal).
            sliding_window_stride: Window stride in samples.
            filters: Dict of {column: value} to filter runs (e.g. {'material': 2}).
            apply_averaging: Whether to denoise with moving average.
            avg_window_size: Moving-average window size.
            mu_data_folder: Path to directory containing .mat files.
            stats_csv_path: Path to signals_stats.csv (SREAL_start/end markers).
            custom_train_runs: Override train run indices (list of ints).
            custom_val_runs: Override val run indices (list of ints).
        """
        self.transformX = transformX
        self.transformProc = transformProc
        self.split = split
        self.seed = seed
        self.debug_plots = debug_plots
        self.signal_group = signal_group
        self.avg_window_size = avg_window_size

        assert split in ["train", "val", "test"]
        assert signal_group in allowed_signal_group

        data_file_name = f'data/mu{f"_ma{avg_window_size}" if apply_averaging else ""}.bin'

        if not os.path.exists(data_file_name):
            print("================== MU-TCM dataset files do not exist\n"
                  "Processing and saving from .mat files...")
            _process_mu_tcm_dataset(
                data_file_name=data_file_name,
                mu_data_folder=mu_data_folder,
                stats_csv_path=stats_csv_path,
                apply_averaging=apply_averaging,
                avg_window_size=avg_window_size,
                debug_plots=debug_plots,
            )
        else:
            print("================== MU-TCM dataset files exist\n"
                  "Loading pre-saved splits...")

        x_list, x_proc, y, run_filenames = load(data_file_name)
        n_total = len(x_list)
        print(f"Loaded {n_total} runs (variable length)")

        if filters is not None:
            keep = np.ones(n_total, dtype=bool)
            for key, value in filters.items():
                col_idx = MU_TCM_Dataset.proc_variable_list.index(key)
                keep &= x_proc[:, col_idx] == value
            idx = np.where(keep)[0]
            x_list = [x_list[i] for i in idx]
            x_proc = x_proc[idx]
            y = y[idx]
            run_filenames = [run_filenames[i] for i in idx]
            print(f"After filters {filters}: {len(x_list)} runs")

        n_runs = len(x_list)

        cond_ids = np.array([
            _condition_id(x_proc[i, 2], x_proc[i, 3], x_proc[i, 1])
            for i in range(n_runs)
        ])

        all_indices = np.arange(n_runs)

        if custom_train_runs is not None and custom_val_runs is not None:
            train_idx = np.array(custom_train_runs, dtype=int)
            val_idx = np.array(custom_val_runs, dtype=int)
            test_idx = np.array([i for i in all_indices
                                 if i not in train_idx and i not in val_idx], dtype=int)
        else:
            train_idx, test_idx = train_test_split(
                all_indices, test_size=split_ratios[2], random_state=self.seed,
            )
            if split_ratios[1] > 0:
                rel_val = split_ratios[1] / (split_ratios[0] + split_ratios[1])
                train_idx, val_idx = train_test_split(
                    train_idx, test_size=rel_val, random_state=self.seed,
                )
            else:
                val_idx = np.array([], dtype=int)

        self.train_runs = train_idx
        self.val_runs = val_idx
        self.test_runs = test_idx

        if split == "train":
            sel = train_idx
        elif split == "val":
            sel = val_idx
        else:
            sel = test_idx

        sel_x = [x_list[i] for i in sel]
        sel_proc = x_proc[sel]
        sel_y = y[sel]
        sel_cond = cond_ids[sel]

        print(f"================== {split} split: {len(sel)}/{n_runs} runs "
              f"({len(sel)/n_runs*100:.1f}%)")

        lengths = [arr.shape[0] for arr in sel_x]
        skipped = sum(1 for l in lengths if l < sliding_window_size)
        if skipped > 0:
            warnings.warn(
                f"{skipped}/{len(sel)} runs shorter than sw={sliding_window_size} "
                f"— they will be skipped. Min length: {min(lengths)}"
            )

        proc_with_cond = np.hstack([sel_proc, sel_cond[:, None]])
        x_windowed, proc_windowed, y_windowed = augment_signal(
            sel_x, proc_with_cond, sel_y,
            window_size=sliding_window_size,
            stride=sliding_window_stride,
            description=f"Windowing MU-TCM {split}",
        )

        self.case_labels = proc_windowed[:, -1].astype(int)
        proc_windowed = proc_windowed[:, :-1]

        print(f"================== Windowed: {x_windowed.shape[0]} windows, "
              f"shape {x_windowed.shape}")

        x_windowed, proc_windowed = self._remove_signals(x_windowed, proc_windowed)

        x_tensor = torch.tensor(x_windowed, dtype=torch.float)
        proc_tensor = torch.tensor(proc_windowed, dtype=torch.float)

        if self.transformX is not None:
            x_tensor = self.transformX(x_tensor).to(dtype=torch.float)
        if self.transformProc is not None:
            proc_tensor = self.transformProc(proc_tensor).to(dtype=torch.float)

        self.data = x_tensor
        self.proc_data = proc_tensor
        self.targets = torch.tensor(y_windowed, dtype=torch.float).unsqueeze(1)

        print(f'================== MU-TCM {split} dataset loaded for "{signal_group}"')

    def __len__(self):
        """Return the number of windowed samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the multimodal input dict and target for sample at *idx*."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {
            "proc_data": self.proc_data[idx],
            "x": self.data[idx],
        }
        return sample, self.targets[idx]

    def get_signal_list(self):
        """Return the list of signal names for the current signal_group.

        Group names are case-insensitive. Canonical lowercase names used in
        CL experiments (ac, dc, actable, dctable) are accepted alongside the
        mixed-case forms (AC, DC, AC_table, DC_table).
        """
        return get_mu_tcm_signal_list(self.signal_group)

    def _remove_signals(self, data, proc_data=None):
        """Remove signals not in the active signal group.

        Args:
            data (np.ndarray): Signal array of shape (N, T, C) where C is the
                full channel count of ``MU_TCM_Dataset.signal_list``.
            proc_data: Unused; kept for API symmetry with NASA_Dataset. Always
                returned unchanged.

        Returns:
            tuple: ``(pruned_data, proc_data)`` where ``pruned_data`` has only
                the channels belonging to the active signal group.
        """
        keep_signals = self.get_signal_list()
        drop_idx = [
            i for i, s in enumerate(MU_TCM_Dataset.signal_list)
            if s not in keep_signals
        ]
        x = np.delete(data, drop_idx, axis=2)
        return x, proc_data


def get_mu_tcm_signal_list(signal_group):
    """Return the list of MU-TCM signal names for a given signal group.

    Standalone counterpart of ``MU_TCM_Dataset.get_signal_list()`` that can be
    called without instantiating the full Dataset class.
    """
    g = (signal_group or "").lower()
    if g in ("dc", "ac"):
        return ["CV3_S"]
    elif g in ("dc_ae", "ac_ae"):
        return ["CV3_S", "AE_F"]
    elif g in ("dc_vib", "ac_vib"):
        return ["CV3_S", "Az"]
    elif g in ("dc_table", "ac_table", "dctable", "actable"):
        return ["CV3_S", "Az", "AE_F"]
    elif g == "table":
        return ["Az", "AE_F"]
    elif g == "internals":
        return [
            "CV3_S", "CV3_X", "CV3_Y", "CV3_Z",
            "FREAL", "SREAL",
            "TV2_S", "TV2_X", "TV2_Y", "TV2_Z",
            "TV50", "TV51",
        ]
    return MU_TCM_Dataset.signal_list.copy()


def get_mu_tcm_raw_split(
    signal_group,
    filters=None,
    custom_train_runs=None,
    custom_val_runs=None,
    avg_window_size=10,
    apply_averaging=False,
    mu_data_folder="data/MU-TCM/",
    stats_csv_path="data/MU-TCM/signals_stats.csv",
):
    """Load raw (pre-windowing) MU-TCM runs for the given condition.

    Replicates the loading + filtering + splitting logic of
    ``MU_TCM_Dataset.__init__`` but skips windowing, transforms, and feature
    extraction.  The caller is responsible for normalisation before feeding
    the returned arrays to ``apply_full_preprocessing``.

    Args:
        signal_group  : Signal group name (e.g. "AC", "AC_table", "internals").
        filters       : Dict of {column: value} condition selectors, e.g.
                        ``{"material": 2, "fz": 0.05, "Vc": 50}``.
        custom_train_runs / custom_val_runs :
                        Run indices from ``get_mu_tcm_fixed_split``; the test
                        set is the remainder not in either list.
        avg_window_size : Moving-average window size — determines which cache
                        file to load (default 10 → ``data/mu_ma10.bin``).
        apply_averaging : If True load the MA-smoothed cache, else the raw
                        cache (default False — matches train.py/champion).
        mu_data_folder / stats_csv_path : Passed to ``_process_mu_tcm_dataset``
                        when the cache file does not yet exist.

    Returns:
        A tuple ((x_tr, proc_tr, y_tr), (x_va, proc_va, y_va), (x_te, proc_te, y_te))
    """

    ma_tag = f"_ma{avg_window_size}" if apply_averaging else ""
    cache_path = f"data/mu{ma_tag}.bin"

    if not os.path.exists(cache_path):
        _process_mu_tcm_dataset(
            data_file_name=cache_path,
            mu_data_folder=mu_data_folder,
            stats_csv_path=stats_csv_path,
            apply_averaging=apply_averaging,
            avg_window_size=avg_window_size,
        )

    x_list, x_proc, y, _ = load(cache_path)
    n_total = len(x_list)

    if filters is not None:
        keep = np.ones(n_total, dtype=bool)
        for key, value in filters.items():
            col_idx = MU_TCM_Dataset.proc_variable_list.index(key)
            keep &= x_proc[:, col_idx] == value
        idx = np.where(keep)[0]
        x_list = [x_list[i] for i in idx]
        x_proc = x_proc[idx]
        y = y[idx]

    n_runs = len(x_list)
    all_idx = np.arange(n_runs)

    if custom_train_runs is not None and custom_val_runs is not None:
        tr_set = set(int(i) for i in custom_train_runs)
        va_set = set(int(i) for i in custom_val_runs)
        train_idx = np.array(sorted(tr_set), dtype=int)
        val_idx   = np.array(sorted(va_set), dtype=int)
        test_idx  = np.array(
            [i for i in all_idx if i not in tr_set and i not in va_set], dtype=int
        )
    else:
        train_idx, test_idx = train_test_split(all_idx, test_size=0.4, random_state=42)
        train_idx, val_idx  = train_test_split(train_idx, test_size=0.2, random_state=42)

    keep_sigs = get_mu_tcm_signal_list(signal_group)
    drop_cols = [
        i for i, s in enumerate(MU_TCM_Dataset.signal_list)
        if s not in keep_sigs
    ]

    def _select_sig(lst):
        """Drop the excluded signal columns from each run array in *lst*."""
        return [np.delete(run, drop_cols, axis=1).astype(np.float32) for run in lst]

    def _split(idx_arr):
        """Extract train/val/test arrays for a given index subset."""
        return (
            _select_sig([x_list[i] for i in idx_arr]),
            x_proc[idx_arr].astype(np.float32),
            y[idx_arr].astype(np.float32),
        )

    return _split(train_idx), _split(val_idx), _split(test_idx)


def get_mu_tcm_scenario_normalizer(signal_group, scenario_filters, avg_window_size=10):
    """Compute a single z-score normaliser from all MU-TCM training runs in a
    scenario.

    Mirrors ``get_nasa_data_pipeline``'s global-fit approach: one shared mean
    and std fitted over **all training runs across all experiences** in the
    scenario so that every experience and every replay-buffer sample share the
    same input coordinate system.

    The split for each experience is determined by the same
    ``get_mu_tcm_fixed_split`` logic used during training (Mode A / Mode B
    is auto-detected from the shape of each filter dict), so the normaliser
    is fitted on exactly the training data that will be seen during CL.

    Args:
        signal_group     : Signal group name (e.g. ``"AC_table"``, ``"internals"``).
        scenario_filters : List of filter dicts — one per experience, exactly
                           as defined in the scenario config
                           (e.g. ``config["mu_filters"]``).
        avg_window_size  : Passed through to cache selection (default 10).

    Returns:
        (mu_sig_mean, mu_sig_std) — each has shape ``(1, C)`` where C is the
        channel count of the requested signal group.  Safe to broadcast
        against run arrays of shape ``(T_i, C)``.
    """
    all_tr_raw: list = []
    for filt in scenario_filters:
        _, tr_idx, val_idx, _ = get_mu_tcm_fixed_split(
            filt, avg_window_size=avg_window_size,
            val_fraction=0.0, apply_averaging=False,
        )
        if len(tr_idx) == 0:
            continue
        (x_tr_raw, _, _), _, _ = get_mu_tcm_raw_split(
            signal_group=signal_group,
            filters=filt,
            custom_train_runs=tr_idx,
            custom_val_runs=val_idx,
            apply_averaging=False,
            avg_window_size=avg_window_size,
        )
        all_tr_raw.extend(x_tr_raw)

    if all_tr_raw:
        all_cat     = np.concatenate(all_tr_raw, axis=0)  # (Σ T_i, C)
        mu_sig_mean = all_cat.mean(axis=0, keepdims=True).astype(np.float32)  # (1, C)
        mu_sig_std  = (all_cat.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    else:
        mu_sig_mean = np.zeros((1, 1), dtype=np.float32)
        mu_sig_std  = np.ones((1, 1),  dtype=np.float32)

    return mu_sig_mean, mu_sig_std


def get_mu_tcm_scenario_data(signal_group, scenario_filters, avg_window_size=10):
    """Full data pipeline for a MU-TCM CL scenario.

    Analogous to ``get_nasa_data_pipeline``: all normalisation is encapsulated
    here so callers receive ready-to-use arrays and never touch mean/std or
    proc bounds directly.

    A single z-score normaliser is fitted across **all training runs of all
    experiences** in the scenario (scenario-level, not per-experience).
    Process parameters are MinMax-normalised using the dataset-level bounds
    ``MU_PROC_MIN`` / ``MU_PROC_MAX``.

    Mode A / B split is auto-detected inside ``get_mu_tcm_fixed_split``
    exactly as during training, so the normaliser is fitted on the same set
    of runs that will be used for CL.

    Args:
        signal_group     : Signal group name (e.g. ``"AC_table"``, ``"internals"``).
        scenario_filters : List of filter dicts — one per experience.
        avg_window_size  : Passed through to cache selection (default 10).
    
    Returns:
        List of dicts, one per experience::

            {
                "filters"  : the filter dict passed in,
                "n_train"  : int,
                "n_val"    : int,
                "n_test"   : int,
                "train"    : (x_tr_z, proc_tr_n, y_tr) or None if no train runs,
                "val"      : (x_va_z, proc_va_n, y_va) or None if no val runs,
                "test"     : (x_te_z, proc_te_n, y_te),
            }
    """
    _proc_min = np.array(MU_PROC_MIN, dtype=np.float32)
    _proc_max = np.array(MU_PROC_MAX, dtype=np.float32)
    _proc_rng = _proc_max - _proc_min

    mu_sig_mean, mu_sig_std = get_mu_tcm_scenario_normalizer(
        signal_group, scenario_filters,
        avg_window_size=avg_window_size,
    )

    results = []
    for filt in scenario_filters:
        test_idx, train_idx, val_idx, n_total = get_mu_tcm_fixed_split(
            filt, avg_window_size=avg_window_size,
            val_fraction=0.2, apply_averaging=False,
        )

        (x_tr, proc_tr, y_tr), (x_va, proc_va, y_va), (x_te, proc_te, y_te) = get_mu_tcm_raw_split(
            signal_group=signal_group,
            filters=filt,
            custom_train_runs=train_idx,
            custom_val_runs=val_idx,
            apply_averaging=False,
            avg_window_size=avg_window_size,
        )

        if len(x_tr) == 0:
            results.append({
                "filters": filt,
                "n_train": len(train_idx),
                "n_val":   len(val_idx),
                "n_test":  len(test_idx),
                "train":   None,
                "val":     None,
                "test":    (
                    [(r - mu_sig_mean) / mu_sig_std for r in x_te],
                    np.clip((proc_te - _proc_min) / _proc_rng, 0.0, 1.0),
                    y_te,
                ),
            })
            continue

        x_tr_z    = [(r - mu_sig_mean) / mu_sig_std for r in x_tr]
        x_te_z    = [(r - mu_sig_mean) / mu_sig_std for r in x_te]
        proc_tr_n = np.clip((proc_tr - _proc_min) / _proc_rng, 0.0, 1.0)
        proc_te_n = np.clip((proc_te - _proc_min) / _proc_rng, 0.0, 1.0)

        val_entry = None
        if len(x_va) > 0:
            x_va_z    = [(r - mu_sig_mean) / mu_sig_std for r in x_va]
            proc_va_n = np.clip((proc_va - _proc_min) / _proc_rng, 0.0, 1.0)
            val_entry = (x_va_z, proc_va_n, y_va)

        results.append({
            "filters": filt,
            "n_train": len(train_idx),
            "n_val":   len(val_idx),
            "n_test":  len(test_idx),
            "train":   (x_tr_z, proc_tr_n, y_tr),
            "val":     val_entry,
            "test":    (x_te_z, proc_te_n, y_te),
        })

    return results

def _safe_downsample(signal, target_len):
    """
    Downsample a signal to target_len using staged IIR downsampling to avoid artifacts.

    Args:
        signal (np.ndarray): 1-D signal array to downsample.
        target_len (int): Desired output length in samples.

    Returns:
        np.ndarray: Downsampled (and zero-edge-padded if necessary) array of length
            ``target_len``.
    """
    current = signal.copy()
    current_len = len(current)
    ratio = current_len / target_len

    if ratio <= 1.0:
        return current[:target_len]

    while len(current) > target_len * 12:
        q = min(10, len(current) // (target_len * 2))
        if q < 2:
            break
        current = decimate(current, q, ftype="iir", zero_phase=True)

    final_ratio = len(current) / target_len
    if final_ratio >= 2.0:
        q = int(final_ratio)
        if q >= 2:
            current = decimate(current, q, ftype="iir", zero_phase=True)

    if len(current) > target_len:
        current = current[:target_len]
    elif len(current) < target_len:
        current = np.pad(current, (0, target_len - len(current)), mode="edge")

    return current

_NEEDED_MAT_KEYS = (
    MU_TCM_Dataset.signal_list
    + MU_TCM_Dataset.proc_variable_list
    + ["VB", "WorkpieceMaterial"]
)


def _process_mu_tcm_dataset(
    data_file_name,
    mu_data_folder="data/MU-TCM/",
    stats_csv_path="data/MU-TCM/signals_stats.csv",
    apply_averaging=True,
    avg_window_size=10,
    debug_plots=False,
):
    """
    Load raw .mat files, downsample, trim, denoise, and save as a list.

    Args:
        data_file_name (str): Output path for the cached joblib file.
        mu_data_folder (str): Root directory containing the .mat run files.
        stats_csv_path (str): Path to ``signals_stats.csv`` with per-signal
            target lengths and trim indices.
        apply_averaging (bool): Apply moving-average denoising after downsampling.
        avg_window_size (int): Kernel size for the moving-average filter.
        debug_plots (bool): If True, plot each signal after processing.

    Saved format (joblib):
        [x_list, x_proc, y, filenames]
    """
    if not os.path.exists(mu_data_folder):
        raise FileNotFoundError(
            f"MU-TCM data directory not found: {mu_data_folder}"
        )

    files = get_file_list(root=mu_data_folder, pattern="*.mat")
    files = sorted(files)
    print(f"Found {len(files)} .mat files")

    if not os.path.exists(stats_csv_path):
        raise FileNotFoundError(
            f"signals_stats.csv not found: {stats_csv_path}. "
            "Place it at data/MU-TCM/signals_stats.csv"
        )
    stats_df = pd.read_csv(stats_csv_path, sep=";", decimal=".")
    stats_dict = stats_df.set_index("_file_name").to_dict(orient="index")

    x_list = []
    x_proc = []
    y_list = []
    loaded_files = []

    for fpath in tqdm(files, desc="Loading MU-TCM .mat files"):
        mat = _load_mat_file(fpath, needed_keys=_NEEDED_MAT_KEYS)

        proc = []
        for pv in MU_TCM_Dataset.proc_variable_list:
            if pv == "material":
                proc.append(MU_TCM_Dataset.material_map.get(mat.get("WorkpieceMaterial", ""), 0))
            else:
                proc.append(float(mat[pv]))

        vb_value = float(mat["VB"])

        if proc[2] == 0:
            warnings.warn(f"Unknown material in {fpath}, skipping")
            continue

        signals = {}
        missing = False
        for sig_name in MU_TCM_Dataset.signal_list:
            if sig_name in mat and hasattr(mat[sig_name], '__len__') and len(mat[sig_name]) > 0:
                signals[sig_name] = np.asarray(mat[sig_name], dtype=np.float64)
            else:
                missing = True
                break
        if missing:
            warnings.warn(f"Missing signals in {fpath}, skipping")
            continue

        ref_len = len(signals["SREAL"])
        for sig_name in MU_TCM_Dataset.signals_50k + MU_TCM_Dataset.signals_1M:
            raw = signals[sig_name]
            if len(raw) != ref_len:
                signals[sig_name] = _safe_downsample(raw, ref_len)

        del mat
        gc.collect()

        run_matrix = np.column_stack(
            [signals[s] for s in MU_TCM_Dataset.signal_list]
        )

        fname = ntpath.basename(fpath)
        if fname in stats_dict:
            s_start = int(stats_dict[fname]["SREAL_start"])
            s_end = int(stats_dict[fname]["SREAL_end"])
            run_matrix = run_matrix[s_start:s_end, :]
        else:
            warnings.warn(f"No SREAL markers for {fname}, using full signal")

        if run_matrix.shape[0] < 10:
            warnings.warn(f"Run {fname} too short after trimming ({run_matrix.shape[0]} samples), skipping")
            continue

        if apply_averaging:
            run_matrix = uniform_filter1d(
                run_matrix, size=avg_window_size, axis=0, mode="nearest"
            )

        x_list.append(run_matrix)
        x_proc.append(proc)
        y_list.append(vb_value)
        loaded_files.append(fname)

    x_proc = np.array(x_proc, dtype=np.float64)
    y_arr = np.array(y_list, dtype=np.float64)

    print(f"================== MU-TCM: loaded {len(x_list)} runs")
    print(f"VB range: [{y_arr.min():.3f}, {y_arr.max():.3f}]")
    for i, pv in enumerate(MU_TCM_Dataset.proc_variable_list):
        print(f"  {pv}: [{x_proc[:, i].min()}, {x_proc[:, i].max()}]")

    lengths = [m.shape[0] for m in x_list]
    print(f"Signal lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}")

    if debug_plots:
        for i, sig in enumerate(MU_TCM_Dataset.signal_list):
            vals = np.concatenate([m[:, i] for m in x_list])
            print(f"  {sig}: min={vals.min():.4f}, max={vals.max():.4f}")

    dump([x_list, x_proc, y_arr, loaded_files], data_file_name)
    print(f"Saved to {data_file_name}")


def _load_mat_file(fpath, needed_keys=None):
    """Load a single .mat file and flatten scalar arrays.

    Args:
        fpath: Path to .mat file.
        needed_keys: If provided, only load these variables (saves memory).
    """
    kw = {}
    if needed_keys is not None:
        kw["variable_names"] = needed_keys
    mat = scipy.io.loadmat(fpath, **kw)
    out = {}
    for k, v in mat.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.ndarray):
            if v.ndim == 2 and v.shape[0] == 1 and v.shape[1] > 1:
                out[k] = v[0]
            elif v.ndim == 2 and v.shape[0] == 1 and v.shape[1] == 1:
                out[k] = v[0, 0]
            elif v.ndim == 2 and v.shape[1] == 1:
                out[k] = v.flatten()
            elif v.ndim == 1 and v.shape[0] == 1:
                val = v[0]
                out[k] = str(val) if isinstance(val, np.str_) else val
            elif v.ndim == 1:
                if v.dtype.kind in ('U', 'S', 'O') and len(v) == 1:
                    out[k] = str(v[0])
                else:
                    out[k] = v
            else:
                out[k] = v
        else:
            out[k] = v
    return out

def get_mu_tcm_fixed_split(
    filters,
    mu_data_folder="data/MU-TCM/",
    avg_window_size=10,
    val_fraction=0.2,
    apply_averaging=False,
):
    """Return fixed, deterministic run-level train/val/test indices for MU-TCM.

    Two split modes are selected automatically from ``filters``:

    **Mode A — single-condition** (``filters`` contains ``material``, ``Vc``,
    and ``fz``):
        Test = last repetition (alphabetical filename order) of **each unique
        VB level** in the condition (typically 4 runs at VB ∈ {0.0, 0.1, 0.2,
        0.3} mm).  This spans the full wear range so R² is well-defined, while
        keeping all earlier reps of each level in training.

    **Mode B — multi-condition** (``filters`` is ``None`` or contains only
    ``material``):
        Test = **all** runs from the last sorted condition per material present
        (CI last = Vc=200/fz=0.2; SS last = Vc=100/fz=0.1).  Analogous to the
        NASA paper-split holding out cases {11,12,15,16}: the held-out
        conditions were never seen during training.
        Returns **all runs** of the test condition(s).

    In both modes the remaining runs are split train/val at run level:
        val   = last ``round(n_remaining * val_fraction)`` sorted runs
        train = all other remaining runs

    The returned indices are positions within the filtered run
    list (0-based local indexing), matching what ``MU_TCM_Dataset`` uses
    internally when ``custom_train_runs`` / ``custom_val_runs`` are supplied.

    Args:
        filters:          Dict of {column: value} condition selectors, e.g.
                          ``{"material": 2, "fz": 0.05, "Vc": 50}`` (Mode A)
                          or ``{"material": 2}`` / ``None`` (Mode B).
        mu_data_folder:   Path to MU-TCM .mat folder (only used to locate bin).
        avg_window_size:  Moving-average window size (used only when
                          ``apply_averaging=True`` to select the correct .bin).
        val_fraction:     Fraction of non-test runs to hold out for validation.
                          Pass 0.0 when using Avalanche's
                          ``benchmark_with_validation_stream`` to avoid
                          wasting training data at the run level.
        apply_averaging:  If True load the MA-smoothed cache (``mu_ma<N>.bin``);
                          if False (default) load the raw cache (``mu.bin``).
                          **Must match the ``apply_averaging`` argument passed
                          to ``get_mu_tcm_raw_split``** so that both calls
                          operate on the same run ordering.

    Returns:
        test_idx  (list[int]) — local indices of test runs
        train_idx (list[int]) — local indices of training runs
        val_idx   (list[int]) — local indices of validation runs
        n_runs    (int)       — total runs after filtering

    Raises:
        ValueError if fewer than 2 runs remain after filtering
    """
    ma_tag = f"_ma{avg_window_size}" if apply_averaging else ""
    data_file_name = f"data/mu{ma_tag}.bin"

    if not os.path.exists(data_file_name):
        _process_mu_tcm_dataset(
            data_file_name=data_file_name,
            mu_data_folder=mu_data_folder,
            apply_averaging=apply_averaging,
            avg_window_size=avg_window_size,
        )

    x_list, x_proc, y, filenames = load(data_file_name)
    n_total = len(x_list)

    keep = np.ones(n_total, dtype=bool)
    if filters is not None:
        for key, value in filters.items():
            col_idx = MU_TCM_Dataset.proc_variable_list.index(key)
            keep &= x_proc[:, col_idx] == value

    global_idx = np.where(keep)[0]

    if len(global_idx) < 2:
        raise ValueError(
            f"Only {len(global_idx)} run(s) match filters={filters} after "
        )

    local_filenames = [filenames[i] for i in global_idx]
    sort_order = np.argsort(local_filenames)

    sorted_local = sort_order.tolist()

    is_single_condition = (
        filters is not None
        and {"material", "Vc", "fz"}.issubset(filters.keys())
    )

    if is_single_condition:
        sorted_vbs = np.array([y[global_idx[i]] for i in sorted_local])
        unique_vbs = sorted(set(round(float(v), 4) for v in sorted_vbs))
        test_k: list[int] = []
        for uv in unique_vbs:
            reps = [k for k, v in enumerate(sorted_vbs)
                    if abs(float(v) - uv) < 1e-4]
            test_k.append(reps[-1])
        test_idx = [sorted_local[k] for k in test_k]
        remaining = [i for i in sorted_local if i not in set(test_idx)]

    else:
        local_procs = x_proc[global_idx]
        _CONDS_BY_MAT = {
            1: sorted([(1, 100, 0.1), (1, 100, 0.2), (1, 200, 0.1), (1, 200, 0.2)]),
            2: sorted([(2,  50, 0.05), (2,  50, 0.1), (2, 100, 0.05), (2, 100, 0.1)]),
        }
        mats_present = sorted(
            set(int(local_procs[i, 2]) for i in range(len(global_idx)))
        )
        test_local_set: set = set()
        for mat in mats_present:
            for _, last_vc, last_fz in reversed(_CONDS_BY_MAT.get(mat, [])):
                if any(
                    int(local_procs[i, 2]) == mat
                    and abs(float(local_procs[i, 3]) - last_vc) < 1e-6
                    and abs(float(local_procs[i, 1]) - last_fz) < 1e-6
                    for i in range(len(global_idx))
                ):
                    for local_i in range(len(global_idx)):
                        if (
                            int(local_procs[local_i, 2]) == mat
                            and abs(float(local_procs[local_i, 3]) - last_vc) < 1e-6
                            and abs(float(local_procs[local_i, 1]) - last_fz) < 1e-6
                        ):
                            test_local_set.add(local_i)
                    break
        test_idx = [i for i in sorted_local if i in test_local_set]
        remaining = [i for i in sorted_local if i not in test_local_set]

    if len(remaining) == 0:
        return test_idx, [], [], len(sorted_local)

    n_val = max(0, round(len(remaining) * val_fraction))
    if n_val >= len(remaining):
        n_val = max(0, len(remaining) - 1)

    if is_single_condition and n_val >= 1 and len(remaining) >= 4:
        remaining_vbs = [(i, float(y[global_idx[i]])) for i in remaining]
        remaining_vbs_sorted = sorted(remaining_vbs, key=lambda x: x[1])
        val_idx = list(dict.fromkeys(
            [remaining_vbs_sorted[0][0], remaining_vbs_sorted[-1][0]]
        ))
        train_idx = [i for i in remaining if i not in set(val_idx)]
    else:
        val_idx = remaining[len(remaining) - n_val:] if n_val > 0 else []
        train_idx = remaining[:len(remaining) - n_val]

    return test_idx, train_idx, val_idx, len(sorted_local)
