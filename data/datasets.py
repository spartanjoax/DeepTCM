################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""NASA Ames Milling Dataset loader for TCM.

Provides NASA_Dataset (PyTorch Dataset), windowed data pipeline, LOCO-CV and
paper-split logic (Zheng et al., 2017), z-score normalisation, and binary cache
management. Entry point: get_nasa_data_pipeline().
"""
import os
import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from helpers import (
    plot_signals,
    augment_signal,
    apply_moving_average,
)
import data.mat_to_csv as mtc

import torch
from torch.utils.data import Dataset
from .transforms import StdScalerTransform, MinMaxScalerTransform

NASA_PROC_MIN = [0.1, 0.05, 1, 50]
NASA_PROC_MAX = [1.5, 0.5, 2, 200]
N_BASE_PROC = 4


allowed_signal_group = [
    "DC",
    "DC_AE",
    "DC_table",
    "DC_Vib",
    "table",
    "all",
    "AC",
    "AC_AE",
    "AC_table",
    "AC_Vib",
    "internals",
    "ACDC",
]


class NASA_Dataset(Dataset):

    signal_list = [
        "smcAC",
        "smcDC",
        "vib_table",
        "vib_spindle",
        "AE_table",
        "AE_spindle",
    ]
    proc_variable_list = [
        "DOC",
        "feed",
        "material",
        "Vc",
    ]  # Material is one of 1 (cast iron) and 2 (stainless steel)

    def __init__(
        self,
        transformX=None,
        transformProc=None,
        split="train",
        split_type="runs",
        split_ratios=(0.64, 0.16, 0.2),
        seed=42,
        signal_group="all",
        debug_plots=False,
        sliding_window_size=250,
        sliding_window_stride=25,
        runs_per_case_test_val=1,
        split_offset=0,
        apply_averaging=True,
        avg_window_size=10,
        windowing=True,
        custom_train_cases=None,
        custom_val_cases=None,
    ):
        """
        Custom DataLoader for the NASA-Ames face-milling dataset with train/val/test split support.

        Args:
            transformX (torch.nn.Module or callable, optional): A transformation function or object applied
                to the signal input data. For example, it could normalize the signals, augment data, or apply
                domain-specific preprocessing. Defaults to None (no transformation).
            transformProc (torch.nn.Module or callable, optional): A transformation function or object applied
                to the process input data. For example, it could normalize the signals, augment data, or apply
                domain-specific preprocessing. Defaults to None (no transformation).
                The split determines which preprocessed file is loaded.
            split_type (str): Specifies the dataset splitting method. Must be one of "paper" or "runs".
            split_ratios (tuple of floats): Proportions used to divide the full dataset into train,
                validation, and test sets. The sum of the ratios should be 1. Defaults to (0.7, 0.15, 0.15).
            seed (int): Seed used for random number generation to ensure reproducible data splitting
                and any other stochastic operations. Defaults to 42.
            signal_group (str): Name of the signal group used during training. This controls which signal
                types are included in the dataset. Must be one of the `allowed_signal_group` values.
            debug_plots (bool): If True, enables debugging features like printing the min/max
                statistics of each signal and generating plots for visualization. Defaults to False.
            sliding_window_size (int): Size of the sliding window (for data augmentation). Defaults to 250.
            sliding_window_stride (int): Stride of the sliding window. Defaults to 25.
            runs_per_case_test_val (int): Number of runs per case assigned to the val and test
                splits when using cyclic splitting. Defaults to 1.
            split_offset (int): Cyclic offset for the first test run index within each case.
                Defaults to 0.
            apply_averaging (bool): If True, apply a moving-average filter to each signal
                before windowing. Defaults to True.
            avg_window_size (int): Kernel size for the moving-average filter. Defaults to 10.
            windowing (bool): If True, apply sliding-window augmentation on load.
                Set to False when the caller will apply windowing externally. Defaults to True.
            custom_train_cases (list[int] | None): Explicit list of case IDs to use for
                training. Overrides the default split when provided. Defaults to None.
            custom_val_cases (list[int] | None): Explicit list of case IDs to use for
                validation. Overrides the default split when provided. Defaults to None.

        Attributes:
            self.data (torch.Tensor): Tensor containing the processed input signals.
            self.proc_data (torch.Tensor): Tensor containing additional processing parameters associated
                with the input data.
            self.targets (torch.Tensor): Tensor containing the target labels for the dataset.

        Raises:
            AssertionError: If `split` is not one of "train", "val", or "test".
            AssertionError: If `split_type` is not one of "paper", "runs" or "cases".
            AssertionError: If `signal_group` is not in the allowed signal groups.

        Notes:
            - The NASA dataset files are expected to be preprocessed and stored in specific locations.
            If the files do not exist, the `_process_dataset()` method is called to generate and save
            the splits.
            - If `debug_plots=True`, signal statistics will be printed to assist with dataset inspection
            and debugging.
        """
        self.transformX = transformX
        self.transformProc = transformProc
        self.split = split
        self.seed = seed
        self.debug_plots = debug_plots
        self.signal_group = signal_group
        self.avg_window_size = avg_window_size

        assert split in [
            "train",
            "val",
            "test",
        ], f'Variable must be one of "train", "val", or "test"'
        assert split_type in [
            "runs",
            "cases",
            "paper",
        ], f'Variable must be one of "paper", "runs", or "cases"'
        assert (
            signal_group in allowed_signal_group
        ), f"Variable must be one of {allowed_signal_group}"

        if windowing:
            data_file_name = f'data/nasa_{sliding_window_size}_{sliding_window_stride}{f"_ma{avg_window_size}" if apply_averaging else ""}.bin'
        else:
            data_file_name = (
                f'data/nasa{f"_ma{avg_window_size}" if apply_averaging else ""}.bin'
            )

        if not os.path.exists(data_file_name):
            print(
                "================== NASA dataset files do not exist\nProcessing and saving splits"
            )
            self._process_dataset(
                data_file_name=data_file_name,
                apply_averaging=apply_averaging,
                avg_window_size=avg_window_size,
            )
        else:
            print(
                "================== NASA dataset files exist\nLoading pre-saved splits"
            )

        # Load dat file with the data
        [x, x_process_params, y] = load(data_file_name)
        print(f"Samples: {x.shape}")

        total_data = len(x)

        if split_type == "runs":
            # Split runs per-case, then join. Ensure at least 1 run per case goes to the test set
            cases = np.unique(x_process_params[:, 0])
            train_pairs = []
            val_pairs = []
            test_pairs = []

            test_ratio = split_ratios[2]
            val_ratio = split_ratios[1]
            train_ratio = split_ratios[0]

            for c in np.sort(cases):
                runs = np.unique(
                    x_process_params[x_process_params[:, 0] == c, 1]
                ).astype(int)
                n = len(runs)
                if n == 0:
                    continue

                rng = np.random.RandomState(self.seed + int(c))
                perm = rng.permutation(n)
                runs_shuffled = runs[perm]

                if runs_per_case_test_val > 0:
                    effective_offset = split_offset % n
                    runs_ordered = np.concatenate(
                        (
                            runs_shuffled[effective_offset:],
                            runs_shuffled[:effective_offset],
                        )
                    )

                    test_sel = runs_ordered[:runs_per_case_test_val]
                    val_sel = runs_ordered[
                        runs_per_case_test_val : 2 * runs_per_case_test_val
                    ]
                    
                    tr = runs_ordered[2 * runs_per_case_test_val :]
                    vl = val_sel

                else:
                    if test_ratio > 0:
                        test_count = max(1, int(np.round(test_ratio * n)))
                        test_count = min(test_count, n)
                    else:
                        test_count = 0
                    test_sel = runs_shuffled[:test_count]

                    remaining = runs_shuffled[test_count:]
                    if val_ratio > 0 and len(remaining) > 1:
                        rel_val = val_ratio / (train_ratio + val_ratio)
                        tr, vl = train_test_split(
                            remaining, test_size=rel_val, random_state=self.seed
                        )
                    else:
                        tr = remaining
                        vl = np.array([], dtype=int)

                for r in tr:
                    train_pairs.append([int(c), int(r)])
                for r in vl:
                    val_pairs.append([int(c), int(r)])
                for r in test_sel:
                    test_pairs.append([int(c), int(r)])

            self.train_runs = (
                np.array(train_pairs, dtype=x_process_params[:, :2].dtype).reshape(
                    (-1, 2)
                )
                if len(train_pairs) > 0
                else np.empty((0, 2))
            )
            self.val_runs = (
                np.array(val_pairs, dtype=x_process_params[:, :2].dtype).reshape(
                    (-1, 2)
                )
                if len(val_pairs) > 0
                else np.empty((0, 2))
            )
            self.test_runs = (
                np.array(test_pairs, dtype=x_process_params[:, :2].dtype).reshape(
                    (-1, 2)
                )
                if len(test_pairs) > 0
                else np.empty((0, 2))
            )

            if split == "train":
                selected_runs = self.train_runs
            elif split == "val":
                selected_runs = self.val_runs
            else:
                selected_runs = self.test_runs

            if selected_runs.size == 0:
                mask = np.zeros(len(x_process_params), dtype=bool)
            else:
                mask = np.any(
                    (x_process_params[:, :2, None] == selected_runs.T).all(axis=1),
                    axis=1,
                )

        elif split_type == "paper":

            # Defined by Zheng 2017 (https://doi.org/10.1109/ICPHM.2017.7998311)
            # (Case 6 excluded)
            test_cases = [11, 12, 15, 16]
            train_pool_cases = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]

            self.test_runs = []
            for c in test_cases:
                runs = np.unique(
                    x_process_params[x_process_params[:, 0] == c, 1]
                ).astype(int)
                for r in runs:
                    self.test_runs.append([c, r])
            self.test_runs = np.array(self.test_runs, dtype=int)

            train_pool_runs = []
            for c in train_pool_cases:
                runs = np.unique(
                    x_process_params[x_process_params[:, 0] == c, 1]
                ).astype(int)
                for r in runs:
                    train_pool_runs.append([c, r])
            train_pool_runs = np.array(train_pool_runs, dtype=int)

            val_cases = custom_val_cases if custom_val_cases is not None else [3, 5]
            train_cases = (
                custom_train_cases
                if custom_train_cases is not None
                else [c for c in train_pool_cases if c not in val_cases]
            )

            self.val_runs = []
            for c in val_cases:
                runs = np.unique(
                    x_process_params[x_process_params[:, 0] == c, 1]
                ).astype(int)
                for r in runs:
                    self.val_runs.append([c, r])
            self.val_runs = np.array(self.val_runs, dtype=int)

            self.train_runs = []
            for c in train_cases:
                runs = np.unique(
                    x_process_params[x_process_params[:, 0] == c, 1]
                ).astype(int)
                for r in runs:
                    self.train_runs.append([c, r])
            self.train_runs = np.array(self.train_runs, dtype=int)

            if split == "train":
                selected_runs = self.train_runs
            elif split == "val":
                selected_runs = self.val_runs
            else:
                selected_runs = self.test_runs

            if selected_runs.size == 0:
                mask = np.zeros(len(x_process_params), dtype=bool)
            else:
                mask = np.any(
                    (x_process_params[:, :2, None] == selected_runs.T).all(axis=1),
                    axis=1,
                )

        x, x_process_params, y = x[mask], x_process_params[mask], y[mask]
        
        _case_col = x_process_params[:, 0:1].copy()
        x_process_params = np.hstack([x_process_params[:, 2:], _case_col])
        print(
            f"================== Data splitted for {split} split: {len(x)/total_data*100}%"
        )
        print(
            f"Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}"
        )

        if debug_plots:
            plot_signals(x, NASA_Dataset.signal_list, signal_chanel=2)

        if windowing:
            x, x_process_params, y = augment_signal(
                x,
                x_process_params,
                y,
                window_size=sliding_window_size,
                stride=sliding_window_stride,
            )
        else:
            print("================== Data NOT augmented (Windowing=False)")

        self.case_labels = x_process_params[:, -1].astype(int)
        x_process_params = x_process_params[:, :-1]

        if debug_plots:
            plot_signals(x, NASA_Dataset.signal_list, signal_chanel=2)

        print(f"================== Data augmented")
        print(
            f"Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}"
        )

        if debug_plots:
            x_stat = np.asarray(x, dtype=float)
            for i, signal in enumerate(NASA_Dataset.signal_list):
                print(
                    f"Signal {signal}: Min = {np.min(x_stat[:,:,i])}, Max = {np.max(x_stat[:,:,i])}"
                )

        x, x_process_params = self.remove_signals(x, x_process_params)

        x_tensor = torch.Tensor(x).to(torch.float)
        proc_tensor = torch.Tensor(x_process_params).to(torch.float)

        if self.transformX is not None:
            print("================== Applying Signal Transform (Eager)")
            x_tensor = self.transformX(x_tensor).to(dtype=torch.float)

        if self.transformProc is not None:
            print("================== Applying Process Transform (Eager)")
            proc_tensor = self.transformProc(proc_tensor).to(dtype=torch.float)

        self.data = x_tensor
        self.proc_data = proc_tensor
        self.targets = torch.Tensor(y).unsqueeze(1).to(torch.float)

        print(f'================== NASA {split} dataset loaded for "{signal_group}"')

    def __len__(self):
        """Return the number of windowed samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return the multimodal input dict and target for sample at *idx*.

        Returns:
            Tuple of (sample_dict, target) where sample_dict has keys
            ``x`` (signal tensor) and ``proc_data``.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sampleProc = self.proc_data[idx]

        sample = {
            "proc_data": sampleProc,
            "x": sample,
        }

        return sample, self.targets[idx]

    def _process_dataset(
        self, apply_averaging=True, avg_window_size=10, data_file_name="nasa.bin"
    ):
        """
        Processes the dataset CSV file. If the expanded CSV file has not been created,
        the mat_to_csv.py script should be run to convert the dataset MAT file into an expanded
        csv file.

        Args:
            apply_averaging (bool): Apply moving average to denoise the signals.
            avg_window_size (int): Size of the moving average window.
            data_file_name (str): Name of the file to save the data.
        """
        csv_file = "data/mill_expanded.csv"
        if not os.path.exists(csv_file):
            mtc.convert_expanded(csv_file=csv_file)
        data_file = pd.read_csv(
            csv_file, sep=";", decimal=".", index_col=0
        ).reset_index(drop=True)
        print("================== NASA dataset loaded from CSV")

        print(f"Shape of signals before filtering bad runs:{data_file.shape}")
        data_file = data_file[(data_file["case"] != 1) | (data_file["run"] <= 15)]
        data_file = data_file[(data_file["case"] != 12) | (data_file["run"] != 1)]
        data_file = data_file[(data_file["case"] != 12) | (data_file["run"] != 12)]
        data_file = data_file[(data_file["case"] != 6)]
        print(f"Shape of signals after filtering bad runs:{data_file.shape}")

        data_file["Vc"] = 200
        data_file["Vc"] = data_file["Vc"].astype(float)
        data_file["VB interpolated"] = None
        data_file["VB interpolated"] = data_file["VB interpolated"].astype(float)
        for x in np.sort(data_file["case"].unique()):
            data_vb = (
                data_file[(data_file["case"] == x)]
                .groupby(["case", "run"])
                .mean()
                .reset_index()
            )
            data_vb = data_vb.interpolate()
            for y in np.sort(data_vb["run"].unique()):
                vb = data_vb.loc[data_vb["run"] == y, "VB"].iloc[0]
                data_file.loc[
                    (data_file["case"] == x) & (data_file["run"] == y),
                    "VB interpolated",
                ] = vb

        data_file = data_file.drop(columns=["time", "VB"]).rename(
            columns={"VB interpolated": "VB"}
        )

        data_file = data_file[(data_file["VB"] <= 0.45)]
        print(f"Shape of signals after filtering tool wear:{data_file.shape}")

        print("================== NASA dataset cleaned and VB interpolated")
        print(f'VB - min: {data_file["VB"].min()} - max: {data_file["VB"].max()}')

        x_proc_params = (
            data_file[["case", "run"] + NASA_Dataset.proc_variable_list]
            .copy()
            .drop_duplicates()
        )
        x_process_params = np.array(x_proc_params)
        for proc_var in NASA_Dataset.proc_variable_list:
            print(
                f"{proc_var} - min: {x_proc_params[proc_var].min()} - max: {x_proc_params[proc_var].max()}"
            )

        y = data_file[["case", "run", "VB"]].copy().drop_duplicates()
        y = np.array(y.drop(columns=["case", "run"])).reshape((-1,))

        print("================== Transposing signals")
        x = NASA_Dataset.transpose_signals(data_file)
        print(f"Shape of signals after transposing:{x.shape}")
        print("================== Dataset splited into X, X_proc and Y")

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list)

        if apply_averaging:
            print("================== Denoising signals with moving average")
            x = apply_moving_average(x, window_size=avg_window_size)
            x = np.array(x)
            print(f"Shape of signals after denoising:{x.shape}")
            print("================== Signals denoised")
        else:
            print("================== Signals were not denoised")

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list)

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list)

        x = np.transpose(x, (0, 2, 1))

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list, signal_chanel=2)

        dump([x, x_process_params, y], data_file_name)

    def get_signal_list(self):
        """Return the list of signal channel names for the active signal group."""
        signals = NASA_Dataset.signal_list.copy()
        if self.signal_group == "DC":
            signals = ["smcDC"]
        elif self.signal_group == "DC_AE":
            signals = ["smcDC", "AE_table"]
        elif self.signal_group == "DC_Vib":
            signals = ["smcDC", "vib_table"]
        elif self.signal_group == "DC_table":
            signals = ["smcDC", "vib_table", "AE_table"]
        elif self.signal_group == "AC":
            signals = ["smcAC"]
        elif self.signal_group == "AC_AE":
            signals = ["smcAC", "AE_table"]
        elif self.signal_group == "AC_Vib":
            signals = ["smcAC", "vib_table"]
        elif self.signal_group == "AC_table":
            signals = ["smcAC", "vib_table", "AE_table"]
        elif self.signal_group == "table":
            signals = ["vib_table", "AE_table"]
        elif self.signal_group == "ACDC":
            signals = ["smcAC", "smcDC"]
        elif self.signal_group == "all":
            signals = [
                "smcAC",
                "smcDC",
                "vib_table",
                "vib_spindle",
                "AE_table",
                "AE_spindle",
            ]
        return signals

    def remove_signals(self, data, proc_data=None):
        """Drop signal channels not in the active signal group from data arrays.

        Args:
            data:      Signal array of shape (N, C, T); channels on axis 2.
            proc_data: Optional process-parameter array of shape (N, P).

        Returns:
            Pruned ``data`` array, or (pruned data, pruned proc_data) tuple.
        """
        idx = []
        signals = self.get_signal_list()

        # Find indices of signals to remove
        idx = [
            index
            for index, element in enumerate(NASA_Dataset.signal_list)
            if element not in signals
        ]

        # Remove from signal data
        x = np.delete(data, idx, axis=2)
        if proc_data is not None:
            return x, proc_data
        else:
            return x

    def transpose_signals(data):
        """Reorder raw run data from row-per-timestep to a list of (channels × time) arrays.

        Args:
            data: Pandas DataFrame with columns ``case``, ``run``, and one
                  column per entry in NASA_Dataset.signal_list.

        Returns:
            List of signal arrays, one per (case, run) pair, each of shape
            (n_channels, n_timesteps).
        """
        x = []

        for case in tqdm(np.sort(data["case"].unique())):
            data_case = data[(data["case"] == case)]
            for run in np.sort(data_case["run"].unique()):
                data_run = data_case[(data_case["run"] == run)]

                signals = []
                for col in NASA_Dataset.signal_list:
                    signals.append([])

                for _, row in data_run.iterrows():
                    for i, col in enumerate(NASA_Dataset.signal_list):
                        signals[i].append(row[col])
                x.append(signals)

        return np.array(x)


def get_nasa_data_pipeline(
    run_name,
    window_size,
    stride,
    split_type="runs",
    split_ratios=(0.64, 0.16, 0.2),
    seed=42,
    stats_strategy="robust",
    avg_window_size=10,
    apply_averaging=False,
    runs_per_case_test_val=1,
    split_offset=0,
    windowing=False,
    debug_plots=False,
    custom_train_cases=None,
    custom_val_cases=None,
):
    """
    Unified factory for NASA Datasets.

    Args:
        run_name (str): Signal group name (e.g., 'AC', 'DC', 'all').
        window_size (int): Sliding window size.
        stride (int): Sliding window stride.
        split_type (str): 'runs' (default) or 'paper'.
        split_ratios (tuple): (train, val, test) ratios. Legacy/Optional if runs_per_case used.
        seed (int): Random seed.
        stats_strategy (str): 'robust' (Winsorization) or 'standard' (Raw Mean/Std).
        avg_window_size (int): Moving average window size.
        apply_averaging (bool): Whether to apply moving average.
        runs_per_case_test_val (int): Runs per case for val/test (activates cyclic split).
        split_offset (int): Offset for cyclic split.
        windowing (bool): Whether to apply windowing on load (default False for pipelines).
        debug_plots (bool): Enable debug plots.
        custom_train_cases (list[int] | None): Explicit list of case IDs used for
            training. Overrides the default split when provided.
        custom_val_cases (list[int] | None): Explicit list of case IDs used for
            validation. Pass ``[]`` for an empty validation set (retrain mode).

    Returns:
        (train_ds, val_ds, test_ds, transforms_dict)
    """

    print(f"--- [Pipeline] Loading Raw Train Data for Stats ({stats_strategy}) ---")
    train_ds_raw = NASA_Dataset(
        split="train",
        signal_group=run_name,
        split_ratios=split_ratios,
        sliding_window_size=window_size,
        sliding_window_stride=stride,
        seed=seed,
        split_type=split_type,
        runs_per_case_test_val=runs_per_case_test_val,
        split_offset=split_offset,
        apply_averaging=False,
        avg_window_size=avg_window_size,
        windowing=False,
        debug_plots=False,
        custom_train_cases=custom_train_cases,
        custom_val_cases=custom_val_cases,
    )

    x_raw_np = train_ds_raw.data.numpy()

    if stats_strategy == "robust":
        # Robust Winsorized Statistics
        # 1st and 99th Percentiles
        p1 = np.percentile(x_raw_np, 1, axis=(0, 1))
        p99 = np.percentile(x_raw_np, 99, axis=(0, 1))
        print(f"--- [Pipeline] Winsorization Stats - 1st: {p1}, 99th: {p99}")

        x_clipped = np.clip(x_raw_np, p1, p99)
        mean = np.mean(x_clipped, axis=(0, 1))
        std = np.std(x_clipped, axis=(0, 1))

    elif stats_strategy == "standard":
        # Standard Statistics (Raw Mean/Std)
        mean = np.mean(x_raw_np, axis=(0, 1))
        std = np.std(x_raw_np, axis=(0, 1))

    else:
        raise ValueError(f"Unknown stats_strategy: {stats_strategy}")

    print(f"--- [Pipeline] Stats - Mean: {mean}, Std: {std}")

    mean_tensor = torch.tensor(mean).float()
    std_tensor = torch.tensor(std).float()

    normalize_transform = StdScalerTransform(mean=mean_tensor, std=std_tensor)

    proc_min_t = NASA_PROC_MIN
    proc_max_t = NASA_PROC_MAX
    minmax_transform = MinMaxScalerTransform(min=proc_min_t, max=proc_max_t)

    transforms = {
        "x": normalize_transform,
        "proc": minmax_transform,
        "mean": mean,
        "std": std,
    }

    print("--- [Pipeline] Instantiating Final Datasets ---")

    ds_train = NASA_Dataset(
        split="train",
        signal_group=run_name,
        split_ratios=split_ratios,
        transformX=normalize_transform,
        transformProc=minmax_transform,
        sliding_window_size=window_size,
        sliding_window_stride=stride,
        seed=seed,
        split_type=split_type,
        runs_per_case_test_val=runs_per_case_test_val,
        split_offset=split_offset,
        apply_averaging=apply_averaging,
        avg_window_size=avg_window_size,
        windowing=windowing,
        debug_plots=debug_plots,
        custom_train_cases=custom_train_cases,
        custom_val_cases=custom_val_cases,
    )

    ds_val = NASA_Dataset(
        split="val",
        signal_group=run_name,
        split_ratios=split_ratios,
        transformX=normalize_transform,
        transformProc=minmax_transform,
        sliding_window_size=window_size,
        sliding_window_stride=stride,
        seed=seed,
        split_type=split_type,
        runs_per_case_test_val=runs_per_case_test_val,
        split_offset=split_offset,
        apply_averaging=apply_averaging,
        avg_window_size=avg_window_size,
        windowing=windowing,
        debug_plots=debug_plots,
        custom_train_cases=custom_train_cases,
        custom_val_cases=custom_val_cases,
    )

    ds_test = NASA_Dataset(
        split="test",
        signal_group=run_name,
        split_ratios=split_ratios,
        transformX=normalize_transform,
        transformProc=minmax_transform,
        sliding_window_size=window_size,
        sliding_window_stride=stride,
        seed=seed,
        split_type=split_type,
        runs_per_case_test_val=runs_per_case_test_val,
        split_offset=split_offset,
        apply_averaging=apply_averaging,
        avg_window_size=avg_window_size,
        windowing=windowing,
        debug_plots=debug_plots,
    )

    transforms["case_labels"] = ds_train.case_labels

    return ds_train, ds_val, ds_test, transforms


__all__ = [
    "NASA_Dataset",
    "get_nasa_data_pipeline",
    "NASA_PROC_MIN",
    "NASA_PROC_MAX",
    "N_BASE_PROC",
]
