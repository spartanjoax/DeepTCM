################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Custom Keras Tuner implementing the PreprocessingRandom joint HP search.

Each outer trial samples preprocessing choices (scalogram type, normalisation,
noise reduction, window parameters) alongside architecture HPs, then fits an
inner model and scores by RMSE. Supports standard val split and 3-fold LOCO-CV.

Use PreprocessingRandom (random search). PreprocessingBayesian is disabled
because its Bayesian acquisition function scores against test RMSE, causing
data leakage.
"""
import keras_tuner as kt
import numpy as np
import os
import torch
import gc
import json

from helpers import evaluate_model
from helpers.preprocessing import apply_full_preprocessing
from models.dl_models import (
    get_callbacks,
    get_model,
)

import time
import keras

NUM_REPS = 3

def resolve_model_name(model_type_str, arch_type_str="cnn_rnn", rnn_type_str="lstm"):
    """Map tuner model_type + arch_type to a dl_models model name string.

    Args:
        model_type_str (str): Tuner model-type key (e.g. ``'sota_search'``,
            ``'cnn'``, ``'resnet'``, ``'lstm'``, ``'cnn_lstm'``).
        arch_type_str (str): Architecture sub-type
            (e.g. ``'cnn_rnn'``, ``'rnn_only'``). Defaults to ``'cnn_rnn'``.
        rnn_type_str (str): RNN cell type (``'lstm'`` or ``'bigru'``).
            Defaults to ``'lstm'``.

    Returns:
        str: Model name recognised by ``dl_models`` factory functions
        (e.g. ``'CNN_LSTM'``, ``'BiGRU'``, ``'CNN'``).
    """
    if model_type_str in ["sota_search", "sota_search_max"]:
        if arch_type_str == "rnn_only":
            return "BiGRU" if rnn_type_str == "bigru" else "LSTM"
        return "CNN_LSTM"
    _map = {
        "cnn": "CNN", "resnet": "ResNet", "lstm": "LSTM",
        "bilstm": "BiGRU", "cnn_lstm": "CNN_LSTM",
        "cnn_lstm_capacity": "CNN_LSTM",
    }
    return _map.get(model_type_str, "CNN")

class PreprocessingTrial:
    """
    Class containing the core logic for run_trial.
    This allows us to reuse the same trial logic for both BayesianOptimization
    and Hyperband tuners.
    """

    def run_trial(self, trial, x, y, validation_data=None, **kwargs):
        """
        Custom trial execution.
        1. Loads preprocessing Hyperparameters.
        2. Applies preprocessing transformations.
        3. Builds the inner model.
        4. Fits the inner model.

        Args:
            trial (kt.Trial): The trial instance.
            x (np.ndarray or list): Input data.
            y (np.ndarray): Target data.
            validation_data (tuple): Validation data (x_val, y_val).
            **kwargs: Additional arguments for training.

        Returns:
            dict: Dictionary containing evaluation metrics (e.g., {'rmse': score}).
        """
        x_train_final = None
        x_val_final = None
        x_test_final = None
        y_val = None
        y_test = None

        model = None
        best_model = None
        history = None

        try:
            hp = trial.hyperparameters
            ma_window_size = kwargs.get("ma_window_size", 10)

            noise_reduction = hp.get("noise_reduction")
            scalogram_type = hp.get("scalogram")

            if self.model_type in ["sota_search", "sota_search_max"]:

                rnn_type = hp.get("rnn_type")
                arch_type = hp.values.get("arch_type", "cnn_rnn")
                filters_base = hp.values.get("filters_base", 0)
                lstm_units = hp.get("lstm_units")
                dropout_rate = hp.get("dropout")
                weight_decay = hp.values.get("weight_decay", 1e-5)
                learning_rate = hp.values.get("learning_rate", 1e-3)
                pooling_type = hp.values.get("pooling", "avg")
                conditioning = hp.values.get("conditioning", "no")
                lstm_layers = hp.values.get("lstm_layers", 2)
                cnn_layers = hp.values.get("cnn_layers", 3)
                jitter_sigma = hp.values.get("jitter_sigma", 0.0)
            else:
                rnn_type = "lstm"
                arch_type = "cnn_rnn"
                filters_base = 32
                lstm_units = 64
                lstm_layers = 2
                cnn_layers = 3
                dropout_rate = 0.3
                weight_decay = 1e-5
                learning_rate = 1e-3
                pooling_type = "avg"
                conditioning = "no"
                jitter_sigma = 0.0

            if self.model_type == "cnn_lstm_capacity":
                scalogram_type = "none"

            downsample_factor = kwargs.pop("downsample_factor", 1)
            preprocessing_hps = {
                "noise_reduction": noise_reduction,
                "scalogram": scalogram_type,
                "ma_window_size": ma_window_size,
                "downsample_factor": downsample_factor,
                "jitter_sigma": jitter_sigma,
            }

            print(f"Running Trial Preprocessing: {preprocessing_hps}")

            sw_config = {
                "window_size": kwargs.pop("sliding_window_size", 250),
                "stride": kwargs.pop("sliding_window_stride", 125),
            }

            cv_folds = kwargs.pop("cv_folds", None)
            case_labels = kwargs.pop("case_labels", None)

            y_orig = y

            x_train_final, y, x_train_proc, preproc_stats = apply_full_preprocessing(
                x,
                y,
                preprocessing_hps,
                split="train",
                preproc_stats=None,
                sliding_window_config=sw_config,
            )

            if validation_data is not None:
                x_val_in, y_val_in = validation_data
                x_val_final, y_val, x_val_proc, _ = apply_full_preprocessing(
                    x_val_in,
                    y_val_in,
                    preprocessing_hps,
                    split="val",
                    preproc_stats=preproc_stats,
                    sliding_window_config=sw_config,
                )
            else:
                x_val_final = None
                x_val_proc = x_val_feat = None

            if "test_data" in kwargs:
                x_test_in, y_test_in = kwargs.get("test_data")
                x_test_final, y_test, x_test_proc, _ = apply_full_preprocessing(
                    x_test_in,
                    y_test_in,
                    preprocessing_hps,
                    split="test",
                    preproc_stats=preproc_stats,
                    sliding_window_config=sw_config,
                )
            else:
                x_test_final = None
                x_test_proc = x_test_feat = None

            kwargs_fit = kwargs.copy()
            if "test_data" in kwargs_fit:
                kwargs_fit.pop("test_data")
            if "ma_window_size" in kwargs_fit:
                kwargs_fit.pop("ma_window_size")

            def _shape(arr):
                """Return arr.shape or None (safe for None arrays)."""
                return arr.shape if arr is not None else None
            print(
                f"[shapes] "
                f"x_train={_shape(x_train_final)}  proc={_shape(x_train_proc)} | "
                f"x_val={_shape(x_val_final)}  proc={_shape(x_val_proc)} | "
                f"x_test={_shape(x_test_final)}  proc={_shape(x_test_proc)}"
            )

            if cv_folds is not None and case_labels is not None:
                _cv_target_model = resolve_model_name(self.model_type, arch_type, rnn_type)
                _cv_trial_hps = {
                    "filters_base": filters_base,
                    "lstm_units": lstm_units,
                    "lstm_layers": lstm_layers,
                    "cnn_layers": cnn_layers,
                    "dropout": dropout_rate,
                    "weight_decay": weight_decay,
                    "rnn_type": rnn_type,
                    "pooling": pooling_type,
                    "learning_rate": learning_rate,
                    "scalogram": scalogram_type if scalogram_type != "none" else "none",
                }

                _cv_epochs = kwargs_fit.get("epochs", 1)
                if "tuner/epochs" in hp.values:
                    _cv_epochs = hp.values["tuner/epochs"]

                trial_folder = (
                    f"{self.directory}/{self.project_name}/preproc_trial_{trial.trial_id}"
                )
                if not os.path.exists(trial_folder):
                    os.makedirs(trial_folder)

                fold_rmse_scores = []
                fold_test_scores = []
                fold_train_scores = []
                _cv_best_fold_score = float("inf")
                _fold_times = []
                _trial_start_time = time.time()

                def _cv_squeeze_signal(arr):
                    """Remove degenerate trailing dims from a 4-D CV signal array."""
                    if arr.ndim == 4 and arr.shape[-1] == 1:
                        arr = np.squeeze(arr, axis=-1)
                    if arr.ndim == 4 and arr.shape[-2] == 1:
                        arr = np.squeeze(arr, axis=-2)
                    return arr

                def _cv_squeeze_split(data):
                    """Apply _cv_squeeze_signal to the signal element of a list or array."""
                    if isinstance(data, list):
                        data[0] = _cv_squeeze_signal(data[0])
                        return data
                    return _cv_squeeze_signal(data)

                for fold_idx, fold_val_cases in enumerate(cv_folds):
                    print(
                        f">>> CV Fold {fold_idx + 1}/{len(cv_folds)} "
                        f"\u2014 held-out cases: {fold_val_cases}"
                    )
                    fold_train_mask = ~np.isin(case_labels, fold_val_cases)
                    fold_val_mask = np.isin(case_labels, fold_val_cases)

                    if isinstance(x, list):
                        x_fold_train_in = [arr[fold_train_mask] for arr in x]
                        x_fold_val_in = [arr[fold_val_mask] for arr in x]
                    else:
                        x_fold_train_in = x[fold_train_mask]
                        x_fold_val_in = x[fold_val_mask]

                    y_fold_train = y_orig[fold_train_mask]
                    y_fold_val = y_orig[fold_val_mask]

                    _fold_preproc_start = time.time()
                    x_fold_train_pp, y_fold_train_pp, x_fold_proc_pp, fold_preproc_stats = (
                        apply_full_preprocessing(
                            x_fold_train_in,
                            y_fold_train,
                            preprocessing_hps,
                            split="train",
                            preproc_stats=None,
                            sliding_window_config=sw_config,
                        )
                    )
                    x_fold_val_pp, y_fold_val_pp, x_fold_val_proc, _ = apply_full_preprocessing(
                        x_fold_val_in,
                        y_fold_val,
                        preprocessing_hps,
                        split="val",
                        preproc_stats=fold_preproc_stats,
                        sliding_window_config=sw_config,
                    )
                    _fold_preproc_time = time.time() - _fold_preproc_start

                    if scalogram_type == "none":
                        x_fold_train_pp = _cv_squeeze_split(x_fold_train_pp)
                        x_fold_val_pp = _cv_squeeze_split(x_fold_val_pp)

                    _fold_signal = x_fold_train_pp
                    _fold_proc = x_fold_proc_pp
                    _use_proc_cv = conditioning != "no" and _fold_proc is not None
                    _fold_suffix = "_Film" if conditioning == "film" else ("_Proc" if conditioning == "proc" else "")
                    _fold_proc_shape = (1, _fold_proc.shape[1]) if _use_proc_cv else None

                    _fold_dummy_shape = (1, *_fold_signal.shape[1:])
                    _fold_name = f"{_cv_target_model}_{trial.trial_id}_fold{fold_idx}"

                    if _use_proc_cv:
                        fold_model = get_model(
                            model_name=f"{_cv_target_model}{_fold_suffix}",
                            input_shape=_fold_dummy_shape,
                            hps=_cv_trial_hps,
                            proc_shape=_fold_proc_shape,
                            name=_fold_name,
                        )
                    else:
                        fold_model = get_model(
                            model_name=_cv_target_model,
                            input_shape=_fold_dummy_shape,
                            hps=_cv_trial_hps,
                            name=_fold_name,
                        )

                    fold_callbacks = get_callbacks(
                        patience=50,
                        min_delta=0.0001,
                        monitor="val_loss",
                        mode="min",
                        reduce_lr=True,
                        cleanup_memory=True,
                        patience_lr=5,
                        factor_lr=0.5,
                        min_lr=1e-6,
                    )

                    _fold_fit_kwargs = {
                        k: v for k, v in kwargs_fit.items()
                        if k not in ["epochs", "initial_epoch"]
                    }

                    _fold_fit_start = time.time()
                    
                    def _fold_assemble(sig, proc):
                        """Assemble single or multi-input list from signal and proc arrays for a CV fold.

                        Args:
                            sig (np.ndarray): Preprocessed signal array.
                            proc (np.ndarray | None): Process-parameter array, or None to exclude.

                        Returns:
                            np.ndarray or list: ``sig`` alone if ``proc`` is None,
                            otherwise ``[sig, proc]``.
                        """
                        inp = [sig]
                        if proc is not None: inp.append(proc)
                        return inp[0] if len(inp) == 1 else inp
                    
                    x_fold_train_fit = _fold_assemble(x_fold_train_pp, _fold_proc if _use_proc_cv else None)
                    x_fold_val_fit   = _fold_assemble(x_fold_val_pp, x_fold_val_proc if _use_proc_cv else None)

                    fold_history = fold_model.fit(
                        x_fold_train_fit,
                        y_fold_train_pp,
                        validation_data=(x_fold_val_fit, y_fold_val_pp),
                        callbacks=fold_callbacks,
                        epochs=_cv_epochs,
                        **_fold_fit_kwargs,
                    )
                    _fold_fit_time = time.time() - _fold_fit_start
                    _fold_total_time = _fold_preproc_time + _fold_fit_time

                    fold_val_score = np.nan
                    fold_train_score = np.nan
                    best_epoch_idx = None
                    if "val_root_mse" in fold_history.history:
                        best_epoch_idx = int(np.argmin(fold_history.history["val_root_mse"]))
                        fold_val_score = fold_history.history["val_root_mse"][best_epoch_idx]
                    elif "val_root_mean_squared_error" in fold_history.history:
                        best_epoch_idx = int(np.argmin(fold_history.history["val_root_mean_squared_error"]))
                        fold_val_score = fold_history.history["val_root_mean_squared_error"][best_epoch_idx]
                    elif "val_loss" in fold_history.history:
                        best_epoch_idx = int(np.argmin(fold_history.history["val_loss"]))
                        fold_val_score = fold_history.history["val_loss"][best_epoch_idx]

                    if best_epoch_idx is not None:
                        if "root_mse" in fold_history.history:
                            fold_train_score = fold_history.history["root_mse"][best_epoch_idx]
                        elif "root_mean_squared_error" in fold_history.history:
                            fold_train_score = fold_history.history["root_mean_squared_error"][best_epoch_idx]

                    fold_test_score = np.nan
                    if "test_data" in kwargs:
                        _test_raw_x, _test_raw_y = kwargs["test_data"]
                        x_fold_test_pp, y_fold_test_pp, x_fold_test_proc, _ = (
                            apply_full_preprocessing(
                                _test_raw_x,
                                _test_raw_y,
                                preprocessing_hps,
                                split="test",
                                preproc_stats=fold_preproc_stats,
                                sliding_window_config=sw_config,
                            )
                        )
                        if scalogram_type == "none":
                            x_fold_test_pp = _cv_squeeze_split(x_fold_test_pp)
                        x_fold_test_fit = _fold_assemble(
                            x_fold_test_pp,
                            x_fold_test_proc if _use_proc_cv else None,
                        )
                        _fold_test_pred = fold_model.predict(x_fold_test_fit, verbose=0)
                        fold_test_score = float(np.sqrt(np.mean(
                            (y_fold_test_pp.flatten() - _fold_test_pred.flatten()) ** 2
                        )))

                    print(
                        f">>> Fold {fold_idx + 1} "
                        f"Train RMSE: {fold_train_score:.4f} | "
                        f"Val RMSE: {fold_val_score:.4f} | "
                        f"Test RMSE: {fold_test_score:.4f} "
                        f"| preproc={_fold_preproc_time:.1f}s "
                        f"fit={_fold_fit_time:.1f}s "
                        f"total={_fold_total_time:.1f}s"
                    )
                    _fold_times.append({
                        "preproc_s": round(_fold_preproc_time, 2),
                        "fit_s": round(_fold_fit_time, 2),
                        "total_s": round(_fold_total_time, 2),
                    })
                    if not np.isnan(fold_val_score):
                        fold_rmse_scores.append(fold_val_score)
                        if fold_val_score < _cv_best_fold_score:
                            _cv_best_fold_score = fold_val_score
                            _champion_path = os.path.join(
                                trial_folder,
                                f"trial_{trial.trial_id}_best.weights.h5",
                            )
                            fold_model.save_weights(_champion_path)
                            print(
                                f">>> New best weights saved "
                                f"(fold {fold_idx+1}, RMSE={fold_val_score:.4f}): "
                                f"{_champion_path}"
                            )
                    if not np.isnan(fold_test_score):
                        fold_test_scores.append(fold_test_score)
                    if not np.isnan(fold_train_score):
                        fold_train_scores.append(fold_train_score)

                    _fold_history_ser = {
                        k: [float(v) for v in vals]
                        for k, vals in fold_history.history.items()
                    }
                    _fold_progress = {
                        "trial_id": trial.trial_id,
                        "fold": fold_idx,
                        "fold_val_cases": fold_val_cases,
                        "fold_val_rmse": (
                            float(fold_val_score)
                            if not np.isnan(fold_val_score)
                            else None
                        ),
                        "fold_test_rmse": (
                            float(fold_test_score)
                            if not np.isnan(fold_test_score)
                            else None
                        ),
                        "fold_train_rmse": (
                            float(fold_train_score)
                            if not np.isnan(fold_train_score)
                            else None
                        ),
                        "preproc_time_s": round(_fold_preproc_time, 2),
                        "fit_time_s": round(_fold_fit_time, 2),
                        "total_time_s": round(_fold_total_time, 2),
                        "scalogram_type": scalogram_type,
                        "noise_reduction": noise_reduction,
                        "rnn_type": rnn_type,
                        "filters_base": int(filters_base),
                        "lstm_units": int(lstm_units),
                        "dropout": float(dropout_rate),
                        "weight_decay": float(weight_decay),
                        "arch_type": arch_type,
                        "lstm_layers": int(lstm_layers),
                        "cnn_layers": int(cnn_layers),
                        "jitter_sigma": float(jitter_sigma),
                        "conditioning": conditioning,
                        "learning_rate": float(learning_rate),
                        "pooling": pooling_type,
                        "history": _fold_history_ser,
                    }
                    _fold_progress_path = os.path.join(
                        trial_folder,
                        f"DL_trial_{trial.trial_id}_fold{fold_idx}_progress.json",
                    )
                    with open(_fold_progress_path, "w") as _fp:
                        json.dump(_fold_progress, _fp, indent=2)

                    del fold_model
                    keras.utils.clear_session()
                    gc.collect()
                    torch.cuda.empty_cache()

                rmse_score = np.mean(fold_test_scores) if fold_test_scores else (
                    np.mean(fold_rmse_scores) if fold_rmse_scores else 1e9
                )
                mean_val = np.mean(fold_rmse_scores) if fold_rmse_scores else np.nan
                mean_train = np.mean(fold_train_scores) if fold_train_scores else np.nan
                
                trial_exceeds_threshold = bool(rmse_score > 0.09)
                print(
                    f"--- Trial {trial.trial_id} CV Complete "
                    f"- Mean Test RMSE (Score): {rmse_score:.4f} "
                    f"| Mean Val RMSE: {mean_val:.4f} "
                    f"| Mean Train RMSE: {mean_train:.4f} "
                    f"| Exceeds Threshold (mean > 0.09): {trial_exceeds_threshold}"
                )

                _trial_total_time = time.time() - _trial_start_time
                _total_preproc = sum(t["preproc_s"] for t in _fold_times)
                _total_fit = sum(t["fit_s"] for t in _fold_times)

                _cv_summary = {
                    "trial_id": f"{trial.trial_id}_cv",
                    "run_name": f"trial_{trial.trial_id}_cv",
                    "training_time": round(_trial_total_time, 2),
                    "total_preproc_time_s": round(_total_preproc, 2),
                    "total_fit_time_s": round(_total_fit, 2),
                    "fold_times": _fold_times,
                    "scalogram_type": scalogram_type,
                    "noise_reduction": noise_reduction,
                    "rnn_type": rnn_type,
                    "filters_base": int(filters_base),
                    "lstm_units": int(lstm_units),
                    "dropout": float(dropout_rate),
                    "weight_decay": float(weight_decay),
                    "arch_type": arch_type,
                    "lstm_layers": int(lstm_layers),
                    "cnn_layers": int(cnn_layers),
                    "jitter_sigma": float(jitter_sigma),
                    "conditioning": conditioning,
                    "learning_rate": float(learning_rate),
                    "pooling": pooling_type,
                    "rmse": float(rmse_score),
                    "mean_test_rmse": float(rmse_score),
                    "mean_val_rmse": float(mean_val) if not np.isnan(mean_val) else None,
                    "mean_train_rmse": float(mean_train) if not np.isnan(mean_train) else None,
                    "cv_fold_val_rmses": [float(s) for s in fold_rmse_scores],
                    "cv_fold_test_rmses": [float(s) for s in fold_test_scores],
                    "cv_fold_train_rmses": [float(s) for s in fold_train_scores],
                    "exceeds_threshold": trial_exceeds_threshold,
                    "history": {
                        "val_root_mse": [float(mean_val) if not np.isnan(mean_val) else 1e9],
                        "root_mse": [float(mean_train) if not np.isnan(mean_train) else 1e9],
                    },
                    "best_inner_trial_id": "cv",
                    "inner_best_hps": {},
                }
                _cv_summary_path = os.path.join(
                    trial_folder,
                    f"DL_trial_{trial.trial_id}_cv_rep0.json",
                )
                with open(_cv_summary_path, "w") as _fp:
                    json.dump(_cv_summary, _fp, indent=2)
                print(f"--- CV summary saved: {_cv_summary_path}")

            x_train_fit = x_train_final
            x_val_fit   = x_val_final
            x_test_fit  = x_test_final

            def _squeeze_signal(arr):
                """Remove degenerate trailing dims from a 4-D signal array."""
                if arr.ndim == 4 and arr.shape[-1] == 1:
                    arr = np.squeeze(arr, axis=-1)
                if arr.ndim == 4 and arr.shape[-2] == 1:
                    arr = np.squeeze(arr, axis=-2)
                return arr

            def _squeeze_split(data):
                """Apply _squeeze_signal to the signal element of a list or array."""
                if isinstance(data, list):
                    data[0] = _squeeze_signal(data[0])
                    return data
                return _squeeze_signal(data)

            if scalogram_type == "none":
                x_train_fit = _squeeze_split(x_train_fit)
                if x_val_fit is not None:
                    x_val_fit = _squeeze_split(x_val_fit)
                if x_test_fit is not None:
                    x_test_fit = _squeeze_split(x_test_fit)

            x_train_signal = x_train_fit

            def _tuner_assemble(sig, proc):
                """Assemble single or multi-input list from signal, proc, and feat arrays for the tuner path.

                Args:
                    sig (np.ndarray): Preprocessed signal array.
                    proc (np.ndarray | None): Process-parameter array, or None to exclude.

                Returns:
                    np.ndarray or list: ``sig`` alone if ``proc`` is None,
                    otherwise ``[sig, proc]``.
                """
                inp = [sig]
                if proc is not None: inp.append(proc)
                return inp[0] if len(inp) == 1 else inp

            _use_proc_data = conditioning != "no" and x_train_proc is not None
            x_train_fit = _tuner_assemble(x_train_fit, x_train_proc if _use_proc_data else None)
            if x_val_fit is not None:
                x_val_fit  = _tuner_assemble(x_val_fit,  x_val_proc if _use_proc_data else None)
            if x_test_fit is not None:
                x_test_fit = _tuner_assemble(x_test_fit, x_test_proc if _use_proc_data else None)

            input_shape = x_train_signal.shape
            dummy_shape = (1, *input_shape[1:])

            reps_rmse = []

            trial_folder = f"{self.directory}/{self.project_name}/preproc_trial_{trial.trial_id}"
            if not os.path.exists(trial_folder):
                os.makedirs(trial_folder)

            tuner_epochs = kwargs_fit.get("epochs", 1)
            tuner_initial_epoch = kwargs_fit.get("initial_epoch", 0)

            if "tuner/epochs" in hp.values:
                tuner_epochs = hp.values["tuner/epochs"]

            if "tuner/initial_epoch" in hp.values:
                tuner_initial_epoch = hp.values["tuner/initial_epoch"]

            parent_trial_id = hp.values.get("tuner/trial_id", None)

            current_reps = NUM_REPS if tuner_initial_epoch == 0 else 1

            print(
                f"--- Starting {current_reps} repetitions for Trial {trial.trial_id} ---"
            )
            print(
                f"    > Initial Epoch: {tuner_initial_epoch}, Target Epochs: {tuner_epochs}"
            )
            if parent_trial_id:
                print(f"    > Resuming from Parent: {parent_trial_id}")

            reps_val_losses = []
            reps_val_scores = []
            reps_train_scores = []
            best_rep_val_loss = float("inf")

            for rep in range(current_reps):
                print(f">>> Repetition {rep+1}/{current_reps}...")

                trial_hps = {
                    "filters_base": filters_base,
                    "lstm_units": lstm_units,
                    "lstm_layers": lstm_layers,
                    "cnn_layers": cnn_layers,
                    "dropout": dropout_rate,
                    "weight_decay": weight_decay,
                    "rnn_type": rnn_type,
                    "pooling": pooling_type,
                    "learning_rate": learning_rate,
                    "scalogram": (
                        scalogram_type if scalogram_type != "none" else "none"
                    ),
                }

                target_model_name = resolve_model_name(self.model_type, arch_type, rnn_type)
                unique_name = f"{target_model_name}_{trial.trial_id}_rep{rep}"

                _use_proc_trial = conditioning != "no" and x_train_proc is not None
                _trial_suffix = "_Film" if conditioning == "film" else ("_Proc" if conditioning == "proc" else "")
                _proc_shape = (1, x_train_proc.shape[1]) if _use_proc_trial else None
                if _use_proc_trial:
                    model = get_model(
                        model_name=f"{target_model_name}{_trial_suffix}",
                        input_shape=dummy_shape,
                        hps=trial_hps,
                        proc_shape=_proc_shape,
                        name=unique_name,
                    )
                else:
                    model = get_model(
                        model_name=target_model_name,
                        input_shape=dummy_shape,
                        hps=trial_hps,
                        name=unique_name,
                    )

                if tuner_initial_epoch > 0 and parent_trial_id:
                    parent_folder = f"{self.directory}/{self.project_name}/preproc_trial_{parent_trial_id}"
                    parent_weights = os.path.join(
                        parent_folder, f"trial_{parent_trial_id}_best.weights.h5"
                    )

                    if os.path.exists(parent_weights):
                        print(f"    > Loading Parent Weights: {parent_weights}")
                        model.load_weights(parent_weights)
                    else:
                        print(
                            f"    > WARNING: Parent weights not found at {parent_weights}. Starting from epoch 0 state (but clock at {tuner_initial_epoch})."
                        )

                callbacks_list = get_callbacks(
                    patience=50,
                    min_delta=0.0001,
                    monitor=(
                        "val_loss"
                        if x_val_final is not None and y_val is not None
                        else "loss"
                    ),
                    mode="min",
                    reduce_lr=True,
                    cleanup_memory=True,
                    patience_lr=5,
                    factor_lr=0.5,
                    min_lr=1e-6,
                )

                start_time = time.time()
                clean_fit_kwargs = {
                    k: v
                    for k, v in kwargs_fit.items()
                    if k
                    not in ["epochs", "initial_epoch"]
                }

                if x_val_final is not None and y_val is not None:
                    history = model.fit(
                        x_train_fit,
                        y,
                        validation_data=(x_val_fit, y_val),
                        callbacks=callbacks_list,
                        epochs=tuner_epochs,
                        initial_epoch=tuner_initial_epoch,
                        **clean_fit_kwargs,
                    )
                else:
                    history = model.fit(
                        x_train_fit,
                        y,
                        callbacks=callbacks_list,
                        epochs=tuner_epochs,
                        initial_epoch=tuner_initial_epoch,
                        **clean_fit_kwargs,
                    )

                training_time = time.time() - start_time

                val_loss = min(history.history["val_loss"])
                val_score = val_loss

                rmse_rep = np.nan

                should_evaluate = True

                if np.isnan(val_loss):
                    print(f"Rep {rep+1} failed with NaN loss. Skipping evaluation.")
                    rmse_rep = np.nan
                    should_evaluate = False
                elif np.isinf(val_loss):
                    print(
                        f"Rep {rep+1} failed with Inf loss. Setting to large value."
                    )
                    rmse_rep = 1e9
                    should_evaluate = False

                if not np.isnan(val_loss):
                    reps_val_losses.append(val_loss)

                    best_epoch_idx = None
                    if "val_root_mse" in history.history:
                        val_rmse_hist = history.history["val_root_mse"]
                        best_epoch_idx = int(np.argmin(val_rmse_hist))
                        val_score = val_rmse_hist[best_epoch_idx]
                    elif "val_root_mean_squared_error" in history.history:
                        val_rmse_hist = history.history["val_root_mean_squared_error"]
                        best_epoch_idx = int(np.argmin(val_rmse_hist))
                        val_score = val_rmse_hist[best_epoch_idx]

                    train_score = np.nan
                    if best_epoch_idx is not None:
                        if "root_mse" in history.history:
                            train_score = history.history["root_mse"][best_epoch_idx]
                        elif "root_mean_squared_error" in history.history:
                            train_score = history.history["root_mean_squared_error"][best_epoch_idx]

                    reps_val_scores.append(val_score)
                    reps_train_scores.append(train_score)

                    if val_loss < best_rep_val_loss:
                        best_rep_val_loss = val_loss
                        best_model = model

                        champion_path = os.path.join(
                            trial_folder, f"trial_{trial.trial_id}_best.weights.h5"
                        )
                        model.save_weights(champion_path)

                model_results = {
                    "trial_id": f"{trial.trial_id}_rep{rep}",
                    "scalogram_type": scalogram_type,
                    "noise_reduction": noise_reduction,
                    "rnn_type": rnn_type,
                    "filters_base": filters_base,
                    "lstm_units": lstm_units,
                    "dropout": dropout_rate,
                    "weight_decay": weight_decay,
                    "best_inner_trial_id": "cnn_ensemble",
                    "inner_best_hps": {},
                }

                target_x = None
                target_y = None
                run_suffix = ""

                if x_test_fit is not None and y_test is not None:
                    target_x = x_test_fit
                    target_y = y_test
                    run_suffix = "test"
                elif x_val_fit is not None and y_val is not None:
                    target_x = x_val_fit
                    target_y = y_val
                    run_suffix = "val"

                rep_run_name_base = f"trial_{trial.trial_id}_{run_suffix}"
                rep_run_name = f"{rep_run_name_base}_rep{rep}"

                if should_evaluate:
                    if target_x is not None and target_y is not None:
                        eval_results = evaluate_model(
                            model=model,
                            x=target_x,
                            y=target_y,
                            folder=trial_folder,
                            run_name=rep_run_name,
                            history=history,
                            training_time=training_time,
                            model_results=model_results,
                            norm_target=False,
                        )
                        if "rmse" in eval_results:
                            rmse_rep = eval_results["rmse"]

                print(f">>>>> Rep {rep+1} Val RMSE: {val_score:.4f} Test RMSE: {rmse_rep:.4f}")

                reps_rmse.append(rmse_rep)

                if model != best_model:
                    del model
                    model = None
                keras.utils.clear_session()
                gc.collect()
                torch.cuda.empty_cache()

            valid_test_scores = [s for s in reps_rmse if not np.isnan(s)]
            valid_scores = [s for s in reps_val_scores if not np.isnan(s)]

            if valid_test_scores and target_x is not None and target_y is not None:
                rmse_score = np.mean(valid_test_scores)
            elif valid_scores:
                rmse_score = np.mean(valid_scores)
                print("WARNING: No valid test scores, falling back to val RMSE.")
            else:
                valid_losses = [l for l in reps_val_losses if not np.isnan(l)]
                if valid_losses:
                    rmse_score = np.mean(valid_losses)
                    print(
                        "WARNING: Using Val Loss as RMSE proxy (metrics missing)."
                    )
                else:
                    rmse_score = 1e9

            mean_val_loss = (
                np.mean([l for l in reps_val_losses if not np.isnan(l)])
                if reps_val_losses
                else 1e9
            )

            valid_train_scores = [s for s in reps_train_scores if not np.isnan(s)]
            mean_train = np.mean(valid_train_scores) if valid_train_scores else float('nan')
            mean_val = np.mean(valid_scores) if valid_scores else float('nan')

            print(f"--- Trial {trial.trial_id} Complete ---")
            print(f"   > Mean Test  RMSE (Score): {rmse_score:.4f}")
            print(f"   > Mean Val   RMSE:         {mean_val:.4f}")
            print(f"   > Mean Train RMSE:         {mean_train:.4f}")

            trial_exceeds_threshold_rep = bool(rmse_score > 0.09)
            final_results = {
                "trial_id": trial.trial_id,
                "exceeds_threshold": trial_exceeds_threshold_rep,
                "metrics": {
                    "val_loss": best_rep_val_loss,
                    "mean_val_loss": mean_val_loss,
                    "rmse": rmse_score,
                    "mean_test_rmse": rmse_score,
                    "mean_val_rmse": mean_val,
                    "mean_train_rmse": mean_train,
                },
                "preprocessing_hps": preprocessing_hps,
                "model_weights_path": f"trial_{trial.trial_id}_best.weights.h5",
                "tuner_type": (
                    "hyperband"
                    if isinstance(self, PreprocessingHyperband)
                    else "bayesian"
                ),
            }

            with open(
                os.path.join(trial_folder, f"DL_trial_{trial.trial_id}.json"),
                "w",
            ) as f:
                json.dump(final_results, f, default=lambda x: str(x), indent=4)

            oracle_rmse_score = rmse_score
            if cv_folds is not None and case_labels is not None:
                print(
                    f">>> Trial {trial.trial_id} mean CV RMSE={rmse_score:.4f}. "
                    f"Fold scores: {[f'{s:.4f}' for s in fold_rmse_scores]}"
                )
            else:
                print(
                    f">>> Trial {trial.trial_id} mean rep RMSE={rmse_score:.4f}. Reporting actual score to oracle."
                )
            
            self.oracle.update_trial(
                trial.trial_id, metrics={"rmse": oracle_rmse_score, "val_loss": oracle_rmse_score}
            )

            return {"rmse": oracle_rmse_score, "val_loss": oracle_rmse_score}

        finally:
            if model is not None:
                del model
            if best_model is not None:
                del best_model
            if history is not None:
                del history

            del x_train_final, x_val_final, x_test_final, y, y_val, y_test

            keras.utils.clear_session()
            gc.collect()
            torch.cuda.empty_cache()


class PreprocessingHyperband(kt.Hyperband, PreprocessingTrial):
    def __init__(self, hypermodel, **kwargs):
        """Initialise PreprocessingHyperband.

        Args:
            hypermodel: Keras Tuner HyperModel instance whose ``build()`` is
                called to construct the Keras model for each trial.
            **kwargs: Forwarded to ``kt.Hyperband.__init__``. The
                ``model_type`` key is consumed here before forwarding.
        """
        self.model_type = kwargs.pop("model_type", "cnn")
        super().__init__(hypermodel, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        """Execute one Hyperband trial.

        Args:
            trial: Keras Tuner ``Trial`` object with sampled HP values.
            *args: Forwarded to ``PreprocessingTrial.run_trial``.
            **kwargs: Forwarded to ``PreprocessingTrial.run_trial``.
        """
        return PreprocessingTrial.run_trial(self, trial, *args, **kwargs)


class PreprocessingRandom(kt.RandomSearch, PreprocessingTrial):
    """Random-search tuner variant.  Simpler than Bayesian — no GP surrogate,
    no warm-up requirement.  Preferred for run-level experiments where the
    full training pipeline is expensive and we want reproducible exploration.
    Use with ``tuner_type='random'`` in ``train_ak.py``.
    """

    def __init__(self, hypermodel, **kwargs):
        """Initialise PreprocessingRandom.

        Args:
            hypermodel: Keras Tuner HyperModel instance.
            **kwargs: Forwarded to ``kt.RandomSearch.__init__``. The
                ``model_type`` key is consumed here before forwarding.
        """
        self.model_type = kwargs.pop("model_type", "cnn")
        super().__init__(hypermodel, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        """Execute one random-search trial.

        Args:
            trial: Keras Tuner ``Trial`` object with sampled HP values.
            *args: Forwarded to ``PreprocessingTrial.run_trial``.
            **kwargs: Forwarded to ``PreprocessingTrial.run_trial``.
        """
        return PreprocessingTrial.run_trial(self, trial, *args, **kwargs)


def get_tuner_class(tuner_type):
    """Factory to return the correct Tuner Class."""
    t = tuner_type.lower()
    if t == "hyperband":
        return PreprocessingHyperband
    elif t == "random":
        return PreprocessingRandom
