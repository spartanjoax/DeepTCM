################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""AutoML joint hyperparameter search for TCM using a custom PreprocessingRandom tuner.

Searches over preprocessing choices (scalogram type, window size, normalisation)
and architecture HPs (filter counts, RNN units, FiLM conditioning) in a single
outer-trial loop. Champion configuration is written to configs/champion_model.json.

Usage:
    python train_ak.py -o 20 -i 50 -e 300
    python train_ak.py --search_mode timefreq_only --use_loco_cv -o 50 -i 100
"""
import argparse
import glob
import os

# Environment Setup
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from keras.losses import MeanSquaredError
import gc

import json
import time
import logging
import datetime
import numpy as np

HYPERBAND_ITERATIONS = 1
HYPERBAND_FACTOR = 3

SPLIT_TYPE = "paper"
NUM_FINAL_REPS = 3

import keras
import keras_tuner as kt
from keras.utils import plot_model

from helpers import (
    evaluate_model,
    NumpyFloatValuesEncoder,
    apply_full_preprocessing,
)
from data import (
    get_nasa_data_pipeline,
)
from models import (
    get_optimizer,
    get_callbacks,
    get_tuner_class,
    get_model,
    resolve_model_name,
)


def setup_logging(timestamp):
    """
    Sets up the logging configuration for the script.

    Args:
        timestamp (str): A timestamp string used to name the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"AK_search_{timestamp}.log")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger


def main(
    epochs=1,
    sliding_window_size=250,
    sliding_window_stride=125,
    num_trials=20,
    run_name="",
    resume_training=False,
    runs_per_case_test_val=1,
    split_offset=5,
    seed=42,
    ma_window_size=10,
    batch_size=16,
    model_type="cnn",
    tuner_type="random",
    search_mode="full",
    use_loco_cv=False,
    downsample_factor=1,
):
    """
    Main function to run the AutoML training pipeline with Support for Split Search & Hyperband.

    Args:
        epochs (int): Number of epochs (Global for Bayesian, Max Epochs for Hyperband).
        sliding_window_size (int): Size of the sliding window.
        sliding_window_stride (int): Stride of the sliding window.
        num_trials (int): Number of outer trials.
        run_name (str): Name of the run.
        resume_training (bool): Whether to resume training.
        runs_per_case_test_val (int): Number of runs per case for test and validation.
        split_offset (int): Offset for splitting the data.
        seed (int): Random seed.
        ma_window_size (int): Size of the moving average window.
        batch_size (int): Batch size.
        model_type (str): Type of model.
        tuner_type (str): 'random' or 'hyperband'.
        search_mode (str): 'split' (Time then Freq), 'time_only', 'timefreq_only', 'full'.
        use_loco_cv (bool): If True, use 11-fold LOCO-CV for champion verification
            instead of the default 3-repetition final evaluation.
        downsample_factor (int): Factor by which to downsample signals before
            processing. Defaults to 1 (no downsampling).
    """

    keras.utils.clear_session()
    gc.collect()
    torch.cuda.empty_cache()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    logger = setup_logging(timestamp)
    logger.info(
        f"Starting Training Pipeline - Tuner: {tuner_type}, Mode: {search_mode}"
    )

    folder = "automl"
    if not os.path.exists(folder):
        os.makedirs(folder)

    logger.info("Loading Data...")
    run_id = run_name[1:]

    allowed_groups = [
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
    if run_id not in allowed_groups:
        run_id = "all"
        logger.info(
            f"Run ID '{run_name[1:]}' not a valid signal group. Defaulting to 'all'."
        )

    if use_loco_cv and tuner_type == "hyperband":
        logger.warning(
            "use_loco_cv=True with tuner_type='hyperband': Hyperband is incompatible "
            "with LOCO-CV (uncontrollable trial count, multi-fidelity noise at low epochs). "
            "Switching to 'random'."
        )
        tuner_type = "random"

    _pipeline_split_type = "paper" if use_loco_cv else SPLIT_TYPE
    _custom_val_cases = [] if use_loco_cv else None

    nasa_train, nasa_val, nasa_test, transforms = get_nasa_data_pipeline(
        run_name=run_id,
        window_size=sliding_window_size,
        stride=sliding_window_stride,
        split_type=_pipeline_split_type,
        seed=seed,
        avg_window_size=ma_window_size,
        apply_averaging=False,
        runs_per_case_test_val=runs_per_case_test_val,
        split_offset=split_offset,
        windowing=False,
        custom_val_cases=_custom_val_cases,
    )

    if use_loco_cv:
        case_labels = transforms["case_labels"]
        cv_folds = [[1, 5, 8, 10], [2, 4, 7, 9], [3, 13, 14]]
        logger.info(f"LOCO-CV enabled. cv_folds={cv_folds}")
    else:
        case_labels = None
        cv_folds = None

    xtr, xtr_proc, ytr = (
        nasa_train.data.cpu().data.numpy(),
        nasa_train.proc_data.cpu().data.numpy(),
        nasa_train.targets.cpu().data.numpy(),
    )
    xval, xval_proc, yval = (
        nasa_val.data.cpu().data.numpy(),
        nasa_val.proc_data.cpu().data.numpy(),
        nasa_val.targets.cpu().data.numpy(),
    )
    xte, xte_proc, yte = (
        nasa_test.data.cpu().data.numpy(),
        nasa_test.proc_data.cpu().data.numpy(),
        nasa_test.targets.cpu().data.numpy(),
    )

    logger.info(f"Train info: X={xtr.shape}, Proc={xtr_proc.shape}, Y={ytr.shape}")

    search_groups = []
    if search_mode == "split":
        search_groups = [
            {"name": "time_domain", "scalograms": ["none"]},
            {"name": "timefreq_domain", "scalograms": ["cwt", "fsst", "wsst", "stft"]},
        ]
    elif search_mode == "time_only":
        search_groups = [{"name": "time_domain", "scalograms": ["none"]}]
    elif search_mode == "timefreq_only":
        search_groups = [
            {"name": "timefreq_domain", "scalograms": ["cwt", "fsst", "wsst", "stft"]}
        ]
    else:
        search_groups = [
            {
                "name": "full",
                "scalograms": ["none", "cwt", "fsst", "wsst", "stft"],
            }
        ]

    global_results = {}
    for group in search_groups:
        group_name = group["name"]
        allowed_scalograms = group["scalograms"]

        logger.info(
            f"=== Starting Search Group: {group_name} (Scalograms: {allowed_scalograms}) ==="
        )

        project_name = f"preprocessing_{model_type}_{group_name}"

        hypermodel = lambda hp: keras.Sequential()

        hp = kt.HyperParameters()
        
        hp.Choice("noise_reduction", ["moving_average", "rms", "none"], default="rms")
        hp.Float("jitter_sigma", 0.0, 0.05, step=0.005, default=0.0)

        if model_type in ["sota_search", "sota_search_max"]:
            hp.Choice("learning_rate", [1e-4, 5e-4, 1e-3], default=1e-3)
            hp.Float("dropout", 0.0, 0.5, step=0.1, default=0.3)
            hp.Choice("arch_type", ["rnn_only", "cnn_rnn"], default="rnn_only")
            hp.Choice("rnn_type", ["lstm", "bigru"], default="lstm")
            hp.Choice("lstm_units", [64, 128, 256], default=64)
            hp.Choice("lstm_layers", [2, 3], default=2)
            hp.Choice("cnn_layers", [2, 3], default=2,
                        parent_name="arch_type", parent_values=["cnn_rnn"])
            hp.Choice("filters_base", [8, 16, 32], default=8,
                        parent_name="arch_type", parent_values=["cnn_rnn"])
          
        hp.Fixed("pooling", value="avg")
        hp.Choice("weight_decay", [1e-5, 1e-4, 1e-3], default=1e-5)
        hp.Choice("conditioning", ["no", "film", "proc"], default="no")
        
        if len(allowed_scalograms) == 1:
            hp.Fixed("scalogram", value=allowed_scalograms[0])
        else:
            hp.Choice("scalogram", allowed_scalograms, default=allowed_scalograms[0])

        logger.info(f"Instantiating {TunerClass.__name__}...")
        TunerClass = get_tuner_class(tuner_type)

        tuner_args = {
            "hypermodel": hypermodel,
            "hyperparameters": hp,
            "objective": kt.Objective("rmse", direction="min"),
            "max_trials": num_trials,
            "directory": f"{folder}/{run_id}",
            "project_name": project_name,
            "overwrite": not resume_training,
            "max_consecutive_failed_trials": 50,
            "seed": seed,
            "model_type": model_type,
        }

        if tuner_type == "hyperband":
            tuner_args["max_epochs"] = epochs
            tuner_args["factor"] = HYPERBAND_FACTOR
            tuner_args["hyperband_iterations"] = HYPERBAND_ITERATIONS
            if "max_trials" in tuner_args:
                del tuner_args["max_trials"]

        tuner = TunerClass(**tuner_args)

        train_input = [xtr, xtr_proc] if xtr_proc is not None else xtr
        val_input   = [xval, xval_proc] if xval_proc is not None else xval
        test_input  = [xte,  xte_proc]  if xte_proc  is not None else xte

        _validation_data = (val_input, yval)
        _cv_kwargs = {"cv_folds": cv_folds, "case_labels": case_labels} if use_loco_cv else {}

        tuner.search(
            x=train_input,
            y=ytr,
            validation_data=_validation_data,
            test_data=(test_input, yte),
            epochs=epochs,
            ma_window_size=ma_window_size,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            batch_size=batch_size,
            downsample_factor=downsample_factor,
            **_cv_kwargs,
        )

        best_hp_outer = tuner.get_best_hyperparameters()[0]
        logger.info(
            f"Group {group_name} Best Preprocessing HPs: {best_hp_outer.values}"
        )

        _n_candidate_trials = 3 if use_loco_cv else 1
        top_trials = tuner.oracle.get_best_trials(
            min(_n_candidate_trials, len(tuner.oracle.trials))
        )
        best_trial = top_trials[0]
        logger.info(
            f"Group {group_name} Best Inner Project for Trial {best_trial.trial_id}"
        )

        if use_loco_cv:
            logger.info("\n" + "=" * 60)
            logger.info(
                f"TOP {len(top_trials)} CANDIDATE TRIALS "
                f"\u2014 GROUP {group_name} \u2014 champion verification"
            )
            logger.info("=" * 60)
            for _rank, _t in enumerate(top_trials):
                logger.info(
                    f"  Rank {_rank + 1}  trial_id={_t.trial_id}  "
                    f"oracle_score={_t.score:.4f}"
                )
                for _hp_name, _hp_val in sorted(_t.hyperparameters.values.items()):
                    logger.info(f"    {_hp_name}: {_hp_val}")
            logger.info("=" * 60 + "\n")

        best_outer_dir = (
            f"{folder}/{run_id}/{project_name}/preproc_trial_{best_trial.trial_id}"
        )
        best_trial_id = best_trial.trial_id

        del tuner
        gc.collect()
        torch.cuda.empty_cache()

        if use_loco_cv:
            loco_cv_folds = [[c] for c in [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]]
            
            def _v_squeeze_signal(arr):
                """Remove degenerate trailing or penultimate dims from a 4-D signal array."""
                if arr.ndim == 4 and arr.shape[-1] == 1:
                    arr = np.squeeze(arr, axis=-1)
                if arr.ndim == 4 and arr.shape[-2] == 1:
                    arr = np.squeeze(arr, axis=-2)
                return arr

            trial_cv_results = []

            for trial_rank, cand_trial in enumerate(top_trials):
                cand_hp = cand_trial.hyperparameters.values
                logger.info(
                    f"\n=== Verification Trial {trial_rank + 1}/{len(top_trials)}: "
                    f"trial_id={cand_trial.trial_id}  "
                    f"oracle_score={cand_trial.score:.4f} ==="
                )

                cand_preprocessing_hps = {
                    "noise_reduction": cand_hp.get("noise_reduction"),
                    "scalogram": cand_hp.get("scalogram"),
                    "ma_window_size": ma_window_size,
                    "downsample_factor": downsample_factor,
                    "jitter_sigma": cand_hp.get("jitter_sigma", 0.0),
                    "rnn_type": cand_hp.get("rnn_type"),
                    "filters_base": cand_hp.get("filters_base"),
                    "lstm_units": cand_hp.get("lstm_units"),
                    "dropout": cand_hp.get("dropout"),
                    "weight_decay": cand_hp.get("weight_decay"),
                    "conditioning": cand_hp.get("conditioning"),
                    "learning_rate": cand_hp.get("learning_rate"),
                    "pooling": cand_hp.get("pooling"),
                }
                cand_sw_config = {
                    "window_size": sliding_window_size,
                    "stride": sliding_window_stride,
                }
                cand_scalogram_type = cand_preprocessing_hps.get("scalogram") or "none"
                _cand_arch_type = cand_hp.get("arch_type")
                _cand_rnn_type = cand_hp.get("rnn_type")
                _cv_target_model = resolve_model_name(model_type, _cand_arch_type, _cand_rnn_type)
                cand_arch_hps = {
                    "filters_base": cand_hp.get("filters_base"),
                    "lstm_units": cand_hp.get("lstm_units"),
                    "lstm_layers": cand_hp.get("lstm_layers", 2),
                    "cnn_layers": cand_hp.get("cnn_layers", 3),
                    "dropout": cand_hp.get("dropout"),
                    "weight_decay": cand_hp.get("weight_decay"),
                    "rnn_type": cand_hp.get("rnn_type"),
                    "pooling": cand_hp.get("pooling"),
                    "learning_rate": cand_hp.get("learning_rate"),
                    "scalogram": cand_scalogram_type,
                }

                fold_rmse_scores = []
                fold_test_scores = []

                _ckpt_dir = os.path.join(
                    folder, run_id,
                    f"preprocessing_{model_type}_{group_name}",
                    "loco_cv_checkpoints",
                )
                os.makedirs(_ckpt_dir, exist_ok=True)
                
                _cand_hp_native = {
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in cand_hp.items()
                }

                for fold_idx, fold_val_cases in enumerate(loco_cv_folds):
                    logger.info(
                        f"  >>> LOCO Fold {fold_idx + 1}/{len(loco_cv_folds)} "
                        f"\u2014 held-out case: {fold_val_cases}"
                    )

                    _ckpt_file = os.path.join(
                        _ckpt_dir,
                        f"trial_{cand_trial.trial_id}_fold{fold_idx}.json",
                    )
                    _ckpt_loaded = False
                    if os.path.exists(_ckpt_file):
                        try:
                            with open(_ckpt_file, "r") as _cf:
                                _ckpt = json.load(_cf)
                            if (
                                _ckpt.get("trial_id") == cand_trial.trial_id
                                and _ckpt.get("hp_values") == _cand_hp_native
                                and "fold_test_score" in _ckpt
                            ):
                                _cached_score = _ckpt.get("fold_val_score", float("nan"))
                                _cached_test = _ckpt.get("fold_test_score", float("nan"))
                                if not np.isnan(_cached_score):
                                    logger.info(
                                        f"  >>> Fold {fold_idx + 1} [CACHED] "
                                        f"Val RMSE: {_cached_score:.4f}  "
                                        f"Test RMSE: {_cached_test:.4f}"
                                    )
                                    fold_rmse_scores.append(_cached_score)
                                    if not np.isnan(_cached_test):
                                        fold_test_scores.append(_cached_test)
                                    _ckpt_loaded = True
                            else:
                                logger.info(
                                    f"  >>> Fold {fold_idx + 1}: checkpoint HP/trial "
                                    f"mismatch — rerunning."
                                )
                        except Exception as _ce:
                            logger.warning(
                                f"  >>> Failed to load checkpoint {_ckpt_file}: {_ce}"
                            )
                    if _ckpt_loaded:
                        continue

                    fold_train_mask = ~np.isin(case_labels, fold_val_cases)
                    fold_val_mask = np.isin(case_labels, fold_val_cases)

                    _raw_in = [xtr, xtr_proc] if xtr_proc is not None else xtr
                    if isinstance(_raw_in, list):
                        x_fold_train_in = [arr[fold_train_mask] for arr in _raw_in]
                        x_fold_val_in = [arr[fold_val_mask] for arr in _raw_in]
                    else:
                        x_fold_train_in = _raw_in[fold_train_mask]
                        x_fold_val_in = _raw_in[fold_val_mask]

                    y_fold_train = ytr[fold_train_mask]
                    y_fold_val = ytr[fold_val_mask]

                    _fp_start = time.time()
                    x_fold_train_pp, y_fold_train_pp, x_fold_proc_pp, fold_preproc_stats = (
                        apply_full_preprocessing(
                            x_fold_train_in,
                            y_fold_train,
                            cand_preprocessing_hps,
                            split="train",
                            preproc_stats=None,
                            sliding_window_config=cand_sw_config,
                        )
                    )
                    x_fold_val_pp, y_fold_val_pp, x_fold_val_proc, _ = apply_full_preprocessing(
                        x_fold_val_in,
                        y_fold_val,
                        cand_preprocessing_hps,
                        split="val",
                        preproc_stats=fold_preproc_stats,
                        sliding_window_config=cand_sw_config,
                    )
                    _fp_time = time.time() - _fp_start

                    if cand_scalogram_type == "none":
                        x_fold_train_pp = _v_squeeze_signal(x_fold_train_pp)
                        x_fold_val_pp = _v_squeeze_signal(x_fold_val_pp)

                    _fsig = x_fold_train_pp
                    _loco_conditioning = cand_hp.get("conditioning", "no")
                    _loco_use_proc = _loco_conditioning != "no" and x_fold_proc_pp is not None
                    _loco_suffix = "_Film" if _loco_conditioning == "film" else ("_Proc" if _loco_conditioning == "proc" else "")
                    _fproc_shape = (1, x_fold_proc_pp.shape[1]) if _loco_use_proc else None
                    _fold_name = f"{_cv_target_model}_{cand_trial.trial_id}_loco{fold_idx}"
                    if _loco_use_proc:
                        fold_model = get_model(
                            model_name=f"{_cv_target_model}{_loco_suffix}",
                            input_shape=(1, *_fsig.shape[1:]),
                            hps=cand_arch_hps,
                            proc_shape=_fproc_shape,
                            name=_fold_name,
                        )
                    else:
                        fold_model = get_model(
                            model_name=_cv_target_model,
                            input_shape=(1, *_fsig.shape[1:]),
                            hps=cand_arch_hps,
                            name=_fold_name,
                        )

                    def _loco_assemble(sig, proc):
                        """Assemble a single or multi-input list from signal and proc arrays.

                        Args:
                            sig (np.ndarray): Preprocessed signal array.
                            proc (np.ndarray | None): Process-parameter array, or None.

                        Returns:
                            np.ndarray or list: ``sig`` alone if ``proc`` is None,
                            otherwise ``[sig, proc]``.
                        """
                        inp = [sig]
                        if proc is not None: inp.append(proc)
                        return inp[0] if len(inp) == 1 else inp
                    x_fold_train_fit = _loco_assemble(x_fold_train_pp, x_fold_proc_pp if _loco_use_proc else None)
                    x_fold_val_fit   = _loco_assemble(x_fold_val_pp,   x_fold_val_proc if _loco_use_proc else None)

                    fold_model.compile(
                        loss=MeanSquaredError(),
                        optimizer=get_optimizer(
                            cand_hp.get("learning_rate", 1e-3),
                            cand_hp.get("optimizer", "adam"),
                        ),
                        metrics=[keras.metrics.RootMeanSquaredError(name="root_mse")],
                    )

                    fold_callbacks = get_callbacks(
                        patience=50,
                        min_delta=0.0001,
                        monitor="val_loss",
                        mode="min",
                        reduce_lr=True,
                    )

                    _ff_start = time.time()
                    fold_history = fold_model.fit(
                        x_fold_train_fit,
                        y_fold_train_pp,
                        validation_data=(x_fold_val_fit, y_fold_val_pp),
                        callbacks=fold_callbacks,
                        epochs=epochs,
                        verbose=1,
                        batch_size=batch_size,
                    )
                    _ff_time = time.time() - _ff_start

                    fold_val_score = np.nan
                    if "val_root_mse" in fold_history.history:
                        fold_val_score = min(fold_history.history["val_root_mse"])
                    elif "val_root_mean_squared_error" in fold_history.history:
                        fold_val_score = min(fold_history.history[
                            "val_root_mean_squared_error"
                        ])
                    elif "val_loss" in fold_history.history:
                        fold_val_score = min(fold_history.history["val_loss"])

                    fold_test_score = np.nan
                    try:
                        _x_test_raw_in = [xte, xte_proc] if xte_proc is not None else xte
                        x_fold_test_pp, y_fold_test_pp, x_fold_test_proc, _ = (
                            apply_full_preprocessing(
                                _x_test_raw_in,
                                yte,
                                cand_preprocessing_hps,
                                split="test",
                                preproc_stats=fold_preproc_stats,
                                sliding_window_config=cand_sw_config,
                            )
                        )
                        if cand_scalogram_type == "none":
                            x_fold_test_pp = _v_squeeze_signal(x_fold_test_pp)
                        x_fold_test_fit = _loco_assemble(
                            x_fold_test_pp,
                            x_fold_test_proc if _loco_use_proc else None,
                        )
                        _test_pred = fold_model.predict(x_fold_test_fit, verbose=0)
                        fold_test_score = float(np.sqrt(np.mean(
                            (y_fold_test_pp.flatten() - _test_pred.flatten()) ** 2
                        )))
                    except Exception as _te:
                        logger.warning(f"  >>> Test eval failed for fold {fold_idx + 1}: {_te}")

                    logger.info(
                        f"  >>> Fold {fold_idx + 1} "
                        f"Val RMSE: {fold_val_score:.4f}  "
                        f"Test RMSE: {fold_test_score:.4f}  "
                        f"preproc={_fp_time:.1f}s  fit={_ff_time:.1f}s"
                    )
                    if not np.isnan(fold_val_score):
                        fold_rmse_scores.append(fold_val_score)
                    if not np.isnan(fold_test_score):
                        fold_test_scores.append(fold_test_score)

                    if not np.isnan(fold_val_score):
                        try:
                            with open(_ckpt_file, "w") as _cf:
                                json.dump(
                                    {
                                        "trial_id": cand_trial.trial_id,
                                        "fold_idx": fold_idx,
                                        "fold_val_cases": fold_val_cases,
                                        "fold_val_score": fold_val_score,
                                        "fold_test_score": float(fold_test_score) if not np.isnan(fold_test_score) else None,
                                        "hp_values": _cand_hp_native,
                                    },
                                    _cf,
                                    indent=4,
                                    cls=NumpyFloatValuesEncoder,
                                )
                        except Exception as _ce:
                            logger.warning(
                                f"  >>> Failed to save checkpoint {_ckpt_file}: {_ce}"
                            )

                    fold_model = None
                    fold_history = None
                    keras.utils.clear_session()
                    gc.collect()
                    torch.cuda.empty_cache()

                mean_loco = (
                    float(np.mean(fold_rmse_scores))
                    if fold_rmse_scores
                    else float("inf")
                )
                std_loco = (
                    float(np.std(fold_rmse_scores))
                    if fold_rmse_scores
                    else float("nan")
                )
                mean_test_loco = (
                    float(np.mean(fold_test_scores))
                    if fold_test_scores
                    else float("inf")
                )
                std_test_loco = (
                    float(np.std(fold_test_scores))
                    if fold_test_scores
                    else float("nan")
                )
                logger.info(
                    f"=== Trial {cand_trial.trial_id}: "
                    f"mean 11-fold Val RMSE = {mean_loco:.4f} \u00b1 {std_loco:.4f} "
                    f"| Test RMSE = {mean_test_loco:.4f} \u00b1 {std_test_loco:.4f} "
                    f"({len(fold_test_scores)}/11 folds) ==="
                )
                trial_cv_results.append(
                    {
                        "trial_id": cand_trial.trial_id,
                        "hp_values": cand_hp,
                        "preprocessing_hps": cand_preprocessing_hps,
                        "mean_rmse": mean_test_loco,
                        "std_rmse": std_test_loco,
                        "fold_rmses": [float(v) for v in fold_test_scores],
                        "mean_val_rmse": mean_loco,
                        "std_val_rmse": std_loco,
                        "fold_val_rmses": [float(v) for v in fold_rmse_scores],
                    }
                )

            best_cv_idx = int(
                np.argmin([r["mean_rmse"] for r in trial_cv_results])
            )
            champion_cv = trial_cv_results[best_cv_idx]

            logger.info("\n" + "=" * 60)
            logger.info(f"CHAMPION VERIFICATION RESULT  (group: {group_name})")
            for r in trial_cv_results:
                tag = (
                    "  <-- CHAMPION"
                    if r["trial_id"] == champion_cv["trial_id"]
                    else ""
                )
                logger.info(
                    f"  {r['trial_id']}  "
                    f"Test: {r['mean_rmse']:.4f} \u00b1 {r['std_rmse']:.4f}  "
                    f"Val: {r['mean_val_rmse']:.4f} \u00b1 {r['std_val_rmse']:.4f}"
                    f"{tag}"
                )
            logger.info("=" * 60)

            _vrf_folder = (
                f"{folder}/{run_id}/preprocessing_{model_type}_{group_name}"
            )
            os.makedirs(_vrf_folder, exist_ok=True)
            _vrf_json = os.path.join(_vrf_folder, "DL_champion_verification.json")
            with open(_vrf_json, "w") as _f:
                json.dump(
                    {
                        "run_id": run_id,
                        "group": group_name,
                        "champion_trial_id": champion_cv["trial_id"],
                        "champion_mean_test_rmse": champion_cv["mean_rmse"],
                        "champion_mean_val_rmse": champion_cv.get("mean_val_rmse"),
                        "champion_hp_values": {
                            k: (
                                float(v)
                                if isinstance(v, (np.floating, float))
                                else v
                            )
                            for k, v in champion_cv["hp_values"].items()
                        },
                        "trial_cv_results": [
                            {
                                "trial_id": r["trial_id"],
                                "mean_test_rmse": r["mean_rmse"],
                                "std_test_rmse": r["std_rmse"],
                                "num_folds": len(r["fold_rmses"]),
                                "fold_test_rmses": r["fold_rmses"],
                                "mean_val_rmse": r.get("mean_val_rmse"),
                                "std_val_rmse": r.get("std_val_rmse"),
                                "fold_val_rmses": r.get("fold_val_rmses", []),
                            }
                            for r in trial_cv_results
                        ],
                    },
                    _f,
                    indent=4,
                    cls=NumpyFloatValuesEncoder,
                )
            logger.info(f"Verification JSON saved: {_vrf_json}")

            selection_score = champion_cv["mean_rmse"]
            best_rmse = float("nan")
            preprocessing_hps = champion_cv["preprocessing_hps"]
            final_base_name = f"{run_id}_{group_name}"

            if not np.isnan(selection_score) and not np.isinf(selection_score):
                global_results[group_name] = {
                    "val_rmse": selection_score,
                    "selection_score": selection_score,
                    "run_name": final_base_name,
                    "folder": f"{folder}/{run_id}/{group_name}",
                    "preprocessing_hps": preprocessing_hps,
                    "trial_cv_results": trial_cv_results,
                }

                _g_config = {
                    k: (v.item() if hasattr(v, "item") else v)
                    for k, v in {
                        "rnn_type": preprocessing_hps.get("rnn_type"),
                        "filters_base": preprocessing_hps.get("filters_base"),
                        "lstm_units": preprocessing_hps.get("lstm_units"),
                        "dropout": preprocessing_hps.get("dropout"),
                        "weight_decay": preprocessing_hps.get("weight_decay"),
                        "noise_reduction": preprocessing_hps.get("noise_reduction"),
                        "scalogram": preprocessing_hps.get("scalogram"),
                        "conditioning": preprocessing_hps.get("conditioning"),
                        "learning_rate": preprocessing_hps.get("learning_rate"),
                        "pooling": preprocessing_hps.get("pooling"),
                    }.items()
                }
                os.makedirs("configs", exist_ok=True)
                _g_config_path = os.path.join("configs", f"champion_{group_name}.json")
                with open(_g_config_path, "w") as _gf:
                    json.dump(_g_config, _gf, indent=4)
                logger.info(f"Group champion config saved: {_g_config_path}")

            logger.info(f"Group {group_name} Completed Successfully.")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        best_inner_trial_id = None
        inner_best_hps = None

        json_path = os.path.join(best_outer_dir, f"DL_trial_{best_trial_id}_test.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(
                best_outer_dir, f"DL_trial_{best_trial_id}_val.json"
            )

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    res_data = json.load(f)
                if "inner_best_hps" in res_data and "best_inner_trial_id" in res_data:
                    inner_best_hps = res_data["inner_best_hps"]
                    best_inner_trial_id = res_data["best_inner_trial_id"]
                    logger.info(f"Found inner HPs in JSON: Trial {best_inner_trial_id}")
            except Exception as e:
                logger.warning(f"Failed to read JSON {json_path}: {e}")

        logger.info(f"Selected Best Inner Trial ID: {best_inner_trial_id}")

        model_to_load = None
        
        weights_path = os.path.join(
            best_outer_dir, f"trial_{best_trial_id}_best.weights.h5"
        )
        if os.path.exists(weights_path):
            model_to_load = weights_path
            logger.info(f"Found weights at {weights_path}")
        else:
            logger.error(f"Weights not found at {weights_path}")

        if not model_to_load:
            logger.error("Could not find a model file to clone. Skipping.")
            continue

        preprocessing_hps = {
            "noise_reduction": best_hp_outer.values.get("noise_reduction"),
            "scalogram": best_hp_outer.values.get("scalogram"),
            "ma_window_size": ma_window_size,
            "downsample_factor": downsample_factor,
            "jitter_sigma": best_hp_outer.values.get("jitter_sigma", 0.0),
            "rnn_type": best_hp_outer.values.get("rnn_type"),
            "filters_base": best_hp_outer.values.get("filters_base"),
            "lstm_units": best_hp_outer.values.get("lstm_units"),
            "dropout": best_hp_outer.values.get("dropout"),
            "weight_decay": best_hp_outer.values.get("weight_decay"),
            "conditioning": best_hp_outer.values.get("conditioning"),
            "pooling": best_hp_outer.values.get("pooling"),
            "learning_rate": best_hp_outer.values.get("learning_rate"),
        }

        sw_size = sliding_window_size
        sw_stride = sliding_window_stride
        sw_config = {"window_size": sw_size, "stride": sw_stride}

        logger.info(
            f"Applying Unified Pipeline to Final Train Data for Group {group_name}..."
        )

        train_input = [xtr, xtr_proc] if xtr_proc is not None else xtr

        x_train_final_fit, y_train_pre, xtr_proc_fit, preproc_stats = apply_full_preprocessing(
            train_input,
            ytr,
            preprocessing_hps,
            split="train",
            preproc_stats=None,
            sliding_window_config=sw_config,
        )

        def _squeeze_signal(arr):
            """Remove degenerate trailing or penultimate dims from a 4-D signal array."""
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

        if preprocessing_hps["scalogram"] == "none":
            x_train_final_fit = _squeeze_split(x_train_final_fit)

        logger.info(f"Loading model template/weights from: {model_to_load}")

        logger.info("Reconstructing model structure for weight loading...")

        _best_arch_type = best_hp_outer.values.get("arch_type", "cnn_rnn")
        _best_rnn_type = best_hp_outer.values.get("rnn_type", "lstm")
        target_model_name = resolve_model_name(model_type, _best_arch_type, _best_rnn_type)

        scalogram_type = best_hp_outer.values.get("scalogram")
        reconst_hps = {
            "filters_base": best_hp_outer.values.get("filters_base"),
            "lstm_units": best_hp_outer.values.get("lstm_units"),
            "dropout": best_hp_outer.values.get("dropout"),
            "weight_decay": best_hp_outer.values.get("weight_decay"),
            "rnn_type": best_hp_outer.values.get("rnn_type"),
            "pooling": best_hp_outer.values.get("pooling", "avg"),
            "learning_rate": best_hp_outer.values.get("learning_rate", 1e-3),
            "scalogram": (scalogram_type if scalogram_type != "none" else "none"),
        }

        input_signal_shape = (
            x_train_final_fit[0].shape
            if isinstance(x_train_final_fit, list)
            else x_train_final_fit.shape
        )
        dummy_shape = (1, *input_signal_shape[1:])

        _final_conditioning = preprocessing_hps.get("conditioning")
        _use_proc_final = _final_conditioning != "no" and xtr_proc_fit is not None
        _final_suffix = "_Film" if _final_conditioning == "film" else ("_Proc" if _final_conditioning == "proc" else "")
        if _use_proc_final:
            _proc_shape = (1, xtr_proc_fit.shape[1])
            loaded_model = get_model(
                model_name=f"{target_model_name}{_final_suffix}",
                input_shape=dummy_shape,
                hps=reconst_hps,
                proc_shape=_proc_shape,
                name=f"Final_{target_model_name}",
            )
        else:
            loaded_model = get_model(
                model_name=target_model_name,
                input_shape=dummy_shape,
                hps=reconst_hps,
                name=f"Final_{target_model_name}",
            )

        loaded_model.load_weights(model_to_load)
        logger.info("Weights loaded successfully.")

        if inner_best_hps is None:
            inner_best_hps = {}

        opt_name = inner_best_hps.get("optimizer", "adam")
        lr = best_hp_outer.values.get("learning_rate", inner_best_hps.get("learning_rate", 1e-3))

        _has_val = xval is not None and len(xval) > 0
        x_val_final_fit = None
        y_val_final_fit = None
        xval_proc_fit = None
        if _has_val:
            val_input = [xval, xval_proc] if xval_proc is not None else xval
            x_val_final_fit, y_val_final_fit, xval_proc_fit, _ = apply_full_preprocessing(
                val_input,
                yval,
                preprocessing_hps,
                split="val",
                preproc_stats=preproc_stats,
                sliding_window_config=sw_config,
            )
            if preprocessing_hps["scalogram"] == "none":
                x_val_final_fit = _squeeze_split(x_val_final_fit)

        xte_proc_fit = None
        if xte is not None:
            test_input = [xte, xte_proc] if xte_proc is not None else xte
            x_test_final, y_test_final, xte_proc_fit, _ = apply_full_preprocessing(
                test_input,
                yte,
                preprocessing_hps,
                split="test",
                preproc_stats=preproc_stats,
                sliding_window_config=sw_config,
            )
            if preprocessing_hps["scalogram"] == "none":
                x_test_final = _squeeze_split(x_test_final)
        else:
            x_test_final = None

        gc.collect()
        torch.cuda.empty_cache()

        if isinstance(x_train_final_fit, list):
            x_train_sig_check = x_train_final_fit[0]
        else:
            x_train_sig_check = x_train_final_fit

        scalogram_type = preprocessing_hps.get("scalogram", "none")

        if scalogram_type == "none" and x_train_sig_check.ndim == 4:
            logger.info("Squeezing extra dimension for Standard CNN (dual-pass)...")
            x_train_final_fit = _squeeze_split(x_train_final_fit)
            if x_val_final_fit is not None:
                x_val_final_fit = _squeeze_split(x_val_final_fit)
            if x_test_final is not None:
                x_test_final = _squeeze_split(x_test_final)

        def _final_assemble(sig, proc):
            """Assemble single or multi-input list from signal and proc arrays.

            None entries are excluded, returning a list or bare array.

            Args:
                sig (np.ndarray): Preprocessed signal array.
                proc (np.ndarray | None): Process-parameter array, or None.

            Returns:
                np.ndarray or list: ``sig`` alone if ``proc`` is None,
                otherwise ``[sig, proc]``.
            """
            inp = [sig]
            if proc is not None: inp.append(proc)
            return inp[0] if len(inp) == 1 else inp

        _use_proc_data = preprocessing_hps.get("conditioning") != "no"
        x_train_final_fit = _final_assemble(x_train_final_fit, xtr_proc_fit if _use_proc_data else None)
        if _has_val and x_val_final_fit is not None:
            x_val_final_fit = _final_assemble(x_val_final_fit, xval_proc_fit if _use_proc_data else None)
        if x_test_final is not None:
            x_test_final = _final_assemble(x_test_final, xte_proc_fit if _use_proc_data else None)

        reps_rmse = []
        reps_info = []

        logger.info(
            f"Starting {NUM_FINAL_REPS} repetitions for Final Model Evaluation (Group: {group_name})..."
        )

        for rep in range(NUM_FINAL_REPS):
            logger.info(f">>> Final Repetition {rep+1}/{NUM_FINAL_REPS}...")

            rep_run_name = f"Final_{run_id}_{group_name}_rep{rep}"
            eval_folder = f"{folder}/{run_id}/preprocessing_{model_type}_{group_name}"
            json_path = f"{eval_folder}/DL_{rep_run_name}.json"

            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        res_entry = json.load(f)
                        rmse_val = res_entry.get("rmse", float("inf"))

                        val_rmse_rep = float("nan")
                        if "history" in res_entry:
                            h = res_entry["history"]
                            if "val_root_mse" in h:
                                val_rmse_rep = min(h["val_root_mse"])
                            elif "val_root_mean_squared_error" in h:
                                val_rmse_rep = min(h["val_root_mean_squared_error"])
                            elif "val_loss" in h:
                                val_rmse_rep = min(h["val_loss"])

                        reps_rmse.append(rmse_val)
                        reps_info.append(
                            {
                                "rep": rep,
                                "rmse": rmse_val,
                                "run_name": rep_run_name,
                                "folder": eval_folder,
                                "valid": True,
                                "val_rmse": val_rmse_rep,
                            }
                        )
                    logger.info(f"Skipping Rep {rep+1} (Resume): Found {json_path}")
                    continue
                except Exception as e:
                    logger.warning(
                        f"Failed to load existing result JSON {json_path}: {e}"
                    )

            final_model_rep = keras.models.clone_model(loaded_model)

            final_model_rep.compile(
                loss=MeanSquaredError(),
                optimizer=get_optimizer(lr, opt_name),
                metrics=[keras.metrics.RootMeanSquaredError(name="root_mse")],
            )

            callbacks_list = get_callbacks(
                patience=50,
                min_delta=0.0001,
                monitor="val_loss",
                mode="min",
                reduce_lr=True,
            )

            start_time_rep = time.time()
            history_rep = final_model_rep.fit(
                x_train_final_fit,
                y_train_pre,
                validation_data=(x_val_final_fit, y_val_final_fit) if _has_val else None,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks_list,
                batch_size=batch_size,
            )
            training_time_rep = time.time() - start_time_rep

            if _has_val and "val_loss" in history_rep.history:
                final_val_loss = min(history_rep.history["val_loss"])
            else:
                final_val_loss = min(history_rep.history["loss"])

            val_rmse_rep = np.nan
            if "val_root_mse" in history_rep.history:
                val_rmse_rep = min(history_rep.history["val_root_mse"])
            elif "val_root_mean_squared_error" in history_rep.history:
                val_rmse_rep = min(history_rep.history["val_root_mean_squared_error"])
            elif "root_mse" in history_rep.history:
                val_rmse_rep = min(history_rep.history["root_mse"])
            elif "root_mean_squared_error" in history_rep.history:
                val_rmse_rep = min(history_rep.history["root_mean_squared_error"])
            elif not np.isnan(final_val_loss):
                val_rmse_rep = final_val_loss

            rmse_rep = np.nan
            should_evaluate = True

            if np.isnan(final_val_loss):
                logger.warning(
                    f"Rep {rep+1} failed with NaN loss. Skipping evaluation."
                )
                rmse_rep = np.nan
                val_rmse_rep = float("inf")
                should_evaluate = False
            elif np.isinf(final_val_loss):
                logger.warning(
                    f"Rep {rep+1} failed with Inf loss. Setting to large value."
                )
                rmse_rep = 1e9
                val_rmse_rep = 1e9
                should_evaluate = False

            rep_run_name = f"Final_{run_id}_{group_name}_rep{rep}"

            eval_folder = f"{folder}/{run_id}/preprocessing_{model_type}_{group_name}"
            if not os.path.exists(eval_folder):
                os.makedirs(eval_folder)

            model_results = {
                "run_id": run_id,
                "group": group_name,
                "rep": rep,
                "preprocessing_hps": preprocessing_hps,
            }

            if should_evaluate:
                eval_results = evaluate_model(
                    model=final_model_rep,
                    x=x_test_final,
                    y=y_test_final,
                    folder=eval_folder,
                    run_name=rep_run_name,
                    history=history_rep,
                    training_time=training_time_rep,
                    model_results=model_results,
                    norm_target=False,
                )

                if "rmse" in eval_results:
                    rmse_rep = eval_results["rmse"]

            logger.info(f">>> Rep {rep+1} RMSE: {rmse_rep:.4f}")

            reps_rmse.append(rmse_rep)
            reps_info.append(
                {
                    "rep": rep,
                    "rmse": rmse_rep,
                    "run_name": rep_run_name,
                    "folder": eval_folder,
                    "valid": should_evaluate,
                    "val_rmse": val_rmse_rep,
                }
            )

            final_model_rep = None
            history_rep = None
            keras.utils.clear_session()
            gc.collect()

        valid_val_rmses = [
            info["val_rmse"] for info in reps_info if not np.isnan(info["val_rmse"])
        ]

        if len(valid_val_rmses) > 0:
            safe_val_rmses = [
                info["val_rmse"] if not np.isnan(info["val_rmse"]) else float("inf")
                for info in reps_info
            ]
            best_rep_idx = np.argmin(safe_val_rmses)

            best_val_rmse = reps_info[best_rep_idx]["val_rmse"]

            best_test_rmse = reps_info[best_rep_idx]["rmse"]

            valid_test_rmses = [r for r in reps_rmse if not np.isnan(r)]
            avg_test_rmse = np.mean(valid_test_rmses) if valid_test_rmses else np.nan

            best_rmse = best_test_rmse
            selection_score = best_val_rmse

        else:
            best_rmse = np.nan
            selection_score = float("inf")
            best_rep_idx = 0

        logger.info(
            f"Group {group_name} Final Results - Best Val RMSE: {selection_score:.4f} -> Test RMSE: {best_rmse:.4f}"
        )

        final_base_name = f"{run_id}_{group_name}"

        for info in reps_info:
            rep = info["rep"]
            rep_name = info["run_name"]
            folder_path = info["folder"]

            if rep == best_rep_idx:
                json_src = os.path.join(folder_path, f"DL_{rep_name}.json")
                if os.path.exists(json_src):
                    with open(json_src, "r") as f:
                        final_meta = json.load(f)

                    final_meta["selection_val_rmse"] = selection_score

                    with open(json_src, "w") as f:
                        json.dump(final_meta, f, indent=4, cls=NumpyFloatValuesEncoder)

                keras_pattern = os.path.join(folder_path, f"DL_{rep_name}*.keras")
                found_keras = glob.glob(keras_pattern)
                if found_keras:
                    src_keras = found_keras[0]
                    dst_keras = os.path.join(folder_path, f"DL_{final_base_name}.keras")
                    os.rename(src_keras, dst_keras)
                else:
                    logger.warning(
                        f"Could not find model file for renaming: {keras_pattern}"
                    )

                other_extensions = ["_training_loss.png", "_test_pred.png", ".json"]
                for ext in other_extensions:
                    src = os.path.join(folder_path, f"DL_{rep_name}{ext}")
                    dst = os.path.join(folder_path, f"DL_{final_base_name}{ext}")
                    if os.path.exists(src):
                        os.rename(src, dst)
            else:
                keras_pattern = os.path.join(folder_path, f"DL_{rep_name}*.keras")
                for f in glob.glob(keras_pattern):
                    os.remove(f)

                other_extensions = ["_training_loss.png", "_test_pred.png", ".json"]
                for ext in other_extensions:
                    src = os.path.join(folder_path, f"DL_{rep_name}{ext}")
                    if os.path.exists(src):
                        os.remove(src)

        try:
            plot_model(
                loaded_model,
                to_file=f"{folder}/{run_id}/{group_name}/DL_{final_base_name}_modelplot.png",
                show_shapes=True,
                show_layer_names=True,
            )
        except Exception as e:
            logger.warning(f"Could not plot model: {e}")

        del loaded_model
        del x_train_final_fit, y_train_pre

        gc.collect()
        torch.cuda.empty_cache()

        if not np.isnan(selection_score) and not np.isinf(selection_score):
            global_results[group_name] = {
                "selection_score": selection_score,
                "run_name": final_base_name,
                "folder": f"{folder}/{run_id}/{group_name}",
                "preprocessing_hps": preprocessing_hps,
            }

            _g_pooling = preprocessing_hps.get("pooling", "avg")
            _g_config = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in {
                    "rnn_type": preprocessing_hps.get("rnn_type"),
                    "filters_base": preprocessing_hps.get("filters_base"),
                    "lstm_units": preprocessing_hps.get("lstm_units"),
                    "dropout": preprocessing_hps.get("dropout"),
                    "weight_decay": preprocessing_hps.get("weight_decay"),
                    "noise_reduction": preprocessing_hps.get("noise_reduction"),
                    "scalogram": preprocessing_hps.get("scalogram"),
                    "learning_rate": preprocessing_hps.get("learning_rate", 1e-3),
                    "pooling": _g_pooling,
                }.items()
            }
            os.makedirs("configs", exist_ok=True)
            _g_config_path = os.path.join("configs", f"champion_{group_name}.json")
            with open(_g_config_path, "w") as _gf:
                json.dump(_g_config, _gf, indent=4)
            logger.info(f"Group champion config saved: {_g_config_path}")

        logger.info(f"Group {group_name} Completed Successfully.")

    logger.info("All Search Groups Completed.")

    if len(global_results) > 0:
        logger.info("\n" + "=" * 40)
        logger.info(" GLOBAL CHAMPION SELECTION")
        logger.info("=" * 40)

        best_group = min(
            global_results, key=lambda k: global_results[k]["selection_score"]
        )
        champion_data = global_results[best_group]

        logger.info(f"Overall Champion: {model_type} ({best_group})")
        _score_label = (
            "Mean 11-fold LOCO RMSE" if use_loco_cv else "Best Val RMSE (Selection)"
        )
        logger.info(
            f"   > {_score_label}: {champion_data['selection_score']:.4f}"
        )
        logger.info(f"   > Location: {champion_data['folder']}")

        def _serialise_group(v):
            """Build the per-group dict that goes into all_results."""
            entry = {
                k2: v2
                for k2, v2 in v.items()
                if k2 not in ("preprocessing_hps", "trial_cv_results")
            }
            if use_loco_cv and "trial_cv_results" in v:
                entry["candidate_trials"] = [
                    {
                        "trial_id": t["trial_id"],
                        "mean_11fold_rmse": t["mean_rmse"],
                        "std_rmse": t["std_rmse"],
                        "fold_rmses": t["fold_rmses"],
                        "hp_values": {
                            hk: (
                                float(hv)
                                if isinstance(hv, (np.floating, float))
                                else hv
                            )
                            for hk, hv in t["hp_values"].items()
                        },
                    }
                    for t in v["trial_cv_results"]
                ]
            return entry

        global_meta = {
            "run_id": run_id,
            "model_type": model_type,
            "use_loco_cv": use_loco_cv,
            "champion_group": best_group,
            "champion_selection_score": champion_data["selection_score"],
            "all_results": {
                k: _serialise_group(v)
                for k, v in global_results.items()
            },
        }

        with open(f"{folder}/{run_id}/global_champion.json", "w") as f:
            json.dump(global_meta, f, indent=4, cls=NumpyFloatValuesEncoder)
        logger.info(f"Global champion JSON saved: {folder}/{run_id}/global_champion.json")

        chp_hps = champion_data.get("preprocessing_hps", {})
        _pooling = chp_hps.get("pooling", "avg")
        champion_config = {
            "rnn_type": chp_hps.get("rnn_type"),
            "filters_base": chp_hps.get("filters_base"),
            "lstm_units": chp_hps.get("lstm_units"),
            "dropout": chp_hps.get("dropout"),
            "weight_decay": chp_hps.get("weight_decay"),
            "noise_reduction": chp_hps.get("noise_reduction"),
            "scalogram": chp_hps.get("scalogram"),
            "learning_rate": chp_hps.get("learning_rate", 1e-3),
            "pooling": _pooling,
        }
        
        champion_config = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in champion_config.items()
        }
        os.makedirs("configs", exist_ok=True)
        champion_config_path = os.path.join("configs", "champion_model.json")
        with open(champion_config_path, "w") as f:
            json.dump(champion_config, f, indent=4)
        logger.info(f"Champion model config saved: {champion_config_path}")
        logger.info(f"  > Champion HPs: {champion_config}")

    else:
        logger.warning("No valid results found across any search group.")

    if "hypermodel" in locals():
        del hypermodel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Training with Split Search")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=300,
        help="Epochs (Global for Bayesian, Max for Hyperband)",
    )
    parser.add_argument(
        "-r", "--resume", type=int, default=1, help="Resume Training (1=True, 0=False)"
    )
    parser.add_argument(
        "-w", "--window_size", type=int, default=10, help="MA Window Size (default 10)"
    )
    parser.add_argument(
        "-o", "--num_trials", type=int, default=50, help="Number of Outer Trials"
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="sota_search",
        help="Model Type (cnn, lstm, sota_search...)",
    )
    parser.add_argument(
        "--tuner_type",
        type=str,
        default="random",
        choices=["hyperband", "random"],
        help="Tuner Type ('hyperband' or 'random')",
    )
    parser.add_argument(
        "--search_mode",
        type=str,
        default="split",
        choices=["split", "time_only", "timefreq_only", "full"],
        help="Search Mode",
    )
    parser.add_argument(
        "--use_loco_cv",
        action="store_true",
        help=(
            "Use 3-fold stratified case CV for HP search, then "
            "11-fold LOCO-CV on top 3 trials for champion verification. "
            "Forces split_type='paper' and custom_val_cases=[]."
        ),
    )
    parser.add_argument(
        "-sw", "--sliding_window_size", type=int, default=250,
        help="Sliding window size (default 250)",
    )
    parser.add_argument(
        "-ss", "--sliding_window_stride", type=int, default=125,
        help="Sliding window stride (default 125)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Decimation factor applied before windowing (e.g. 2 = take every 2nd sample)",
    )

    args = parser.parse_args()

    epochs = args.epochs
    resume = bool(args.resume)
    ma_window_size = args.window_size
    num_trials = args.num_trials
    model_type = args.model_type
    tuner_type = args.tuner_type
    search_mode = args.search_mode

    for run_name in ["_all"]:
        print(f"================== Training {run_name}")
        main(
            epochs=epochs,
            run_name=run_name,
            resume_training=resume,
            ma_window_size=ma_window_size,
            num_trials=num_trials,
            model_type=model_type,
            tuner_type=tuner_type,
            search_mode=search_mode,
            use_loco_cv=args.use_loco_cv,
            sliding_window_size=args.sliding_window_size,
            sliding_window_stride=args.sliding_window_stride,
            downsample_factor=args.downsample,
        )
