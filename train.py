################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Train expert-defined DL models for tool condition monitoring on the NASA Ames Milling Dataset.

Architectures include CNN, ResNet, BiGRU, LSTM, meta-learning ensembles, CNN+RNN,  
and their process-conditioned multimodal variants.
Results (RMSE and R²) and model checkpoints are written per-fold.

Usage:
    python train.py --signal_group all
    python train.py --config configs/champion_timefreq_domain.json
"""

import glob
import gc
import torch
import argparse

import os

# Available backend options are: "jax", "tensorflow", "torch"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from keras.utils import clear_session
from keras import saving
from keras import optimizers
from keras.metrics import RootMeanSquaredError

from models import (
    get_optimizer,
    get_model,
    get_callbacks,
)

# Clear all previously registered custom objects
saving.get_custom_objects().clear()

from helpers import (
    evaluate_model,
    NumpyFloatValuesEncoder,
    apply_full_preprocessing,
)
from data import (
    get_nasa_data_pipeline,
)
import os
import json
import time
import logging
import datetime
from collections import defaultdict

sns.set_context("paper")
sns.set(font_scale=1)
bgcolor = ""
color = ["hls", "Plasma"]

NUM_FINAL_REPS = 3
es_patience = 50
es_min_delta = 0.0001
split_type = "paper"

PAPER_TRAIN_POOL = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]

def get_expected_models(test_mode=False):
    """
    Returns a list of expected model names based on configuration.

    Args:
        test_mode (bool): If True, returns a smaller subset of models.

    Returns:
        list: List of model names.
    """
    models = []
    if not test_mode:
        models.extend(
            [
                "CNN_LSTM",
                "CNN_LSTM_Proc",
                "RobustCNN_LSTM",
                "RobustCNN_LSTM_Proc",
            ]
        )
    models.extend(["CNN", "LSTM", "BiGRU", "Ensemble", "Ensemble_Proc"])

    if not test_mode:
        models.extend(
            [
                "ResNet",
                "RobustResNet",
                "ResNet_Proc",
                "RobustResNet_Proc",
            ]
        )

    return models


def get_best_model_path(folder, model_name, run):
    """
    Finds the model file with the lowest RMSE score per fold from the filename.
    Pattern: DL_{model_name}_{run}_rep{i}_{score}.keras

    Args:
        folder (str): Directory to search for ``.keras`` files.
        model_name (str): Architecture name (e.g. ``'CNN_LSTM'``).
        run (str): Signal group / run identifier embedded in the filename.

    Returns:
        str | None: Path to the ``.keras`` file with the lowest RMSE, or
        None if no matching files are found.
    """
    file_pattern = f"{folder}/DL_{model_name}_{run}_rep*.keras"
    files = glob.glob(file_pattern)

    if not files:
        return None

    best_rmse = float("inf")
    best_file = None

    for f in files:
        try:
            basename = os.path.basename(f)
            name_no_ext = os.path.splitext(basename)[0]
            rmse_str = name_no_ext.split("_")[-1]
            rmse = float(rmse_str)

            if rmse < best_rmse:
                best_rmse = rmse
                best_file = f
        except Exception:
            continue

    return best_file

def main(
    epochs=300,
    sliding_window_size=250,
    sliding_window_stride=125,
    run="all",
    resume_training=True,
    seed=42,
    test_mode=False,
    hp_file=None,
    scheduler="plateau",
    batch_size=16,
    hps_override=None,
    experiment=None,
):
    """
    Main training function.

    Args:
        epochs (int): Epoch ceiling (ES may fire earlier in eval/calibrate modes).
        sliding_window_size (int): Segment size.
        sliding_window_stride (int): Stride.
        run (str): Signal-group name passed to the data pipeline (e.g. 'all').
        resume_training (bool): Skip already-completed reps when True.
        seed (int): Random seed.
        test_mode (bool): If True, uses smaller model subset.
        hp_file (str): Path to hyperparameters JSON file.
        scheduler (str): 'plateau' or 'cosine'.
        batch_size (int): Batch size.
        hps_override (dict | None): Key-value overrides applied on top of hp_file.
        experiment (str | None): Prefix for the output folder name.
    """
    folder_name = "final_optimized" if hp_file else "final"
    folder_prefix = experiment if experiment is not None else folder_name
    folder = f"{folder_prefix}_{run}_sw{sliding_window_size}_ss{sliding_window_stride}"
    Path(folder).mkdir(parents=True, exist_ok=True)

    best_hps = {
        "rnn_type": "lstm",
        "filters_base": 32,
        "lstm_units": 64,
        "dropout": 0.2,
        "weight_decay": 1e-5,
        "noise_reduction": "moving_average",
        "scalogram": "none",
    }

    if hp_file:
        print(f"Loading Hyperparameters from {hp_file}")
        try:
            with open(hp_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    data = data[0] if len(data) > 0 else {}

                best_hps.update(data)

                if "values" in data:
                    best_hps.update(data["values"])

        except Exception as e:
            print(f"Error loading HP file: {e}. Using defaults.")

    if hps_override:
        print("--- Applying HP Overrides ---")
        best_hps.update(hps_override)

    print("--- Applied Configuration ---")
    print(json.dumps(best_hps, indent=4))

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{folder}/manual_{timestamp}.log")
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    script_time = time.time()
    logger.info(f"---Starting script\n--- {(time.time() - script_time):.4f} seconds ---")

    sw_config_preproc = {
        "window_size": sliding_window_size,
        "stride": sliding_window_stride,
    }

    def _prepare_data(train_cases=None, val_cases=None, *,
                       preloaded_tr=None, preloaded_val=None, preloaded_te=None):
        """Build windowed train/val/test arrays.

        Either pass ``train_cases``/``val_cases`` (calls get_nasa_data_pipeline)
        or pass pre-loaded raw arrays via ``preloaded_tr``, ``preloaded_val``,
        ``preloaded_te`` — each is a tuple (signal, proc, feat, targets).
        The pre-loaded path is used by LOCO calibrate so the pipeline is only
        called once for the whole pool (mirroring train_ak.py).

        Args:
            train_cases (list | None): Case IDs for the training split.
                None uses the default paper split.
            val_cases (list | None): Case IDs for the validation split.
            preloaded_tr (tuple | None): Pre-loaded ``(signal, proc, feat, targets)``
                for training. Bypasses the pipeline call when provided.
            preloaded_val (tuple | None): Same for validation.
            preloaded_te (tuple | None): Same for test.

        Returns:
            dict: Windowed arrays with keys ``xtr_sw``, ``xval_sw``,
            ``xte_sw``, ``xtr_sw_1d``, proc arrays, feat arrays, and targets.
        """
        if preloaded_tr is not None:
            xtr_orig,  xtr_proc_orig,  xtr_feat_orig,  ytr_orig  = preloaded_tr
            xval_orig, xval_proc_orig, xval_feat_orig, yval_orig = preloaded_val
            xte_orig,  xte_proc_orig,  xte_feat_orig,  yte_orig  = preloaded_te
            logger.info(
                f"Train/Val/Test (pre-loaded). Sizes: {xtr_orig.shape}, "
                f"{xval_orig.shape}, {xte_orig.shape}"
            )
        else:
            nasa_train, nasa_val, nasa_test, transforms = get_nasa_data_pipeline(
                run_name=run,
                window_size=sliding_window_size,
                stride=sliding_window_stride,
                split_type="paper",
                seed=seed,
                apply_averaging=False,
                windowing=False,
                custom_train_cases=train_cases,
                custom_val_cases=val_cases,
            )

            xtr_orig = nasa_train.data.cpu().numpy()
            xval_orig = nasa_val.data.cpu().numpy()
            xte_orig  = nasa_test.data.cpu().numpy()
            ytr_orig  = nasa_train.targets.cpu().numpy()
            yval_orig = nasa_val.targets.cpu().numpy()
            yte_orig  = nasa_test.targets.cpu().numpy()
            xtr_proc_orig  = nasa_train.proc_data.cpu().numpy()
            xval_proc_orig = nasa_val.proc_data.cpu().numpy()
            xte_proc_orig  = nasa_test.proc_data.cpu().numpy()
            xtr_feat_orig  = nasa_train.feat_data.cpu().numpy()
            xval_feat_orig = nasa_val.feat_data.cpu().numpy()
            xte_feat_orig  = nasa_test.feat_data.cpu().numpy()

            logger.info(
                f"Train/Val/Test loaded. Sizes: {nasa_train.data.shape}, "
                f"{nasa_val.data.shape}, {nasa_test.data.shape}"
            )

        best_hps_1d = best_hps.copy()
        best_hps_1d["scalogram"] = "none"

        x_train_input = [xtr_orig, np.hstack([xtr_proc_orig, xtr_feat_orig])]
        x_val_input   = [xval_orig, np.hstack([xval_proc_orig, xval_feat_orig])]
        x_test_input  = [xte_orig,  np.hstack([xte_proc_orig,  xte_feat_orig])]

        logger.info("Generating STRICT 1D Dataset (Scalogram=None)...")
        xtr_sw_1d, ytr_sw, xtr_proc_sw, stats = apply_full_preprocessing(
            x_train_input, ytr_orig, best_hps_1d, split="train",
            sliding_window_config=sw_config_preproc,
        )
        if len(yval_orig) > 0:
            xval_sw_1d, yval_sw, xval_proc_sw, _ = apply_full_preprocessing(
                x_val_input, yval_orig, best_hps_1d, split="val",
                preproc_stats=stats, sliding_window_config=sw_config_preproc,
            )
        else:
            xval_sw_1d = np.empty((0,) + xtr_sw_1d.shape[1:], dtype=np.float32)
            yval_sw    = np.empty((0,), dtype=np.float32)
            xval_proc_sw = (np.empty((0, xtr_proc_sw.shape[1]), dtype=np.float32)
                            if xtr_proc_sw is not None else None)
        xte_sw_1d, yte_sw, xte_proc_sw, _ = apply_full_preprocessing(
            x_test_input, yte_orig, best_hps_1d, split="test",
            preproc_stats=stats, sliding_window_config=sw_config_preproc,
        )

        if best_hps.get("scalogram", "none") != "none":
            logger.info(f"Generating MAIN Dataset 2D ({best_hps['scalogram']})...")
            xtr_sw, _, _, stats_2d = apply_full_preprocessing(
                x_train_input, ytr_orig, best_hps, split="train",
                sliding_window_config=sw_config_preproc,
            )
            if len(yval_orig) > 0:
                xval_sw, _, _, _ = apply_full_preprocessing(
                    x_val_input, yval_orig, best_hps, split="val",
                    preproc_stats=stats_2d, sliding_window_config=sw_config_preproc,                   
                )
            else:
                xval_sw = np.empty((0,) + xtr_sw.shape[1:], dtype=np.float32)
            xte_sw, _, _, _ = apply_full_preprocessing(
                x_test_input, yte_orig, best_hps, split="test",
                preproc_stats=stats_2d, sliding_window_config=sw_config_preproc,               
            )
        else:
            logger.info("Main Dataset is 1D (Same as Strict 1D).")
            xtr_sw, xval_sw, xte_sw = xtr_sw_1d, xval_sw_1d, xte_sw_1d

        is_scalogram = best_hps.get("scalogram", "none") != "none"
        squeeze_tags = [("1d", [xtr_sw_1d, xval_sw_1d, xte_sw_1d])]
        if not is_scalogram:
            squeeze_tags = [("main", [xtr_sw, xval_sw, xte_sw])] + squeeze_tags
        for tag, arr_list in squeeze_tags:
            for i, arr in enumerate(arr_list):
                if arr.ndim == 4 and arr.shape[-1] == 1:
                    arr_list[i] = np.squeeze(arr, axis=-1)
                if arr.ndim == 4 and arr.shape[-2] == 1:
                    arr_list[i] = np.squeeze(arr, axis=-2)
            if tag == "main":
                xtr_sw, xval_sw, xte_sw = arr_list
            else:
                xtr_sw_1d, xval_sw_1d, xte_sw_1d = arr_list

        logger.info(f"Windowed Shapes: Main={xtr_sw.shape}, 1D={xtr_sw_1d.shape}")

        return dict(
            xtr_sw=xtr_sw, xval_sw=xval_sw, xte_sw=xte_sw,
            xtr_sw_1d=xtr_sw_1d, xval_sw_1d=xval_sw_1d, xte_sw_1d=xte_sw_1d,
            xtr_proc_sw=xtr_proc_sw, xval_proc_sw=xval_proc_sw, xte_proc_sw=xte_proc_sw,
            ytr_sw=ytr_sw, yval_sw=yval_sw, yte_sw=yte_sw,
            ytr_orig=ytr_orig, yval_orig=yval_orig, yte_orig=yte_orig,
            xtr_orig=xtr_orig, xval_orig=xval_orig, xte_orig=xte_orig,
        )

    def _run_model_loop(d, fold_tag, save_models):
        """Train and evaluate all models in the zoo for one data split.

        Args:
            d: Data dict returned by _prepare_data(), containing windowed train/
                val/test arrays for signals, process parameters, and targets.
            fold_tag: Filename suffix, e.g. "_fold{case_id}" for LOCO-CV folds.
            save_models: If True, write .keras checkpoints for each model.
        """
        xtr_sw      = d["xtr_sw"];  xval_sw     = d["xval_sw"];  xte_sw     = d["xte_sw"]
        xtr_sw_1d   = d["xtr_sw_1d"]; xval_sw_1d = d["xval_sw_1d"]; xte_sw_1d = d["xte_sw_1d"]
        xtr_proc_sw = d["xtr_proc_sw"]; xval_proc_sw = d["xval_proc_sw"]; xte_proc_sw = d["xte_proc_sw"]
        ytr_sw      = d["ytr_sw"]; yval_sw = d["yval_sw"]; yte_sw = d["yte_sw"]

        data_shapes = {
            "main_train": getattr(xtr_sw, "shape", None),
            "1d_train":   getattr(xtr_sw_1d, "shape", None),
            "proc_train": getattr(xtr_proc_sw, "shape", None),
        }

        if best_hps.get("model_name"):
            model_names = [best_hps["model_name"]]
        else:
            model_names = get_expected_models(test_mode=test_mode)

        fold_results = []
        pretrained_baselines = {}
        final_stats = []

        for name in model_names:
            logger.info(f"\n--- Training Logic for {name}{fold_tag} ---")

            if name not in pretrained_baselines:
                pretrained_baselines[name] = []

            needed_reps = 1
            rep_idx = 0
            valid_reps_count = 0
            model_scores = []
            model_times = []

            while valid_reps_count < needed_reps:
                rep_key = f"{name}_{run}{fold_tag}_rep{rep_idx}"
                res_json_path = f"{folder}/DL_{rep_key}.json"
                rep_file_pattern = f"{folder}/DL_{name}_{run}{fold_tag}_rep{rep_idx}_*.keras"
                existing_keras = glob.glob(rep_file_pattern)

                if resume_training and (existing_keras or os.path.exists(res_json_path)):
                    save_path = existing_keras[0] if existing_keras else ""
                    logger.info(f"Skipping {name} Rep {rep_idx} (Resume)")
                    if name in ["CNN", "LSTM", "BiGRU"]:
                        pretrained_baselines[name].append(save_path)
                    if os.path.exists(res_json_path):
                        try:
                            with open(res_json_path, "r") as f:
                                res_entry = json.load(f)
                                fold_results.append(res_entry)
                                model_scores.append(res_entry.get("rmse", float("inf")))
                                _t = res_entry.get("training_time") or res_entry.get("time")
                                if _t is not None:
                                    model_times.append(float(_t))
                        except Exception as e:
                            logger.warning(f"Failed to load {res_json_path}: {e}")
                            model_scores.append(float("inf"))
                    else:
                        model_scores.append(float("inf"))
                    valid_reps_count += 1
                    rep_idx += 1
                    continue

                logger.info(f">>> Training Rep {rep_idx} ({valid_reps_count+1}/{needed_reps}) for {name}{fold_tag} <<<")

                is_strict_1d = name in ["LSTM", "BiGRU"]
                is_ensemble  = "Ensemble" in name
                best_baselines_paths = {}

                ytr_use  = ytr_sw; yval_use = yval_sw; yte_use = yte_sw
                if is_ensemble:
                    tr_data = []; val_data = []; te_data = []
                    for base in ["CNN", "LSTM", "BiGRU"]:
                        best_path = get_best_model_path(folder, base, f"{run}{fold_tag}")
                        if best_path:
                            best_baselines_paths[base] = best_path
                            arr = xtr_sw if base == "CNN" else xtr_sw_1d
                            tr_data.append(arr)
                            val_data.append(xval_sw if base == "CNN" else xval_sw_1d)
                            te_data.append(xte_sw  if base == "CNN" else xte_sw_1d)
                        else:
                            logger.warning(f"Ensemble: Missing baseline {base}!")
                    if not best_baselines_paths:
                        logger.warning("Ensemble requested but NO baselines found!")
                    if "Proc" in name:
                        tr_data.append(xtr_proc_sw); val_data.append(xval_proc_sw); te_data.append(xte_proc_sw)
                    input_shape_arg = xtr_sw.shape
                elif is_strict_1d:
                    tr_data = xtr_sw_1d
                    val_data = xval_sw_1d
                    te_data = xte_sw_1d
                    input_shape_arg = xtr_sw_1d.shape
                else:
                    if "Proc" in name or "Film" in name:
                        tr_data  = [xtr_sw, xtr_proc_sw]
                        val_data = [xval_sw, xval_proc_sw]
                        te_data  = [xte_sw, xte_proc_sw]
                    else:
                        tr_data = xtr_sw
                        val_data = xval_sw
                        te_data = xte_sw
                    input_shape_arg = xtr_sw.shape

                model_hps = best_hps.copy()
                if ("Proc" in name or "Film" in name) and model_hps.get("conditioning") in (None, "no", "none"):
                    model_hps["conditioning"] = "film"
                try:
                    model = get_model(
                        model_name=name,
                        input_shape=input_shape_arg,
                        hps=model_hps,
                        proc_shape=xtr_proc_sw.shape if xtr_proc_sw is not None else None,
                        pretrained_baselines=best_baselines_paths if is_ensemble else {},
                        original_name=name,
                    )
                except Exception as _build_err:
                    logger.warning(f"Skipping {name}: get_model failed — {_build_err}")
                    break

                if is_ensemble:
                    try:
                        dummy_inputs = (
                            [arr[:1] for arr in tr_data]
                            if isinstance(tr_data, list)
                            else tr_data[:1]
                        )
                        model(dummy_inputs)
                        logger.info(f"Built {name} via dummy forward pass.")
                    except Exception as _build_err:
                        logger.warning(f"Dummy build for {name} failed: {_build_err}")

                _monitor = "val_loss"
                _epochs = epochs

                callbacks_list = get_callbacks(
                    patience=es_patience,
                    min_delta=es_min_delta,
                    monitor=_monitor,
                    mode="min",
                    reduce_lr=(scheduler != "cosine"),
                    disable_es=False,
                )

                if scheduler == "cosine":
                    n_samp = len(tr_data[0]) if isinstance(tr_data, list) else len(tr_data)
                    steps_per_epoch = n_samp // batch_size
                    lr_schedule = optimizers.schedules.CosineDecay(
                        initial_learning_rate=best_hps.get("learning_rate", 1e-3),
                        decay_steps=epochs * steps_per_epoch,
                    )
                    model.compile(
                        loss=model.loss,
                        optimizer=get_optimizer(lr_schedule),
                        metrics=[RootMeanSquaredError(name="root_mse")],
                    )

                start_time = time.time()
                _fit_batch_size = batch_size
                fit_kwargs = dict(batch_size=_fit_batch_size, epochs=_epochs, callbacks=callbacks_list)
                history = model.fit(tr_data, ytr_use,
                                    validation_data=(val_data, yval_use), **fit_kwargs)

                res_entry = {
                    "name": f"{name}_rep{rep_idx}",
                    "arch": name,
                    "fold_tag": fold_tag,
                    "val_loss":     (min(history.history["val_loss"])     if "val_loss"     in history.history else 0),
                    "val_root_mse": (min(history.history["val_root_mse"]) if "val_root_mse" in history.history else 0),
                }
                
                _save_this = save_models
                res_entry = evaluate_model(
                    model=model,
                    x=te_data,
                    y=yte_use,
                    folder=folder,
                    run_name=rep_key,
                    history=history,
                    training_time=time.time() - start_time,
                    model_results=res_entry,
                    norm_target=False,
                    save_model=_save_this,
                )

                test_rmse = res_entry.get("rmse", 0.0)
                logger.info(f"Test RMSE (Rep {rep_idx}){fold_tag}: {test_rmse:.4f}")

                alldata_csv_path = None
                try:
                    assert tr_data is not None and val_data is not None and te_data is not None
                    assert ytr_use is not None and yval_use is not None and yte_use is not None
                    if isinstance(tr_data, list):
                        all_x = [np.concatenate([tr, va, te], axis=0)
                                 for tr, va, te in zip(tr_data, val_data, te_data)]
                    else:
                        all_x = np.concatenate([tr_data, val_data, te_data], axis=0)
                    all_y = np.concatenate([ytr_use, yval_use, yte_use], axis=0)
                    all_pred = model.predict(all_x, verbose=0).squeeze()
                    alldata_csv_path = f"{folder}/DL_{rep_key}_alldata_pred.csv"
                    pd.DataFrame({"gt": all_y.squeeze(), "pred": all_pred}).to_csv(
                        alldata_csv_path, index=False
                    )
                    logger.info(f"alldata_pred saved → {alldata_csv_path}")
                except Exception as _e:
                    logger.warning(f"alldata_pred save failed for {rep_key}: {_e}")

                res_entry["alldata_pred_csv"] = alldata_csv_path
                fold_results.append(res_entry)
                model_scores.append(test_rmse)
                _rep_time = res_entry.get("training_time") or res_entry.get("time")
                if _rep_time is not None:
                    model_times.append(float(_rep_time))

                save_path = res_entry.get("model_file", "")
                if name in ["CNN", "LSTM", "BiGRU"]:
                    pretrained_baselines[name].append(save_path)

                del model
                clear_session()
                gc.collect()

                valid_reps_count += 1
                rep_idx += 1

            valid_scores = [s for s in model_scores if s != float("inf")]
            if valid_scores:
                _stat = {
                    "Model":      name + fold_tag,
                    "Avg_RMSE":   np.mean(valid_scores),
                    "Best_RMSE":  np.min(valid_scores),
                    "Worst_RMSE": np.max(valid_scores),
                    "Std_RMSE":   np.std(valid_scores),
                    "Reps":       len(valid_scores),
                }
                if model_times:
                    _stat["Total_Time_s"] = sum(model_times)
                    _stat["Avg_Time_s"]   = sum(model_times) / len(model_times)
                final_stats.append(_stat)

        return fold_results, final_stats, data_shapes

    all_results = []
    all_stats   = []
    dispatch_shapes = {}

    logger.info("Loading full PAPER_TRAIN_POOL data for LOCO-CV (single load)...")
    _nasa_pool, _, _nasa_test, _transforms = get_nasa_data_pipeline(
        run_name=run,
        window_size=sliding_window_size,
        stride=sliding_window_stride,
        split_type="paper",
        seed=seed,
        apply_averaging=False,
        windowing=False,
        custom_train_cases=PAPER_TRAIN_POOL,
        custom_val_cases=[],
    )
    _case_labels = _transforms["case_labels"] 
    _xtr_pool  = _nasa_pool.data.cpu().numpy()
    _ytr_pool  = _nasa_pool.targets.cpu().numpy()
    _xpr_pool  = _nasa_pool.proc_data.cpu().numpy()
    _xft_pool  = _nasa_pool.feat_data.cpu().numpy()
    _xte_raw   = _nasa_test.data.cpu().numpy()
    _yte_raw   = _nasa_test.targets.cpu().numpy()
    _xte_proc  = _nasa_test.proc_data.cpu().numpy()
    _xte_feat  = _nasa_test.feat_data.cpu().numpy()

    loco_cv_folds = [[c] for c in PAPER_TRAIN_POOL]
    for fold_idx, fold_val_list in enumerate(loco_cv_folds):
        left_out = fold_val_list[0]
        logger.info(
            f"\n=== LOCO Fold {fold_idx + 1}/{len(loco_cv_folds)} "
            f"— held-out case: {left_out} ==="
        )
        tr_mask  = ~np.isin(_case_labels, fold_val_list)
        val_mask =  np.isin(_case_labels, fold_val_list)
        try:
            data = _prepare_data(
                preloaded_tr=(
                    _xtr_pool[tr_mask], _xpr_pool[tr_mask],
                    _xft_pool[tr_mask], _ytr_pool[tr_mask],
                ),
                preloaded_val=(
                    _xtr_pool[val_mask], _xpr_pool[val_mask],
                    _xft_pool[val_mask], _ytr_pool[val_mask],
                ),
                preloaded_te=(_xte_raw, _xte_proc, _xte_feat, _yte_raw),
            )
            fold_res, fold_stats, fold_shapes = _run_model_loop(
                data, fold_tag=f"_fold{left_out}", save_models=True
            )
            all_results.extend(fold_res)
            all_stats.extend(fold_stats)
            dispatch_shapes = fold_shapes
        except Exception as _fold_err:
            logger.error(f"LOCO fold {left_out} failed: {_fold_err}")
            import traceback as _tb
            logger.error(_tb.format_exc())

    logger.info("\n=== Best-fold selection ===")
    arch_results = defaultdict(list)
    for r in all_results:
        if r.get("arch") and r.get("model_file"):
            arch_results[r["arch"]].append(r)

    best_fold_selection = {}
    for arch, entries in arch_results.items():
        best = min(entries, key=lambda e: e.get("rmse", float("inf")))
        best_fold_selection[arch] = {
            "fold_tag":        best["fold_tag"],
            "rmse":            best.get("rmse"),
            "model_file":      best.get("model_file", ""),
            "val_loss":        best.get("val_loss"),
            "alldata_pred_csv": best.get("alldata_pred_csv", ""),
        }
        
        for e in entries:
            if e is not best:
                kf = e.get("model_file", "")
                if kf and os.path.exists(kf):
                    try:
                        os.remove(kf)
                        logger.info(f"Deleted non-best fold model: {os.path.basename(kf)}")
                    except Exception as _e:
                        logger.warning(f"Could not delete {kf}: {_e}")
                cf = e.get("alldata_pred_csv", "")
                if cf and os.path.exists(cf):
                    try:
                        os.remove(cf)
                        logger.info(f"Deleted non-best alldata CSV: {os.path.basename(cf)}")
                    except Exception as _e:
                        logger.warning(f"Could not delete {cf}: {_e}")

        logger.info(
            f"{arch}: best fold={best['fold_tag']}, "
            f"RMSE={best.get('rmse', 'N/A'):.4f}, "
            f"model={os.path.basename(best.get('model_file', ''))}"
        )

    best_fold_path = f"{folder}/best_fold_selection.json"
    with open(best_fold_path, "w") as _bf:
        json.dump(best_fold_selection, _bf, indent=2, cls=NumpyFloatValuesEncoder)
    logger.info(f"Best-fold written → {best_fold_path}")

    histories = {}
    for r in all_results:
        if "history" in r:
            histories[r["name"]] = r["history"]

    with open(f"{folder}/DL_{run}_history.json", "w") as f:
        json.dump(histories, f, cls=NumpyFloatValuesEncoder)

    df = pd.DataFrame(all_results)
    if "history" in df.columns:
        df = df.drop(["history"], axis=1)
    df.to_csv(f"{folder}/DL_{run}_scores.csv", sep=";", decimal=".")
    logger.info(df)

    if all_stats:
        df_stats = pd.DataFrame(all_stats)
        df_stats.to_csv(
            f"{folder}/DL_{run}_final_benchmark_stats.csv", sep=";", decimal="."
        )
        logger.info("\n=== FINAL BENCHMARK STATS ===")
        logger.info(df_stats)

    logger.info("================== Training finished")
    logger.info(f"---Script time\n--- {(time.time() - script_time):.4f} seconds ---")
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    clear_session()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "results": all_results,
        "shapes": dispatch_shapes,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Manual training pipeline for NASA data."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3000, help="Number of epochs"
    )
    parser.add_argument(
        "-r",
        "--retrain",
        action="store_true",
        help="Retrain models even if they exist (opposite of resume_training)",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Run in test mode (smaller subset of models)",
    )
    parser.add_argument(
        "-sw", "--window", type=int, default=250, help="Sliding Window Size"
    )
    parser.add_argument(
        "-ss", "--stride", type=int, default=125, help="Sliding Window Stride"
    )
    parser.add_argument(
        "-hp",
        "--hp_file",
        type=str,
        default="configs/champion_model.json",
        help="Path to HPs json (from preprocessing_tuner)",
    )
    parser.add_argument(
        "-run",
        "--run",
        type=str,
        default="",
        help="Specific run name (e.g., 'all', 'AC', 'DC')",
    )
    parser.add_argument(
        "-sched",
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine"],
        help="Learning rate scheduler: 'plateau' or 'cosine'",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Label for the first part of the output folder name "
            "(e.g. 'timefreq_033'). Defaults to 'final' or 'final_optimized'. "
            "Keeps --run free for dataset signal-group selection."
        ),
    )

    args = parser.parse_args()

    if args.run != "":
        run_names = [args.run]
    else:
        run_names = ["all"]

    print(f"================== Starting Final Benchmark ==================")
    print(f"Epochs: {args.epochs}")
    print(f"Window/Stride: {args.window} / {args.stride}")
    print(f"HP File: {args.hp_file if args.hp_file else 'Default (MA)'}")
    print(f"Test Mode: {args.test}")
    print(f"Resume: {not args.retrain}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Batch Size: {args.batch_size}")

    for run_name in run_names:
        print(f"\n>>> Processing Signal Group: {run_name} <<<")
        main(
            epochs=args.epochs,
            sliding_window_size=args.window,
            sliding_window_stride=args.stride,
            run=run_name,
            resume_training=not args.retrain,
            seed=42,
            test_mode=args.test,
            hp_file=args.hp_file,
            scheduler=args.scheduler,
            batch_size=args.batch_size,
            experiment=args.experiment,
        )
