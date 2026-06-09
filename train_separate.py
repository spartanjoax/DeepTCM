################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

#!/usr/bin/env python
"""Separate-model LOCO-CV baseline for DeepTCM continual learning comparison.

Trains one independent model per experience (condition) using only that
experience's own training data.

Usage:
    python continual/train_separate.py --signal_group AC --datasets nasa --seed 42
    python continual/train_separate.py --signal_group internals --datasets muss muci
"""

import argparse
import datetime
import json
import logging
import math
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import re
import time

import numpy as np
import torch
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.evaluation.metrics import (
    loss_metrics,
    timing_metrics,
    ram_usage_metrics,
    gpu_usage_metrics,
    rmse_metrics,
    r2_metrics,
)
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
from avalanche.logging import TextLogger, InteractiveLogger

from data.datasets import get_nasa_data_pipeline
from data.mu_tcm_datasets import get_mu_tcm_scenario_data
from helpers import NumpyFloatValuesEncoder
from helpers.preprocessing import apply_full_preprocessing
from models.dl_models_torch import get_torch_model

NASA_TRAIN_CASES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]

class NumpyTCMDataset(torch.utils.data.Dataset):
    """Wraps preprocessed numpy arrays in Avalanche-compatible dict format."""

    def __init__(self, x_sig, x_proc, y):
        """Initialise the dataset from pre-windowed numpy arrays.

        Args:
            x_sig:  Signal array of shape (N, C, T) or (N, F, T, 1).
            x_proc: Process-parameter array of shape (N, P).
            y:      Target VB array of shape (N,) or (N, 1).
        """
        self.data      = torch.tensor(x_sig,  dtype=torch.float32)
        self.proc_data = torch.tensor(x_proc, dtype=torch.float32)
        targets = np.array(y, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx):
        """Return the multimodal input dict and target for sample at *idx*."""
        return (
            {"x": self.data[idx], "proc_data": self.proc_data[idx]},
            self.targets[idx],
        )

MU_SS_CONDITIONS = [
    {"material": 2},
]

MU_CI_CONDITIONS = [
    {"material": 1},
]

NASA_SIGNAL_GROUPS  = ["AC", "AC_table", "DC", "DC_table"]
MU_SIGNAL_GROUPS    = ["AC", "AC_table", "DC", "DC_table", "internals"]


def _condition_label(filters):
    """Human-readable string for a condition filter dict."""
    mat = {1: "CI", 2: "SS"}.get(filters.get("material", 0), "?")
    return f"{mat}_Vc{filters.get('Vc','?')}_fz{filters.get('fz','?')}"


def _extract_rmse(eval_result):
    """Extract RMSE for experience 0 from an Avalanche eval result dict."""
    for key, val in eval_result.items():
        if re.search(r"Top1_RMSE_Exp/eval_phase/test_stream/Exp0", key):
            return float(val[0]) if isinstance(val, (list, tuple)) else float(val)
    for key, val in eval_result.items():
        if "Top1_RMSE_Exp" in key and "test_stream" in key:
            return float(val[0]) if isinstance(val, (list, tuple)) else float(val)
    return float("nan")


def _extract_r2(eval_result):
    """Extract R² for experience 0."""
    for key, val in eval_result.items():
        if re.search(r"Top1_R2_Exp/eval_phase/test_stream/Exp0", key):
            return float(val[0]) if isinstance(val, (list, tuple)) else float(val)
    for key, val in eval_result.items():
        if "Top1_R2_Exp" in key and "test_stream" in key:
            return float(val[0]) if isinstance(val, (list, tuple)) else float(val)
    return float("nan")


def _train_one_model(
    signal_group,
    train_ds,
    test_ds,
    champion_hps,
    group_model_configs,
    device,
    args,
    scenario_dir,
    label,
    logger,
):
    """Train a single separate model and return (rmse, r2, time_s).

    Args:
        signal_group (str): Signal group key (e.g. ``'AC_table'``,
            ``'internals'``).
        train_ds (NumpyTCMDataset): Pre-processed training dataset.
        test_ds (NumpyTCMDataset): Pre-processed test dataset.
        champion_hps (dict): Champion preprocessing and architecture HP dict.
        group_model_configs (dict): Mapping of signal_group \u2192 model config dict
            with at least a ``'model_name'`` key.
        device (str or torch.device): Device to train on.
        args: Parsed ``argparse.Namespace`` with ``epochs``, ``patience``, and
            ``seed`` attributes.
        scenario_dir (str): Output directory for text logs.
        label (str): Human-readable label for this fold used in filenames and
            log messages.
        logger: Logger instance for info messages.

    Returns:
        tuple: ``(rmse, r2, time_s)`` where ``rmse`` and ``r2`` are the final
        test-set metrics and ``time_s`` is the elapsed wall-clock time.
    """
    batch_size = 8 if "table" in signal_group else 16

    av_train = AvalancheDataset(train_ds)
    av_test  = AvalancheDataset(test_ds)
    benchmark = benchmark_from_datasets(train=[av_train], test=[av_test])
    benchmark = benchmark_with_validation_stream(benchmark, validation_size=0.2, shuffle=True, seed=args.seed)

    sample_dict, _ = train_ds[0]
    input_shape = tuple(sample_dict["x"].shape)
    proc_shape  = tuple(sample_dict["proc_data"].shape)

    sg_model_cfg  = group_model_configs.get(signal_group, {})
    model_name    = sg_model_cfg.get("model_name", "CNN_LSTM_Film")
    hps = {k: v for k, v in champion_hps.items()}
    model = get_torch_model(model_name, input_shape, hps, proc_shape=proc_shape)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    Model: {model_name}, params={n_params:,}")

    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=champion_hps.get("learning_rate", 1e-3),
        eps=1e-7,
        decoupled_weight_decay=True,
        weight_decay=champion_hps.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = torch.nn.SmoothL1Loss()

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    text_log_path = os.path.join(scenario_dir, f"sep_{label}_{run_ts}.txt")
    text_logger = TextLogger(open(text_log_path, "a"))
    std_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        loss_metrics(epoch=True, experience=True, stream=True),
        rmse_metrics(epoch=True, experience=True, stream=True),
        r2_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(experience=True, stream=True),
        ram_usage_metrics(experience=True, stream=True),
        gpu_usage_metrics(0, epoch=True, experience=True, stream=True),
        loggers=[text_logger, std_logger],
    )

    es = EarlyStoppingPlugin(
        patience=args.patience,
        val_stream_name="valid_stream",
        metric_name="Loss_Exp",
        mode="min",
        margin=0.0001,
    )
    sched_plugin = LRSchedulerPlugin(scheduler, metric="val_loss")

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=batch_size,
        train_epochs=args.epochs,
        eval_every=1,
        plugins=[es, sched_plugin],
        evaluator=eval_plugin,
        eval_mb_size=batch_size,
        device=device,
    )

    start = time.time()
    strategy.train(
        benchmark.train_stream[0],
        eval_streams=[benchmark.valid_stream[0]],
    )
    eval_result = strategy.eval(benchmark.test_stream)
    elapsed = time.time() - start

    rmse = _extract_rmse(eval_result)
    r2   = _extract_r2(eval_result)
    logger.info(f"    {label}: RMSE={rmse:.4f}  R²={r2:.3f}  ({elapsed:.0f}s)")

    del strategy, model, optimizer, scheduler, criterion, eval_plugin, es, sched_plugin
    del text_logger, benchmark, av_train, av_test
    torch.cuda.empty_cache()

    return rmse, r2, elapsed


def run_nasa(signal_group, args, champion_hps, group_model_configs, device, output_dir, logger):
    """Train and evaluate independent models on the NASA Ames Milling paper split.

    Args:
        signal_group:       Signal group key (e.g. "all", "AC", "DC").
        args:               Parsed argparse namespace.
        champion_hps:       Champion preprocessing HP dict.
        group_model_configs: List of model config dicts for this signal group.
        device:             Torch device string.
        output_dir:         Directory in which to write result JSON.
        logger:             Logger instance.
    """
    result_path = os.path.join(
        output_dir, f"{signal_group}_nasa_seed{args.seed}_results.json"
    )
    if args.resume and os.path.exists(result_path):
        logger.info(f"  Skipping NASA {signal_group} (exists)")
        return

    logger.info(f"  NASA | signal_group={signal_group}")

    sw_cfg = {"window_size": args.window_size, "stride": args.window_stride}

    train_raw, _, test_raw, _ = get_nasa_data_pipeline(
        run_name=signal_group,
        window_size=args.window_size,
        stride=args.window_stride,
        split_type="paper",
        seed=args.seed,
        apply_averaging=False,
        windowing=False,
        custom_train_cases=NASA_TRAIN_CASES,
        custom_val_cases=[],
    )

    xtr  = train_raw.data.cpu().numpy()
    ptr  = train_raw.proc_data.cpu().numpy()
    ytr  = train_raw.targets.cpu().numpy().ravel()
    xte  = test_raw.data.cpu().numpy()
    pte  = test_raw.proc_data.cpu().numpy()
    yte  = test_raw.targets.cpu().numpy().ravel()

    x_sw, ytr_sw, x_proc_sw, stats_2d = apply_full_preprocessing(
        [xtr, ptr], ytr, champion_hps, split="train",
        sliding_window_config=sw_cfg
    )
    xte_sw, yte_sw, xte_proc_sw, _ = apply_full_preprocessing(
        [xte, pte], yte, champion_hps, split="test",
        preproc_stats=stats_2d,
        sliding_window_config=sw_cfg
    )

    train_ds = NumpyTCMDataset(x_sw, x_proc_sw, ytr_sw)
    test_ds  = NumpyTCMDataset(xte_sw, xte_proc_sw, yte_sw)

    rmse, r2, elapsed = _train_one_model(
        signal_group, train_ds, test_ds,
        champion_hps, group_model_configs, device, args, output_dir, "nasa", logger,
    )

    out = {
        "type": "separate",
        "signal_group": signal_group,
        "dataset": "nasa",
        "seed": args.seed,
        "n_folds": 1,
        "rmse": rmse,
        "r2": r2,
        "eae": rmse,
        "eas": r2,
        "time_seconds": elapsed,
        "train_cases": NASA_TRAIN_CASES,
        "test_cases": [11, 12, 15, 16],
        "folds": [{
            "condition": "paper_split",
            "rmse": rmse,
            "r2": r2,
            "time_s": elapsed,
            "skipped": False,
        }],
    }
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyFloatValuesEncoder)
    logger.info(f"    Saved {result_path}")


def run_mu_tcm(signal_group, conditions, dataset_tag, args, champion_hps,
               group_model_configs, device, output_dir, logger):
    """Train and evaluate independent models per MU-TCM cutting condition.

    Args:
        signal_group:       Signal group key.
        conditions:         List of (condition_id, condition_label) pairs.
        dataset_tag:        Short tag string for filenames (e.g. "mu_tcm_GG30_dry").
        args:               Parsed argparse namespace.
        champion_hps:       Champion preprocessing HP dict.
        group_model_configs: List of model config dicts for this signal group.
        device:             Torch device string.
        output_dir:         Directory in which to write result JSON.
        logger:             Logger instance.
    """
    result_path = os.path.join(
        output_dir, f"{signal_group}_{dataset_tag}_seed{args.seed}_results.json"
    )
    if args.resume and os.path.exists(result_path):
        logger.info(f"  Skipping {dataset_tag} {signal_group} (exists)")
        return

    logger.info(f"  {dataset_tag} | signal_group={signal_group} | {len(conditions)} folds")

    sw_cfg = {"window_size": args.window_size, "stride": args.window_stride}

    folds = []
    total_start = time.time()

    mu_data = get_mu_tcm_scenario_data(
        signal_group=signal_group,
        scenario_filters=conditions,
    )

    for exp in mu_data:
        cond_filters = exp["filters"]
        cond_label   = _condition_label(cond_filters)
        logger.info(f"    Fold: {cond_label}")
        logger.info(
            f"      {exp['n_train'] + exp['n_val'] + exp['n_test']} runs → "
            f"train={exp['n_train']}, val={exp['n_val']}, test={exp['n_test']}"
        )

        if exp["train"] is None:
            logger.warning(f"      No train runs for {cond_label} — skipping fold")
            folds.append({
                "condition": cond_label,
                "filters": cond_filters,
                "rmse": float("nan"),
                "r2": float("nan"),
                "n_train_runs": exp["n_train"],
                "n_val_runs":   exp["n_val"],
                "n_test_runs":  exp["n_test"],
                "time_s": 0.0,
                "skipped": True,
            })
            continue

        x_tr_z, proc_tr_n, y_tr = exp["train"]
        x_te_z, proc_te_n, y_te = exp["test"]

        x_sw, ytr_sw, x_proc_sw, stats_2d = apply_full_preprocessing(
            [x_tr_z, proc_tr_n], y_tr, champion_hps, split="train",
            sliding_window_config=sw_cfg
        )
        if x_sw.shape[0] == 0:
            logger.warning(f"      No train windows for {cond_label} — skipping fold")
            folds.append({
                "condition": cond_label, "filters": cond_filters,
                "rmse": float("nan"), "r2": float("nan"),
                "n_train_runs": exp["n_train"], "n_val_runs": exp["n_val"],
                "n_test_runs": exp["n_test"], "time_s": 0.0, "skipped": True,
            })
            continue

        xte_sw, yte_sw, xte_proc_sw, _ = apply_full_preprocessing(
            [x_te_z, proc_te_n], y_te, champion_hps, split="test",
            preproc_stats=stats_2d,
            sliding_window_config=sw_cfg
        )

        train_ds = NumpyTCMDataset(x_sw, x_proc_sw, ytr_sw)
        test_ds  = NumpyTCMDataset(xte_sw, xte_proc_sw, yte_sw)

        rmse, r2, fold_time = _train_one_model(
            signal_group, train_ds, test_ds,
            champion_hps, group_model_configs, device, args, output_dir, cond_label, logger,
        )

        folds.append({
            "condition": cond_label,
            "filters": cond_filters,
            "rmse": rmse,
            "r2": r2,
            "n_train_runs": exp["n_train"],
            "n_val_runs":   exp["n_val"],
            "n_test_runs":  exp["n_test"],
            "time_s": fold_time,
            "skipped": False,
        })

    total_elapsed = time.time() - total_start
    valid_folds   = [f for f in folds if not f.get("skipped") and math.isfinite(f.get("rmse", float("nan")))]
    valid_r2_folds = [f for f in folds if not f.get("skipped") and math.isfinite(f.get("r2", float("nan")))]
    eae = float(np.mean([f["rmse"] for f in valid_folds]))    if valid_folds    else float("nan")
    eas = float(np.mean([f["r2"]   for f in valid_r2_folds])) if valid_r2_folds else float("nan")

    out = {
        "type": "separate",
        "signal_group": signal_group,
        "dataset": dataset_tag,
        "seed": args.seed,
        "n_folds": len(conditions),
        "rmse": eae,
        "r2": eas,
        "eae": eae,
        "eas": eas,
        "time_seconds": total_elapsed,
        "folds": folds,
    }
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyFloatValuesEncoder)
    logger.info(f"    EAE={eae:.4f}  EAS={eas}  Saved {result_path}")


def main(args):
    """Run per-experience independent model training for all signal groups.

    Iterates over signal groups, datasets (NASA and MU-TCM), and conditions.
    Each combination trains a freshly initialised model with no continual
    learning, providing a no-CL lower-bound reference.

    Args:
        args: Parsed argparse namespace (see parse_args()).
    """
    output_dir = os.path.join("continual", "results", "separate")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger("train_separate")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_dir, f"sep_{timestamp}.log"))
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Arguments: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    champion_cfg_path = os.path.join("configs", "champion_timefreq_domain.json")
    with open(champion_cfg_path) as f:
        champion_hps = json.load(f)
    logger.info(f"Champion HPs loaded from {champion_cfg_path}")

    group_cfg_path = os.path.join("configs", "cl_models.json")
    with open(group_cfg_path) as f:
        group_model_configs = json.load(f)
    logger.info(f"Group model configs loaded from {group_cfg_path}")

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["nasa", "muss", "muci"]

    signal_groups = args.signal_groups
    if "all" in signal_groups:
        signal_groups = list(set(NASA_SIGNAL_GROUPS + MU_SIGNAL_GROUPS))

    for sg in signal_groups:
        logger.info(f"=== Signal group: {sg} ===")

        if "nasa" in datasets:
            if sg not in NASA_SIGNAL_GROUPS:
                logger.info(f"  Skipping NASA for signal_group={sg} (not applicable)")
            else:
                run_nasa(sg, args, champion_hps, group_model_configs, device, output_dir, logger)

        if "muss" in datasets:
            run_mu_tcm(sg, MU_SS_CONDITIONS, "muss", args, champion_hps,
                       group_model_configs, device, output_dir, logger)

        if "muci" in datasets:
            run_mu_tcm(sg, MU_CI_CONDITIONS, "muci", args, champion_hps,
                       group_model_configs, device, output_dir, logger)


def parse_args():
    """Parse command-line arguments for the separate-model baseline script."""
    p = argparse.ArgumentParser(
        description="DeepTCM separate-model LOCO-CV baseline"
    )
    p.add_argument(
        "--signal_groups",
        nargs="+",
        default=["all"],
        help="Signal group(s) to run, or 'all' (default: all)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["nasa", "muss", "muci", "all"],
        help="Datasets to run (default: all)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--window_size", type=int, default=500)
    p.add_argument("--window_stride", type=int, default=500)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
