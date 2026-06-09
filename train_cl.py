################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Continual Learning training script for DeepTCM.

Usage:
    python continual/train_cl.py --epochs 200 --patience 20
    python continual/train_cl.py --scenarios scenario_1 scenario_3
    python continual/train_cl.py --strategies ewc si --no-resume
"""

import argparse
import datetime
import json
import logging
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
logging.captureWarnings(True)

import random
import re
import time

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid

from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training import Naive, SynapticIntelligence
from continual.strategies.gem import AGEM
from avalanche.evaluation.metrics import (
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    ram_usage_metrics,
    gpu_usage_metrics,
    rmse_metrics,
    r2_metrics,
)
from avalanche.evaluation.metrics.regression_forgetting import (
    rmse_forgetting_metrics,
    r2_forgetting_metrics,
)
from avalanche.training.plugins import (
    EvaluationPlugin,
    EarlyStoppingPlugin,
    ReplayPlugin,
    LRSchedulerPlugin,
)
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint

from data.datasets import get_nasa_data_pipeline
from data.mu_tcm_datasets import get_mu_tcm_scenario_data
from helpers import NumpyFloatValuesEncoder
from helpers.preprocessing import apply_full_preprocessing

from continual.strategies import Cumulative, EWC, MASMultimodal

from models.dl_models_torch import get_torch_model

import copy as _copy

_REPLAY_SC125 = [0, 75, 150]
_REPLAY_SC346 = [0, 50, 100]

STRATEGIES = {
    "cumulative": {"replay_mem": [0]},
    "naive": {},
    "ewc": [
        {
            "lambda": [0.1, 1, 10, 100],
            "mode": ["online"],
            "decayfactor": [0.70, 0.90, 0.95],
        },
        {
            "lambda": [0.1, 1, 10, 100],
            "mode": ["separate"],
        },
    ],
    "si": {
        "lambda": [0.1, 1, 10, 100],
    },
    "mas": {
        "lambda_reg": [0.1, 1, 10, 100],
        "alpha": [0.3, 0.5, 0.7],
    },
    "agem": {
        "patterns_per_exp": [48],
    },
}

SCENARIO_1 = {
    "mu_filters": [
        {"material": 2},
        {"material": 1},
    ],
    "runs": ["AC", "AC_table", "DC", "DC_table"],
    "batch_size": [16, 8, 16, 8],
    "nasa_position": "first",
}

SCENARIO_2 = {
    "mu_filters": [
        {"material": 2, "fz": 0.05, "Vc": 50},
        {"material": 2, "fz": 0.1,  "Vc": 50},
        {"material": 2, "fz": 0.05, "Vc": 100},
        {"material": 2, "fz": 0.1,  "Vc": 100},
        {"material": 1, "fz": 0.1,  "Vc": 100},
        {"material": 1, "fz": 0.2,  "Vc": 100},
        {"material": 1, "fz": 0.1,  "Vc": 200},
        {"material": 1, "fz": 0.2,  "Vc": 200},
    ],
    "runs": ["AC", "AC_table", "DC", "DC_table"],
    "batch_size": [16, 8, 16, 8],
    "nasa_position": "first",
}

SCENARIO_3 = {
    "mu_filters": [
        {"material": 1},
        {"material": 2, "fz": 0.05, "Vc": 50},
        {"material": 2, "fz": 0.1,  "Vc": 50},
        {"material": 2, "fz": 0.05, "Vc": 100},
        {"material": 2, "fz": 0.1,  "Vc": 100},
    ],
    "runs": ["internals"],
    "batch_size": [16],
    "nasa_position": "none",
}

SCENARIO_4 = {
    "mu_filters": [
        {"material": 2},
        {"material": 1, "fz": 0.1,  "Vc": 100},
        {"material": 1, "fz": 0.2,  "Vc": 100},
        {"material": 1, "fz": 0.1,  "Vc": 200},
        {"material": 1, "fz": 0.2,  "Vc": 200},
    ],
    "runs": ["internals"],
    "batch_size": [16],
    "nasa_position": "none",
}

SCENARIO_5 = {
    "mu_filters": [
        {"material": 1},
        {"material": 2},
    ],
    "runs": ["AC", "AC_table", "DC", "DC_table"],
    "batch_size": [16, 8, 16, 8],
    "nasa_position": "last",
}

SCENARIO_6 = {
    "mu_filters": [
        {"material": 1, "fz": 0.2, "Vc": 200},
        {"material": 1, "fz": 0.1, "Vc": 200},
        {"material": 1, "fz": 0.2, "Vc": 100},
        {"material": 1, "fz": 0.1, "Vc": 100},
        {"material": 2, "fz": 0.1, "Vc": 100},
        {"material": 2, "fz": 0.05, "Vc": 100},
        {"material": 2, "fz": 0.1,  "Vc": 50},
        {"material": 2, "fz": 0.05, "Vc": 50},
    ],
    "runs": ["internals"],
    "batch_size": [16],
    "nasa_position": "none",
}

SCENARIO_1["replay_mem"]            = _REPLAY_SC125
SCENARIO_1["agem_patterns_per_exp"] = [48]

SCENARIO_2["replay_mem"]            = _REPLAY_SC125
SCENARIO_2["agem_patterns_per_exp"] = [8]

SCENARIO_3["replay_mem"]            = _REPLAY_SC346
SCENARIO_3["agem_patterns_per_exp"] = [20]

SCENARIO_4["replay_mem"]            = _REPLAY_SC346
SCENARIO_4["agem_patterns_per_exp"] = [48]

SCENARIO_5["replay_mem"]            = _REPLAY_SC125
SCENARIO_5["agem_patterns_per_exp"] = [48]

SCENARIO_6["replay_mem"]            = _REPLAY_SC346
SCENARIO_6["agem_patterns_per_exp"] = [20]

SCENARIOS = {
    "scenario_1": SCENARIO_1,
    "scenario_2": SCENARIO_2,
    "scenario_3": SCENARIO_3,
    "scenario_4": SCENARIO_4,
    "scenario_5": SCENARIO_5,
    "scenario_6": SCENARIO_6,
}

PROC_MIN = [0.1, 0.05, 1, 50]
PROC_MAX = [1.5, 0.5, 2, 200]


class NumpyTCMDataset(torch.utils.data.Dataset):
    """Wraps preprocessed (signal, proc, labels) numpy arrays in the
    Avalanche-compatible dict format expected by the model zoo.

    Args:
        x_sig  : np.ndarray  shape (N, ...)  — windowed signal array
                 (may be 3-D (N,W,C) for raw or 4-D (N,F,T,C) for STFT).
        x_proc : np.ndarray  shape (N, P)   — normalised proc params.
        y      : np.ndarray  shape (N,) or (N,1) — VB targets.
    """

    def __init__(self, x_sig, x_proc, y):
        """Initialise the dataset from pre-windowed numpy arrays.

        Args:
            x_sig:  Signal array of shape (N, C, T) or (N, F, T, 1).
            x_proc: Process-parameter array of shape (N, P).
            y:      Target VB array of shape (N,) or (N, 1).
        """
        self.proc_data = torch.tensor(x_proc, dtype=torch.float32)
        targets = np.array(y, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]
        self.targets   = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, idx):
        """Return the multimodal input dict and target for sample at *idx*."""
        return (
            {"x": self.data[idx], "proc_data": self.proc_data[idx]},
            self.targets[idx],
        )


def _apply_scenario_overrides(strat_key, base_grid, scenario_config):
    """Apply per-scenario replay_mem and agem_patterns_per_exp overrides.

    Injects scenario-specific grid values so that strategies with a global
    default receive the correct per-scenario values without mutating STRATEGIES.

    Args:
        strat_key       : e.g. "ewc", "agem"
        base_grid       : dict or list-of-dicts from STRATEGIES[strat_key]
        scenario_config : the SCENARIO_N dict for the active scenario

    Returns:
        dict or list-of-dicts with replay_mem / patterns_per_exp overridden.
    """
    sc_replay = scenario_config.get("replay_mem")
    sc_ppe    = scenario_config.get("agem_patterns_per_exp")

    def _patch(d):
        """Apply scenario-specific override to a single strategy config dict."""
        out = dict(d)
        if (strat_key not in ("cumulative")
                and "replay_mem" in out and sc_replay is not None):
            out["replay_mem"] = sc_replay
        if strat_key == "agem" and "patterns_per_exp" in out and sc_ppe is not None:
            out["patterns_per_exp"] = sc_ppe
        return out

    if isinstance(base_grid, list):
        return [_patch(d) for d in base_grid]
    return _patch(base_grid)


def truncate_to_batch_size(dataset, batch_size):
    """Truncate dataset so its length is divisible by batch_size.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to truncate.
        batch_size (int): Target batch-size divisor.

    Returns:
        torch.utils.data.Dataset: Original dataset if already divisible, or a
        ``torch.utils.data.Subset`` truncated to be exactly divisible.
    """
    n = len(dataset)
    new_n = (n // batch_size) * batch_size
    if 0 < new_n < n:
        return torch.utils.data.Subset(dataset, list(range(new_n)))
    return dataset


def extract_experience_rmse(eval_result):
    """Extract per-experience RMSE values from an Avalanche eval result dict.

    Returns:
        dict mapping experience_id (int) → RMSE (float).
    """
    rmses = {}
    for key, val in eval_result.items():
        m = re.search(r"Top1_RMSE_Exp/eval_phase/test_stream/Exp(\d+)", key)
        if m:
            exp_id = int(m.group(1))
            rmses[exp_id] = float(val[0]) if isinstance(val, tuple) else float(val)
    return rmses


def compute_max_forgetting(strat_results, n_experiences):
    """Max-drop forgetting: worst-case RMSE increase from any experience's best.

    For each experience k, find its best (lowest) RMSE across all eval steps
    and its final RMSE. Returns the maximum and mean positive drops.
    Positive values = catastrophic forgetting; 0 = no forgetting or improvement.

    Args:
        strat_results (list[dict]): List of Avalanche eval-result dicts, one per
            training step.
        n_experiences (int): Total number of CL experiences.

    Returns:
        dict: ``{"max": float, "mean": float}`` — max and mean positive RMSE
        increases (i.e. forgetting) across all experiences.
    """
    if n_experiences < 2 or len(strat_results) < 2:
        return {"max": 0.0, "mean": 0.0}

    rmse_per_step = [
        extract_experience_rmse(sr) for sr in strat_results
    ]
    final_rmses = rmse_per_step[-1]

    drops = []
    for k in range(n_experiences):
        rmse_k_series = [
            step[k] for step in rmse_per_step if k in step
        ]
        if not rmse_k_series:
            continue
        best = min(rmse_k_series)
        final = final_rmses.get(k)
        if final is not None:
            drops.append(max(0.0, final - best))

    if not drops:
        return {"max": 0.0, "mean": 0.0}
    return {"max": float(max(drops)), "mean": float(np.mean(drops))}


def make_serializable(obj):
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def main(args):
    """Run all CL experiments defined by the scenario/strategy matrix.

    Iterates over every combination of CL scenario, strategy, and random seed.
    For each run, constructs the Avalanche benchmark, instantiates the chosen
    strategy, trains experience-by-experience, and writes per-seed result CSVs
    to continual/results/.

    Args:
        args: Parsed argparse namespace (see parse_args()).
    """
    output_dir = os.path.join("continual", "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = logging.getLogger("train_cl")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        os.path.join(output_dir, f"cl_training_{timestamp}.log")
    )
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Arguments: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        logger.info(
            f"Device: {device} — {gpu.name} "
            f"({gpu.total_memory // 1024**2} MB VRAM)"
        )
    else:
        logger.warning("CUDA not available — training on CPU")

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    RNGManager.set_random_seeds(args.seed)
    logger.info(f"Seed: {args.seed}")

    champion_cfg_path = os.path.join("configs", "champion_timefreq_domain.json")
    with open(champion_cfg_path) as f:
        champion_hps = json.load(f)
    logger.info(f"Champion HPs: {champion_hps}")

    group_cfg_path = os.path.join("configs", "cl_models.json")
    with open(group_cfg_path) as f:
        group_model_configs = json.load(f)
    logger.info(f"Group model configs loaded from {group_cfg_path}")

    sw_cfg = {"window_size": args.window_size, "stride": args.window_stride}

    training_strategies = args.strategies
    training_scenarios = args.scenarios

    for scenario_name in training_scenarios:
        config = SCENARIOS[scenario_name]
        scenario_dir = os.path.join(
            output_dir, f"seed_{args.seed}", scenario_name
        )
        os.makedirs(scenario_dir, exist_ok=True)
        os.makedirs(os.path.join(scenario_dir, "checkpoints"), exist_ok=True)

        for idx, run in enumerate(config["runs"]):
            logger.info("=" * 60)
            logger.info(f"Scenario: {scenario_name} | Signal group: {run}")
            logger.info("=" * 60)

            if args.resume:
                _all_done = True
                for _sk in training_strategies:
                    _pg = _apply_scenario_overrides(
                        _sk, STRATEGIES[_sk], config
                    )
                    for _p in ParameterGrid(_pg):
                        _fn = _sk + "_" + "_".join(
                            f"{k}-{v}" for k, v in _p.items()
                        )
                        _rp = os.path.join(
                            scenario_dir, f"{run}_{_fn}_results.json"
                        )
                        if not os.path.exists(_rp):
                            _all_done = False
                            break
                    if not _all_done:
                        break
                if _all_done:
                    logger.info(
                        f"  All strategies for {run} already done — skipping"
                    )
                    continue

            if config["mu_filters"] is None:
                logger.info(
                    f"  {scenario_name}/{run}: mu_filters=None — skipping"
                )
                continue

            nasa_position = config.get("nasa_position", "none")

            batch_size = config["batch_size"][
                min(idx, len(config["batch_size"]) - 1)
            ]

            datasets_train = []
            datasets_val   = []
            datasets_test  = []

            def _load_nasa_experience():
                """Load and preprocess the NASA paper-split experience.

                Applies champion preprocessing (scalogram, windowing, normalisation)
                to the NASA training and test sets and returns the windowed arrays.
                """
                logger.info("Loading NASA dataset (paper split, champion preprocessing)")
                _nasa_train_cases = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14]
                _tr_ds, _, _te_ds, _ = get_nasa_data_pipeline(
                    run_name=run,
                    window_size=args.window_size,
                    stride=args.window_stride,
                    split_type="paper",
                    seed=args.seed,
                    apply_averaging=False,
                    windowing=False,
                    custom_train_cases=_nasa_train_cases,
                    custom_val_cases=[],
                )
                _xtr = _tr_ds.data.cpu().numpy()
                _ptr = _tr_ds.proc_data.cpu().numpy()
                _ytr = _tr_ds.targets.cpu().numpy().ravel()
                _xte = _te_ds.data.cpu().numpy()
                _pte = _te_ds.proc_data.cpu().numpy()
                _yte = _te_ds.targets.cpu().numpy().ravel()
                _x_sw, _ytr_sw, _x_proc_sw, _stats = apply_full_preprocessing(
                    [_xtr, _ptr], _ytr, champion_hps, split="train",
                    sliding_window_config=sw_cfg,
                )
                _xte_sw, _yte_sw, _xte_proc_sw, _ = apply_full_preprocessing(
                    [_xte, _pte], _yte, champion_hps, split="test",
                    preproc_stats=_stats,
                    sliding_window_config=sw_cfg,
                )
                
                _n = len(_ytr_sw)
                _rng = np.random.default_rng(args.seed)
                _perm = _rng.permutation(_n)
                _n_val = max(1, int(0.2 * _n))
                _va_idx, _tr_idx = _perm[:_n_val], _perm[_n_val:]
                _nasa_tr = truncate_to_batch_size(
                    NumpyTCMDataset(_x_sw[_tr_idx], _x_proc_sw[_tr_idx], _ytr_sw[_tr_idx]), batch_size
                )
                _nasa_va = NumpyTCMDataset(_x_sw[_va_idx], _x_proc_sw[_va_idx], _ytr_sw[_va_idx])
                _nasa_te = truncate_to_batch_size(
                    NumpyTCMDataset(_xte_sw, _xte_proc_sw, _yte_sw), batch_size
                )
                logger.info(f"  NASA  train={len(_nasa_tr)}, val={len(_nasa_va)}, test={len(_nasa_te)}")
                return _nasa_tr, _nasa_va, _nasa_te

            if nasa_position == "first":
                _nasa_tr, _nasa_va, _nasa_te = _load_nasa_experience()
                datasets_train.append(_nasa_tr)
                datasets_val.append(_nasa_va)
                datasets_test.append(_nasa_te)

            logger.info("Loading MU-TCM datasets (fixed run-level holdout, champion preprocessing)")
            mu_data = get_mu_tcm_scenario_data(
                signal_group=run,
                scenario_filters=config["mu_filters"],
            )

            _all_mu_tr_x    = []
            _all_mu_tr_proc = []
            _all_mu_tr_y    = []
            for _exp in mu_data:
                if _exp["train"] is not None:
                    _x, _p, _y = _exp["train"]
                    _all_mu_tr_x.extend(_x)
                    _all_mu_tr_proc.append(_p)
                    _all_mu_tr_y.append(_y)
                if _exp["val"] is not None:
                    _x, _p, _y = _exp["val"]
                    _all_mu_tr_x.extend(_x)
                    _all_mu_tr_proc.append(_p)
                    _all_mu_tr_y.append(_y)
            mu_stft_stats = None
            if _all_mu_tr_x:
                _pooled_proc = np.concatenate(_all_mu_tr_proc, axis=0)
                _pooled_y    = np.concatenate(_all_mu_tr_y,    axis=0)
                _, _, _, mu_stft_stats = apply_full_preprocessing(
                    [_all_mu_tr_x, _pooled_proc], _pooled_y,
                    champion_hps, split="train",
                    sliding_window_config=sw_cfg,
                )
                logger.info(
                    f"  Scenario STFT stats computed from {len(_all_mu_tr_x)} pooled training runs"
                )

            for j, exp in enumerate(mu_data):
                logger.info(f"  MU-TCM exp {j}: {exp['filters']}")
                logger.info(
                    f"    Fixed split: {exp['n_train'] + exp['n_val'] + exp['n_test']} runs → "
                    f"train={exp['n_train']}, val={exp['n_val']}, test={exp['n_test']}"
                )

                if exp["train"] is None:
                    logger.warning(f"    No train runs for {exp['filters']} — skipping")
                    continue

                x_tr_z, proc_tr_n, y_tr = exp["train"]
                x_te_z, proc_te_n, y_te = exp["test"]

                x_sw, ytr_sw, x_proc_sw, _ = apply_full_preprocessing(
                    [x_tr_z, proc_tr_n], y_tr, champion_hps, split="train",
                    preproc_stats=mu_stft_stats,
                    sliding_window_config=sw_cfg,
                )
                if x_sw.shape[0] == 0:
                    logger.warning(f"    No train windows for {exp['filters']} — skipping")
                    continue
                    
                xte_sw, yte_sw, xte_proc_sw, _ = apply_full_preprocessing(
                    [x_te_z, proc_te_n], y_te, champion_hps, split="test",
                    preproc_stats=mu_stft_stats,
                    sliding_window_config=sw_cfg,
                )

                mu_train = NumpyTCMDataset(x_sw, x_proc_sw, ytr_sw)
                mu_test  = NumpyTCMDataset(xte_sw, xte_proc_sw, yte_sw)

                mu_train = truncate_to_batch_size(mu_train, batch_size)
                mu_test  = truncate_to_batch_size(mu_test, batch_size)

                if exp["val"] is not None:
                    x_va_z, proc_va_n, y_va = exp["val"]
                    xva_sw, yva_sw, xva_proc_sw, _ = apply_full_preprocessing(
                        [x_va_z, proc_va_n], y_va, champion_hps, split="test",
                        preproc_stats=mu_stft_stats,
                        sliding_window_config=sw_cfg,
                    )
                    if xva_sw.shape[0] > 0:
                        mu_val = NumpyTCMDataset(xva_sw, xva_proc_sw, yva_sw)
                    else:
                        logger.warning(f"    Val run produced 0 windows; using last train window")
                        mu_val = NumpyTCMDataset(x_sw[-1:], x_proc_sw[-1:], ytr_sw[-1:])
                else:
                    logger.warning(f"    No val run for {exp['filters']}; using last train window")
                    mu_val = NumpyTCMDataset(x_sw[-1:], x_proc_sw[-1:], ytr_sw[-1:])

                datasets_train.append(mu_train)
                datasets_val.append(mu_val)
                datasets_test.append(mu_test)
                logger.info(
                    f"    train={len(mu_train)}, val={len(mu_val)}, test={len(mu_test)}"
                )

            if nasa_position == "last":
                _nasa_tr, _nasa_va, _nasa_te = _load_nasa_experience()
                datasets_train.append(_nasa_tr)
                datasets_val.append(_nasa_va)
                datasets_test.append(_nasa_te)

            logger.info("Creating Avalanche benchmark")
            avalanche_train = [AvalancheDataset(ds) for ds in datasets_train]
            avalanche_val   = [AvalancheDataset(ds) for ds in datasets_val]
            avalanche_test  = [AvalancheDataset(ds) for ds in datasets_test]
            
            benchmark = benchmark_from_datasets(
                train=avalanche_train, test=avalanche_test, valid=avalanche_val,
            )

            total_train = sum(
                len(s.dataset) for s in benchmark.train_stream
            )
            logger.info(f"Total training samples: {total_train}")
            for i in range(len(benchmark.train_stream)):
                nt = len(benchmark.train_stream[i].dataset)
                nv = len(benchmark.valid_stream[i].dataset)
                logger.info(f"  Exp {i}: train={nt}, val={nv}")
                if nt == 0:
                    raise RuntimeError(
                        f"Experience {i} has 0 training windows — cannot train. "
                        f"Check the run-level split and windowing config."
                    )
                if nv == 0:
                    raise RuntimeError(
                        f"Experience {i} has 0 validation windows — EarlyStoppingPlugin "
                        f"and forgetting metrics require at least 1. "
                        f"Increase val_fraction or add more runs to this condition."
                    )

            n_experiences = len(benchmark.train_stream)
            _min_exp_size = min(
                len(benchmark.train_stream[i].dataset)
                for i in range(n_experiences)
            )
            effective_batch_size = max(1, min(batch_size, _min_exp_size))
            if effective_batch_size < batch_size:
                logger.warning(
                    f"  Smallest post-split train experience has {_min_exp_size} samples "
                    f"(< batch_size={batch_size}); capping train_mb_size to {effective_batch_size}"
                )

            sample_dict, _ = datasets_train[0][0]
            input_shape = tuple(sample_dict["x"].shape)  # (W, C) or (F, T, C)
            proc_shape = tuple(sample_dict["proc_data"].shape)  # (4,)
            logger.info(
                f"Model input_shape={input_shape}, proc_shape={proc_shape}"
            )

            for strat_key in training_strategies:
                param_grid = _apply_scenario_overrides(
                    strat_key, STRATEGIES[strat_key], config
                )
                logger.info(f"Strategy: {strat_key}")

                for params in ParameterGrid(param_grid):
                    full_name = (
                        strat_key
                        + "_"
                        + "_".join(f"{k}-{v}" for k, v in params.items())
                    )
                    result_path = os.path.join(
                        scenario_dir, f"{run}_{full_name}_results.json"
                    )

                    if args.resume and os.path.exists(result_path):
                        logger.info(f"  Skipping {full_name} (exists)")
                        continue

                    logger.info(f"  Running {full_name}")

                    hps = {k: v for k, v in champion_hps.items()}
                    sg_model_cfg = group_model_configs.get(run, {})
                    model_name_str = sg_model_cfg.get("model_name", "CNN_LSTM_Film")

                    model = get_torch_model(
                        model_name_str,
                        input_shape,
                        hps,
                        proc_shape=proc_shape,
                    )
                    model = model.to(device)
                    n_params = sum(p.numel() for p in model.parameters())
                    logger.info(
                        f"    Model: {model_name_str}, params={n_params:,}"
                    )

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

                    scheduler_plugin = LRSchedulerPlugin(
                        scheduler, metric="val_loss"
                    )
                    es_plugin = EarlyStoppingPlugin(
                        patience=args.patience,
                        val_stream_name="valid_stream",
                        metric_name="Loss_Exp",
                        mode="min",
                        margin=0.0001,
                    )
                    plugins = [es_plugin, scheduler_plugin]

                    if params.get("replay_mem", 0) != 0:
                        plugins.append(
                            ReplayPlugin(mem_size=params["replay_mem"])
                        )

                    run_ts = datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )
                    text_log_path = os.path.join(
                        scenario_dir,
                        f"{run}_{strat_key}_{run_ts}.txt",
                    )
                    text_logger = TextLogger(open(text_log_path, "a"))
                    std_logger = InteractiveLogger()
                    loggers = [text_logger, std_logger]

                    eval_plugin = EvaluationPlugin(
                        loss_metrics(
                            epoch=True, experience=True, stream=True
                        ),
                        rmse_metrics(
                            epoch=True, experience=True, stream=True
                        ),
                        r2_metrics(
                            epoch=True, experience=True, stream=True
                        ),
                        timing_metrics(
                            epoch=True,
                            epoch_running=True,
                            experience=True,
                            stream=True,
                        ),
                        rmse_forgetting_metrics(
                            experience=True, stream=True
                        ),
                        r2_forgetting_metrics(
                            experience=True, stream=True
                        ),
                        cpu_usage_metrics(experience=True, stream=True),
                        gpu_usage_metrics(
                            0, epoch=True, experience=True, stream=True
                        ),
                        ram_usage_metrics(
                            epoch=True, experience=True, stream=True
                        ),
                        loggers=loggers,
                    )

                    common_kw = dict(
                        model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        train_mb_size=effective_batch_size,
                        train_epochs=args.epochs,
                        eval_every=1,
                        plugins=plugins,
                        evaluator=eval_plugin,
                        eval_mb_size=effective_batch_size,
                        device=device,
                    )

                    strategy = _create_strategy(
                        strat_key, params, common_kw, logger
                    )
                    if strategy is None:
                        continue

                    av_ckpt = os.path.join(
                        scenario_dir, "checkpoints", f"{run}_{full_name}.pkl"
                    )
                    meta_ckpt = os.path.join(
                        scenario_dir, "checkpoints", f"{run}_{full_name}.meta.pt"
                    )
                    if args.resume:
                        strategy, initial_exp = maybe_load_checkpoint(
                            strategy, av_ckpt, map_location=device
                        )
                        if initial_exp > 0:
                            strategy.evaluator = eval_plugin
                    else:
                        for _stale in (av_ckpt, meta_ckpt):
                            if os.path.exists(_stale):
                                os.remove(_stale)
                        initial_exp = 0

                    strat_results = []
                    b_bar = None
                    elapsed_offset = 0.0
                    _fisher_diag = {}
                    _agem_eff_ppe = []

                    if initial_exp > 0 and os.path.exists(meta_ckpt):
                        meta = torch.load(meta_ckpt, map_location="cpu",
                                          weights_only=False)
                        strat_results = meta["strat_results"]
                        b_bar = meta["b_bar"]
                        elapsed_offset = meta.get("elapsed", 0.0)
                        _fisher_diag = meta.get("fisher_diag", {})
                        _agem_eff_ppe = meta.get("agem_eff_ppe", [])
                        logger.info(
                            f"    Resumed from experience {initial_exp} "
                            f"({len(strat_results)} results loaded)"
                        )

                    start = time.time()

                    if initial_exp == 0:
                        b_bar_eval = strategy.eval(benchmark.test_stream)
                        b_bar = [
                            extract_experience_rmse(b_bar_eval).get(k)
                            for k in range(n_experiences)
                        ]

                        if hasattr(strategy, "initial_weights"):
                            strategy.initial_weights = _copy.deepcopy(
                                strategy.model.state_dict()
                            )

                    for i, experience in enumerate(
                        benchmark.train_stream[initial_exp:],
                        start=initial_exp,
                    ):
                        logger.info(
                            f"    Experience {i}/{n_experiences - 1}"
                        )
                        strategy.train(
                            experience,
                            eval_streams=[benchmark.valid_stream[i]],
                        )
                        logger.info("      Training completed")

                        if strat_key == "ewc":
                            for _pl in strategy.plugins:
                                if isinstance(_pl, EWC.EWCPlugin if hasattr(EWC, 'EWCPlugin') else type(None)):
                                    break
                            _importances = getattr(strategy, "importances", {})
                            if not _importances:
                                for _pl in strategy.plugins:
                                    _importances = getattr(_pl, "importances", {})
                                    if _importances:
                                        break
                            _fi = _importances.get(i, {})
                            if _fi:
                                _all_f = torch.cat(
                                    [v.data.flatten() for v in _fi.values()
                                     if v is not None and v.data is not None]
                                )
                                _fisher_diag[i] = {
                                    "mean": float(_all_f.mean()),
                                    "max":  float(_all_f.max()),
                                }
                                logger.info(
                                    f"      Fisher diag (exp {i}): "
                                    f"mean={_all_f.mean():.3e}, "
                                    f"max={_all_f.max():.3e}"
                                )

                        if strat_key == "agem":
                            _actual = len(experience.dataset)
                            _nom    = params.get("patterns_per_exp", _actual)
                            _eff    = min(_nom, _actual)
                            _agem_eff_ppe.append(_eff)
                            logger.info(
                                f"      AGEM effective_patterns_per_exp={_eff} "
                                f"(nominal={_nom}, exp_size={_actual})"
                            )

                        strat_results.append(
                            strategy.eval(benchmark.test_stream)
                        )

                        save_checkpoint(strategy, av_ckpt)
                        torch.save(
                            {
                                "strat_results": strat_results,
                                "b_bar": b_bar,
                                "elapsed": elapsed_offset + (time.time() - start),
                                "fisher_diag": _fisher_diag,
                                "agem_eff_ppe": _agem_eff_ppe,
                            },
                            meta_ckpt,
                        )

                    elapsed = elapsed_offset + (time.time() - start)
                    logger.info(f"    Total time: {elapsed:.1f}s")

                    ckpt_path = os.path.join(
                        scenario_dir,
                        "checkpoints",
                        f"{run}_{full_name}.pt",
                    )
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        ckpt_path,
                    )

                    for _p in (av_ckpt, meta_ckpt):
                        if os.path.exists(_p):
                            os.remove(_p)

                    fwt = compute_fwt(strat_results, n_experiences, b_bar=b_bar)
                    max_forgetting = compute_max_forgetting(
                        strat_results, n_experiences
                    )
                    model_size_mb = (
                        sum(
                            p.numel() * p.element_size()
                            for p in model.parameters()
                        )
                        / 1e6
                    )

                    _all_m = eval_plugin.get_all_metrics()

                    def _epoch_vals(key):
                        """Return list of values from an epoch-level metric (steps, values)."""
                        v = _all_m.get(key)
                        if v is None:
                            return []
                        if isinstance(v, (list, tuple)) and len(v) == 2:
                            return [float(x) for x in v[1]]
                        return []

                    wall_clock_s_per_exp = (
                        [elapsed / n_experiences] * n_experiences
                        if n_experiences > 0 else [elapsed]
                    )
                    
                    ram_vals = _epoch_vals("MaxRAMUsage_Epoch/train_phase/train_stream")
                    gpu_vals = _epoch_vals("MaxGPU0Usage_Epoch/train_phase/train_stream")
                    ram_peak_mb = float(max(ram_vals)) if ram_vals else float("nan")
                    gpu_peak_mb = float(max(gpu_vals)) if gpu_vals else float("nan")

                    results = {
                        "scenario": scenario_name,
                        "signal_group": run,
                        "strategy": strat_key,
                        "params": make_serializable(params),
                        "model_name": model_name_str,
                        "input_shape": list(input_shape),
                        "n_experiences": n_experiences,
                        "seed": args.seed,
                        "time_seconds": elapsed,
                        "model_size_mb": model_size_mb,
                        "wall_clock_s_per_exp": wall_clock_s_per_exp,
                        "ram_peak_mb": ram_peak_mb,
                        "gpu_peak_mb": gpu_peak_mb,
                        "fwt": fwt,
                        "b_bar": b_bar,
                        "max_forgetting": max_forgetting,
                        "fisher_diag": make_serializable(_fisher_diag),
                        "agem_effective_patterns_per_exp": _agem_eff_ppe,
                        "results": make_serializable(strat_results),
                        "all_metrics": make_serializable(
                            eval_plugin.get_all_metrics()
                        ),
                    }

                    with open(result_path, "w") as f:
                        json.dump(
                            results, f, indent=2, cls=NumpyFloatValuesEncoder
                        )
                    logger.info(f"    Results saved to {result_path}")

                    del strategy, model, optimizer, scheduler, criterion
                    del es_plugin, scheduler_plugin, eval_plugin
                    del text_logger, std_logger, loggers, plugins
                    torch.cuda.empty_cache()

            del benchmark, datasets_train, datasets_test
            del avalanche_train, avalanche_test
            torch.cuda.empty_cache()


def _create_strategy(strat_key, params, common_kw, logger):
    """Instantiate an Avalanche CL strategy from its name and parameters.

    Args:
        strat_key (str): Strategy identifier (e.g. ``'ewc'``, ``'agem'``,
            ``'cumulative'``, ``'naive'``).
        params (dict): Strategy-specific hyperparameters from the strategy grid.
        common_kw (dict): Keyword arguments forwarded to every strategy
            constructor (model, optimizer, criterion, etc.).
        logger: Logger instance used to emit info messages.

    Returns:
        Instantiated Avalanche CL strategy object.
    """
    if strat_key == "cumulative":
        logger.info("    Cumulative training (oracle upper bound)")
        return Cumulative(**common_kw, reset_weights=True)

    if strat_key in ("naive"):
        logger.info(
            "    Naive fine-tuning (forgetting baseline)"
        )
        return Naive(**common_kw)
    
    if strat_key == "ewc":
        decay = (
            params.get("decayfactor") if params["mode"] == "online" else None
        )
        logger.info(
            f"    EWC: lambda={params['lambda']}, "
            f"mode={params['mode']}, decay={decay!r}"
        )
        return EWC(
            **common_kw,
            ewc_lambda=params["lambda"],
            mode=params["mode"],
            decay_factor=decay,
        )

    if strat_key == "si":
        logger.info(f"    SI: lambda={params['lambda']}")
        return SynapticIntelligence(
            **common_kw, si_lambda=params["lambda"]
        )

    if strat_key == "mas":
        _alpha = params.get("alpha", 0.5)
        logger.info(
            f"    MAS: lambda_reg={params['lambda_reg']}, alpha={_alpha}"
        )
        return MASMultimodal(
            **common_kw,
            lambda_reg=params["lambda_reg"],
            alpha=_alpha,
        )

    if strat_key == "agem":
        _ppe = params["patterns_per_exp"]
        _ss = min(64, _ppe)
        logger.info(
            f"    A-GEM: patterns_per_exp={_ppe}, "
            f"sample_size={_ss}"
        )
        return AGEM(
            **common_kw,
            patterns_per_exp=_ppe,
            sample_size=_ss,
        )

    logger.warning(f"    Unknown strategy: {strat_key}, skipping")
    return None


def parse_args():
    """Parse command-line arguments for the CL training script."""
    p = argparse.ArgumentParser(
        description="DeepTCM Continual Learning Training",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Max training epochs per experience (default: 200)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (default: 20)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=500,
        help="Sliding window size in samples (default: 500)",
    )
    p.add_argument(
        "--window-stride",
        type=int,
        default=500,
        help="Sliding window stride in samples (default: 500 = no overlap)",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["scenario_1", "scenario_3", "scenario_4", "scenario_5", "scenario_6"],
        choices=list(SCENARIOS.keys()),
        help="Scenarios to run (default: scenario_1 scenario_3 scenario_4 scenario_5)",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES.keys()),
        choices=list(STRATEGIES.keys()),
        help="CL strategies to evaluate (default: all)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip completed experiments (default: True)",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-run all experiments even if results exist",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed (default: None = auto-generated). "
            "The chosen seed is saved in results so runs are reconstructible. "
            "For multi-seed evaluation, run the script multiple times without "
            "specifying --seed."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
