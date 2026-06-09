################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Custom MAS strategy with multimodal (dict) input support.

Avalanche's built-in MASPlugin calls `x.to(device)` which fails when the
dataloader yields `x` as a dict (our {x: tensor, proc_data: tensor} format).
This module overrides only `_get_importance` to handle that case.
"""

import torch
from torch.utils.data import DataLoader

from typing import Callable, Optional, List, Union

from tqdm import tqdm

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.mas import MASPlugin as _MASPlugin
from avalanche.training.utils import zerolike_params_dict

from torch.nn import Module
from torch.optim import Optimizer
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import EvaluationPlugin


class MASPluginMultimodal(_MASPlugin):
    """MASPlugin that handles dict minibatches (multimodal inputs)."""

    def _get_importance(self, strategy):
        """Compute weight importance as the mean L2-norm of the output gradient.

        Iterates over the current experience's dataset, performs forward and
        backward passes with no labels, and accumulates the absolute gradient
        magnitude for each parameter as its importance score.

        Args:
            strategy: The active Avalanche strategy instance.

        Returns:
            dict: Mapping from parameter name to importance tensor.
        """
        # Initialize importance matrix
        importance = dict(zerolike_params_dict(strategy.model))

        if not strategy.experience:
            raise ValueError("Current experience is not available")

        if strategy.experience.dataset is None:
            raise ValueError("Current dataset is not available")

        # Do forward and backward pass to accumulate L2-loss gradients
        strategy.model.train()
        collate_fn = (
            strategy.experience.dataset.collate_fn
            if hasattr(strategy.experience.dataset, "collate_fn")
            else None
        )
        dataloader = DataLoader(
            strategy.experience.dataset,
            batch_size=strategy.train_mb_size,
            collate_fn=collate_fn,
        )  # type: ignore

        # Progress bar
        if self.verbose:
            print("Computing importance")
            dataloader = tqdm(dataloader)

        for _, batch in enumerate(dataloader):
            # Get batch
            if len(batch) == 2 or len(batch) == 3:
                x, _, t = batch[0], batch[1], batch[-1]
            else:
                raise ValueError("Batch size is not valid")

            # --- dict-aware device transfer (multimodal extension) ---
            if isinstance(x, dict):
                x = {k: v.to(strategy.device) for k, v in x.items()}
            else:
                x = x.to(strategy.device)

            # Forward pass
            strategy.optimizer.zero_grad()
            out = avalanche_forward(strategy.model, x, t)

            # Average L2-Norm of the output
            loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
            loss.backward()

            # Accumulate importance
            for name, param in strategy.model.named_parameters():
                if param.requires_grad:
                    # In multi-head architectures, the gradient is going
                    # to be None for all the heads different from the
                    # current one.
                    if param.grad is not None:
                        importance[name].data += param.grad.abs()

        # Normalize importance
        for k in importance.keys():
            importance[k].data /= float(len(dataloader))

        return importance


class MASMultimodal(SupervisedTemplate):
    """MAS strategy built on SupervisedTemplate base + multimodal-aware MASPlugin.

    Inherits from SupervisedTemplate (not from Avalanche's MAS) to avoid plugin conflict.
    The MASPluginMultimodal is appended after super().__init__() so it is
    the only MAS plugin in the list.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        lambda_reg: float = 1.0,
        alpha: float = 0.5,
        verbose: bool = False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
        ):
        """Initialise MASMultimodal with the given model, optimiser, and MAS hyperparameters.

        Args:
            model:          PyTorch model to train.
            optimizer:      Gradient-descent optimiser.
            criterion:      Loss function.
            lambda_reg:     MAS regularisation strength.
            alpha:          Exponential moving-average coefficient for importance update.
            verbose:        Print importance update progress if True.
            train_mb_size:  Minibatch size during training.
            train_epochs:   Epochs per experience.
            eval_mb_size:   Minibatch size during evaluation.
            device:         Device string or torch.device.
            plugins:        Additional Avalanche plugins.
            evaluator:      Avalanche EvaluationPlugin instance or factory.
            eval_every:     Evaluation frequency in steps (-1 = end of experience).
            **base_kwargs:  Extra kwargs forwarded to SupervisedTemplate.
        """
        # Instantiate plugin
        mas = MASPluginMultimodal(lambda_reg=lambda_reg, alpha=alpha, verbose=verbose)

        # Add plugin to the strategy
        if plugins is None:
            plugins = [mas]
        else:
            plugins.append(mas)

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

