################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Cumulative training strategy for Avalanche.

Retrains on the union of all experience data seen so far at each step.
Serves as the oracle upper bound for CL experiments (no forgetting by design).
"""
from typing import Callable, Optional, List, Union
import torch

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType

import copy

class Cumulative(SupervisedTemplate):
    """Cumulative training strategy.

    At each experience, train model with data from all previous experiences
        and current experience.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        reset_weights: bool=False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

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
            **kwargs
        )

        self.dataset = None  # cumulative dataset
        self.reset_weights = reset_weights
        self.initial_weights = copy.deepcopy(model.state_dict())

    def train_dataset_adaptation(self, **kwargs):
        """
        Concatenates all the previous experiences.
        """
        exp = self.experience
        assert exp is not None
        if self.dataset is None:
            self.dataset = exp.dataset
        else:
            self.dataset = concat_datasets([self.dataset, exp.dataset])
        self.adapted_dataset = self.dataset

    def _before_training_exp(self, **kwargs):
        """Prepend current experience data to the cumulative dataset, then optionally reset model weights."""
        super()._before_training_exp(**kwargs)

        # Reset model weights
        if self.reset_weights:
            self.model.load_state_dict(self.initial_weights)


__all__ = ["Cumulative"]
