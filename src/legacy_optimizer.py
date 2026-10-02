"""Adam update used by legacy Keras, with epsilon before bias correction.

Reference: keras-team/keras tag 2.2.4, keras/optimizers.py Adam.get_updates.
The authors' exact Keras version is unknown; this documented convention is
different from torch.optim.Adam, even with the same nominal learning rate.
"""
from __future__ import annotations

import math

import torch


class LegacyAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-7):
        if lr <= 0 or eps <= 0 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("Invalid legacy Adam settings")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            first, second = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError("Legacy Adam requires dense gradients")
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state["step"] += 1
                average, square = state["exp_avg"], state["exp_avg_sq"]
                average.mul_(first).add_(gradient, alpha=1 - first)
                square.mul_(second).addcmul_(gradient, gradient, value=1 - second)
                adjusted_rate = group["lr"] * math.sqrt(1 - second ** state["step"]) / (1 - first ** state["step"])
                denominator = square.sqrt().add_(group["eps"])
                parameter.addcdiv_(average, denominator, value=-adjusted_rate)
        return loss
