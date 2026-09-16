# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPAM optimizer — Spike-Aware Adam with Momentum Reset.
#
# Based on: "SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training"
# (arxiv 2501.06842, Jan 2025)
#
# Key ideas:
# 1. Spike-aware gradient clipping: scale down outlier gradients while
#    preserving direction
# 2. Periodic momentum reset: clear accumulated moments every DeltaT steps
#    to prevent stale momentum from causing instability

import math

import torch
from torch.optim import Optimizer


class SPAM(Optimizer):
    """SPAM: Spike-Aware Adam with Momentum Reset.

    Extends AdamW with two mechanisms for stable LLM training:
    - Gradient spike detection and directional clipping
    - Periodic momentum reset every delta_t steps

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Adam beta coefficients (default: (0.9, 0.999)).
        eps: Adam epsilon (default: 1e-8).
        weight_decay: Decoupled weight decay (default: 0.1).
        spike_threshold: Gradient norm ratio threshold for spike detection.
            A gradient is considered a spike if its norm exceeds
            spike_threshold * EMA of gradient norms (default: 2.0).
        delta_t: Reset momentum every delta_t steps (default: 100).
            Set to 0 to disable momentum reset.
        ema_beta: EMA coefficient for tracking gradient norm (default: 0.999).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        spike_threshold: float = 2.0,
        delta_t: int = 100,
        ema_beta: float = 0.999,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "spike_threshold": spike_threshold,
            "delta_t": delta_t,
            "ema_beta": ema_beta,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            spike_threshold = group["spike_threshold"]
            delta_t = group["delta_t"]
            ema_beta = group["ema_beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["grad_norm_ema"] = 0.0

                state["step"] += 1
                step = state["step"]

                # --- Spike-aware gradient clipping ---
                grad_norm = grad.norm().item()

                if state["grad_norm_ema"] > 0 and spike_threshold > 0:
                    ratio = grad_norm / state["grad_norm_ema"]
                    if ratio > spike_threshold:
                        # Scale gradient to threshold * EMA, preserving direction
                        clip_scale = (
                            spike_threshold * state["grad_norm_ema"] / grad_norm
                        )
                        grad = grad * clip_scale

                # Update gradient norm EMA
                state["grad_norm_ema"] = (
                    ema_beta * state["grad_norm_ema"]
                    + (1.0 - ema_beta) * grad_norm
                )

                # --- Periodic momentum reset ---
                if delta_t > 0 and step % delta_t == 0:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

                # --- AdamW update ---
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decoupled weight decay
                p.mul_(1.0 - lr * wd)

                # Adam moment updates
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                # After momentum reset, use steps-since-reset for correction
                steps_since_reset = (
                    step % delta_t if delta_t > 0 else step
                )
                if steps_since_reset == 0:
                    steps_since_reset = delta_t if delta_t > 0 else step

                bias_correction1 = 1.0 - beta1**steps_since_reset
                bias_correction2 = 1.0 - beta2**steps_since_reset
                step_size = lr / bias_correction1

                denom = (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                ).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
