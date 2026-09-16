# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Mano optimizer — Manifold-Normalized Optimizer for LLM Training.
#
# Based on: "Mano: Restriking Manifold Optimization for LLM Training"
# (arxiv 2601.23000, Jan 2026)
#
# Key idea: project momentum onto tangent space of a rotating Oblique
# manifold. Cheaper than Muon (vector-based ops vs Newton-Schulz) and
# 1.75x faster wall-clock convergence.

import math

import torch
from torch.optim import Optimizer


class Mano(Optimizer):
    """Mano: Manifold-Normalized Optimizer.

    Applies Mano updates to 2D parameters (hidden layers) and falls back
    to AdamW for 1D parameters and embeddings/heads.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 3e-4).
        momentum: Momentum coefficient for Mano (default: 0.95).
        weight_decay: Decoupled weight decay (default: 0.1).
        adamw_betas: Beta values for AdamW fallback (default: (0.9, 0.95)).
        adamw_eps: Epsilon for AdamW fallback (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
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
            mu = group["momentum"]
            wd = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Decide: Mano for 2D hidden layers, AdamW for rest
                use_mano = p.ndim == 2 and max(p.shape) <= 10000

                if len(state) == 0:
                    state["step"] = 0
                    state["use_mano"] = use_mano
                    if use_mano:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    else:
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1

                if state["use_mano"]:
                    self._mano_step(p, grad, state, lr, mu, wd)
                else:
                    self._adamw_step(
                        p, grad, state, lr, beta1, beta2, eps, wd
                    )

        return loss

    @staticmethod
    def _mano_step(
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        lr: float,
        momentum: float,
        weight_decay: float,
    ):
        """Mano update: tangent projection on rotating Oblique manifold."""
        step = state["step"]
        buf = state["momentum_buffer"]

        # Accumulate momentum
        buf.mul_(momentum).add_(grad)

        # Rotating manifold: alternate between row-norm (k=0) and col-norm (k=1)
        k = step % 2

        # Normalize parameters along the chosen dimension
        if k == 0:
            # Row-wise normalization
            p_norm = p.norm(dim=1, keepdim=True).clamp(min=1e-8)
            p_hat = p / p_norm
            # Tangent projection: remove component along p_hat
            dot = (buf * p_hat).sum(dim=1, keepdim=True)
            v = buf - p_hat * dot
            # Normalize update
            v_norm = v.norm(dim=1, keepdim=True).clamp(min=1e-8)
            v_hat = v / v_norm
            n_k = p.shape[1]
        else:
            # Column-wise normalization
            p_norm = p.norm(dim=0, keepdim=True).clamp(min=1e-8)
            p_hat = p / p_norm
            dot = (buf * p_hat).sum(dim=0, keepdim=True)
            v = buf - p_hat * dot
            v_norm = v.norm(dim=0, keepdim=True).clamp(min=1e-8)
            v_hat = v / v_norm
            n_k = p.shape[0]

        # Scale factor: 0.2 * sqrt(n_k)
        scale = 0.2 * math.sqrt(n_k)

        # Update: Euclidean descent with weight decay
        p.add_(scale * v_hat + weight_decay * p, alpha=-lr)

    @staticmethod
    def _adamw_step(
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ):
        """Standard AdamW for embeddings/head/1D params."""
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        # Decoupled weight decay
        p.mul_(1.0 - lr * weight_decay)

        # Adam update
        exp_avg.lerp_(grad, 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr / bias_correction1

        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-step_size)
