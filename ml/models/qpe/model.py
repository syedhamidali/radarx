#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Network for the rain rate of ``radarx.retrieve.rain_rate_ml``.

A multilayer perceptron on the physical values of the features of the
registry (``dbz``, ``zdr``, ``kdp``, ``rhohv``, ``height``, ``temperature``,
``wavelength``). Each feature is transformed (the specific differential phase
by the inverse hyperbolic sine, which is linear near 0 and logarithmic for
large values and keeps the sign; the correlation coefficient by -log10(1 -
rho), which spreads the values close to 1; the wavelength by its logarithm)
and standardized with the mean and the standard deviation of the training
data. The transformation and the statistics are part of the graph, so the
exported ONNX model takes the values that radarx passes.

The network predicts the mean ``mu`` and the standard deviation ``sigma`` of
the natural logarithm of the rain rate (mm/h); the rain rate is the mean of
the log-normal distribution, ``exp(mu + sigma**2 / 2)``, which keeps the sum
of the rain (the accumulation) unbiased, and the bounds are the 16th and 84th
percentile ``exp(mu -+ sigma)``. Training minimizes the
Gaussian negative log-likelihood in the logarithm (Nix and Weigend 1994),
which makes ``sigma`` the predicted uncertainty and removes the dominance of
heavy rain in the squared error.
"""

from __future__ import annotations

import math

import torch
from torch import nn

FEATURES = ("dbz", "zdr", "kdp", "rhohv", "height", "temperature", "wavelength")
LN10 = math.log(10.0)


def transform(name, x):
    """Transformation of one feature (before standardization)."""
    if name == "kdp":  # asinh(x / 0.2), written out for the ONNX exporter
        u = x / 0.2
        return torch.sign(u) * torch.log(torch.abs(u) + torch.sqrt(u * u + 1.0))
    if name == "rhohv":
        return -torch.log(1.0005 - torch.clamp(x, 0.5, 1.0)) / LN10
    if name == "wavelength":
        return torch.log(x)
    return x


class QPENet(nn.Module):
    """Features (N, F) -> mean and log standard deviation of ln(rain rate)."""

    def __init__(self, features, hidden=(64, 64, 64)):
        super().__init__()
        unknown = [f for f in features if f not in FEATURES]
        if unknown:
            raise ValueError(f"unknown features {unknown}; use some of {FEATURES}")
        self.features = tuple(features)
        n = len(self.features)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))
        layers = []
        for h in hidden:
            layers += [nn.Linear(n, h), nn.SiLU()]
            n = h
        layers.append(nn.Linear(n, 2))
        self.mlp = nn.Sequential(*layers)

    def transformed(self, x):
        return torch.stack(
            [transform(f, x[:, i]) for i, f in enumerate(self.features)], dim=1
        )

    def set_scaling(self, x):
        """Mean and standard deviation of the transformed training features."""
        t = self.transformed(x)
        self.mean.copy_(t.mean(dim=0))
        self.std.copy_(t.std(dim=0).clamp_min(1e-6))

    def forward(self, x):
        out = self.mlp((self.transformed(x) - self.mean) / self.std)
        return out[:, 0], torch.clamp(out[:, 1], -5.0, 1.5)


class Exported(nn.Module):
    """The graph of the registry: mean rain rate and one-sigma bounds."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, features):
        mu, log_sigma = self.net(features)
        sigma = torch.exp(log_sigma)
        mean = torch.exp(mu + 0.5 * sigma * sigma)
        return mean, torch.exp(mu - sigma), torch.exp(mu + sigma)


def nll(mu, log_sigma, target_ln):
    """Mean Gaussian negative log-likelihood of ln(rain rate)."""
    return (log_sigma + 0.5 * ((target_ln - mu) / torch.exp(log_sigma)) ** 2).mean()
