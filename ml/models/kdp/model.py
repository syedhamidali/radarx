#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Network and physics-constrained loss for KDP and backscatter phase.

The network is a one-dimensional U-Net along range (see ``KDPNet``), fully
convolutional, so it runs on rays of any length. It returns three channels per gate: K_DP (deg/km), the
backscatter phase delta (deg) and the standard deviation of K_DP.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

N_FEATURES = 7  # radarx.retrieve.kdp.ML_FEATURES
KDP_SCALE = 10.0  # the dphi feature is 0.1 x half the phase derivative
DELTA_SCALE = 10.0


class Res(nn.Module):
    """Residual block: two dilated convolutions with batch normalisation."""

    def __init__(self, width, dilation=1, kernel=5):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(width, width, kernel, dilation=dilation, padding=pad),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel, dilation=dilation, padding=pad),
            nn.BatchNorm1d(width),
        )

    def forward(self, x):
        return F.gelu(x + self.body(x))


def down(cin, cout):
    return nn.Sequential(
        nn.Conv1d(cin, cout, 5, stride=2, padding=2),
        nn.BatchNorm1d(cout),
        nn.GELU(),
        Res(cout, 2),
    )


class KDPNet(nn.Module):
    """
    One-dimensional U-Net along range: four halvings of the gate resolution
    (to 1/16) with skip connections, dilated residual blocks at the coarsest
    level (receptive field of about +-500 gates), batch normalisation folded
    into the convolutions at export. Rays of any length.
    """

    def __init__(self, n_in=N_FEATURES, widths=(16, 32, 48, 64, 64)):
        super().__init__()
        w0, w1, w2, w3, w4 = widths
        self.stem = nn.Sequential(nn.Conv1d(n_in, w0, 1), Res(w0, 1), Res(w0, 2))
        self.enc1 = down(w0, w1)
        self.enc2 = down(w1, w2)
        self.enc3 = down(w2, w3)
        self.enc4 = nn.Sequential(down(w3, w4), *[Res(w4, d) for d in (1, 2, 4, 8)])
        self.dec3 = self._up(w4 + w3, w3)
        self.dec2 = self._up(w3 + w2, w2)
        self.dec1 = self._up(w2 + w1, w1)
        self.dec0 = self._up(w1 + w0, w0)
        self.head = nn.Conv1d(w0, 3, 1)

    @staticmethod
    def _up(cin, cout):
        return nn.Sequential(
            nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout), nn.GELU(), Res(cout, 1)
        )

    @staticmethod
    def _merge(x, skip):
        up = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        return torch.cat([up, skip], 1)

    def forward(self, features):
        e0 = self.stem(features)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        h = self.enc4(e3)
        h = self.dec3(self._merge(h, e3))
        h = self.dec2(self._merge(h, e2))
        h = self.dec1(self._merge(h, e1))
        h = self.dec0(self._merge(h, e0))
        out = self.head(h)
        kdp = KDP_SCALE * out[:, 0]
        delta = DELTA_SCALE * out[:, 1]
        std = F.softplus(out[:, 2]) + 0.02
        return kdp, delta, std


class Exported(nn.Module):
    """ONNX graph: features -> (kdp, delta, kdp_std)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, features):
        return self.net(features)


def integrate(kdp, dr):
    """Twice the trapezoidal range integral of KDP, starting at zero."""
    steps = dr[:, None] * (kdp[:, :-1] + kdp[:, 1:])
    return torch.cat([torch.zeros_like(kdp[:, :1]), torch.cumsum(steps, 1)], 1)


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def physics_terms(kdp, delta, psi, features):
    """
    Self-supervised terms, usable without truth:

    - consistency: Psi_DP = 2 * integral(K_DP) + delta + constant at valid
      gates (the constant is the least-squares fit per ray), in units of a
      3 degree phase noise, Huber;
    - non-negativity of K_DP in rain (rho_hv >= 0.97 and Z_H >= 30 dBZ);
    - roughness of K_DP (squared second difference).
    """
    valid = features[:, 1]
    dr = features[:, 6, 0]
    res = psi - delta - integrate(kdp, dr)
    n = valid.sum(1).clamp(min=1.0)
    c = (res * valid).sum(1) / n
    cons = masked_mean(
        F.huber_loss(
            (res - c[:, None]) / 3.0, torch.zeros_like(res), reduction="none", delta=2.0
        ),
        valid,
    )
    rain = (
        valid
        * (features[:, 2] >= 0.97)
        * (features[:, 4] * 50.0 >= 30.0)
        * features[:, 3]
        * features[:, 5]
    )
    neg = masked_mean(F.relu(-kdp) ** 2, rain)
    d2 = kdp[:, 2:] - 2 * kdp[:, 1:-1] + kdp[:, :-2]
    rough = masked_mean(d2**2, valid[:, 1:-1])
    return cons, neg, rough


def supervised_terms(kdp, delta, std, kdp_true, delta_true, valid):
    """Huber errors of K_DP and delta, and the Gaussian NLL of the std."""
    lk = masked_mean(F.huber_loss(kdp, kdp_true, reduction="none", delta=1.0), valid)
    ld = masked_mean(
        F.huber_loss(delta / 2.0, delta_true / 2.0, reduction="none", delta=1.0), valid
    )
    z = (kdp.detach() - kdp_true) / std
    nll = masked_mean(0.5 * z**2 + torch.log(std), valid)
    return lk, ld, nll
