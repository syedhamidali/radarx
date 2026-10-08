"""
Network: a small 3-D U-Net with a domain-mean (VAD-like) context branch.

Input ``features`` ``(N, C, Z, Y, X)`` as built by
``radarx.retrieve.single_doppler._features``; output the wind ``(N, 3, Z,
Y, X)`` in m s-1. The network predicts the departure from the background
wind (channels ``background_u``, ``background_v`` of the input), so an
untrained network returns the background. ``Z``, ``Y`` and ``X`` must be
multiples of 4 (two poolings).

At the coarsest level the features are averaged over the horizontal domain
at every height (weighted by the observation mask) and fed back to every
cell: a learned analogue of the velocity-azimuth display, which gives the
network the domain-scale wind that the radial velocities define.
"""

import torch
from torch import nn

from radarx.retrieve.single_doppler import FEATURES, WIND_SCALE

N_FEATURES = len(FEATURES)
I_MASK = FEATURES.index("radial_velocity_mask")
I_UBG = FEATURES.index("background_u")
I_VBG = FEATURES.index("background_v")


def block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
        nn.Conv3d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
    )


class SingleDopplerNet(nn.Module):
    def __init__(self, width=24, uv_scale=10.0, w_scale=5.0):
        super().__init__()
        c = width
        self.enc1 = block(N_FEATURES, c)
        self.enc2 = block(c, 2 * c)
        self.enc3 = block(2 * c, 4 * c)
        self.context = nn.Sequential(nn.Conv3d(4 * c, 4 * c, 1), nn.SiLU())
        self.mid = block(8 * c, 4 * c)
        self.up2 = nn.Conv3d(4 * c, 2 * c, 1)
        self.dec2 = block(4 * c, 2 * c)
        self.up1 = nn.Conv3d(2 * c, c, 1)
        self.dec1 = block(2 * c, c)
        self.head = nn.Conv3d(c, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.pool = nn.AvgPool3d(2)
        self.register_buffer(
            "scale", torch.tensor([uv_scale, uv_scale, w_scale]).view(1, 3, 1, 1, 1)
        )

    @staticmethod
    def _up(x):
        return nn.functional.interpolate(
            x, scale_factor=2.0, mode="trilinear", align_corners=False
        )

    def forward(self, features):
        e1 = self.enc1(features)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # observation-weighted horizontal mean at every level
        mask = self.pool(self.pool(features[:, I_MASK : I_MASK + 1]))
        num = (e3 * mask).sum(dim=(3, 4), keepdim=True)
        den = mask.sum(dim=(3, 4), keepdim=True) + 1e-3
        ctx = self.context(num / den).expand_as(e3)
        m = self.mid(torch.cat([e3, ctx], dim=1))
        d2 = self.dec2(torch.cat([self._up(self.up2(m)), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(self.up1(d2)), e1], dim=1))
        delta = self.head(d1) * self.scale
        bg = torch.cat(
            [
                features[:, I_UBG : I_UBG + 1] * WIND_SCALE,
                features[:, I_VBG : I_VBG + 1] * WIND_SCALE,
                torch.zeros_like(features[:, :1]),
            ],
            dim=1,
        )
        return bg + delta
