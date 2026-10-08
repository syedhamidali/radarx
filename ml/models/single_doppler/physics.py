"""
The variational cost of ``radarx.retrieve.multi_doppler`` as a PyTorch loss.

:func:`variational_cost` re-implements, for a batch of single-radar samples,
the terms of the radarx cost function without the vorticity term,

    J = J_o + J_m + J_s + J_b,

with exactly the discrete operators of ``radarx/retrieve/_multidoppler.cpp``
(centred first differences, one-sided at the edges; second differences in
grid units, zero at the edges). PyTorch's autograd then provides the adjoint.
``verify_physics.py`` checks the value and the gradient against the compiled
kernel ``radarx.retrieve._multidoppler.cost_gradient``.

:class:`KernelCost` wraps the compiled kernel itself as an autograd function
(the cost from the kernel, the gradient from its hand-written adjoint) for
one sample at a time on the CPU, as an exact alternative.
"""

import numpy as np
import torch

from radarx.retrieve import multidoppler as md

WEIGHTS = {
    "observation": 1.0,
    "mass_continuity": 10.0,
    "smoothness": 0.5,
    "background": 0.001,
}


def d1(f, dim, d):
    """Centred first difference along ``dim``, one-sided at the edges."""
    n = f.shape[dim]
    inner = (f.narrow(dim, 2, n - 2) - f.narrow(dim, 0, n - 2)) / (2.0 * d)
    first = (f.narrow(dim, 1, 1) - f.narrow(dim, 0, 1)) / d
    last = (f.narrow(dim, n - 1, 1) - f.narrow(dim, n - 2, 1)) / d
    return torch.cat([first, inner, last], dim=dim)


def d2(f, dim):
    """Second difference in grid units along ``dim``, zero at the edges."""
    n = f.shape[dim]
    inner = (
        f.narrow(dim, 0, n - 2)
        - 2.0 * f.narrow(dim, 1, n - 2)
        + f.narrow(dim, 2, n - 2)
    )
    zero = torch.zeros_like(f.narrow(dim, 0, 1))
    return torch.cat([zero, inner, zero], dim=dim)


def variational_cost(
    wind,
    coef,
    target,
    obs_weight,
    rho,
    bg,
    bg_weight,
    dx,
    dy,
    dz,
    weights=WEIGHTS,
    reduce="sum",
):
    """
    Terms of the radarx variational cost for a batch.

    Parameters
    ----------
    wind : torch.Tensor
        ``(B, 3, Z, Y, X)`` wind ``u, v, w`` (m s-1).
    coef : torch.Tensor
        ``(B, 3, Z, Y, X)`` beam direction cosines of the radar.
    target : torch.Tensor
        ``(B, Z, Y, X)`` radial velocity plus ``coef_z`` times the fall speed.
    obs_weight : torch.Tensor
        ``(B, Z, Y, X)`` 1 where observed, 0 elsewhere (times C_o).
    rho : torch.Tensor
        ``(B, Z, Y, X)`` air density.
    bg, bg_weight : torch.Tensor
        ``(B, 3, Z, Y, X)`` background and its weight (C_b where defined).
    dx, dy, dz : float
        Grid spacing (m).
    reduce : {"sum", "mean"}
        Sum over the grid (as radarx) or mean per grid cell.

    Returns
    -------
    dict of torch.Tensor
        ``observation``, ``mass_continuity``, ``smoothness``, ``background``,
        each of shape ``(B,)``.
    """
    u, v, w = wind[:, 0], wind[:, 1], wind[:, 2]
    dims = (1, 2, 3)
    r = coef[:, 0] * u + coef[:, 1] * v + coef[:, 2] * w - target
    jo = (obs_weight * r * r).sum(dims)
    rb = wind - bg
    jb = (bg_weight * rb * rb).sum((1, 2, 3, 4))
    cs = weights["smoothness"]
    js = 0.0
    for q in range(3):
        f = wind[:, q]
        for dim in (3, 2, 1):  # x, y, z
            s = d2(f, dim)
            js = js + cs * (s * s).sum(dims)
    h = float(np.sqrt(abs(dx * dy)))
    m = h / rho * (d1(rho * u, 3, dx) + d1(rho * v, 2, dy) + d1(rho * w, 1, dz))
    jm = weights["mass_continuity"] * (m * m).sum(dims)
    terms = {
        "observation": jo,
        "mass_continuity": jm,
        "smoothness": js,
        "background": jb,
    }
    if reduce == "mean":
        n = float(np.prod(u.shape[1:]))
        terms = {k: t / n for k, t in terms.items()}
    return terms


def kernel_arguments(
    coef, target, obs_weight, rho, bg, bg_weight, dx, dy, dz, weights=WEIGHTS
):
    """Arguments of ``_multidoppler.cost_gradient`` for one sample (NumPy)."""
    shape = rho.shape
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
    cs = weights["smoothness"]
    return dict(
        coef=f64(coef.reshape((3,) + shape)),
        target=f64(target[None]),
        weight=f64(obs_weight[None]),
        rho=f64(rho),
        bg=f64(bg),
        bg_weight=f64(bg_weight),
        vort_weight=np.zeros(shape),
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        cm=float(weights["mass_continuity"]),
        csx=cs,
        csy=cs,
        csz=cs,
        cv=0.0,
        ut=0.0,
        vt=0.0,
        coriolis=0.0,
        h=float(np.sqrt(abs(dx * dy))),
        vort_scale=0.0,
    )


class KernelCost(torch.autograd.Function):
    """Total cost of one sample from the compiled radarx kernel, with its adjoint."""

    @staticmethod
    def forward(ctx, wind, problem):
        state = wind.detach().cpu().double().numpy()
        use_compiled = md.HAS_COMPILED_KERNEL
        terms, grad = md._cost_gradient(problem, state, use_compiled, 0)
        ctx.save_for_backward(
            torch.as_tensor(grad, dtype=wind.dtype, device=wind.device)
        )
        return torch.as_tensor(
            float(np.sum(terms[:4])), dtype=wind.dtype, device=wind.device
        )

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return g * grad, None
