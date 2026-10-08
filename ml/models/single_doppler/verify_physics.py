"""
Check the PyTorch physics loss against the compiled radarx kernel.

Compares, on a synthetic single-radar sample and a random wind, every cost
term and the gradient (PyTorch autograd vs the kernel's adjoint), and checks
that :class:`physics.KernelCost` gives the same gradient.
"""

import numpy as np
import synthetic
import torch
from physics import WEIGHTS, KernelCost, kernel_arguments, variational_cost

from radarx.retrieve import multidoppler as md


def main(seed=0):
    rng = np.random.default_rng(seed)
    s = synthetic.sample(rng, ny=40, nx=48)
    nz, ny, nx = s["rho"].shape
    obs = np.isfinite(s["vr"]).astype(float) * WEIGHTS["observation"]
    target = np.where(
        obs > 0, np.nan_to_num(s["vr"]) + s["coef"][2] * s["fall_speed"], 0.0
    )
    coef = s["coef"] * (obs > 0)
    bg = np.stack([s["u_bg"], s["v_bg"], np.zeros_like(s["u_bg"])])
    bgw = np.stack(
        [np.full(bg[0].shape, WEIGHTS["background"])] * 2 + [np.zeros_like(bg[0])]
    )
    wind = s["truth"] + rng.normal(0, 2.0, s["truth"].shape)
    p = kernel_arguments(coef, target, obs, s["rho"], bg, bgw, 1000.0, 1000.0, 500.0)
    terms_k, grad_k = md._cost_gradient(p, wind, md.HAS_COMPILED_KERNEL, 0)

    t = lambda a: torch.tensor(a[None], dtype=torch.float64)  # noqa: E731
    wt = t(wind).requires_grad_(True)
    terms_t = variational_cost(
        wt,
        t(coef),
        t(target),
        t(obs),
        t(s["rho"]),
        t(bg),
        t(bgw),
        1000.0,
        1000.0,
        500.0,
    )
    total = sum(v.sum() for v in terms_t.values())
    total.backward()
    grad_t = wt.grad[0].numpy()
    print("compiled kernel:", md.HAS_COMPILED_KERNEL)
    names = ["observation", "mass_continuity", "smoothness", "background"]
    for i, name in enumerate(names):
        a, b = float(terms_t[name]), float(terms_k[i])
        print(
            f"{name:16s} torch {a:14.6f} kernel {b:14.6f} rel diff {abs(a - b) / max(abs(b), 1e-30):.1e}"
        )
    rel = np.abs(grad_t - grad_k).max() / np.abs(grad_k).max()
    print(f"gradient: max |torch - kernel| / max |kernel| = {rel:.1e}")

    wk = torch.tensor(wind, dtype=torch.float64, requires_grad=True)
    KernelCost.apply(wk, p).backward()
    rel2 = np.abs(wk.grad.numpy() - grad_t).max() / np.abs(grad_t).max()
    print(f"KernelCost autograd vs torch gradient: {rel2:.1e}")
    assert rel < 1e-10 and rel2 < 1e-10
    for i, name in enumerate(names):
        assert abs(float(terms_t[name]) - terms_k[i]) <= 1e-10 * max(
            abs(terms_k[i]), 1.0
        )
    print("OK")


if __name__ == "__main__":
    main()
