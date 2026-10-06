"""
Synthetic single-Doppler samples with an exact wind truth.

The wind is a background profile with shear plus a perturbation that
satisfies the anelastic continuity equation :math:`\\nabla\\cdot(\\rho
\\mathbf{v}) = 0`:

- a random vector potential :math:`\\mathbf{A}` (Gaussian random field with a
  random correlation length), :math:`\\rho \\mathbf{v} = \\nabla \\times
  \\mathbf{A}`, tapered so that :math:`w = 0` at the lowest level and the top;
- a Beltrami flow (Shapiro 1993) of random wavelength, orientation and phase,
  divided by the density.

Reflectivity is a random precipitation field enhanced in updrafts; the fall
speed follows the relations of :func:`radarx.retrieve.fall_speed`. A virtual
radar at a random position samples the radial velocity (with noise) where
there is echo, inside its elevation coverage. The background given to the
network is the true background profile plus a random, vertically correlated
error, as for a sounding or ERA5.
"""

import numpy as np

from radarx.retrieve.multidoppler import _beam_angles

DX = 1000.0
DZ = 500.0
Z = np.arange(500.0, 12e3 + 1, DZ)


def _random_field(rng, shape, length, d):
    """Gaussian random field (unit variance) with correlation ``length``."""
    noise = rng.standard_normal(shape)
    k = [np.fft.fftfreq(n, d=dd) for n, dd in zip(shape, d)]
    kk = np.meshgrid(*k, indexing="ij")
    k2 = sum((ki * li) ** 2 for ki, li in zip(kk, length))
    spec = np.fft.fftn(noise) * np.exp(-2.0 * np.pi**2 * k2)
    f = np.real(np.fft.ifftn(spec))
    return (f - f.mean()) / (f.std() + 1e-12)


def _profile(rng, z, sigma, length):
    """Random smooth vertical profile with standard deviation ``sigma``."""
    n = len(z)
    f = _random_field(rng, (n * 3,), (length,), (z[1] - z[0],))[n : 2 * n]
    return sigma * f


def background_profile(rng, z):
    """Random background wind profile (u, v) with directional shear."""
    s0 = rng.uniform(0.0, 12.0)
    s1 = rng.uniform(5.0, 45.0)
    d0 = rng.uniform(0, 2 * np.pi)
    d1 = d0 + rng.normal(0.6, 0.6)
    t = (z - z[0]) / (z[-1] - z[0])
    shape = t ** rng.uniform(0.5, 1.5)
    speed = s0 + (s1 - s0) * shape
    direction = d0 + (d1 - d0) * shape
    u = speed * np.sin(direction) + _profile(rng, z, 2.0, 1500.0)
    v = speed * np.cos(direction) + _profile(rng, z, 2.0, 1500.0)
    return u, v


def beltrami(X, Y, Z_, z0, ztop, rng, wmax):
    """Beltrami flow (Shapiro 1993) as a mass flux divided by a reference density."""
    lx = rng.uniform(15e3, 60e3)
    k = l_ = 2 * np.pi / lx
    m = np.pi / (ztop - z0)
    lam = np.sqrt(k * k + l_ * l_ + m * m)
    kh2 = k * k + l_ * l_
    a = rng.uniform(0, 2 * np.pi)
    xr_ = np.cos(a) * X - np.sin(a) * Y + rng.uniform(0, lx)
    yr_ = np.sin(a) * X + np.cos(a) * Y + rng.uniform(0, lx)
    zz = Z_ - z0
    fu = (
        -wmax
        / kh2
        * (
            lam * l_ * np.cos(k * xr_) * np.sin(l_ * yr_) * np.sin(m * zz)
            + m * k * np.sin(k * xr_) * np.cos(l_ * yr_) * np.cos(m * zz)
        )
    )
    fv = (
        wmax
        / kh2
        * (
            lam * k * np.sin(k * xr_) * np.cos(l_ * yr_) * np.sin(m * zz)
            - m * l_ * np.cos(k * xr_) * np.sin(l_ * yr_) * np.cos(m * zz)
        )
    )
    fw = wmax * np.cos(k * xr_) * np.cos(l_ * yr_) * np.sin(m * zz)
    # rotate the horizontal components back to the grid axes
    fu, fv = np.cos(a) * fu + np.sin(a) * fv, -np.sin(a) * fu + np.cos(a) * fv
    return fu, fv, fw


def potential_flow(rng, shape, z, wmax):
    """Anelastic mass flux from a random vector potential, tapered in z."""
    nz, ny, nx = shape
    # horizontal scale of the cells; a vertical scale of the same order keeps
    # the horizontal perturbation comparable to w, as in convective cells
    length = rng.uniform(4e3, 15e3)
    lz = float(np.clip(length * rng.uniform(0.6, 1.2), 2500.0, 8000.0))
    # pad in z so the field is not periodic there, then taper
    a = [
        _random_field(rng, (2 * nz, ny, nx), (lz, length, length), (DZ, DX, DX))[:nz]
        for _ in range(3)
    ]
    t = (z - z[0]) / (z[-1] - z[0] + DZ)
    taper = np.sin(np.pi * t)[:, None, None]
    ax, ay, az = a[0] * taper, a[1] * taper, a[2]
    d = (DZ, DX, DX)
    # rho v = curl A (centred differences)
    fu = np.gradient(az, d[1], axis=1) - np.gradient(ay, d[0], axis=0)
    fv = np.gradient(ax, d[0], axis=0) - np.gradient(az, d[2], axis=2)
    fw = np.gradient(ay, d[2], axis=2) - np.gradient(ax, d[1], axis=1)
    s = wmax / (np.abs(fw).max() + 1e-12)
    return fu * s, fv * s, fw * s


def fall_speed(dbz, rho, freezing_level, z):
    """Fall speed as in radarx.retrieve.fall_speed (rain / snow, density)."""
    zl = 10.0 ** (np.minimum(dbz, 70.0) / 10.0)
    vt = np.where(
        z[:, None, None] > freezing_level, 0.817 * zl**0.063, 2.65 * zl**0.114
    )
    return vt * (1.2 / rho) ** 0.4


def sample(rng, ny=96, nx=96, z=Z, noise=1.0):
    """
    One synthetic single-Doppler sample.

    Returns a dict of NumPy arrays: ``vr`` and ``dbz`` (NaN where missing),
    ``coef`` (3, z, y, x), ``u_bg``, ``v_bg`` (the erroneous background given
    to the network), ``rho``, ``fall_speed``, ``truth`` (3, z, y, x),
    ``weight`` (supervision weight), ``distance`` (y, x), and the grid.
    """
    nz = len(z)
    x = (np.arange(nx) - (nx - 1) / 2) * DX
    y = (np.arange(ny) - (ny - 1) / 2) * DX
    Z_, Y_, X_ = np.meshgrid(z, y, x, indexing="ij")
    H = rng.uniform(8500.0, 10500.0)
    rho = 1.2 * np.exp(-Z_ / H)
    rho_ref = 1.2 * np.exp(-z.mean() / H)

    ub, vb = background_profile(rng, z)
    wmax = rng.uniform(1.0, 20.0)
    share = rng.uniform(0.0, 1.0)
    fu, fv, fw = potential_flow(rng, (nz, ny, nx), z, wmax * share + 0.5)
    if rng.uniform() < 0.6:
        ztop = z[-1] + rng.uniform(0.0, 4000.0)
        bu, bv, bw = beltrami(X_, Y_, Z_, z[0], ztop, rng, wmax * (1 - share) + 0.5)
        fu, fv, fw = fu + bu, fv + bv, fw + bw
    u = ub[:, None, None] + fu * rho_ref / rho
    v = vb[:, None, None] + fv * rho_ref / rho
    w = fw * rho_ref / rho

    # reflectivity: random precipitation area, stronger in updrafts
    base = _random_field(rng, (ny, nx), (rng.uniform(8e3, 30e3),) * 2, (DX, DX))
    cover = rng.uniform(0.25, 0.9)
    thr = np.quantile(base, 1 - cover)
    top = rng.uniform(6000.0, 13000.0)
    freezing = rng.uniform(2500.0, 4800.0)
    dbz = 15.0 + 12.0 * (base - thr)[None] + 3.0 * np.clip(w, 0, None)
    dbz = dbz - 4.0 * np.clip((Z_ - freezing) / 1000.0, 0, None)
    dbz += 3.0 * rng.standard_normal(dbz.shape)
    echo = (base > thr)[None] & (Z_ < top) & (dbz > 0)
    clear = (Z_ < rng.uniform(1000.0, 2500.0)) & (rng.uniform() < 0.5)
    dbz = np.where(echo, np.clip(dbz, 0.0, 65.0), np.nan)
    vt = np.where(echo, fall_speed(np.nan_to_num(dbz), rho, freezing, z), 0.0)

    # virtual radar
    rng_r = rng.uniform(10e3, 160e3)
    ang = rng.uniform(0, 2 * np.pi)
    rx_, ry_ = rng_r * np.sin(ang), rng_r * np.cos(ang)
    alt = rng.uniform(0.0, 500.0)
    az, el = _beam_angles(X_ - rx_, Y_ - ry_, Z_, np.full(Z_.shape, alt))
    a_, e_ = np.radians(az), np.radians(el)
    coef = np.stack([np.cos(e_) * np.sin(a_), np.cos(e_) * np.cos(a_), np.sin(e_)])
    dist = np.hypot(X_[0] - rx_, Y_[0] - ry_)
    covered = (el >= 0.0) & (el <= 20.0) & (dist[None] > 2e3) & (dist[None] < 230e3)
    seen = covered & (echo | (clear & (dist[None] < 60e3)))
    vr = coef[0] * u + coef[1] * v + coef[2] * (w - vt)
    vr = vr + noise * rng.standard_normal(vr.shape)
    vr = np.where(seen, vr, np.nan)
    dbz = np.where(seen | echo & covered, dbz, np.nan)

    # background given to the network: true profile + correlated error
    err = rng.uniform(0.5, 3.0)
    u_bg = np.broadcast_to(
        (ub + _profile(rng, z, err, 2000.0))[:, None, None], Z_.shape
    )
    v_bg = np.broadcast_to(
        (vb + _profile(rng, z, err, 2000.0))[:, None, None], Z_.shape
    )
    weight = np.where(echo | seen, 1.0, 0.25)
    return dict(
        vr=vr,
        dbz=dbz,
        coef=coef,
        u_bg=np.array(u_bg),
        v_bg=np.array(v_bg),
        rho=rho,
        fall_speed=vt,
        truth=np.stack([u, v, w]),
        weight=weight,
        distance=dist,
        x=x,
        y=y,
        z=z,
        radar=(rx_, ry_, alt),
        freezing_level=np.full((ny, nx), freezing),
    )
