---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  name: python3
  language: python
---

# Exercise on Fundamentals
This notebook reproduces several textbook radar problems, solved with functions from `radarx.fundamentals`.
These probelms are taken from
 - Rahman, H. (2019). Fundamental principles of radar. CRC Press.

```{code-cell} ipython3
import numpy as np
from radarx.fundamentals import constants, doppler, principles, timing, power, system
```

### Example 2.8 — S‑Band Pulsed Radar

> **Problem**
> Consider an **S‑band** pulsed radar transmitting **250 kW** of peak power with a pulse width of **1.5 µs** and a PRF of **500 pps**.
> The radar operates at a frequency of **3000 MHz**.
> 1. **(a)** Determine the maximum *unambiguous range*, *range resolution*, and *duty factor*.
> 2. **(b)** Compute the *average transmitted power* and the *energy radiated in the first 10 ms*.
> 3. **(c)** Calculate the *Doppler shift* for a target approaching with a radial velocity of **30 m s⁻¹**.

```{code-cell} ipython3
# Given values
P_peak = 250e3  # W
PRF = 500  # Hz
tau = 1.5e-6  # s
f0 = 3.0e9  # Hz
lam = constants.C / f0
v_r = 30  # m/s
t_interval = 10e-3  # 10 ms

# (a)
R_unamb = doppler.unambiguous_range(PRF)
R_res = principles.compute_range_resolution(tau)
duty = timing.compute_duty_cycle(tau, PRF)
duty_factor = tau * PRF

# (b)
P_avg = power.compute_average_power(P_peak, duty)
E_10ms = P_avg * t_interval

# (c)
f_d = doppler.doppler_frequency_shift(f0, v_r)

print(f"Unambiguous range      : {R_unamb:,.0f} m")
print(f"Range resolution       : {R_res:,.2f} m")
print(f"Duty cycle             : {duty:.4f}  (fraction)")
print(f"Duty factor (τ·PRF)    : {duty_factor:.4f}")
print(f"Average power          : {P_avg:,.1f} W")
print(f"Energy in first 10 ms  : {E_10ms:.4f} J")
print(f"Doppler shift (30 m/s) : {f_d:.2f} Hz")
```

### Example 3.1 — X‑Band Antenna Gain

> **Problem**
> Calculate the maximum gain of an **X‑band** antenna operating at **8 GHz** with a diameter of **1 m**.
> Repeat for diameters **1.5 m** and **2.0 m**.
> Assume aperture efficiency ρₐ = 1 so Aₑ = A.

```{code-cell} ipython3
f_x = 8e9
lam_x = constants.C / f_x
for D in [1.0, 1.5, 2.0]:
    Ae = np.pi * (D / 2) ** 2
    G = 4 * np.pi * Ae / lam_x**2  # linear
    G_dB = 10 * np.log10(G)
    print(f"Diameter {D:>3.1f} m → Gain = {G_dB:.2f} dB")
```

### Example 3.9 — Required Peak Power

> **Problem**
> An X‑band radar uses the same 3 m circular antenna (efficiency **0.8**) for Tx/Rx at **8 GHz**.
> The system should yield an average received power of **3 × 10⁻¹⁴ W** from a **1 m²** target at **100 km**.
> Total system loss is **3 dB**.  Determine the required **transmitter peak power**.

```{code-cell} ipython3
f = 8e9
lam = constants.C / f
D = 3.0
rho = 0.8
R = 100e3
sigma = 1.0
P_rx = 3e-14
L_lin = 10 ** (3 / 10)  # 3 dB loss

Ae = rho * np.pi * (D / 2) ** 2
G = 4 * np.pi * Ae / lam**2

P_tx = system.solve_peak_power(P_rx, G, lam, sigma, R, L_lin)
P_tx
# print(f"Required peak power ≈ {P_tx/1e6:.2f} MW")
```

### Multipath Propagation Example

> **Problem**
> A radar at **30 m** height tracks a target at **50 m**, range **40 km**, RCS **4 m²**.
> Antenna gain **34.77 dB**, peak power **1 MW**, frequency **5.4 GHz**.
> Above a smooth plane, estimate:
> 1. **(a)** Multipath propagation factor (two‑ray).
> 2. **(b)** Received signal power.

```{code-cell} ipython3
ht, hr, R = 30.0, 50.0, 40e3
sigma = 4.0
P_t = 1e6
f_c = 5.4e9
lam_c = constants.C / f_c
G_lin = 10 ** (34.77 / 10)

delta_r = 2 * ht * hr / R
delta_phi = 4 * np.pi * delta_r / lam_c
F = 2 * np.abs(np.cos(delta_phi / 2))
print(f"Propagation factor |F| ≈ {F:.2f}")

P_r = system.radar_equation(P_t, G_lin, lam_c, sigma, R, 1.0) * F**2
print(f"Received power ≈ {P_r:.2e} W")
```

### Missile Doppler Shift

> **Problem**
> A missile closing at **300 m s⁻¹** is illuminated by a **12 GHz** radar.
> 1. Compute the **exact** Doppler frequency.
> 2. Compute the **approximate** Doppler frequency (v ≪ c).

```{code-cell} ipython3
f0 = 12e9
v = 300
f_d_exact = doppler.doppler_frequency_shift(f0, v, exact=True)
f_d_approx = doppler.doppler_frequency_shift(f0, v, exact=False)
print(f"Exact Doppler  : {f_d_exact:.2f} Hz")
print(f"Approx Doppler : {f_d_approx:.2f} Hz")
```

```{code-cell} ipython3

```
