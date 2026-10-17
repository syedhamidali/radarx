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

# Classification of Cloud Radar Echoes

`classify_cloud_targets` sorts the echoes of a cloud radar into cloud droplets, drizzle or rain, ice,
melting ice, insects and (with a dual-frequency ratio) large ice, from the reflectivity, the temperature and
the depolarization ratio. The categories are those of Cloudnet, which needs a lidar and a model besides the
radar; the classification of radarx uses the radar and a temperature only, so it has no class for supercooled
liquid, aerosol or mixtures. The example is a day of open data of the Cloudnet station at Lindenberg
(Germany, 1 May 2023): the categorization file, with the 35.5 GHz radar, the model temperature and the
Cloudnet classification to compare with. The cell downloads two files (13 MB) into the temporary directory
and keeps the hours from 18 to 24 UTC, in which an ice cloud, melting ice, drizzle and liquid layers pass.

```{code-cell} ipython3
import tempfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

import radarx  # noqa: F401  registers the .radarx accessors
from radarx.retrieve import classify_cloud_targets
```

```{code-cell} ipython3
BASE = "https://cloudnet.fmi.fi/api/download/product/"
FILES = {
    "categorize": "231c0919-f399-4fea-aa04-3435754f96b9/20230501_lindenberg_categorize.nc",
    "classification": "9eac625e-a74b-478c-b5f6-ff66aaf6771d/20230501_lindenberg_classification.nc",
}
cache = Path(tempfile.gettempdir()) / "radarx_cloudnet"
cache.mkdir(exist_ok=True)


def fetch(key, attempts=4):
    """Download a file once (to a temporary name first, with retries) and open it."""
    path = cache / Path(FILES[key]).name
    for attempt in range(attempts):
        if path.exists():
            break
        part = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(BASE + FILES[key], part)  # noqa: S310
            part.rename(path)
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
    return xr.open_dataset(path, engine="netcdf4")


window = slice("2023-05-01T18:00", "2023-05-01T23:59:59")
cat = fetch("categorize").sel(time=window).load()
cls = fetch("classification").sel(time=window).load()
print(cat.Z.shape, float(cat.radar_frequency))
```

`classify_cloud_targets` takes the reflectivity, the temperature (here the wet-bulb temperature of the
categorization file, as Cloudnet places the melting layer) and the depolarization ratio. The comparison merges
the classes of Cloudnet into the ones a radar alone can tell.

```{code-cell} ipython3
section = xr.Dataset({"Z": cat.Z, "ldr": cat.ldr, "Tw": cat.Tw}).rename(height="range")
classes = classify_cloud_targets(section, "Tw", ldr="ldr")
names = ["no echo", "droplets", "drizzle/rain", "ice", "melting ice", "insects"]
merge = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 0, 9: 5, 10: 5}
reference = np.vectorize(lambda v: merge.get(int(v), 0) if np.isfinite(v) else 0)(cls.target_classification.values)
echo = np.isfinite(cat.Z.values)
print(f"gates with an echo: {echo.sum()}, agreement with Cloudnet: {(classes.values == reference)[echo].mean():.3f}")
for i, name in enumerate(names):
    sel = (reference == i) & echo
    if sel.sum():
        print(f"  Cloudnet {name:13s}: {sel.sum():7d} gates, radarx gives the same for {(classes.values[sel] == i).mean():.2f}")

colors = ["white", "tab:cyan", "tab:blue", "0.7", "tab:orange", "tab:green"]
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
for ax, data, title in ((axes[0], reference, "Cloudnet (radar, lidar and model), merged"), (axes[1], classes.values, "radarx, radar only")):
    ax.pcolormesh(cat.time.values, cat.height.values / 1e3, np.where(echo, data, 0).T, cmap=cmap, norm=norm, shading="auto")
    ax.set(ylabel="height (km)", title=title, ylim=(0, 9))
handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="k", lw=0.3) for c in colors[1:]]
axes[1].legend(handles, names[1:], ncol=5, loc="upper right", fontsize=8)
```

Ice is found in every gate of Cloudnet, and the melting layer and the rain agree to most of the gates. The weak
echoes of the boundary layer are insects for Cloudnet (it has the lidar) and for radarx as long as the
depolarization ratio is given; without it they become drizzle. In the weakest echoes of the night radarx puts
a melting layer where the temperature of the model is in the zone of 0 to 2 degrees Celsius, whatever the echo is.
The thresholds are radarx choices (see `radarx.retrieve.cloud_classification`).

```{code-cell} ipython3
same = section.radarx.classify_cloud_targets("Tw", ldr="ldr")
print(bool((same == classes).all()))
blind = classify_cloud_targets(section, "Tw")  # without the depolarization ratio
print("insects without ldr:", int((blind == 5).sum()), " drizzle or rain without ldr:", int((blind == 2).sum()),
      " with ldr:", int((classes == 2).sum()))
```

## References

- Illingworth, A. J., et al., 2007: Cloudnet: Continuous evaluation of cloud profiles in seven operational models using ground-based observations. *Bulletin of the American Meteorological Society*, **88**, 883-898, <https://doi.org/10.1175/BAMS-88-6-883>
- Tukiainen, S., E. O'Connor, and A. Korpinen, 2020: CloudnetPy: A Python package for processing cloud remote sensing data. *Journal of Open Source Software*, **5**, 2123, <https://doi.org/10.21105/joss.02123>
- Data: ACTRIS Cloud Remote Sensing Data Centre Unit (CLU), Cloudnet products of Lindenberg (DWD Meteorological Observatory Lindenberg), 1 May 2023, <https://cloudnet.fmi.fi> (CC BY 4.0); the product identifiers are in the `pid` attribute of the files.
