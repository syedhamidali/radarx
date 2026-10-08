# Machine learning with radarx

This folder holds the machine-learning code that lives **outside** the
`radarx` package: dataset builders (`ml/data/`) and, later, training code
(`ml/models/<name>/`). Nothing here is imported by `radarx`.

## Training data from NEXRAD with radarx labels (`ml/data/`)

`ml/data/build_dataset.py` builds training sets from NEXRAD Level II volumes
in the NOAA open-data archive on AWS (`s3://unidata-nexrad-level2`, read
anonymously). The labels come from radarx's physical retrievals, or are
known exactly because the inputs are degraded artificially:

| task | input | label | radarx function |
|---|---|---|---|
| `qc` | DBZH, ZDR, RHOHV, PHIDP | echo class, meteorological score | `echo_mask` |
| `kdp` | raw PHIDP, RHOHV, DBZH, ZDR | processed PHIDP, KDP | `estimate_kdp` |
| `hid` | DBZH, ZDR, RHOHV, KDP, temperature, meteo mask | hydrometeor class, confidence | `hid` (Park et al. 2009 at S band) |
| `dealias` | velocity folded at an artificial Nyquist velocity | true velocity, fold number (exact) | `dealias_velocity` (truth check and baseline) |
| `inpaint` | reflectivity with artificial blockage sectors | clean reflectivity (exact) | `echo_mask` + `apply_mask` |
| `nowcast` | composite reflectivity frames | next frame(s) | `grid_cones`, `estimate_motion`, `advect` (baseline) |
| `multidoppler` | gridded radial velocities of two radars + beam geometry | 3-D wind u, v, w | `multi_doppler_input`, `multi_doppler` |

### Quick start

```bash
pip install -e . zarr pyyaml        # radarx (with its compiled kernels), xradar, boto3
python ml/data/build_dataset.py ml/data/configs/demo.yaml --dry-run
python ml/data/build_dataset.py ml/data/configs/demo.yaml -o demo_out -w 8
```

```python
import sys; sys.path.insert(0, "ml/data")
import mldata

ds = mldata.open_dataset("demo_out", "dealias", "train")
mldata.validate(ds, "dealias")        # raises if the schema does not match
x = ds[["VRADH_folded", "nyquist_velocity"]]
y = ds["VRADH"]
```

The demo (`configs/demo.yaml`, 20 volumes, three events, one per split)
builds in a few minutes once the files are downloaded; downloads (about
10-20 MB per volume) are cached in `~/.cache/radarx-ml/nexrad` (`-c`).
`configs/conus_cases.yaml` is a starting point for a full set: 14 events
over four seasons and eight regimes (squall line, supercell, derecho,
tropical cyclones, summer convection, snow, sleet, upslope snow), about 1000
volumes.

### How it works

1. **Cases and splits** (`mldata/cases.py`). A case is one radar over a time
   window; an *event* groups cases that see the same storms (by default all
   cases starting on the same date). Splits are assigned per event: an
   explicit `split:` of any case of the event, else `test` if a radar in
   `holdout_radars` takes part, else a hash of the event key and the seed
   (fractions in `splits.fractions`). The build refuses to run if cases in
   different splits are closer than `min_gap_hours` (default 24 h) or a
   held-out radar appears outside `test`, so no storm, day or held-out radar
   leaks between train, val and test.
2. **Volumes in parallel** (`mldata/build.py:process_volume`). One process
   per volume (`concurrent.futures.ProcessPoolExecutor`); each one fetches the
   file, reads it with xradar (sweeps up to `max_elevation`; Nyquist velocity
   from the Message 31 headers; NEXRAD no-data codes set to NaN), and runs
   `echo_mask`, `estimate_kdp`, `hid` and `dealias_velocity` on the whole
   volume: each is one call of a multithreaded C++ kernel. Workers share the
   cores (`threads = cores // workers`).
3. **Samples.** Polar tasks give one sample per sweep: polarimetric sweeps
   for `qc`, `kdp`, `hid` and `inpaint`, sweeps with velocity for `dealias`;
   repeated SAILS cuts of the same elevation are skipped. All polar samples
   are put on one fixed grid (default 360 one-degree azimuth bins and 920
   gates of 250 m from 2.125 km) by taking the nearest native ray and gate,
   so inputs and labels come from the same native gate and classes stay
   valid. The quality-controlled volume is also gridded (cone interpolation)
   to a 256 x 256 km composite for nowcasting.
4. **Sequences and pairs.** Consecutive composites of a case (gaps up to
   `max_gap_minutes`) become nowcasting samples; configured radar pairs
   become multi-Doppler samples (both volumes quality-controlled and
   dealiased, advected to the analysis time with the motion of the first
   radar's last two frames).
5. **Zarr** (`consolidate`). Per-volume parts are appended into one store
   per task and split, `<out>/<task>/<split>.zarr` (Zarr format 2,
   consolidated metadata, one chunk per sample and variable), and
   `<out>/manifest.json` records the provenance: config, cases and their
   splits, every volume (S3 key, split, sample counts, timings, errors),
   radarx version, git commit, creation time and the data licence.

Randomness (artificial Nyquist velocity, blockage sectors) is drawn from a
generator seeded by `(seed, volume, sweep, task)`, so a sample does not
depend on the number of workers or the processing order.

### Label details and caveats

- **dealias.** The *truth* is the measured velocity when radarx's dealiasing
  leaves every gate in its fold (`truth_source = 0`), else radarx's
  dealiased field (`truth_source = 1`); either is accepted only if fewer
  than `max_jump_fraction` of neighbouring gate pairs differ by more than the
  Nyquist velocity (no fold lines left). The truth is then folded exactly at
  an artificial Nyquist velocity drawn from `nyquist` (default 8-20 m/s):
  `VRADH = VRADH_folded + 2 * FOLD * nyquist_velocity`. Filter on
  `truth_source == 0` for observation-only truth. `VRADH_radarx` is radarx's
  single-sweep dealiasing of the folded input, a baseline to beat; without a
  reference wind it can miss the absolute fold when echo covers only part of
  the circle.
- **inpaint.** Blocked sectors start at an obstacle range and extend to the
  end of the ray, as real blockage does. Total blockage gives NaN; partial
  blockage of power fraction `f` lowers reflectivity by `-10 log10(1 - f)`
  dB. The label is the quality-controlled reflectivity (non-meteorological
  echo removed).
- **hid.** Temperature comes from the case's `freezing_level` (constant
  6.5 K/km lapse rate, `mldata.labels.standard_profile`); put the value of
  the nearest sounding in the config. Only meteorological gates
  (`METEO_MASK`) are classified; class 0 means not classified.
- **kdp.** `PHIDP_processed` bridges masked gates and holds its end values
  beyond the first and last valid gate (as radarx returns it); `KDP` is NaN
  at non-meteorological gates.
- **Physical labels are not ground truth.** `qc`, `kdp`, `hid` and
  `multidoppler` labels are what radarx's published methods give; a model
  trained on them learns to reproduce (and speed up or regularise) those
  methods. The `dealias` and `inpaint` targets are exact by construction.

### Dataset schemas (version 1)

All stores stack samples on `sample`. Polar tasks share the dimensions
`(sample, azimuth, range)` with coordinates `azimuth` (degrees, bin centres)
and `range` (m), and the per-sample coordinates `radar` (ICAO), `volume_id`
(`RADAR_YYYYmmddTHHMMSS`), `sweep` (index in the volume), `elevation`
(degrees), `time` (sweep start), `latitude`, `longitude`, `altitude` (radar
site) and `case`. Missing data are NaN in float variables and 0 in class
variables. Every variable carries `units`, `long_name` and `role`
(`input`, `label`, `baseline`, `meta`); `mldata.schema.SCHEMAS` is the
machine-readable form and `mldata.validate(ds, task)` checks a dataset.

**qc** — inputs `DBZH` (dBZ), `ZDR` (dB), `RHOHV` (1), `PHIDP` (degrees,
raw), all float32; labels `ECHO_CLASS` (int8: 0 no echo, 1 meteorological,
2 non-meteorological, 3 speckle), `METEO_SCORE` (float32, 0-1).

**kdp** — inputs `PHIDP` (raw), `RHOHV`, `DBZH`, `ZDR`; labels
`PHIDP_processed` (degrees), `KDP` (degrees/km).

**hid** — inputs `DBZH`, `ZDR`, `RHOHV`, `KDP` (degrees/km, from radarx),
`TEMPERATURE` (degC at the gate, NaN without a freezing level), `METEO_MASK`
(bool); labels `HID` (int8; `flag_values`/`flag_meanings` attributes, Park
classes at S band: 1 DS, 2 WS, 3 CR, 4 GR, 5 BD, 6 RA, 7 HR, 8 RH; 0 not
classified), `HID_confidence` (float32).

**dealias** — inputs `VRADH_folded` (m/s), `DBZH`, `nyquist_velocity`
(sample, m/s); labels `VRADH` (m/s), `FOLD` (int8); baseline `VRADH_radarx`;
meta `nyquist_velocity_original`, `truth_source` (int8), `folded_fraction`.

**inpaint** — inputs `DBZH_blocked` (dBZ), `BLOCKAGE` (blocked power
fraction 0-1); label `DBZH` (dBZ).

**nowcast** — dimensions `(sample, lead, y, x)` and `(sample, lead_target,
y, x)` on a radar-centred grid (`x`, `y` in m; default 1 km, 256 x 256);
`lead` counts volumes relative to the last input frame (default `[-1, 0]`),
`lead_target` the frames to predict (default `[1]`). Inputs `DBZH`
(column-maximum reflectivity; -10 dBZ for no echo within 230 km, NaN beyond),
label `DBZH_target`, baselines `u`, `v` (m/s on `(sample, y, x)`, motion of
the input frames, tiled) and `DBZH_extrapolated` (last input advected to each
target time); meta `frame_time`, `target_time`; coordinates `radar`, `case`,
`time` (lead-0 time). Volume intervals vary (about 4-10 min): use the times.

**multidoppler** — dimensions `(sample, radar_index, z, y, x)` and
`(sample, z, y, x)`; grid from the config relative to the first radar
(`x`, `y` east/north in m, `z` m above sea level). Inputs `VRADH`
(dealiased radial velocity), `DBZH`, `beam_azimuth`, `beam_elevation`
(degrees); labels `u`, `v`, `w` (m/s), `n_radars` (int8),
`beam_crossing_angle` (degrees); coordinates `radar` (`(sample,
radar_index)`), `case`, `time` (analysis time). For single-Doppler wind
models take one `radar_index` as input and `u`, `v` where `n_radars == 2`
and the crossing angle exceeds 30 degrees as the target.

### Licence and citation

NEXRAD Level II data are NOAA open data without restrictions on use. Cite
them as: NOAA National Weather Service Radar Operations Center (1991): NOAA
Next Generation Radar (NEXRAD) Level II Base Data. NOAA National Centers for
Environmental Information, https://doi.org/10.7289/V5W9574V. The methods
behind the labels are cited in the docstrings of the radarx functions
listed above. radarx and these builders are MIT licensed.

### Tests

`tests/ml_data/` checks the label builders and the whole pipeline on
synthetic sweeps (no network): exact folding, blockage geometry, fixed-grid
resampling, splits and leakage checks, reproducibility and the schemas.
