#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for radarx.io.surface and radarx.io.profiler (format readers)."""

import numpy as np
import pytest
import xarray as xr

from radarx.io import profiler, read_mrr, read_pips, read_sticknet, surface
from radarx.retrieve import bulk_shear, cold_pool_perturbation

LOCATIONS = """ID,Latitude,Longitude,Elevation,Array_Type
101A,33.89005,-89.019192,111.7,Coarse
103A,33.714987,-88.450477,72.4,Fine
"""


def write_sticknet(tmp_path, sid="0101A", iop=2, level=3, n=120, start="16:00:00"):
    lines = ["Time,T,RH,P,WS,WD"] if level != 1 else ["TIME,T,RH,P,WS,WD,TFLAG,WFLAG"]
    t0 = np.datetime64(f"2022-03-30T{start}")
    for i in range(n):
        t = str(t0 + np.timedelta64(i, "s")).replace("T", " ")
        temp = 20.0 if i < n // 2 else 15.0
        row = [t, f"{temp:.1f}", "80.0", "1000.0", "5.0", "270.0"]
        if level == 1:
            row += ["1" if i == 3 else "0", "1" if i == 4 else "0"]
        if i == 5:
            row[2] = ""  # missing humidity
        lines.append(",".join(row))
    path = tmp_path / f"{sid}_IOP{iop}_level{level}.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_sticknet_directory(tmp_path):
    (tmp_path / "IOP2_StickNet_Locations.csv").write_text(LOCATIONS)
    write_sticknet(tmp_path, "0101A")
    write_sticknet(tmp_path, "0103A", start="16:00:30")
    write_sticknet(tmp_path, "0101A", iop=3)
    with pytest.raises(ValueError, match="several deployments"):
        read_sticknet(tmp_path)
    ds = read_sticknet(tmp_path, iop=2)
    assert ds.sizes == {"station": 2, "time": 150}
    assert list(ds.station.values) == ["101A", "103A"]
    np.testing.assert_allclose(ds.latitude, [33.89005, 33.714987])
    np.testing.assert_allclose(ds.altitude, [111.7, 72.4])
    assert list(ds.array_type.values) == ["Coarse", "Fine"]
    assert list(ds.deployment.values) == ["IOP2", "IOP2"]
    t = ds.temperature.sel(station="101A")
    assert float(t[0]) == pytest.approx(293.15)
    assert ds.pressure.attrs["units"] == "Pa" and float(ds.pressure[0, 0]) == 1e5
    assert float(ds.relative_humidity[0, 0]) == pytest.approx(0.8)
    assert np.isnan(ds.relative_humidity.sel(station="101A")[5])
    # wind from the west: blowing toward the east
    assert float(ds.u[0, 0]) == pytest.approx(5.0)
    assert float(ds.v[0, 0]) == pytest.approx(0.0, abs=1e-12)
    # stations sampled at different times share the union of times
    assert np.isnan(ds.temperature.sel(station="103A")[0])
    assert np.isfinite(ds.dewpoint[0, 0]) and float(ds.dewpoint[0, 0]) < 293.15
    # time window
    sub = read_sticknet(
        tmp_path, iop=2, time=slice("2022-03-30T16:00:10", "2022-03-30T16:00:20")
    )
    assert sub.sizes["time"] == 11
    # the network feeds the cold-pool diagnostics
    pert = cold_pool_perturbation(ds, slice("2022-03-30T16:00", "2022-03-30T16:00:50"))
    assert float(pert.buoyancy.sel(station="101A").min()) < -0.1


def test_read_sticknet_level1_flags_and_explicit_locations(tmp_path):
    loc = tmp_path / "locs.csv"
    loc.write_text(LOCATIONS)
    f = write_sticknet(tmp_path, "0103A", level=1, n=20)
    ds = read_sticknet([f], locations=loc)
    assert np.isnan(ds.temperature[0, 3]) and np.isnan(ds.pressure[0, 3])
    assert np.isnan(ds.wind_speed[0, 4]) and np.isfinite(ds.temperature[0, 4])
    assert ds.attrs["level"] == 1
    raw = read_sticknet(
        f, locations=surface.read_sticknet_locations(loc), apply_flags=False
    )
    assert np.isfinite(raw.temperature[0, 3])
    # no table found: NaN locations
    other = tmp_path / "other"
    other.mkdir()
    g = write_sticknet(other, "0105A")
    assert np.isnan(read_sticknet(g).latitude[0])
    with pytest.raises(FileNotFoundError):
        read_sticknet([])
    with pytest.raises(ValueError, match="StickNet file name"):
        surface._sticknet_id("station.txt")
    bad = tmp_path / "bad.csv"
    bad.write_text("Name,Lat\nA,1\n")
    with pytest.raises(ValueError, match="column"):
        surface.read_sticknet_locations(bad)
    badfile = tmp_path / "0107A_IOP2_level3.txt"
    badfile.write_text("Time,T,RH\n2022-03-30 16:00:00,1,2\n")
    with pytest.raises(ValueError, match="column"):
        read_sticknet(badfile)


def write_pips(path, name, start, temp):
    time = np.arange(
        np.datetime64(start), np.datetime64(start) + np.timedelta64(60, "s")
    ).astype("datetime64[ns]")
    n = time.size
    ds = xr.Dataset(
        {
            "fasttemp": ("time", np.full(n, temp)),
            "slowtemp": ("time", np.full(n, temp + 0.2)),
            "RH": ("time", np.full(n, 90.0)),
            "pressure": ("time", np.full(n, 990.0)),
            "windspd": ("time", np.full(n, 3.0)),
            "winddirabs": ("time", np.full(n, 180.0)),
        },
        coords={"time": time[::-1]},  # out of order on purpose
        attrs={
            "probe_name": name,
            "location": "(33.75, -88.45, 70.7)",
            "deployment_name": "IOP2_033022",
        },
    )
    ds.to_netcdf(path)


def test_read_pips_and_concat_with_sticknet(tmp_path):
    write_pips(
        tmp_path / "conventional_raw_A.nc", "PIPS1A", "2022-03-30T16:00:30", 18.0
    )
    write_pips(
        tmp_path / "conventional_raw_B.nc", "PIPS1B", "2022-03-30T16:00:00", 19.0
    )
    pp = read_pips(tmp_path)
    assert list(pp.station.values) == ["PIPS1A", "PIPS1B"]
    assert pp.sizes["time"] == 90
    assert float(pp.altitude[0]) == pytest.approx(70.7)
    assert float(
        pp.temperature.sel(station="PIPS1A").dropna("time")[0]
    ) == pytest.approx(291.15)
    assert bool((pp.time.diff("time") > np.timedelta64(0, "ns")).all())
    assert float(pp.v[0].dropna("time")[0]) == pytest.approx(3.0)
    slow = read_pips(tmp_path / "conventional_raw_A.nc", temperature="slowtemp")
    assert float(slow.temperature[0, 0]) == pytest.approx(291.35)
    with pytest.raises(KeyError):
        read_pips(tmp_path / "conventional_raw_A.nc", temperature="hottemp")
    with pytest.raises(FileNotFoundError):
        read_pips([])
    (tmp_path / "IOP2_StickNet_Locations.csv").write_text(LOCATIONS)
    write_sticknet(tmp_path, "0101A")
    sn = read_sticknet(tmp_path, iop=2)
    both = xr.concat([sn, pp], dim="station", join="outer")
    assert both.sizes["station"] == 3
    assert list(both.platform.values) == ["TTU StickNet", "PIPS", "PIPS"]
    sub = read_pips(tmp_path, time=slice("2022-03-30T16:00:40", None))
    assert sub.time.values.min() >= np.datetime64("2022-03-30T16:00:40")


def test_station_duplicates_rejected():
    rec = (np.array(["2022-01-01"], dtype="datetime64[ns]"), {}, {})
    with pytest.raises(ValueError, match="duplicate"):
        surface._station_dataset([("A",) + rec, ("A",) + rec], source="x")


def test_read_mrr(tmp_path):
    nt, ng, nb = 4, 5, 3
    gates = np.tile(np.arange(1, ng + 1) * 150.0, (nt, 1))
    data = {
        "MRR rangegate": (("time", "MRR rangegate"), gates),
        "MRR_Capital_Z": (("time", "MRR rangegate"), np.full((nt, ng), 30.0)),
        "MRR_W": (("time", "MRR rangegate"), np.full((nt, ng), 6.0)),
        "MRR_RR": (("time", "MRR rangegate"), np.full((nt, ng), 5.0)),
        "MRR_N": (
            ("time", "MRR rangegate", "MRR spectralclass"),
            np.ones((nt, ng, nb)),
        ),
    }
    raw = xr.Dataset(
        data,
        coords={"time": ("time", 1648651350.0 + 30 * np.arange(nt))},
        attrs={"system": "MRR"},
    )
    raw.time.attrs["units"] = "seconds since 1970-01-01 00:00:00"
    f = tmp_path / "mrr.nc"
    raw.to_netcdf(f)
    ds = read_mrr(f, latitude=33.6, longitude=-89.0, altitude=87.0, spectra=True)
    assert ds.DBZ.dims == ("time", "height")
    np.testing.assert_allclose(ds.height, 87.0 + np.arange(1, ng + 1) * 150.0)
    np.testing.assert_allclose(ds.height_agl, np.arange(1, ng + 1) * 150.0)
    assert ds.time.values[0] == np.datetime64("2022-03-30T14:42:30")
    assert float(ds.altitude) == 87.0
    assert ds.drop_number_density.dims == ("time", "height", "bin")
    assert ds.DBZ.attrs["units"] == "dBZ"
    plain = read_mrr(f)
    np.testing.assert_allclose(plain.height, plain.height_agl)
    assert "drop_number_density" not in plain and np.isnan(plain.latitude)
    gates[1, 0] = 140.0
    raw["MRR rangegate"] = (("time", "MRR rangegate"), gates)
    raw.to_netcdf(tmp_path / "bad.nc")
    with pytest.raises(ValueError, match="change"):
        read_mrr(tmp_path / "bad.nc")


def write_rwp(path):
    nt, nh = 3, 6
    height = 126.0 + 100.0 * np.arange(nh)
    u = np.tile(np.linspace(2.0, 12.0, nh), (nt, 1))
    u[0, 0] = 999.9
    qc = np.full((nt, nh), 5.0)
    qc[2, :] = 1.0
    ds = xr.Dataset(
        {
            "epochTime": ("time", 1648652400.0 + 300 * np.arange(nt)),
            "u": (("time", "height"), u),
            "v": (("time", "height"), np.zeros((nt, nh))),
            "w": (("time", "height"), np.full((nt, nh), 999.9)),
            "qcTag": (("time", "height"), qc),
            "beam_azimuths": ("beamAZ", np.array([35.0, 95.0])),
            "Vel_1": (("time", "height"), np.ones((nt, nh))),
            "Vel_2": (("time", "height"), -np.ones((nt, nh))),
            "SNR_1": (("time", "height"), np.ones((nt, nh))),
            "SNR_2": (("time", "height"), np.ones((nt, nh))),
        },
        coords={
            "height": height,
            "latitude": ("latitude", [33.6]),
            "longitude": ("longitude", [-88.99]),
            "altitude": ("altitude", [87.0]),
        },
        attrs={"System": "RWP"},
    )
    ds.to_netcdf(path)


def test_read_wind_profiler(tmp_path):
    f = tmp_path / "rwp.nc"
    write_rwp(f)
    ds = profiler.read_wind_profiler(f, beams=True)
    assert ds.u.dims == ("time", "height")
    assert np.isnan(ds.u[0, 0]) and np.isnan(ds.w).all()
    np.testing.assert_allclose(ds.height, 87.0 + 126.0 + 100.0 * np.arange(6))
    assert float(ds.wind_direction[1, 0]) == pytest.approx(270.0)
    assert ds.beam_velocity.shape == (3, 6, 2)
    np.testing.assert_allclose(ds.beam_azimuth, [35.0, 95.0])
    assert "beam_spectrum_width" not in ds
    assert ds.time.values[0] == np.datetime64("2022-03-30T15:00:00")
    good = profiler.read_wind_profiler(f, min_qc=3.0)
    assert np.isnan(good.u[2]).all() and np.isfinite(good.u[1]).all()
    # profiler winds feed the wind-profile diagnostics, one value per time
    sh = bulk_shear(good, 0, 400, ground=good.altitude + 126.0)
    np.testing.assert_allclose(sh.shear_u, [np.nan, 8.0, np.nan])
