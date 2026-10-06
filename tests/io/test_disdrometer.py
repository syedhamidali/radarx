#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the disdrometer readers
=================================
"""

import numpy as np
import pytest
import xarray as xr

from radarx.io import parsivel_classes, read_parsivel, read_pips_netcdf


def _spectrum(seed):
    rnd = np.random.default_rng(seed)
    c = np.zeros((32, 32), int)
    c[10:20, 5:15] = rnd.integers(0, 5, (10, 10))
    return c


def _telegram(counts, hms="00:29:22", dmy="31.03.2022", rate=12.5, n=None):
    n = int(counts.sum()) if n is None else n
    head = ["304545", f"{rate:08.3f}", "0008.44", "45.680", "00010", "14119"]
    head += [f"{n:05d}", "-55", "11.6", hms, dmy]
    return ";".join(head + [f"{x:03d}" for x in counts.ravel()]) + ";"


def test_parsivel_classes():
    ds = parsivel_classes()
    assert ds.sizes == {"diameter": 32, "velocity": 32}
    np.testing.assert_allclose(ds.diameter_upper[:-1], ds.diameter_lower[1:])
    np.testing.assert_allclose(ds.velocity_upper[:-1], ds.velocity_lower[1:])
    assert ds.diameter_lower[0] == 0 and ds.diameter_upper[-1] == 26.0
    assert ds.velocity_upper[-1] == pytest.approx(22.4)
    np.testing.assert_allclose(ds.diameter[:3], [0.0625, 0.1875, 0.3125])
    np.testing.assert_allclose(ds.velocity[[0, 10, 31]], [0.05, 1.1, 20.8])


def _pips_file(tmp_path, gps=True):
    header = (
        "TIMESTAMP,BattV,PTemp_C,WindDir,WS_ms,WSDiag,FastTemp,SlowTemp,RH,"
        "Pressure,FluxDirection,GPSTime,GPSStatus,GPSLat,GPSLatHem,GPSLon,"
        "GPSLonHem,GPSSpd,GPSDir,GPSDate,GPSMagVar,GPSAlt,WindDirAbs,Dewpoint,"
        "RHDer,ParsivelStr"
    )
    lines = [header]
    spectra = []
    for s in range(30):  # 1 Hz records, a telegram every 10 s
        sec = 10 + s
        stamp = f"2022-03-31 00:29:{sec:02d}"
        gtime = f"0029{sec + 3:02d}" if gps else ""
        tel = "NaN"
        if s % 10 == 9:
            spec = _spectrum(s)
            spectra.append(spec)
            tel = _telegram(spec, hms=f"00:29:{sec:02d}")
        wind = 2.0 + s % 10  # 2 .. 11 m/s within each interval
        lines.append(
            f"{stamp},12.6,21.9,183,{wind},0,17.8,18.0,94.0,993.0,329.7,"
            f"{gtime},A,33.4548,N,88.2677,W,0,217.4,310322,1.9,70.5,"
            f"{90 if s % 2 else 270},17.3,96.7,{tel}"
        )
    lines.append(lines[-1])  # duplicated record
    lines.append(
        "2022-03-31 00:29:41,12.6,21.9,183,2,0,17.8,18.0,94.0,993.0,329.7,"
        ",,NaN,,NaN,,NaN,NaN,,,NaN,0,17.3,96.7,NAN"
    )
    path = tmp_path / "PIPS1A_test_merged.txt"
    path.write_text("\n".join(lines) + "\n")
    return path, spectra


def test_read_pips_merged(tmp_path):
    path, spectra = _pips_file(tmp_path)
    ds = read_parsivel(path)
    assert ds.sizes["time"] == 3
    assert ds.counts.dims == ("time", "velocity", "diameter")
    np.testing.assert_array_equal(ds.counts.values, np.stack(spectra))
    # GPS time: logger + 3 s
    assert str(ds.time.values[0])[:19] == "2022-03-31T00:29:22"
    assert ds.attrs["time_source"] == "GPS"
    assert ds.station.item() == "PIPS1A"
    assert ds.latitude.item() == pytest.approx(33 + 45.48 / 60)
    assert ds.longitude.item() == pytest.approx(-(88 + 26.77 / 60))
    assert ds.altitude.item() == pytest.approx(70.5)
    assert ds.rain_rate_instrument.values[0] == pytest.approx(12.5)
    assert ds.particle_count.values[1] == spectra[1].sum()
    assert ds.sample_interval.values[0] == 10
    # 1 Hz wind averaged over each interval (2 .. 11 m/s), gust = maximum
    assert ds.wind_speed.values[1] == pytest.approx(6.5)
    assert ds.wind_speed_max.values[1] == pytest.approx(11.0)
    # circular mean of 90 and 270 alternating is undefined-ish; just in range
    assert 0 <= ds.wind_direction.values[0] < 360
    assert ds.air_pressure.attrs["units"] == "hPa"
    logger = read_parsivel(path, time="logger")
    assert str(logger.time.values[0])[:19] == "2022-03-31T00:29:19"
    inst = read_parsivel(path, time="instrument", station="X", latitude=1.0)
    assert inst.station.item() == "X" and inst.latitude.item() == 1.0
    assert inst.attrs["time_source"] == "instrument clock"


def test_read_pips_without_gps(tmp_path):
    path, _ = _pips_file(tmp_path, gps=False)
    ds = read_parsivel(path)
    assert ds.attrs["time_source"] == "logger"
    with pytest.raises(ValueError, match="no GPS time"):
        read_parsivel(path, time="gps")


def test_read_toa5_and_plain(tmp_path):
    spec = _spectrum(3)
    toa5 = tmp_path / "TOA5_Parsivel_TenHz0.dat"
    toa5.write_text(
        '"TOA5","2580","CR6","2580","CR6.Std.09.02","CPU:x.CR6","58156","Ten_Hz"\n'
        '"TIMESTAMP","RECORD","ParsivelStr"\n"TS","RN","ParsivelRawData"\n'
        '"","","Smp"\n"2022-02-28 17:51:30",0,"NAN"\n'
        f'"2022-02-28 17:51:40",1,"{_telegram(spec, hms="17:51:38", dmy="28.02.2022")}"\n'
    )
    ds = read_parsivel(toa5)
    assert ds.sizes["time"] == 1 and ds.attrs["time_source"] == "logger"
    assert str(ds.time.values[0])[:19] == "2022-02-28T17:51:40"
    assert ds.station.item() == "Parsivel 304545"
    assert np.isnan(ds.latitude.item())
    np.testing.assert_array_equal(ds.counts.values[0], spec)

    plain = tmp_path / "telegrams.txt"
    plain.write_text(
        _telegram(spec, hms="10:00:10", dmy="01.06.2024")
        + "\n"
        + _telegram(spec * 2, hms="10:00:00", dmy="01.06.2024")
        + "\nbroken;line\n"
    )
    ds = read_parsivel(plain)
    assert ds.sizes["time"] == 2
    assert str(ds.time.values[0])[:19] == "2024-06-01T10:00:00"  # sorted
    np.testing.assert_array_equal(ds.counts.values[0], spec * 2)
    with pytest.raises(ValueError, match="no logger time"):
        read_parsivel(plain, time="logger")


def test_read_errors(tmp_path):
    with pytest.raises(ValueError, match="time must be"):
        read_parsivel(tmp_path / "x", time="bad")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_parsivel(empty)
    nocol = tmp_path / "nocol.csv"
    nocol.write_text("TIMESTAMP,A\n2022-01-01 00:00:00,1\n")
    with pytest.raises(ValueError, match="no ParsivelStr"):
        read_parsivel(nocol)
    notel = tmp_path / "notel.csv"
    notel.write_text("TIMESTAMP,ParsivelStr\n2022-01-01 00:00:00,NAN\n")
    with pytest.raises(ValueError, match="no complete Parsivel telegram"):
        read_parsivel(notel)
    bad = tmp_path / "bad.txt"
    bad.write_text(_telegram(_spectrum(1)).replace(";000;", ";x0;", 1) + "\n")
    with pytest.raises(ValueError, match="no complete"):
        read_parsivel(bad)


def test_read_pips_netcdf(tmp_path):
    vd = np.full((3, 32, 32), np.nan)
    vd[:, 12, 8] = [1, 2, 3]
    src = xr.Dataset(
        {
            "VD_matrix": (("time", "fallspeed_bin", "diameter_bin"), vd),
            "pcount": ("time", [1, 2, 3]),
            "precipintensity": ("time", [0.1, 0.2, 0.3]),
            "sample_interval": ("time", [10.0, 10.0, 10.0]),
            "windspd": ("time", [1.0, 2.0, 3.0]),
        },
        coords={
            "time": np.array(
                ["2022-03-31T00:00:00", "2022-03-31T00:00:10", "2022-03-31T00:00:20"],
                "datetime64[ns]",
            )
        },
        attrs={
            "probe_name": "PIPS2A",
            "location": "(33.8, -88.5, 73.6)",
            "deployment_name": "IOP2",
        },
    )
    path = tmp_path / "parsivel_combined_test.nc"
    src.to_netcdf(path)
    ds = read_pips_netcdf(path)
    assert ds.station.item() == "PIPS2A"
    assert (ds.latitude.item(), ds.longitude.item(), ds.altitude.item()) == (
        33.8,
        -88.5,
        73.6,
    )
    assert ds.counts.sum().item() == 6 and ds.counts.dtype == np.int32
    assert ds.counts.values[2, 12, 8] == 3
    assert ds.attrs["deployment_name"] == "IOP2"
    assert "wind_speed" in ds and "rain_rate_instrument" in ds
    src.drop_vars("VD_matrix").to_netcdf(tmp_path / "no_vd.nc")
    with pytest.raises(ValueError, match="no VD_matrix"):
        read_pips_netcdf(tmp_path / "no_vd.nc")
