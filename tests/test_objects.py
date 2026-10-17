#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Tests for the object kernels on stacks of frames
================================================

Labelling, the nested threshold regions, the watershed, the sums over labels and the
overlap counts have analytic answers on small images, and the compiled kernel and
the NumPy twin must give the same labels (the same tie breaks) on random fields.
"""

import numpy as np
import pytest

from radarx.retrieve import _cells

ENGINES = [False] + ([True] if _cells.HAS_COMPILED_KERNEL else [])
compiled_only = pytest.mark.skipif(
    not _cells.HAS_COMPILED_KERNEL, reason="compiled kernel not built"
)


def smooth_field(seed, shape=(3, 40, 50), sigma=2.0):
    from scipy import ndimage

    rnd = np.random.default_rng(seed)
    out = ndimage.gaussian_filter(rnd.normal(size=shape), (0, sigma, sigma))
    return out / out.std()


@pytest.mark.parametrize("use", ENGINES)
@pytest.mark.parametrize("connectivity", [4, 8])
def test_label_matches_scipy(use, connectivity):
    from scipy import ndimage

    mask = smooth_field(1) > 0.4
    labels, counts = _cells.label(mask, connectivity, use, 2)
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    for t in range(mask.shape[0]):
        expected, n = ndimage.label(mask[t], structure=structure)
        np.testing.assert_array_equal(labels[t], expected)
        assert counts[t] == n


@pytest.mark.parametrize("use", ENGINES)
def test_features_are_the_leaves_of_the_nested_regions(use):
    frame = np.zeros((1, 30, 40))
    frame[0, 5:12, 5:12] = 25.0  # a weak storm: its own feature at 20
    frame[0, 15:28, 5:25] = 25.0  # a region that holds two stronger regions
    frame[0, 18:23, 8:12] = 45.0
    frame[0, 18:23, 16:21] = 45.0
    frame[0, 19:21, 9:11] = 55.0  # a feature of the highest threshold inside the first
    markers, counts, levels = _cells.features(
        frame, np.array([20.0, 40.0, 50.0]), 4, 8, use, 1
    )
    assert counts.tolist() == [3]
    # numbered in the raster order of the first pixel: the weak storm, the 45 dBZ
    # region on the right (its first pixel is in row 18) and the 55 dBZ region
    assert levels.tolist() == [0, 1, 2]
    assert markers[0, 8, 8] == 1 and markers[0, 19, 18] == 2 and markers[0, 20, 9] == 3
    assert markers[0, 16, 6] == 0  # the 25 dBZ region around is replaced by its parts
    # a region smaller than min_size at a threshold does not replace its parent
    markers, counts, levels = _cells.features(
        frame, np.array([20.0, 40.0, 50.0]), 6, 8, use, 1
    )
    assert counts.tolist() == [3] and levels.tolist() == [0, 1, 1]


@pytest.mark.parametrize("use", ENGINES)
def test_features_ignore_nan_and_keep_one_threshold(use):
    frame = np.full((2, 10, 10), np.nan)
    frame[0, 2:5, 2:5] = 31.0
    frame[1, 2:5, 2:5] = 29.0
    markers, counts, levels = _cells.features(frame, np.array([30.0]), 1, 4, use, 0)
    assert counts.tolist() == [1, 0] and levels.tolist() == [0]
    assert markers[0].sum() == 9 and markers[1].sum() == 0


@pytest.mark.parametrize("use", ENGINES)
def test_watershed_divides_two_peaks_at_the_saddle(use):
    x = np.arange(41, dtype=float)
    ridge = np.exp(-((x - 10) ** 2) / 20) + 0.8 * np.exp(-((x - 30) ** 2) / 20)
    field = np.broadcast_to(ridge, (1, 5, 41)).copy()
    markers = np.zeros(field.shape, dtype=np.int32)
    markers[0, 2, 10], markers[0, 2, 30] = 1, 2
    everywhere = np.ones(field.shape, dtype=np.uint8)
    out = _cells.watershed(field, markers, everywhere, 4, use, 1)
    saddle = 10 + int(np.argmin(ridge[10:31]))
    # every row holds the first label up to the saddle and the second after it
    assert (out[0, :, :saddle] == 1).all() and (out[0, :, saddle + 1 :] == 2).all()
    # outside the mask nothing is labelled; a marker outside the mask is dropped
    low = np.zeros(field.shape, dtype=np.uint8)
    low[0, 2, 5:15] = 1
    out = _cells.watershed(field, markers, low, 4, use, 1)
    assert (out[0, 2, 5:15] == 1).all() and out.sum() == 10
    # NaN is outside, too
    field[0, 2, 12] = np.nan
    out = _cells.watershed(field, markers, everywhere, 4, use, 1)
    assert out[0, 2, 12] == 0 and out[0, 2, 13] == 1  # reached around the gap


@compiled_only
@pytest.mark.parametrize("connectivity", [4, 8])
def test_engines_agree_on_random_fields(connectivity):
    field = smooth_field(7)
    thresholds = np.array([0.0, 0.5, 1.0, 1.5])
    kernel = _cells.features(field, thresholds, 5, connectivity, True, 3)
    twin = _cells.features(field, thresholds, 5, connectivity, False, 1)
    for a, b in zip(kernel, twin):
        np.testing.assert_array_equal(a, b)
    markers = kernel[0]
    mask = field > 0.0
    np.testing.assert_array_equal(
        _cells.watershed(field, markers, mask, connectivity, True, 3),
        _cells.watershed(field, markers, mask, connectivity, False, 1),
    )


@pytest.mark.parametrize("use", ENGINES)
def test_region_sums(use):
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[:, :2, :2] = 1
    labels[:, 2:, 2:] = 2
    values = np.arange(32, dtype=float).reshape(1, 2, 4, 4).repeat(2, axis=0)
    values[1] = values[0] * 2.0
    values[0, 0, 0, 0] = np.nan
    total, count, high, low = _cells.region_sums(labels, values, 3, use, 2)
    assert total.shape == (2, 2, 3)
    assert count[0, 0].tolist() == [3, 4, 0] and count[1, 0].tolist() == [4, 4, 0]
    assert total[1, 0, 0] == 2 * (0 + 1 + 4 + 5)
    assert high[0, 0, 1] == 15 and low[0, 0, 0] == 1
    assert np.isnan(high[0, 0, 2]) and np.isnan(low[0, 0, 2])


@pytest.mark.parametrize("use", ENGINES)
def test_overlap_counts_after_a_shift(use):
    l0 = np.zeros((2, 6, 8), dtype=np.int32)
    l1 = np.zeros((2, 6, 8), dtype=np.int32)
    l0[:, 1:3, 1:4] = 1  # six cells
    l0[:, 4:6, 5:8] = 2  # six cells
    l1[0, 1:3, 3:6] = 1  # the first moved two to the east
    l1[0, 4:6, 5:8] = 2
    l1[1, 2:4, 1:4] = 1  # the first moved one to the north (row +1)
    n = np.array([2, 2], dtype=np.int32)
    n1 = np.array([2, 1], dtype=np.int32)
    dy = np.array([[[0]], [[1]]], dtype=np.int32)
    dx = np.array([[[2]], [[0]]], dtype=np.int32)
    counts = _cells.overlap(l0, l1, n, n1, dy, dx, use, 1)
    assert counts.size == 4 + 2
    first = counts[:4].reshape(2, 2)
    # (the second cell moves two to the east and leaves the grid, then lands outside)
    assert first[0, 0] == 6 and first[0, 1] == 0 and first[1, 0] == 0
    second = counts[4:].reshape(2, 1)
    assert second[0, 0] == 6 and second[1, 0] == 0
    # a shift per pixel: the first cell moves, the second stays
    dy = np.zeros((2, 6, 8), dtype=np.int32)
    dx = np.zeros((2, 6, 8), dtype=np.int32)
    dx[0, 1:3, 1:4] = 2
    counts = _cells.overlap(l0[:1], l1[:1], n[:1], n1[:1], dy[:1], dx[:1], use, 1)
    assert counts.reshape(2, 2).tolist() == [[6, 0], [0, 6]]


@pytest.mark.parametrize("use", ENGINES)
def test_overlap_of_no_pairs(use):
    empty = np.zeros((0, 4, 4), dtype=np.int32)
    out = _cells.overlap(
        empty,
        empty,
        empty[:, 0, 0],
        empty[:, 0, 0],
        empty[:, :1, :1],
        empty[:, :1, :1],
        use,
        1,
    )
    assert out.size == 0


def test_connectivity_is_checked():
    with pytest.raises(ValueError, match="connectivity"):
        _cells.check_connectivity(6)
    assert _cells.check_connectivity(4) == 4


@compiled_only
def test_kernel_arguments_are_checked():
    from radarx.retrieve import _objects

    mask = np.ones((1, 3, 3), dtype=np.uint8)
    field = np.ones((1, 3, 3))
    markers = np.ones((1, 3, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="connectivity"):
        _objects.label(mask, 5)
    with pytest.raises(ValueError, match="3-D"):
        _objects.label(mask[0], 8)
    with pytest.raises(ValueError, match="connectivity"):
        _objects.features(field, np.array([0.0]), 1, 3)
    with pytest.raises(ValueError, match="3-D"):
        _objects.features(field[0], np.array([0.0]), 1, 8)
    with pytest.raises(ValueError, match="threshold"):
        _objects.features(field, np.array([]), 1, 8)
    with pytest.raises(ValueError, match="connectivity"):
        _objects.watershed(field, markers, mask, 7)
    with pytest.raises(ValueError, match="3-D"):
        _objects.watershed(field[0], markers, mask, 8)
    with pytest.raises(ValueError, match="shape"):
        _objects.watershed(field, markers[:, :2], mask, 8)
    with pytest.raises(ValueError, match="dimensions"):
        _objects.region_sums(markers[0], field[None], 1)
    with pytest.raises(ValueError, match="shape"):
        _objects.region_sums(markers[:, :2], field[None], 1)
    z = np.zeros((1, 1, 1), dtype=np.int32)
    n = np.array([1], dtype=np.int32)
    with pytest.raises(ValueError, match="dimensions"):
        _objects.overlap(markers[0], markers, n, n, z, z)
    with pytest.raises(ValueError, match="inconsistent"):
        _objects.overlap(markers, markers[:, :2], n, n, z, z)
    with pytest.raises(ValueError, match="dy and dx"):
        _objects.overlap(
            markers,
            markers,
            n,
            n,
            np.zeros((1, 2, 1), dtype=np.int32),
            np.zeros((1, 2, 1), dtype=np.int32),
        )


@pytest.mark.parametrize("use", ENGINES)
def test_smoothing_is_the_gaussian_filter_of_scipy(use):
    from scipy import ndimage

    field = smooth_field(3, shape=(3, 30, 37), sigma=1.0)
    for sigma in (0.7, 1.0, 2.5):
        out = _cells.smooth(field, sigma, use, 2)
        expected = ndimage.gaussian_filter(field, (0, sigma, sigma), mode="nearest")
        np.testing.assert_allclose(out, expected, atol=1e-12)
    # a constant field stays constant, also at the edges
    flat = np.full((1, 9, 9), 4.0)
    np.testing.assert_allclose(_cells.smooth(flat, 1.5, use, 1), 4.0)


@compiled_only
def test_smoothing_arguments_are_checked():
    from radarx.retrieve import _objects

    with pytest.raises(ValueError, match="3-D"):
        _objects.smooth(np.ones((3, 3)), 1.0)
    with pytest.raises(ValueError, match="sigma"):
        _objects.smooth(np.ones((1, 3, 3)), 0.0)
