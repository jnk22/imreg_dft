"""Tests for utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy import fft, int64
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_array_less,
    assert_equal,
)

from imreg_dft import utils

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray


@pytest.fixture
def rng() -> Generator:
    """Return default random generator with pre-defined seed."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def size_setup() -> tuple[tuple[int, int], int64]:
    """Return default array size for tests."""
    whatshape = (20, 11)
    whatsize = np.prod(whatshape)
    return whatshape, whatsize


@pytest.mark.parametrize("whs", [(20, 11), (21, 12), (22, 13), (50, 60)])
def test_undo(
    whs: tuple[int, int], size_setup: tuple[tuple[int, int], int64], rng: Generator
) -> None:
    whatshape, _ = size_setup
    what = rng.random(whatshape)

    where = np.zeros(whs)
    embd = utils.embed_to(where, what.copy())
    undone = utils.undo_embed(embd, what.shape)
    assert what.shape == undone.shape
    assert_equal(what, undone)


@pytest.mark.parametrize("dst", [2, 3, 4])
def test_extend(
    dst: int, size_setup: tuple[tuple[int, int], int64], rng: Generator
) -> None:
    whatshape, whatsize = size_setup
    what = rng.random(whatshape)
    whaty = what.shape[0]
    what[:] += np.arange(whaty, dtype=float)[:, np.newaxis] * 5 / whaty

    dftscore0 = __dft_score(what, whatsize)

    ext = utils.extend_by(what, dst)

    # Bigger distance should mean better "DFT score"
    dftscore = __dft_score(ext, whatsize)
    assert dftscore < dftscore0 * 1.1
    dftscore0 = dftscore

    undone = utils.unextend_by(ext, dst)
    assert what.shape == undone.shape
    # TODO: unextend does not work 100% precisely
    # assert_equal(what, undone)


def test_subarray() -> None:
    arr = np.arange(20)
    arr = arr.reshape((4, 5))

    # trivial subarray
    suba = utils._get_subarr(arr, (1, 1), 1)
    ret = arr[:3, :3]
    assert_equal(suba, ret)

    # subarray with zero radius
    suba = utils._get_subarr(arr, (1, 1), 0)
    ret = arr[1, 1]
    assert_equal(suba, ret)

    # subarray that wraps through two edges
    suba = utils._get_subarr(arr, (0, 0), 1)
    ret = np.zeros((3, 3), int)
    ret[1:, 1:] = arr[:2, :2]
    ret[0, 0] = arr[-1, -1]
    ret[0, 1] = arr[-1, 0]
    ret[0, 2] = arr[-1, 1]
    ret[1, 0] = arr[0, -1]
    ret[2, 0] = arr[1, -1]
    assert_equal(suba, ret)


def test_filter() -> None:
    src = np.zeros((20, 30))

    __wrap_filter(src, [(0.8, 0.8)], (0.8, 1.0))
    __wrap_filter(src, [(0.1, 0.2)], None, (0.3, 0.4))

    src2 = __add_freq(src.copy(), (0.1, 0.4))
    __wrap_filter(src2, [(0.8, 0.8), (0.1, 0.2)], (0.8, 1.0), (0.3, 0.4))


def test_argmax_ext() -> None:
    src = np.array([[1, 3, 1], [0, 0, 0], [1, 3.01, 0]])
    infres = utils._argmax_ext(src, "inf")  # element 3.01
    assert tuple(infres) == (2.0, 1.0)
    n10res = utils._argmax_ext(src, 10)  # element 1 in the rows with 3s
    n10res = np.round(n10res)
    assert tuple(n10res) == (1, 1)


def test_select() -> None:
    inshp = np.array((5, 8))

    start = np.array((0, 0))
    dim = np.array((2, 3))
    slis = utils.mkCut(inshp, dim, start)

    sliarrs = np.array([__slice2arr(sli) for sli in slis])
    assert_array_equal(sliarrs[:, 2], dim)
    assert_array_equal(sliarrs[:, 0], start)
    assert_array_equal(sliarrs[:, 1], (2, 3))

    start = np.array((3, 6))
    dim = np.array((2, 3))
    slis = utils.mkCut(inshp, dim, start)

    sliarrs = np.array([__slice2arr(sli) for sli in slis])
    assert_array_equal(sliarrs[:, 2], dim)
    assert_array_equal(sliarrs[:, 0], (3, 5))
    assert_array_equal(sliarrs[:, 1], inshp)


def test_cuts() -> None:
    big = np.array((30, 50))
    small = np.array((20, 20))
    res = utils.getCuts(big, small, 0.25)
    # first is (0, 0), second is (0, 1)
    assert res[1][1] == 5
    # Last element of the row has beginning at 40
    assert res[5][1] == 25
    assert res[6][1] == 30
    assert res[7][1] == 0
    # (50 / 5) + 1 = 11th should be (5, 5) - 2nd of the 2nd row
    assert res[8] == (5, 5)

    small = np.array((10, 20))
    res = utils.getCuts(big, small, 1.0)
    assert res[1] == (0, 15)
    assert res[2] == (0, 30)
    assert res[3] == (10, 0)
    assert res[4] == (10, 15)
    assert res[5] == (10, 30)
    assert res[6] == (20, 0)


def test_cut() -> None:
    # Tests of those private functions are ugly
    res = utils._getCut(14, 5, 3)
    assert_array_equal(res, (0, 3, 6, 9))

    res = utils._getCut(130, 50, 50)
    assert_array_equal(res, (0, 40, 80))


def test_decomps(rng: Generator) -> None:
    smallshp = (30, 50)
    inarr = rng.random(smallshp)
    recon = np.zeros_like(inarr)
    # Float tile dimensions are possible, but they may cause problems.
    # Our code should handle them well.
    coef = 0.8
    tileshp = (7.4, 6.3)
    tileshp_round = tuple(np.round(tileshp))
    decomps = utils.decompose(inarr, tileshp, coef)
    for decarr, start in decomps:
        sshp = decarr.shape
        assert tileshp_round == sshp
        recon[start[0] : start[0] + sshp[0], start[1] : start[1] + sshp[1]] = decarr
    assert_array_equal(inarr, recon)

    starts = list(zip(*decomps, strict=False))[1]
    dshape = np.array(utils.starts2dshape(starts), int)
    # vvv generic conditions decomp shape has to satisfy vvv
    # assert((dshape - 1) * tileshp * coef < smallshp)
    # assert(dshape * tileshp * coef >= smallshp)
    assert_array_equal(dshape, (6, 10))


def test_subpixel() -> None:
    anarr = np.zeros((4, 5))
    anarr[2, 3] = 1
    # The correspondence principle should hold
    first_guess = (2, 3)
    second_guess = utils._interpolate(anarr, first_guess, rad=1)
    assert_equal(second_guess, (2, 3))

    # Now something more meaningful
    anarr[2, 4] = 1
    second_guess = utils._interpolate(anarr, first_guess, rad=1)
    assert_almost_equal(second_guess, (2, 3.5))


def test_subpixel_edge() -> None:
    anarr = np.zeros((4, 5))
    anarr[3, 0] = 1
    anarr[3, 4] = 1
    first_guess = (4, 0)
    second_guess = utils._interpolate(anarr, first_guess, rad=2)
    assert_almost_equal(second_guess, (3, -0.5))

    anarr[3, 0] += 1
    anarr[0, 4] = 1
    second_guess = utils._interpolate(anarr, first_guess, rad=2)
    assert_almost_equal(second_guess, (3.25, -0.5))


def test_subpixel_crazy() -> None:
    anarr = np.zeros((4, 5))
    first_guess = (0, 0)
    second_guess = utils._interpolate(anarr, first_guess, rad=2)
    assert_array_less(second_guess, anarr.shape)


def test_clusters() -> None:
    shifts = [(0, 1), (0, 1.1), (0.2, 1), (-0.1, 0.9), (-0.1, 0.8)]
    shifts = np.array(shifts)
    clusters = utils.get_clusters(shifts, 0.11)
    cluster = clusters[0]
    assert_array_equal(cluster, (1, 1, 0, 1, 0))

    nshifts = len(shifts)
    scores = np.zeros(nshifts)
    scores[0] = 1
    angles = np.arange(nshifts)
    scales = np.ones(nshifts)
    scales[2] = np.nan

    shift, angle, scale, score = utils.get_values(
        cluster, shifts, scores, angles, scales
    )

    assert_array_equal(shift, shifts[0])
    assert angle == angles[0]
    assert scale == scales[0]
    assert score == scores[0]


def __dft_score(arr: NDArray, whatsize: int64) -> float:
    dft = fft.fft2(arr) * whatsize
    dft /= dft.size

    yfreqs = fft.fftfreq(arr.shape[0])[:, np.newaxis]
    xfreqs = fft.fftfreq(arr.shape[1])[np.newaxis, :]
    weifun = xfreqs**2 + yfreqs**2

    ret = np.abs(dft) * weifun
    return ret.sum()


def __wrap_filter(src: NDArray, vecs: list[tuple[float, float]], *args) -> None:
    dest = src.copy()
    for vec in vecs:
        __add_freq(dest, vec)

    filtered = utils.imfilter(dest, *args)
    mold, mnew = [__arr_diff(src, arr)[0] for arr in (dest, filtered)]
    assert mold * 1e-10 > mnew


def __add_freq(src: NDArray, vec: tuple[float, float]) -> NDArray:
    # Modifies 'src' array in-place.
    dom = np.zeros(src.shape)
    dom += np.arange(src.shape[0])[:, np.newaxis] * np.pi * vec[0]
    dom += np.arange(src.shape[1])[np.newaxis, :] * np.pi * vec[1]

    src += np.sin(dom)

    return src


def __arr_diff(a: NDArray, b: NDArray) -> tuple[float, float]:
    adiff = np.abs(a - b)
    return adiff.mean(), adiff.max()


def __slice2arr(sli: slice) -> NDArray:
    return np.array([sli.start, sli.stop, sli.stop - sli.start])
