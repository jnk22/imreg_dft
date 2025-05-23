# utils.py

# Copyright (c) 2014-?, Matěj Týč
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""FFT based image registration. --- utility functions."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.ndimage as ndi
from numpy import fft, intp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from imreg_dft.reporting import ReportsWrapper


def wrap_angle(angles: float | NDArray, ceil: float = 2 * np.pi) -> float | NDArray:
    """Args:
    angles (float or NDArray, unit depends on kwarg ``ceil``)
    ceil (float): Turnaround value.

    """
    angles += ceil / 2.0
    angles %= ceil
    angles -= ceil / 2.0
    return angles


def rot180(arr: NDArray) -> NDArray:
    """Rotate the input array over 180°."""
    return np.rot90(arr, 2)


def _get_angles(shape: tuple[int, int]) -> NDArray:
    """In the log-polar spectrum, the (first) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective angle.
    """
    zeros = np.zeros(shape, dtype=np.float64)
    return zeros - np.linspace(0, np.pi, shape[0], endpoint=False)[:, None]


def _get_lograd(shape: tuple[int, int], log_base: float) -> NDArray:
    """In the log-polar spectrum, the (second) coord corresponds to an angle.
    This function returns a mapping of (the two) coordinates
    to the respective scale.

    Returns:
        2D np.ndarray of shape ``shape``, -1 coord contains scales
            from 0 to log_base ** (shape[1] - 1)

    """
    zeros = np.zeros(shape, dtype=np.float64)
    return zeros + np.power(log_base, np.arange(shape[1], dtype=float))[None, :]


def _get_constraint_mask(
    shape: tuple[int, int],
    log_base: float,
    constraints: dict[str, Any] | None = None,
) -> NDArray:
    """Prepare mask to apply to constraints to a cross-power spectrum."""
    if constraints is None:
        constraints = {}

    mask = np.ones(shape, float)

    # Here, we create masks that modulate picking the best correspondence.
    # Generally, we look at the log-polar array and identify mapping of
    # coordinates to values of quantities.
    if "scale" in constraints:
        scale, sigma = constraints["scale"]
        scales = fft.ifftshift(_get_lograd(shape, log_base))
        # vvv This issome kind of transformation of result of _get_lograd
        # vvv (log radius in pixels) to the linear scale.
        scales *= log_base ** (-shape[1] / 2.0)
        # This makes the scales array low near where scales is near 'scale'
        scales -= 1.0 / scale
        if sigma == 0:
            # there isn't: ascales = np.abs(scales - scale)
            # because scales are already low for values near 'scale'
            ascales = np.abs(scales)
            scale_min = ascales.min()
            mask[ascales > scale_min] = 0
        elif sigma is not None:
            mask *= np.exp(-(scales**2) / sigma**2)

    if "angle" in constraints:
        angle, sigma = constraints["angle"]
        angles = _get_angles(shape)
        # We flip the sign on purpose
        # TODO: ^^^ Why???
        angles += np.deg2rad(angle)
        # TODO: Check out the wrapping. It may be tricky since pi+1 != 1
        angles = wrap_angle(angles, np.pi)
        angles = np.rad2deg(angles)
        if sigma == 0:
            aangles = np.abs(angles)
            angle_min = aangles.min()
            mask[aangles > angle_min] = 0
        elif sigma is not None:
            mask *= np.exp(-(angles**2) / sigma**2)

    return fft.fftshift(mask)


def argmax_angscale(
    array: NDArray,
    log_base: float,
    exponent: str,
    constraints: dict[str, Any] | None = None,
    reports: ReportsWrapper | None = None,
) -> tuple[NDArray, float]:
    """Given a power spectrum, we choose the best fit.

    The power spectrum is treated with constraint masks and then
    passed to :func:`_argmax_ext`.
    """
    mask = _get_constraint_mask(array.shape, log_base, constraints)
    array_orig = array.copy()

    array *= mask
    ret = _argmax_ext(array, exponent)
    ret_final = _interpolate(array, ret)

    if reports is not None and reports.show("scale_angle"):
        reports["amas-orig"] = array_orig.copy()
        reports["amas-postproc"] = array.copy()

    success = _get_success(array_orig, tuple(ret_final), 0)
    return ret_final, success


def argmax_translation(
    array: NDArray,
    filter_pcorr: int,
    constraints: dict[str, Any] | None = None,
    reports: ReportsWrapper | None = None,
) -> tuple[NDArray, float]:
    if constraints is None:
        constraints = {"tx": (0, None), "ty": (0, None)}

    # We want to keep the original and here is obvious that
    # it won't get changed inadvertently
    array_orig = array.copy()
    if filter_pcorr > 0:
        array = ndi.minimum_filter(array, filter_pcorr)

    ashape = np.array(array.shape, int)
    mask = np.ones(ashape, float)
    # first goes Y, then X
    for dim, key in enumerate(("ty", "tx")):
        if constraints.get(key, (0, None))[1] is None:
            continue
        pos, sigma = constraints[key]
        alen = ashape[dim]
        dom = np.linspace(-alen // 2, -alen // 2 + alen, alen, endpoint=False)
        if sigma == 0:
            # generate a binary array closest to the position
            idx = np.argmin(np.abs(dom - pos))
            vals = np.zeros(dom.size)
            vals[idx] = 1.0
        else:
            vals = np.exp(-((dom - pos) ** 2) / sigma**2)
        if dim == 0:
            mask *= vals[:, np.newaxis]
        else:
            mask *= vals[np.newaxis, :]

    array *= mask

    # WE ARE FFTSHIFTED already.
    # ban translations that are too big
    aporad = (ashape // 6).min()
    mask2 = get_apofield(ashape, aporad)
    array *= mask2
    # Find what we look for
    tvec = _argmax_ext(array, "inf")
    tvec = _interpolate(array_orig, tvec)

    # If we use constraints or min filter,
    # array_orig[tvec] may not be the maximum
    success = _get_success(array_orig, tuple(tvec), 2)

    if reports is not None and reports.show("translation"):
        reports["amt-orig"] = array_orig.copy()
        reports["amt-postproc"] = array.copy()

    return tvec, success


def _get_success(
    array: NDArray,
    coord: tuple[float, float],
    radius: int = 2,
) -> float:
    """Given a coord, examine the array around it and return a number signifying
    how good is the "match".

    Args:
        radius: Get the success as a sum of neighbor of coord of this radius
        coord: Coordinates of the maximum. Float numbers are allowed
            (and converted to int inside)

    Returns:
        Success as float between 0 and 1 (can get slightly higher than 1).
        The meaning of the number is loose, but the higher the better.

    """
    coord_x = tuple(np.round(coord).astype(int))

    subarr = _get_subarr(array, coord_x, 2)

    theval = subarr.sum()
    theval2 = array[coord_x]
    # bigval = np.percentile(array, 97)
    # success = theval / bigval
    # TODO: Think this out
    return np.sqrt(theval * theval2)


def _argmax2D(array: NDArray, reports: ReportsWrapper | None = None) -> NDArray:
    """Simple 2D argmax function with simple sharpness indication."""
    amax = np.argmax(array)
    max_indices = list(np.unravel_index(amax, array.shape))

    return np.array(max_indices)


def _get_subarr(
    array: NDArray,
    center: NDArray | tuple[int, int],
    rad: int,
) -> NDArray:
    """Args:
    array (NDArray): The array to search
    center (2-tuple): The point in the array to search around
    rad (int): Search radius, no radius (i.e. get the single point)
        implies rad == 0.

    """
    dim = 1 + 2 * rad
    subarr = np.zeros((dim,) * 2)
    corner = np.array(center) - rad
    for ii in range(dim):
        yidx = corner[0] + ii
        yidx %= array.shape[0]
        for jj in range(dim):
            xidx = corner[1] + jj
            xidx %= array.shape[1]
            subarr[ii, jj] = array[yidx, xidx]
    return subarr


def _interpolate(
    array: NDArray,
    rough: NDArray | tuple[int, int],
    rad: int = 2,
) -> NDArray:
    """Returns index that is in the array after being rounded.

    The result index tuple is in each of its components between zero and the
    array's shape.
    """
    rough = np.round(rough).astype(int)
    surroundings = _get_subarr(array, rough, rad)
    com = _argmax_ext(surroundings, 1)
    offset = com - rad
    ret = rough + offset
    # similar to win.wrap, so
    # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
    # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
    # which is rounded to len() - 1. Yeah!
    ret += 0.5
    ret %= np.array(array.shape).astype(int)
    ret -= 0.5
    return ret


def _argmax_ext(array: NDArray, exponent: str | int) -> NDArray:
    """Calculate coordinates of the COM (center of mass) of the provided array.

    Args:
        array (NDArray): The array to be examined.
        exponent (float or 'inf'): The exponent we power the array with. If the
            value 'inf' is given, the coordinage of the array maximum is taken.

    Returns:
        np.ndarray: The COM coordinate tuple, float values are allowed!

    """
    # When using an integer exponent for _argmax_ext, it is good to have the
    # neutral rotation/scale in the center rather near the edges

    if exponent == "inf":
        ret = _argmax2D(array)
    else:
        col = np.arange(array.shape[0])[:, np.newaxis]
        row = np.arange(array.shape[1])[np.newaxis, :]

        arr2 = array**exponent
        arrsum = arr2.sum()
        if arrsum == 0:
            # We have to return SOMETHING, so let's go for (0, 0)
            return np.zeros(2)
        arrprody = np.sum(arr2 * col) / arrsum
        arrprodx = np.sum(arr2 * row) / arrsum
        ret = [arrprody, arrprodx]
        # We don't use it, but it still tells us about value distribution

    return np.array(ret)


def _get_emslices(
    shape1: tuple[int, int],
    shape2: NDArray | tuple[int, int],
) -> tuple[list[slice], list[slice]]:
    """Common code used by :func:`embed_to` and :func:`undo_embed`."""
    slices_from = []
    slices_to = []
    for dim1, dim2 in zip(shape1, shape2, strict=False):
        diff = dim2 - dim1
        # In fact: if diff == 0:
        slice_from = slice(None)
        slice_to = slice(None)

        # dim2 is bigger => we will skip some of their pixels
        if diff > 0:
            # diff // 2 + rem == diff
            rem = diff - (diff // 2)
            slice_from = slice(diff // 2, dim2 - rem)
        if diff < 0:
            diff *= -1
            rem = diff - (diff // 2)
            slice_to = slice(diff // 2, dim1 - rem)
        slices_from.append(slice_from)
        slices_to.append(slice_to)
    return slices_from, slices_to


def undo_embed(what: NDArray, orig_shape: NDArray | tuple[int, int]) -> NDArray:
    """Undo an embed operation.

    Args:
        what: What has once be the destination array
        orig_shape: The shape of the once original array

    Returns:
        The closest we got to the undo

    """
    _, slices_to = _get_emslices(what.shape, orig_shape)

    return what[slices_to[0], slices_to[1]].copy()


def embed_to(where: NDArray, what: NDArray) -> NDArray:
    """Given a source and destination arrays, put the source into
    the destination so it is centered and perform all necessary operations
    (cropping or aligning).

    Args:
        where: The destination array (also modified inplace)
        what: The source array

    Returns:
        The destination array

    """
    slices_from, slices_to = _get_emslices(where.shape, what.shape)

    where[slices_to[0], slices_to[1]] = what[slices_from[0], slices_from[1]]
    return where


def extend_to_3D(what: NDArray, newdim_2D: tuple[int, ...]) -> NDArray:
    """Extend 2D and 3D arrays (when being supplied with their x--y shape)."""
    assert len(newdim_2D) == 2, (
        f"You were supposed to provide 2D dimensions, got {newdim_2D}"
    )
    if what.ndim == 3:
        height = what.shape[2]
        res = np.empty((*newdim_2D, height), what.dtype)

        for dim in range(height):
            res[:, :, dim] = extend_to(what[:, :, dim], newdim_2D)
    else:
        res = extend_to(what, newdim_2D)

    return res


def extend_to(what: NDArray, newdim: NDArray | tuple[int, ...]) -> NDArray:
    """Given an image, it puts it in a (typically larger) array.
    To prevent rough edges from appearing, the containing array has a color
    that is close to the image's border color, and image edges
    smoothly blend into the background.

    Args:
        what (NDArray): What to extend
        newdim (tuple): The resulting dimension

    """
    mindim = min(what.shape)
    dst = int(mindim * 0.12)
    bgval = get_borderval(what, dst // 2)

    dest = np.zeros(newdim, what.dtype)
    res = dest.copy() + bgval
    res = embed_to(res, what)

    apofield = get_apofield(what.shape, dst)
    apoemb = embed_to(dest.copy().astype(what.dtype), apofield)

    # res is a convex combination of its previous self and the bg value
    return apoemb * res + (1 - apoemb) * bgval


def extend_by(what: NDArray, dst: int) -> NDArray:
    """Given a source array, extend it by given number of pixels and try
    to make the extension smooth (not altering the original array).
    """
    olddim = np.array(what.shape, dtype=int)
    newdim = olddim + 2 * dst

    return extend_to(what, newdim)


def unextend_by(what: NDArray, dst: int) -> NDArray:
    """Try to undo as much as the :func:`extend_by` does.
    Some things can't be undone, though.
    """
    newdim = np.array(what.shape, dtype=int)
    origdim = newdim - 2 * dst

    return undo_embed(what, origdim)


def imfilter(
    img: NDArray,
    low: tuple[float, float] | None = None,
    high: tuple[float, float] | None = None,
    cap: tuple[float, float] | None = None,
) -> NDArray:
    """Given an image, it a high-pass and/or low-pass filters on its
    Fourier spectrum.

    Args:
        img (NDArray): The image to be filtered
        low (tuple): The low-pass filter parameters, 0..1
        high (tuple): The high-pass filter parameters, 0..1
        cap (tuple): The quantile cap parameters, 0..1.
            A filtered image will have extremes below the lower quantile and
            above the upper one cut.

    Returns:
        np.ndarray: The real component of the image after filtering

    """
    dft = fft.fft2(img)

    if low is not None:
        _lowpass(dft, low[0], low[1])
    if high is not None:
        _highpass(dft, high[0], high[1])

    ret = fft.ifft2(dft)
    # if the input was a real number array, return real numbers,
    # otherwise let it be complex.
    if not np.iscomplexobj(img):
        ret = np.real(ret)

    if cap is None:
        cap = (0, 1)

    low_x, high_x = cap
    if low_x > 0.0:
        low_val = np.percentile(ret, low_x * 100.0)
        ret[ret < low_val] = low_val
    if high_x < 1.0:
        high_val = np.percentile(ret, high_x * 100.0)
        ret[ret > high_val] = high_val

    return ret


def _highpass(dft: NDArray, lo: float, hi: float) -> None:
    mask = _xpass((dft.shape), lo, hi)
    dft *= 1 - mask


def _lowpass(dft: NDArray, lo: float, hi: float) -> None:
    mask = _xpass((dft.shape), lo, hi)
    dft *= mask


def _xpass(shape: tuple[int, int], lo: float, hi: float) -> NDArray:
    """Compute a pass-filter mask with values ranging from 0 to 1.0
    The mask is low-pass, application has to be handled by a calling funcion.
    """
    assert lo <= hi, f"Filter order wrong, low '{lo:g}', high '{hi:g}'"
    assert lo >= 0, f"Low filter lower than zero ({lo:g})"
    # High can be as high as possible

    dom_x = np.fft.fftfreq(shape[0])[:, np.newaxis]
    dom_y = np.fft.fftfreq(shape[1])[np.newaxis, :]

    # freq goes 0..0.5, we want from 0..1, so we multiply it by 2.
    dom = np.sqrt(dom_x**2 + dom_y**2) * 2

    res = np.ones(dom.shape)
    res[dom >= hi] = 0.0
    mask = (dom > lo) * (dom < hi)
    res[mask] = 1 - (dom[mask] - lo) / (hi - lo)

    return res


def _apodize(
    what: NDArray, aporad: int | None = None, ratio: float | None = None
) -> NDArray:
    """Given an image, it apodizes it (so it becomes quasi-seamless).
    When ``ratio`` is None, color near the edges will converge
    to the same colour, whereas when ratio is a float number, a blurred
    original image will serve as background.

    Args:
        what: The original image
        aporad (int): Radius [px], width of the band near the edges
            that will get modified
        ratio (float or None): When None, the apodization background will
            be a flat color.
            When a float number, the background will be the image itself
            convolved with Gaussian kernel of sigma (aporad / ratio).

    Returns:
        The apodized image

    """
    if aporad is None:
        mindim = min(what.shape)
        aporad = int(mindim * 0.12)
    apofield = get_apofield(what.shape, aporad)
    res = what * apofield
    if ratio is not None:
        ratio = float(ratio)
        bg = ndi.gaussian_filter(what, aporad / ratio, mode="wrap")
    else:
        bg = get_borderval(what, aporad // 2)
    res += bg * (1 - apofield)
    return res


def get_apofield(shape: NDArray | tuple[int, int], aporad: int) -> NDArray:
    """Returns an array between 0 and 1 that goes to zero close to the edges."""
    if aporad == 0:
        return np.ones(shape, dtype=float)
    apos = np.hanning(aporad * 2)
    vecs = []
    for dim in shape:
        assert dim > aporad * 2, "Apodization radius %d too big for shape dim. %d" % (
            aporad,
            dim,
        )
        toapp = np.ones(dim)
        toapp[:aporad] = apos[:aporad]
        toapp[-aporad:] = apos[-aporad:]
        vecs.append(toapp)
    return np.outer(vecs[0], vecs[1])


# TODO: Refactor this function, the current shape looks covoluted.
def frame_img(
    img: NDArray, mask: NDArray, dst: int, apofield: NDArray | None = None
) -> NDArray:
    """Given an array, a mask (floats between 0 and 1), and a distance,
    alter the area where the mask is low (and roughly within dst from the edge)
    so it blends well with the area where the mask is high.
    The purpose of this is removal of spurious frequencies in the image's
    Fourier spectrum.

    Args:
        img (np.array): What we want to alter
        maski (np.array): The indicator what can be altered (0)
            and what can not (1)
        dst (int): Parameter controlling behavior near edges, value could be
            probably deduced from the mask.

    """
    import scipy.ndimage as ndimg

    radius = dst / 1.8

    convmask0 = mask + 1e-10

    krad_max = radius * 6
    convimg = img
    convmask = convmask0
    convimg0 = img
    krad0 = 0.8
    krad = krad0

    while krad < krad_max:
        convimg = ndimg.gaussian_filter(convimg0 * convmask0, krad, mode="wrap")
        convmask = ndimg.gaussian_filter(convmask0, krad, mode="wrap")
        convimg /= convmask

        convimg = convimg * (convmask - convmask0) + convimg0 * (
            1 - convmask + convmask0
        )
        krad *= 1.8

        convimg0 = convimg
        convmask0 = convmask

    if apofield is not None:
        ret = convimg * (1 - apofield) + img * apofield
    else:
        ret = convimg
        ret[mask >= 1] = img[mask >= 1]

    return ret


def get_borderval(img: NDArray, radius: int | None = None) -> float:
    """Given an image and a radius, examine the average value of the image
    at most radius pixels from the edge.
    """
    if radius is None:
        mindim = min(img.shape)
        radius = max(1, mindim // 20)
    mask = np.zeros_like(img, dtype=bool)
    mask[:, :radius] = True
    mask[:, -radius:] = True
    mask[:radius, :] = True
    mask[-radius:, :] = True

    return np.median(img[mask])


def slices2start(slices: list[slice]) -> NDArray:
    """Convenience function.
    Given a tuple of slices, it returns an array of their starts.
    """
    starts = (slices[0].start, slices[1].start)
    return np.array(starts)


def decompose(
    what: NDArray, outshp: tuple[float, float], coef: float
) -> list[tuple[NDArray, NDArray]]:
    """Given an array and a shape, it creates a decomposition of the array in form
    of subarrays and their respective position.

    Args:
        what (np.ndarray): The array to be decomposed
        outshp (tuple-like): The shape of decompositions

    Returns:
        list: Decomposition --- a list of tuples (subarray (np.ndarray),
        coordinate (np.ndarray))

    """
    slices = getSlices(np.array(what.shape), np.array(outshp), coef)
    return [(what[tuple(slic)], slices2start(slic)) for slic in slices]


def starts2dshape(starts: tuple[NDArray, ...]) -> tuple[int, int]:
    """Given starts of tiles, deduce the shape of the decomposition from them.

    Args:
        starts (list of ints)

    Return:
        tuple: shape of the decomposition

    """
    ncols = 1
    for start in starts[1:]:
        if start[1] == 0:
            break
        ncols += 1
    nrows = len(starts) // ncols
    assert len(starts) == nrows * ncols
    return (nrows, ncols)


def getSlices(inshp: NDArray, outshp: NDArray, coef: float) -> list[list[slice]]:
    shape = inshp
    starts = getCuts(shape, outshp, coef)
    return [mkCut(shape, outshp, start) for start in starts]


def getCuts(shp0: NDArray, shp1: NDArray, coef: float = 0.5) -> list[Any]:
    """Given an array shape, tile shape and density coefficient, return list of
    possible points of the array decomposition.

    Args:
        shp0 (np.ndarray): Shape of the big array
        shp1 (np.ndarray): Shape of the tile
        coef (float): Density coefficient --- lower means higher density and
            1.0 means no overlap, 0.5 50% overlap, 0.1 90% overlap etc.

    Returns:
        list: List of tuples (y, x) coordinates of possible tile corners.

    """
    # * coef = possible increase of density
    offsets = (shp1 * coef).astype(int)
    starts = [
        _getCut(shap0, shap1, offset)
        for shap0, shap1, offset in zip(shp0, shp1, offsets, strict=False)
    ]
    assert len(starts) == 2
    return list(product(starts[0], starts[1]))


def _getCut(big: int, small: int, offset: int) -> list[int]:
    """Given a big array length and small array length and an offset,
    output a list of starts of small arrays, so that they cover the
    big one and their offset is <= the required offset.

    Args:
        big (int): The source length array
        small (float): The small length

    Returns:
        list - list of possible start locations

    """
    # big "reduced"
    big_r = big - small
    count = int(big_r / offset)
    if count * offset != big_r:
        count += 1
    true_offset = big_r / float(count)
    assert offset >= true_offset
    begins = [int(true_offset * ii) for ii in range(count)]
    # big:   ----------------| - hidden small -
    # small: +---
    # begins:*...*...*...*..*
    begins.append(big_r)
    return begins


def mkCut(
    shp0: NDArray,
    dims: NDArray,
    start: tuple[float, float]
    | NDArray
    | tuple[int, int]
    | tuple[float, int]
    | tuple[int, float],
) -> list[slice]:
    """Make a cut from shp0 and keep the given dimensions.
    Also obey the start, but if it is not possible, shift it backwards.

    Returns:
        list - List of slices defining the subarray.

    """
    assert np.all(shp0 > dims), (
        f"The array is too small - shape {shp0} vs shape {dims} of cuts "
    )
    # If dims (or even start )are float, the resulting shape may be different
    # due to the rounding stuff.
    start = np.round(start).astype(int)
    dims = np.round(dims).astype(int)
    end = start + dims
    # The difference between end of the cut and the original shape.
    # If it is >= 0, then everything is OK.
    diff = shp0 - end
    for ii, num in enumerate(diff):
        # no-op, the end fits into our shape
        if num > 0:
            diff[ii] = 0

    # If the end would be outside, we shift both the start AND the end.
    rstart = start + diff
    rend = end + diff
    return [slice(rstart[dim], rend[dim]) for dim in range(dims.size)]


def _get_dst1(pt: NDArray, pts: NDArray) -> NDArray:
    """Given a point in 2D and vector of points, return vector of distances
    according to Manhattan metrics.
    """
    return np.max(np.abs(pts - pt), axis=1)


def get_clusters(points: NDArray, rad: float = 0) -> NDArray:
    """Given set of points and radius upper bound, return a binary matrix
    telling whether a given point is close to other points according to
    :func:`_get_dst1`.
    (point = matrix row).

    Args:
        points (np.ndarray): Shifts.
        rad (float): What is closer than ``rad`` is considered *close*.

    The result matrix has always True on diagonals.

    """
    num = len(points)
    clusters = np.zeros((num, num), bool)
    # some distances may be NaNs
    for ii, shift in enumerate(points):
        clusters[ii] = _get_dst1(shift, points) <= rad
    return clusters


def get_best_cluster(
    points: NDArray, scores: NDArray, rad: int = 0
) -> tuple[NDArray, intp]:
    """Given some additional data, choose the best cluster and the index
    of the best point in the best cluster.
    Score of a cluster is sum of scores of points in it.

    Note that the point of the best score may not be in the best cluster
    and a point may be members of multiple cluster.

    Args:
        points: Array of bools, indices that belong to the cluster are True
        scores: Rates a point by a number --- higher is better.

    """
    clusters = get_clusters(points, rad)
    cluster_scores = np.zeros(len(points))
    for ii, cluster in enumerate(clusters):
        cluster_scores[ii] = sum(cluster * scores)
    amax = np.argmax(cluster_scores)
    ret = clusters[amax]
    return ret, amax


def _ang2complex(angles: NDArray) -> NDArray:
    """Transform angle in degrees to complex phasor."""
    return np.exp(1j * np.deg2rad(angles))


def _complex2ang(cplx: complex) -> float:
    """Inversion of :func:`_ang2complex`."""
    return np.rad2deg(np.angle(cplx))


def get_values(
    cluster: NDArray, shifts: NDArray, scores: NDArray, angles: NDArray, scales: NDArray
) -> tuple[NDArray, float, float, float]:
    """Given a cluster and some vectors, return average values of the data
    in the cluster.
    Treat the angular data carefully.
    """
    weights = scores[cluster]
    s_weights = sum(weights)
    if s_weights == 0:
        # When scores are all zero, we just select the mean of proposed values
        weights = np.ones_like(weights)
    else:
        weights /= s_weights

    shift = np.sum(shifts[cluster] * weights[:, np.newaxis], axis=0)
    scale = sum(scales[cluster] * weights)
    score = sum(scores[cluster] * weights)

    angles = _ang2complex(angles)
    angle = sum(angles[cluster] * weights)
    angle = _complex2ang(angle)

    return shift, angle, scale, score
