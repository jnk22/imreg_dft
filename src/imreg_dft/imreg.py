# imreg.py

# Copyright (c) 2014-?, Matěj Týč
# Copyright (c) 2011-2014, Christoph Gohlke
# Copyright (c) 2011-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
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

"""FFT based image registration. --- main functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NoReturn

import numpy as np
import scipy.ndimage as ndi

try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    from numpy import fft

from imreg_dft import utils

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from imreg_dft.reporting import ReportsWrapper


def _logpolar_filter(shape: tuple[int, int]):
    """Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    val = np.pi / 2.0
    linspace_yx = (np.linspace(-val, val, i) for i in reversed(shape[:2]))
    grid = np.array(np.meshgrid(*linspace_yx))

    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt((grid**2).sum(axis=0))
    filt = np.sin(rads) ** 2

    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[rads > val] = 1
    return filt


def _get_pcorr_shape(shape: tuple[float, float]) -> tuple[int, int]:
    return (s := int(max(shape)), s)


def _get_ang_scale(
    ims: Sequence[NDArray],
    bgval: float,
    exponent: float | str = "inf",
    constraints: dict[str, Any] | None = None,
    reports: ReportsWrapper | None = None,
) -> tuple[float, float]:
    """Given two images, return their scale and angle difference.

    Args:
        ims (2-tuple-like of 2D ndarrays): The images
        bgval: We also pad here in the :func:`map_coordinates`
        exponent (float or 'inf'): The exponent stuff, see :func:`similarity`
        constraints (dict, optional)
        reports (optional)

    Returns:
        tuple: Scale, angle. Describes the relationship of
        the subject image to the first one.

    """
    assert len(ims) == 2, "Only two images are supported as input"
    shape = ims[0].shape

    ims_apod = [utils._apodize(im) for im in ims]
    dfts = [fft.fftshift(fft.fft2(im)) for im in ims_apod]
    filt = _logpolar_filter(shape)
    dfts = [dft * filt for dft in dfts]

    # High-pass filtering used to be here, but we have moved it to a higher
    # level interface

    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    stuffs = [_logpolar(np.abs(dft), pcorr_shape, log_base) for dft in dfts]

    (arg_ang, arg_rad), success = _phase_correlation(
        stuffs[0],
        stuffs[1],
        utils.argmax_angscale,
        log_base,
        exponent,
        constraints,
        reports,
    )

    angle = -np.pi * arg_ang / float(pcorr_shape[0])
    angle = np.rad2deg(angle)
    angle = utils.wrap_angle(angle, 360)
    scale = log_base**arg_rad

    angle = -angle
    scale = 1.0 / scale

    if reports is not None:
        reports["shape"] = filt.shape
        reports["base"] = log_base

        if reports.show("spectra"):
            reports["dfts_filt"] = dfts
        if reports.show("inputs"):
            reports["ims_filt"] = [fft.ifft2(np.fft.ifftshift(dft)) for dft in dfts]
        if reports.show("logpolar"):
            reports["logpolars"] = stuffs

        if reports.show("scale_angle"):
            reports["amas-result-raw"] = (arg_ang, arg_rad)
            reports["amas-result"] = (scale, angle)
            reports["amas-success"] = success
            extent_el = pcorr_shape[1] / 2.0
            reports["amas-extent"] = (
                log_base ** (-extent_el),
                log_base**extent_el,
                -90,
                90,
            )

    if not 0.5 < scale < 2:
        msg = f"Images are not compatible. Scale change {scale:g} too big to be true."
        raise ValueError(msg)

    return scale, angle


def translation(
    im0: NDArray,
    im1: NDArray,
    filter_pcorr: int = 0,
    odds: float = 1,
    constraints: dict[str, Any] | None = None,
    reports: ReportsWrapper | None = None,
) -> dict[str, NDArray | float | int]:
    """Return translation vector to register images.
    It tells how to translate the im1 to get im0.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        filter_pcorr (int): Radius of the minimum spectrum filter
            for translation detection, use the filter when detection fails.
            Values > 3 are likely not useful.
        constraints (dict or None): Specify preference of seeked values.
            For more detailed documentation, refer to :func:`similarity`.
            The only difference is that here, only keys ``tx`` and/or ``ty``
            (i.e. both or any of them or none of them) are used.
        odds (float): The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.
            The value 1 is neutral, the converse of 2 is 1 / 2 etc.

    Returns:
        dict: Contains following keys: ``angle``, ``tvec`` (Y, X),
            and ``success``.

    """
    angle = 0
    report_one = report_two = None
    if reports is not None and reports.show("translation"):
        report_one = reports.copy_empty()
        report_two = reports.copy_empty()

    # We estimate translation for the original image...
    tvec, succ = _translation(im0, im1, filter_pcorr, constraints, report_one)
    # ... and for the 180-degrees rotated image (the rotation estimation
    # doesn't distinguish rotation of x vs x + 180deg).
    tvec2, succ2 = _translation(
        im0, utils.rot180(im1), filter_pcorr, constraints, report_two
    )

    pick_rotated = succ2 * odds > succ or odds == -1
    if reports is not None and reports.show("translation"):
        reports["t0-orig"] = report_one["amt-orig"]
        reports["t0-postproc"] = report_one["amt-postproc"]
        reports["t0-success"] = succ
        reports["t0-tvec"] = tuple(tvec)

        reports["t1-orig"] = report_two["amt-orig"]
        reports["t1-postproc"] = report_two["amt-postproc"]
        reports["t1-success"] = succ2
        reports["t1-tvec"] = tuple(tvec2)

    if reports is not None and reports.show("transformed"):
        toapp = [
            transform_img(utils.rot180(im1), tvec=tvec2, mode="wrap", order=3),
            transform_img(im1, tvec=tvec, mode="wrap", order=3),
        ]
        if pick_rotated:
            toapp.reverse()
        reports["after_tform"].extend(toapp)

    if pick_rotated:
        tvec = tvec2
        succ = succ2
        angle += 180

    return {"tvec": tvec, "success": succ, "angle": angle}


def _get_precision(shape: tuple[int, int], scale: float = 1) -> tuple[float, float]:
    """Given the parameters of the log-polar transform, get width of the interval
    where the correct values are.

    Args:
        shape (tuple): Shape of images
        scale (float): The scale difference (precision varies)

    """
    pcorr_shape = _get_pcorr_shape(shape)
    log_base = _get_log_base(shape, pcorr_shape[1])
    # * 0.5 <= max deviation is half of the step
    # * 0.25 <= we got subpixel precision now and 0.5 / 2 == 0.25
    # sccale: Scale deviation depends on the scale value
    Dscale = scale * (log_base - 1) * 0.25
    # angle: Angle deviation is constant
    Dangle = 180.0 / pcorr_shape[0] * 0.25
    return Dangle, Dscale


def _similarity(
    im0: NDArray,
    im1: NDArray,
    numiter: int = 1,
    order: int = 3,
    constraints: dict[str, Any] | None = None,
    filter_pcorr: int = 0,
    exponent: float | str = "inf",
    bgval: float | None = None,
    reports: ReportsWrapper | None = None,
) -> dict[str, Any]:
    """This function takes some input and returns mutual rotation, scale
    and translation.
    It does these things during the process:

    * Handles correct constraints handling (defaults etc.).
    * Performs angle-scale determination iteratively.
      This involves keeping constraints in sync.
    * Performs translation determination.
    * Calculates precision.

    Returns:
        Dictionary with results.

    """
    if bgval is None:
        bgval = utils.get_borderval(im1, 5)

    shape = im0.shape
    if shape != im1.shape:
        msg = "Images must have same shapes."
        raise ValueError(msg)
    if im0.ndim != 2:
        msg = "Images must be 2-dimensional."
        raise ValueError(msg)

    # We are going to iterate and precise scale and angle estimates
    scale = 1.0
    angle = 0.0
    im2 = im1

    constraints_default = {"angle": [0, None], "scale": [1, None]}
    if constraints is None:
        constraints = constraints_default

    # We guard against case when caller passes only one constraint key.
    # Now, the provided ones just replace defaults.
    constraints_default |= constraints
    constraints = constraints_default

    # During iterations, we have to work with constraints too.
    # So we make the copy in order to leave the original intact
    constraints_dynamic = constraints.copy()
    constraints_dynamic["scale"] = list(constraints["scale"])
    constraints_dynamic["angle"] = list(constraints["angle"])

    if reports is not None and reports.show("transformed"):
        reports["after_tform"] = [im2.copy()]

    for _ in range(numiter):
        newscale, newangle = _get_ang_scale(
            [im0, im2], bgval, exponent, constraints_dynamic, reports
        )
        scale *= newscale
        angle += newangle

        constraints_dynamic["scale"][0] /= newscale
        constraints_dynamic["angle"][0] -= newangle

        im2 = transform_img(im1, scale, angle, bgval=bgval, order=order)

        if reports is not None and reports.show("transformed"):
            reports["after_tform"].append(im2.copy())

    # Here we look how is the turn-180
    target, stdev = constraints.get("angle", (0, None))
    odds = _get_odds(angle, target, stdev)

    # now we can use pcorr to guess the translation
    res = translation(im0, im2, filter_pcorr, odds, constraints, reports)

    # The log-polar transform may have got the angle wrong by 180 degrees.
    # The phase correlation can help us to correct that
    angle += res["angle"]
    res["angle"] = utils.wrap_angle(angle, 360)

    # don't know what it does, but it alters the scale a little bit
    # scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    d_angle, d_scale = _get_precision(shape, scale)

    res["scale"] = scale
    res["Dscale"] = d_scale
    res["Dangle"] = d_angle
    # 0.25 because we go subpixel now
    res["Dt"] = 0.25

    return res


def similarity(
    im0: NDArray,
    im1: NDArray,
    numiter: int = 1,
    order: int = 3,
    constraints: dict[str, Any] | None = None,
    filter_pcorr: int = 0,
    exponent: float | str = "inf",
    reports: ReportsWrapper | None = None,
) -> dict[str, Any]:
    """Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Args:
        im0 (2D numpy array): The first (template) image
        im1 (2D numpy array): The second (subject) image
        numiter (int): How many times to iterate when determining scale and
            rotation
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc.
        filter_pcorr (int): Radius of a spectrum filter for translation
            detection
        exponent (float or 'inf'): The exponent value used during processing.
            Refer to the docs for a thorough explanation. Generally, pass "inf"
            when feeling conservative. Otherwise, experiment, values below 5
            are not even supposed to work.
        constraints (dict or None): Specify preference of seeked values.
            Pass None (default) for no constraints, otherwise pass a dict with
            keys ``angle``, ``scale``, ``tx`` and/or ``ty`` (i.e. you can pass
            all, some of them or none of them, all is fine). The value of a key
            is supposed to be a mutable 2-tuple (e.g. a list), where the first
            value is related to the constraint center and the second one to
            softness of the constraint (the higher is the number,
            the more soft a constraint is).

            More specifically, constraints may be regarded as weights
            in form of a shifted Gaussian curve.
            However, for precise meaning of keys and values,
            see the documentation section :ref:`constraints`.
            Names of dictionary keys map to names of command-line arguments.

    Returns:
        dict: Contains following keys: ``scale``, ``angle``, ``tvec`` (Y, X),
        ``success`` and ``timg`` (the transformed subject image)

    .. note:: There are limitations

        * Scale change must be less than 2.
        * No subpixel precision (but you can use *resampling* to get
          around this).

    """
    bgval = utils.get_borderval(im1, 5)

    res = _similarity(
        im0, im1, numiter, order, constraints, filter_pcorr, exponent, bgval, reports
    )

    im2 = transform_img_dict(im1, res, bgval, order)
    # Order of mask should be always 1 - higher values produce strange results.
    imask = transform_img_dict(np.ones_like(im1), res, 0, 1)
    # This removes some weird artifacts
    imask[imask > 0.8] = 1.0

    # Framing here = just blending the im2 with its BG according to the mask
    res["timg"] = utils.frame_img(im2, imask, 10)

    return res


def _get_odds(angle: float, target: float, stdev: float | None) -> float:
    """Determine whether we are more likely to choose the angle, or angle + 180°.

    Args:
        angle (float, degrees): The base angle.
        target (float, degrees): The angle we think is the right one.
            Typically, we take this from constraints.
        stdev (float, degrees): The relevance of the target value.
            Also typically taken from constraints.

    Return:
        float: The greater the odds are, the higher is the preferrence
            of the angle + 180 over the original angle. Odds of -1 are the same
            as inifinity.

    """
    if stdev is None:
        return 1

    angles = np.array((ta := target - angle, ta + 180.0))
    diffs = np.abs(utils.wrap_angle(angles, 360))

    if stdev <= 0:
        return -1 if diffs[0] >= diffs[1] else 0

    odds = np.exp(-(diffs**2) / stdev**2)

    if odds[0] == 0:
        return -1 if (diffs[0] >= diffs[1] or odds[1] > 0) else 0

    return odds[1] / odds[0]


def _translation(
    im0: NDArray,
    im1: NDArray,
    filter_pcorr: int = 0,
    constraints: dict[str, Any] | None = None,
    reports: ReportsWrapper | None = None,
) -> tuple[NDArray, float]:
    """The plain wrapper for translation phase correlation, no big deal."""
    return _phase_correlation(
        im0, im1, utils.argmax_translation, filter_pcorr, constraints, reports
    )


def _phase_correlation(
    im0: NDArray, im1: NDArray, callback: Callable | None = None, *args: Any
) -> tuple[NDArray, float]:
    """Computes phase correlation between im0 and im1.

    Args:
        im0
        im1
        callback (function): Process the cross-power spectrum (i.e. choose
            coordinates of the best element, usually of the highest one).
            Defaults to :func:`imreg_dft.utils.argmax2D`

    Returns:
        tuple: The translation vector (Y, X). Translation vector of (0, 0)
            means that the two images match.

    """
    if callback is None:
        callback = utils._argmax2D

    # TODO: Implement some form of high-pass filtering of PHASE correlation
    f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
    # spectrum can be filtered (already),
    # so we have to take precaution against dividing by 0
    eps = abs(f1).max() * 1e-15
    # cps == cross-power spectrum of im0 and im1
    cps = abs(fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
    # scps = shifted cps
    scps = fft.fftshift(cps)

    (t0, t1), success = callback(scps, *args)
    ret = np.array((t0, t1))

    # _compensate_fftshift is not appropriate here, this is OK.
    t0 -= f0.shape[0] // 2
    t1 -= f0.shape[1] // 2

    ret -= np.array(f0.shape, int) // 2
    return ret, success


def transform_img_dict(
    img: NDArray,
    tdict: dict[str, Any],
    bgval: float | None = None,
    order: int = 1,
    invert: bool = False,
) -> NDArray:
    """Wrapper of :func:`transform_img`, works well with the :func:`similarity`
    output.

    Args:
        img
        tdict (dictionary): Transformation dictionary --- supposed to contain
            keys "scale", "angle" and "tvec"
        bgval
        order
        invert (bool): Whether to perform inverse transformation --- doesn't
            work very well with the translation.

    Returns:
        np.ndarray: .. seealso:: :func:`transform_img`

    """
    scale = tdict["scale"]
    angle = tdict["angle"]
    tvec = np.array(tdict["tvec"])
    if invert:
        scale = 1.0 / scale
        angle *= -1
        tvec *= -1
    return transform_img(img, scale, angle, tvec, bgval=bgval, order=order)


def transform_img(
    img: NDArray,
    scale: float = 1.0,
    angle: float = 0.0,
    tvec: tuple[int, int] | NDArray = (0, 0),
    mode: str = "constant",
    bgval: float | None = None,
    order: int = 1,
) -> NDArray:
    """Return translation vector to register images.

    Args:
        img (2D or 3D numpy array): What will be transformed.
            If a 3D array is passed, it is treated in a manner in which RGB
            images are supposed to be handled - i.e. assume that coordinates
            are (Y, X, channels).
            Complex images are handled in a way that treats separately
            the real and imaginary parts.
        scale (float): The scale factor (scale > 1.0 means zooming in)
        angle (float): Degrees of rotation (clock-wise)
        tvec (2-tuple): Pixel translation vector, Y and X component.
        mode (string): The transformation mode (refer to e.g.
            :func:`scipy.ndimage.shift` and its kwarg ``mode``).
        bgval (float): Shade of the background (filling during transformations)
            If None is passed, :func:`imreg_dft.utils.get_borderval` with
            radius of 5 is used to get it.
        order (int): Order of approximation (when doing transformations). 1 =
            linear, 3 = cubic etc. Linear works surprisingly well.

    Returns:
        np.ndarray: The transformed img, may have another
        i.e. (bigger) shape than the source.

    """
    if img.ndim == 3:
        # A bloody painful special case of RGB images
        ret = np.empty_like(img)
        for idx in range(img.shape[2]):
            sli = (slice(None), slice(None), idx)
            ret[sli] = transform_img(img[sli], scale, angle, tvec, mode, bgval, order)
        return ret
    if np.iscomplexobj(img):
        decomposed = np.empty((*img.shape, 2), float)
        decomposed[:, :, 0] = img.real
        decomposed[:, :, 1] = img.imag
        # The bgval makes little sense now, as we decompose the image
        res = transform_img(decomposed, scale, angle, tvec, mode, None, order)
        return res[:, :, 0] + 1j * res[:, :, 1]

    if bgval is None:
        bgval = utils.get_borderval(img)

    bigshape = np.round(np.array(img.shape) * 1.2).astype(int)
    bg = np.zeros(bigshape, img.dtype) + bgval

    dest0 = utils.embed_to(bg, img.copy())
    # TODO: We have problems with complex numbers
    # that are not supported by zoom(), rotate() or shift()
    if scale != 1.0:
        dest0 = ndi.zoom(dest0, scale, order=order, mode=mode, cval=bgval)
    if angle != 0.0:
        dest0 = ndi.rotate(dest0, angle, order=order, mode=mode, cval=bgval)

    if tvec[0] != 0 or tvec[1] != 0:
        dest0 = ndi.shift(dest0, tvec, order=order, mode=mode, cval=bgval)

    bg = np.zeros_like(img) + bgval
    return utils.embed_to(bg, dest0)


def similarity_matrix(
    scale: float, angle: float, vector: NDArray | Sequence[float]
) -> NoReturn:
    """Return homogeneous transformation matrix from similarity parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector (of size 2).

    The order of transformations is: scale, rotate, translate.

    """
    msg = "We have no idea what this is supposed to do"
    raise NotImplementedError(msg)


EXCESS_CONST: Final = 1.1


def _get_log_base(shape: tuple[int, int], new_r: int) -> float:
    r"""Basically common functionality of :func:`_logpolar`
    and :func:`_get_ang_scale`.

    This value can be considered fixed, if you want to mess with the logpolar
    transform, mess with the shape.

    Args:
        shape: Shape of the original image.
        new_r (float): The r-size of the log-polar transform array dimension.

    Returns:
        float: Base of the log-polar transform.
        The following holds:
        :math:`log\_base = \exp( \ln [ \mathit{spectrum\_dim} ] / \mathit{loglpolar\_scale\_dim} )`,
        or the equivalent :math:`log\_base^{\mathit{loglpolar\_scale\_dim}} = \mathit{spectrum\_dim}`.

    """
    # The highest radius we have to accomodate is 'old_r',
    # However, we cut some parts out as only a thin part of the spectra has
    # these high frequencies
    old_r = shape[0] * EXCESS_CONST
    # We are radius, so we divide the diameter by two.
    old_r /= 2.0
    # we have at most 'new_r' of space.
    return np.exp(np.log(old_r) / new_r)


def _logpolar(
    image: NDArray, shape: tuple[int, int], log_base: float, bgval: float | None = None
) -> NDArray:
    """Return log-polar transformed image
    Takes into account anisotropicity of the freq spectrum
    of rectangular images.

    Args:
        image: The image to be transformed
        shape: Shape of the transformed image
        log_base: Parameter of the transformation, get it via
            :func:`_get_log_base`
        bgval: The backround value. If None, use minimum of the image.

    Returns:
        The transformed image

    """
    if bgval is None:
        bgval = np.percentile(image, 1).item()
    imshape = np.array(image.shape)
    center = imshape[0] / 2.0, imshape[1] / 2.0
    # 0 .. pi = only half of the spectrum is used
    theta = utils._get_angles(shape)
    radius_x = utils._get_lograd(shape, log_base)
    radius_y = radius_x.copy()
    ellipse_coef = imshape[0] / float(imshape[1])
    # We have to acknowledge that the frequency spectrum can be deformed
    # if the image aspect ratio is not 1.0
    # The image is x-thin, so we acknowledge that the frequency spectra
    # scale in x is shrunk.
    radius_x /= ellipse_coef

    y = radius_y * np.sin(theta) + center[0]
    x = radius_x * np.cos(theta) + center[1]
    output = np.empty_like(y)
    ndi.map_coordinates(
        image, [y, x], output=output, order=3, mode="constant", cval=bgval
    )
    return output


def imshow(
    im0: NDArray,
    im1: NDArray,
    im2: NDArray,
    cmap: str | None = None,
    fig: Figure | None = None,
    **kwargs: Any,
) -> Figure:
    """Plot images using matplotlib.
    Opens a new figure with four subplots:

    ::

      +----------------------+---------------------+
      |                      |                     |
      |   <template image>   |   <subject image>   |
      |                      |                     |
      +----------------------+---------------------+
      | <difference between  |                     |
      |  template and the    |<transformed subject>|
      | transformed subject> |                     |
      +----------------------+---------------------+

    Args:
        im0 (np.ndarray): The template image
        im1 (np.ndarray): The subject image
        im2: The transformed subject --- it is supposed to match the template
        cmap (optional): colormap
        fig (optional): The figure you would like to have this plotted on

    Returns:
        matplotlib figure: The figure with subplots

    """
    from matplotlib import pyplot as plt

    if fig is None:
        fig = plt.figure()
    if cmap is None:
        cmap = "coolwarm"
    # We do the difference between the template and the result now
    # To increase the contrast of the difference, we norm images according
    # to their near-maximums
    norm = np.percentile(np.abs(im2), 99.5) / np.percentile(np.abs(im0), 99.5)
    # Divide by zero is OK here
    phase_norm = np.median(np.angle(im2 / im0) % (2 * np.pi))
    if phase_norm != 0:
        norm *= np.exp(1j * phase_norm)
    im3 = abs(im2 - im0 * norm)
    pl0 = fig.add_subplot(221)
    pl0.imshow(im0.real, cmap, **kwargs)
    pl0.grid()
    share = {"sharex": pl0, "sharey": pl0}
    pl = fig.add_subplot(222, **share)
    pl.imshow(im1.real, cmap, **kwargs)
    pl.grid()
    pl = fig.add_subplot(223, **share)
    pl.imshow(im3, cmap, **kwargs)
    pl.grid()
    pl = fig.add_subplot(224, **share)
    pl.imshow(im2.real, cmap, **kwargs)
    pl.grid()
    return fig
