# reporting.py

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

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, ItemsView, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

# We intentionally don't import matplotlib on this level - we want this module
# to be importable even if one doesn't have matplotlib

TEXT_MODE = "plain"


def _t(stri: str) -> str:
    return f"\textrm{{{stri}}}" if TEXT_MODE == "tex" else stri


@contextlib.contextmanager
def report_wrapper(
    orig: ReportsWrapper | None, index: int
) -> Generator[ReportsWrapper | None]:
    if orig is None:
        yield None
    else:
        prefix = f"{index:03d}-"
        orig.push_prefix(prefix)
        yield orig
        orig.pop_prefix(prefix)


class ReportsWrapper:
    """A wrapped dictionary.
    It allows a parent function to put it in a mode, in which it will
    prefix keys of items set.
    """

    def __init__(self, toshow: str = "") -> None:
        self.prefixes = [""]
        #: Keys by prefix
        self._stuff = {"": {}}
        self.idx = ""
        self._toshow = toshow
        self._show = {
            "inputs": "i" in toshow,
            "spectra": "s" in toshow,
            "logpolar": "l" in toshow,
            "tile_info": "t" in toshow,
            "scale_angle": "1" in toshow,
            "transformed": "a" in toshow,
            "translation": "2" in toshow,
        }

    def get_contents(self) -> ItemsView:
        return self._stuff.items()

    def copy_empty(self) -> ReportsWrapper:
        ret = ReportsWrapper(self._toshow)
        ret.idx = self.idx
        ret.prefixes = self.prefixes
        for prefix in self.prefixes:
            ret._stuff[prefix] = {}
        return ret

    def set_global(self, key: str, value: Any) -> None:
        self._stuff[""][key] = value

    def get_global(self, key: str) -> Any:
        return self._stuff[""][key]

    def show(self, *args: Any) -> Any:
        ret = False
        for arg in args:
            ret |= self._show[arg]
        return ret

    def __setitem__(self, key: str, value: Any) -> None:
        self._stuff[self.idx][key] = value

    def __getitem__(self, key: str) -> Any:
        return self._stuff[self.idx][key]

    def push_prefix(self, idx: str) -> None:
        self._stuff.setdefault(idx, {})
        self.idx = idx
        self.prefixes.append(self.idx)

    def pop_prefix(self, idx: str) -> None:
        assert self.prefixes[-1] == idx, (
            f"Real previous prefix ({self.prefixes[-1]}) differs from the specified ({idx})"
        )
        assert len(self.prefixes) > 1, (
            "There is not more than 1 prefix left, you can't remove any."
        )
        self.prefixes.pop()
        self.idx = self.prefixes[-1]


class Rect_callback:
    def __call__(self, idx: int, LLC: NDArray, dims: NDArray) -> None:
        self._call(idx, LLC, dims)

    def _call(self, idx: int, LLC: NDArray, dims: NDArray) -> None:
        raise NotImplementedError


class Rect_mpl(Rect_callback):
    """A class that can draw image tiles nicely."""

    def __init__(self, subplot: Axes, shape: tuple[int, ...]) -> None:
        self.subplot = subplot
        self.ecs = ("w", "k")
        self.shape = shape

    def _get_color(
        self, coords: tuple[int, ...], dic: dict[str, Any] | None = None
    ) -> str:
        lidx = sum(coords)
        ret = self.ecs[lidx % 2]
        if dic is not None:
            dic["ec"] = ret
        return ret

    def _call(
        self, idx: int, LLC: NDArray, dims: NDArray, special: bool = False
    ) -> None:
        from matplotlib.patches import Rectangle

        # Get from the numpy -> MPL coord system
        LLC = LLC[::-1]
        URC = LLC + np.array((dims[1], dims[0]))
        kwargs = {"fc": "none", "lw": 4, "alpha": 0.5}
        coords = np.unravel_index(idx, self.shape)
        color = self._get_color(coords, kwargs)
        if special:
            kwargs["fc"] = "w"
        rect = Rectangle(LLC, dims[1], dims[0], **kwargs)
        self.subplot.add_artist(rect)
        center = (URC + LLC) / 2.0
        self.subplot.text(
            center[0],
            center[1],
            f"{idx:02d}\n({coords[0]}, {coords[1]})",
            va="center",
            ha="center",
            color=color,
        )


def slices2rects(slices: list[list[slice]], rect_cb: Rect_callback) -> None:
    """Args:
    slices: List of slice objects
    rect_cb (callable): Check :class:`Rect_callback`.

    """
    for ii, (sly, slx) in enumerate(slices):
        llc = np.array((sly.start, slx.start))
        urc = np.array((sly.stop, slx.stop))
        dims = urc - llc
        rect_cb(ii, llc, dims)


def imshow_spectra(fig: Figure, spectra: list[NDArray]) -> Figure:
    import mpl_toolkits.axes_grid1 as axg

    dfts_filt_extent = (-1, 1, -1, 1)
    grid = axg.ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 2),
        add_all=True,
        axes_pad=0.4,
        label_mode="L",
    )
    what = ("template", "subject")
    for ii, im in enumerate(spectra):
        grid[ii].set_title(_t(f"log abs dfts - {what[ii]}"))
        im = grid[ii].imshow(
            np.log(np.abs(im)),
            cmap="viridis",
            extent=dfts_filt_extent,
        )
        grid[ii].set_xlabel(_t("2 X / px"))
        grid[ii].set_ylabel(_t("2 Y / px"))
    return fig


def imshow_logpolars(
    fig: Figure, spectra: list[NDArray], log_base: float, im_shape: tuple[int, ...]
) -> Figure:
    import mpl_toolkits.axes_grid1 as axg
    from matplotlib.ticker import ScalarFormatter

    low = 1.0
    high = log_base ** spectra[0].shape[1]
    logpolars_extent = (low, high, 0, 180)
    grid = axg.ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 1),
        add_all=True,
        aspect=False,
        axes_pad=0.4,
        label_mode="L",
    )
    ims = [np.log(np.abs(im)) for im in spectra]
    for ii, im in enumerate(ims):
        vmin = np.percentile(im, 1)
        vmax = np.percentile(im, 99)
        grid[ii].set_xscale("log", basex=log_base)
        grid[ii].get_xaxis().set_major_formatter(ScalarFormatter())
        im = grid[ii].imshow(
            im,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=logpolars_extent,
        )
        grid[ii].set_xlabel(_t("log radius"))
        grid[ii].set_ylabel(_t("azimuth / degrees"))

        xticklabels = [
            f"{tick * 2 / im_shape[0]:.3g}" for tick in grid[ii].get_xticks()
        ]

        grid[ii].set_xticklabels(
            xticklabels, rotation=40, rotation_mode="anchor", ha="right"
        )

    return fig


def imshow_plain(
    fig: Figure, images: list[NDArray], what: Sequence[str], also_common: bool = False
) -> Figure:
    import mpl_toolkits.axes_grid1 as axg

    ncols = len(images)
    nrows = 1
    if also_common:
        nrows = 2
    elif len(images) == 4:
        # not also_common and we have 4 images --- we make a grid of 2x2
        nrows = ncols = 2

    grid = axg.ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        add_all=True,
        axes_pad=0.4,
        label_mode="L",
    )
    images = [im.real for im in images]

    for ii, im in enumerate(images):
        vmin = np.percentile(im, 2)
        vmax = np.percentile(im, 98)
        grid[ii].set_title(_t(what[ii]))
        grid[ii].imshow(im, cmap="gray", vmin=vmin, vmax=vmax)

    if also_common:
        vmin = min(np.percentile(im, 2) for im in images)
        vmax = max(np.percentile(im, 98) for im in images)
        for ii, im in enumerate(images):
            grid[ii + ncols].set_title(_t(what[ii]))
            im = grid[ii + ncols].imshow(im, cmap="viridis", vmin=vmin, vmax=vmax)

    return fig


def imshow_pcorr_translation(
    fig: Figure,
    cpss: list[NDArray],
    extent: NDArray,
    results: list[NDArray],
    successes: list[float],
) -> Figure:
    import mpl_toolkits.axes_grid1 as axg

    ncols = 2
    grid = axg.ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(1, ncols),
        add_all=True,
        axes_pad=0.4,
        aspect=False,
        cbar_pad=0.05,
        label_mode="L",
        cbar_mode="single",
        cbar_size="3.5%",
    )
    vmax = max(cpss[0].max(), cpss[1].max())
    imshow_kwargs = {
        "vmin": 0,
        "vmax": vmax,
        "aspect": "auto",
        "origin": "lower",
        "extent": extent,
        "cmap": "viridis",
    }
    titles = (_t("CPS — translation 0°"), _t("CPS — translation 180°"))
    labels = (_t("translation y / px"), _t("translation x / px"))
    for idx, pl in enumerate(grid):
        # TODO: Code duplication with imshow_pcorr
        pl.set_title(titles[idx])
        center = np.array(results[idx])
        im = pl.imshow(cpss[idx], **imshow_kwargs)

        # Otherwise plot would change xlim
        pl.autoscale(False)
        pl.plot(
            center[0], center[1], "o", color="r", fillstyle="none", markersize=18, lw=8
        )
        pl.annotate(
            _t(f"succ: {successes[idx]:.3g}"),
            xy=center,
            xytext=(0, 9),
            textcoords="offset points",
            color="red",
            va="bottom",
            ha="center",
        )
        pl.annotate(
            _t("({:.3g}, {:.3g})".format(*center)),
            xy=center,
            xytext=(0, -9),
            textcoords="offset points",
            color="red",
            va="top",
            ha="center",
        )
        pl.grid(c="w")
        pl.set_xlabel(labels[1])

    grid.cbar_axes[0].colorbar(im)
    grid[0].set_ylabel(labels[0])

    return fig


def imshow_pcorr(
    fig: Figure,
    raw: NDArray,
    filtered: NDArray,
    extent: NDArray,
    result: NDArray,
    success: bool,
    log_base: float | None = None,
    terse: bool = False,
) -> Figure:
    import mpl_toolkits.axes_grid1 as axg
    from matplotlib.ticker import ScalarFormatter

    ncols = 1 if terse else 2
    grid = axg.ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(1, ncols),
        add_all=True,
        axes_pad=0.4,
        aspect=False,
        cbar_pad=0.05,
        label_mode="L",
        cbar_mode="single",
        cbar_size="3.5%",
    )
    vmax = raw.max()
    imshow_kwargs = {
        "vmin": 0,
        "vmax": vmax,
        "aspect": "auto",
        "origin": "lower",
        "extent": extent,
        "cmap": "viridis",
    }
    grid[0].set_title(_t("CPS"))
    labels = (_t("translation y / px"), _t("translation x / px"))
    im = grid[0].imshow(raw, **imshow_kwargs)

    center = np.array(result)
    # Otherwise plot would change xlim
    grid[0].autoscale(False)
    grid[0].plot(
        center[0], center[1], "o", color="r", fillstyle="none", markersize=18, lw=8
    )
    grid[0].annotate(
        _t(f"succ: {success:.3g}"),
        xy=center,
        xytext=(0, 9),
        textcoords="offset points",
        color="red",
        va="bottom",
        ha="center",
    )
    grid[0].annotate(
        _t("({:.3g}, {:.3g})".format(*center)),
        xy=center,
        xytext=(0, -9),
        textcoords="offset points",
        color="red",
        va="top",
        ha="center",
    )
    # Show the grid only on the annotated image
    grid[0].grid(c="w")
    if not terse:
        grid[1].set_title(_t("CPS — constrained and filtered"))
        im = grid[1].imshow(filtered, **imshow_kwargs)
    grid.cbar_axes[0].colorbar(im)

    if log_base is not None:
        for dim in range(ncols):
            grid[dim].set_xscale("log", basex=log_base)
            grid[dim].get_xaxis().set_major_formatter(ScalarFormatter())
            xlabels = grid[dim].get_xticklabels(False, "both")
            for x in xlabels:
                x.set_ha("right")
                x.set_rotation_mode("anchor")
                x.set_rotation(40)
        labels = (_t("rotation / degrees"), _t("scale change"))

    # The common stuff
    for idx in range(ncols):
        grid[idx].set_xlabel(labels[1])

    grid[0].set_ylabel(labels[0])

    return fig


def imshow_tiles(
    fig: Figure, im0: NDArray, slices: list[list[slice]], shape: tuple[int, ...]
) -> None:
    axes = fig.add_subplot(111)
    axes.imshow(im0, cmap="viridis")
    callback = Rect_mpl(axes, shape)
    slices2rects(slices, callback)


def imshow_results(
    fig: Figure, successes: NDArray, shape: tuple[int, ...], cluster: NDArray
) -> None:
    toshow = successes.reshape(shape)

    axes = fig.add_subplot(111)
    img = axes.imshow(toshow, cmap="viridis", interpolation="none")
    fig.colorbar(img)

    axes.set_xticks(np.arange(shape[1]))
    axes.set_yticks(np.arange(shape[0]))

    coords = np.unravel_index(np.arange(len(successes)), shape)
    for idx, coord in enumerate(zip(*coords, strict=False)):
        color = "w"
        if cluster[idx]:
            color = "r"
        label = f"{idx:02d}\n({coord[1]},{coord[0]})"
        axes.text(
            coord[1],
            coord[0],
            label,
            va="center",
            ha="center",
            color=color,
        )


def mk_factory(
    prefix: str, basedim: NDArray, dpi: int = 150, ftype: str = "png"
) -> Callable:
    import matplotlib.pyplot as plt

    @contextlib.contextmanager
    def _figfun(
        basename: str, x: float, y: float, use_aspect: bool = True
    ) -> Generator[Figure]:
        _basedim = basedim
        if use_aspect is False:
            _basedim = basedim[0]
        fig = plt.figure(figsize=_basedim * np.array((x, y)))
        yield fig
        fname = f"{prefix}{basename}.{ftype}"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        del fig

    return _figfun


def report_tile(reports: ReportsWrapper, prefix: str, multiplier: float = 5.5) -> None:
    multiplier = reports.get_global("size")
    dpi = reports.get_global("dpi")
    ftype = reports.get_global("ftype")
    terse = reports.get_global("terse")

    aspect = reports.get_global("aspect")
    basedim = multiplier * np.array((aspect, 1), float)
    for index, contents in reports.get_contents():
        fig_factory = mk_factory(f"{prefix}-{index}", basedim, dpi, ftype)
        for key, value in contents.items():
            _report_switch(fig_factory, key, value, reports, contents, terse)


def _report_switch(
    fig_factory: Callable,
    key: str,
    value: Any,
    reports: ReportsWrapper,
    contents: dict[str, Any],
    terse: bool,
) -> None:
    if "ims_filt" in key and reports.show("inputs"):
        with fig_factory(key, 2, 2) as fig:
            imshow_plain(fig, value, ("template", "subject"), not terse)
    elif "dfts_filt" in key and reports.show("spectra"):
        with fig_factory(key, 2, 1, False) as fig:
            imshow_spectra(fig, value)
    elif "logpolars" in key and reports.show("logpolar"):
        with fig_factory(key, 1, 1.4, False) as fig:
            imshow_logpolars(fig, value, contents["base"], contents["shape"])
    elif "amas-orig" in key and reports.show("scale_angle"):
        with fig_factory("sa", 2, 1) as fig:
            center = np.array(contents["amas-result"], float)
            imshow_pcorr(
                fig,
                value[:, ::-1],
                contents["amas-postproc"][:, ::-1],
                contents["amas-extent"],
                center,
                contents["amas-success"],
                log_base=contents["base"],
            )
    elif "tiles_successes" in key and reports.show("tile_info"):
        with fig_factory("tile-successes", 1, 1) as fig:
            imshow_results(
                fig,
                value,
                reports.get_global("tiles-shape"),
                reports.get_global("tiles-cluster"),
            )
    elif "tiles_decomp" in key and reports.show("tile_info"):
        with fig_factory("tile-decomposition", 1, 1) as fig:
            imshow_tiles(
                fig,
                reports.get_global("tiles-whole"),
                value,
                reports.get_global("tiles-shape"),
            )
    elif "after_tform" in key and reports.show("transformed"):
        shape = (len(value), 2)
        if terse:
            shape = (2, 2)
        with fig_factory(key, *shape) as fig:
            imshow_plain(
                fig,
                value,
                (
                    "plain",
                    "rotated--scaled",
                    "translated — bad rotation",
                    "translated — good rotation",
                ),
                not terse,
            )
    elif "t0-orig" in key and reports.show("translation"):
        origs = [contents[f"t{idx}-orig"] for idx in range(2)]
        tvecs = [contents[f"t{idx}-tvec"][::-1] for idx in range(2)]
        successes = [contents[f"t{idx}-success"] for idx in range(2)]
        halves = np.array(origs[0].shape) / 2.0
        extent = np.array((-halves[1], halves[1], -halves[0], halves[0]))

        if terse:
            with fig_factory("t", 2, 1) as fig:
                imshow_pcorr_translation(fig, origs, extent, tvecs, successes)
        else:
            t_flip = ("0", "180")
            for idx in range(2):
                basename = f"t_{t_flip[idx]}"
                ncols = 2
                with fig_factory(basename, ncols, 1) as fig:
                    imshow_pcorr(
                        fig,
                        origs[idx],
                        contents[f"t{idx}-postproc"],
                        extent,
                        tvecs[idx],
                        successes[idx],
                        terse=terse,
                    )
