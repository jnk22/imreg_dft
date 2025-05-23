# cli.py
#

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

"""FFT based image registration. --- CLI frontend."""

from __future__ import annotations

import argparse as ap
import sys
from typing import TYPE_CHECKING, Any, Final, Literal

import imreg_dft as ird
from imreg_dft import loader, tiles, utils

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__CONSTRAINT_BOUNDS: Final[dict[Literal["angle", "scale"], tuple[float, float]]] = {
    "angle": (-180, 180),
    "scale": (0.5, 2.0),
}


def assure_constraint(possible_constraints: dict[str, Any]) -> None:
    pass


def _constraints(
    what: Literal["angle", "scale", "shift"],
) -> Callable:
    def constraint(string: str) -> tuple[float, float | None]:
        components = string.split(",")
        if not (0 < len(components) <= 2):
            msg = f"We accept at most {len(components)} (but at least 1) comma-delimited numbers, you have passed us 2"
            raise ap.ArgumentTypeError(msg)
        try:
            mean = float(components[0])
        except Exception as e:
            msg = f"The {what} value must be a float number, got '{components[0]}'."
            raise ap.ArgumentTypeError(msg) from e
        if what in __CONSTRAINT_BOUNDS:
            lo, hi = __CONSTRAINT_BOUNDS[what]
            if not lo <= mean <= hi:
                msg = f"The {what} value must be a number between {lo:g} and {hi:g}, got {mean:g}."
                raise ap.ArgumentTypeError(msg)

        if len(components) != 2:
            return mean, 0

        std = components[1]
        if not std:
            return mean, None

        try:
            return (mean, float(std))
        except Exception as e:
            msg = f"The {what} standard deviation spec must be either a float number or nothing, got '{std}'."
            raise ap.ArgumentTypeError(msg) from e

    return constraint


def _float_tuple(string: str) -> list[float]:
    """Support function for parsing string of two floats delimited by a comma."""
    vals = string.split(",")
    if len(vals) != 2:
        msg = f"'{string}' are not two values delimited by comma"
        raise ap.ArgumentTypeError(msg)
    try:
        return [float(val) for val in vals]
    except ValueError as e:
        msg = f"{vals} are not two float values"
        raise ap.ArgumentTypeError(msg) from e


def _exponent(string: str) -> str | float:
    """Converts the passed string to a float or "inf"."""
    if string == "inf":
        return string
    try:
        ret = float(string)
    except Exception as e:
        msg = f"'{string}' should be either 'inf' or a float value"
        raise ap.ArgumentTypeError(msg) from e
    return ret


def outmsg(msg: str) -> str:
    """Support function for checking of validity of the output format string.
    A test interpolation is performed and exceptions handled.
    """
    tpl = "The string '%s' is not a good format string"
    fake_data = {
        "scale": 1.0,
        "angle": 2.0,
        "tx": 2,
        "ty": 2,
        "Dscale": 0.1,
        "Dangle": 0.2,
        "Dt": 0.5,
        "success": 0.99,
    }
    try:
        msg % fake_data
    except KeyError as exc:
        raise ap.ArgumentTypeError(
            f"{tpl}. The correct string has to contain at most %s, but this one also contains an invalid value '%s'."
            % (msg, fake_data.keys(), exc.args[0])
        ) from exc
    except Exception as exc:
        raise ap.ArgumentTypeError(f"{tpl} - %s" % (msg, str(exc))) from exc
    return msg


def create_base_parser(parser: ap.ArgumentParser) -> None:
    parser.add_argument(
        "--extend",
        type=int,
        metavar="PIXELS",
        default=0,
        help="Extend images by the specified amount of pixels "
        "before the processing (thus eliminating "
        "edge effects)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Interpolation order (1 = linear, 3 = cubic etc)",
    )


def update_parser_imreg(parser: ap.ArgumentParser) -> None:
    parser.add_argument("template")
    parser.add_argument("subject")
    parser.add_argument(
        "--lowpass",
        type=_float_tuple,
        default=None,
        metavar="LOW_THRESH,HIGH_THRESH",
        help="1,1 means no-op, 0.8,0.9 is a mild filter",
    )
    parser.add_argument(
        "--highpass",
        type=_float_tuple,
        default=None,
        metavar="LOW_THRESH,HIGH_THRESH",
        help="0,0 means no-op, 0.1,0.2 is a mild filter",
    )
    parser.add_argument(
        "--cut",
        type=_float_tuple,
        default=None,
        metavar="LOW_THRESH,HIGH_THRESH",
        help="Cap values of the image according to "
        "quantile values. "
        "0,1 means no-op, 0.01,0.99 is a mild filter",
    )
    parser.add_argument(
        "--resample", type=float, default=1, help="Work with resampled images."
    )
    # parser.add_argument('--exponent', type=_exponent, default="inf",
    #                     help="Either 'inf' or float. See the docs.")
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="How many iterations to guess the right scale and angle",
    )
    create_base_parser(parser)
    parser.add_argument(
        "--filter-pcorr",
        type=int,
        default=0,
        help="Whether to filter during translation detection. Normally not "
        "needed, but when using low-pass filtering, you may need to increase "
        "filter radius (0 means no filtering, 4 should be enough)",
    )
    parser.add_argument(
        "--print-result",
        action="store_true",
        default=False,
        help="We don't print anything unless this option is specified",
    )
    parser.add_argument(
        "--print-format",
        type=outmsg,
        default="scale: %(scale).5g +-%(Dscale).4g\n"
        "angle: %(angle).6g +-%(Dangle).5g\n"
        "shift (x, y): %(tx).6g, %(ty).6g +-%(Dt).4g\nsuccess: %(success).4g\n",
        help="Print a string (to stdout) in a given format. A dictionary "
        "containing the 'scale', 'angle', 'tx', 'ty', 'Dscale', 'Dangle', "
        "'Dt' and 'success' keys will be passed for string interpolation",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        default=False,
        help="If the template "
        "is larger than the subject, break the template to pieces of size "
        "similar to subject size.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"imreg_dft {ird.__version__}",
        help="Just print version and exit",
    )
    parser.add_argument(
        "--angle",
        type=_constraints("angle"),
        metavar="MEAN[,STD]",
        default=(0, None),
        help="The mean and standard deviation of the expected angle. ",
    )
    parser.add_argument(
        "--scale",
        type=_constraints("scale"),
        metavar="MEAN[,STD]",
        default=(1, None),
        help="The mean and standard deviation of the expected scale. ",
    )
    parser.add_argument(
        "--tx",
        type=_constraints("shift"),
        metavar="MEAN[,STD]",
        default=(0, None),
        help="The mean and standard deviation of the expected X translation. ",
    )
    parser.add_argument(
        "--ty",
        type=_constraints("shift"),
        metavar="MEAN[,STD]",
        default=(0, None),
        help="The mean and standard deviation of the expected Y translation. ",
    )
    parser.add_argument("--output", "-o", help="Where to save the transformed subject.")
    loader.update_parser(parser)


def create_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    update_parser_imreg(parser)
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Whether to show registration result",
    )
    return parser


def args2dict(args: ap.Namespace) -> dict[str, Any]:
    loaders = loader.settle_loaders(args, (args.template, args.subject))

    # We need tuples in the parser and lists further in the code.
    # So we have to do it like this.
    constraints = {
        "angle": list(args.angle),
        "scale": list(args.scale),
        "tx": list(args.tx),
        "ty": list(args.ty),
    }

    print_format = args.print_format if args.print_result else None
    return {
        "order": args.order,
        "filter_pcorr": args.filter_pcorr,
        "extend": args.extend,
        "low": args.lowpass,
        "high": args.highpass,
        "cut": args.cut,
        "print_format": print_format,
        "iters": args.iters,
        "exponent": "inf",
        "resample": args.resample,
        "tile": args.tile,
        "constraints": constraints,
        "output": args.output,
        "loaders": loaders,
        "reports": None,
    }


def main() -> None:
    parser = create_parser()

    args = parser.parse_args()

    opts = args2dict(args)
    opts["show"] = args.show
    run(args.template, args.subject, opts)


def _get_resdict(
    imgs: list[NDArray], opts: dict[str, Any], tosa: NDArray | None = None
) -> dict[str, Any]:
    import numpy as np

    reports = opts.get("reports")
    tiledim = None
    if opts["tile"]:
        shapes = np.array([np.array(img.shape) for img in imgs])
        if (shapes[0] / shapes[1]).max() > 1.7:
            tiledim = np.min(shapes, axis=0) * 1.1
            # TODO: Establish a translate region constraint of width tiledim * coef
    if tiledim is not None:
        resdict = tiles.settle_tiles(imgs, tiledim, opts, reports)

        # TODO: This "tosa" occurence is convoluted - it is not needed
        #  in process_images
        if tosa is not None:
            tosa[:] = ird.transform_img_dict(tosa, resdict)
    else:
        resdict = tiles.process_images(
            imgs, opts, tosa, get_unextended=True, reports=reports
        )

    if reports is not None:
        reports.set_global("aspect", imgs[0].shape[1] / float(imgs[0].shape[0]))

    return resdict


def run(template: NDArray, subject: NDArray, opts: dict[str, Any]) -> None:
    # lazy import so no imports before run() is really called
    from imreg_dft import imreg

    fnames = (template, subject)
    loaders = opts["loaders"]
    loader_img = loaders[1]
    imgs = [loa.load2reg(fname) for fname, loa in zip(fnames, loaders, strict=False)]

    # The array where the result should be placed
    tosa = None
    saver = None
    outname = opts["output"]
    if outname is not None:
        tosa = loader_img.get2save()
        saver = loader.LOADERS.get_loader(outname)
        tosa = utils.extend_to_3D(tosa, imgs[0].shape[:3])

    resdict = _get_resdict(imgs, opts, tosa)
    im0, im1, im2 = resdict["unextended"]

    if opts["print_format"] is not None:
        msg = opts["print_format"] % resdict
        msg = msg.encode("utf-8")
        msg = msg.decode("unicode_escape")
        sys.stdout.write(msg)

    if outname is not None:
        saver.save(outname, tosa, loader_img)

    if opts["show"]:
        import pylab as pyl

        fig = pyl.figure()
        imreg.imshow(im0, im1, im2, fig=fig)
        pyl.show()


if __name__ == "__main__":
    main()
