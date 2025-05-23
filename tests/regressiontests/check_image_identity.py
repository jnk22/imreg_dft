from __future__ import annotations

import argparse as ap
import sys
from typing import TYPE_CHECKING, Final

import imageio.v3 as iio
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

STR2NORM: Final[dict[str, Callable]] = {
    "max": np.max,
    "mean": np.mean,
}


RET2RES: Final = ("same", "different")


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("one")
    parser.add_argument("two")
    parser.add_argument("thresh", type=float, default=0.01, nargs="?")
    parser.add_argument("--grayscale", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--norm", choices=("max", "mean"), default="max")
    args = parser.parse_args()

    imgs = [iio.imread(fname).astype(float) for fname in (args.one, args.two)]
    diff = np.abs(imgs[1] - imgs[0])
    mean = np.abs(imgs[0] + imgs[1]) / 2.0
    if not args.grayscale:
        # format (y, x, RGB)
        diff, mean = [np.sum(arr, axis=-1) / 3.0 for arr in (diff, mean)]

    norm = STR2NORM[args.norm]
    if args.verbose:
        print(f"Reference norm: {norm(mean):.4g}\nDifference norm: {norm(diff):.4g}")
    ret = 0
    if norm(mean) * args.thresh < norm(diff):
        # The difference is too big
        ret = 1
        msg = f"{args.one} and {args.two} are significantly different :-(\n"
        sys.stderr.write(msg)

    if args.verbose:
        print(f"The two images are {RET2RES[ret]}.")

    sys.exit(ret)


if __name__ == "__main__":
    main()
