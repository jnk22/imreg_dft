from __future__ import annotations

import argparse as ap

import imageio.v3 as iio
from scipy import io


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("imgfile")
    parser.add_argument("outfile")
    parser.add_argument("--var", default="img")
    parser.add_argument("--dummy-vars", default="", metavar="NAME1,NAME2,...")
    args = parser.parse_args()

    img = iio.imread(args.imgfile)
    tosave = {dvar: 0 for dvar in args.dummy_vars.split(",") if len(dvar) != 0}
    tosave[args.var] = img
    io.savemat(args.outfile, tosave)


if __name__ == "__main__":
    main()
