import argparse as ap

from scipy import io, misc


def parse():
    parser = ap.ArgumentParser()
    parser.add_argument("imgfile")
    parser.add_argument("outfile")
    parser.add_argument("--var", default="img")
    parser.add_argument("--dummy-vars", default="", metavar="NAME1,NAME2,...")
    return parser.parse_args()


def main() -> None:
    args = parse()
    img = misc.imread(args.imgfile)
    tosave = {}
    for dvar in args.dummy_vars.split(","):
        if len(dvar) == 0:
            continue
        tosave[dvar] = 0
    tosave[args.var] = img
    io.savemat(args.outfile, tosave)


if __name__ == "__main__":
    main()
