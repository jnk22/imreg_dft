import os
from pathlib import Path

import imageio.v3 as iio

import imreg_dft as ird

basedir = Path("../examples")
# the TEMPLATE
im0 = iio.imread(basedir / "sample1.png", mode="F")
# the image to be transformed
im1 = iio.imread(basedir / "sample3.png", mode="F")
result = ird.similarity(im0, im1, numiter=3)

assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt

    ird.imshow(im0, im1, result["timg"])
    plt.show()
