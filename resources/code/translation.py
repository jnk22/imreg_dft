import os
from pathlib import Path

import imageio.v3 as iio

import imreg_dft as ird

basedir = Path("../examples")
# the TEMPLATE
im0 = iio.imread(basedir / "sample1.png", mode="F")
# the image to be transformed
im1 = iio.imread(basedir / "sample2.png", mode="F")
result = ird.translation(im0, im1)
tvec = result["tvec"].round(4)
# the Transformed IMaGe.
timg = ird.transform_img(im1, tvec=tvec)

# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt

    ird.imshow(im0, im1, timg)
    plt.show()

print("Translation is {}, success rate {:.4g}".format(tuple(tvec), result["success"]))
