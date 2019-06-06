from pathlib import Path

import cv2
import mrcfile
import pandas as pd
import matplotlib.pyplot as plt

import image

for f in Path(".").glob("FoilHole_24681684_Data_24671838_24671839_20181024_1545-78085.mrc"):
    with mrcfile.open(f, permissive=True) as mrc:
        h = mrc.header
        d = mrc.data

    d_cv = image.normalise_8bit(d)
    d_cv = image.scale_down(d_cv, 8)

    star_filename = f"{f.stem}_manualpick.star"

    df = pd.read_csv(f"{f.stem}_manualpick.star", delim_whitespace=True, skiprows=10, names=["x", "y", "class_number", "angle_psi", "fom"], usecols=[0, 1])
    df["x"] = df["x"] / (d.shape[1]/d_cv.shape[1])
    df["y"] = df["y"] / (d.shape[0]/d_cv.shape[0])

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_cv, cmap="bone")
    df.plot.scatter("x", "y", ax=cv_image_ax, color="none", facecolors='none', edgecolors='red', s=1000)

    plt.show()
