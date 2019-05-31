from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

import image

for f in Path(".").glob("FoilHole_24681684_Data_24671838_24671839_20181024_1545-78085.mrc"):
    with mrcfile.open(f, permissive=True) as mrc:
        h = mrc.header
        d = mrc.data

    d_cv = image.scale_down(d, 8)
    d_cv = image.adjust_gradient(d_cv)

    #d_cv = d_cv[10:-10, 10:-10]
    d_cv = d_cv[50:50+32, 30:30+32]
    #d_cv = d_cv[100:164, 170:170+64]

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_cv, cmap="bone")

    transformed = cv2.dft(d_cv, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(transformed)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    th3 = image.normalise_8bit(magnitude_spectrum)
    _, th3 = cv2.threshold(th3, 200, 255, cv2.THRESH_BINARY)

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(magnitude_spectrum, cmap="bone")

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(th3, cmap="bone")

    plt.show()
