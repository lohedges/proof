from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

import image
import lsm

for f in Path(".").glob("*.mrc"):
    with mrcfile.open(f) as mrc:
        h = mrc.header
        d = mrc.data

    start = time.time()

    d = image.normalise_8bit(d)
    d_equalised = cv2.equalizeHist(d)
    d_cv = image.scale_down(d_equalised, 8)
    blur = cv2.GaussianBlur(d_cv, (7, 7), 3)

    image_for_blobs = cv2.GaussianBlur(image.scale_down(d_equalised, 4), (7, 7), 3)
    blob_mask = image.find_blobs(image_for_blobs)
    blob_mask = image.scale_down(blob_mask, 2)

    thresholded = image.find_threshold(blur, threshold_mask=blob_mask)
    centres = image.find_centre_lines(thresholded)

    end = time.time()
    print("Line finding", end - start)

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_cv, cmap="bone")
    overlay = np.where(centres, 3, np.nan)
    plt.cm.Set1.set_bad(color="#00000000")
    cv_image_ax.imshow(overlay, cmap='Set1')

    plt.show()

    #break
