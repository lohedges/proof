from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

import image
import lsm

for f in Path(".").glob("*.mrc"):
    with mrcfile.open(f, permissive=True) as mrc:
        h = mrc.header
        d = mrc.data

    start = time.time()

    d_cv = image.scale_down(d, 8)
    d_cv = image.adjust_gradient(d_cv)
    d_cv = image.normalise_8bit(d_cv)
    blur = cv2.GaussianBlur(d_cv, (7, 7), 3)

    image_for_blobs = image.scale_down(d, 4)
    image_for_blobs = image.adjust_gradient(image_for_blobs)
    image_for_blobs = image.normalise_8bit(image_for_blobs)
    image_for_blobs = cv2.GaussianBlur(image_for_blobs, (7, 7), 3)
    blob_mask = image.find_blobs(image_for_blobs)
    blob_mask = image.scale_down(blob_mask, 2)

    thresholded = image.find_thresholded_ridges(blur, threshold_mask=blob_mask)
    centres = image.find_centre_lines(thresholded)

    # TODO Maybe look at fast arguments for this
    distances = cv2.distanceTransform(cv2.bitwise_not(centres), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)

    contours = cv2.findContours(centres.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    end = time.time()
    print("Line finding", end - start)

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_cv, cmap="bone")
    overlay = np.where(centres, 3, np.nan)
    plt.cm.Set1.set_bad(color="#00000000")
    cv_image_ax.imshow(overlay, cmap='Set1')
    cv_image_ax.set_xlim(365, 405)
    cv_image_ax.set_ylim(300, 260)

    from matplotlib.lines import Line2D
    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_cv, cmap="bone")
    cv_image_ax.set_xlim(365, 405)
    cv_image_ax.set_ylim(300, 260)
    for c in contours[0][30:30+1]:
        c = c[:, 0, :]
        line = Line2D(c[:, 0], c[:, 1])
        cv_image_ax.add_line(line)

    plt.show()

    #break
