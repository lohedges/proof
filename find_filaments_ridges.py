from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

import image
import lsm
from trace_filament import trace_filaments

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

    lines = cv2.HoughLinesP(centres, 1, np.pi / 180, threshold=10, minLineLength=5, maxLineGap=5)[:, 0, :]
    line_segments = [lsm.LineSegment(x1, y1, x2, y2) for x1, y1, x2, y2 in lines]
    merged_lines = lsm.merge_lines(line_segments, tau_theta=0.1, xi_s=0.5)
    polys = trace_filaments(merged_lines)

    end = time.time()
    print("Line finding", end - start)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(d_cv, cmap='bone', alpha=0.99)

    for poly in polys:
        if poly.length < 30:
            continue
        x = [p.x for p in poly.points]
        y = [p.y for p in poly.points]
        ax.plot(x, y, linewidth=5, alpha=0.8)

    plt.show()

    #break
