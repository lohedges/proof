from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mrcfile

for f in Path(".").glob("*.mrc"):
    with mrcfile.open(f) as mrc:
        h = mrc.header
        d = mrc.data

    start = time.time()

    d_min = d.min()
    d_max = d.max()
    shifted = d - d_min
    shifted /= ((d_max - d_min) / 255)
    d_cv = shifted.astype(np.uint8)
    d_cv = cv2.equalizeHist(d_cv)
    scale = 8
    d_cv = cv2.resize(d_cv, (int(d.shape[0]/scale), int(d.shape[0]/scale)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(d_cv, (7, 7), 3)

    # Tune the kernel size
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create(ksize=3)
    ridges = ridge_filter.getRidgeFilteredImage(blur)

    erosion_size = 1
    element_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    eroded = cv2.erode(ridges, element_d)
    _, th3 = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The Hough line parameters need to be tuned as well
    lines = cv2.HoughLinesP(th3, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=3)

    end = time.time()
    print(end - start)


    #image_fig = plt.figure(figsize=(10, 9))
    #image_ax = image_fig.subplots()
    #image_ax.imshow(eroded, cmap="bone")

    #image_fig = plt.figure(figsize=(10, 9))
    #image_ax = image_fig.subplots()
    #image_ax.imshow(th3, cmap="bone")

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_cv, cmap="bone")
    for x1, y1, x2, y2 in (l[0] for l in lines):
        cv_image_ax.plot((x1, x2), (y1, y2), color="red")

    plt.show()

    #break
