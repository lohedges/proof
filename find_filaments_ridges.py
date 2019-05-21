from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mrcfile

import image

for f in Path(".").glob("*.mrc"):
    with mrcfile.open(f) as mrc:
        h = mrc.header
        d = mrc.data

    start = time.time()

    d = image.normalise_8bit(d)
    d_cv = cv2.equalizeHist(d)
    scale = 8
    d_cv = cv2.resize(d_cv, (int(d.shape[0]/scale), int(d.shape[0]/scale)), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(d_cv, (7, 7), 3)

    lines = image.find_lines(blur)

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
    for x1, y1, x2, y2 in lines:
        cv_image_ax.plot((x1, x2), (y1, y2), color="red")

    plt.show()

    #break
