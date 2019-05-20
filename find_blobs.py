from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from skimage import exposure, transform

for f in Path(".").glob("*.mrc"):
    with mrcfile.open(f) as mrc:
        h = mrc.header
        d = mrc.data

    start = time.time()

    d = exposure.equalize_hist(d)
    scale = 4
    d = transform.resize(d, (int(d.shape[0]/scale), int(d.shape[0]/scale)))
    d_cv = (d * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(d_cv, (7, 7), 3)

    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    erosion_size = 4
    element_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    eroded = cv2.dilate(th3, element_d)
    eroded = cv2.dilate(eroded, element_d)
    eroded = cv2.erode(eroded, element_d)
    eroded = cv2.erode(eroded, element_d)

    eroded_blur = cv2.GaussianBlur(eroded, (3, 3), 1)

    end = time.time()
    print(end - start)


    image_fig = plt.figure(figsize=(10, 9))
    image_ax = image_fig.subplots()
    image_ax.imshow(d, cmap="bone")

    image_fig = plt.figure(figsize=(10, 9))
    image_ax = image_fig.subplots()
    image_ax.imshow(eroded_blur, cmap="bone")

    plt.show()

    #break
