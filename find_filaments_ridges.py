from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import mrcfile

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

    raw_lines = image.find_lines(blur, threshold_mask=blob_mask)

    end = time.time()
    print("Line finding", end - start)

    line_segments = []
    for x1, y1, x2, y2 in raw_lines:
        l = lsm.LineSegment(x1, y1, x2, y2)
        line_segments.append(l)


    start = time.time()
    merged_lines = lsm.merge_lines(line_segments, tau_theta=0.01, xi_s=20)
    end = time.time()
    print("Merging:", end - start)

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_cv, cmap="bone")
    for line in merged_lines:
        cv_image_ax.plot((line.x1, line.x2), (line.y1, line.y2), color="red")

    plt.show()

    #break
