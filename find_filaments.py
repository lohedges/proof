from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from skimage import exposure, transform, feature, filters


def extract_edge_segments(d: np.ndarray) -> list:
    """
    Given an image, ``d``, extract line segments along the edges
    """
    edges = feature.canny(d, sigma=2)
    lines = transform.probabilistic_hough_line(edges, threshold=5, line_length=7, line_gap=2)
    return lines


def extract_edge_segments2(d: np.ndarray) -> list:
    """
    Given an image, ``d``, extract line segments along the edges
    """
    d_cv = (d * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(d_cv, (7, 7), 3)
    edges = cv2.Canny(blur, 25, 75, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=3, maxLineGap=2)
    return lines


def merge_line_segments(lines: list) -> list:
    """
    Tavares and Padilha
    """
    return lines


def join_line_segments(lines: list) -> list:
    """
    Given a list of lines, join those that are close together.

    doi:10.1.1.38.4011 maybe
    """
    return lines


for f in Path(".").glob("*.mrc"):
    with mrcfile.open(f) as mrc:
        h = mrc.header
        d = mrc.data

    d = exposure.equalize_hist(d)
    scale = 8
    d = transform.resize(d, (int(d.shape[0]/scale), int(d.shape[0]/scale)))

    lines = extract_edge_segments(d)
    merged_lines = merge_line_segments(lines)
    joined_lines = join_line_segments(merged_lines)

    image_fig = plt.figure(figsize=(10, 9))
    image_ax = image_fig.subplots()
    image_ax.imshow(d, cmap="bone")#, vmin=29)#, vmax=27)
    for p0, p1 in lines:
        image_ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color="red")

    cv2_lines = extract_edge_segments2(d)

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d, cmap="bone")
    for x1, y1, x2, y2 in (l[0] for l in cv2_lines):
        cv_image_ax.plot((x1, x2), (y1, y2), color="red")

    plt.show()

    #break
