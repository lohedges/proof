from pathlib import Path

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
    d = filters.gaussian(d, sigma=0.5)

    lines = extract_edge_segments(d)
    merged_lines = merge_line_segments(lines)
    joined_lines = join_line_segments(merged_lines)

    image_fig = plt.figure(figsize=(10, 9))
    image_ax = image_fig.subplots()
    image_ax.imshow(d, cmap="bone")#, vmin=29)#, vmax=27)

    #edges_fig = plt.figure(figsize=(10, 9))
    #edges_ax = edges_fig.subplots()
    #edges_ax.imshow(edges)

    for p0, p1 in lines:
        image_ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color="red")

    #fig = plt.figure(figsize=(10, 9))
    #ax = fig.subplots()
    #ax.hist(d.ravel(), bins=1000)
    plt.show()

    #break
