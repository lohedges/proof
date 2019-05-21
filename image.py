import cv2
import numpy as np


def normalise_8bit(d):
    """
    Given an image with values in an arbitrary range, shift the values so they
    lie on a range from 0 to 255 and convert them to 8 bit integers.

    Args:
        d: the image file to be normalised

    Returns: an image of the same size as d

    """
    d_min = d.min()
    d_max = d.max()
    shifted = d - d_min
    shifted /= ((d_max - d_min) / 255)
    d_cv = shifted.astype(np.uint8)
    return d_cv


def find_lines(d) -> list:
    """
    Given an image, find the centre lines of any lines in the image.
    It will return quite fragmented lines which will need to be filtered.

    Args:
        d: the image file to be processed

    Returns: a list of lines

    """
    # Tune the kernel size
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create(ksize=3)
    ridges = ridge_filter.getRidgeFilteredImage(d)

    erosion_size = 1
    element_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                          (erosion_size, erosion_size))
    eroded = cv2.erode(ridges, element_d)
    _, th3 = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The Hough line parameters need to be tuned as well
    lines = cv2.HoughLinesP(th3, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=3)
    return [l[0] for l in lines]
