import cv2
import numpy as np
import sklearn.linear_model


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


def find_lines(d, threshold_mask=None) -> list:
    """
    Given an image, find the centre lines of any lines in the image.
    It will return quite fragmented lines which will need to be filtered.

    Args:
        d: the image file to be processed
        threshold_mask: an images of the same size as ``d`` which
                        will be applied to the post thresholding
                        image before Hough

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

    if threshold_mask is not None:
        masked = ((th3 / 255.0 * threshold_mask / 255.0) * 255).astype(np.uint8)
    else:
        masked = th3

    # The Hough line parameters need to be tuned as well
    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=3)
    return lines[:, 0, :]


def find_threshold(d, threshold_mask=None) -> list:
    # Tune the kernel size
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create(ksize=3)
    ridges = ridge_filter.getRidgeFilteredImage(d)

    erosion_size = 1
    element_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                          (erosion_size, erosion_size))
    eroded = cv2.erode(ridges, element_d)
    eroded = cv2.dilate(eroded, element_d)
    _, th3 = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if threshold_mask is not None:
        masked = ((th3 / 255.0 * threshold_mask / 255.0) * 255).astype(np.uint8)
    else:
        masked = th3

    return masked


def find_centre_lines(image):
    skeleton = cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    components = cv2.connectedComponents(skeleton, connectivity=8)[1]  # TODO Use connectedComponentsWithStats()?

    # Remove any components which are small
    unique, counts = np.unique(components, return_counts=True)
    keep_components = np.where(np.logical_and(10 < counts, counts < components.size / 10), unique, 0)
    with np.nditer(components, op_flags=['readwrite']) as it:
        for x in it:
            if x not in keep_components:
                x[...] = 0

    centres = np.where(components != 0, 255, 0).astype(np.uint8)
    return centres


def find_blobs(d):
    _, th3 = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    erosion_size = 4
    element_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                          (erosion_size, erosion_size))
    eroded = cv2.dilate(th3, element_d)
    eroded = cv2.dilate(eroded, element_d)
    eroded = cv2.erode(eroded, element_d)
    eroded = cv2.erode(eroded, element_d)
    eroded = cv2.erode(eroded, element_d)
    eroded = cv2.erode(eroded, element_d)

    eroded_blur = cv2.GaussianBlur(eroded, (3, 3), 1)
    return eroded_blur


def scale_down(d, scale):
    return cv2.resize(d, (int(d.shape[0] / scale), int(d.shape[0] / scale)), interpolation=cv2.INTER_AREA)


def adjust_gradient(image):
    m, n = image.shape
    R, C = np.mgrid[:m, :n]
    out = np.column_stack((C.ravel(), R.ravel(), image.ravel()))

    reg = sklearn.linear_model.LinearRegression().fit(out[:, 0:2], out[:, 2])

    plane = np.fromfunction(lambda x, y: y * reg.coef_[0] + x * reg.coef_[1] + reg.intercept_, image.shape)

    return image - plane
