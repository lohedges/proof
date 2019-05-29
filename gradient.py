from pathlib import Path
import time

import matplotlib.pyplot as plt
import mrcfile
import numpy as np

import image


def plot_3d(image):
    yvalues = np.indices(image.shape)[0][:, 0]
    xvalues = np.indices(image.shape)[1][0, :]

    xgrid, ygrid = np.meshgrid(xvalues, yvalues)
    zvalues = image

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(xgrid, ygrid, zvalues, rstride=5, cstride=5, linewidth=0, cmap="bone")
    return surf


for f in Path(".").glob("*.mrc"):
    with mrcfile.open(f) as mrc:
        h = mrc.header
        d = mrc.data

    d_scaled = image.scale_down(d, 8)

    #plot_3d(d_scaled)

    start = time.time()

    d_flattened = image.adjust_gradient(d_scaled)

    end = time.time()
    print("Gradient adjustment", end - start)

    #plot_3d(d_flattened)

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_scaled, cmap="bone")

    cv_image_fig = plt.figure(figsize=(10, 9))
    cv_image_ax = cv_image_fig.subplots()
    cv_image_ax.imshow(d_flattened, cmap="bone")

    plt.show()
