import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import cv2

# generate some sample data
import scipy.misc

def view_surface(path):
    img = cv2.imread(path, 0)

    # downscaling has a "smoothing" effect
    img = cv2.resize(img, (100,100))/250

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # create the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)


    ax.view_init(elev=80, azim=25)

    plt.figure(figsize=(18, 16), dpi=180)

    # show it
    plt.show()

