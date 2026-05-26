import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -(xx ** 2 + yy ** 2) / (2 * sigma ** 2)
    )
    kernel = kernel / np.sum(kernel)

    filtered = convolve(img_in, kernel).astype(int)
    return kernel, filtered


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    # convolve flips the kernel, therefore use flipped Sobel kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    gx = convolve(img_in, kx).astype(int)
    gy = convolve(img_in, ky).astype(int)

    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    g = np.sqrt(gx ** 2 + gy ** 2).astype(int)
    theta = np.arctan2(gy, gx)
    return g, theta




def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    angle = np.rad2deg(angle) % 180

    if angle < 22.5 or angle >= 157.5:
        return 0
    elif angle < 67.5:
        return 45
    elif angle < 112.5:
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    max_sup = np.zeros_like(g).astype(int)
    rows, cols = g.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            angle = convertAngle(theta[y, x])
            current = g[y, x]

            if angle == 0:
                n1, n2 = g[y, x - 1], g[y, x + 1]
            elif angle == 45:
                n1, n2 = g[y + 1, x - 1], g[y - 1, x + 1]
            elif angle == 90:
                n1, n2 = g[y - 1, x], g[y + 1, x]
            else:
                n1, n2 = g[y - 1, x - 1], g[y + 1, x + 1]

            if current >= n1 and current >= n2:
                max_sup[y, x] = current

    return max_sup



def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    result = np.zeros_like(max_sup).astype(int)
    rows, cols = max_sup.shape

    strong = max_sup > t_high
    weak = (max_sup > t_low) & (max_sup <= t_high)

    for y in range(rows):
        for x in range(cols):
            if strong[y, x]:
                result[y, x] = 255

                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny = y + dy
                        nx = x + dx

                        if 0 <= ny < rows and 0 <= nx < cols:
                            if weak[ny, nx]:
                                result[ny, nx] = 255

    return result


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result
