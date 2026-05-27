import numpy.lib.stride_tricks
from PIL import Image
import numpy as np
#from numpy.matlib import zeros
import matplotlib.pyplot as plt
from pathlib import Path
import os


def make_kernel(ksize, sigma):
    # check for valid ksize
    if ksize % 2 == 0:
        raise ValueError("Kernelsize must be uneven.")
    if ksize <= 0:
        raise ValueError("Kernelsize must be bigger than 0.")


    # create coordinates
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    # calculate gaussian values
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma * sigma)

    abs_val = np.sum(kernel)

    return kernel / abs_val


def slow_convolve(arr, k):
    # flip kernel for true convolution
    k_flip = np.flip(k)

    # only apply padding if size of k is bigger than 1 in this dimension
    kh, kw = k.shape
    if arr.ndim == 3:
        # create padding with 0 values around arr
        arr_padded = np.pad(arr, pad_width=((kh // 2, (kh - 1) // 2),(kw // 2, (kw - 1) // 2),(0,0)), mode='constant', constant_values=0)
        # create 4d array with the kernel sized windows for every position
        windows = numpy.lib.stride_tricks.sliding_window_view(arr_padded, k.shape,axis=(0,1))
        # calculate the sum over the single kernel sized windows and return the values
        return  np.sum(windows * k_flip, axis=(3,4))
    else:
        # create padding with 0 values around arr
        arr_padded = np.pad(arr, pad_width=((kh // 2, (kh - 1) // 2),(kw // 2, (kw - 1) // 2)), mode='constant', constant_values=0)
        # create 4d array with the kernel sized windows for every position
        windows = numpy.lib.stride_tricks.sliding_window_view(arr_padded, k.shape, axis=(0, 1))
        # calculate the sum over the single kernel sized windows and return the values
        return np.sum(windows * k_flip, axis=(2,3))



if __name__ == '__main__':
    # Load the images and perform histogram equalization.
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    output_dir = data_dir / 'output'
    kernel_size = 77
    sig = kernel_size / 5
    k = make_kernel(kernel_size, sig)   # todo: find better parameters

    input1_path = data_dir / 'input1.jpg'
    input2_path = data_dir / 'input2.jpg'
    input3_path = data_dir / 'input3.jpg'

    # TODO: chose the image you prefer
    #im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    im = np.array(Image.open(input1_path))

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    convolution_result = slow_convolve(im, k)

    result = im - convolution_result + im
    result_clipped = np.clip(result, 0, 255).astype(np.uint8)

    # Figure mit zwei Subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Erstes Bild
    axes[0].imshow(im)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Zweites Bild
    axes[1].imshow(result_clipped)
    axes[1].set_title("Result")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    path = str(data_dir / 'results/convolution.png')

    print(os.getcwd())
    print("Saving to:", os.path.abspath(path))
    plt.imsave(path, result_clipped)
