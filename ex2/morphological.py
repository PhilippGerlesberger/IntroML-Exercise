import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    """
    The function receives a padded image (pad_image) and the current pixel of our padded image.
    Return the surrounding area around that center pixel with the given size (window_size).
    """
    half_window = window_size // 2

    row_start = center_row - half_window
    row_end = row_start + window_size

    col_start = center_col - half_window
    col_end = col_start + window_size

    return padded_image[row_start : row_end, col_start : col_end]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    """
    Pad the image with zeros.
    """
    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """
    Apply erosion on the given image using the structuring element.
    """
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."
    active = structuring_element == 1

    # Create the padded image and an empty output image that can be filled later.
    padded_image = pad_image(image, se_size // 2)
    output = np.zeros_like(image)

    image_y = image.shape[0]
    image_x = image.shape[1]


    # Iterate over the provided image and perform erosion around each pixel.
    # Hint: Use the extract_region function to get the area around each pixel.
    # Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    for row in range(image_y):
        for col in range(image_x):
            region = extract_region(padded_image, row + (se_size // 2), col + (se_size // 2), se_size)

            # Check if the structuring element fits in the region (i.e. all pixels under the SE are 1).
            if np.all(region[active] == 1):
                output[row, col] = 1

    return output


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    """
    Apply dilation on the given image using the structuring element.
    """
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."
    active = structuring_element == 1

    # Create the padded image and an empty output image that can be filled later.
    padded_image = pad_image(image, se_size // 2)
    output = np.zeros_like(image)

    image_y = image.shape[0]
    image_x = image.shape[1]

    # Log padded image to file with all values


    # Iterate over the provided image and perform dilation around each pixel.
    # Hint: Use the extract_region function to get the area around each pixel.
    # Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.
    for row in range(image_y):
        for col in range(image_x):
            region = extract_region(padded_image, row + (se_size // 2), col + (se_size // 2), se_size)

            # Check if the structuring element fits in the region (i.e. any pixel under the SE is 1).
            if np.any(region[active] == 1):
                output[row, col] = 1

    return output

def repeat_operation(operation_func, image: np.ndarray, structuring_element: np.ndarray, iterations: int) -> np.ndarray:
    result = image
    for _ in range(iterations):
        result = operation_func(result, structuring_element)
    return result

def open_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Perform opening (erosion followed by dilation) on the input image.
    """
    result = input_image.copy()
    result = repeat_operation(erode_binary, result, structuring_element, iterations)
    result = repeat_operation(dilate_binary, result, structuring_element, iterations)
    return result


def close_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Perform closing (dilation followed by erosion) on the input image.
    """
    result = input_image.copy()
    result = repeat_operation(dilate_binary, result, structuring_element, iterations)
    result = repeat_operation(erode_binary, result, structuring_element, iterations)
    return result


def load_binary(filepath: str) -> np.ndarray:
    """
    Load the image and binarize it again with a simple threshold.
    """
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)  # type: ignore
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    """
    Save the binary image to the specified filepath.
    """
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    """
    Display the image using matplotlib.
    """
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def perform_opening(input_image, SE, output_dir, iterations=1):
    """
    Perform opening on the input image and save the result.
    """
    opened = open_binary(input_image, SE, iterations=iterations)
    show_image(opened, f'Opened Image {iterations} iterations')
    save_binary(opened, output_dir / f'opened_{iterations}_iterations.png')


def perform_closing(input_image, SE, output_dir, iterations=1):
    """
    Perform closing on the input image and save the result.
    """
    closed = close_binary(input_image, SE, iterations=iterations)
    show_image(closed, f'Closed Image {iterations} iterations')
    save_binary(closed, output_dir / f'closed_{iterations}_iterations.png')


def perform_erosion(input_image, SE, output_dir, iterations=3):
    """
    Perform erosion on the input image and save the result.
    """
    eroded = repeat_operation(erode_binary, input_image, SE, iterations)
    show_image(eroded, f'Eroded Image {iterations} iterations')
    save_binary(eroded, output_dir / f'eroded_{iterations}_iterations.png')


def perform_dilation(input_image, SE, output_dir, iterations=7):
    """
    Perform dilation on the input image and save the result.
    """
    dilated = repeat_operation(dilate_binary, input_image, SE, iterations)
    show_image(dilated, f'Dilated Image {iterations} iterations')
    save_binary(dilated, output_dir / f'dilated_{iterations}_iterations.png')


if __name__ == '__main__':
    # Paths.
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    output_dir = data_dir / 'output'
    raw_erosion_image_path = data_dir / 'erosion_image_raw.png'
    raw_dilation_image_path = data_dir / 'dilation_image_raw.png'

    # Load images.
    erosion_input = load_binary(raw_erosion_image_path)
    dilation_input = load_binary(raw_dilation_image_path)

    # Structuring element.
    SE = np.ones((5, 5), dtype=np.uint8)

    # Perform opening and closing.
    #perform_opening(erosion_input, SE, output_dir)
    #perform_closing(dilation_input, SE, output_dir)

    # Perform erosion:
    # Repeatetly shrink the image until the circles are completely separated.
    perform_erosion(erosion_input, SE, output_dir, 3)
    perform_erosion(erosion_input, SE, output_dir, 4)

    # Perform dilation:
    # Repeatetly grow the shapes until the central hole is filled.
    perform_dilation(dilation_input, SE, output_dir, 7)
    perform_dilation(dilation_input, SE, output_dir, 8)
