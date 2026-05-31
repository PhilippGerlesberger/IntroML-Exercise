import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    # The function receives a padded image (pad_image) and the current pixel of our padded image.
    # ToDo: Return the surrounding area around that center pixel with the given size (window_size).
    # ToDo: Use slicing.

    half_window = window_size // 2

    row_start = center_row - half_window
    row_end = row_start + window_size

    col_start = center_col - half_window
    col_end = col_start + window_size

    return padded_image[row_start : row_end, col_start : col_end]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    # Pad the image with zeros.
    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply erosion on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    padded_image = pad_image(image, se_size // 2)
    output = np.zeros_like(image)

    image_y = image.shape[0]
    image_x = image.shape[1]


    # ToDo: Iterate over the provided image and perform erosion around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.

    for row in range(image_y):
        for col in range(image_x):
            region = extract_region(padded_image, row + (se_size // 2), col + (se_size // 2), se_size)

            neigbourhood = np.sum(region)
            if neigbourhood == se_size**2:
                output[row, col] = 1

    return output


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply dilation on the given image using the structuring element.
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    # ToDo: Create the padded image and an empty output image that can be filled later.
    padded_image = pad_image(image, se_size // 2)
    output = np.zeros_like(image)

    image_y = image.shape[0]
    image_x = image.shape[1]

    # Log padded image to file with all values


    # ToDo: Iterate over the provided image and perform dilation around each pixel.
    # ToDo: Hint: Use the extract_region function to get the area around each pixel.
    # ToDo: Hint: Don't forget that the extract region function receives the padded image and the corresponding centers.

    for row in range(image_y):
        for col in range(image_x):
            region = extract_region(padded_image, row + (se_size // 2), col + (se_size // 2), se_size)

            neigbourhood = np.sum(region)
            if neigbourhood != 0:
                output[row, col] = 1
    return output

def repeat_operation(image: np.ndarray, operation_func, structuring_element: np.ndarray, iterations: int) -> np.ndarray:
    result = image
    for _ in range(iterations):
        result = operation_func(result, structuring_element)
    return result

def open_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform opening (erosion followed by dilation).
    result = input_image.copy()
    result = repeat_operation(result, erode_binary, structuring_element, iterations)
    result = repeat_operation(result, dilate_binary, structuring_element, iterations)
    return result


def close_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    # ToDo: Perform closing (dilation followed by erosion).
    result = input_image.copy()
    result = repeat_operation(result, dilate_binary, structuring_element, iterations)
    result = repeat_operation(result, erode_binary, structuring_element, iterations)
    return result


def load_binary(filepath: str) -> np.ndarray:
    # Load the image and binarize it again with a simple threshold.
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)  # type: ignore
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    # Save the binary image.
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Paths.
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    output_dir = data_dir / 'output'

    raw_erosion_image_path = data_dir / 'erosion_image_raw.png'
    raw_dilation_image_path = data_dir / 'dilation_image_raw.png'
    erosion_out_path = output_dir / 'erosion_output.png'
    dilation_out_path = output_dir / 'dilation_output.png'

    # Load images.
    erosion_input = load_binary(raw_erosion_image_path)
    dilation_input = load_binary(raw_dilation_image_path)

    eroded = load_binary(raw_erosion_image_path)
    dilated = dilation_input.copy()


    # Structuring element.
    SE = np.ones((5, 5), dtype=np.uint8)

    # Erosion.
    # ToDo: Perform erosion multiple times until the circles separate from each other.

    for i in range(11):
        eroded = erode_binary(eroded, SE)
        erosion_file = erosion_out_path.with_name(f'{erosion_out_path.stem}_{i}{erosion_out_path.suffix}')
        save_binary(eroded, erosion_file)
        if i % 10 == 0: show_image(eroded, f'Erosion Output{i}')

    # Dilation.
    # ToDo: Perform dilation multiple times until the hole closes.
    for i in range(11):
        dilated = dilate_binary(dilated, SE)
        dilation_file = dilation_out_path.with_name(f'{dilation_out_path.stem}_{i}{dilation_out_path.suffix}')
        save_binary(dilated, dilation_file)

        if i % 10 == 0: show_image(dilated, f'Dilation Output{i}')
