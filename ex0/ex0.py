import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"

class FlipMode(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    HORIZONTAL_AND_VERTICAL = 2


class ImageProcessor:
    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
        image_path (str): Path to the input image.
        colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        # Save the colour type and load the image using CV2.
        self._colour_type: str = colour_type
        self._image: np.ndarray = np.zeros(0)

        if colour_type == "BGR":
            self._image = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)
        elif colour_type == "RGB":
            self._image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        else:
            self._image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    def get_image_data(self) -> tuple[np.ndarray, str]:
        """
        Return the image data (image and colour scheme).

        Returns:
        tuple(np.ndarray, str): Loaded image and current colour scheme.
        """
        return self._image, self._colour_type


    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # Show the image depending on the colour type.
        if self._colour_type == "BGR":
            plt.imshow(self._image[:, :, ::-1])
        elif self._colour_type == "RGB":
            plt.imshow(self._image)
        else:
            plt.imshow(self._image, cmap="gray")
        plt.axis("off")
        plt.show()


    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        # Save the image.
        image_to_save = self._image[:,:,::-1] if self._colour_type == "RGB" else self._image
        cv2.imwrite(total_image_path, image_to_save)


    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        # Perform the colour conversion Update the colour type.
        self._image = self._image[:,:,::-1]
        self._colour_type = "RGB" if self._colour_type == "BGR" else "BGR"


    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
        clip_min (int): Minimum image colour intensity.
        clip_max (int): Maximum image colour intensity.
        """
        # Clip the image values to the given values.
        self._image[self._image < clip_min] = clip_min
        self._image[self._image > clip_max] = clip_max


    def convert_to_grayscale(self, method: str = "lightness"):
        """
        Convert a colour image to a grayscale image.
        Write the different options from scratch.

        Args:
        method (str): Method for the colour conversion, either lightness, average or luminosity.
        """
        if method not in ["lightness", "average", "luminosity"]:
            raise ValueError("The given method is not supported!")
        if self._colour_type not in ["BGR", "RGB"]:
            raise ValueError("The function only works for colour images!")

        luminosity_factor = np.array([0.2126, 0.7152, 0.0722])
        if method == "lightness":
            self._image = (np.max(self._image, axis=2) + np.min(self._image, axis=2)) / 2

        if method == "average":
            self._image = np.mean(self._image, axis=2)

        if method == "luminosity":
            if self._colour_type == "RGB":
               self._image = self._image @ luminosity_factor
            else:
                self._image = self._image @ luminosity_factor[::-1]

        # Update the colour type.
        self._colour_type = "Gray"


    def __transpose_image(self):
        """
        Transpose the image by swapping height and width while preserving colour channels.
        """

        if self._colour_type == "Grey":
            self._image = np.transpose(self._image)
        else:
            self._image = np.transpose(self._image, (1, 0, 2))


    def rotate_image(self, degrees: int = 0):
        """
        Rotate an image by a given angle (k * 90) clockwise.
        Do not use functions from external libraries apart from numpy.transpose.

        Args:
        degrees (int): Rotation angle.
        """
        if degrees % 90 != 0:
            raise ValueError("The provided rotation angle must be a multiple of 90!")

        num_rotation : int = (degrees // 90) % 4

        # Rotate the image depending on the given rotation value.
        if num_rotation == 0:
            return
        elif num_rotation == 1:
            self.flip_image(FlipMode.VERTICAL)
            self.__transpose_image()
        elif num_rotation == 2:
            self.flip_image(FlipMode.HORIZONTAL_AND_VERTICAL)
        else:
            self.flip_image(FlipMode.HORIZONTAL)
            self.__transpose_image()


    def flip_image(self, flip_value: int):
        """
        Flip an image either horizontally (0), vertically (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # Flip the image using indexing.
        if flip_value == FlipMode.HORIZONTAL:
            self._image = self._image[:, ::-1]
        elif flip_value == FlipMode.VERTICAL:
            self._image = self._image[::-1, :]
        elif flip_value == FlipMode.HORIZONTAL_AND_VERTICAL:
            self._image = self._image[::-1, ::-1]


    def crop_center(self, new_height: int, new_width: int):
        """
        Crop the image to a given size around the center.
        Do not use functions from external libraries.

        Args:
        new_height (int): Height of the cropped image.
        new_width (int): Width of the cropped image.
        """
        # Check that the given parameters are valid!
        if new_height <= 0 or new_width <= 0:
            raise ValueError("The given dimensions for cropping must be positive!")
        if new_height > self._image.shape[0] or new_width > self._image.shape[1]:
            raise ValueError("The given dimensions for cropping are too large!")

        # Calculate center
        y_center = self._image.shape[0] // 2
        x_center = self._image.shape[1] // 2

        # Calculate start and end indices for cropping
        y_start = y_center - new_height // 2
        x_start = x_center - new_width // 2
        y_end = y_start + new_height
        x_end = x_start + new_width

        # Crop the image around the center.
        self._image = self._image[y_start:y_end, x_start:x_end]


    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """
        # Resize the image. Research the available options in CV2.
        cv2.resize(self._image, (new_width, new_height))


if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
