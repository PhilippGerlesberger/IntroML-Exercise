import cv2
import os
import numpy as np
from PIL import Image
from ex0 import IMAGE_PATH, ImageProcessor

IMAGE_PARENT_DIRECTORY: str = os.path.dirname(IMAGE_PATH)


def create_image(image_name: str) -> tuple[np.ndarray, str]:
    # Create a random image, create the full path to store it and then save it using CV2.
    test_image: np.ndarray = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    full_image_path: str = os.path.join(".\\", os.path.join(IMAGE_PARENT_DIRECTORY, image_name))
    cv2.imwrite(full_image_path, test_image)
    return test_image, full_image_path


def delete_image(full_image_path: str):
    # Remove the randomly created image.
    if os.path.exists(full_image_path):
        os.remove(full_image_path)

def print_matrix(X):
    print(X[0, 0:3, ::])

def compare_matrix(A, B):
    print("--------------------------")
    print("A:")
    print_matrix(A)
    print("B:")
    print_matrix(B)
    print("--------------------------")


class Tests():
    
    def test_init_rgb(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        image_rgb, image_rgb_colour_type = processor.get_image_data()


        delete_image(image_path)

    def test_init_bgr(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        image_bgr, image_rgb_colour_type = processor.get_image_data()


        delete_image(image_path)
    
    def test_init_gray(self):
        test_image, image_path = create_image("test.png")
        test_image_shape: np.ndarray = np.array(test_image.shape)
        test_image_shape[-1] = 1

        processor: ImageProcessor = ImageProcessor(image_path, "Gray")
        image_gray, image_rgb_colour_type = processor.get_image_data()
        image_gray_shape: np.ndarray = np.array(image_gray.shape)


        delete_image(image_path)

    def test_bgr_to_rgb(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        processor.convert_colour()
        image_converted, image_converted_colour_type = processor.get_image_data()

        delete_image(image_path)
    
    def test_rgb_to_bgr(self):
        test_image, image_path = create_image("test.png")

        print("test_image")
        print_matrix(test_image)
        processor: ImageProcessor = ImageProcessor(image_path, "RGB")
        processor.convert_colour()
        image_converted, image_converted_colour_type = processor.get_image_data()

        test_image_converted: np.ndarray = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        compare_matrix(image_converted, test_image_converted)
        delete_image(image_path)

    def test_clip(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        processor.clip_image(60, 180)
        clipped_image, _ = processor.get_image_data()

        test_image_clipped: np.ndarray = np.clip(test_image, 60, 180)

        delete_image(image_path)

    def test_flip_vertical(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        processor.flip_image(0)
        flipped_image, _ = processor.get_image_data()
        

        test_image_flipped: np.ndarray = np.fliplr(test_image)
        print(test_image[0][:3])
        print("-----------")
        print(flipped_image[0][:3])
        print("-----------")
        print(test_image_flipped[0][:3])



        delete_image(image_path)

    def test_flip_horizontal(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        processor.flip_image(1)
        flipped_image, _ = processor.get_image_data()

        test_image_flipped: np.ndarray = np.flipud(test_image)



        delete_image(image_path)

    def test_flip_both(self):
        test_image, image_path = create_image("test.png")

        processor: ImageProcessor = ImageProcessor(image_path, "BGR")
        processor.flip_image(2)
        flipped_image, _ = processor.get_image_data()

        test_image_flipped: np.ndarray = np.flip(test_image, axis=[0, 1])


        delete_image(image_path)

test = Tests()
test.test_rgb_to_bgr()