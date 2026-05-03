import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ex0 import IMAGE_PATH, ImageProcessor, FlipMode

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



class VisualTester:
    def __init__(self, image_path: str | None = None, colour_type: str = "RGB"):
        base_dir = os.path.dirname(__file__)
        if image_path is not None:
            resolved_path = image_path
        else:
            # ex0.py uses "data/Image01.png" relative to workspace root.
            resolved_path = os.path.normpath(os.path.join(base_dir, "..", IMAGE_PATH))

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image not found at path: {resolved_path}")

        self._image_path: str = resolved_path
        self._colour_type: str = colour_type

    def _new_processor(self) -> ImageProcessor:
        return ImageProcessor(self._image_path, self._colour_type)

    @staticmethod
    def _to_display(image: np.ndarray, colour_type: str) -> np.ndarray:
        if colour_type == "BGR" and image.ndim == 3:
            return image[:, :, ::-1]
        return image

    def collect_results(self) -> list[tuple[str, np.ndarray, str]]:
        results: list[tuple[str, np.ndarray, str]] = []

        p = self._new_processor()
        img, ctype = p.get_image_data()
        results.append(("Original", img.copy(), ctype))

        p = self._new_processor()
        p.convert_colour()
        img, ctype = p.get_image_data()
        results.append(("convert_colour", img.copy(), ctype))

        p = self._new_processor()
        p.clip_image(60, 190)
        img, ctype = p.get_image_data()
        results.append(("clip_image(60,190)", img.copy(), ctype))

        p = self._new_processor()
        p.clip_image(0, 60)
        img, ctype = p.get_image_data()
        results.append(("clip_image(0,60)", img.copy(), ctype))

        p = self._new_processor()
        p.clip_image(190, 255)
        img, ctype = p.get_image_data()
        results.append(("clip_image(190,255)", img.copy(), ctype))

        p = self._new_processor()
        p.convert_to_grayscale("luminosity")
        img, ctype = p.get_image_data()
        results.append(("convert_to_grayscale luminosity", img.copy(), ctype))

        p = self._new_processor()
        p.convert_to_grayscale("lightness")
        img, ctype = p.get_image_data()
        results.append(("convert_to_grayscale lightness", img.copy(), ctype))

        p = self._new_processor()
        p.convert_to_grayscale("average")
        img, ctype = p.get_image_data()
        results.append(("convert_to_grayscale average", img.copy(), ctype))

        p = self._new_processor()
        p.rotate_image(90)
        img, ctype = p.get_image_data()
        results.append(("rotate_image(90)", img.copy(), ctype))

        p = self._new_processor()
        p.rotate_image(180)
        img, ctype = p.get_image_data()
        results.append(("rotate_image(180)", img.copy(), ctype))

        p = self._new_processor()
        p.rotate_image(270)
        img, ctype = p.get_image_data()
        results.append(("rotate_image(270)", img.copy(), ctype))

        p = self._new_processor()
        p.flip_image(FlipMode.HORIZONTAL)
        img, ctype = p.get_image_data()
        results.append(("flip_image(HORIZONTAL)", img.copy(), ctype))

        p = self._new_processor()
        p.flip_image(FlipMode.VERTICAL)
        img, ctype = p.get_image_data()
        results.append(("flip_image(VERTICAL)", img.copy(), ctype))

        p = self._new_processor()
        p.flip_image(FlipMode.HORIZONTAL_AND_VERTICAL)
        img, ctype = p.get_image_data()
        results.append(("flip_image(H_AND_V)", img.copy(), ctype))

        p = self._new_processor()
        image, _ = p.get_image_data()
        p.crop_center(max(10, image.shape[0] // 2), max(10, image.shape[1] // 2))
        img, ctype = p.get_image_data()
        results.append(("crop_center", img.copy(), ctype))

        p = self._new_processor()
        image, _ = p.get_image_data()
        p.resize_image(10, 10)
        img, ctype = p.get_image_data()
        results.append(("resize_image 10x10", img.copy(), ctype))

        return results

    def collect_results_gray(self) -> list[tuple[str, np.ndarray, str]]:
        results: list[tuple[str, np.ndarray, str]] = []

        p = self._new_processor()
        img, ctype = p.get_image_data()
        results.append(("Original (Gray)", img.copy(), ctype))

        p = self._new_processor()
        p.clip_image(60, 190)
        img, ctype = p.get_image_data()
        results.append(("clip_image(60,190)", img.copy(), ctype))

        p = self._new_processor()
        p.clip_image(0, 60)
        img, ctype = p.get_image_data()
        results.append(("clip_image(0,60)", img.copy(), ctype))

        p = self._new_processor()
        p.clip_image(190, 255)
        img, ctype = p.get_image_data()
        results.append(("clip_image(190,255)", img.copy(), ctype))

        p = self._new_processor()
        p.rotate_image(90)
        img, ctype = p.get_image_data()
        results.append(("rotate_image(90)", img.copy(), ctype))

        p = self._new_processor()
        p.rotate_image(180)
        img, ctype = p.get_image_data()
        results.append(("rotate_image(180)", img.copy(), ctype))

        p = self._new_processor()
        p.rotate_image(270)
        img, ctype = p.get_image_data()
        results.append(("rotate_image(270)", img.copy(), ctype))

        p = self._new_processor()
        p.flip_image(FlipMode.HORIZONTAL)
        img, ctype = p.get_image_data()
        results.append(("flip_image(HORIZONTAL)", img.copy(), ctype))

        p = self._new_processor()
        p.flip_image(FlipMode.VERTICAL)
        img, ctype = p.get_image_data()
        results.append(("flip_image(VERTICAL)", img.copy(), ctype))

        p = self._new_processor()
        p.flip_image(FlipMode.HORIZONTAL_AND_VERTICAL)
        img, ctype = p.get_image_data()
        results.append(("flip_image(H_AND_V)", img.copy(), ctype))

        p = self._new_processor()
        image, _ = p.get_image_data()
        p.crop_center(max(10, image.shape[0] // 2), max(10, image.shape[1] // 2))
        img, ctype = p.get_image_data()
        results.append(("crop_center", img.copy(), ctype))

        p = self._new_processor()
        p.resize_image(10, 10)
        img, ctype = p.get_image_data()
        results.append(("resize_image 10x10", img.copy(), ctype))

        return results

    def plot_results(self, output_file: str | None = None, show: bool = True, gray: bool = False):
        results = self.collect_results_gray() if gray else self.collect_results()
        n = len(results)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
        axes = np.array(axes).reshape(-1)

        for i, (title, image, ctype) in enumerate(results):
            ax = axes[i]
            #disp = self._to_display(image, ctype)
            disp = image
            if ctype == "Gray" or disp.ndim == 2:
                ax.imshow(disp, cmap="gray")
            else:
                ax.imshow(disp)
            ax.set_title(title)
            ax.axis("off")

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Visual Overview", fontsize=14)
        fig.tight_layout()

        if output_file is not None:
            fig.savefig(output_file, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    # Original image (colour)
    tester = VisualTester()
    out_path = os.path.join(os.path.dirname(__file__), "data", "ex0_visual_overview.png")
    tester.plot_results(output_file=out_path, show=True)

    # Image01 as grayscale
    gray_tester = VisualTester(colour_type="Gray")
    out_path_gray = os.path.join(os.path.dirname(__file__), "data", "ex0_visual_overview_gray.png")
    gray_tester.plot_results(output_file=out_path_gray, show=True, gray=True)

    # Random image
    _, random_image_path = create_image("random_test.png")
    try:
        random_tester = VisualTester(image_path=random_image_path)
        out_path_random = os.path.join(os.path.dirname(__file__), "data", "ex0_visual_overview_random.png")
        random_tester.plot_results(output_file=out_path_random, show=True)
    finally:
        delete_image(random_image_path)


