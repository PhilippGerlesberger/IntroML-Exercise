import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng()

def load_image(file_path: str) -> np.ndarray:
    # Load the image (either gray or colour).
    loaded_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if loaded_image is None:
        raise FileNotFoundError(f"Cannot load image at {file_path}")
    return loaded_image


def save_image(image: np.ndarray, file_path: str) -> None:
    # Save the image.
    cv2.imwrite(file_path, image)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 10.0) -> np.ndarray:
    """
    Generate gaussian noise and add it to the image.
    """

    noisy_image = image + rng.normal(loc = mean, scale = sigma, size = image.shape)
    noisy_image = noisy_image.clip(0, 255)

    return noisy_image.astype(np.uint8)


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
    """
    Generate random salt and pepper noise based on the provided probabilities.
    """
    noisy_image = image.copy()
    random_values = rng.random(image.shape)

    salt_mask = random_values <= salt_prob
    pepper_mask = random_values >= 1 - pepper_prob

    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0

    return noisy_image.astype(dtype=np.uint8)


def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    """
    Add poisson noise to the image.
    """

    # TODO image als rate? wieso kein willkuerlicher wert?
    noisy_image = rng.poisson(lam = image, size=image.shape)
    noisy_image = noisy_image.clip(0, 255)

    return noisy_image.astype(np.uint8)


def add_uniform_noise(image: np.ndarray, low: float = -20.0, high: float = 20.0) -> np.ndarray:
    """
    Add uniform noise to the image, which is sampled uniformly from the available values.
    """

    noisy_image = image + rng.uniform(low, high, image.shape)
    noisy_image = noisy_image.clip(0, 255)

    return noisy_image.astype(np.uint8)


def display_images(original: np.ndarray, processed: np.ndarray, title: str) -> None:
    # Transform the colour image (BGR) into an RGB image.
    def to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image

    adapted_original_image = to_rgb(original)
    adapted_noise_image = to_rgb(processed)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    print(original.ndim, processed.ndim)

    plt.imshow(
        adapted_original_image,
        cmap=None if adapted_original_image.ndim == 3 else 'gray',
        vmin=None if adapted_original_image.ndim == 3 else 0,
        vmax=None if adapted_original_image.ndim == 3 else 255,
    )
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(
        adapted_noise_image,
        cmap=None if adapted_noise_image.ndim == 3 else 'gray',
        vmin=None if adapted_noise_image.ndim == 3 else 0,
        vmax=None if adapted_noise_image.ndim == 3 else 255,
    )
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    input_file = str(data_dir / 'hello.png')
    gaussian_file = str(data_dir / 'hello_gaussian.png')
    salt_pepper_file = str(data_dir / 'hello_salt_pepper.png')
    poisson_file = str(data_dir / 'hello_poisson.png')
    uniform_file = str(data_dir / 'hello_uniform.png')

    original_image = load_image(input_file)

    # Apply noise to the images.
    gaussian = add_gaussian_noise(original_image)
    save_image(gaussian, gaussian_file)

    salt_pepper = add_salt_and_pepper_noise(original_image)
    save_image(salt_pepper, salt_pepper_file)

    poisson = add_poisson_noise(original_image)
    save_image(poisson, poisson_file)

    uniform = add_uniform_noise(original_image)
    save_image(uniform, uniform_file)

    # Display the images side by side.
    display_images(original_image, gaussian, 'Gaussian Noise')
    display_images(original_image, salt_pepper, 'Salt & Pepper Noise')
    display_images(original_image, poisson, 'Poisson Noise')
    display_images(original_image, uniform, 'Uniform Noise')
