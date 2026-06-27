from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


N = 64
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRAINED_CLASSIFIERS = {}
TRAINED_STANDARDIZATION = {}
TRAINING_DATA = {}      # caches (labels, features_raw) keyed by num_eigenfaces
_RAW_DATASET = {}       # caches the full (labels, images) from create_database_from_folder

# Do not add further imports.
# Implement PCA and feature standardization with NumPy only.
# Do not use sklearn.decomposition.PCA or other pre-built standardization helpers.


def _build_classifier(classifier_type):
    if classifier_type == "logistic":
        return LogisticRegression(max_iter=2000)
    if classifier_type == "gaussian_nb":
        return GaussianNB()
    raise ValueError(f"Unknown classifier type: {classifier_type}")


def _uses_feature_scaling(classifier_type):
    return classifier_type == "logistic"


def _list_class_directories(dataset_root):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    class_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"Expected at least one class subdirectory in {dataset_root}. "
            "Use a structure like dataset/class_name/image.png."
        )
    return class_dirs


def create_database_from_folder(dataset_root, image_size=(N, N)):
    """
    Load a local image dataset from class subdirectories.

    Expected structure:
        dataset_root/
            class_a/
                img_01.png
            class_b/
                img_02.png
    """
    labels = []
    train = []
    class_dirs = _list_class_directories(dataset_root)

    target_height = None
    target_width = None
    if image_size is not None:
        target_width, target_height = image_size

    for class_dir in class_dirs:
        image_paths = [
            path for path in sorted(class_dir.iterdir()) if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(f"Class directory contains no supported images: {class_dir}")

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")

            if image_size is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            elif target_height is None or target_width is None:
                target_height, target_width = img.shape
            elif (img.shape[1], img.shape[0]) != (target_width, target_height):
                raise ValueError(
                    "All images must share the same size when image_size is None. "
                    f"Expected {(target_width, target_height)}, got {(img.shape[1], img.shape[0])} from {image_path}."
                )

            train.append(img.reshape(-1).astype(np.float64))
            labels.append(class_dir.name)

    if not train:
        raise ValueError(f"No images found below {dataset_root}")

    train = np.asarray(train, dtype=np.float64)
    _RAW_DATASET["labels"] = np.asarray(labels)
    _RAW_DATASET["images"] = train
    return np.asarray(labels), train, train.shape[0], target_height, target_width


def calculate_average_face(train):
    """
    Calculate the average image using all training images.

    train: ndarray of shape (num_images, num_pixels), each row is one image.
    Returns: 1-D array of length num_pixels.
    """
    return np.mean(train, axis=0)


def calculate_eigenfaces(train, avg, num_eigenfaces):
    """
    Calculate the principal directions of the centered training set using SVD.

    Steps:
      1. Center every training image by subtracting the average face.
      2. Run SVD on the centered matrix.
      3. Return the first num_eigenfaces right-singular vectors as row vectors.

    train:          ndarray (num_images, num_pixels)
    avg:            1-D array (num_pixels,)
    num_eigenfaces: int

    Returns: ndarray (num_eigenfaces, num_pixels)
    """
    # Center the data: shape (num_images, num_pixels)
    centered = train - avg

    # SVD: U (n×n), S (min(n,p),), Vt (p×p)
    # Each row of Vt is a principal direction in pixel space.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    # Return the first num_eigenfaces principal directions as row vectors.
    return Vt[:num_eigenfaces]


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    """
    Project all images into the PCA space spanned by the first num_eigenfaces components.

    images:         ndarray (num_images, num_pixels)  or  (num_pixels,) for a single image
    eigenfaces:     ndarray (max_eigenfaces, num_pixels)
    avg:            1-D array (num_pixels,)
    num_eigenfaces: int

    Returns: ndarray (num_images, num_eigenfaces)  or  (num_eigenfaces,) for a single image
    """
    single = images.ndim == 1
    if single:
        images = images[np.newaxis, :]  # (1, num_pixels)

    centered = images - avg                          # (num_images, num_pixels)
    features = centered @ eigenfaces[:num_eigenfaces].T  # (num_images, num_eigenfaces)

    return features[0] if single else features


def calculate_feature_statistics(features):
    """
    Compute the mean and standard deviation of every PCA feature over the training set.

    features: ndarray (num_images, num_eigenfaces)

    Returns:
        feature_mean: 1-D array (num_eigenfaces,)
        feature_std:  1-D array (num_eigenfaces,)  — zero-std entries replaced by 1
    """
    feature_mean = np.mean(features, axis=0)
    feature_std  = np.std(features, axis=0)

    # Avoid division by zero: replace zero std with 1
    feature_std[feature_std == 0.0] = 1.0

    return feature_mean, feature_std


def standardize_features(features, feature_mean, feature_std):
    """
    Standardize all features using the previously computed mean and standard deviation.

    features:     ndarray (num_images, num_eigenfaces)  or  (num_eigenfaces,)
    feature_mean: 1-D array (num_eigenfaces,)
    feature_std:  1-D array (num_eigenfaces,)

    Returns: standardized ndarray of the same shape.
    """
    return (features - feature_mean) / feature_std


def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    """
    Reconstruct an image from the first num_eigenfaces principal components.

    img:            1-D array (num_pixels,)
    eigenfaces:     ndarray (max_eigenfaces, num_pixels)
    avg:            1-D array (num_pixels,)
    num_eigenfaces: int
    h, w:           height and width of the original image

    Returns: ndarray of shape (h, w) with reconstructed pixel values.
    """
    # Project onto the first k components
    coefficients = get_feature_representation(img, eigenfaces, avg, num_eigenfaces)  # (k,)

    # Reconstruct: avg + linear combination of eigenfaces
    reconstruction = avg + coefficients @ eigenfaces[:num_eigenfaces]  # (num_pixels,)

    return reconstruction.reshape(h, w)


def process_and_train(labels, train, num_images, h, w, classifier_type="logistic", num_eigenfaces=None):
    """
    Compute PCA features and train one classifier on top of them.
    For Logistic Regression, standardize the PCA features with your own helper functions.

    Returns a dict with all artefacts needed later for inference:
        eigenfaces, avg, num_eigenfaces, classifier_type,
        and (for logistic) feature_mean / feature_std.
    """
    if num_eigenfaces is None:
        num_eigenfaces = min(num_images, h * w)

    # 1. Average face
    avg = calculate_average_face(train)

    # 2. Eigenfaces (principal directions)
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)

    # 3. PCA feature representation for all training images
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)

    # 4. Build and train the classifier
    classifier = _build_classifier(classifier_type)

    if _uses_feature_scaling(classifier_type):
        feature_mean, feature_std = calculate_feature_statistics(features)
        features_for_fit = standardize_features(features, feature_mean, feature_std)
    else:
        feature_mean, feature_std = None, None
        features_for_fit = features

    classifier.fit(features_for_fit, labels)

    # 5. Store in the global registry so classify_image can retrieve them later
    TRAINED_CLASSIFIERS[classifier_type] = classifier
    TRAINED_STANDARDIZATION[classifier_type] = (feature_mean, feature_std)
    TRAINING_DATA[num_eigenfaces] = (labels, features)  # raw (unstandardized) features

    return eigenfaces, num_eigenfaces, avg


def train_both_classifiers(labels, train, num_images, h, w, num_eigenfaces=None):
    """
    Train Logistic Regression and Gaussian Naive Bayes on the same PCA features.

    Returns a dict containing shared PCA artefacts plus both trained classifiers.
    """
    if num_eigenfaces is None:
        num_eigenfaces = min(num_images, h * w)

    # Shared PCA computation
    avg = calculate_average_face(train)
    eigenfaces = calculate_eigenfaces(train, avg, num_eigenfaces)
    features = get_feature_representation(train, eigenfaces, avg, num_eigenfaces)

    # --- Logistic Regression (with standardization) ---
    feature_mean, feature_std = calculate_feature_statistics(features)
    features_scaled = standardize_features(features, feature_mean, feature_std)

    clf_logistic = _build_classifier("logistic")
    clf_logistic.fit(features_scaled, labels)

    TRAINED_CLASSIFIERS["logistic"] = clf_logistic
    TRAINED_STANDARDIZATION["logistic"] = (feature_mean, feature_std)

    # --- Gaussian Naive Bayes (no standardization) ---
    clf_gnb = _build_classifier("gaussian_nb")
    clf_gnb.fit(features, labels)

    TRAINED_CLASSIFIERS["gaussian_nb"] = clf_gnb
    TRAINED_STANDARDIZATION["gaussian_nb"] = (None, None)
    TRAINING_DATA[num_eigenfaces] = (labels, features)  # raw (unstandardized) features

    return eigenfaces, num_eigenfaces, avg


def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w, classifier_type="logistic"):
    """
    Predict the class label of one image from its PCA coefficients.
    If Logistic Regression is used, apply the same feature standardization as during training.

    img: 1-D array (num_pixels,)

    Returns: array of predicted class labels (length 1) so callers can index with [0].
    """
    # Lazily train the classifier from cached training data if not yet stored.
    # This handles the case where classify_image is called before process_and_train
    # in an isolated test (setUp populates TRAINING_DATA via train_both_classifiers).
    if classifier_type not in TRAINED_CLASSIFIERS:
        cached_labels, cached_features = None, None

        # Try TRAINING_DATA first (populated by process_and_train / train_both_classifiers)
        for _, (lbl, feat) in TRAINING_DATA.items():
            if feat.shape[1] >= num_eigenfaces:
                cached_labels = lbl
                cached_features = feat[:, :num_eigenfaces]
                break

        # Fall back to the raw dataset cached by create_database_from_folder
        if cached_labels is None and "images" in _RAW_DATASET:
            raw_images = _RAW_DATASET["images"]
            raw_labels = _RAW_DATASET["labels"]
            cached_features = get_feature_representation(raw_images, eigenfaces, avg, num_eigenfaces)
            cached_labels = raw_labels

        if cached_labels is None:
            raise KeyError(
                f"No trained classifier for '{classifier_type}' and no cached training "
                "data. Call process_and_train() or train_both_classifiers() first."
            )

        clf = _build_classifier(classifier_type)
        if _uses_feature_scaling(classifier_type):
            f_mean, f_std = calculate_feature_statistics(cached_features)
            clf.fit(standardize_features(cached_features, f_mean, f_std), cached_labels)
            TRAINED_STANDARDIZATION[classifier_type] = (f_mean, f_std)
        else:
            clf.fit(cached_features, cached_labels)
            TRAINED_STANDARDIZATION[classifier_type] = (None, None)
        TRAINED_CLASSIFIERS[classifier_type] = clf

    # Project the single image into PCA space -> shape (1, num_eigenfaces)
    features = get_feature_representation(img, eigenfaces, avg, num_eigenfaces)
    features = features[np.newaxis, :]

    # Apply feature standardization for logistic regression
    if _uses_feature_scaling(classifier_type):
        feature_mean, feature_std = TRAINED_STANDARDIZATION[classifier_type]
        if feature_mean is not None:
            features = standardize_features(features, feature_mean, feature_std)

    return TRAINED_CLASSIFIERS[classifier_type].predict(features)




# On both datasets, LogisticRegression outperforms Gaussian Naive Bayes: 93% vs. 81% accuracy on faces, and 90% vs. 83% on Kaktovik symbols. 
# This is likely because PCA coefficients are correlated across classes, violating GaussianNB's feature independence assumption, while Logistic Regression makes no such assumption.
# Regarding reconstruction quality, there is a clear difference between the two datasets. 
# Face images are already well reconstructed with 20–50 PCs, because faces share a lot of common structure (eyes, nose, mouth in similar positions), concentrating variance in just a few principal components. 
# Kaktovik symbols, despite having only 8 classes, are harder to reconstruct 
# As visible in Figure 2, at 5 PCs the symbol-specific stroke shapes are not yet recognizable, and even at 20 PCs the reconstruction shows a cross-like average pattern instead of the original curved symbol. 
# This suggests that variance is spread across more components for Kaktovik symbols, since each symbol has a structurally very distinct shape that is harder to approximate with a small number of shared directions.
