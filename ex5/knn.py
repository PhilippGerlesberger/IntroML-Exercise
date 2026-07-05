from pathlib import Path

import numpy as np
from scipy.stats import ansari
from sklearn.metrics.pairwise import cosine_distances

from visualization import plot_knn_neighbors


class KNNClassifier:
    def __init__(
        self,
        n_neighbors=3,
        metric="euclidean",
        plot_neighbors=False,
        image_shape=None,
        plot_dir=None,
        plot_prefix="knn",
    ):
        # Number of nearest neighbours to use for prediction.
        self.n_neighbors = n_neighbors
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.plot_neighbors = plot_neighbors
        self.image_shape = image_shape
        self.plot_dir = plot_dir
        self.plot_prefix = plot_prefix

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - check that X has shape (n_samples, n_features)
            - check that len(X) == len(y)
            - return self
        """
        # conversion to NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if len(X) != len(y):
            raise ValueError("Lengths of X and y must be equal.")

        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        diff = np.subtract(self.X_train, x)
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Hint:
            cosine_distance = 1 - cosine_similarity
            cosine_similarity = (a · b) / (||a|| * ||b||)

        Make sure that zero vectors do not cause a division-by-zero error.
        """
        dot_prod = self.X_train @ x
        denominator = np.linalg.norm(x) * np.linalg.norm(self.X_train, axis=1)

        # Prevent division by 0
        denominator[denominator == 0] = 1e-9

        cos_similarity = dot_prod / denominator
        return 1- cos_similarity


    def _majority_vote(self, neighbor_labels):
        """
        Return the most frequent label among the nearest neighbours.

        Hint:
            np.unique(..., return_counts=True) is useful here.
            If there is a tie, choose the smallest label after sorting.
        """
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        count_max = np.max(counts)
        label_max = labels[counts == count_max]
        # If more than one element is in the array then index 0 is the smallest label
        return label_max[0]

    def predict(self, X):
        """
        Predict labels for one or more input samples.

        Requirements:
            - convert X to a NumPy array
            - allow a single sample with shape (n_features,)
            - compute distances with the selected metric
            - select the k nearest neighbours
            - predict by majority vote
            - optionally save neighbour plots when plot_neighbors is True
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Not fitted yet.")
        # Convert input data to a NumPy array.
        X = np.asarray(X)

        # If a single sample is passed, turn it into shape (1, n_features).
        if X.ndim == 1:
            X = X[None, :]
        predictions = []
        # Iterate over each test sample to predict its label.
        for sample_index, x in enumerate(X):
            if self.metric == "euclidean":
                # Compute Euclidean distances from x to all training samples.
                distance = self._euclidean_distances(x)
            elif self.metric == "cosine":
                # Compute cosine distances from x to all training samples.
                distance = self._cosine_distances(x)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            # Find the indices of the k nearest neighbours.
            nn_indices = np.argsort(distance)[:self.n_neighbors]

            # Get the labels of the nearest neighbours.
            nn_labels = self.y_train[nn_indices]

            # Find the most common label and append it to predictions.
            predicted_label = self._majority_vote(nn_labels)
            predictions.append(predicted_label)

            if self.plot_neighbors and self.image_shape is not None:
                test_image = x.reshape(self.image_shape)
                neighbor_images = [
                    self.X_train[idx].reshape(self.image_shape) for idx in nn_indices
                ]

                save_path = None
                if self.plot_dir is not None:
                    save_path = Path(self.plot_dir) / f"{self.plot_prefix}_{sample_index:02d}.png"

                plot_knn_neighbors(
                    test_image=test_image,
                    neighbor_images=neighbor_images,
                    neighbor_labels=nn_labels,
                    test_label=None,
                    pred_label=predicted_label,
                    save_path=save_path,
                )

        # Return predictions as a NumPy array.
        return np.asarray(predictions)
