import numpy as np


class NBNNClassifier:
    def __init__(self, metric="euclidean"):
        # Distance metric: "euclidean" or "cosine".
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Store training data and labels as NumPy arrays.

        Requirements:
            - convert X and y to NumPy arrays
            - validate shapes
            - store the sorted unique class labels in self.classes_
            - return self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must have 2 dims (samples, features).")
        if y.ndim != 1:
            raise ValueError("y must have 1 dimension")
        if len(X) != len(y):
            raise ValueError("Lengths of X and y must be equal.")

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)

        return self

    def _euclidean_distances(self, x):
        """Return the Euclidean distance from x to all training samples."""
        diff = self.X_train - x
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _cosine_distances(self, x):
        """
        Return the cosine distance from x to all training samples.

        Use the same convention as in knn.py:
            cosine_distance = 1 - cosine_similarity
        """
        dot_prod = self.X_train @ x.T
        abs_a = np.linalg.norm(x)
        abs_b = np.linalg.norm(self.X_train, axis=1)
        denominator = abs_b * abs_a
        # Prevent division by 0
        denominator[denominator==0] = 1e-9

        cos_similarity = dot_prod / denominator
        return 1 - cos_similarity

    def _class_scores(self, distances):
        """
        Compute one score per class.

        For each class, use the distance of the nearest training sample from
        that class. The predicted class is the class with the smallest score.
        """
        distance_c = []
        for c in self.classes_:
            class_index = np.where(c == self.y_train)
            distance_c.append(min(distances[class_index]))
        prediction_index = np.argmin(distance_c)
        return self.classes_[prediction_index]

    def predict(self, X):
        """
        Predict labels for one or more samples with the NBNN rule.

        Requirements:
            - allow either a single sample or a batch
            - compute distances to all training samples
            - convert them into class-wise scores
            - return the class label with the smallest score
        """
        X = np.asarray(X)

        if X.ndim == 1:
            X = X[None, :]

        prediction = []
        for sample_index, x in enumerate(X):
            if self.metric == "euclidean":
                distances = self._euclidean_distances(x)
            elif self.metric == "cosine":
                distances = self._cosine_distances(x)
            else:
                raise ValueError("Invalid metric")

            prediction.append(self._class_scores(distances))
        return np.asarray(prediction)