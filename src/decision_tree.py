import numpy as np


class DecisionTreeClassifier:
    def __init__(self, n_classes, max_depth=None, max_features=None):
        self.max_depth_ = max_depth
        self.max_features_ = max_features
        self.tree_ = None
        self.n_classes_ = n_classes

    def fit(self, X, y):
        n_features = X.shape[1]
        effective_max_features = (
            n_features if self.max_features_ is None else self.max_features_
        )

        self.tree_ = self._splitter(
            X, y, effective_max_features, current_depth=0
        )
        return self

    def predict_probabilities(self, X):
        if self.tree_ is None:
            raise RuntimeError("You must call fit before predict")
        return np.array([self._predict_one(self.tree_, x) for x in X])

    def predict(self, X):
        probas = self.predict_probabilities(X)
        return np.argmax(probas, axis=1)

    def _splitter(self, X, y, max_features, current_depth=0):
        if (
            self.max_depth_ is not None and current_depth >= self.max_depth_
        ) or self._gini_impurity(y) == 0:
            return {
                "value": self._compute_class_proportions(y, self.n_classes_)
            }

        feature_index, threshold, info_gain = self._find_best_split(
            X, y, max_features
        )

        if info_gain == 0:
            return {
                "value": self._compute_class_proportions(y, self.n_classes_)
            }

        left_indices = X[:, feature_index] < threshold
        right_indices = ~left_indices
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        left_subtree = self._splitter(
            X_left, y_left, max_features, current_depth + 1
        )
        right_subtree = self._splitter(
            X_right, y_right, max_features, current_depth + 1
        )

        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _find_best_split(self, X, y, max_features):
        best_gain = -1
        best_feature_index, best_threshold = None, None
        current_impurity = self._gini_impurity(y)
        n_features = X.shape[1]

        feature_indices = np.random.choice(
            n_features, max_features, replace=False
        )

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                y_left, y_right = y[left_indices], y[~left_indices]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                p_left = len(y_left) / len(y)
                weighted_impurity = p_left * self._gini_impurity(y_left) + (
                    1 - p_left
                ) * self._gini_impurity(y_right)
                info_gain = current_impurity - weighted_impurity

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold, best_gain

    def _predict_one(self, tree, x):
        node = tree
        while "value" not in node:
            if x[node["feature_index"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    @staticmethod
    def _gini_impurity(y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)

    @staticmethod
    def _compute_class_proportions(y, n_classes):
        if len(y) == 0:
            return np.zeros(n_classes)
        counts = np.bincount(y, minlength=n_classes)
        return counts / len(y)
