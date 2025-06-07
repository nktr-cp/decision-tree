import numpy as np

from decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self, n_classes, n_estimators=100, max_depth=None, max_features=None
    ):
        self.n_classes_ = n_classes
        self.n_estimators_ = n_estimators
        self.max_depth_ = max_depth
        self.max_features_ = max_features
        self.trees_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators_):
            # Bootstrap
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(
                n_classes=self.n_classes_,
                max_depth=self.max_depth_,
                max_features=self.max_features_,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

    def predict(self, X):
        # Aggregate predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )
