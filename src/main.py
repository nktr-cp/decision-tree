import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def visualize_iris(dataset):
    X, y = dataset.data, dataset.target
    df = pd.DataFrame(X, columns=dataset.feature_names)
    df["target"] = y

    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(8, 8))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                axes[i, j].hist(X[:, i], bins=20, color="skyblue")
                axes[i, j].set_title(dataset.feature_names[i])
            else:
                for target_class in range(len(iris.target_names)):
                    axes[i, j].scatter(
                        X[y == target_class, j],
                        X[y == target_class, i],
                        label=iris.target_names[target_class],
                        alpha=0.5,
                    )
                axes[i, j].set_xlabel(iris.feature_names[j])
                axes[i, j].set_ylabel(iris.feature_names[i])

    plt.tight_layout()
    plt.show()


def gini_impurity(y):
    """Calculate the Gini impurity for a list of labels."""
    if len(y) == 0:
        return 0
    p = pd.Series(y).value_counts(normalize=True)
    return 1 - sum(p**2)


def find_best_split(X, y):
    current_impurity = gini_impurity(y)

    best_gain = 0
    best_feature_index = None
    best_threshold = None

    n_features = X.shape[1]

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])

        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2

            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold

            y_left, y_right = y[left_indices], y[right_indices]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            p_left = len(y_left) / len(y)
            p_right = len(y_right) / len(y)
            weighted_impurity = p_left * gini_impurity(
                y_left
            ) + p_right * gini_impurity(y_right)

            # Note
            # this problem can also be solved by minimizing weighted_impurity
            information_gain = current_impurity - weighted_impurity

            if information_gain > best_gain:
                best_gain = information_gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold, best_gain


def compute_class_proportions(y, n_classes):
    """Compute the proportions of each class in y."""
    counts = np.bincount(y, minlength=n_classes)
    if counts.sum() == 0:
        return np.zeros(n_classes)
    return counts / counts.sum()


def splitter(X, y, n_classes, max_depth=None, current_depth=0):
    if max_depth is not None and current_depth >= max_depth:
        leaf_value = compute_class_proportions(y, n_classes=n_classes)
        return {"value": leaf_value}

    if gini_impurity(y) == 0:
        leaf_value = compute_class_proportions(y, n_classes=n_classes)
        return {"value": leaf_value}

    feature_index, threshold, info_gain = find_best_split(X, y)

    if info_gain == 0:
        leaf_value = compute_class_proportions(y, n_classes=n_classes)
        return {"value": leaf_value}

    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold

    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    left_subtree = splitter(
        X_left,
        y_left,
        n_classes=n_classes,
        max_depth=max_depth,
        current_depth=current_depth + 1,
    )
    right_subtree = splitter(
        X_right,
        y_right,
        n_classes=n_classes,
        max_depth=max_depth,
        current_depth=current_depth + 1,
    )

    return {
        "feature_index": feature_index,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree,
    }


def predict_one(tree, x):
    """Predict the class label for a single instance x."""
    node = tree

    while "value" not in node:
        feature_index = node["feature_index"]
        threshold = node["threshold"]

        if x[feature_index] <= threshold:
            node = node["left"]
        else:
            node = node["right"]

    return node["value"]


def predict_probabilities(tree, X):
    probability_list = [predict_one(tree, x) for x in X]
    return np.array(probability_list)


def predict(tree, X):
    """Predict class labels for all instances in X."""
    probabilities = predict_probabilities(tree, X)
    return np.argmax(probabilities, axis=1)


if __name__ == "__main__":
    iris = load_iris()
    # visualize_iris(iris)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for depth in range(1, 5):
        print(f"Training decision tree with max depth = {depth}")
        tree = splitter(X_train, y_train, n_classes=3, max_depth=depth)

        y_pred = predict(tree, X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}\n")
