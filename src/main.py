import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


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


def splitter(X, y):
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

    return best_feature_index, best_threshold


if __name__ == "__main__":
    iris = load_iris()
    # visualize_iris(iris)
    X, y = iris.data, iris.target
    best_feature_index, best_threshold = splitter(X, y)
    print("Best feature:", iris.feature_names[best_feature_index])
    print("Best threshold:", best_threshold)
