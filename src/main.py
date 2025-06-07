import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris


def visualize(iris):
    X, y = iris.data, iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["target"] = y

    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(8, 8))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                axes[i, j].hist(X[:, i], bins=20, color="skyblue")
                axes[i, j].set_title(iris.feature_names[i])
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


# Gini impurity function
def impurity(y):
    """Calculate the Gini impurity for a list of labels."""
    if len(y) == 0:
        return 0
    p = pd.Series(y).value_counts(normalize=True)
    return 1 - sum(p**2)


if __name__ == "__main__":
    iris = load_iris()
    visualize(iris)
