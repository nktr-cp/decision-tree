import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from decision_tree_classifier import DecisionTreeClassifier


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


if __name__ == "__main__":
    iris = load_iris()
    # visualize_iris(iris)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for depth in range(1, 5):
        print(f"Training decision tree with max depth = {depth}")
        tree = DecisionTreeClassifier(n_classes=3, max_depth=depth)

        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}\n")
