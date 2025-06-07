from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

n_features = X.shape[1]
fig, axes = plt.subplots(n_features, n_features, figsize=(8, 8))

for i in range(n_features):
    for j in range(n_features):
        if i == j:
            axes[i, j].hist(X[:, i], bins=20, color='skyblue')
            axes[i, j].set_title(iris.feature_names[i])
        else:
            for target_class in range(len(iris.target_names)):
                axes[i, j].scatter(X[y == target_class, j], X[y == target_class, i],
                                    label=iris.target_names[target_class], alpha=0.5)
            axes[i, j].set_xlabel(iris.feature_names[j])
            axes[i, j].set_ylabel(iris.feature_names[i])

plt.tight_layout()
plt.show()

