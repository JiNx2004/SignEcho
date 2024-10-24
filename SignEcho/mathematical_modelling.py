import numpy as np
from collections import Counter
from sklearn.datasets import make_classification  # Just to generate a toy dataset

# Generate a toy dataset
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth or n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=True)

        # find the best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gini = 1.0
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gini = self._gini(X_column, y, threshold)
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _gini(self, X_column, y, threshold):
        left_idxs = np.where(X_column <= threshold)
        right_idxs = np.where(X_column > threshold)
        left_gini = self._gini_impurity(y[left_idxs])
        right_gini = self._gini_impurity(y[right_idxs])
        m = len(y)
        return (len(left_idxs) / m) * left_gini + (len(right_idxs) / m) * right_gini

    def _gini_impurity(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_samp, y_samp = self._bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._most_common_label(tree_pred) for tree_pred in tree_preds])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

# Using the toy dataset
rf = RandomForest(n_trees=10, max_depth=10)
rf.fit(X, y)
predictions = rf.predict(X)

# Calculate accuracy
accuracy = np.sum(predictions == y) / len(y)
print(f"Accuracy: {accuracy:.4f}")
