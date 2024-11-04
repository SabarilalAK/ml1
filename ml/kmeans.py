import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# 2. Initialize K-means with k=4 clusters
kmeans = KMeans(n_clusters=4, init='random', n_init=1, max_iter=300, random_state=42)

# 3. Initial clustering (centroids after initialization)
initial_centroids = kmeans.cluster_centers_
print("Initial Centroids:", initial_centroids)

# 4. Fit K-means model and get final clusters
kmeans.fit(X)
final_centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# 5. Calculate Error Rate (MSE)
mse = mean_squared_error(X, kmeans.cluster_centers_[kmeans.labels_])

# 6. Plot Initial vs Final Clusters
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='red', marker='x')
plt.title("Initial Clusters")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x')
plt.title("Final Clusters with Epochs: {}".format(kmeans.n_iter_))

plt.show()
print("Epochs (Iterations):", kmeans.n_iter_)
print("Final MSE Error Rate:", mse)
