import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# 2. Create the linkage matrix for dendrogram
Z = linkage(X, 'ward')

# 3. Plot dendrogram
plt.figure(figsize=(10, 7))

dendrogram(Z)
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()

# 4. Perform Agglomerative Clustering with 4 clusters
agg_clustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
labels = agg_clustering.fit_predict(X)

# 5. Calculate Error Rate (MSE)
centroids = np.array([X[labels == i].mean(axis=0) for i in range(4)])
mse = mean_squared_error(X, centroids[labels])

# 6. Plot Final Clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title("Final Clusters (Hierarchical Clustering)")
plt.show()

print("Final MSE Error Rate:", mse)
