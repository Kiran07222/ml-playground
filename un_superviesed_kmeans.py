from sklearn.cluster import KMeans 
import numpy as np
import matplotlib.pyplot as plt
# Generate synthetic data
from sklearn.datasets import make_blobs
# Create synthetic data with 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=21, n_init=10)
kmeans.fit(X)   
# Get the cluster centers and labels
centers = kmeans.cluster_centers_   
labels = kmeans.labels_
# Plot the clusters and their centers
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
