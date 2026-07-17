from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = load_iris()
X = data.data
y = data.target
model = PCA(n_components=2,random_state=42)
X_pca = model.fit_transform(X)
print("Transformed shape:", X_pca.shape)
plt.scatter(X_pca[:,0],X_pca[:,1], c='pink', edgecolor='k', s=10)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()
plt.close()

