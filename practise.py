from sklearn.cluster import KMeans
import numpy as np
X=np.random.randint(1,50,100).reshape(-1,2)
print(X)
np.random.seed(10)
model1=KMeans(n_clusters=5,random_state=21,n_init=1)
model1.fit(X)
model2=KMeans(n_clusters=5,random_state=21,n_init=10)
model2.fit(X)
print(model1.predict(X))

print(model2.predict(X))

