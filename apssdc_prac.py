import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("C:\\Users\\nani3\\Desktop\\indian_bikes_dataset_1000_with_clusters.csv")
# Data Preprocessing
df.dropna(inplace=True) 
# Feature Selection
features = ['cc', 'mileage_kmpl', 'top_speed_kmh', 'overall_score', 'on_road_price_inr']
X = df[features]
# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=10, random_state=42,n_init='auto')
model.fit(X_scaled)

# Predicting cluster labels
model.predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, df['Rating'])
print(f'Silhouette Score: {silhouette_avg}')    