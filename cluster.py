from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
# Load dataset
df = pd.read_csv("C:/Users/nani3/Desktop/indian_bikes_dataset_1000.csv")

# Select important features
features = ['cc', 'top_speed_kmh', 'mileage_kmpl']
X = df[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans model
kmeans = KMeans(n_clusters=10, random_state=0)

# Create cluster labels
df['cluster'] = kmeans.fit_predict(X_scaled)

# Display top 5 clustered records
result = df[features + ['cluster']].head(5)

print("\nTop 5 Clustered Bike Records:\n")
print(result.to_string(index=False))