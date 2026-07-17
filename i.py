from sklearn.cluster import KMeans
import pandas as pd
df=pd.read_csv("C:\\Users\\nani3\\Desktop\\indian_bikes_dataset_1000_with_clusters.csv")
# Replace cluster numbers with labels
df['cluster'] = df['cluster'].map({
    0: '1',
    1: '2',
    2: '3',
    3: '4',
    4: '5',
    5: '6',
    6: '7',
    7: '8',
    8: '9',
    9: '10'
})

# Save updated dataset
df.to_csv(
    r"C:\Users\nani3\Desktop\indian_bikes_dataset_1000_with_clusters.csv",
    index=False
)


print("Dataset Updated Successfully")
print(df.head())