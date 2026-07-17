import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = [
    {"CustomerID": 1, "Name": "John", "Age": 28, "Subscribed": "Yes", "Region": "North"},
    {"CustomerID": 2, "Name": "Sophia", "Age": 34, "Subscribed": "No", "Region": "South"},
    {"CustomerID": 3, "Name": "Amit", "Age": 25, "Subscribed": "Yes", "Region": "East"},
    {"CustomerID": 4, "Name": "Emma", "Age": 40, "Subscribed": "No", "Region": "West"},
    {"CustomerID": 5, "Name": "Liam", "Age": 30, "Subscribed": "Yes", "Region": "South"}
]
df=pd.DataFrame(data)
print("Original Data:")
print(df)

# Step 2: Label Encoding (Subscribed)
label_encoder = LabelEncoder()
# Convert the "Subscribed" column into numeric values using LabelEncoder
df["Subscribed_Label"] = label_encoder.fit_transform(df["Subscribed"])
print("\nAfter Label Encoding Subscribed:")
print(df[["Name", "Subscribed", "Subscribed_Label"]])

# Step 3: One-Hot Encoding (Region)
# Convert the "Region" column into multiple binary columns
df_onehot = pd.get_dummies(df,columns=["Region"])

# Step 4: Display results
print("\nAfter One-Hot Encoding Region:")
print(df_onehot)
