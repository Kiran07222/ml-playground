import pandas as pd

# TODO: Import train_test_split from sklearn
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("team_performanc.csv")   # TODO: Fill in the CSV filename

def split_team_data():
    # Step 2: Features (Experience, Score) and Label (Promoted)
    X = df[["Experience","Score"]]           # TODO: Fill in the feature columns
    y = df["Promoted"]             # TODO: Fill in the label column

    # Step 3: Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
       X,y, test_size=0.2, random_state=42  # TODO: Fill in X and y
    )

    # Step 4: Return the results(training and testing sets)
    return X_train,X_test,y_train,y_test

# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_team_data()
    print("Training set:")
    print(X_train)
    print(y_train)
    print("\nTesting set:")
    print(X_test)
    print(y_test)
